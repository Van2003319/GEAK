#!/usr/bin/env python3
"""Emit the device assembly a build produced, and rebuild the build around an edited copy.

Why this exists
---------------
phase2 optimises the ASSEMBLY rather than the source. That needs two things the tree did not have:
a `.s` an engineer can edit, and a way to get the edited `.s` back into the artifact the harness
loads -- without changing the launcher, the kernarg layout, or anything else the host side depends
on.

`isa_capture.py` deliberately does NOT recompile: it reads the artifact that was measured, which is
the right answer to "what did the thing I just timed contain". This module answers the other
question, "build me one with this instruction stream instead", and the two must not be confused.
Capture is evidence; this is a build step.

WHY NOT THE PAPER'S ROUTE. AsmEvo (arXiv 2608.20711) recovers a reassemblable representation from a
compiled code object, because its setting is source-free: the deployed binary is all it has, so it
must reconstruct sections, symbols, kernel descriptors and AMDGPU metadata, and prove the
reconstruction round-trips before any edit is trusted. We are not in that setting. `hipcc -S` hands
us a `.s` that already carries `.amdhsa_kernel` and `.amdgpu_metadata` -- the declaration layer raw
disassembly lacks -- so the recovery half of the paper is unnecessary here and the fidelity gate
becomes a cheap check rather than a subsystem.

THE CHAIN, and it is measured rather than assumed (see `verify`):

    hipcc <cuda_cflags> --cuda-device-only -S      X.hip     -> X.s          (edit this)
    clang -x assembler --target=amdgcn-amd-amdhsa  X.s       -> X.dev.o
    clang --target=amdgcn-amd-amdhsa -Wl,-shared   X.dev.o   -> X.hsaco
    clang-offload-bundler --type=o                 X.hsaco   -> X.fatbin
    hipcc <cuda_cflags> --cuda-host-only -Xclang -fcuda-include-gpubinary -Xclang X.fatbin
                                                   X.hip     -> X.cuda.o

The last step is the load-bearing one: it compiles the HOST half normally and hands it our device
binary instead of letting it compile the device half. So the host object keeps its own symbols, its
kernarg marshalling and its launch stubs, and only the code object inside `.hip_fatbin` is ours.
That is what makes the result a drop-in for the object ninja would have produced.

WHERE THE FLAGS COME FROM. They are read, not reconstructed. `torch.utils.cpp_extension.load()`
writes a `build.ninja` carrying the exact device compiler and the exact `cuda_cflags`, and
`ir_capture.py` already parses it for the same reason -- a plausible hand-written flag set compiles
a slightly different program, and a slightly different program is the one failure this whole layer
exists to avoid. Those parsers are imported rather than copied.

HOW THE REBUILT OBJECT REACHES THE HARNESS, and why no task needs changing. `rebuild` writes to the
path the build edge names, inside the torch extension directory. ninja decides to recompile from
mtime plus the command string, and neither moves: the object is newer than its source and the
command in `build.ninja` is untouched. So the next `load()` finds the object up to date, skips the
compile, and links what we wrote -- measured, not assumed:

    [1/1] c++ k.cuda.o -shared -L/opt/rocm/lib -lamdhip64 -o probe.so

which means phase2 needs no build shim and no per-task loader hook. The corollary is a real hazard
and the reason `rebuild` resolves its default output against the build directory rather than the
caller's cwd: an object written anywhere else leaves the ORIGINAL one in place, the link succeeds,
and the round reads as "the edit did not help" when the edit was never built.

Usage:
  asm_roundtrip.py emit    --source-root <dir> [--source-file X.hip] --out X.s
  asm_roundtrip.py rebuild --source-root <dir> [--source-file X.hip] --asm X.s [--out X.cuda.o]
  asm_roundtrip.py verify  --source-root <dir> [--source-file X.hip]

Exit codes (the `isa_capture.py` / `ir_capture.py` convention):
  0  complete
  2  HOLE: the build was never recorded (no build.ninja / no device edge), so nothing can be replayed
  3  partial: a tool was missing or a step failed
  1  usage or write error
"""
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ir_capture as IRC        # noqa: E402  -- build.ninja parsing, read not rebuilt
import isa_capture as ISA       # noqa: E402  -- EM_AMDGPU code-object extraction, tool resolution

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_HOLE = 2
EXIT_PARTIAL = 3

DEVICE_TRIPLE = "amdgcn-amd-amdhsa"


def _emit(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def run(argv: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(argv, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def arch_from_flags(flags: list[str]) -> str | None:
    """The gfx target this build compiles for, taken from its own flags.

    Not defaulted to gfx942. A wrong -mcpu reassembles against a different instruction encoding and
    the failure surfaces as a link or load error much later, pointing at the edit rather than at the
    guess that caused it.
    """
    for flag in flags:
        if flag.startswith("--offload-arch="):
            return flag.split("=", 1)[1].strip()
        if flag.startswith("--amdgpu-target="):     # older spelling, same meaning
            return flag.split("=", 1)[1].strip()
    return None


def host_triple(clang: str) -> str:
    rc, out, _ = run([clang, "-print-target-triple"])
    return out.strip() if rc == 0 and out.strip() else "x86_64-unknown-linux-gnu"


def resolve_build(source_root: Path, source_file: str | None,
                  ninja: Path | None) -> tuple[dict | None, list[str]]:
    """The device compiler, its flags, and the edge that compiles our translation unit."""
    holes: list[str] = []
    if ninja is None:
        ninja, found_holes = IRC.find_ninja(source_root, source_file)
        holes.extend(found_holes)
    if ninja is None:
        return None, holes
    try:
        parsed = IRC.parse_ninja(Path(ninja).read_text(encoding="utf-8"))
    except OSError as exc:
        return None, holes + [f"ninja:unreadable({ninja}: {exc})"]

    if not parsed.get("compiler"):
        return None, holes + ["ninja:no_compiler(build.ninja defines no `nvcc` variable, so the "
                              "device compiler cannot be replayed)"]
    edge, edge_holes = IRC.select_edge(parsed.get("edges") or [], source_file)
    holes.extend(edge_holes)
    if edge is None:
        return None, holes

    flags = list(parsed.get("cuda_cflags") or [])
    arch = arch_from_flags(flags)
    if not arch:
        return None, holes + ["ninja:no_offload_arch(cuda_cflags names no --offload-arch, so the "
                              "gfx target to reassemble for is unknown; it is not guessed)"]
    # A ninja target is relative to the directory holding build.ninja, not to the caller's cwd.
    # Resolving it against cwd wrote the patched object somewhere the build never looks, and the
    # link then picked up the ORIGINAL object -- a silent no-op that reads as "the edit did not
    # help" rather than as "the edit was not built".
    build_dir = Path(ninja).resolve().parent
    output = Path(edge["output"])
    return {
        "ninja": str(ninja),
        "build_dir": str(build_dir),
        "compiler": parsed["compiler"],
        "cuda_cflags": flags,
        "cuda_post_cflags": list(parsed.get("cuda_post_cflags") or []),
        "source": edge["source"],
        "output": str(output if output.is_absolute() else build_dir / output),
        "arch": arch,
        "variables": parsed.get("variables") or {},
    }, holes


def tools() -> tuple[dict, list[str]]:
    """clang and the bundler, reported when absent rather than worked around."""
    found, missing = {}, []
    for name in ("clang", "clang-offload-bundler", "llvm-objdump", "llvm-readelf"):
        path = ISA.resolve_tool(name)
        if path:
            found[name] = path
        else:
            missing.append(f"tool:absent({name})")
    return found, missing


# ---------------------------------------------------------------------------
# emit
# ---------------------------------------------------------------------------

def emit(build: dict, out: Path) -> tuple[int, dict]:
    """The device assembly this build's own flags produce, ready to edit."""
    argv = [build["compiler"], *build["cuda_cflags"], *build["cuda_post_cflags"],
            "--cuda-device-only", "-S", build["source"], "-o", str(out)]
    rc, _, err = run(argv)
    if rc != 0 or not out.is_file():
        return EXIT_PARTIAL, {"step": "emit", "command": shlex.join(argv), "stderr": err[-4000:]}
    text = out.read_text(encoding="utf-8", errors="replace")
    return EXIT_OK, {
        "asm_path": str(out),
        "lines": text.count("\n") + 1,
        # The declaration layer is what makes this reassemblable. Reported so a caller can tell a
        # real `.s` from a disassembly listing that merely looks like one.
        "has_amdhsa_kernel": ".amdhsa_kernel" in text,
        "has_amdgpu_metadata": ".amdgpu_metadata" in text,
        "command": shlex.join(argv),
    }


# ---------------------------------------------------------------------------
# rebuild
# ---------------------------------------------------------------------------

def rebuild(build: dict, tool: dict, asm: Path, out: Path,
            scratch: Path) -> tuple[int, dict]:
    """A host object carrying OUR device code, drop-in for the one ninja would have produced."""
    arch = build["arch"]
    clang = tool["clang"]
    dev_o = scratch / "device.o"
    hsaco = scratch / "device.hsaco"
    fatbin = scratch / "device.fatbin"
    steps: list[dict] = []

    def step(name: str, argv: list[str], produces: Path) -> str | None:
        rc, _, err = run(argv)
        steps.append({"step": name, "command": shlex.join(argv), "returncode": rc})
        if rc != 0 or not produces.is_file():
            steps[-1]["stderr"] = err[-4000:]
            return name
        steps[-1]["bytes"] = produces.stat().st_size
        return None

    failed = step("assemble", [clang, "-x", "assembler", f"--target={DEVICE_TRIPLE}",
                               f"-mcpu={arch}", "-c", str(asm), "-o", str(dev_o)], dev_o)
    if failed:
        return EXIT_PARTIAL, {"failed_step": failed, "steps": steps}

    failed = step("link_code_object", [clang, f"--target={DEVICE_TRIPLE}", f"-mcpu={arch}",
                                       "-nostdlib", "-Wl,-shared", str(dev_o), "-o", str(hsaco)],
                  hsaco)
    if failed:
        return EXIT_PARTIAL, {"failed_step": failed, "steps": steps}

    targets = f"host-{host_triple(clang)},hipv4-{DEVICE_TRIPLE}--{arch}"
    failed = step("bundle", [tool["clang-offload-bundler"], "--type=o", f"--targets={targets}",
                             "--input=/dev/null", f"--input={hsaco}", f"--output={fatbin}"], fatbin)
    if failed:
        return EXIT_PARTIAL, {"failed_step": failed, "steps": steps}

    # The load-bearing step. `--cuda-host-only` keeps the host half exactly as the normal build
    # produces it -- launch stubs, kernarg marshalling, symbol names -- and
    # `-fcuda-include-gpubinary` substitutes our code object for the one it would have compiled.
    failed = step("host_object", [build["compiler"], *build["cuda_cflags"],
                                  *build["cuda_post_cflags"], "--cuda-host-only",
                                  "-Xclang", "-fcuda-include-gpubinary", "-Xclang", str(fatbin),
                                  "-c", build["source"], "-o", str(out)], out)
    if failed:
        return EXIT_PARTIAL, {"failed_step": failed, "steps": steps}

    return EXIT_OK, {"object_path": str(out), "arch": arch, "steps": steps}


# ---------------------------------------------------------------------------
# verify -- the round-trip fidelity gate
# ---------------------------------------------------------------------------

def device_instructions(tool: dict, blob_path: Path, arch: str) -> list[str] | None:
    """The instruction stream of the largest AMDGPU code object inside a file.

    Largest rather than first: a host object can carry more than one, and the bundle header itself
    is not one. Extraction is `isa_capture`'s, not a second copy of the ELF walk.
    """
    try:
        blob = blob_path.read_bytes()
    except OSError:
        return None
    # `iter_code_objects` yields (offset, LENGTH), not (start, end).
    best = b""
    for offset, length in ISA.iter_code_objects(blob):
        if length > len(best):
            best = blob[offset:offset + length]
    if not best:
        return None
    with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=False) as fh:
        fh.write(best)
        tmp = Path(fh.name)
    try:
        rc, out, _ = run([tool["llvm-objdump"], "-d", f"--triple={DEVICE_TRIPLE}",
                          f"--mcpu={arch}", str(tmp)])
    finally:
        tmp.unlink(missing_ok=True)
    if rc != 0:
        return None
    insns: list[str] = []
    for line in out.splitlines():
        if not line.startswith("\t"):
            continue
        body = line[1:].split("//", 1)[0].rstrip()
        if body and not body.startswith("."):
            insns.append(body)
    return insns


def verify(build: dict, tool: dict, scratch: Path) -> tuple[int, dict]:
    """Does an UNEDITED round trip reproduce the build's own device code?

    The paper's one-time gate, and it earns its place for the same reason: it establishes that later
    edits operate on a faithful, rebuildable representation. If this fails, an edit that changes
    nothing would already change the binary, and every measurement afterwards would be attributing
    the round trip's own damage to the engineer's edit.
    """
    ref_o = scratch / "reference.o"
    argv = [build["compiler"], *build["cuda_cflags"], *build["cuda_post_cflags"],
            "-c", build["source"], "-o", str(ref_o)]
    rc, _, err = run(argv)
    if rc != 0 or not ref_o.is_file():
        return EXIT_PARTIAL, {"failed_step": "reference_build", "command": shlex.join(argv),
                              "stderr": err[-4000:]}

    asm = scratch / "roundtrip.s"
    code, info = emit(build, asm)
    if code != EXIT_OK:
        return code, {"failed_step": "emit", **info}

    rt_o = scratch / "roundtrip.o"
    code, info = rebuild(build, tool, asm, rt_o, scratch)
    if code != EXIT_OK:
        return code, {"failed_step": "rebuild", **info}

    ref = device_instructions(tool, ref_o, build["arch"])
    rt = device_instructions(tool, rt_o, build["arch"])
    if not ref or not rt:
        return EXIT_PARTIAL, {"failed_step": "disassemble",
                              "reference_instructions": len(ref or []),
                              "roundtrip_instructions": len(rt or [])}

    identical = ref == rt
    first_diff = None
    if not identical:
        for i, (a, b) in enumerate(zip(ref, rt)):
            if a != b:
                first_diff = {"index": i, "reference": a, "roundtrip": b}
                break
        if first_diff is None:
            first_diff = {"index": min(len(ref), len(rt)),
                          "reference": "(end)" if len(ref) < len(rt) else ref[len(rt)],
                          "roundtrip": "(end)" if len(rt) < len(ref) else rt[len(ref)]}
    return (EXIT_OK if identical else EXIT_PARTIAL), {
        "round_trip_identical": identical,
        "reference_instructions": len(ref),
        "roundtrip_instructions": len(rt),
        "first_difference": first_diff,
        "arch": build["arch"],
        "source": build["source"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=("emit", "rebuild", "verify"))
    p.add_argument("--source-root", required=True, type=Path,
                   help="the candidate's source tree; build.ninja is found from here or its parent")
    p.add_argument("--source-file", default=None,
                   help="which device translation unit, named either as the file an engineer edits "
                        "(X.hip) or as the hipified twin torch compiles")
    p.add_argument("--ninja", default=None, type=Path,
                   help="name the build.ninja outright when more than one exists")
    p.add_argument("--asm", default=None, type=Path, help="rebuild: the edited .s")
    p.add_argument("--out", default=None, type=Path,
                   help="emit: where to write the .s. rebuild: the host object to produce "
                        "(default: the path the build edge names, so a normal link picks it up)")
    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv[1:])

    build, holes = resolve_build(args.source_root, args.source_file, args.ninja)
    if build is None:
        _emit({"action": args.action, "exit_code": EXIT_HOLE, "holes": holes})
        return EXIT_HOLE

    tool, missing = tools()
    if missing:
        _emit({"action": args.action, "exit_code": EXIT_PARTIAL, "holes": holes + missing})
        return EXIT_PARTIAL

    with tempfile.TemporaryDirectory(prefix="asmrt_") as tmp:
        scratch = Path(tmp)
        if args.action == "emit":
            if not args.out:
                _emit({"action": "emit", "exit_code": EXIT_ERROR, "error": "--out is required"})
                return EXIT_ERROR
            args.out.parent.mkdir(parents=True, exist_ok=True)
            code, info = emit(build, args.out)

        elif args.action == "rebuild":
            if not args.asm or not args.asm.is_file():
                _emit({"action": "rebuild", "exit_code": EXIT_ERROR,
                       "error": f"--asm must name an existing file (got {args.asm})"})
                return EXIT_ERROR
            out = args.out or Path(build["output"])
            out.parent.mkdir(parents=True, exist_ok=True)
            code, info = rebuild(build, tool, args.asm, out, scratch)

        else:
            code, info = verify(build, tool, scratch)

    _emit({"action": args.action, "exit_code": code, "holes": holes,
           "ninja": build["ninja"], **info})
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
