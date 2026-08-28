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

WHY `restore` EXISTS, and it is not a convenience wrapper around `rebuild`. An engineer's patch
carries the edited `.s`, and NOTHING IN THE BUILD GRAPH COMPILES A `.s`. `build.ninja` has one
`cuda_compile` edge per translation unit and its input is the `.hip`; the only thing that ever made
the edited assembly matter was `rebuild` overwriting the object that edge names. So a verifier that
applies an assembly patch to a clean tree and builds it normally measures THE UNPATCHED KERNEL: the
`.s` sits in the tree unread, ninja compiles the untouched `.hip`, correctness passes, and the round
is recorded as "the assembly edit did not help" for an edit that was never built. That is finding
(87)'s failure shape one layer further down, and it is worse there, because at least a source edit
IS in the file ninja compiles.

`restore` is the entry point that closes it: given a workspace, it finds every `.s` this module
emitted, resolves each back to the translation unit it came from, and rebuilds the object. It is
one command with one receipt because the alternative -- telling an agent to grep for the `.s`, read
its provenance, and compose a `rebuild` invocation -- is three steps that fail silently when any one
of them is skipped, and the whole point is that skipping is what we are defending against.

WHY THE `.s` CARRIES ITS OWN PROVENANCE. `restore` must know which translation unit an assembly file
belongs to, and it cannot derive it: a task tree legitimately holds several builds (this one holds
the candidate AND the immutable rocBLAS oracle), so `find_ninja` REFUSES rather than guessing when
no `--source-file` names one. Nothing else in the patch records the answer -- the `.s` is a
generated file with a name the engineer chose. So `emit` stamps it into the file itself, as an
assembler comment, where it cannot drift from the thing it describes the way a sidecar can. The
round-trip gate exercises the stamp on every run, which is why it is a comment and not a convention
anyone has to remember.

Usage:
  asm_roundtrip.py emit    --source-root <dir> [--source-file X.hip] --out X.s
  asm_roundtrip.py rebuild --source-root <dir> [--source-file X.hip] --asm X.s [--out X.cuda.o]
  asm_roundtrip.py verify    --source-root <dir> [--source-file X.hip]
  asm_roundtrip.py restore   --source-root <dir> [--scan <dir>]...      (the verifier's entry point)
  asm_roundtrip.py incumbent --source-root <dir> [--scan <dir>]...      (phase2's "am I continuing?")

Exit codes (the `isa_capture.py` / `ir_capture.py` convention):
  0  complete
  2  HOLE: the build was never recorded (no build.ninja / no device edge), so nothing can be replayed
  3  partial: a tool was missing or a step failed
  1  usage or write error
"""
from __future__ import annotations

import argparse
import hashlib
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

SCHEMA_PROVENANCE = "geak.asm-provenance/v1"
SCHEMA_RESTORE = "geak.asm-restore/v1"
SCHEMA_INCUMBENT = "geak.asm-incumbent/v1"

# `;` is AMDGCN assembly's comment character -- the same one `hipcc -S` writes `; %bb.0:` with -- so
# a stamped line assembles to nothing. It is the FIRST line so a reader sees it before the listing,
# and so `read_provenance` does not have to scan a 20k-line file to decide the stamp is absent.
_PROVENANCE_PREFIX = f"; {SCHEMA_PROVENANCE} "
# How far in to look. A stamp this module wrote is on line 1; the allowance exists only so a file an
# engineer prepended a note to still resolves, and it is small on purpose -- an unbounded search
# would happily find the string inside a `.asciz` and treat it as provenance.
_PROVENANCE_SEARCH_LINES = 8


def _emit(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# provenance -- which translation unit an emitted `.s` came from
# ---------------------------------------------------------------------------

def source_sha(path: Path) -> str | None:
    """The digest of the translation unit an assembly file was emitted from.

    This is what makes SUPERSESSION detectable. During phase2 the source is read-only, so a `.s`
    cannot go stale -- but the ladder leaves phase2, and a phase1 or deep round that edits this
    file makes the assembly a description of a program that no longer exists. Rebuilding from it
    then does something far worse than nothing: it silently reverts the source edit and builds the
    OLD kernel, which passes correctness and reads as a measurement of the new one.
    """
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def stamp_provenance(text: str, source_file: str, arch: str, sha: str | None = None) -> str:
    """The emitted assembly with its origin recorded in its first line."""
    note = {"source_file": source_file, "arch": arch}
    if sha:
        note["source_sha"] = sha
    return f"{_PROVENANCE_PREFIX}{json.dumps(note, sort_keys=True)}\n{text}"


def read_provenance(text: str) -> dict | None:
    """The stamp `emit` left, or None for an assembly file this module did not write."""
    for line in text.splitlines()[:_PROVENANCE_SEARCH_LINES]:
        if not line.startswith(_PROVENANCE_PREFIX):
            continue
        try:
            note = json.loads(line[len(_PROVENANCE_PREFIX):])
        except json.JSONDecodeError:
            return None
        return note if isinstance(note, dict) and note.get("source_file") else None
    return None


def instruction_lines(text: str) -> list[str]:
    """The lines of a `.s` that carry content, with directives, labels and comments dropped.

    Used to answer "was this assembly actually edited", and it compares lines rather than bytes for
    a reason that would otherwise make the check useless: `hipcc -S` writes the absolute source path
    into `.file` and the toolchain version into `.ident`, so the SAME program emitted in two
    different workspaces differs in text. A byte comparison would report every candidate as edited,
    including one whose engineer saved a pristine copy by mistake -- which is the single case this
    check exists to catch.
    """
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.split(";", 1)[0].split("//", 1)[0].strip()
        if not line or line.startswith(".") or line.endswith(":"):
            continue
        out.append(line)
    return out


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
    """The device assembly this build's own flags produce, stamped with its origin, ready to edit."""
    argv = [build["compiler"], *build["cuda_cflags"], *build["cuda_post_cflags"],
            "--cuda-device-only", "-S", build["source"], "-o", str(out)]
    rc, _, err = run(argv)
    if rc != 0 or not out.is_file():
        return EXIT_PARTIAL, {"step": "emit", "command": shlex.join(argv), "stderr": err[-4000:]}
    source_name = Path(build["source"]).name
    sha = source_sha(Path(build["source"]))
    text = stamp_provenance(out.read_text(encoding="utf-8", errors="replace"),
                            source_name, build["arch"], sha)
    try:
        out.write_text(text, encoding="utf-8")
    except OSError as exc:
        return EXIT_ERROR, {"step": "stamp", "asm_path": str(out), "error": str(exc)}
    return EXIT_OK, {
        "asm_path": str(out),
        "lines": text.count("\n") + 1,
        # The declaration layer is what makes this reassemblable. Reported so a caller can tell a
        # real `.s` from a disassembly listing that merely looks like one.
        "has_amdhsa_kernel": ".amdhsa_kernel" in text,
        "has_amdgpu_metadata": ".amdgpu_metadata" in text,
        # Reported so `restore`'s ability to find its way home is visible at emit time rather than
        # discovered missing in a verifier three phases later.
        "provenance": {"source_file": source_name, "arch": build["arch"], "source_sha": sha},
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
# restore -- put every edited `.s` in a workspace back into the build it came from
# ---------------------------------------------------------------------------

def find_assembly(scan_roots: list[Path]) -> tuple[list[tuple[Path, dict | None]], list[str]]:
    """Every `.s` under the scan roots, each with the stamp it carries or None.

    Unstamped files are RETURNED rather than filtered out. Dropping them would make the one case
    worth refusing -- a hand-made assembly file whose translation unit nobody recorded -- read
    exactly like a clean tree with no assembly patch in it.
    """
    found: list[tuple[Path, dict | None]] = []
    scanned: list[str] = []
    seen: set[Path] = set()
    for root in scan_roots:
        if not root.is_dir():
            continue
        scanned.append(str(root))
        for path in sorted(root.rglob("*.s")):
            if any(part == ".git" for part in path.parts):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                found.append((path, None))
                continue
            found.append((path, read_provenance(text)))
    return found, scanned


def restore_one(source_root: Path, asm: Path, source_file: str, tool: dict,
                scratch: Path) -> tuple[bool, dict]:
    """Rebuild one edited `.s` into the object its own build edge names."""
    row: dict = {"asm": str(asm), "source_file": source_file}
    build, holes = resolve_build(source_root, source_file, None)
    if build is None:
        row["holes"] = holes
        return False, row
    row["arch"] = build["arch"]
    row["output"] = build["output"]

    text = asm.read_text(encoding="utf-8", errors="replace")
    stamp = read_provenance(text)
    # An arch disagreement is refused rather than reconciled. `rebuild` reassembles against the
    # build's arch, so a `.s` emitted for another one would be assembled against a different
    # instruction encoding -- and where that does not fail outright it produces a wrong kernel that
    # builds, links and loads.
    if stamp and stamp.get("arch") and stamp["arch"] != build["arch"]:
        row["holes"] = [f"asm:arch_mismatch(the assembly was emitted for {stamp['arch']} and this "
                        f"build targets {build['arch']}; reassembling it would encode against the "
                        "wrong ISA)"]
        return False, row

    # SUPERSESSION. The source this assembly was emitted from has since changed, so the `.s`
    # describes a program that no longer exists. Refused, and refused LOUDLY, because the quiet
    # outcome is the dangerous one: rebuilding here would overwrite the object compiled from the
    # NEW source with code generated from the OLD, which reverts the source edit, passes
    # correctness, and reads as a measurement of the edit it just undid.
    now = source_sha(Path(build["source"]))
    if stamp and stamp.get("source_sha") and now and stamp["source_sha"] != now:
        row["superseded"] = True
        row["holes"] = [f"asm:source_superseded({source_file} has changed since this assembly was "
                        "emitted, so the .s describes a program that no longer exists. A phase1 or "
                        "deep round edited it and its result already beat this assembly on the "
                        "measured incumbent -- so the .s is superseded and must be DELETED, not "
                        "rebuilt)"]
        return False, row

    # Did the engineer actually change anything? This is the assembly phase's `hip_twin_sync`: a
    # pristine `.s` rebuilds cleanly, measures within noise of its parent, and closes a direction
    # that was never tried. Reported, never enforced here -- the verifier owns that verdict.
    code, info = emit(build, scratch / "pristine.s")
    if code == EXIT_OK:
        pristine = instruction_lines((scratch / "pristine.s").read_text(encoding="utf-8",
                                                                       errors="replace"))
        candidate = instruction_lines(text)
        row["edited"] = candidate != pristine
        row["instruction_lines"] = len(candidate)
        row["pristine_instruction_lines"] = len(pristine)
    else:
        # Unknown, not False. Reporting "unedited" here would refuse a real edit over a failed
        # control build, and the verifier's gate reads `edited is False` as a refusal.
        row["edited"] = None
        why = json.dumps(info, sort_keys=True)[:500]
        row["holes"] = ["asm:pristine_unavailable(the unedited assembly could not be emitted for "
                        f"comparison, so whether this file was edited is unknown: {why})"]

    code, info = rebuild(build, tool, asm, Path(build["output"]), scratch)
    row.update(info)
    return code == EXIT_OK, row


def restore(source_root: Path, scan_roots: list[Path], source_file: str | None,
            tool: dict, scratch: Path) -> tuple[int, dict]:
    """Every assembly file in this workspace, rebuilt into the build it belongs to."""
    found, scanned = find_assembly(scan_roots)
    out: dict = {"schema": SCHEMA_RESTORE, "scanned": scanned,
                 "assembly_found": len(found), "restored": [], "unresolved": [], "holes": []}

    if not found:
        # The caller reached here because the patch carried a `.s`. Finding none means it did not
        # land where the scan looks, and a silent success would report the unpatched kernel's time
        # as the candidate's.
        out["holes"].append("asm:none_found(no .s under "
                            f"{scanned or [str(r) for r in scan_roots]}; restore was called, so "
                            "one was expected -- check the patch applied and the scan root)")
        return EXIT_HOLE, out

    unstamped = [p for p, stamp in found if stamp is None]
    if unstamped and not source_file:
        out["unresolved"] = [str(p) for p in unstamped]
        out["holes"].append(
            f"asm:no_provenance({[p.name for p in unstamped]} carry no {SCHEMA_PROVENANCE} stamp, "
            "so the translation unit to rebuild cannot be derived -- a task tree holds the "
            "candidate AND the immutable oracle, and rebuilding the wrong one is a real object "
            "built from the wrong program. Pass --source-file to name it)")
        return EXIT_HOLE, out
    if len(unstamped) > 1:
        out["unresolved"] = [str(p) for p in unstamped]
        out["holes"].append(f"asm:ambiguous_source_file({len(unstamped)} unstamped files share one "
                            f"--source-file {source_file!r}; only one can have come from it)")
        return EXIT_HOLE, out

    # Two assembly files for ONE translation unit, refused rather than ordered.
    #
    # This is the EXPECTED shape, not an edge case: `asm_engineer.md` step 2 tells the engineer to
    # keep the pristine copy to diff against, and `git add -A` sweeps up whatever is in the
    # workspace. Both files then resolve to the same build edge, both rebuilds write the same
    # object, and the LAST one wins -- so with the names this repo's own tests use (`e.edit.s` and
    # `e.s`) the pristine copy sorts last and silently reverts the edit. The candidate is then
    # measured as its own parent and the round reads as "the assembly edit did not help".
    #
    # Refusing here rather than picking one is the whole point. There is no rule that could choose
    # correctly -- "the edited one" is not knowable without already knowing the answer -- and a
    # heuristic that got it right most of the time would be worse than none, because the times it
    # got it wrong would look exactly like the times it got it right.
    by_unit: dict[str, list[Path]] = {}
    for path, stamp in found:
        by_unit.setdefault(stamp["source_file"] if stamp else str(source_file), []).append(path)
    clashes = {unit: paths for unit, paths in by_unit.items() if len(paths) > 1}
    if clashes:
        out["unresolved"] = sorted(str(p) for paths in clashes.values() for p in paths)
        for unit, paths in sorted(clashes.items()):
            out["holes"].append(
                f"asm:multiple_assembly_for_one_unit({[p.name for p in paths]} all rebuild "
                f"{unit!r}, so each would overwrite the last and the winner would be whichever "
                "sorted last. Keep the pristine copy OUTSIDE the workspace -- $OUTPUT_DIR, not "
                "$KERNEL_PATH -- so it does not ride in the patch)")
        return EXIT_HOLE, out

    ok = True
    for index, (path, stamp) in enumerate(found):
        unit = stamp["source_file"] if stamp else source_file
        per = scratch / f"restore_{index}"
        per.mkdir(parents=True, exist_ok=True)
        succeeded, row = restore_one(source_root, path, unit, tool, per)
        row["provenance"] = "stamp" if stamp else "--source-file"
        out["restored"].append(row)
        ok = ok and succeeded

    if not ok:
        return EXIT_PARTIAL, out
    return EXIT_OK, out


# ---------------------------------------------------------------------------
# incumbent -- what assembly this workspace is already carrying
# ---------------------------------------------------------------------------
#
# phase2 EVOLVES the assembly. That only works if each round starts where the last one finished,
# and before this action there was no way for a round to find out that it had a predecessor: the
# engineer was told to `emit`, unconditionally, which regenerates the assembly FROM SOURCE -- the
# compiler's own output, with every previous round's edits gone. The search could re-discover one
# window's win forever and never stack two.
#
# So this answers the question the round has to ask first: is there an incumbent `.s` here, is it
# still the assembly of THIS source, and therefore am I continuing or starting?
#
# It is a read. It never compiles and never writes, so a round may call it before deciding anything.

def incumbent(source_root: Path, scan_roots: list[Path],
              source_file: str | None) -> tuple[int, dict]:
    """The assembly already in this workspace, and whether it still describes this source."""
    found, scanned = find_assembly(scan_roots)
    out: dict = {"schema": SCHEMA_INCUMBENT, "scanned": scanned,
                 "assembly_found": len(found), "incumbent": [], "holes": []}
    if not found:
        # NOT a hole. "There is no incumbent" is the correct and expected answer on the first
        # phase2 round of a cycle, and reporting it as a failure would push the engineer into
        # treating a normal cold start as something to work around.
        out["continue_from_assembly"] = False
        return EXIT_OK, out

    for path, stamp in found:
        row: dict = {"asm": str(path), "has_provenance": stamp is not None}
        unit = (stamp or {}).get("source_file") or source_file
        row["source_file"] = unit
        if not unit:
            row["usable"] = False
            row["why"] = "no provenance stamp and no --source-file, so its build cannot be resolved"
            out["incumbent"].append(row)
            continue
        build, holes = resolve_build(source_root, unit, None)
        if build is None:
            row["usable"] = False
            row["why"] = "; ".join(holes)
            out["incumbent"].append(row)
            continue
        now = source_sha(Path(build["source"]))
        was = (stamp or {}).get("source_sha")
        row["superseded"] = bool(was and now and was != now)
        row["usable"] = not row["superseded"]
        if row["superseded"]:
            row["why"] = ("the source has changed since this was emitted, so it describes a "
                          "program that no longer exists -- delete it and emit fresh")
        out["incumbent"].append(row)

    usable = [r for r in out["incumbent"] if r.get("usable")]
    out["continue_from_assembly"] = bool(usable)
    if not usable:
        out["holes"].append("asm:no_usable_incumbent(assembly is present but none of it can be "
                            "continued from; see each row's `why`)")
    return EXIT_OK, out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=("emit", "rebuild", "verify", "restore", "incumbent"))
    p.add_argument("--source-root", required=True, type=Path,
                   help="the candidate's source tree; build.ninja is found from here or its parent")
    p.add_argument("--source-file", default=None,
                   help="which device translation unit, named either as the file an engineer edits "
                        "(X.hip) or as the hipified twin torch compiles. For `restore` this is the "
                        "fallback for an assembly file carrying no provenance stamp")
    p.add_argument("--ninja", default=None, type=Path,
                   help="name the build.ninja outright when more than one exists")
    p.add_argument("--asm", default=None, type=Path, help="rebuild: the edited .s")
    p.add_argument("--out", default=None, type=Path,
                   help="emit: where to write the .s. rebuild: the host object to produce "
                        "(default: the path the build edge names, so a normal link picks it up)")
    p.add_argument("--scan", action="append", default=[], type=Path,
                   help="restore: where to look for .s files (repeatable). Defaults to the source "
                        "root AND its parent, because `.torch_ext/` conventionally sits beside "
                        "`src/` and an engineer may save the .s on either side")
    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv[1:])

    tool, missing = tools()
    roots = args.scan or [args.source_root, args.source_root.parent]

    # A pure read: no compile, no write, and no toolchain. It answers "am I continuing or
    # starting", which a round must be able to ask before it has decided to build anything.
    if args.action == "incumbent":
        code, info = incumbent(args.source_root, roots, args.source_file)
        _emit({"action": "incumbent", "exit_code": code, **info})
        return code

    # `restore` resolves a build PER assembly file, from the stamp each one carries, so it must not
    # resolve one up front: a workspace holding the candidate and the immutable oracle has two, and
    # asking for "the" build here would refuse the whole run over an ambiguity `restore` answers
    # file by file.
    if args.action == "restore":
        if missing:
            _emit({"action": "restore", "exit_code": EXIT_PARTIAL, "holes": missing})
            return EXIT_PARTIAL
        with tempfile.TemporaryDirectory(prefix="asmrt_") as tmp:
            code, info = restore(args.source_root, roots, args.source_file, tool, Path(tmp))
        _emit({"action": "restore", "exit_code": code, **info})
        return code

    build, holes = resolve_build(args.source_root, args.source_file, args.ninja)
    if build is None:
        _emit({"action": args.action, "exit_code": EXIT_HOLE, "holes": holes})
        return EXIT_HOLE

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
