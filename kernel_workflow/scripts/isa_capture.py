#!/usr/bin/env python3
"""Archive the AMDGCN machine code out of a tree that was already built.

Reads the artifact the benchmark actually ran, and does NOT recompile. That is
the point rather than a shortcut: a side compile with reconstructed flags answers
"what would this source become", and the question a round needs answered is "what
did the thing I just measured contain". Finding (87) is the same distinction one
level up -- `hip_twin_sync.py` exists because the file that was edited and the
file that was compiled can differ -- and re-deriving the build here would open a
third copy for them to differ from. It also means capture cannot break the build,
which is what keeps this safe to leave on.

HOW THE DEVICE CODE IS FOUND. A HIP shared object carries its device code objects
inside a clang offload bundle in `.hip_fatbin`; a Triton cache entry or a
standalone `.hsaco` IS a code object. Rather than shell out to `roc-obj-ls` /
`roc-obj-extract` and parse their listing -- whose text has moved between ROCm
releases, and a capture that silently stops matching reads as "this kernel has no
device code" -- this scans the bytes for ELF headers whose `e_machine` is
`EM_AMDGPU` and slices each one out at the length its own header declares. That
is deterministic, needs no ROCm tool to locate anything, works identically on a
`.so`, a bare fatbin and a `.hsaco`, and is testable with a synthesized header
and no GPU in the room.

Only two ROCm tools are needed, and each is reported when absent instead of being
worked around:
  llvm-objdump -d --arch-name=amdgcn   -> the instruction stream
  llvm-readelf --notes                 -> AMDGPU metadata: vgpr/agpr/sgpr counts,
                                          .private_segment_fixed_size (spill!),
                                          .group_segment_fixed_size (LDS), target

The archive is immutable once written, and its `manifest.json` carries the
`source_hash` of the tree it came from, computed with `source_hash.tree_hash`
-- the same hasher, with the same exclusions, that `kernel_lane.js` uses for its
fresh-workspace tar-copy. Without that field these signals are numbers with no
owner, and `kernel_launch_facts.py`'s finding (144) is the standing warning about
what accurate numbers about the wrong object cost: "the more trusted a source is
the less often anyone re-checks which object it is about."

SCOPE: AMD ONLY, and not incidentally. The ELF filter accepts `EM_AMDGPU` and
nothing else, the disassembler is `llvm-objdump --arch-name=amdgcn`, and the
register/scratch/LDS budgets are read out of AMDGPU metadata. An NVIDIA cubin is
`EM_CUDA`, carries no `amdhsa.kernels` block, and disassembles only under
nvdisasm/cuobjdump, so pointing this at a CUDA build yields the HOLE exit code
every time -- an archive reporting no device code, which is indistinguishable from
a clean capture of a simple kernel. Supporting cuda therefore means porting all
three layers plus `isa_signals.py`'s AMDGCN opcode vocabulary, not widening a
language list somewhere. `kernel_lane.js` deliberately does NOT gate this layer on
`TARGET_LANGUAGE` (see its comment above `ISA_ENABLED`): a language whitelist is an
indirect proxy for "did the build produce an AMDGPU code object", and it failed
SILENTLY. The HOLE exit code below is the direct measurement, so a CUDA or Triton
tree degrades to inert-and-visible rather than to a gate everyone believes is
running.

Usage:
  isa_capture.py --out <archive-dir> --source-root <dir> [--scan <path> ...]
                 [--arch gfx942] [--force]

Exit codes (the `hip_twin_sync.py` convention):
  0  complete: at least one code object archived with both disassembly and notes
  2  HOLE: no AMDGPU code object found -- nothing was captured
  3  partial: code objects found, but a tool was missing or a step failed
  1  usage or write error
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SCHEMA = "geak.isa-archive/v1"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_HOLE = 2
EXIT_PARTIAL = 3

EM_AMDGPU = 224
ELF_MAGIC = b"\x7fELF"
_ELF_TYPES_ACCEPTED = (1, 2, 3)  # ET_REL, ET_EXEC, ET_DYN

# Where a built tree keeps things that can hold device code. `.o` is included
# because a `--save-temps` or object-only build leaves the code object there, and
# excluded extensions are not listed at all: the ELF scan below rejects a
# non-object cheaply, so a narrow allowlist would only add a way to miss one.
ARTIFACT_SUFFIXES = (".so", ".hsaco", ".co", ".o", ".a", ".bundle", ".fatbin")

# Bounded so a stray `--scan /` cannot walk a filesystem. A real eval workspace
# has a handful of objects; a thousand means the wrong root was passed, and
# saying so beats spending an hour proving it.
MAX_ARTIFACTS = 64
MAX_ARTIFACT_BYTES = 512 * 1024 * 1024

_ARCH_RE = re.compile(r"amdgcn-amd-amdhsa--(gfx[0-9a-z]+)")

ROCM_TOOL_DIRS = ("llvm/bin", "bin")


def _emit(payload: dict) -> None:
    print(json.dumps(payload, sort_keys=True, indent=2))


def resolve_tool(name: str) -> str | None:
    """`name` on PATH, else under ROCM_PATH / the conventional /opt/rocm."""
    found = shutil.which(name)
    if found:
        return found
    roots = [os.environ.get("ROCM_PATH"), "/opt/rocm"]
    for root in roots:
        if not root:
            continue
        for sub in ROCM_TOOL_DIRS:
            candidate = Path(root) / sub / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    return None


def resolve_tools(objdump: str | None = None, readelf: str | None = None) -> dict:
    """The two tool paths, with an explicit override taking precedence.

    Separated from `capture` so both are injectable. An operator needs it because
    a container can carry ROCm somewhere neither PATH nor `/opt/rocm` names, and
    the alternative to a flag is a capture that reports `tool:missing` on a box
    that has the tool. The tests need it for the same reason from the other side:
    a `runner` that is never called because discovery already failed tests
    nothing, which is exactly what the first version of this file did.
    """
    return {
        "llvm-objdump": objdump or resolve_tool("llvm-objdump"),
        "llvm-readelf": readelf or resolve_tool("llvm-readelf"),
    }


def elf_object_span(blob: bytes, offset: int) -> tuple[int, int] | None:
    """(offset, length) of the ELF64 object starting at `offset`, or None.

    None covers every way this is not an AMDGPU code object: not 64-bit, not
    little-endian, not a type we expect, not `EM_AMDGPU`, or a declared length
    that runs past the end of the blob. A rejection is silent because the scan
    walks every `\\x7fELF` in a shared object and most of them legitimately are
    not device code -- what must never happen is accepting one and slicing a
    wrong length, which yields a disassembly of garbage that reads as a real one.
    """
    header = blob[offset:offset + 64]
    if len(header) < 64:
        return None
    if header[:4] != ELF_MAGIC:
        return None
    if header[4] != 2 or header[5] != 1:  # EI_CLASS=ELFCLASS64, EI_DATA=ELFDATA2LSB
        return None
    e_type, e_machine = struct.unpack_from("<HH", header, 16)
    if e_machine != EM_AMDGPU or e_type not in _ELF_TYPES_ACCEPTED:
        return None
    e_phoff, e_shoff = struct.unpack_from("<QQ", header, 32)
    e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum = struct.unpack_from(
        "<HHHHH", header, 52)
    end = max(e_ehsize,
              e_phoff + e_phentsize * e_phnum,
              e_shoff + e_shentsize * e_shnum)
    if end <= 0 or offset + end > len(blob):
        return None
    return offset, end


def iter_code_objects(blob: bytes):
    """Every AMDGPU code object embedded in `blob`, in order, non-overlapping."""
    position = 0
    while True:
        found = blob.find(ELF_MAGIC, position)
        if found < 0:
            return
        span = elf_object_span(blob, found)
        if span is None:
            position = found + 4
            continue
        offset, length = span
        yield offset, length
        position = offset + length


def find_artifacts(scan_roots: list[Path]) -> tuple[list[Path], list[str]]:
    """Candidate files that may hold device code, plus the reasons any were skipped."""
    holes: list[str] = []
    found: list[Path] = []
    seen: set[Path] = set()
    for root in scan_roots:
        try:
            if root.is_file():
                candidates = [root]
            elif root.is_dir():
                candidates = sorted(p for p in root.rglob("*")
                                    if p.suffix in ARTIFACT_SUFFIXES)
            else:
                holes.append(f"scan:missing({root})")
                continue
        except OSError as exc:
            holes.append(f"scan:unreadable({root}: {exc})")
            continue
        for path in candidates:
            try:
                resolved = path.resolve()
                if resolved in seen or not path.is_file():
                    continue
                size = path.stat().st_size
            except OSError as exc:
                holes.append(f"artifact:unreadable({path}: {exc})")
                continue
            if size > MAX_ARTIFACT_BYTES:
                holes.append(f"artifact:too_large({path}, {size} bytes)")
                continue
            seen.add(resolved)
            found.append(path)
    if len(found) > MAX_ARTIFACTS:
        holes.append(f"scan:too_many_artifacts({len(found)} candidates, keeping the first "
                     f"{MAX_ARTIFACTS}; this usually means --scan was pointed above the "
                     "build directory)")
        found = found[:MAX_ARTIFACTS]
    return found, holes


def arch_from_text(text: str) -> str | None:
    match = _ARCH_RE.search(text)
    return match.group(1) if match else None


def _default_runner(argv: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    return proc.returncode, proc.stdout, proc.stderr


def source_hash(root: Path) -> tuple[str | None, str | None]:
    """(hash, hole). Uses the lane's own hasher so the digest is comparable with
    every other `source_hash` in the run rather than being a second convention."""
    try:
        import source_hash
    except ImportError as exc:
        return None, f"source_hash:helper_missing({exc})"
    try:
        return source_hash.tree_hash(str(root)), None
    except OSError as exc:
        return None, f"source_hash:unreadable({root}: {exc})"


def capture(out_dir: Path, source_root: Path, scan_roots: list[Path],
            arch: str | None = None, force: bool = False, runner=_default_runner,
            tools: dict | None = None) -> dict:
    holes: list[str] = []

    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        return {"schema": SCHEMA, "exit_code": EXIT_ERROR, "archive": str(out_dir),
                "holes": [f"archive:not_empty({out_dir}; an ISA archive is immutable once "
                          "written so a round cannot silently re-attribute its evidence -- "
                          "pass --force only when replacing a capture on purpose)"]}

    tools = resolve_tools() if tools is None else tools
    objdump = tools.get("llvm-objdump")
    readelf = tools.get("llvm-readelf")
    if objdump is None:
        holes.append("tool:llvm-objdump missing (not on PATH, not under $ROCM_PATH or "
                     "/opt/rocm/llvm/bin) -- no instruction stream can be read")
    if readelf is None:
        holes.append("tool:llvm-readelf missing -- register, scratch and LDS budgets will be "
                     "UNAVAILABLE; they will not be reported as zero")

    digest, hash_hole = source_hash(source_root)
    if hash_hole:
        holes.append(hash_hole)

    artifacts, scan_holes = find_artifacts(scan_roots)
    holes.extend(scan_holes)

    objects_dir = out_dir / "objects"
    objects_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    arches: set[str] = set()
    index = 0
    for artifact in artifacts:
        try:
            blob = artifact.read_bytes()
        except OSError as exc:
            holes.append(f"artifact:unreadable({artifact}: {exc})")
            continue
        for offset, length in iter_code_objects(blob):
            raw = objects_dir / f"{index}.co"
            payload = blob[offset:offset + length]
            raw.write_bytes(payload)
            entry = {
                "index": index, "origin": str(artifact), "offset": offset,
                "bytes": length, "object": f"objects/{index}.co",
                # The actual bytes of the code object, so "did codegen change" can be
                # answered by comparison rather than by inference. `isa_signals.py`'s
                # `_identical` compares an opcode multiset plus the register/LDS/scratch
                # budget, and its own docstring is careful to say "as far as these
                # signals can see" -- an edit that changes operands, immediates or
                # instruction ORDER without changing the census is invisible to it and
                # is currently reported as byte-identical codegen, which
                # `mechanism_verdict` then converts into a hard `refuted`.
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            notes_text = ""
            if readelf is not None:
                code, stdout, stderr = runner([readelf, "--notes", str(raw)])
                if code == 0 and stdout.strip():
                    notes_text = stdout
                    (objects_dir / f"{index}.notes.txt").write_text(stdout, encoding="utf-8")
                    entry["notes"] = f"objects/{index}.notes.txt"
                    entry["notes_tool"] = readelf
                else:
                    holes.append(f"notes:failed(object {index} from {artifact.name}: "
                                 f"exit {code} {stderr.strip()[:160]})")
            if objdump is not None:
                code, stdout, stderr = runner(
                    [objdump, "-d", "--arch-name=amdgcn", str(raw)])
                if code == 0 and stdout.strip():
                    (objects_dir / f"{index}.disasm.txt").write_text(stdout, encoding="utf-8")
                    entry["disasm"] = f"objects/{index}.disasm.txt"
                    entry["disasm_tool"] = objdump
                    notes_text = notes_text or stdout
                else:
                    holes.append(f"disasm:failed(object {index} from {artifact.name}: "
                                 f"exit {code} {stderr.strip()[:160]})")
            observed = arch_from_text(notes_text)
            entry["arch"] = observed
            if observed:
                arches.add(observed)
            index += 1
            entries.append(entry)

    # An `--arch` filter keeps a fat binary built for two targets from mixing two
    # kernels' signals under one name. Filtered objects stay on disk and are named
    # in the manifest: dropping them silently is how a capture of the wrong target
    # reads as a capture of the right one.
    filtered_out: list[int] = []
    if arch:
        keep = []
        for entry in entries:
            if entry["arch"] in (arch, None):
                keep.append(entry)
            else:
                filtered_out.append(entry["index"])
                stale = objects_dir / f"{entry['index']}.disasm.txt"
                if stale.is_file():
                    stale.rename(objects_dir / f"{entry['index']}.disasm.other-arch")
                stale_notes = objects_dir / f"{entry['index']}.notes.txt"
                if stale_notes.is_file():
                    stale_notes.rename(objects_dir / f"{entry['index']}.notes.other-arch")
        if filtered_out:
            holes.append(f"arch:filtered({len(filtered_out)} object(s) were not {arch}: "
                         f"indices {filtered_out}; their dumps are kept with an "
                         ".other-arch suffix so isa_signals.py will not read them)")
        entries = keep

    with_disasm = [e for e in entries if e.get("disasm")]
    if not with_disasm:
        exit_code = EXIT_HOLE
        holes.append("capture:nothing(no AMDGPU code object yielded a disassembly. Either the "
                     "scanned tree was not built, the build targets a different device, or the "
                     "objects were stripped. This is a HOLE, not a clean capture: a caller must "
                     "not read an empty archive as evidence that the kernel is simple)")
    elif holes or not all(e.get("notes") for e in with_disasm):
        exit_code = EXIT_PARTIAL
    else:
        exit_code = EXIT_OK

    manifest = {
        "schema": SCHEMA,
        "exit_code": exit_code,
        "archive": str(out_dir),
        "source_root": str(source_root),
        "source_hash": digest,
        "arch": arch or (sorted(arches)[0] if len(arches) == 1 else None),
        "arches_observed": sorted(arches),
        "scanned": [str(p) for p in artifacts],
        "objects": entries,
        "object_count": len(entries),
        "tools": {"llvm-objdump": objdump, "llvm-readelf": readelf},
        "holes": holes,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--out", required=True, help="archive directory to create")
    parser.add_argument("--source-root", required=True,
                        help="candidate-owned source tree, hashed into the manifest so the "
                             "evidence has an owner")
    parser.add_argument("--scan", action="append", default=[],
                        help="repeatable; a built directory or artifact to scan for device "
                             "code. Defaults to --source-root.")
    parser.add_argument("--arch", default=None,
                        help="keep only code objects for this target, e.g. gfx942")
    parser.add_argument("--force", action="store_true",
                        help="replace a non-empty archive directory")
    parser.add_argument("--objdump", default=None,
                        help="path to llvm-objdump, when it is neither on PATH nor under "
                             "$ROCM_PATH or /opt/rocm")
    parser.add_argument("--readelf", default=None, help="path to llvm-readelf, likewise")
    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv[1:])
    out_dir = Path(args.out)
    source_root = Path(args.source_root)
    scan_roots = [Path(s) for s in args.scan] or [source_root]
    manifest = capture(out_dir, source_root, scan_roots, arch=args.arch, force=args.force,
                       tools=resolve_tools(args.objdump, args.readelf))
    _emit(manifest)
    return int(manifest["exit_code"])


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except OSError as exc:
        print(json.dumps({"schema": SCHEMA, "exit_code": EXIT_ERROR, "error": str(exc)},
                         sort_keys=True))
        sys.exit(EXIT_ERROR)
