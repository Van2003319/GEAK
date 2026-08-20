#!/usr/bin/env python3
"""Archive the lowering trajectory -- every pass that CHANGED the IR -- of one kernel.

This is the L3 evidence source, and it is a different question from the one
`isa_capture.py` answers. That module reads the artifact the benchmark actually
ran and deliberately does NOT recompile, because "what did the thing I measured
contain" is the question a verification receipt needs. Machine code is the END
of the pipeline; it can say a vector load is not there, and it can never say
which stage of lowering dropped it. A trajectory can, and a trajectory only
exists if you compile again.

So this module recompiles, and that is a liability rather than a feature. Two
rules contain it:

  1. It works in a scratch copy and never in WORKSPACE. A tree rebuilt with
     evidence flags is a tree whose later timings nobody sanctioned.
  2. It re-derives the code object WITHOUT the evidence flags and proves that
     object matches the archive of the binary that was measured. Until that
     check passes, the stages in this archive describe *a* program, not *the*
     program, and `manifest.json` says so in `provenance`. A caller that
     attributes a plateau to a trajectory it has not tied to the measured
     kernel has done the same thing `kernel_launch_facts.py` finding (144)
     warns about: accurate numbers about the wrong object.

HOW THE HIP COMMAND IS OBTAINED. It is not reconstructed. `torch.utils.
cpp_extension.load()` writes a `build.ninja` next to the extension, and that
file carries the exact device compiler (`nvcc = /opt/rocm/bin/hipcc`), the exact
`cuda_cflags` (`--offload-arch=gfx942 -fno-gpu-rdc -O3 ...`), and one build edge
per device translation unit. Reconstructing those flags by hand would produce a
plausible command that compiles a slightly different program, which is the exact
failure mode rule 2 exists to catch -- so the flags are read, quoted into the
manifest, and replayed verbatim.

THE HIPIFY TWIN. The edge does not name the file an engineer edits. torch
hipifies `src/X.hip` into `src/X_hip.hip` and compiles the twin, so every `!dbg`
location in these stages names the twin. `hip_twin_sync.py` is the existing
proof that the pair is in lockstep; this module records the mapping so a reader
of an IR location is never left to assume it.

WHAT THE TRACE COVERS. `-mllvm -print-changed=quiet` emits a full dump after
each pass that changed anything, and on this toolchain that spans the whole
pipeline -- middle-end IR passes (`SROAPass`, `InferAddressSpacesPass`,
`InstCombinePass`) AND the MachineFunction passes that follow instruction
selection (`si-fold-operands`, register allocation, `prologepilog`). So "which
pass declined to widen the load" and "which pass inserted the wait" are both
answerable from one capture. `-mllvm -filter-print-funcs=<symbol>` narrows it to
the hot kernel, which is the difference between a few dozen stages and a
thousand.

Usage:
  ir_capture.py --out <archive-dir> --source-root <dir> [--backend hip|triton|auto]
                [--ninja <build.ninja>] [--source-file X.hip] [--kernel <symbol>]
                [--isa-archive <dir>] [--triton-command "<cmd>"] [--force]

Exit codes (the `hip_twin_sync.py` / `isa_capture.py` convention):
  0  complete: stages captured AND tied to the measured binary
  2  HOLE: nothing replayable was found -- no trajectory was captured
  3  partial: stages captured, but a step failed or provenance is unproven
  1  usage or write error
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SCHEMA = "geak.ir-archive/v1"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_HOLE = 2
EXIT_PARTIAL = 3

# `*** IR Dump After <description> on <scope> ***`, where description is either a
# new-PM class name (`SROAPass`) or a legacy pass's human name followed by its
# command-line argument (`SI Fold Operands (si-fold-operands)`), and scope is a
# mangled symbol, `[module]`, or an SCC written `(sym1, sym2)`.
#
# `description` is GREEDY on purpose. A non-greedy match would split at the first
# " on ", and legacy AMDGPU pass names contain ordinary English -- the scope, by
# contrast, is a symbol or a bracketed literal and never contains " on ", so
# binding to the LAST separator is the one that cannot be fooled by a pass name.
STAGE_RE = re.compile(r"^\*\*\* IR Dump After (?P<desc>.+) on (?P<scope>.+) \*\*\*$")

# Triton's MLIR pass manager writes its own banner. Both directions appear; the
# `Before` form is what a `MLIR_ENABLE_DUMP=1` run emits per pass.
MLIR_STAGE_RE = re.compile(
    r"^// -----// IR Dump (?P<when>Before|After) (?P<desc>.+?) "
    r"(?:\((?P<arg>[^)]*)\) )?\((?P<scope>[^)]*)\) //----- //$")

# A legacy pass banner ends with its command-line argument in parentheses. That
# argument is the stable identity -- the human name has been reworded between
# LLVM releases while the flag stayed put -- so it is preferred as the pass id.
# The argument is case-sensitive and not always lowercase: `Post RA hazard
# recognizer (post-RA-hazard-rec)` keeps the subtarget's capitals. A lowercase-only
# class silently fell through to the prose name for exactly those passes, so the
# same pass would be identified two different ways depending on its spelling.
PASS_ARG_RE = re.compile(r"^(?P<name>.+?)\s+\((?P<arg>[A-Za-z0-9][A-Za-z0-9._-]*)\)$")

# A MachineFunction dump opens with this line. Used to pick the file extension,
# because a reader (and `ir_signals.py`) needs to know whether it is looking at
# LLVM IR or MIR before it counts anything.
MIR_MARKER = "# Machine code for function "

# Bounded so a runaway trace cannot fill a lane's disk. A filtered capture of one
# kernel runs to a few dozen stages; four figures means the filter did not apply
# and the archive would be unreadable anyway.
MAX_STAGES = 2000
MAX_TRACE_BYTES = 512 * 1024 * 1024

# The evidence flags. Split out because the provenance replay must use the build's
# own flags and NOTHING else -- if these leaked into that command the comparison
# would be against a binary this module itself perturbed.
TRACE_FLAGS = ("-mllvm", "-print-changed=quiet")
FRONTEND_FLAGS = ("-S", "-emit-llvm", "-Xclang", "-disable-llvm-passes")


def _emit(payload: dict) -> None:
    print(json.dumps(payload, sort_keys=True, indent=2))


def default_runner(cmd: list[str], cwd: str | None = None,
                   env: dict | None = None) -> tuple[int, str, str]:
    """Run `cmd`, returning (exit_code, stdout, stderr).

    Injectable for the same reason `isa_capture.resolve_tools` is: a test that
    cannot reach the compiler still has to exercise the parsing, selection and
    provenance logic, and a runner that is never called because discovery
    already failed tests nothing.
    """
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True,
                          errors="replace")
    return proc.returncode, proc.stdout, proc.stderr


# ---------------------------------------------------------------------------
# build.ninja -- the exact device compile command, read rather than rebuilt
# ---------------------------------------------------------------------------

def find_ninja_candidates(source_root: Path) -> list[Path]:
    """Every `build.ninja` under the source root or its parent.

    Searched in both because `--source-root` is conventionally the candidate's
    `src/` while `.torch_ext/` sits beside it.
    """
    seen: list[Path] = []
    for root in (source_root, source_root.parent):
        if not root.is_dir():
            continue
        for pattern in ("build.ninja", ".torch_ext/*/build.ninja",
                        "*/.torch_ext/*/build.ninja"):
            for hit in sorted(root.glob(pattern)):
                if hit not in seen:
                    seen.append(hit)
    return seen


def find_ninja(source_root: Path, source_file: str | None = None) -> tuple[Path | None, list[str]]:
    """The one build file that compiles the source we were asked to trace.

    A task tree legitimately holds more than one extension: `dense_bf16_gemm_fused`
    builds `..._candidate` AND `..._oracle`, the second being the immutable
    rocBLAS comparison. Tracing the oracle would produce a real, well-formed
    trajectory of the wrong program, and nothing downstream could tell.

    So when several build files exist the tie is broken by a FACT rather than a
    preference: which of them actually has a device edge for the requested
    source. Nothing to break the tie with -- no `--source-file`, or several
    builds compiling it -- is a refusal, not a default.
    """
    seen = find_ninja_candidates(source_root)
    if not seen:
        return None, ["ninja:absent(no build.ninja under the source root or its parent -- "
                      "the tree has not been built by torch.utils.cpp_extension, so there is "
                      "no recorded device compile command to replay)"]
    if len(seen) == 1:
        return seen[0], []

    if source_file:
        matching: list[Path] = []
        for path in seen:
            try:
                edges = parse_ninja(path.read_text(encoding="utf-8"))["edges"]
            except OSError:
                continue
            if any(edge_matches(e, source_file) for e in edges):
                matching.append(path)
        if len(matching) == 1:
            return matching[0], []
        if len(matching) > 1:
            return None, [f"ninja:ambiguous({source_file!r} is compiled by {len(matching)} "
                          f"builds {[str(p) for p in matching]}; pass --ninja to name the one "
                          "that produced the measured artifact)"]
        return None, [f"ninja:no_build_compiles({source_file!r} appears in none of "
                      f"{[str(p) for p in seen]})"]

    return None, [f"ninja:ambiguous({len(seen)} build.ninja files: {[str(p) for p in seen]}; "
                  "pass --source-file so the build that compiles it can be identified, or "
                  "--ninja to name one outright. A task tree holds the candidate AND the "
                  "immutable oracle, and tracing the oracle would attribute a real trajectory "
                  "to the wrong program)"]


def parse_ninja(text: str) -> dict:
    """The device compiler, its flags, and one entry per device translation unit.

    Only what is needed to replay: `nvcc`, `cuda_cflags`, `cuda_post_cflags`,
    and the `cuda_compile` edges. Host `compile` edges are ignored -- they carry
    no device code and replaying them would produce an empty trajectory that
    reads exactly like a kernel with nothing in it.
    """
    variables: dict[str, str] = {}
    edges: list[dict] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" ") and "=" in line and not line.startswith("build "):
            key, _, value = line.partition("=")
            key = key.strip()
            if key and " " not in key:
                variables[key] = value.strip()
            continue
        if line.startswith("build "):
            target, _, rest = line[len("build "):].partition(":")
            rule, _, inputs = rest.strip().partition(" ")
            if rule.strip() != "cuda_compile":
                continue
            sources = shlex.split(inputs.strip())
            if not sources:
                continue
            edges.append({"output": target.strip(), "source": sources[0]})
    return {
        "compiler": variables.get("nvcc", ""),
        "cuda_cflags": shlex.split(variables.get("cuda_cflags", "")),
        "cuda_post_cflags": shlex.split(variables.get("cuda_post_cflags", "")),
        "edges": edges,
        "variables": variables,
    }


def edge_matches(edge: dict, source_file: str) -> bool:
    """Whether this device edge compiles `source_file`.

    Accepts the file an engineer edits (`X.hip`) as well as the hipified twin
    (`X_hip.hip`) torch actually compiles. Kept separate from `select_edge` so
    "which build contains this source" stays a containment question: routing it
    through the selector made a build with two matching edges report that it did
    not compile the file at all, which is the opposite of what was wrong.
    """
    wanted = Path(source_file).name
    stem = wanted[:-4] if wanted.endswith(".hip") else wanted
    return Path(edge["source"]).name in {wanted, f"{stem}_hip.hip", f"{stem}.hip"}


def select_edge(edges: list[dict], source_file: str | None) -> tuple[dict | None, list[str]]:
    """Which translation unit to trace, refusing to guess when it is ambiguous.

    `source_file` may name the file an engineer edits (`X.hip`) even though the
    edge compiles torch's hipified twin (`X_hip.hip`); both spellings resolve to
    the same edge. With no selector and more than one device edge, this is a
    HOLE rather than a default -- an archive attributed to the wrong translation
    unit is worse than no archive, because it is a real trajectory of a real
    kernel and nothing downstream can tell it is the wrong one.
    """
    if not edges:
        return None, ["ninja:no_device_edge(build.ninja has no cuda_compile edge; nothing "
                      "in this build produces device code)"]
    if source_file:
        hits = [e for e in edges if edge_matches(e, source_file)]
        if not hits:
            names = [Path(e["source"]).name for e in edges]
            return None, [f"ninja:no_such_source({source_file!r} matches no device edge; "
                          f"this build compiles {names})"]
        if len(hits) > 1:
            return None, [f"ninja:ambiguous_source({source_file!r} matches "
                          f"{[Path(e['source']).name for e in hits]})"]
        return hits[0], []
    if len(edges) > 1:
        names = [Path(e["source"]).name for e in edges]
        return None, [f"ninja:ambiguous_edge({len(edges)} device translation units {names}; "
                      "pass --source-file to name the one that holds the hot kernel. Tracing "
                      "an unnamed one would attribute this plateau to whichever file ninja "
                      "listed first)"]
    return edges[0], []


# ---------------------------------------------------------------------------
# Splitting a trace into stages
# ---------------------------------------------------------------------------

def pass_identity(desc: str) -> tuple[str, str | None]:
    """(pass_id, pass_arg) for a banner description.

    New-PM passes announce a class name and have no flag; legacy ones announce a
    sentence plus the flag that runs them. The flag wins when present because it
    is what a reader will grep for and what `-print-after=` would accept.
    """
    match = PASS_ARG_RE.match(desc.strip())
    if match:
        return match.group("arg"), match.group("arg")
    return desc.strip(), None


def split_stages(text: str) -> tuple[list[dict], list[str]]:
    """Every `IR Dump After` block in a `-print-changed` trace, in order.

    Anything before the first banner is compiler chatter (warnings, remarks) and
    is returned as a hole rather than dropped: a trace whose head is a
    diagnostic usually means the flags were rejected, and silently starting at
    the first banner would hide that.
    """
    stages: list[dict] = []
    holes: list[str] = []
    current: dict | None = None
    body: list[str] = []
    preamble: list[str] = []

    for line in text.splitlines():
        match = STAGE_RE.match(line)
        if match:
            if current is not None:
                current["body"] = "\n".join(body).strip("\n")
                stages.append(current)
            desc = match.group("desc").strip()
            pass_id, pass_arg = pass_identity(desc)
            current = {
                "index": len(stages),
                "pass_id": pass_id,
                "pass_name": desc,
                "pass_arg": pass_arg,
                "scope": match.group("scope").strip(),
            }
            body = []
            continue
        if current is None:
            preamble.append(line)
        else:
            body.append(line)

    if current is not None:
        current["body"] = "\n".join(body).strip("\n")
        stages.append(current)

    noise = "\n".join(preamble).strip()
    if noise:
        holes.append(f"trace:preamble({len(noise)} chars before the first IR banner; the first "
                     f"line is {noise.splitlines()[0][:160]!r}. If the flags were rejected there "
                     "will be no banners at all, so this is reported rather than skipped)")
    if len(stages) > MAX_STAGES:
        holes.append(f"trace:truncated({len(stages)} stages exceeds the {MAX_STAGES} bound; the "
                     "kernel filter almost certainly did not apply, and an unfiltered trace "
                     f"cannot be navigated. Only the first {MAX_STAGES} were written)")
        stages = stages[:MAX_STAGES]
    return stages, holes


def split_mlir_stages(text: str) -> tuple[list[dict], list[str]]:
    """The same, for Triton's MLIR pass manager banner."""
    stages: list[dict] = []
    holes: list[str] = []
    current: dict | None = None
    body: list[str] = []
    for line in text.splitlines():
        match = MLIR_STAGE_RE.match(line.strip())
        if match:
            if current is not None:
                current["body"] = "\n".join(body).strip("\n")
                stages.append(current)
            desc = match.group("desc").strip()
            arg = match.group("arg")
            current = {
                "index": len(stages),
                "pass_id": arg or desc,
                "pass_name": desc,
                "pass_arg": arg,
                "scope": (match.group("scope") or "").strip(),
                "when": match.group("when"),
            }
            body = []
            continue
        if current is not None:
            body.append(line)
    if current is not None:
        current["body"] = "\n".join(body).strip("\n")
        stages.append(current)
    if not stages:
        holes.append("trace:no_mlir_banner(MLIR_ENABLE_DUMP produced no `// -----// IR Dump` "
                     "banner. Either the command compiled nothing, or the kernel was served "
                     "from cache -- set TRITON_ALWAYS_COMPILE=1 and an isolated "
                     "TRITON_CACHE_DIR)")
    return stages, holes


def stage_extension(body: str) -> str:
    """`.mir` for a MachineFunction dump, `.ll` otherwise.

    Cheap and load-bearing: `ir_signals.py` counts different things in the two,
    and counting SSA values in MIR (or virtual registers in IR) would produce
    numbers that look like signals and mean nothing.
    """
    head = body.lstrip("\n")[:400]
    return ".mir" if MIR_MARKER in head else ".ll"


def write_stages(out_dir: Path, stages: list[dict]) -> list[dict]:
    stage_dir = out_dir / "stages"
    stage_dir.mkdir(parents=True, exist_ok=True)
    written: list[dict] = []
    for stage in stages:
        body = stage.get("body", "")
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", stage["pass_id"]).strip("-") or "pass"
        name = f"{stage['index']:03d}-{safe}{stage_extension(body)}"
        (stage_dir / name).write_text(body + "\n", encoding="utf-8")
        entry = {k: v for k, v in stage.items() if k != "body"}
        entry["file"] = f"stages/{name}"
        entry["lines"] = body.count("\n") + 1 if body else 0
        written.append(entry)
    return written


# ---------------------------------------------------------------------------
# Provenance -- does the replayed object match the binary that was measured?
# ---------------------------------------------------------------------------

def compare_to_measured(probe_archive: Path, isa_archive: Path,
                        kernel: str | None) -> dict:
    """Per-kernel identity between the replayed object and the measured one.

    Deliberately NOT `isa_signals.py diff`'s whole-archive
    `unchanged_machine_code`. That field is scoped to a whole archive and turns
    false as soon as the symbol sets differ -- and they legitimately differ here,
    because the measured artifact is a `.so` linked from every translation unit
    while this probe compiles one. Kernels present only in the measured archive
    are other translation units, not drift.

    The claim made is therefore the narrow one that is actually true: every
    kernel this probe produced which also exists in the measured binary is
    identical there. `matched` being empty is NOT a pass -- it is the case where
    nothing was compared, which must never read as agreement.
    """
    try:
        import isa_signals
    except ImportError as exc:  # pragma: no cover - environment defect, not logic
        return {"checked": False, "ir_binary_equals_measured": False,
                "reason": f"isa_signals is not importable ({exc}), so the replayed object "
                          "cannot be tied to the measured one"}

    # Named rather than reached for inline: if this predicate is ever renamed,
    # the failure must be a loud "the identity rule is gone", not a provenance
    # check that quietly starts comparing nothing.
    identical = getattr(isa_signals, "_identical", None)
    if identical is None:
        return {"checked": False, "ir_binary_equals_measured": False,
                "reason": "isa_signals no longer exposes the codegen-identity predicate this "
                          "check is defined against; provenance cannot be asserted"}

    try:
        probe = isa_signals.build_signals(probe_archive)
        measured = isa_signals.build_signals(isa_archive)
    except Exception as exc:  # noqa: BLE001 - any read failure is a provenance failure
        return {"checked": False, "ir_binary_equals_measured": False,
                "reason": f"could not read both archives ({exc})"}

    left = {k["name"]: k for k in probe.get("kernels", [])}
    right = {k["name"]: k for k in measured.get("kernels", [])}
    shared = sorted(set(left) & set(right))
    drifted = [n for n in shared if not identical(left[n], right[n])]
    matched = [n for n in shared if n not in drifted]

    if kernel and kernel not in shared:
        return {"checked": True, "ir_binary_equals_measured": False,
                "matched_kernels": matched, "drifted_kernels": drifted,
                "probe_only": sorted(set(left) - set(right)),
                "measured_only": sorted(set(right) - set(left)),
                "reason": f"the selected kernel {kernel!r} is not present in both archives, so "
                          "this trajectory cannot be attributed to the measured kernel"}
    if not shared:
        return {"checked": True, "ir_binary_equals_measured": False,
                "matched_kernels": [], "drifted_kernels": [],
                "probe_only": sorted(left), "measured_only": sorted(right),
                "reason": "no kernel symbol is present in both archives -- nothing was "
                          "compared, which is not the same as nothing having changed"}
    if drifted:
        return {"checked": True, "ir_binary_equals_measured": False,
                "matched_kernels": matched, "drifted_kernels": drifted,
                "probe_only": sorted(set(left) - set(right)),
                "measured_only": sorted(set(right) - set(left)),
                "reason": f"{len(drifted)} kernel(s) differ between the replayed object and the "
                          f"measured binary ({drifted[:4]}); the replay is a different program "
                          "and its trajectory describes a neighbour, not this kernel"}
    return {"checked": True, "ir_binary_equals_measured": True,
            "matched_kernels": matched, "drifted_kernels": [],
            "probe_only": sorted(set(left) - set(right)),
            "measured_only": sorted(set(right) - set(left)),
            "reason": f"{len(matched)} kernel(s) replayed byte-comparable to the measured "
                      "binary as far as these signals can see"}


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def capture_hip(out_dir: Path, source_root: Path, *, ninja: Path | None,
                source_file: str | None, kernel: str | None,
                isa_archive: Path | None, runner, scratch: Path) -> dict:
    holes: list[str] = []

    if ninja is None:
        ninja, ninja_holes = find_ninja(source_root, source_file)
        holes.extend(ninja_holes)
    if ninja is None:
        return {"exit_code": EXIT_HOLE, "holes": holes, "backend": "hip"}

    try:
        parsed = parse_ninja(Path(ninja).read_text(encoding="utf-8"))
    except OSError as exc:
        holes.append(f"ninja:unreadable({ninja}: {exc})")
        return {"exit_code": EXIT_HOLE, "holes": holes, "backend": "hip"}

    edge, edge_holes = select_edge(parsed["edges"], source_file)
    holes.extend(edge_holes)
    if edge is None:
        return {"exit_code": EXIT_HOLE, "holes": holes, "backend": "hip",
                "ninja": str(ninja)}

    compiler = parsed["compiler"]
    if not compiler:
        holes.append("ninja:no_compiler(build.ninja defines no `nvcc` variable; the device "
                     "compiler cannot be replayed)")
        return {"exit_code": EXIT_HOLE, "holes": holes, "backend": "hip",
                "ninja": str(ninja)}

    src = edge["source"]
    base = parsed["cuda_cflags"]
    post = parsed["cuda_post_cflags"]
    # The build edge compiles for host and device in one invocation. Device-only
    # keeps the host half's passes out of the trajectory, where they would appear
    # as real stages of a function that never runs on the GPU.
    device_only = ["--cuda-device-only"]

    endpoints = out_dir / "endpoints"
    endpoints.mkdir(parents=True, exist_ok=True)

    commands: dict[str, list[str]] = {}

    # (1) Entry point. `-print-changed` has no "before the first pass" dump, so
    # without this the earliest stage in the archive is already optimized and the
    # front end's own choices are invisible.
    frontend_out = endpoints / "frontend.ll"
    cmd_fe = [compiler, *base, *device_only, *FRONTEND_FLAGS, src, "-o", str(frontend_out)]
    commands["frontend"] = cmd_fe
    code, _out, err = runner(cmd_fe, cwd=str(scratch))
    if code != 0 or not frontend_out.is_file():
        holes.append(f"frontend:failed(exit {code}: {err.strip()[:200]}) -- the trajectory will "
                     "start at the first pass that changed something, with no unoptimized "
                     "reference to diff the first stage against")

    # (2) The trajectory itself.
    trace_filter = ["-mllvm", f"-filter-print-funcs={kernel}"] if kernel else []
    trace_obj = scratch / "trace.o"
    cmd_trace = [compiler, *base, *device_only, "-c", src, "-o", str(trace_obj),
                 *TRACE_FLAGS, *trace_filter, *post]
    commands["trace"] = cmd_trace
    code, _out, trace_text = runner(cmd_trace, cwd=str(scratch))
    if code != 0:
        holes.append(f"trace:failed(exit {code}: {trace_text.strip()[:300]})")
        return {"exit_code": EXIT_HOLE, "holes": holes, "backend": "hip",
                "ninja": str(ninja), "commands": {k: shlex.join(v) for k, v in commands.items()}}
    if len(trace_text) > MAX_TRACE_BYTES:
        holes.append(f"trace:oversize({len(trace_text)} bytes; truncated to {MAX_TRACE_BYTES})")
        trace_text = trace_text[:MAX_TRACE_BYTES]

    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (out_dir / "raw" / "print-changed.txt").write_text(trace_text, encoding="utf-8")

    stages, stage_holes = split_stages(trace_text)
    holes.extend(stage_holes)
    written = write_stages(out_dir, stages)

    # (3) Provenance. The build's own flags and nothing else -- adding an
    # evidence flag here would compare the measured binary against an object this
    # module perturbed, which proves the perturbation and not the identity.
    provenance: dict = {"checked": False, "ir_binary_equals_measured": False,
                        "reason": "no --isa-archive was supplied, so this trajectory is not "
                                  "tied to the binary that was measured"}
    if isa_archive is not None:
        probe_dir = scratch / "probe"
        probe_dir.mkdir(parents=True, exist_ok=True)
        probe_obj = probe_dir / "probe.o"
        cmd_probe = [compiler, *base, "-c", src, "-o", str(probe_obj), *post]
        commands["provenance_build"] = cmd_probe
        code, _out, err = runner(cmd_probe, cwd=str(scratch))
        if code != 0:
            provenance = {"checked": False, "ir_binary_equals_measured": False,
                          "reason": f"the unflagged replay failed (exit {code}: "
                                    f"{err.strip()[:200]}), so nothing could be compared"}
        else:
            probe_archive = scratch / "probe_isa"
            try:
                import isa_capture
                isa_capture.capture(probe_archive, source_root, [probe_dir], force=True)
                provenance = compare_to_measured(probe_archive, Path(isa_archive), kernel)
                if probe_archive.is_dir():
                    shutil.copytree(probe_archive, out_dir / "provenance",
                                    dirs_exist_ok=True)
            except Exception as exc:  # noqa: BLE001
                provenance = {"checked": False, "ir_binary_equals_measured": False,
                              "reason": f"the probe object could not be archived ({exc})"}

    if not provenance.get("ir_binary_equals_measured"):
        holes.append(f"provenance:unproven({provenance.get('reason', 'unknown')})")

    if not written:
        exit_code = EXIT_HOLE
        holes.append("capture:nothing(no pass changed the IR, which on a real kernel means the "
                     "flags did not apply rather than that the kernel is already optimal)")
    elif holes:
        exit_code = EXIT_PARTIAL
    else:
        exit_code = EXIT_OK

    return {
        "exit_code": exit_code,
        "backend": "hip",
        "ninja": str(ninja),
        "compiled_source": src,
        "edited_source": twin_origin(src),
        "kernel_filter": kernel,
        "commands": {k: shlex.join(v) for k, v in commands.items()},
        "stages": written,
        "stage_count": len(written),
        "endpoints": sorted(p.name for p in endpoints.iterdir()) if endpoints.is_dir() else [],
        "provenance": provenance,
        "holes": holes,
    }


def twin_origin(compiled: str) -> str | None:
    """The file an engineer edits, given the hipified twin ninja compiles.

    Returned as its own manifest field rather than folded into `compiled_source`
    because the two are genuinely different files and every IR source location
    in this archive names the second one.
    """
    name = Path(compiled).name
    if name.endswith("_hip.hip"):
        return str(Path(compiled).with_name(name[: -len("_hip.hip")] + ".hip"))
    return None


def capture_triton(out_dir: Path, source_root: Path, *, command: str | None,
                   kernel: str | None, runner, scratch: Path) -> dict:
    holes: list[str] = []
    if not command:
        return {"exit_code": EXIT_HOLE, "backend": "triton",
                "holes": ["triton:no_command(--triton-command is required: this backend has no "
                          "build file to read, so the caller must name the command that "
                          "compiles the kernel)"]}

    endpoints = out_dir / "endpoints"
    endpoints.mkdir(parents=True, exist_ok=True)
    cache = scratch / "triton-cache"
    cache.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.update({
        # An isolated cache plus a forced compile, because a cache hit produces
        # no pass banners at all and an empty trajectory is indistinguishable
        # from a kernel whose passes changed nothing.
        "TRITON_CACHE_DIR": str(cache),
        "TRITON_ALWAYS_COMPILE": "1",
        "TRITON_KERNEL_DUMP": "1",
        "TRITON_DUMP_DIR": str(endpoints),
        "MLIR_ENABLE_DUMP": "1",
    })

    code, _out, trace_text = runner(["bash", "-lc", command], cwd=str(source_root), env=env)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (out_dir / "raw" / "mlir-dump.txt").write_text(trace_text, encoding="utf-8")
    if code != 0:
        holes.append(f"triton:command_failed(exit {code}: {trace_text.strip()[-300:]})")

    stages, stage_holes = split_mlir_stages(trace_text)
    holes.extend(stage_holes)
    written = write_stages(out_dir, stages)

    dumped = sorted(p.name for p in endpoints.iterdir()) if endpoints.is_dir() else []
    if len(dumped) > 1 and not kernel:
        holes.append(f"triton:ambiguous_kernel({len(dumped)} kernels were dumped {dumped[:6]} "
                     "and none was selected; binding a trajectory to the measured winner is the "
                     "caller's job, and choosing here would be a guess)")

    if not written:
        exit_code = EXIT_HOLE
    elif holes:
        exit_code = EXIT_PARTIAL
    else:
        exit_code = EXIT_OK
    return {
        "exit_code": exit_code, "backend": "triton", "kernel_filter": kernel,
        "commands": {"triton": command},
        "stages": written, "stage_count": len(written), "endpoints": dumped,
        "provenance": {"checked": False, "ir_binary_equals_measured": False,
                       "reason": "the Triton backend recompiles into an isolated cache; tying "
                                 "that code object to the measured one is not implemented, so "
                                 "this trajectory is unproven provenance by construction"},
        "holes": holes,
    }


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------

def toolchain_facts(runner) -> dict:
    facts: dict = {}
    clang = shutil.which("amdclang++") or "/opt/rocm/llvm/bin/amdclang++"
    if Path(clang).is_file():
        code, out, _err = runner([clang, "--version"])
        if code == 0:
            facts["compiler_version"] = out.strip().splitlines()[0] if out.strip() else ""
    version_file = Path("/opt/rocm/.info/version")
    if version_file.is_file():
        try:
            facts["rocm_version"] = version_file.read_text(encoding="utf-8").strip()
        except OSError:
            pass
    return facts


def capture(out_dir: Path, source_root: Path, *, backend: str = "auto",
            ninja: Path | None = None, source_file: str | None = None,
            kernel: str | None = None, isa_archive: Path | None = None,
            triton_command: str | None = None, force: bool = False,
            runner=default_runner) -> dict:
    # Absolute, because every replayed command runs with `cwd` set to the scratch
    # tree. A relative `--out` then points at a directory that does not exist
    # there, and the compiler's `-o` fails with a path error that reads like a
    # permissions problem rather than like the caller's argument.
    out_dir = Path(out_dir).resolve()
    source_root = Path(source_root).resolve()

    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        return {"schema": SCHEMA, "exit_code": EXIT_ERROR, "archive": str(out_dir),
                "holes": [f"archive:not_empty({out_dir}; an IR archive is immutable once "
                          "written so a round cannot silently re-attribute its evidence -- "
                          "pass --force only when replacing a capture on purpose)"]}
    out_dir.mkdir(parents=True, exist_ok=True)

    holes: list[str] = []
    try:
        import source_hash as source_hash_mod
        digest = source_hash_mod.tree_hash(source_root)
    except Exception as exc:  # noqa: BLE001
        digest = None
        holes.append(f"source_hash:failed({exc}) -- these stages have no owner, so nothing "
                     "downstream can prove which tree produced them")

    if backend == "auto":
        backend = "triton" if triton_command else "hip"

    with tempfile.TemporaryDirectory(prefix="ir_capture_") as tmp:
        scratch = Path(tmp)
        if backend == "hip":
            result = capture_hip(out_dir, source_root, ninja=ninja, source_file=source_file,
                                 kernel=kernel, isa_archive=isa_archive, runner=runner,
                                 scratch=scratch)
        elif backend == "triton":
            result = capture_triton(out_dir, source_root, command=triton_command,
                                    kernel=kernel, runner=runner, scratch=scratch)
        else:
            result = {"exit_code": EXIT_ERROR, "backend": backend,
                      "holes": [f"backend:unsupported({backend!r}). This is reported rather "
                                "than degraded to an empty archive: a backend with no adapter "
                                "has NO trajectory, which is a different fact from a kernel "
                                "whose passes changed nothing)"]}

    holes.extend(result.pop("holes", []))
    manifest = {
        "schema": SCHEMA,
        "archive": str(out_dir),
        "source_root": str(source_root),
        "source_hash": digest,
        "toolchain": toolchain_facts(runner),
        "holes": holes,
        **result,
    }
    if holes and manifest["exit_code"] == EXIT_OK:
        manifest["exit_code"] = EXIT_PARTIAL
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--out", required=True, help="archive directory to create")
    parser.add_argument("--source-root", required=True,
                        help="candidate-owned source tree, hashed into the manifest so the "
                             "trajectory has an owner")
    parser.add_argument("--backend", default="auto", choices=("auto", "hip", "triton"))
    parser.add_argument("--ninja", default=None,
                        help="path to the build.ninja torch wrote, when it cannot be found "
                             "from --source-root or more than one exists")
    parser.add_argument("--source-file", default=None,
                        help="which device translation unit to trace, named either as the file "
                             "an engineer edits (X.hip) or as the hipified twin (X_hip.hip)")
    parser.add_argument("--kernel", default=None,
                        help="mangled kernel symbol to narrow the trace to "
                             "(-mllvm -filter-print-funcs)")
    parser.add_argument("--isa-archive", default=None,
                        help="the ISA archive of the binary that was MEASURED. Without it the "
                             "trajectory cannot be tied to the measured kernel and the "
                             "manifest says so.")
    parser.add_argument("--triton-command", default=None,
                        help="for --backend triton: the command that compiles the kernel")
    parser.add_argument("--force", action="store_true",
                        help="replace a non-empty archive directory")
    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv[1:])
    manifest = capture(
        Path(args.out), Path(args.source_root), backend=args.backend,
        ninja=Path(args.ninja) if args.ninja else None,
        source_file=args.source_file, kernel=args.kernel,
        isa_archive=Path(args.isa_archive) if args.isa_archive else None,
        triton_command=args.triton_command, force=args.force)
    _emit(manifest)
    return int(manifest["exit_code"])


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except OSError as exc:
        print(json.dumps({"schema": SCHEMA, "exit_code": EXIT_ERROR, "error": str(exc)},
                         sort_keys=True))
        sys.exit(EXIT_ERROR)
