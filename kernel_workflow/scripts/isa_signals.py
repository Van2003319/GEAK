#!/usr/bin/env python3
"""Machine-readable signals over the AMDGCN a candidate actually compiled to.

Finding (87) wired `hip_twin_sync.py` because an edit applied to the primary of a
hipify twin pair is never compiled: the build succeeds, correctness passes, and
the benchmark reports a clean null result for a change that never ran. The
comment at that gate states the cost exactly -- "it is indistinguishable from an
honest negative -- and an honest negative is what closes a search direction. So
the cost of missing this is not one wasted round, it is a mechanism written off."

This module answers the next question in the same family. `hip_twin_sync`
establishes that the file which was edited is the file that was compiled. It
cannot establish that the *mechanism* which was written survived compilation. An
engineer that widens a staging load to `uint4`, moves an accumulator onto the
matrix core, or deletes a conversion burst can be silently undone by the backend
-- vectorization refused for an alignment it could not prove, a builtin lowered
to the opcodes it was meant to replace, a conversion re-materialized. Every
existing check still passes, the candidate measures within noise of its parent,
and the round is recorded as "tried X, no effect". That ledger entry is false,
and `tech_lead` will not propose X again. In greedy search the ledger is the only
memory there is, so one false negative closes a direction permanently.

So this reads the opcodes. `verify_engineer.md` already established the principle
-- "Classify `compute_primitive` from the disassembly, never from the include
list ... You are the independent check on a classification the engineer made from
its own source; reading the same source back is not independence" -- and gave the
reason it matters on this hardware: on
gfx942 rocWMMA lowers its 32x32x8 bf16 fragment to `v_mfma_f32_16x16x16_bf16`,
so two sources that differ can compile to byte-identical machine code. What is
checked here is the converse: two sources that differ can also compile to *the
same* machine code, because the edit did not take.

SIGNALS ONLY. Nothing here decides whether a kernel is good, and nothing here
produces a speedup, a headroom or a score. `compare-perf`-equivalent verified
geomean remains the sole authority on performance (the same separation cannot be
`sol_*` and `peak_pct`, per `profile_engineer.md` finding (89)). What this
produces is one narrow, mechanical, falsifiable fact per claim: did the machine
code move in the direction the candidate said it would.

NEVER FABRICATES. Every signal that cannot be read is reported `unavailable` or
`indeterminate` and never as a zero or a False. `asm_loop_audit.py` states the
reason for that rule on the LDS field it shares with this module: a structural
zero reads as "this kernel uses no LDS", which is a confident wrong answer, and a
confident wrong answer is worse than declining.

RELATION TO `asm_loop_audit.py`. That tool
(`perf_knowledge/expert_skills/skills/gluon_authoring/scripts/asm_loop_audit.py`)
remains the deep, human- and agent-facing hot-loop auditor: it scopes to the
innermost back-edge loop, prints the op-class symbol stream, and leaves the
structural-vs-schedulable verdict to the reader. This module is the
machine-readable gate: whole-kernel scope, JSON out, stable keys, no verdict. The
two wait spellings and the ALU-wait split below are the rules that tool
established, restated here rather than imported because it lives under an expert
skill that `kernel_lane.js` gates off by default (`use_expert_skills=false`), and
a gate must not depend on an optional tree. The op-class regexes and the
`s_nop`-imm summing convention are kept deliberately identical so the two tools'
numbers stay comparable; `test_isa_signals.py` pins the wait cases against the
same assertions that tool's own selftest carries.

SCOPE: AMDGCN ONLY. Every op class, access-width suffix, wait spelling and metadata
field below is an AMDGCN or AMDGPU-metadata name. NVIDIA SASS spells all of it
differently -- HMMA for the matrix op, LDS/LDG.E.128 for the memory ops, BAR.SYNC
for the barrier -- so a cubin fed through here would classify as entirely `other`
and every claim would come back `indeterminate`: a run that looks checked and is
not. Supporting SASS means a second vocabulary and a second disassembler, not a
wider language list in the lane.

Usage:
  isa_signals.py signals --archive <dir>
  isa_signals.py diff --from <dir> --to <dir> [--claim <id> ...]
  isa_signals.py checks --archive <dir>
  isa_signals.py claims

Exit codes (the `hip_twin_sync.py` convention, because the callers already read
it that way and a second spelling of "nothing was checked" is how a HOLE gets
summarized into a pass):
  0  complete: at least one kernel read, with register/LDS metadata
  2  HOLE: nothing was checked -- no kernel found in the archive
  3  partial: kernels read but some evidence was unavailable
  1  usage or read error
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

SCHEMA_SIGNALS = "geak.isa-signals/v1"
SCHEMA_DIFF = "geak.isa-diff/v1"
SCHEMA_CHECKS = "geak.isa-checks/v1"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_HOLE = 2
EXIT_PARTIAL = 3

# Op classes. DISJOINT by construction and first-match-wins, so `valu` excludes
# the conversion and accvgpr families that are broken out above it -- both are
# `v_` instructions whose *counts* are the signal, and folding them into a single
# `valu` bucket is the granularity `asm_loop_audit.py` calls "one level too
# coarse to act on": a dtype-conversion sequence, an address chain and a layout
# permute all land in `valu` and have opposite fixes.
_CLASSES = [
    ("mfma", re.compile(r"^v_mfma|^v_smfmac|^v_wmma")),
    ("atomic", re.compile(r"^(global|buffer|flat|ds|scratch)_atomic")),
    ("lds_read", re.compile(r"^ds_read|^ds_load")),
    ("lds_write", re.compile(r"^ds_write|^ds_store")),
    ("global_load", re.compile(r"^(global|buffer|flat|scratch)_load")),
    ("global_store", re.compile(r"^(global|buffer|flat|scratch)_store")),
    # ALU-dependency waits are split OUT of the memory-wait class: `s_wait_alu
    # depctr_*` (gfx11/12) and `s_delay_alu` (gfx11) resolve a register hazard,
    # not an outstanding memory op, so folding them in inflates the drain ratio.
    ("alu_wait", re.compile(r"^s_wait_alu|^s_delay_alu")),
    ("wait", re.compile(r"^s_waitcnt|^s_wait_")),
    ("barrier", re.compile(r"^s_barrier")),
    ("nop", re.compile(r"^s_nop")),
    ("conversion", re.compile(r"^v_cvt")),
    ("accvgpr_move", re.compile(r"^v_accvgpr_(read|write)")),
    ("valu", re.compile(r"^v_")),
    ("salu", re.compile(r"^s_")),
]
CLASS_NAMES = tuple(name for name, _rx in _CLASSES) + ("other",)

# Access-width suffixes, longest first because `global_load_dwordx4` contains
# `dword`. Value is BYTES PER LANE for one access.
_WIDTH_SUFFIXES = (
    ("dwordx4", 16), ("dwordx3", 12), ("dwordx2", 8), ("dwordx1", 4), ("dword", 4),
    ("b128", 16), ("b96", 12), ("b64", 8), ("b32", 4), ("b16", 2), ("b8", 1),
    ("ushort", 2), ("sshort", 2), ("short", 2), ("u16", 2), ("i16", 2),
    ("ubyte", 1), ("sbyte", 1), ("byte", 1), ("u8", 1), ("i8", 1),
)

# Two wait spellings; matching only the first is why every RDNA kernel used to
# report "100% full drains" in the tool this is restated from.
#   gfx9      s_waitcnt vmcnt(0) lgkmcnt(1)   -- counter NAMED, value in parens
#   gfx11/12  s_wait_dscnt 0x1                -- counter in the MNEMONIC, imm value
#             s_waitcnt_vscnt null, 0x0       -- gfx10 store-counter spelling
_CNT_RE = re.compile(r"(\w*cnt)\s*\(\s*(\d+)\s*\)")
# The counter group excludes `_` deliberately: `_` separates fused counters, so
# letting the inner class match it makes the match exponential in the number of
# `cnt_` repetitions.
_WAIT_IMM_RE = re.compile(
    r"^s_wait(?:cnt)?_([a-z]*cnt(?:_[a-z]*cnt)*)\s+(?:null\s*,\s*)?"
    r"(?:0[xX]([0-9a-fA-F]+)|(\d+))\s*$")

# Symbol label, two emitters:
#   (a) `llvm-objdump -d` of a code object:  "0000000000001234 <_Z3fooPf>:"
#   (b) plain assembler `.s` / `.amdgcn`:    "_Z3fooPf:"
_LABEL_RE = re.compile(
    r"^\s*(?:[0-9a-fA-F]+\s+)?<(?P<obj>[\w.$@]+)>:\s*$"
    r"|^\s*(?P<asm>\.?\w[\w.$@]*):\s*(?:;.*)?$")

# AMDGPU metadata, from `llvm-readelf --notes` over a code object OR from the
# `.amdgpu_metadata` YAML tail of a `.s`. Both spell the kernel list the same way
# (`amdhsa.kernels:` then `- .agpr_count: 0`), so one parser covers both.
_MD_ENTRY_RE = re.compile(r"^\s*-\s+\.(\w+):\s*(.*?)\s*$")
_MD_FIELD_RE = re.compile(r"^\s*\.(\w+):\s*(.*?)\s*$")

_MD_NUMERIC = {
    "vgpr_count": "vgpr_count",
    "agpr_count": "agpr_count",
    "sgpr_count": "sgpr_count",
    "private_segment_fixed_size": "scratch_bytes",
    "group_segment_fixed_size": "lds_bytes",
    "vgpr_spill_count": "vgpr_spill_count",
    "sgpr_spill_count": "sgpr_spill_count",
    "max_flat_workgroup_size": "max_flat_workgroup_size",
}


def _emit(payload: dict) -> None:
    """One writer, so `sort_keys=True` is a property of the module rather than of
    each call site. These receipts are diffed against each other across rounds."""
    print(json.dumps(payload, sort_keys=True, indent=2))


def mnemonic(line: str) -> str | None:
    """First token of a real instruction line, else None.

    Handles the trailing encoding comment `llvm-objdump` appends
    (`s_endpgm  // 000000000100: BF810000`) by splitting the comment off first.
    """
    text = re.split(r";|//", line, maxsplit=1)[0].strip()
    if not text or text.startswith(("#", ".")) or text.endswith(":"):
        return None
    return text.split()[0]


def classify(mn: str) -> str:
    for name, rx in _CLASSES:
        if rx.match(mn):
            return name
    return "other"


def access_bytes(mn: str) -> int | None:
    """Bytes per lane for one memory access, or None when no suffix is recognised.

    None is a real answer and is counted separately (`width_unparsed`): a width
    this parser cannot name must not be averaged in as a 4. Matching is anchored
    -- the suffix must END the mnemonic or appear as a `_suffix_` segment -- so an
    unfamiliar opcode declines rather than being guessed at from a substring that
    happens to occur inside it.
    """
    for suffix, width in _WIDTH_SUFFIXES:
        if mn.endswith(f"_{suffix}") or f"_{suffix}_" in mn:
            return width
    return None


def classify_wait(line: str):
    """(kind, [(counter, value), ...]) for one wait instruction.

    kind is 'drain' (every named counter 0 -- serialized), 'relaxed' (some
    counter > 0 -- pipelined), 'alu' (register-dependency wait, not a memory
    drain), or 'unknown' (a spelling this parser does not know). An unknown is
    counted as a drain by the caller, conservatively, AND reported, so a new
    spelling surfaces as a parser gap instead of quietly biasing the ratio.
    """
    text = re.split(r";|//", line, maxsplit=1)[0].strip()
    if not text:
        return "unknown", []
    mn = text.split()[0]
    if mn.startswith(("s_wait_alu", "s_delay_alu")):
        return "alu", [(mn, -1)]
    counters = [(k, int(v)) for k, v in _CNT_RE.findall(text)]
    if not counters:
        m = _WAIT_IMM_RE.match(text)
        if m:
            value = int(m.group(2), 16) if m.group(2) else int(m.group(3))
            counters = [(m.group(1), value)]
    if not counters:
        return "unknown", []
    return ("drain" if all(v == 0 for _k, v in counters) else "relaxed"), counters


def kernel_spans(lines: list[str]) -> list[tuple[str, int, int]]:
    """(symbol, first_instruction_index, end_exclusive) for every kernel body.

    A body runs from its symbol label to its own `s_endpgm`. Basic-block labels
    (`.LBB*`, `.Lfunc*`, `%bb.*`) are not kernel entries and are skipped, so a
    kernel with control flow is one span and not several.
    """
    spans: list[tuple[str, int, int]] = []
    current: tuple[str, int] | None = None
    for i, line in enumerate(lines):
        m = _LABEL_RE.match(line)
        if m:
            name = m.group("obj") or m.group("asm")
            if name.startswith((".L", ".Lfunc", ".Ltmp")):
                continue
            current = (name, i + 1)
            continue
        if mnemonic(line) == "s_endpgm" and current is not None:
            spans.append((current[0], current[1], i + 1))
            current = None
    return spans


def analyze_body(lines: list[str], start: int, end: int) -> dict:
    classes: Counter = Counter()
    opcodes: Counter = Counter()
    load_bytes: Counter = Counter()
    store_bytes: Counter = Counter()
    lds_bytes: Counter = Counter()
    mfma_shapes: set[str] = set()
    relaxed = drain = unknown_wait = alu_waits = 0
    nops = nop_cycles = 0
    width_unknown = 0
    lds_multi = 0

    for line in lines[start:end]:
        mn = mnemonic(line)
        if mn is None:
            continue
        name = classify(mn)
        classes[name] += 1
        opcodes[mn] += 1

        if name in ("global_load", "global_store", "lds_read", "lds_write"):
            width = access_bytes(mn)
            if width is None:
                width_unknown += 1
            elif name == "global_load":
                load_bytes[width] += 1
            elif name == "global_store":
                store_bytes[width] += 1
            else:
                lds_bytes[width] += 1
                if re.search(r"(read2|write2)", mn):
                    lds_multi += 1
        if name == "mfma":
            shape = re.search(r"(\d+x\d+x\d+)", mn)
            mfma_shapes.add(shape.group(1) if shape else mn)
        if name == "nop":
            nops += 1
            m = re.match(r"^s_nop\s+(?:0[xX]([0-9a-fA-F]+)|(\d+))", line.strip())
            if m:
                nop_cycles += int(m.group(1), 16) if m.group(1) else int(m.group(2))
        if name == "wait":
            kind, _counters = classify_wait(line)
            if kind == "drain":
                drain += 1
            elif kind == "relaxed":
                relaxed += 1
            else:
                drain += 1
                unknown_wait += 1
        if name == "alu_wait":
            alu_waits += 1

    wait_total = relaxed + drain
    return {
        "instructions": sum(classes.values()),
        "classes": {k: classes.get(k, 0) for k in CLASS_NAMES if classes.get(k)},
        "opcodes": dict(opcodes),
        "global_load_bytes": _width_block(load_bytes),
        "global_store_bytes": _width_block(store_bytes),
        "lds_access_bytes": _width_block(lds_bytes),
        "lds_multi_access": lds_multi,
        "width_unparsed": width_unknown,
        "mfma_shapes": sorted(mfma_shapes),
        "waits": {
            "relaxed": relaxed, "full_drain": drain, "alu": alu_waits,
            "unrecognised_counted_as_drain": unknown_wait,
            # None, not 1.0, when there is nothing to take a ratio of. A kernel
            # with no memory waits has no drain quality, and 0.0 would read as
            # "perfectly pipelined".
            "drain_ratio": (drain / wait_total) if wait_total else None,
        },
        "barriers": classes.get("barrier", 0),
        "nops": nops,
        "nop_stall_cycles": nop_cycles,
        "accvgpr_moves": classes.get("accvgpr_move", 0),
        "conversions": classes.get("conversion", 0),
    }


def _width_block(hist: Counter) -> dict:
    return {
        "max": max(hist) if hist else None,
        "histogram": {str(k): v for k, v in sorted(hist.items())},
        "accesses": sum(hist.values()),
    }


def analyze_disasm(text: str) -> dict[str, dict]:
    lines = text.splitlines()
    out: dict[str, dict] = {}
    for name, start, end in kernel_spans(lines):
        body = analyze_body(lines, start, end)
        if name in out:
            # Two spans under one symbol means the disassembly holds the same
            # kernel twice (multiple code objects for one arch, or a bundle
            # disassembled twice). Keeping the larger is a guess; refuse instead.
            out[name]["duplicate_spans"] = out[name].get("duplicate_spans", 1) + 1
            continue
        out[name] = body
    return out


def parse_kernel_metadata(text: str) -> list[dict]:
    """Kernel register/LDS/scratch facts from AMDGPU metadata.

    `scratch_bytes` is `.private_segment_fixed_size` and is the single
    highest-value field here: nonzero means the kernel spills to HBM, which
    `isa_verify.md` records as a 3-5x slow path and marks "MUST be 0".
    """
    entries: list[dict] = []
    current: dict | None = None
    for line in text.splitlines():
        m = _MD_ENTRY_RE.match(line)
        if m:
            current = {}
            entries.append(current)
            _md_assign(current, m.group(1), m.group(2))
            continue
        m = _MD_FIELD_RE.match(line)
        if m and current is not None:
            _md_assign(current, m.group(1), m.group(2))
    return [e for e in entries if e.get("name") or e.get("symbol")]


def _md_assign(entry: dict, key: str, raw: str) -> None:
    value = raw.strip().strip("'\"")
    if key == "name":
        entry["name"] = value
    elif key == "symbol":
        entry["symbol"] = value
        entry.setdefault("name", value[:-3] if value.endswith(".kd") else value)
    elif key in _MD_NUMERIC:
        try:
            entry[_MD_NUMERIC[key]] = int(value)
        except ValueError:
            pass


def archive_unreadable(archive: Path) -> str | None:
    """Why the archive cannot be read, or None when it can.

    A gate must not turn an unreachable path into a traceback, and it must not
    turn one into a silent pass either. Both were live here: `is_file()` raises
    `PermissionError` rather than returning False when a parent directory denies
    stat, so a mistyped `--archive` exited 1 with an errno in place of a verdict.
    The answer is the third option -- exit HOLE, which every caller already reads
    as "nothing was checked", and name the path in `unavailable` so a typo is
    still diagnosable instead of reading as an honest empty capture.
    """
    try:
        if not archive.exists():
            return f"{archive} does not exist"
        if not archive.is_dir():
            return f"{archive} is not a directory"
    except OSError as exc:
        return f"{archive} cannot be stat'd: {exc}"
    return None


def read_archive(archive: Path) -> dict:
    """Load an `isa_capture.py` archive. Tolerant of a bare directory of dumps so
    the tool is usable by hand on a `.s` a human dumped, but a manifest is what
    ties evidence to a source tree and its absence is reported, not assumed away.
    """
    manifest = None
    manifest_path = archive / "manifest.json"
    try:
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = None
    disasm_text = _concat(archive, ("*.disasm.txt", "*.s", "*.amdgcn"))
    notes_text = _concat(archive, ("*.notes.txt",))
    return {
        "manifest": manifest if isinstance(manifest, dict) else None,
        "disasm": disasm_text,
        "notes": notes_text or disasm_text,  # a .s carries its own metadata tail
    }


def _concat(archive: Path, patterns: tuple[str, ...]) -> str:
    chunks: list[str] = []
    for pattern in patterns:
        try:
            paths = sorted(archive.rglob(pattern))
        except OSError:
            continue
        for path in paths:
            try:
                if path.is_file():
                    chunks.append(path.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
    return "\n".join(chunks)


def _empty_signals(archive: Path, unavailable: list[str]) -> dict:
    return {
        "schema": SCHEMA_SIGNALS, "exit_code": EXIT_HOLE, "archive": str(archive),
        "arch": None, "source_hash": None, "kernel_count": 0, "kernels": [],
        "unavailable": sorted(unavailable),
    }


def build_signals(archive: Path) -> dict:
    why = archive_unreadable(archive)
    if why is not None:
        return _empty_signals(archive, [f"archive:unreadable({why})"])
    raw = read_archive(archive)
    kernels_asm = analyze_disasm(raw["disasm"])
    metadata = {e["name"]: e for e in parse_kernel_metadata(raw["notes"]) if e.get("name")}
    unavailable: list[str] = []
    manifest = raw["manifest"] or {}

    if not raw["manifest"]:
        unavailable.append("manifest:absent(no manifest.json, so these signals are not tied to a "
                           "source_hash and must not be quoted as evidence about a named candidate)")

    kernels = []
    for name in sorted(kernels_asm):
        body = dict(kernels_asm[name])
        md = metadata.get(name)
        if md is None:
            body["resources"] = {"available": False,
                                 "why": "no AMDGPU metadata entry for this symbol"}
            unavailable.append(f"resources:{name}")
        else:
            body["resources"] = {
                "available": True,
                "vgpr_count": md.get("vgpr_count"),
                "agpr_count": md.get("agpr_count"),
                "sgpr_count": md.get("sgpr_count"),
                "scratch_bytes": md.get("scratch_bytes"),
                "lds_bytes": md.get("lds_bytes"),
                "vgpr_spill_count": md.get("vgpr_spill_count"),
            }
        body["name"] = name
        kernels.append(body)

    if not kernels:
        exit_code = EXIT_HOLE
    elif unavailable:
        exit_code = EXIT_PARTIAL
    else:
        exit_code = EXIT_OK

    # The digests of the code objects themselves, when the capture recorded them.
    # Sorted, because the ORDER objects were sliced out of a fat binary is not
    # meaningful and two captures of the same build must compare equal.
    digests = sorted(
        str(o["sha256"]) for o in (manifest.get("objects") or [])
        if isinstance(o, dict) and o.get("sha256"))

    return {
        "schema": SCHEMA_SIGNALS,
        "exit_code": exit_code,
        "archive": str(archive),
        "arch": manifest.get("arch"),
        "source_hash": manifest.get("source_hash"),
        "kernel_count": len(kernels),
        "kernels": kernels,
        "code_object_digests": digests,
        "unavailable": sorted(unavailable),
    }


# ---------------------------------------------------------------------------
# Claims. A claim is what the candidate said its edit would do to the machine
# code, in a CLOSED vocabulary of things that are mechanically observable. The
# vocabulary is closed for the reason `source_hash.extract_descriptor_evidence`
# is conservative: an open-ended claim is one the checker has to interpret, and a
# checker that interprets is a checker that can be talked into a pass.
#
# Three verdicts, and the third is load-bearing. `indeterminate` is NOT a soft
# fail: it means the evidence needed to judge this claim was not in the archive,
# and reporting that as `realized: false` would manufacture exactly the false
# negative this module exists to prevent.
# ---------------------------------------------------------------------------

def _max_or_none(block) -> int | None:
    return block.get("max") if isinstance(block, dict) else None


def _res(kernel: dict, field: str):
    res = kernel.get("resources") or {}
    return res.get(field) if res.get("available") else None


def _grew(before, after):
    if before is None or after is None:
        return None
    return after > before


def _shrank(before, after):
    if before is None or after is None:
        return None
    return after < before


def _appeared(before: int, after: int):
    return before == 0 and after > 0


def _claim_widen_global_load(a, b):
    before, after = _max_or_none(a["global_load_bytes"]), _max_or_none(b["global_load_bytes"])
    return _grew(before, after), f"max global load bytes/lane {before} -> {after}"


def _claim_widen_global_store(a, b):
    before, after = _max_or_none(a["global_store_bytes"]), _max_or_none(b["global_store_bytes"])
    return _grew(before, after), f"max global store bytes/lane {before} -> {after}"


def _claim_widen_lds_access(a, b):
    before, after = _max_or_none(a["lds_access_bytes"]), _max_or_none(b["lds_access_bytes"])
    return _grew(before, after), f"max LDS access bytes/lane {before} -> {after}"


def _claim_introduce_matrix_core(a, b):
    before = a["classes"].get("mfma", 0)
    after = b["classes"].get("mfma", 0)
    return _appeared(before, after), (
        f"matrix-core instructions {before} -> {after}"
        f" (shapes {a['mfma_shapes']} -> {b['mfma_shapes']})")


def _claim_change_mfma_shape(a, b):
    before, after = a["mfma_shapes"], b["mfma_shapes"]
    if not before and not after:
        return None, "neither build has a matrix-core instruction"
    return before != after, f"matrix-core shapes {before} -> {after}"


def _claim_introduce_lds_staging(a, b):
    before = a["classes"].get("lds_read", 0) + a["classes"].get("lds_write", 0)
    after = b["classes"].get("lds_read", 0) + b["classes"].get("lds_write", 0)
    return _appeared(before, after), f"LDS accesses {before} -> {after}"


def _claim_remove_spill(a, b):
    before, after = _res(a, "scratch_bytes"), _res(b, "scratch_bytes")
    if before is None or after is None:
        return None, "scratch bytes unavailable on one side (no AMDGPU metadata)"
    if before == 0:
        return None, "the parent build already spills 0 bytes; nothing to remove"
    return after == 0, f"scratch bytes {before} -> {after}"


def _claim_reduce_vgpr(a, b):
    before, after = _res(a, "vgpr_count"), _res(b, "vgpr_count")
    return _shrank(before, after), f"vgpr_count {before} -> {after}"


def _claim_reduce_lds(a, b):
    before, after = _res(a, "lds_bytes"), _res(b, "lds_bytes")
    return _shrank(before, after), f"lds_bytes {before} -> {after}"


def _claim_relax_waitcnt(a, b):
    before, after = a["waits"]["drain_ratio"], b["waits"]["drain_ratio"]
    if before is None or after is None:
        return None, "one build has no memory waits, so it has no drain quality"
    return after < before, f"full-drain ratio {before:.3f} -> {after:.3f}"


def _claim_reduce_barriers(a, b):
    before, after = a["barriers"], b["barriers"]
    return after < before, f"barriers {before} -> {after}"


def _claim_remove_conversion_burst(a, b):
    before, after = a["conversions"], b["conversions"]
    return after < before, f"v_cvt instructions {before} -> {after}"


def _claim_remove_accvgpr_moves(a, b):
    before, after = a["accvgpr_moves"], b["accvgpr_moves"]
    if before == 0:
        return None, "the parent build has no accvgpr moves; nothing to remove"
    return after < before, f"v_accvgpr_read/write {before} -> {after}"


def _claim_introduce_atomics(a, b):
    before, after = a["classes"].get("atomic", 0), b["classes"].get("atomic", 0)
    return _appeared(before, after), f"atomic instructions {before} -> {after}"


def _claim_none_observable(_a, _b):
    return None, ("declared to have no ISA signature (host, wrapper, launch-shape or "
                  "algorithmic change); the machine-code diff neither confirms nor "
                  "refutes it and must not be cited either way")


CLAIMS = {
    "widen_global_load": _claim_widen_global_load,
    "widen_global_store": _claim_widen_global_store,
    "widen_lds_access": _claim_widen_lds_access,
    "introduce_matrix_core": _claim_introduce_matrix_core,
    "change_mfma_shape": _claim_change_mfma_shape,
    "introduce_lds_staging": _claim_introduce_lds_staging,
    "remove_spill": _claim_remove_spill,
    "reduce_vgpr": _claim_reduce_vgpr,
    "reduce_lds": _claim_reduce_lds,
    "relax_waitcnt": _claim_relax_waitcnt,
    "reduce_barriers": _claim_reduce_barriers,
    "remove_conversion_burst": _claim_remove_conversion_burst,
    "remove_accvgpr_moves": _claim_remove_accvgpr_moves,
    "introduce_atomics": _claim_introduce_atomics,
    "none_observable": _claim_none_observable,
}


def _by_name(signals: dict) -> dict[str, dict]:
    return {k["name"]: k for k in signals.get("kernels", [])}


def _identical(a: dict, b: dict) -> bool:
    """Byte-for-byte equal codegen as far as these signals can see: the same
    opcode multiset AND the same register/LDS/scratch budget."""
    if a.get("opcodes") != b.get("opcodes"):
        return False
    fields = ("vgpr_count", "agpr_count", "sgpr_count", "scratch_bytes", "lds_bytes")
    return all(_res(a, f) == _res(b, f) for f in fields)


def diff_signals(before: dict, after: dict, claims: list[str],
                 hot_kernels: list[str] | None = None) -> dict:
    hot_kernels = [k for k in (hot_kernels or []) if k]
    left, right = _by_name(before), _by_name(after)
    shared = sorted(set(left) & set(right))
    only_before = sorted(set(left) - set(right))
    only_after = sorted(set(right) - set(left))

    per_kernel = []
    for name in shared:
        a, b = left[name], right[name]
        delta = {}
        for op in sorted(set(a["opcodes"]) | set(b["opcodes"])):
            change = b["opcodes"].get(op, 0) - a["opcodes"].get(op, 0)
            if change:
                delta[op] = change
        resource_delta = {}
        for field in ("vgpr_count", "agpr_count", "sgpr_count", "scratch_bytes", "lds_bytes"):
            x, y = _res(a, field), _res(b, field)
            if x is not None and y is not None and x != y:
                resource_delta[field] = y - x
        per_kernel.append({
            "name": name, "identical": _identical(a, b),
            "opcode_delta": delta, "resource_delta": resource_delta,
            "instructions": [a["instructions"], b["instructions"]],
        })

    # The claim-independent fact, and the one worth the most. Scoped to matched
    # kernels only: if the symbol set moved, codegen certainly changed, and
    # calling that "unchanged" would be the false reassurance in the opposite
    # direction.
    #
    # Two ways to establish it, and the stronger one is preferred when both
    # archives carry digests. `_identical` compares an opcode multiset and the
    # register/LDS/scratch budget -- which is what the disassembly can show, and
    # is blind to a change in operands, immediates or instruction ORDER. An edit
    # that reschedules a loop without changing its census reads as byte-identical
    # codegen under that test, and `mechanism_verdict` converts "unchanged" into a
    # hard `refuted` regardless of what the per-claim checkers found. So the weaker
    # test can refute a candidate that did change the binary.
    #
    # The digest can only ever move a verdict from "unchanged" to "changed", so
    # this makes the gate strictly less trigger-happy and can never newly refuse
    # anything. Which test was used is reported, because a receipt that does not
    # say how it decided cannot be audited.
    left_digests = before.get("code_object_digests") or []
    right_digests = after.get("code_object_digests") or []
    if left_digests and right_digests:
        unchanged = left_digests == right_digests
        unchanged_basis = "code_object_digest"
    else:
        unchanged = bool(shared) and not only_before and not only_after and \
            all(k["identical"] for k in per_kernel)
        unchanged_basis = "opcode_and_resource_census"

    verdicts = []
    for claim in claims:
        checker = CLAIMS.get(claim)
        if checker is None:
            verdicts.append({"claim": claim, "realized": None,
                             "evidence": "unknown claim id; run `isa_signals.py claims` for the "
                                         "closed vocabulary. An unrecognised claim is never a pass."})
            continue
        if not shared:
            verdicts.append({"claim": claim, "realized": None,
                             "evidence": "no kernel symbol is present in both archives"})
            continue
        # A claim holds for the candidate if it is realized in ANY shared kernel:
        # an operator may fuse or split kernels, and requiring every kernel to
        # move would fail a real widening that landed in the one hot kernel.
        results = []
        for name in shared:
            realized, evidence = checker(left[name], right[name])
            results.append((name, realized, evidence))
        if any(r is True for _n, r, _e in results):
            hit = next(x for x in results if x[1] is True)
            verdict = {"claim": claim, "realized": True,
                       "evidence": f"{hit[0]}: {hit[2]}",
                       # WHICH kernel satisfied it. The any-kernel rule above is
                       # deliberate and stays, but it means a claim can pass on a
                       # symbol nobody was optimizing, and the receipt did not say
                       # so. Reported, never enforced: making the hot kernel
                       # mandatory would reinstate the false negative the rule
                       # exists to prevent (`learned_rules.md`, "Two ISA-evidence
                       # validity traps", where a verified -4.72% patch came back
                       # refuted because every pre-existing symbol was legitimately
                       # unchanged).
                       "realized_in": hit[0]}
            if hot_kernels and hit[0] not in hot_kernels:
                verdict["realized_outside_target"] = True
                verdict["evidence"] += (
                    f" -- NOTE: {hit[0]} is not among the kernels this round targeted "
                    f"({', '.join(sorted(hot_kernels))}), so the mechanism landed somewhere "
                    "the plateau was not measured on")
            verdicts.append(verdict)
        elif all(r is None for _n, r, _e in results):
            verdicts.append({"claim": claim, "realized": None,
                             "evidence": "; ".join(f"{n}: {e}" for n, _r, e in results)})
        elif only_after:
            # A symbol that exists only in the candidate has no counterpart to
            # diff against, so a claim the shared kernels do not carry may still
            # have been realized in it and this comparison cannot see it.
            # Reporting that as False is the missing-evidence-as-refutation error
            # `mechanism_verdict` refuses one level up, and it has already cost a
            # real candidate: a verified, correctness-passing -4.72% patch that
            # added a re-waved instantiation came back `refuted` because every
            # pre-existing symbol was legitimately unchanged (`learned_rules.md`,
            # "Two ISA-evidence validity traps"). Scoring the new symbol against a
            # guessed parent would be worse than declining; `indeterminate` is the
            # answer this evidence supports, and it refuses nothing downstream.
            verdicts.append({"claim": claim, "realized": None,
                             "evidence": "no shared kernel moved in the claimed direction, and the "
                                         f"candidate adds {', '.join(only_after)}, which has no "
                                         "parent to diff against -- so this claim is unjudged here, "
                                         "not contradicted: "
                                         + "; ".join(f"{n}: {e}" for n, r, e in results
                                                     if r is not True)})
        else:
            verdicts.append({"claim": claim, "realized": False,
                             "evidence": "; ".join(f"{n}: {e}" for n, r, e in results
                                                   if r is not True)})

    observable = [v for v in verdicts if v["claim"] != "none_observable"]
    refuted = [v["claim"] for v in observable if v["realized"] is False]
    indeterminate = [v["claim"] for v in observable if v["realized"] is None]

    if not shared:
        exit_code = EXIT_HOLE
    elif indeterminate:
        exit_code = EXIT_PARTIAL
    else:
        exit_code = EXIT_OK

    return {
        "schema": SCHEMA_DIFF,
        "exit_code": exit_code,
        "from": {"archive": before.get("archive"), "source_hash": before.get("source_hash")},
        "to": {"archive": after.get("archive"), "source_hash": after.get("source_hash")},
        "matched_kernels": shared,
        "only_in_from": only_before,
        "only_in_to": only_after,
        "unchanged_machine_code": unchanged,
        "unchanged_machine_code_basis": unchanged_basis,
        "per_kernel": per_kernel,
        "claims": verdicts,
        "claims_refuted": refuted,
        "claims_indeterminate": indeterminate,
        # Advisory, and separate from `claims_refuted` on purpose: a claim that
        # landed on a non-target symbol is REALIZED, and the round is entitled to
        # say so. What it is not entitled to do is present that as evidence about
        # the route it was measuring.
        "claims_realized_outside_target": [v["claim"] for v in verdicts
                                           if v.get("realized_outside_target")],
        "target_kernels": sorted(hot_kernels),
        "mechanism_realized": mechanism_verdict(observable, unchanged),
    }


def mechanism_verdict(observable: list[dict], unchanged: bool) -> bool | None:
    """The one field the orchestrator reads. Three-valued, and the null matters.

    False means the machine code positively contradicts the candidate's own story.
    None means the archive did not carry the evidence needed to judge it -- which
    must never be collapsed into False, because a gate that reads missing evidence
    as a refutation manufactures exactly the false negative this module exists to
    prevent, only now with a receipt behind it.
    """
    if not observable:
        return None
    if any(v["realized"] is False for v in observable):
        return False
    # An edit that claimed an observable mechanism and produced byte-identical
    # codegen is refuted whatever the per-claim checkers concluded. This is the
    # (48)/(71) shape at the ISA boundary: a mutation that silently did nothing
    # must not be indistinguishable from one that tried.
    if unchanged:
        return False
    if any(v["realized"] is True for v in observable):
        return True
    return None


# ---------------------------------------------------------------------------
# Checks. The rule table, executable. Every rule is `advisory` except the spill
# one, because only that one has a threshold that is not a judgement call:
# `isa_verify.md` states scratch "MUST be 0". A narrow load can be correct for a
# genuine strided gather, an all-drain wait pattern can be structural with no
# slack to relax into, and a rule that called either a defect would report its
# own false positives as findings.
# ---------------------------------------------------------------------------

# One card per rule, each stating the mechanism, the source condition a fix must
# satisfy, and -- the part that keeps an advisory rule honest -- the anti-signals
# that make it a false positive. Each card links onward to the deeper existing
# knowledge file rather than restating it.
_CARDS = "perf_knowledge/isa_signals/rule_cards"
CHECK_CARDS = {
    "spill_to_scratch": f"{_CARDS}/spill_to_scratch.md",
    "narrow_global_load": f"{_CARDS}/narrow_global_load.md",
    "narrow_lds_access": f"{_CARDS}/narrow_lds_access.md",
    "all_memory_waits_drain": f"{_CARDS}/all_memory_waits_drain.md",
    "accvgpr_moves_in_kernel": f"{_CARDS}/accvgpr_moves_in_kernel.md",
    "nop_stall_exposed": f"{_CARDS}/nop_stall_exposed.md",
}


def run_checks(signals: dict, nop_cycle_budget: int = 32) -> dict:
    findings = []

    def add(rule, kernel, severity, observed, expected, note):
        findings.append({"rule": rule, "kernel": kernel, "severity": severity,
                         "observed": observed, "expected": expected,
                         "reference": CHECK_CARDS[rule], "note": note})

    for k in signals.get("kernels", []):
        name = k["name"]
        scratch = _res(k, "scratch_bytes")
        if scratch is not None and scratch > 0:
            add("spill_to_scratch", name, "high", scratch, 0,
                "nonzero .private_segment_fixed_size is a register spill to HBM, recorded as a "
                "3-5x slow path. Reduce live accumulators, tile the K loop, or set "
                "__launch_bounds__ / waves_per_eu; re-dump and confirm it returns to 0.")
        loads = k["global_load_bytes"]
        if loads["accesses"] >= 4 and loads["max"] is not None and loads["max"] < 16:
            add("narrow_global_load", name, "advisory", loads["max"], 16,
                "widest global load is under dwordx4. Expected for a genuine strided gather; a "
                "defect for a contiguous tile, where it usually means the compiler could not "
                "prove alignment or contiguity. Decide from the access pattern, not this count.")
        lds = k["lds_access_bytes"]
        if lds["accesses"] >= 4 and lds["max"] is not None and lds["max"] < 16:
            add("narrow_lds_access", name, "advisory", lds["max"], 16,
                "widest LDS access is under b128. On gfx942 dot-operand reads should reach "
                "ds_read_b128 with kpack=2; if they do not, BLOCK_K may be under 64 or the "
                "swizzle did not apply.")
        waits = k["waits"]
        if waits["drain_ratio"] == 1.0 and (waits["relaxed"] + waits["full_drain"]) >= 4:
            add("all_memory_waits_drain", name, "advisory",
                f"{waits['full_drain']}/{waits['relaxed'] + waits['full_drain']} full drains", "<1.0",
                "every memory wait fully drains its counter, so no load overlaps compute. Whether "
                "that is conservative-waitcnt or structural (no independent ready op to overlap "
                "with) is not decidable from counts; that judgement is yours.")
        if k["accvgpr_moves"] > 0:
            add("accvgpr_moves_in_kernel", name, "advisory", k["accvgpr_moves"], 0,
                "v_accvgpr_read/write means the accumulator is moving between the AGPR and ArchVGPR "
                "files. Inside a hot loop this is pure overhead; in a prologue or epilogue it is "
                "expected. This count is whole-kernel, so use asm_loop_audit.py to scope it to the "
                "loop before acting.")
        if k["nop_stall_cycles"] > nop_cycle_budget:
            add("nop_stall_exposed", name, "advisory", k["nop_stall_cycles"], f"<={nop_cycle_budget}",
                "s_nop requests this many stall cycles, an exposed fixed-latency hazard (typically "
                "MFMA-write to VALU-read). The fix is more unroll or occupancy to fill it, NOT "
                "reordering -- reordering cannot create slack.")

    return {
        "schema": SCHEMA_CHECKS,
        "exit_code": signals.get("exit_code", EXIT_OK),
        "archive": signals.get("archive"),
        "source_hash": signals.get("source_hash"),
        "findings": findings,
        "high": sum(1 for f in findings if f["severity"] == "high"),
        "advisory": sum(1 for f in findings if f["severity"] == "advisory"),
        "kernels_checked": signals.get("kernel_count", 0),
        "unavailable": signals.get("unavailable", []),
    }


# ---------------------------------------------------------------------------
# Observed descriptor. What the machine code says the kernel's mechanism IS, as
# opposed to what its author says it is.
#
# `verify_engineer.md` already established the rule this implements: "Classify
# `compute_primitive` from the disassembly, never from
# the include list ... You are the independent check on a classification the
# engineer made from its own source; reading the same source back is not
# independence." It gives the reason too -- on gfx942 rocWMMA lowers its 32x32x8
# bf16 fragment to `v_mfma_f32_16x16x16_bf16`, so source and machine code are not
# in one-to-one correspondence in either direction.
#
# NOT WIRED TO ANY ADMISSION GATE. This is a capability, not a policy: nothing in
# the lane files a candidate by mechanism, so there is nothing here for it to
# protect. It exists so a reader can cross-check a self-declared descriptor
# against the opcodes, and so the axes below have one implementation rather than
# being re-derived per caller.
#
# Every axis is reported with its evidence, and any axis the opcodes cannot decide
# is `null` -- never a plausible default. A descriptor axis guessed from absence is
# how a candidate ends up filed under a mechanism it does not have.
# ---------------------------------------------------------------------------

DESCRIPTOR_AXES = ("compute_primitive", "k_pipeline", "output_path", "wave_schedule")


def observed_descriptor(kernel: dict) -> dict:
    """Mechanism axes decidable from one kernel's opcode profile, plus evidence.

    Deliberately partial. `compute_primitive` distinguishes matrix-core from VALU
    but CANNOT separate `rocwmma` from `native_mfma`: they emit the same opcodes,
    which is the whole point of the rule this implements. Saying `native_mfma`
    would be a guess dressed as a reading, so the value is `matrix_core` -- a
    coarser answer that is true.
    """
    classes = kernel.get("classes") or {}
    lds_reads = classes.get("lds_read", 0)
    lds_writes = classes.get("lds_write", 0)
    mfma = classes.get("mfma", 0)
    atomics = classes.get("atomic", 0)
    barriers = kernel.get("barriers", 0)
    waits = kernel.get("waits") or {}
    axes: dict = {}
    evidence: dict = {}

    if mfma > 0:
        axes["compute_primitive"] = "matrix_core"
        evidence["compute_primitive"] = (
            f"{mfma} matrix-core instruction(s), shapes {kernel.get('mfma_shapes')}. "
            "Cannot distinguish rocwmma from a native builtin: on gfx942 rocWMMA "
            "lowers to the same v_mfma opcodes, so a finer value would be a guess.")
    elif classes.get("valu", 0) > 0:
        axes["compute_primitive"] = "valu"
        evidence["compute_primitive"] = (
            f"no v_mfma/v_wmma present; {classes.get('valu', 0)} VALU instruction(s)")
    else:
        axes["compute_primitive"] = None
        evidence["compute_primitive"] = "neither matrix-core nor VALU arithmetic seen"

    if lds_reads == 0 and lds_writes == 0:
        axes["k_pipeline"] = "direct_global"
        evidence["k_pipeline"] = "no ds_read/ds_write: operands are not staged through LDS"
    elif barriers >= 2:
        axes["k_pipeline"] = "lds_multi_barrier"
        evidence["k_pipeline"] = (
            f"{lds_reads} LDS read(s) + {lds_writes} write(s) with {barriers} barriers, "
            "which is the shape of a multi-stage or ping-pong LDS pipeline. The exact "
            "stage count is NOT decidable from a whole-kernel opcode profile -- scope "
            "to the hot loop with asm_loop_audit.py before naming one.")
    else:
        axes["k_pipeline"] = "lds_single"
        evidence["k_pipeline"] = (
            f"{lds_reads} LDS read(s) + {lds_writes} write(s) with {barriers} barrier(s)")

    if atomics > 0:
        axes["output_path"] = "atomic_fixup"
        evidence["output_path"] = f"{atomics} atomic instruction(s) on the output path"
    elif lds_writes > 0 and classes.get("global_store", 0) > 0:
        axes["output_path"] = None
        evidence["output_path"] = (
            "both LDS writes and global stores are present, which is consistent with "
            "a staged store AND with LDS used only for operands; not decidable here")
    elif classes.get("global_store", 0) > 0:
        axes["output_path"] = "direct_store"
        evidence["output_path"] = "global stores with no atomics and no LDS writes"
    else:
        axes["output_path"] = None
        evidence["output_path"] = "no store or atomic seen"

    # wave_schedule is NOT inferable from an opcode histogram. Producer/consumer
    # asymmetry and symmetric interleave are properties of which wave executes
    # which block, which a whole-kernel count cannot see. Reported as null with the
    # reason, rather than mapped onto the barrier count -- barriers appear in every
    # one of those schedules.
    axes["wave_schedule"] = None
    evidence["wave_schedule"] = (
        "not decidable from a whole-kernel opcode profile: every wave schedule in "
        f"the vocabulary uses barriers (this kernel has {barriers}), and which wave "
        "runs which block is not visible in a histogram")

    return {
        "name": kernel.get("name"),
        "descriptor": axes,
        "evidence": evidence,
        "undecided": sorted(k for k, v in axes.items() if v is None),
        "drain_ratio": waits.get("drain_ratio"),
    }


def build_descriptors(signals: dict) -> dict:
    kernels = [observed_descriptor(k) for k in signals.get("kernels", [])]
    if not kernels:
        exit_code = EXIT_HOLE
    elif any(k["undecided"] for k in kernels):
        exit_code = EXIT_PARTIAL
    else:
        exit_code = EXIT_OK
    return {
        "schema": "geak.isa-descriptor/v1",
        "exit_code": exit_code,
        "archive": signals.get("archive"),
        "source_hash": signals.get("source_hash"),
        "arch": signals.get("arch"),
        "axes": list(DESCRIPTOR_AXES),
        "kernels": kernels,
        "note": "observed from opcodes, not from source. An axis this cannot decide is "
                "null with a reason; it is never filled with a plausible default.",
        "unavailable": signals.get("unavailable", []),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    signals = sub.add_parser("signals", help="signals for one ISA archive")
    signals.add_argument("--archive", required=True)

    diff = sub.add_parser("diff", help="parent-vs-candidate machine-code diff and claim verdicts")
    diff.add_argument("--from", dest="from_archive", required=True)
    diff.add_argument("--to", dest="to_archive", required=True)
    diff.add_argument("--claim", action="append", default=[],
                      help="repeatable; a claim id from `isa_signals.py claims`")
    diff.add_argument("--hot-kernel", action="append", default=[], dest="hot_kernel",
                      help="repeatable; a kernel symbol this round was targeting. A claim that "
                           "lands only outside this set is still REALIZED and is reported as "
                           "such -- the flag adds `realized_outside_target` so a reader can see "
                           "the mechanism arrived somewhere the plateau was not measured. It "
                           "never causes a refusal.")

    checks = sub.add_parser("checks", help="run the rule table over one archive")
    checks.add_argument("--archive", required=True)
    checks.add_argument("--nop-cycle-budget", type=int, default=32)

    descriptor = sub.add_parser(
        "descriptor",
        help="mechanism axes as the OPCODES report them, with evidence; undecidable "
             "axes come back null rather than defaulted")
    descriptor.add_argument("--archive", required=True)

    sub.add_parser("claims", help="print the closed claim vocabulary")
    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv[1:])
    if args.command == "claims":
        _emit({"schema": "geak.isa-claims/v1",
               "claims": sorted(CLAIMS),
               "note": "closed vocabulary; an unrecognised claim id is reported "
                       "realized=null and is never treated as a pass"})
        return EXIT_OK
    if args.command == "signals":
        payload = build_signals(Path(args.archive))
        _emit(payload)
        return payload["exit_code"]
    if args.command == "checks":
        payload = run_checks(build_signals(Path(args.archive)), args.nop_cycle_budget)
        _emit(payload)
        return payload["exit_code"]
    if args.command == "descriptor":
        payload = build_descriptors(build_signals(Path(args.archive)))
        _emit(payload)
        return payload["exit_code"]
    before = build_signals(Path(args.from_archive))
    after = build_signals(Path(args.to_archive))
    payload = diff_signals(before, after, list(args.claim), list(args.hot_kernel))
    _emit(payload)
    return payload["exit_code"]


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except OSError as exc:
        print(json.dumps({"schema": SCHEMA_SIGNALS, "exit_code": EXIT_ERROR,
                          "error": str(exc)}, sort_keys=True))
        sys.exit(EXIT_ERROR)
