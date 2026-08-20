#!/usr/bin/env python3
"""Navigate an IR archive and attribute structural change to the pass that made it.

`isa_signals.py` answers "what is in the machine code". This answers the
question one level up and one step earlier: **which pass changed what**. That is
the L3 evidence shape -- "this pass added 2 loads and 20 sync ops" -- and it is
the only shape from which L4 can be handed a narrow question, because a
compiler-source escalation with no pass named has nothing to look up.

Five subcommands, in the order an analysis uses them:

  list-stages          the trajectory, optionally ranked by how much each pass changed
  find-changes         the adjacent-pass deltas, ranked -- the attribution command
  stage-summary        one stage's structural census
  diff-stages          two stages, side by side
  performance-signals  the whole trajectory as one JSON receipt

FACTS ONLY. Every number here is a count over text this module parsed, and
nothing in the output says what a count means. "Widest memory access falls from
16 to 4 bytes between stage 22 and 23" is a fact and belongs here; "the load was
not vectorized because the stride is dynamic" is a diagnosis and belongs to the
analyst reading it, who has to be able to check the fact independently. A helper
that mixes the two produces conclusions nobody can audit, and the audit is the
entire reason the evidence ladder is worth its cost.

TWO IR LANGUAGES, COUNTED SEPARATELY. A trajectory crosses instruction selection
partway through: before it the stages are LLVM IR, after it they are MIR. The
same word means different things on either side -- a `load` in IR is an
addressing-mode-free abstract access, a `GLOBAL_LOAD_DWORDX4` in MIR is a
specific 16-byte instruction -- so they are parsed by different readers and the
census records which one it used. Counting them under one vocabulary would make
instruction selection look like a pass that deleted every load.

AMBIGUITY IS AN ERROR. A stage selector that matches two stages fails and lists
them. The reference implementation this follows makes the same choice for the
same reason: a helper that picks one produces a real census of the wrong stage,
and the reader has no way to notice.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

SCHEMA = "geak.ir-signals/v1"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_HOLE = 2

# ---------------------------------------------------------------------------
# LLVM IR
# ---------------------------------------------------------------------------

# Lines that are not instructions. Kept explicit rather than inferred from
# indentation, because a dump's leading whitespace has changed between LLVM
# releases and an indentation rule would silently start counting labels.
_LL_SKIP_PREFIXES = (
    ";", "!", "define", "declare", "attributes", "target", "source_filename",
    "}", "{", "@", "$", "module asm", "uselistorder",
)

# `tail call` and friends put a modifier before the opcode. Everything else in
# LLVM IR puts modifiers after it, so this is the whole exception list.
_LL_LEADING_MODIFIERS = frozenset({"tail", "musttail", "notail"})

_LL_INSTR_RE = re.compile(
    r"^\s*(?:%[^\s=]+\s*=\s*)?(?P<rest>[a-zA-Z][a-zA-Z0-9._]*\b.*)$")
_LL_VECTOR_TYPE_RE = re.compile(r"<\s*(?P<count>\d+)\s*x\s*(?P<elem>[a-zA-Z0-9_]+)\s*>")
_LL_ALIGN_RE = re.compile(r"\balign\s+(?P<align>\d+)\b")
_LL_ADDRSPACE_RE = re.compile(r"addrspace\((?P<space>\d+)\)")
_LL_DBG_RE = re.compile(r"!dbg\s+!(?P<id>\d+)")
_LL_CALLEE_RE = re.compile(r"@(?P<name>[A-Za-z0-9_.$\\]+)")

# Element widths in bits, for turning `<4 x float>` into 16 bytes. Only the types
# that actually appear in AMDGPU kernels; an unknown element yields no width
# rather than a guessed one, because a fabricated byte count would flow straight
# into a "the access narrowed" observation.
_LL_ELEM_BITS = {
    "i1": 1, "i8": 8, "i16": 16, "i32": 32, "i64": 64, "i128": 128,
    "half": 16, "bfloat": 16, "float": 32, "double": 64,
    "ptr": 64,
}

LL_FAMILIES = {
    "memory_load": frozenset({"load"}),
    "memory_store": frozenset({"store"}),
    "atomic": frozenset({"atomicrmw", "cmpxchg", "fence"}),
    "address_arith": frozenset({"getelementptr", "ptrtoint", "inttoptr", "addrspacecast"}),
    "vector_shuffle": frozenset({"shufflevector", "insertelement", "extractelement"}),
    "convert": frozenset({"fptrunc", "fpext", "trunc", "zext", "sext", "bitcast",
                          "sitofp", "uitofp", "fptosi", "fptoui", "fpto", "ptrtoaddr"}),
    "control_flow": frozenset({"br", "switch", "ret", "phi", "select", "unreachable"}),
    "call": frozenset({"call", "invoke"}),
}

# Intrinsics whose presence is structural rather than arithmetic. Matched on the
# callee name so a rename in the arithmetic intrinsics cannot inflate `sync`.
_LL_SYNC_CALLEES = ("llvm.amdgcn.s.barrier", "llvm.amdgcn.fence", "llvm.amdgcn.s.waitcnt",
                    "llvm.amdgcn.sched.barrier", "llvm.amdgcn.sched.group.barrier",
                    "llvm.amdgcn.wave.barrier", "llvm.amdgcn.iglp.opt")
_LL_MATRIX_CALLEES = ("llvm.amdgcn.mfma", "llvm.amdgcn.smfmac", "llvm.amdgcn.wmma")


def _ll_opcode(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    for prefix in _LL_SKIP_PREFIXES:
        if stripped.startswith(prefix):
            return None
    if stripped.endswith(":") and " " not in stripped:
        return None  # a basic-block label
    match = _LL_INSTR_RE.match(line)
    if not match:
        return None
    tokens = match.group("rest").split()
    if not tokens:
        return None
    opcode = tokens[0]
    if opcode in _LL_LEADING_MODIFIERS and len(tokens) > 1:
        opcode = tokens[1]
    if not opcode or not opcode[0].isalpha():
        return None
    return opcode


def _ll_access_bytes(line: str) -> int | None:
    """Width of a load/store, from its vector type or its alignment.

    The type is preferred; `align` is the fallback because a scalar access
    carries no vector type and its alignment is the only width the text states.
    An access whose width neither source gives is reported as unknown, never as
    the element size everyone expects.
    """
    match = _LL_VECTOR_TYPE_RE.search(line)
    if match:
        bits = _LL_ELEM_BITS.get(match.group("elem"))
        if bits:
            return int(match.group("count")) * bits // 8
    scalar = re.search(r"\b(?:load|store)\b\s+(?:volatile\s+|atomic\s+)*"
                       r"(?P<ty>[a-zA-Z0-9_]+)", line)
    if scalar:
        bits = _LL_ELEM_BITS.get(scalar.group("ty"))
        if bits:
            return bits // 8
    align = _LL_ALIGN_RE.search(line)
    if align:
        return int(align.group("align"))
    return None


def parse_ll(text: str) -> dict:
    opcodes: Counter = Counter()
    families: Counter = Counter()
    load_widths: Counter = Counter()
    store_widths: Counter = Counter()
    addrspaces: Counter = Counter()
    callees: Counter = Counter()
    blocks = 0
    functions = 0
    dbg_refs = 0

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("define"):
            functions += 1
        if stripped.endswith(":") and " " not in stripped and stripped[:-1].isdigit():
            blocks += 1
        elif re.match(r"^[A-Za-z0-9._-]+:\s*(;.*)?$", stripped):
            blocks += 1
        opcode = _ll_opcode(line)
        if opcode is None:
            continue
        opcodes[opcode] += 1
        if _LL_DBG_RE.search(line):
            dbg_refs += 1
        for space in _LL_ADDRSPACE_RE.findall(line):
            addrspaces[f"addrspace({space})"] += 1
        for family, members in LL_FAMILIES.items():
            if opcode in members:
                families[family] += 1
                break
        if opcode == "load":
            width = _ll_access_bytes(line)
            load_widths[str(width) if width else "unknown"] += 1
        elif opcode == "store":
            width = _ll_access_bytes(line)
            store_widths[str(width) if width else "unknown"] += 1
        elif opcode in ("call", "invoke"):
            name = _LL_CALLEE_RE.search(line)
            if name:
                callee = name.group("name")
                callees[callee] += 1
                if any(callee.startswith(p) for p in _LL_SYNC_CALLEES):
                    families["sync"] += 1
                elif any(callee.startswith(p) for p in _LL_MATRIX_CALLEES):
                    families["matrix"] += 1

    return {
        "kind": "llvm-ir",
        "opcodes": dict(opcodes),
        "families": dict(families),
        "load_widths_bytes": dict(load_widths),
        "store_widths_bytes": dict(store_widths),
        "address_spaces": dict(addrspaces),
        "callees": dict(callees),
        "instructions": int(sum(opcodes.values())),
        "basic_blocks": blocks,
        "functions": functions,
        "dbg_annotated": dbg_refs,
    }


# ---------------------------------------------------------------------------
# MIR
# ---------------------------------------------------------------------------

_MIR_BB_RE = re.compile(r"^\s*bb\.\d+")
_MIR_FUNC_RE = re.compile(r"^# Machine code for function (?P<name>\S+?):")
_MIR_OPCODE_RE = re.compile(r"^[A-Z][A-Za-z0-9_]*$")

# Words that may sit between the `=` and the opcode, or in front of an
# instruction that has no destination. This list is why the opcode is found by
# walking tokens rather than by one regex: the first version anchored on a
# `%`/`$` destination and therefore stopped seeing instructions the moment
# register allocation started writing `renamable $sgpr0_sgpr1 = S_LOAD_...`.
# Every post-RA pass then looked like it had deleted the loads it had merely
# renamed -- a fabricated structural change, which is the one output this module
# must never produce.
_MIR_LEADING_MODIFIERS = frozenset({
    "renamable", "killed", "undef", "implicit", "implicit-def", "internal", "dead",
    "early-clobber", "frame-setup", "frame-destroy", "debug-instr-number",
    "nofpexcept", "nnan", "ninf", "nsz", "arcp", "contract", "afn", "reassoc",
    "nsw", "nuw", "exact", "noconvergent", "unpredictable", "pre-instr-symbol",
    "post-instr-symbol", "heap-alloc-marker", "nomerge", "samesign", "disjoint",
})

# Opcodes with no underscore, which the "looks like an opcode" test would
# otherwise reject along with header words such as `Function` and `IsSSA`.
_MIR_BARE_OPCODES = frozenset({
    "COPY", "PHI", "REG_SEQUENCE", "INSERT_SUBREG", "EXTRACT_SUBREG", "SUBREG_TO_REG",
    "IMPLICIT_DEF", "KILL", "BUNDLE", "INLINEASM", "INLINEASM_BR", "DBG_VALUE",
    "DBG_INSTR_REF", "DBG_LABEL", "LIFETIME_START", "LIFETIME_END", "FAKE_USE",
})


def _mir_opcode(line: str) -> str | None:
    """The instruction mnemonic on a MIR line, or None if this is not one.

    Everything before the first ` = ` is the destination list and is discarded;
    what remains starts with zero or more flag words followed by the opcode.
    """
    text = line.split("::", 1)[0].split(";", 1)[0].strip()
    if not text:
        return None
    _dest, sep, rest = text.partition(" = ")
    tokens = (rest if sep else text).split()
    for token in tokens:
        if token in _MIR_LEADING_MODIFIERS:
            continue
        if not _MIR_OPCODE_RE.match(token):
            return None
        if "_" in token or token in _MIR_BARE_OPCODES:
            return token
        return None
    return None

# First match wins, so the specific families must precede the `V_` / `S_` catch-alls.
# Without that ordering `S_BARRIER` lands in scalar ALU and the sync count -- the
# single number the paper's own L3 example turns on -- reads zero.
MIR_FAMILY_PREFIXES = (
    ("sync", ("S_BARRIER", "S_WAITCNT", "ATOMIC_FENCE", "BUFFER_WBINVL", "BUFFER_INV",
              "BUFFER_WBL2", "S_WAKEUP", "SCHED_BARRIER", "SCHED_GROUP_BARRIER",
              "IGLP_OPT")),
    ("matrix", ("V_MFMA", "V_SMFMAC", "V_WMMA")),
    ("accvgpr", ("V_ACCVGPR",)),
    ("scratch", ("SCRATCH_", "SI_SPILL", "BUFFER_LOAD_DWORD_OFFEN_SPILL")),
    ("lds", ("DS_",)),
    ("global_memory", ("GLOBAL_LOAD", "GLOBAL_STORE", "GLOBAL_ATOMIC")),
    ("buffer_memory", ("BUFFER_LOAD", "BUFFER_STORE", "BUFFER_ATOMIC")),
    ("flat_memory", ("FLAT_LOAD", "FLAT_STORE", "FLAT_ATOMIC")),
    ("scalar_memory", ("S_LOAD", "S_BUFFER_LOAD", "S_STORE")),
    ("exec_mask", ("S_AND_SAVEEXEC", "S_OR_SAVEEXEC", "S_XOR_SAVEEXEC", "SI_IF",
                   "SI_ELSE", "SI_END_CF", "S_CBRANCH_EXECZ", "S_CBRANCH_EXECNZ")),
    ("control_flow", ("S_BRANCH", "S_CBRANCH", "S_ENDPGM", "S_SETPC", "S_SWAPPC", "PHI")),
    ("copy", ("COPY", "V_MOV_B32", "S_MOV_B")),
    # Register plumbing that carries no hardware cost of its own. Named rather
    # than left in `other`, because an unnamed bucket that moves by a dozen ops
    # at instruction selection reads like an unexplained structural change.
    ("pseudo", ("REG_SEQUENCE", "IMPLICIT_DEF", "INSERT_SUBREG", "EXTRACT_SUBREG",
                "KILL", "BUNDLE", "SUBREG_TO_REG", "LOCAL_ESCAPE")),
    ("vector_alu", ("V_",)),
    ("scalar_alu", ("S_",)),
)

# Width suffixes, longest first so DWORDX4 is not read as DWORD.
_MIR_WIDTH_SUFFIXES = (
    ("DWORDX4", 16), ("DWORDX3", 12), ("DWORDX2", 8), ("DWORDX1", 4), ("DWORD", 4),
    ("B128", 16), ("B96", 12), ("B64", 8), ("B32", 4), ("B16", 2), ("B8", 1),
    ("X4", 16), ("X2", 8),
    ("USHORT", 2), ("SSHORT", 2), ("SHORT", 2), ("UBYTE", 1), ("SBYTE", 1), ("BYTE", 1),
    ("D16_HI", 2), ("D16", 2),
)


def mir_family(opcode: str) -> str:
    for family, prefixes in MIR_FAMILY_PREFIXES:
        if opcode.startswith(prefixes):
            return family
    return "other"


def mir_access_bytes(opcode: str) -> int | None:
    for suffix, width in _MIR_WIDTH_SUFFIXES:
        if suffix in opcode:
            return width
    return None


def parse_mir(text: str) -> dict:
    opcodes: Counter = Counter()
    families: Counter = Counter()
    load_widths: Counter = Counter()
    store_widths: Counter = Counter()
    blocks = 0
    functions = 0
    virtual_regs: set[str] = set()

    for line in text.splitlines():
        if _MIR_FUNC_RE.match(line):
            functions += 1
            continue
        if _MIR_BB_RE.match(line):
            blocks += 1
            continue
        opcode = _mir_opcode(line)
        if opcode is None:
            continue
        opcodes[opcode] += 1
        family = mir_family(opcode)
        families[family] += 1
        virtual_regs.update(re.findall(r"%\d+", line))
        width = mir_access_bytes(opcode)
        if width is not None:
            if "LOAD" in opcode or "READ" in opcode:
                load_widths[str(width)] += 1
            elif "STORE" in opcode or "WRITE" in opcode:
                store_widths[str(width)] += 1

    return {
        "kind": "mir",
        "opcodes": dict(opcodes),
        "families": dict(families),
        "load_widths_bytes": dict(load_widths),
        "store_widths_bytes": dict(store_widths),
        "address_spaces": {},
        "callees": {},
        "instructions": int(sum(opcodes.values())),
        "basic_blocks": blocks,
        "functions": functions,
        "virtual_registers": len(virtual_regs),
    }


def parse_stage_text(text: str, file_name: str = "") -> dict:
    """Pick the reader from the dump's own opening line, not the extension.

    The extension was assigned by `ir_capture.py` from the same marker, so this
    is a second opinion rather than a duplicate: an archive hand-assembled for a
    test, or one whose files were renamed, still gets counted correctly.
    """
    head = text.lstrip("\n")[:400]
    if "# Machine code for function " in head or file_name.endswith(".mir"):
        return parse_mir(text)
    return parse_ll(text)


# ---------------------------------------------------------------------------
# Archive access
# ---------------------------------------------------------------------------

def read_manifest(archive: Path) -> dict:
    path = Path(archive) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist -- this is not an IR archive. `ir_capture.py` writes the "
            "manifest last, so its absence usually means the capture died partway.")
    return json.loads(path.read_text(encoding="utf-8"))


def stage_entries(manifest: dict) -> list[dict]:
    return list(manifest.get("stages", []))


def stage_text(archive: Path, entry: dict) -> str:
    return (Path(archive) / entry["file"]).read_text(encoding="utf-8", errors="replace")


def resolve_stage(entries: list[dict], selector: str) -> dict:
    """One stage, or an error that lists every candidate.

    Accepts an index, an exact pass id, an exact file name, or a substring --
    tried in that order, so the precise forms can never be shadowed by a loose
    one. A selector matching more than one stage raises; see the module
    docstring for why picking is worse than failing.
    """
    if not entries:
        raise LookupError("this archive has no stages")
    text = str(selector).strip()

    if text.isdigit():
        index = int(text)
        hits = [e for e in entries if int(e.get("index", -1)) == index]
        if hits:
            return hits[0]
        raise LookupError(f"no stage with index {index}; this archive has "
                          f"{len(entries)} stages (0..{len(entries) - 1})")

    for key in ("file", "pass_id"):
        hits = [e for e in entries if str(e.get(key, "")) == text
                or Path(str(e.get(key, ""))).name == text]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise LookupError(
                f"{text!r} matches {len(hits)} stages by {key} "
                f"(indices {[e['index'] for e in hits]}); name one by index. A pass that ran "
                "more than once is the normal case, and choosing for you would produce a real "
                "census of a stage you did not ask for.")

    hits = [e for e in entries
            if text.lower() in str(e.get("pass_id", "")).lower()
            or text.lower() in str(e.get("pass_name", "")).lower()]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise LookupError(
            f"{text!r} matches {len(hits)} stages (indices {[e['index'] for e in hits]}: "
            f"{[e.get('pass_id') for e in hits][:8]}); name one by index")
    raise LookupError(f"{text!r} matches no stage in this archive")


# ---------------------------------------------------------------------------
# Census and deltas
# ---------------------------------------------------------------------------

_DELTA_KEYS = ("opcodes", "families", "load_widths_bytes", "store_widths_bytes",
               "address_spaces", "callees")


def census(archive: Path, entry: dict) -> dict:
    parsed = parse_stage_text(stage_text(archive, entry), entry.get("file", ""))
    parsed["index"] = entry.get("index")
    parsed["pass_id"] = entry.get("pass_id")
    parsed["pass_name"] = entry.get("pass_name")
    parsed["scope"] = entry.get("scope")
    parsed["file"] = entry.get("file")
    return parsed


def delta(before: dict, after: dict) -> dict:
    """What changed between two censuses, split into added and removed.

    Split rather than signed because that is the shape the evidence is quoted in
    -- "added 2 loads, 20 sync ops" -- and because a pass that both adds and
    removes in one family nets to a number that hides both halves.
    """
    added: dict = {}
    removed: dict = {}
    for key in _DELTA_KEYS:
        left, right = before.get(key, {}) or {}, after.get(key, {}) or {}
        for name in sorted(set(left) | set(right)):
            change = int(right.get(name, 0)) - int(left.get(name, 0))
            if change > 0:
                added.setdefault(key, {})[name] = change
            elif change < 0:
                removed.setdefault(key, {})[name] = -change

    scalars = {}
    for key in ("instructions", "basic_blocks", "functions", "virtual_registers"):
        left, right = before.get(key), after.get(key)
        if isinstance(left, int) and isinstance(right, int) and left != right:
            scalars[key] = right - left

    magnitude = sum(sum(block.values()) for block in added.values())
    magnitude += sum(sum(block.values()) for block in removed.values())
    return {
        "added": added, "removed": removed, "scalar_delta": scalars,
        "magnitude": magnitude,
        # The two IR languages are counted by different readers, so a delta that
        # straddles instruction selection is not a pass's doing. Flagged rather
        # than suppressed: the boundary itself is a real, locatable event.
        "crosses_ir_boundary": before.get("kind") != after.get("kind"),
    }


def rank_key(move: dict) -> tuple:
    """Ranking for `find-changes`: real pass effects before the language boundary.

    Instruction selection rewrites every instruction in the function, so by raw
    magnitude it always wins -- and it is never the answer to "which pass changed
    this", because the delta is an artefact of two readers counting two
    vocabularies. Sorting it behind the genuine changes keeps the top of the list
    usable while leaving the boundary itself visible, which matters: where it
    sits is exactly where an IR-level explanation stops and a codegen-level one
    starts.
    """
    return (1 if move.get("crosses_ir_boundary") else 0, -move["magnitude"], move["to_index"])


def widest(width_map: dict) -> int | None:
    values = [int(k) for k in width_map or {} if str(k).isdigit()]
    return max(values) if values else None


def trajectory(archive: Path, manifest: dict) -> list[dict]:
    return [census(archive, entry) for entry in stage_entries(manifest)]


def transitions(censuses: list[dict]) -> list[dict]:
    out: list[dict] = []
    for i in range(1, len(censuses)):
        before, after = censuses[i - 1], censuses[i]
        change = delta(before, after)
        change.update({
            "from_index": before.get("index"), "to_index": after.get("index"),
            "pass_id": after.get("pass_id"), "pass_name": after.get("pass_name"),
            "scope": after.get("scope"),
        })
        out.append(change)
    return out


MAX_OBSERVATIONS = 40


def observations(censuses: list[dict], moves: list[dict]) -> list[str]:
    """Locatable facts about the trajectory. No interpretation, by contract.

    Repeats are folded. `widest load` is a maximum over the whole function, so a
    pass that clones a block moves it transiently and the same transition
    reappears three or four times in one trajectory. Listing each occurrence
    makes a receipt whose length is dominated by its least informative line;
    folding them keeps every distinct fact and the first place it happened.
    """
    notes: list[str] = []
    if not censuses:
        return notes

    for i in range(1, len(censuses)):
        before, after = censuses[i - 1], censuses[i]
        if before.get("kind") != after.get("kind"):
            notes.append(
                f"IR language changes from {before.get('kind')} to {after.get('kind')} at stage "
                f"{after.get('index')} ({after.get('pass_id')}); counts before and after this "
                "point come from different readers and are not comparable")

    for key, label in (("load_widths_bytes", "load"), ("store_widths_bytes", "store")):
        folded: dict[tuple, list] = {}
        for i in range(1, len(censuses)):
            before, after = censuses[i - 1], censuses[i]
            if before.get("kind") != after.get("kind"):
                continue
            was, now = widest(before.get(key, {})), widest(after.get(key, {}))
            if was is not None and now is not None and now < was:
                folded.setdefault((was, now), []).append(after)
        for (was, now), hits in folded.items():
            first = hits[0]
            where = (f"stage {first.get('index')} ({first.get('pass_id')})" if len(hits) == 1
                     else f"{len(hits)} stages, first at {first.get('index')} "
                          f"({first.get('pass_id')})")
            notes.append(f"widest {label} falls {was} -> {now} bytes at {where}")

    for family in ("sync", "scratch", "matrix", "lds", "accvgpr"):
        for move in moves:
            gained = move.get("added", {}).get("families", {}).get(family, 0)
            lost = move.get("removed", {}).get("families", {}).get(family, 0)
            if gained:
                notes.append(f"{family} ops +{gained} at stage {move.get('to_index')} "
                             f"({move.get('pass_id')})")
            if lost:
                notes.append(f"{family} ops -{lost} at stage {move.get('to_index')} "
                             f"({move.get('pass_id')})")

    first_scratch = next((c for c in censuses if c.get("families", {}).get("scratch")), None)
    if first_scratch is not None:
        notes.append(f"scratch first appears at stage {first_scratch.get('index')} "
                     f"({first_scratch.get('pass_id')})")

    if len(notes) > MAX_OBSERVATIONS:
        elided = len(notes) - MAX_OBSERVATIONS
        notes = notes[:MAX_OBSERVATIONS]
        notes.append(f"({elided} further observation(s) elided; `find-changes` is the "
                     "complete, ranked list)")
    return notes


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def _emit(payload: dict) -> int:
    print(json.dumps(payload, sort_keys=True, indent=2))
    return int(payload.get("exit_code", EXIT_OK))


def _base(archive: Path, manifest: dict) -> dict:
    return {
        "schema": SCHEMA,
        "archive": str(archive),
        "source_hash": manifest.get("source_hash"),
        "backend": manifest.get("backend"),
        "kernel_filter": manifest.get("kernel_filter"),
        "compiled_source": manifest.get("compiled_source"),
        "edited_source": manifest.get("edited_source"),
        "provenance": manifest.get("provenance"),
        "capture_exit_code": manifest.get("exit_code"),
        "capture_holes": manifest.get("holes", []),
    }


def cmd_list_stages(args) -> int:
    archive = Path(args.archive)
    manifest = read_manifest(archive)
    entries = stage_entries(manifest)
    payload = _base(archive, manifest)
    payload["stage_count"] = len(entries)
    if not entries:
        payload["exit_code"] = EXIT_HOLE
        payload["stages"] = []
        payload["note"] = ("this archive captured no stages, which is a HOLE and not a kernel "
                           "whose passes changed nothing")
        return _emit(payload)

    rows = [{"index": e.get("index"), "pass_id": e.get("pass_id"),
             "pass_name": e.get("pass_name"), "scope": e.get("scope"),
             "file": e.get("file"), "lines": e.get("lines")} for e in entries]
    if args.sort_by == "interesting":
        moves = transitions(trajectory(archive, manifest))
        by_index = {m["to_index"]: m for m in moves}
        for row in rows:
            move = by_index.get(row["index"])
            row["change_magnitude"] = move["magnitude"] if move else 0
            row["crosses_ir_boundary"] = bool(move and move["crosses_ir_boundary"])
        rows.sort(key=lambda r: (r.get("crosses_ir_boundary", False),
                                 -r.get("change_magnitude", 0), r["index"]))
    payload["stages"] = rows
    payload["exit_code"] = EXIT_OK
    return _emit(payload)


def cmd_find_changes(args) -> int:
    archive = Path(args.archive)
    manifest = read_manifest(archive)
    censuses = trajectory(archive, manifest)
    payload = _base(archive, manifest)
    if len(censuses) < 2:
        payload["exit_code"] = EXIT_HOLE
        payload["changes"] = []
        payload["note"] = ("fewer than two stages: there is no adjacent pair to attribute a "
                           "change to")
        return _emit(payload)
    moves = transitions(censuses)
    ranked = sorted(moves, key=rank_key)
    if args.top:
        ranked = ranked[: args.top]
    payload["changes"] = ranked
    payload["stage_count"] = len(censuses)
    payload["exit_code"] = EXIT_OK
    return _emit(payload)


def cmd_stage_summary(args) -> int:
    archive = Path(args.archive)
    manifest = read_manifest(archive)
    entry = resolve_stage(stage_entries(manifest), args.stage)
    payload = _base(archive, manifest)
    payload["stage"] = census(archive, entry)
    payload["exit_code"] = EXIT_OK
    return _emit(payload)


def cmd_diff_stages(args) -> int:
    archive = Path(args.archive)
    manifest = read_manifest(archive)
    entries = stage_entries(manifest)
    before = census(archive, resolve_stage(entries, getattr(args, "from")))
    after = census(archive, resolve_stage(entries, args.to))
    payload = _base(archive, manifest)
    payload["from"] = {k: before.get(k) for k in ("index", "pass_id", "pass_name", "kind", "file")}
    payload["to"] = {k: after.get(k) for k in ("index", "pass_id", "pass_name", "kind", "file")}
    payload["delta"] = delta(before, after)
    payload["exit_code"] = EXIT_OK
    return _emit(payload)


def cmd_performance_signals(args) -> int:
    archive = Path(args.archive)
    manifest = read_manifest(archive)
    censuses = trajectory(archive, manifest)
    moves = transitions(censuses)
    payload = _base(archive, manifest)
    payload["stage_count"] = len(censuses)
    payload["trajectory"] = [
        {"index": c.get("index"), "pass_id": c.get("pass_id"), "kind": c.get("kind"),
         "instructions": c.get("instructions"), "families": c.get("families"),
         "widest_load_bytes": widest(c.get("load_widths_bytes", {})),
         "widest_store_bytes": widest(c.get("store_widths_bytes", {}))}
        for c in censuses
    ]
    payload["top_changes"] = sorted(moves, key=rank_key)[:10]
    payload["observations"] = observations(censuses, moves)
    payload["exit_code"] = EXIT_OK if censuses else EXIT_HOLE
    return _emit(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("list-stages", help="the trajectory, in order or by how much changed")
    listing.add_argument("--archive", required=True)
    listing.add_argument("--sort-by", default="index", choices=("index", "interesting"))
    listing.set_defaults(func=cmd_list_stages)

    changes = sub.add_parser("find-changes",
                             help="adjacent-pass deltas, ranked -- the attribution command")
    changes.add_argument("--archive", required=True)
    changes.add_argument("--top", type=int, default=15)
    changes.set_defaults(func=cmd_find_changes)

    summary = sub.add_parser("stage-summary", help="one stage's structural census")
    summary.add_argument("--archive", required=True)
    summary.add_argument("--stage", required=True,
                         help="index, exact pass id, file name, or unambiguous substring")
    summary.set_defaults(func=cmd_stage_summary)

    diff = sub.add_parser("diff-stages", help="two stages, side by side")
    diff.add_argument("--archive", required=True)
    diff.add_argument("--from", required=True, dest="from")
    diff.add_argument("--to", required=True)
    diff.set_defaults(func=cmd_diff_stages)

    signals = sub.add_parser("performance-signals", help="the whole trajectory as one receipt")
    signals.add_argument("--archive", required=True)
    signals.add_argument("--format", default="json", choices=("json",))
    signals.set_defaults(func=cmd_performance_signals)
    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv[1:])
    return args.func(args)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (LookupError, FileNotFoundError, OSError, ValueError) as exc:
        print(json.dumps({"schema": SCHEMA, "exit_code": EXIT_ERROR, "error": str(exc)},
                         sort_keys=True, indent=2))
        sys.exit(EXIT_ERROR)
