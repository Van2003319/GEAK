#!/usr/bin/env python3
"""Flag vector casts whose destination stride cannot carry the cast's alignment.

Finding 143. Widening a `global -> LDS` staging load is an obvious and popular
mutation: the B tile already moves 16 bytes per instruction, the A tile moves 4,
so an agent widens A and every counting check comes back green -- fewer
instructions, fewer registers, same LDS, no spills. The code is still wrong,
because none of those checks is about an *address*.

`reinterpret_cast<uint4*>(&as[row][kk])` is a promise that the address is
16-byte aligned, and the compiler is entitled to believe it. On the kernel this
was found in, `as` is `Bf16[32][68]`: a row is 136 bytes, 136 % 16 == 8, so
exactly half the destinations are 8-byte aligned. The compiler believed the
promise and emitted `ds_write_b128`. That does not fault on gfx942 -- the
address is truncated within the 16-byte granule -- so the failure mode is a
silently wrong tile, on every shape, discovered at correctness time if at all.

WHAT THIS CHECKS, precisely: for a cast to a W-byte vector type applied to
`&arr[i0][i1]...[in]`, every dimension stride *except the innermost* must be a
multiple of W. Those strides are fixed by the declaration, so they are decidable
from the source alone.

WHAT IT DOES NOT CHECK: the innermost term. `&as[row][kk]` contributes
`kk * sizeof(elem)`, and whether that is a multiple of W depends on the loop that
produces `kk` -- in the kernel above `kk` is always a multiple of 8, which makes
the innermost term 16-byte aligned, and a checker that flagged it would report
correct code as broken. So this is a necessary condition, not a sufficient one,
and it says so rather than implying coverage it does not have.

The same asymmetry is why the fix is not "give up on the wide load": the *source*
side of that staging copy is 16-byte aligned for every lane. One `uint4` global
load and two `uint2` LDS stores keeps the whole mechanism and tells the truth
about both sides.

Unknown element types and unresolved dimension constants are REPORTED, never
skipped. A checker that quietly passes what it could not parse is a checker whose
clean exit means nothing (141).

    lds_cast_alignment.py path/to/kernel.hip [more.hip ...]
    lds_cast_alignment.py --json path/to/kernel.hip
    lds_cast_alignment.py --baseline orig.hip candidate.hip     # gate mode

`--baseline` is what makes this usable as a gate. The shipped kernel already
contains one of these -- the B tile has the identical 8-byte-aligned `uint4`
cast, benign today only because the current compiler declines to believe it --
so a fail-closed absolute check would reject every candidate including an
unmodified one, and an exit code that is always 1 says exactly as much as one
that is always 0. In gate mode the tool reports pre-existing findings as
inherited and fails only on a signature the baseline did not have. Signatures
are (array, declaration, cast type, dimension), never line numbers, because a
candidate that only moves code is not a candidate that introduced a hazard.

Exit codes: 0 no findings (or, in gate mode, none the baseline lacked);
1 at least one new finding; 2 something was unparseable and the verdict would
have been a guess.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Width in bytes of the cast target types worth checking. Scalar casts are in
# the table too: `unsigned` on a 2-byte element type is the pattern this kernel
# family already uses, and it has the same decidable property.
CAST_WIDTH = {
    "uint4": 16, "float4": 16, "int4": 16, "double2": 16, "ulonglong2": 16,
    "uint2": 8, "float2": 8, "int2": 8, "double": 8, "ulonglong": 8,
    "unsigned": 4, "unsigned int": 4, "int": 4, "float": 4, "uint32_t": 4,
}

# Element types these kernels actually declare LDS with. Deliberately a closed
# table: a type not in it is reported as unknown rather than assumed, because
# the whole verdict is arithmetic on sizeof.
ELEM_SIZE = {
    "Bf16": 2, "bfloat16_t": 2, "rocwmma::bfloat16_t": 2, "__hip_bfloat16": 2,
    "half": 2, "__half": 2, "_Float16": 2, "rocwmma::float16_t": 2,
    "float": 4, "int": 4, "unsigned": 4, "unsigned int": 4, "uint32_t": 4,
    "double": 8, "long": 8,
    # Byte-sized element types, i.e. a raw LDS arena: `__shared__ __align__(16)
    # char smem[kSmemBytes];` with typed views placed into it by hand. Omitting
    # them did not make the scan conservative, it made it VACUOUS AND LOUD: an
    # unknown element type is never registered in `arrays`, so every cast
    # against that arena hits the `name not in arrays` continue and no finding
    # can be produced -- yet `passed` goes false on the unparseable entry alone.
    # Observed twice on this task (round 1, engineer_1): findings [] and four
    # unparseable rows for one `char smem[]`, refusing a candidate before its
    # build, correctness run, or a single timing. A 1-D arena is doubly
    # undecidable here, since the stride loop walks `strides[:-1]` and there is
    # no outer stride to walk. `char` is 1 byte by definition, so this resolves
    # the size rather than guessing it; the arena still gets the same
    # innermost-index caveat every other array gets.
    "char": 1, "int8_t": 1, "uint8_t": 1, "std::byte": 1, "std::int8_t": 1,
}

# sizeof for the scalar bases a vector typedef can be built on. Same closed-table
# discipline as ELEM_SIZE: a base that is not here leaves the typedef unresolved.
SIZEOF = {
    "char": 1, "signed char": 1, "unsigned char": 1, "int8_t": 1, "uint8_t": 1,
    "short": 2, "unsigned short": 2, "int16_t": 2, "uint16_t": 2,
    "_Float16": 2, "half": 2, "__half": 2, "Bf16": 2, "__hip_bfloat16": 2,
    "int": 4, "unsigned": 4, "unsigned int": 4, "int32_t": 4, "uint32_t": 4,
    "float": 4,
    "long": 8, "unsigned long": 8, "int64_t": 8, "uint64_t": 8, "double": 8,
}

# A vector typedef declared IN THE FILE BEING SCANNED, in either of the two
# spellings these kernels use:
#   typedef __attribute__((__vector_size__(4 * sizeof(short)))) short shortx4_t;
#   typedef short shortx4_t __attribute__((ext_vector_type(4)));
VEC_SIZE_TYPEDEF = re.compile(
    r"typedef\s+__attribute__\(\(\s*_*vector_size_*\s*\(\s*(?P<expr>[^)]*(?:\([^)]*\))?[^)]*)\)\s*\)\)"
    r"\s+(?P<base>[\w:]+(?:\s+\w+)?)\s+(?P<name>\w+)\s*;")
EXT_VEC_TYPEDEF = re.compile(
    r"typedef\s+(?P<base>[\w:]+(?:\s+\w+)?)\s+(?P<name>\w+)\s*"
    r"__attribute__\(\(\s*ext_vector_type\s*\(\s*(?P<n>\d+)\s*\)\s*\)\)\s*;")
_SIZEOF_CALL = re.compile(r"sizeof\s*\(\s*([\w:]+(?:\s+\w+)?)\s*\)")


def _vector_size_bytes(expr: str) -> int | None:
    """Bytes for a `__vector_size__` argument, or None if not decidable here.

    Accepts an integer literal, `sizeof(T)`, and products of those. Anything
    else -- a named constant, a division, arithmetic we have not seen -- is
    left unresolved so the cast is reported unknown rather than assumed.
    """
    resolved = _SIZEOF_CALL.sub(
        lambda m: str(SIZEOF[m.group(1).strip()]) if m.group(1).strip() in SIZEOF else "?",
        expr)
    if "?" in resolved or not re.fullmatch(r"[\d\s*]+", resolved or ""):
        return None
    total = 1
    for factor in resolved.split("*"):
        factor = factor.strip()
        if not factor:
            return None
        total *= int(factor)
    return total or None


def vector_typedefs(text: str) -> dict[str, int]:
    """Cast-target widths for vector typedefs the scanned file declares itself.

    Why this is not a loosening. The tool's rule is resolve-or-report, and it
    already resolves `constexpr` array dimensions out of the same source rather
    than keeping a table of them. A vector typedef is the same kind of fact,
    written a few lines above the cast that uses it, and leaving it out did not
    make the scan conservative -- it made it refuse a candidate on which it had
    produced no finding, three times in one round on this task. The width is
    READ here, never defaulted: an expression this cannot evaluate yields no
    entry, and the cast stays unparseable exactly as before.
    """
    out: dict[str, int] = {}
    for m in VEC_SIZE_TYPEDEF.finditer(text):
        width = _vector_size_bytes(m.group("expr"))
        if width:
            out[m.group("name")] = width
    for m in EXT_VEC_TYPEDEF.finditer(text):
        base = SIZEOF.get(m.group("base").strip())
        if base:
            out[m.group("name")] = base * int(m.group("n"))
    return out


DECL = re.compile(
    r"__shared__\s+(?:__align__\(\d+\)\s+)?"
    r"(?P<type>[A-Za-z_][\w:]*(?:\s+int)?)\s+"
    r"(?P<name>\w+)\s*(?P<dims>(?:\[[^\]]+\]\s*)+);")
CONST = re.compile(r"constexpr\s+(?:int|unsigned|size_t)\s+(\w+)\s*=\s*([^;]+);")
CAST = re.compile(
    r"reinterpret_cast\s*<\s*(?:const\s+)?(?P<type>[\w:]+(?:\s+int)?)\s*\*\s*>"
    r"\s*\(\s*&\s*(?P<name>\w+)\s*(?P<idx>(?:\[[^\]]*\]\s*)+)\)")


class Unresolved(ValueError):
    """A dimension whose extent is not a compile-time constant we can read."""


def constants(text: str) -> dict[str, int]:
    """`constexpr int` values, resolved against each other in declaration order.

    Only integer literals and expressions over already-known names. Anything
    else is left out, which makes the dimension it feeds Unresolved and thus
    reported -- the deliberate alternative to guessing.
    """
    known: dict[str, int] = {}
    for name, expr in CONST.findall(text):
        try:
            known[name] = int(eval(expr, {"__builtins__": {}}, dict(known)))
        except Exception:
            continue
    return known


def dims(spec: str, known: dict[str, int]) -> list[int | None]:
    """Extents, outermost first. `None` marks the outermost one when it is not
    a compile-time constant we can read.

    That single hole is safe to leave open, and leaving it open is the whole
    point: `strides` computes `out[i] = out[i+1] * extents[i+1]`, so it reads
    `extents[1:]` and never `extents[0]`. The outermost extent therefore
    contributes to no stride and cannot change any verdict -- refusing to
    decide the array because of it would fail closed on a question that was
    never being asked. Template-parameterised leading dimensions
    (`__shared__ Bf16 as[TM][kLdsStride]` with `TM = NM * kTile`) are the
    common case, and they say nothing about alignment.

    Any *inner* unresolved extent still raises: those do land in a stride, and
    guessing one would be guessing the verdict.
    """
    raw = re.findall(r"\[([^\]]+)\]", spec)
    out: list[int | None] = []
    for pos, item in enumerate(raw):
        try:
            out.append(int(eval(item.strip(), {"__builtins__": {}}, dict(known))))
        except Exception as exc:
            if pos == 0:
                out.append(None)
                continue
            raise Unresolved(f"dimension {item.strip()!r}") from exc
    return out


def strides(extents: list[int | None], elem: int) -> list[int]:
    """Byte stride of each dimension, outermost first. Last entry is `elem`.

    Reads `extents[1:]` only, which is why `dims` may leave `extents[0]` None.
    """
    out = [elem] * len(extents)
    for i in range(len(extents) - 2, -1, -1):
        out[i] = out[i + 1] * extents[i + 1]
    return out


def scan(path: Path) -> tuple[list[dict], list[dict]]:
    """(findings, unparseable). Both are reported; neither is inferred."""
    text = path.read_text(encoding="utf-8", errors="replace")
    known = constants(text)
    # Module table first, then widths this file declares for itself. A file-local
    # typedef wins on a name clash: it is the definition actually in scope here.
    widths = dict(CAST_WIDTH)
    widths.update(vector_typedefs(text))
    findings: list[dict] = []
    unknown: list[dict] = []

    arrays: dict[str, dict] = {}
    for m in DECL.finditer(text):
        name, typename = m.group("name"), m.group("type").strip()
        line = text[:m.start()].count("\n") + 1
        elem = ELEM_SIZE.get(typename)
        if elem is None:
            unknown.append({"path": str(path), "line": line, "array": name,
                            "reason": f"element type {typename!r} is not in ELEM_SIZE; "
                                      "its size decides the whole verdict"})
            continue
        try:
            extents = dims(m.group("dims"), known)
        except Unresolved as exc:
            unknown.append({"path": str(path), "line": line, "array": name,
                            "reason": f"{exc} is not a resolvable constant"})
            continue
        src = [d.strip() for d in re.findall(r"\[([^\]]+)\]", m.group("dims"))]
        shown = [str(e) if e is not None else s for e, s in zip(extents, src)]
        arrays[name] = {"elem": elem, "extents": extents, "extents_shown": shown,
                        "strides": strides(extents, elem), "type": typename}

    for m in CAST.finditer(text):
        name = m.group("name")
        if name not in arrays:
            continue  # not an array this file declares; nothing decidable here
        width = widths.get(m.group("type").strip())
        line = text[:m.start()].count("\n") + 1
        if width is None:
            unknown.append({"path": str(path), "line": line, "array": name,
                            "reason": f"cast type {m.group('type')!r} is not in CAST_WIDTH"})
            continue
        info = arrays[name]
        # Every stride but the innermost. The innermost term's alignment depends
        # on the index expression, which is not decidable here; see the module
        # docstring for why flagging it would be worse than not checking it.
        for depth, stride in enumerate(info["strides"][:-1]):
            if stride % width:
                findings.append({
                    "path": str(path), "line": line, "array": name,
                    "declared": f"{info['type']}{''.join(f'[{e}]' for e in info['extents_shown'])}",
                    "cast_to": m.group("type").strip(), "cast_width": width,
                    "dimension": depth, "stride_bytes": stride,
                    "misaligned_by": stride % width,
                    "detail": f"dimension {depth} advances {stride} bytes, which is not a "
                              f"multiple of {width}; every odd step lands {stride % width} "
                              f"bytes off and the cast promises alignment the address "
                              f"does not have",
                })
                break
    return findings, unknown


def signature(finding: dict) -> tuple:
    """What makes two findings the same hazard.

    Not the line number and not the path: a candidate that moves a kernel into
    a new file, or inserts twenty lines above it, has not introduced anything.
    Gate mode would otherwise fail on every reformat, which trains the reader to
    pass `--no-verify` and is worse than having no gate.
    """
    return (finding["array"], finding["declared"], finding["cast_to"],
            finding["dimension"])


def scan_all(paths: list[str]) -> tuple[list[dict], list[dict]]:
    findings: list[dict] = []
    unknown: list[dict] = []
    for raw in paths:
        f, u = scan(Path(raw))
        findings += f
        unknown += u
    return findings, unknown


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--baseline", action="append", default=[],
                    help="gate mode: fail only on hazards these files do not "
                         "already have (repeatable)")
    args = ap.parse_args(argv)

    findings, unknown = scan_all(args.paths)
    inherited: list[dict] = []
    if args.baseline:
        base, base_unknown = scan_all(args.baseline)
        if base_unknown:
            # The baseline being unreadable makes "new" undecidable. Reporting
            # the candidate clean here would be a pass derived from a file we
            # could not read.
            unknown += [{**u, "reason": f"baseline: {u['reason']}"} for u in base_unknown]
        known = {signature(f) for f in base}
        inherited = [f for f in findings if signature(f) in known]
        findings = [f for f in findings if signature(f) not in known]

    if args.json:
        print(json.dumps({"findings": findings, "inherited": inherited,
                          "unparseable": unknown,
                          "gate_mode": bool(args.baseline),
                          "passed": not findings and not unknown}, sort_keys=True))
    else:
        for f in findings:
            print(f"{f['path']}:{f['line']}: {f['array']} declared {f['declared']}, "
                  f"cast to {f['cast_to']}*: {f['detail']}")
        for f in inherited:
            print(f"{f['path']}:{f['line']}: INHERITED (the baseline has this too) "
                  f"{f['array']} cast to {f['cast_to']}*", file=sys.stderr)
        for u in unknown:
            print(f"{u['path']}:{u['line']}: UNPARSEABLE {u['array']}: {u['reason']}",
                  file=sys.stderr)
        if not findings and not unknown:
            print("no misaligned vector casts introduced (outer strides only; "
                  "innermost index granularity is not decidable here)")
    if unknown:
        return 2
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
