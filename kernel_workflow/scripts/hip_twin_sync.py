#!/usr/bin/env python3
"""Check that every hipified `X.hip` / `X_hip.hip` pair is in lockstep.

Why this exists (finding 87). torch's cpp_extension hipify rewrites `X.hip` into
`X_hip.hip`, and ninja then compiles *the twin*:

    custom_gemm_hip.cuda.o  <-  src/custom_gemm_hip.hip

The build succeeds, the kernel runs, correctness passes, and a benchmark can
report a null result for a change that was never compiled. That is the worst
shape a measurement bug can take: it is indistinguishable from an honest
negative, and an honest negative is what closes a search direction.

WHICH DIRECTION ACTUALLY DRIFTS, MEASURED. This docstring used to say the twin is
skipped whenever it already exists, so an edit to `src/custom_gemm.hip` alone
"changes nothing that runs". That is not what this toolchain does, and the
correction matters because it points the guard at the other file. On torch
2.11.0 / ROCm 7.2.3, `hipify_python.preprocessor` ends:

    do_write = True
    if os.path.exists(fout_path):
        do_write = open(fout_path).read() != output_source
    if do_write: ... "[ok]"
    else:        ... "[skipped, already hipified]"

so the skip means the twin ALREADY EQUALS the freshly hipified original -- it is
up to date, not stale. Verified end to end: editing only the original and
re-running hipify reports `[ok]` and the twin picks up the edit; deleting both
twins and rebuilding the task regenerates them byte-identically in 12s.

The consequence runs the other way. The twin is DERIVED, so an edit applied only
to `src/custom_gemm_hip.hip` is what gets silently discarded, overwritten from
the original by the next build whose sources changed. That is the drift this
tool is worth running for, and it is why the twins are no longer tracked in git
(see .gitignore): a committed copy of generated output is an invitation to edit
the copy.

The `[skipped, no changes]` status is a third case worth knowing: when a source
has no CUDA construct to rewrite, hipify produces NO twin and the original is
compiled directly. That is why `gemm_bindings.cpp` has no partner and why a
missing twin is not by itself a defect.

The rule enforced here: the two files may differ ONLY in hipify's launch-syntax
rewriting. Everything else -- kernel bodies, template parameters, constants,
guards -- must match line for line.

THE HOLE THAT WAS HERE, AND WHAT REPLACED IT. Launch-carrying lines are excluded
from the line comparison by construction, because that is exactly where the two
files are *supposed* to differ. For several rounds this docstring recorded the
consequence as an open hole (finding 54: a hole is worth as much as a pass, but
only if it is labelled): an edit whose entire effect lives on a launch statement
-- a grid dimension, a block size, a stream, a launch-time template argument --
was invisible, and that is a plausible tuning move rather than a theoretical
one. It said closing it "needs the launch sites parsed and normalized into a
common form, which is a different and much larger tool".

It turned out to be a smaller tool than that. hipify's rewrite is mechanical:

    primary:  name<targs><<<grid, block, shmem, stream>>>(args)
    twin:     hipLaunchKernelGGL((name<targs>), dim3(grid), dim3(block),
                                 shmem, stream, args)

The expressions inside are copied verbatim, so parsing both into
`(name, grid, block, shmem, stream, args)` -- unwrapping `dim3(...)`, splitting
on top-level commas only, and discarding whitespace -- makes them directly
comparable. `check_launches` does that, and a launch-only edit now fails.

The residual hole is narrower and it is *loud*: a launch statement this parser
cannot normalize is reported and exits 3, never silently skipped. Two things
still slip through by construction: a difference expressible only as two
spellings of one expression (`n*2` vs `2*n`) reads as drift rather than
agreement -- the safe direction -- and a launch built by a macro this never
sees expanded is compared as written.

Exit codes:  0 = all pairs in lockstep, launches included
             1 = at least one pair drifted (line body or launch)
             2 = nothing to check (no twins found) -- a hole, not a pass
             3 = pairs compared, but a launch could not be normalized -- the
                 launch half is UNCHECKED for that pair. Also not a pass.
"""

import re
import sys
from pathlib import Path

# Lines carrying launch syntax in either dialect. These are the only lines
# hipify is allowed to have rewritten, so they are the only lines excluded.
LAUNCH_RE = re.compile(r"<<<|>>>|hipLaunchKernelGGL|\bdim3\s*\(")


def _statement_span(lines, i):
    """The full statement containing launch-token line `i`, as [start, end].

    Dropping only the token-bearing line is not enough, and the failure is not
    hypothetical -- it produced a false positive on a tree that was actually in
    lockstep. hipify moves the kernel name onto its own rewritten line:

        primary:  tiled_bf16_gemm_kernel<CTA_M, ...>          <- no launch token
                      <<<grid, ...>>>(m, n, k, ...)           <- token
        twin:     hipLaunchKernelGGL(( tiled_bf16_gemm_kernel<CTA_M, ...>)   <- token
                      , dim3(grid), ..., m, n, k, ...         <- token

    The primary's first line carries no token, so a line-wise filter keeps it
    and drops nothing opposite it in the twin. The two kept sequences shift by
    one line and every subsequent line compares unequal. Whole statements have
    to go, or the check reports drift on files that agree.
    """
    start = i
    while start > 0:
        prev = lines[start - 1].strip()
        if prev == "" or prev.endswith((";", "{", "}", ":")) or prev.startswith("#"):
            break
        start -= 1
    end = i
    while end < len(lines) - 1 and ";" not in lines[end]:
        end += 1
    return start, end


def significant_lines(text):
    """Line sequence with whole launch statements removed.

    Returns (kept_lines, n_dropped). Trailing whitespace is normalized because
    hipify reindents around the launches it rewrites, and reindentation is not
    a semantic difference.

    Widening the exclusion from lines to statements widens the blind spot
    described in the module docstring: any edit confined to a launch statement
    is invisible here, not just an edit confined to a launch line.
    """
    lines = text.splitlines()
    drop = [False] * len(lines)
    for i, line in enumerate(lines):
        if LAUNCH_RE.search(line):
            start, end = _statement_span(lines, i)
            for j in range(start, end + 1):
                drop[j] = True
    kept = [ln.rstrip() for ln, d in zip(lines, drop) if not d]
    return kept, sum(drop)


def _split_top(text):
    """Split on commas at paren/bracket/brace depth zero.

    Angle brackets are deliberately NOT tracked: `<` is also less-than, and a
    depth counter that cannot tell the two apart mis-splits real expressions.
    The only place template commas appear is inside the kernel name, which both
    dialects keep behind parentheses or before `<<<`, so paren depth is enough.
    """
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p.strip() for p in parts]


def _inner(text, open_index):
    """Contents of the parenthesis opening at `open_index`, or None if unclosed."""
    depth = 0
    for i in range(open_index, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[open_index + 1:i]
    return None


def _strip_outer_parens(text):
    text = text.strip()
    while text.startswith("(") and _inner(text, 0) is not None \
            and len(_inner(text, 0)) == len(text) - 2:
        text = text[1:-1].strip()
    return text


def _unwrap_dim3(text):
    text = text.strip()
    if text.startswith("dim3"):
        rest = text[4:].lstrip()
        if rest.startswith("(") and _inner(rest, 0) is not None \
                and len(_inner(rest, 0)) == len(rest) - 2:
            return _inner(rest, 0).strip()
    return text


def _squeeze(text):
    return "".join(str(text).split())


def parse_launch(statement):
    """Normalize one launch statement, or None if it is not one this can read.

    Returns `(name, grid, block, shmem, stream, args)` with all whitespace
    removed, so the two dialects land on the same tuple.
    """
    text = " ".join(str(statement).split())
    if "hipLaunchKernelGGL" in text:
        start = text.index("hipLaunchKernelGGL")
        paren = text.find("(", start)
        inner = _inner(text, paren) if paren >= 0 else None
        if inner is None:
            return None
        parts = _split_top(inner)
        if len(parts) < 5:
            return None
        return (_squeeze(_strip_outer_parens(parts[0])),
                _squeeze(_unwrap_dim3(parts[1])), _squeeze(_unwrap_dim3(parts[2])),
                _squeeze(parts[3]), _squeeze(parts[4]),
                tuple(_squeeze(p) for p in parts[5:] if p.strip()))
    if "<<<" in text and ">>>" in text:
        head, rest = text.split("<<<", 1)
        config, tail = rest.split(">>>", 1)
        parts = _split_top(config)
        if len(parts) < 2:
            return None
        # `<<<grid, block>>>` leaves shmem and stream at the CUDA/HIP defaults;
        # the twin always writes them out, so the defaults must be filled in
        # here or every two-argument launch would read as drift.
        while len(parts) < 4:
            parts.append("0")
        paren = tail.find("(")
        inner = _inner(tail, paren) if paren >= 0 else None
        if inner is None:
            return None
        return (_squeeze(_strip_outer_parens(head)),
                _squeeze(_unwrap_dim3(parts[0])), _squeeze(_unwrap_dim3(parts[1])),
                _squeeze(parts[2]), _squeeze(parts[3]),
                tuple(_squeeze(a) for a in _split_top(inner) if a.strip()))
    return None


def launch_statements(text):
    """Every launch statement in `text`, whole, in source order.

    Same spans `significant_lines` removes -- these two functions partition the
    file between them, which is the property that makes the pair of checks
    complete rather than two overlapping samples of it.
    """
    lines = text.splitlines()
    spans, seen = [], set()
    for i, line in enumerate(lines):
        if LAUNCH_RE.search(line):
            span = _statement_span(lines, i)
            if span not in seen:
                seen.add(span)
                spans.append(span)
    return [" ".join(lines[s:e + 1]).strip() for s, e in spans]


def check_launches(primary, twin):
    """Compare the launch statements the line check must exclude.

    Returns `(status, detail)` with status in `{'ok', 'drift', 'hole'}`.
    `hole` is not a pass: it means a statement could not be normalized, so the
    launch half of this pair went unchecked and says nothing either way.
    """
    a = launch_statements(primary.read_text(errors="replace"))
    b = launch_statements(twin.read_text(errors="replace"))
    pa, pb = [parse_launch(s) for s in a], [parse_launch(s) for s in b]
    unreadable = [s for s, p in zip(a, pa) if p is None] + \
                 [s for s, p in zip(b, pb) if p is None]
    if unreadable:
        return "hole", ("launch statement(s) this check cannot normalize, so the launch "
                        f"half of this pair is UNCHECKED:\n      {unreadable[0][:150]}")
    if len(pa) != len(pb):
        return "drift", (f"{len(pa)} launch statement(s) in the primary, {len(pb)} in the "
                         "twin -- one file launches something the other does not")
    for index, (x, y) in enumerate(zip(pa, pb)):
        if x != y:
            field = next((f for f, u, v in zip(
                ("kernel", "grid", "block", "shared", "stream", "arguments"), x, y) if u != v),
                "kernel")
            return "drift", (
                f"launch {index + 1} differs in {field}\n"
                f"      primary: {x}\n      twin:    {y}")
    return "ok", f"{len(pa)} launch statement(s) normalize identically"


def find_pairs(root):
    """Every (primary, twin) pair under root. Primaries that are themselves
    twins are skipped, so `a_hip.hip` never pairs with `a_hip_hip.hip`."""
    pairs = []
    for primary in sorted(Path(root).rglob("*.hip")):
        if primary.stem.endswith("_hip"):
            continue
        twin = primary.with_name(primary.stem + "_hip.hip")
        if twin.is_file():
            pairs.append((primary, twin))
    return pairs


def check_pair(primary, twin):
    """Return (ok, detail)."""
    a, dropped_a = significant_lines(primary.read_text(errors="replace"))
    b, dropped_b = significant_lines(twin.read_text(errors="replace"))
    if a == b:
        return True, f"{len(a)} significant lines match ({dropped_a}/{dropped_b} launch lines excluded)"

    # Report the first divergence with its line number in each file. A count
    # alone ("17 lines differ") does not tell you whether the twin is stale or
    # genuinely different, which is the question being asked.
    for i in range(max(len(a), len(b))):
        av = a[i] if i < len(a) else "<end of file>"
        bv = b[i] if i < len(b) else "<end of file>"
        if av != bv:
            ndiff = sum(1 for j in range(max(len(a), len(b)))
                        if (a[j] if j < len(a) else None) != (b[j] if j < len(b) else None))
            return False, (
                f"{ndiff} significant line(s) differ; first at significant-line {i + 1}\n"
                f"      primary: {av.strip()[:110]}\n"
                f"      twin:    {bv.strip()[:110]}"
            )
    return False, "lengths differ but no differing line found (should be unreachable)"


def main(argv):
    if len(argv) < 2:
        print(__doc__.strip().splitlines()[0])
        print("usage: hip_twin_sync.py <dir> [<dir> ...]", file=sys.stderr)
        return 2

    pairs = []
    for root in argv[1:]:
        if not Path(root).is_dir():
            print(f"  FAIL: not a directory: {root}", file=sys.stderr)
            return 1
        pairs.extend(find_pairs(root))

    if not pairs:
        print("  HOLE: no X.hip/X_hip.hip pairs found -- nothing was checked.")
        print("        This is not evidence that the build is in lockstep. If the twins")
        print("        have not been generated yet, run the build once and re-check.")
        return 2

    bad = 0
    holes = 0
    for primary, twin in pairs:
        # Counted per pair, not per check: a pair whose lines AND launches both
        # drifted is one stale twin, and "2 drifted" out of one pair would read
        # as a second problem that does not exist.
        pair_bad = False
        ok, detail = check_pair(primary, twin)
        if ok:
            print(f"  ok:   {primary.name} <-> {twin.name} -- {detail}")
        else:
            pair_bad = True
            print(f"  FAIL: {primary.name} <-> {twin.name} -- {detail}")
            print(f"        Ninja compiles {twin.name}, so whatever differs here is NOT in")
            print(f"        the binary being measured. Apply the edit to both, or delete the")
            print(f"        twin and let hipify regenerate it. (finding 87)")

        # The other half of the file. `significant_lines` drops exactly the
        # statements this reads, so between them the two checks cover the pair
        # rather than sampling it twice.
        status, ldetail = check_launches(primary, twin)
        if status == "ok":
            print(f"  ok:   {primary.name} <-> {twin.name} -- launches: {ldetail}")
        elif status == "drift":
            pair_bad = True
            print(f"  FAIL: {primary.name} <-> {twin.name} -- launches: {ldetail}")
            print(f"        A launch-only edit reaches the binary only through {twin.name}.")
            print(f"        This is the case the line comparison cannot see, which is why")
            print(f"        it is checked separately. (finding 87)")
        else:
            holes += 1
            print(f"  HOLE: {primary.name} <-> {twin.name} -- launches: {ldetail}")
            print(f"        Not a pass. Nothing is claimed about the launch half of this")
            print(f"        pair, so an edit confined to it would still be invisible.")

        bad += 1 if pair_bad else 0

    print(f"\n{len(pairs) - bad} pair(s) in lockstep, {bad} drifted, "
          f"{holes} with an unreadable launch")
    if bad:
        return 1
    return 3 if holes else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
