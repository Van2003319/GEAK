#!/usr/bin/env python3
"""Assemble a QD persistence payload from checksummed parts -- finding (120).

Round 15 of the BF16 GEMM search died after 48 minutes with

    QD bootstrap failed closed: canonical artifact/cell manifest was not
    durably persisted

and the receipt said:

    payload checksum mismatch: expected 3acd1667:92453, got 3e05d8cf:92450
    "the payload text was not recoverable verbatim after context compaction"

The second line is the agent's hypothesis, and it is wrong. The file it wrote is
still on disk, and it has exactly one non-ASCII character in it: a literal
em-dash at offset 8892, inside an evidence string reading "correctness fails
silently -- no route survives a narrowing of it". The lane emits pure ASCII, so
what it sent there was the six characters `\\u2014`; three UTF-8 bytes came back
instead. 92453 - 6 + 3 = 92450. The whole discrepancy, to the byte.

That is worth stating plainly because the two diagnoses call for opposite fixes,
and the wrong one is the plausible one. A `\\uXXXX` escape is the single
construct in this transport that a transcriber can "correct" while sincerely
believing it is copying: the escape and the character it denotes render the
same and hash differently. No amount of "byte for byte" in the prompt addresses
that. So the lane now folds the payload to 7-bit ASCII before serializing it --
`—` becomes `--` -- and there is no escape left to unescape.

The rest of this script addresses the other half: not what corrupted the
payload, but why any corruption at all was fatal. The QD archive is written by
`qd_persist_manifest.py`, which is deterministic and has no judgement in it
(96). But the workflow sandbox cannot touch the filesystem, so the payload
reaches that script by being printed into an agent's prompt and retyped into a
file, and the old prompt asked for all 92 KB in one piece:

  * ~92 KB of one-line ASCII JSON is call it 30k tokens;
  * writing it back out as a heredoc costs the same again;
  * so one attempt is ~60k tokens of pure transcription, and the retry the
    prompt demands on mismatch is another 60k.

One wrong character therefore cost the run everything. The fix is to stop
requiring that 92 KB survive in one piece:

  * the payload is split into small parts, each with its own checksum, so a
    corrupted transcription is localized instead of total;
  * each part is written the moment it is read, so the distance between reading
    a byte and writing it is a few thousand bytes rather than the whole file;
  * a retry rewrites only the parts that failed -- a few KB, not 92.

This script is the receiving half: it checks every part against the checksum
the lane computed, refuses to assemble if any part is wrong, and names the
parts to rewrite. It writes nothing unless every part is right, so a partial
transcription can never reach `qd_persist_manifest.py` as a smaller-but-valid
payload.

Usage:
    qd_payload_assemble.py --parts-dir DIR --out FILE \\
        --expect <fnv1a32>:<bytes>,<fnv1a32>:<bytes>,...   # one per part, in order
        [--expect-total <fnv1a32>:<bytes>]                 # of the concatenation

Exit codes: 0 assembled, 3 a part is missing/wrong (nothing written),
2 usage error.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple


def fnv1a32(text: str) -> str:
    """FNV-1a/32 over the ASCII bytes of `text`.

    Must stay byte-for-byte identical to `qdFnv1a32` in kernel_lane.js and to
    `fnv1a32` in qd_persist_manifest.py. All three mask to 32 bits after every
    multiply.
    """
    h = 0x811C9DC5
    for byte in text.encode("utf-8", "surrogatepass"):
        h = ((h ^ byte) * 0x01000193) & 0xFFFFFFFF
    return f"{h:08x}"


def part_path(parts_dir: Path, index: int) -> Path:
    return parts_dir / f"{index:03d}.part"


def parse_expect(spec: str) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for i, item in enumerate(spec.split(",")):
        item = item.strip()
        if not item:
            continue
        digest, _, length = item.partition(":")
        if len(digest) != 8 or not all(c in "0123456789abcdef" for c in digest) \
                or not length.isdigit():
            raise ValueError(f"--expect entry {i} is not <fnv1a32>:<bytes>: {item!r}")
        out.append((digest, int(length)))
    if not out:
        raise ValueError("--expect listed no parts")
    return out


def read_part(path: Path) -> str:
    """The content of one part, with the newline a heredoc appends removed.

    `cat > f <<'EOF'` always terminates the last line, so a part whose payload
    text does not end in a newline arrives with one extra byte. Exactly one is
    stripped -- not `rstrip()`, which would also eat a newline that was part of
    the payload and make two different payloads hash the same.
    """
    text = path.read_text()
    if text.endswith("\n"):
        text = text[:-1]
    return text


def check_parts(parts_dir: Path, expect: List[Tuple[str, int]]) -> Tuple[List[str], List[int]]:
    """Returns (report lines, indices of parts that need rewriting)."""
    lines: List[str] = []
    bad: List[int] = []
    for i, (digest, length) in enumerate(expect):
        path = part_path(parts_dir, i)
        if not path.is_file():
            lines.append(f"part {i:03d} MISSING {path}")
            bad.append(i)
            continue
        text = read_part(path)
        if "\n" in text:
            # The payload is one line of JSON by construction, so an interior
            # newline means the part was pretty-printed or line-wrapped on the
            # way through. Reported separately from a plain mismatch because
            # the fix is different: the transcriber reformatted rather than
            # dropped, and telling it "rewrite it" without saying that invites
            # the same reformatting again.
            lines.append(f"part {i:03d} REFLOWED (contains {text.count(chr(10))} embedded "
                         f"newline(s); the payload is a single line -- write it with a quoted "
                         f"heredoc and do not wrap it)")
            bad.append(i)
            continue
        nonascii = [(off, ch) for off, ch in enumerate(text) if ord(ch) > 0x7F]
        if nonascii:
            # This is round 15's actual failure. The payload is emitted as pure
            # ASCII; a `—` in it came back as a literal em-dash, three bytes
            # instead of six, and the only visible symptom was a checksum three
            # bytes short. "MISMATCH" would send the transcriber back to copy the
            # same characters the same way, because on the screen they are the
            # same characters. Naming them is the difference between one retry
            # and an unwinnable loop.
            where = ", ".join(f"offset {off} U+{ord(c):04X} {c!r}" for off, c in nonascii[:4])
            lines.append(f"part {i:03d} NON-ASCII ({len(nonascii)} character(s): {where}"
                         f"{', ...' if len(nonascii) > 4 else ''}). The payload is 7-bit ASCII "
                         f"throughout; a JSON \\uXXXX escape was written as the character it "
                         f"denotes. Copy the escape sequence literally, backslash and all.")
            bad.append(i)
            continue
        got = f"{fnv1a32(text)}:{len(text.encode())}"
        want = f"{digest}:{length}"
        if got != want:
            lines.append(f"part {i:03d} MISMATCH expected={want} got={got}")
            bad.append(i)
        else:
            lines.append(f"part {i:03d} OK {want}")
    return lines, bad


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parts-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect", required=True,
                    help="comma-separated <fnv1a32>:<bytes>, one per part, in order")
    ap.add_argument("--expect-total", dest="total",
                    help="<fnv1a32>:<bytes> of the concatenated payload")
    args = ap.parse_args(argv)

    try:
        expect = parse_expect(args.expect)
    except ValueError as exc:
        print(f"ASSEMBLE FAILED: {exc}", file=sys.stderr)
        return 2

    parts_dir = Path(args.parts_dir)
    if not parts_dir.is_dir():
        print(f"ASSEMBLE FAILED: --parts-dir {parts_dir} is not a directory", file=sys.stderr)
        return 3

    lines, bad = check_parts(parts_dir, expect)
    for line in lines:
        print(line)

    # A leftover part from a longer previous payload is not itself an error --
    # nothing indexes it -- but it is worth naming, because the usual reason a
    # stale one exists is that a rewrite went to the wrong directory.
    strays = sorted(p.name for p in parts_dir.glob("*.part")
                    if p.name not in {part_path(parts_dir, i).name for i in range(len(expect))})
    if strays:
        print(f"note: {len(strays)} unindexed .part file(s) in {parts_dir} are being ignored: "
              f"{', '.join(strays[:6])}{', ...' if len(strays) > 6 else ''}")

    if bad:
        names = ", ".join(f"{i:03d}" for i in bad)
        print(f"ASSEMBLE FAILED: {len(bad)} of {len(expect)} part(s) do not match. "
              f"Rewrite ONLY these part file(s) and run this command again: {names}. "
              f"Nothing was written to {args.out}.")
        return 3

    text = "".join(read_part(part_path(parts_dir, i)) for i in range(len(expect)))
    total = f"{fnv1a32(text)}:{len(text.encode())}"
    if args.total and total != args.total.strip():
        # Every part matched and the whole does not. That is not a
        # transcription fault -- it is this script's own arithmetic disagreeing
        # with the lane's, e.g. a part list that does not cover the payload.
        # Writing the file anyway would hand qd_persist_manifest.py something
        # that fails its checksum with no indication of which layer was wrong.
        print(f"ASSEMBLE FAILED: every part matched but the concatenation is {total}, "
              f"expected {args.total.strip()}. The part list does not describe this payload; "
              f"this is a lane fault, not a transcription fault. Nothing was written.")
        return 3

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(out)
    # Re-read rather than trust the write: this is the same discipline the
    # persist receipt is built on, and it costs one stat and one read.
    back = out.read_text()
    landed = f"{fnv1a32(back)}:{len(back.encode())}"
    if landed != total:
        print(f"ASSEMBLE FAILED: {out} reads back as {landed}, not {total}.")
        return 3
    print(f"ASSEMBLED {out} {landed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
