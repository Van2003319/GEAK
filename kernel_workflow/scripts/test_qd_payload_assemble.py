#!/usr/bin/env python3
"""Tests for the chunked QD payload transport -- finding (120).

The thing under test is a transcription channel, so the tests are mostly about
what happens when the transcription is imperfect: a dropped byte, a dropped
part, a reflowed part, a stale part left behind. The one property that matters
more than any individual failure mode is that nothing reaches
`qd_persist_manifest.py` unless every part is right -- a payload that is
*almost* correct is not a smaller update, it is an unknown one.

The last class checks the emitting half in `kernel_lane.js`: a receiver that
verifies parts is useless if the lane still prints one 92 KB blob.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "qd_payload_assemble.py"
LANE = HERE.parent / "kernel_lane.js"

sys.path.insert(0, str(HERE))
from qd_payload_assemble import fnv1a32  # noqa: E402


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True, timeout=120)


def chunk(text: str, size: int) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


class Harness(unittest.TestCase):
    """A payload shaped like the real one: a single line of ASCII JSON, big
    enough to need several parts."""

    CHUNK = 512

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.parts_dir = self.root / "parts"
        self.parts_dir.mkdir()
        self.out = self.root / "payload.json"
        payload = {"archive_dir": "/eval/qd_archive", "manifest": {"generation": 0},
                   "admissions": [{"elite_id": f"seed_case_{i}_abc123def456",
                                   "cell": f"case_{i}|native_mfma|symmetric_interleave",
                                   "evidence": "disasm: v_mfma_f32_16x16x16_bf16 " * 6}
                                  for i in range(12)]}
        self.text = json.dumps(payload)
        self.parts = chunk(self.text, self.CHUNK)
        self.assertGreater(len(self.parts), 3, "the fixture must span several parts")

    # -- helpers ---------------------------------------------------------------

    def write_parts(self, parts=None, trailing_newline=True):
        for i, part in enumerate(parts if parts is not None else self.parts):
            (self.parts_dir / f"{i:03d}.part").write_text(part + ("\n" if trailing_newline else ""))

    def expect_spec(self, parts=None) -> str:
        return ",".join(f"{fnv1a32(p)}:{len(p.encode())}"
                        for p in (parts if parts is not None else self.parts))

    def total_spec(self) -> str:
        return f"{fnv1a32(self.text)}:{len(self.text.encode())}"

    def assemble(self, *extra: str) -> subprocess.CompletedProcess:
        return run("--parts-dir", str(self.parts_dir), "--out", str(self.out),
                   "--expect", self.expect_spec(), *extra)


class HappyPathTest(Harness):
    def test_correctly_transcribed_parts_assemble_to_the_payload(self):
        self.write_parts()
        proc = self.assemble("--expect-total", self.total_spec())
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(self.out.read_text(), self.text)
        self.assertIn("ASSEMBLED", proc.stdout)

    def test_the_heredoc_newline_is_stripped_but_only_one(self):
        """`cat <<'EOF'` always terminates the last line. If the assembler used
        rstrip() instead of removing exactly one newline, two different payloads
        would concatenate to the same bytes."""
        self.write_parts(trailing_newline=True)
        a = self.assemble(); self.assertEqual(a.returncode, 0, a.stdout)
        first = self.out.read_text()
        self.write_parts(trailing_newline=False)
        b = self.assemble(); self.assertEqual(b.returncode, 0, b.stdout)
        self.assertEqual(first, self.out.read_text())
        self.assertEqual(first, self.text)

    def test_it_reports_every_part_it_checked(self):
        self.write_parts()
        proc = self.assemble()
        for i in range(len(self.parts)):
            self.assertIn(f"part {i:03d} OK", proc.stdout)


class TranscriptionFailureTest(Harness):
    def test_a_single_dropped_byte_is_caught_and_localized(self):
        """Round 15 in miniature: 92453 bytes emitted, 92450 arrived. The whole
        point of the parts is that the report names which few KB to redo."""
        damaged = list(self.parts)
        damaged[2] = damaged[2][:-3]
        self.write_parts(damaged)
        proc = self.assemble()
        self.assertEqual(proc.returncode, 3)
        self.assertIn("part 002 MISMATCH", proc.stdout)
        self.assertIn("Rewrite ONLY these part file(s)", proc.stdout)
        self.assertIn("002", proc.stdout.rsplit("Rewrite ONLY", 1)[1])
        self.assertFalse(self.out.exists(),
                         "a payload missing three bytes was assembled anyway")

    def test_only_the_damaged_parts_are_named(self):
        damaged = list(self.parts)
        damaged[1] = damaged[1].replace("seed_", "SEED_", 1)
        damaged[4] = damaged[4] + "x"
        self.write_parts(damaged)
        proc = self.assemble()
        self.assertEqual(proc.returncode, 3)
        named = proc.stdout.rsplit("Rewrite ONLY", 1)[1]
        self.assertIn("001", named)
        self.assertIn("004", named)
        self.assertNotIn("003", named)

    def test_a_missing_part_is_not_silently_skipped(self):
        """The failure that would be worst: a shorter payload that still parses
        as JSON is exactly the thing qd_persist_manifest.py's checksum exists to
        refuse, and it should never get that far."""
        self.write_parts()
        (self.parts_dir / f"{3:03d}.part").unlink()
        proc = self.assemble()
        self.assertEqual(proc.returncode, 3)
        self.assertIn("part 003 MISSING", proc.stdout)
        self.assertFalse(self.out.exists())

    def test_a_reflowed_part_says_so_rather_than_just_mismatching(self):
        damaged = list(self.parts)
        damaged[0] = damaged[0].replace(",", ",\n", 3)
        self.write_parts(damaged)
        proc = self.assemble()
        self.assertEqual(proc.returncode, 3)
        self.assertIn("part 000 REFLOWED", proc.stdout)
        self.assertIn("quoted heredoc", proc.stdout)

    def test_an_unescaped_json_escape_is_named_as_such(self):
        """Round 15 exactly: the lane sent the six characters \\u2014 and the
        transcriber wrote the em-dash they denote. On screen those are the same
        thing, so a bare MISMATCH sends it back to make the same substitution."""
        damaged = list(self.parts)
        damaged[1] = damaged[1].replace("disasm:", "disasm—", 1)
        self.write_parts(damaged)
        proc = self.assemble()
        self.assertEqual(proc.returncode, 3)
        self.assertIn("part 001 NON-ASCII", proc.stdout)
        self.assertIn("U+2014", proc.stdout)
        self.assertIn("escape sequence literally", proc.stdout)
        self.assertFalse(self.out.exists())

    def test_the_byte_arithmetic_of_round_15_is_reproduced(self):
        """Six ASCII characters out, three UTF-8 bytes back: -3 bytes, which is
        the entire discrepancy the round 15 receipt reported."""
        before = len("\\u2014".encode())
        after = len("—".encode())
        self.assertEqual(before - after, 3)
        self.assertEqual(92453 - before + after, 92450)

    def test_an_empty_part_file_is_a_mismatch_not_an_assembly(self):
        self.write_parts()
        (self.parts_dir / "002.part").write_text("")
        proc = self.assemble()
        self.assertEqual(proc.returncode, 3)
        self.assertIn("part 002 MISMATCH", proc.stdout)

    def test_a_repaired_part_assembles_on_the_second_run(self):
        damaged = list(self.parts)
        damaged[2] = damaged[2][:-3]
        self.write_parts(damaged)
        self.assertEqual(self.assemble().returncode, 3)
        (self.parts_dir / "002.part").write_text(self.parts[2] + "\n")
        proc = self.assemble("--expect-total", self.total_spec())
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertEqual(self.out.read_text(), self.text)


class RefusalTest(Harness):
    def test_a_stale_extra_part_is_reported_and_ignored(self):
        self.write_parts()
        (self.parts_dir / "099.part").write_text("left over from a longer payload\n")
        proc = self.assemble()
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("099.part", proc.stdout)
        self.assertEqual(self.out.read_text(), self.text)

    def test_a_part_list_that_does_not_describe_the_payload_is_a_lane_fault(self):
        """Every part matches its own checksum and the whole still disagrees.
        Blaming the transcriber here would send it into an unwinnable rewrite
        loop, so the message says which layer is wrong."""
        self.write_parts()
        proc = self.assemble("--expect-total", "deadbeef:12345")
        self.assertEqual(proc.returncode, 3)
        self.assertIn("lane fault", proc.stdout)
        self.assertFalse(self.out.exists())

    def test_a_malformed_expect_spec_is_a_usage_error(self):
        self.write_parts()
        proc = run("--parts-dir", str(self.parts_dir), "--out", str(self.out),
                   "--expect", "notachecksum")
        self.assertEqual(proc.returncode, 2)
        self.assertFalse(self.out.exists())

    def test_a_missing_parts_dir_fails_closed(self):
        proc = run("--parts-dir", str(self.root / "nope"), "--out", str(self.out),
                   "--expect", self.expect_spec())
        self.assertEqual(proc.returncode, 3)


class ChecksumParityTest(unittest.TestCase):
    """Three implementations of FNV-1a/32 now have to agree: this script,
    qd_persist_manifest.py, and qdFnv1a32 in kernel_lane.js. Two of them are
    checkable here without a JS runtime."""

    def test_it_matches_the_persister(self):
        sys.path.insert(0, str(HERE))
        from qd_persist_manifest import fnv1a32 as persister_fnv
        for sample in ("", "a", '{"archive_dir": "/eval"}', "x" * 4096,
                       '{"k": "v_mfma_f32_16x16x16_bf16"}'):
            with self.subTest(sample=sample[:24]):
                self.assertEqual(fnv1a32(sample), persister_fnv(sample))

    def test_the_known_javascript_value_is_reproduced(self):
        # qdFnv1a32('') is the FNV-1a offset basis, 0x811c9dc5, in both
        # implementations -- the one value that pins the seed rather than the
        # multiply.
        self.assertEqual(fnv1a32(""), "811c9dc5")


class LaneEmitsPartsTest(unittest.TestCase):
    """The receiving half is only useful if the lane stopped printing the
    payload as one blob. Round 15's prompt said 'Write PERSIST_PAYLOAD below to
    <file>, byte for byte ... 92453 bytes', and that instruction is what an
    agent cannot reliably carry out."""

    def setUp(self):
        self.lane = LANE.read_text()
        match = re.search(r"const qdPersistPrompt = \(payload, tag\) => \{.*?\n\};",
                          self.lane, re.S)
        self.assertIsNotNone(match, "qdPersistPrompt is no longer where the test looks")
        self.fn = match.group(0)

    def test_the_prompt_chunks_the_payload(self):
        self.assertIn("qd_payload_assemble.py", self.fn,
                      "the persist prompt does not route the payload through the "
                      "checksummed part transport")
        self.assertIn("QD_PAYLOAD_CHUNK_BYTES", self.fn)

    def test_the_chunk_size_is_small_enough_to_rewrite_cheaply(self):
        match = re.search(r"const QD_PAYLOAD_CHUNK_BYTES = (\d+)", self.lane)
        self.assertIsNotNone(match, "the chunk size is not a named constant")
        size = int(match.group(1))
        self.assertGreaterEqual(size, 256)
        self.assertLessEqual(size, 8192,
                             "a part that costs tens of thousands of tokens to rewrite "
                             "reintroduces the compaction that finding (120) is about")

    def test_the_whole_payload_is_still_checksummed_end_to_end(self):
        """Per-part checksums catch a damaged part; only the total catches a
        part list that does not cover the payload."""
        self.assertIn("--expect-total", self.fn)
        self.assertIn("--expect-checksum", self.fn,
                      "qd_persist_manifest.py must still verify the assembled file")

    def test_the_prompt_no_longer_asks_for_one_verbatim_blob(self):
        self.assertNotIn("Write PERSIST_PAYLOAD below to", self.fn)


class LaneFoldsToAsciiTest(unittest.TestCase):
    """The emitted payload must contain no `\\uXXXX` escape for a non-ASCII
    character, because that escape is what round 15's transcriber normalized
    away. Run in a real JS engine where one is available -- a grep for the fold
    table would pass on a fold nobody calls, which is finding (55)."""

    FOLD_CASES = [
        ("em dash — here", "em dash -- here"),
        ("curly ‘quotes’ and “doubles”", "curly 'quotes' and \"doubles\""),
        ("ellipsis… and 4×16", "ellipsis... and 4x16"),
        ("≥ 1.15× at ±2%", ">= 1.15x at +/-2%"),
        ("unmapped ☃ snowman", "unmapped ? snowman"),
    ]

    def setUp(self):
        try:
            from py_mini_racer import MiniRacer
        except ImportError:  # pragma: no cover - environment without an engine
            self.skipTest("no JS engine available (py_mini_racer not installed)")
        src = LANE.read_text()
        block = re.search(r"const QD_ASCII_FOLD = \{.*?\n\};", src, re.S)
        fold = re.search(r"const qdAsciiFoldText = .*?\n(?=const qdAsciiFold =)", src, re.S)
        deep = re.search(r"const qdAsciiFold = \(v\) => \{.*?\n\};", src, re.S)
        ascii_json = re.search(r"const qdAsciiJson = \(obj\) => .*?;\n", src, re.S)
        for name, m in [("QD_ASCII_FOLD", block), ("qdAsciiFoldText", fold),
                        ("qdAsciiFold", deep), ("qdAsciiJson", ascii_json)]:
            self.assertIsNotNone(m, f"{name} is not in kernel_lane.js where the test looks")
        self.ctx = MiniRacer()
        self.ctx.eval("\n".join(m.group(0) for m in (block, fold, deep, ascii_json)))

    def test_the_fold_spells_the_characters_that_actually_appear_in_evidence(self):
        for raw, want in self.FOLD_CASES:
            with self.subTest(raw=raw):
                self.assertEqual(self.ctx.call("qdAsciiFoldText", raw), want)

    def test_folding_reaches_nested_strings_keys_and_arrays(self):
        self.ctx.eval("function foldJson(o){return qdAsciiJson(qdAsciiFold(o));}")
        payload = {"admissions": [{"evidence": ["fails silently — no route"],
                                   "note…": "4×16 tile"}]}
        out = self.ctx.call("foldJson", payload)
        self.assertNotIn("\\u", out, f"an escape survived the fold: {out}")
        self.assertIn("fails silently -- no route", out)
        self.assertIn("note...", out)
        self.assertIn("4x16 tile", out)

    def test_a_folded_payload_is_pure_ascii_and_hashes_the_same_on_both_sides(self):
        fnv = re.search(r"const qdFnv1a32 = .*?\n\};", LANE.read_text(), re.S).group(0)
        self.ctx.eval(fnv)
        self.ctx.eval("function foldJson(o){return qdAsciiJson(qdAsciiFold(o));}")
        text = self.ctx.call("foldJson", {"e": ["— ’ … × ≥ ± µ ° • → ☃"]})
        self.assertTrue(all(ord(c) < 0x80 for c in text), text)
        # qdFnv1a32 throws above U+007F, so this call also proves the fold was
        # total rather than merely mostly right.
        self.assertEqual(self.ctx.call("qdFnv1a32", text), fnv1a32(text))

    def test_the_persist_prompt_folds_before_it_serializes(self):
        fn = re.search(r"const qdPersistPrompt = \(payload, tag\) => \{.*?\n\};",
                       LANE.read_text(), re.S).group(0)
        self.assertIn("qdAsciiJson(qdAsciiFold(payload))", fn,
                      "the payload is serialized without folding, so a \\uXXXX escape "
                      "can still reach the transcriber")


if __name__ == "__main__":
    unittest.main()
