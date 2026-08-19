#!/usr/bin/env python3
"""Tests for `isa_signals.py`.

Every fixture here is hand-written text, so the whole suite runs with no GPU, no
ROCm and no compiler -- the same property that makes `hip_twin_sync.py`'s tests
runnable anywhere. The numbers in the fixtures were chosen so the expected counts
are obvious by inspection; nothing is excerpted from a real kernel.

Two things are pinned deliberately rather than incidentally:

  * the wait-classification cases are the SAME assertions
    `asm_loop_audit.py::_selftest` carries. `isa_signals.py` restates that logic
    instead of importing it (that tool lives under an expert skill the lane gates
    off by default), and a restatement with no shared test is a fork waiting to
    drift. If either side changes, this file should fail.

  * `mechanism_verdict`'s three-valued truth table, exhaustively. That function is
    the whole gate: a bug that turns its None into False converts "we could not
    tell" into "the engineer was wrong", which is the false negative the module
    was written to prevent, now carrying a receipt.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import isa_signals as S  # noqa: E402

HERE = Path(__file__).resolve().parent

KERNEL = "_Z11gemm_kernelPfS_S_iii"

DISASM = f"""custom_gemm.hsaco:\tfile format elf64-amdgpu

Disassembly of section .text:

0000000000000000 <{KERNEL}>:
\ts_load_dwordx4 s[0:3], s[4:5], 0x0          // 000000000000: C00A0002
\tv_mov_b32_e32 v0, 0                         // 000000000008: 7E000280
\ts_waitcnt lgkmcnt(0)                        // 00000000000C: BF8CC07F
;  %bb.1:
\tglobal_load_dwordx4 v[2:5], v[0:1], off     // 000000000010: DC5C0000
\tds_write_b128 v6, v[2:5]                    // 000000000018: D9BE0000
\ts_barrier                                   // 00000000001C: BF8A0000
\tds_read_b128 v[8:11], v7                    // 000000000020: D9FE0000
\ts_waitcnt lgkmcnt(0) vmcnt(1)               // 000000000024: BF8C0F71
\tv_mfma_f32_16x16x16_bf16 a[0:3], v[8:9], v[10:11], a[0:3]
\ts_nop 7                                     // 00000000002C: BF800007
\tv_cvt_f16_f32_e32 v12, v12                  // 000000000030: 7E188B0C
\tglobal_store_dwordx4 v[0:1], v[2:5], off    // 000000000034: DC7C0000
\ts_endpgm                                    // 000000000038: BF810000
"""

# The same kernel with the staging load NARROWED and the accumulator moved off
# the matrix core: the "parent" in a widen/matrix-core claim test.
DISASM_NARROW = DISASM.replace(
    "global_load_dwordx4 v[2:5], v[0:1], off", "global_load_dword v2, v[0:1], off"
).replace(
    "ds_write_b128 v6, v[2:5]", "ds_write_b32 v6, v2"
).replace(
    "ds_read_b128 v[8:11], v7", "ds_read_b32 v8, v7"
).replace(
    "v_mfma_f32_16x16x16_bf16 a[0:3], v[8:9], v[10:11], a[0:3]",
    "v_fmac_f32_e32 v20, v8, v9"
)


# The shape of the -4.72% patch recorded in `isa_signals/learned_rules.md`: the
# mechanism is instantiated as an ADDITIONAL symbol dispatched on the routes that
# qualify, and every pre-existing symbol is left alone. Appended to a parent's
# disassembly it produces a candidate whose shared kernels are all legitimately
# unchanged.
RE_WAVED_SYMBOL = f"""
0000000000000000 <{KERNEL}_8wave>:
\tglobal_load_dwordx4 v[2:5], v[0:1], off
\tds_write_b128 v6, v[2:5]
\ts_endpgm
"""


# Enough narrow accesses in ONE kernel body to clear the rule thresholds. Kept
# separate from DISASM_NARROW rather than made by repeating it: two spans under one
# symbol are refused as a duplicate, not summed, so a repeated fixture silently
# tests nothing.
NARROW_LDS_LOOP = f"""0000000000000000 <{KERNEL}>:
\tglobal_load_dword v2, v[0:1], off
\tglobal_load_dword v3, v[0:1], off
\tglobal_load_dword v4, v[0:1], off
\tglobal_load_dword v5, v[0:1], off
\tds_read_b32 v8, v7
\tds_read_b32 v9, v7
\tds_read_b32 v10, v7
\tds_write_b32 v6, v2
\ts_endpgm
"""


def notes(scratch: int = 0, vgpr: int = 96, lds: int = 8192, name: str = KERNEL) -> str:
    return f"""Displaying notes found in: .note
  Owner                Data size\tDescription
  AMDGPU               0x00000123\tNT_AMDGPU_METADATA (AMDGPU Metadata)
    AMDGPU Metadata:
        ---
        amdhsa.kernels:
          - .agpr_count:    4
            .group_segment_fixed_size: {lds}
            .max_flat_workgroup_size: 256
            .name:          {name}
            .private_segment_fixed_size: {scratch}
            .sgpr_count:    24
            .symbol:        {name}.kd
            .vgpr_count:    {vgpr}
        amdhsa.target:   amdgcn-amd-amdhsa--gfx942
        amdhsa.version:
          - 1
          - 2
"""


def make_archive(root: Path, disasm: str, note_text: str | None,
                 source_hash: str = "a" * 64, arch: str = "gfx942") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "objects").mkdir(exist_ok=True)
    (root / "objects" / "0.disasm.txt").write_text(disasm, encoding="utf-8")
    if note_text is not None:
        (root / "objects" / "0.notes.txt").write_text(note_text, encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps(
        {"schema": "geak.isa-archive/v1", "arch": arch, "source_hash": source_hash,
         "exit_code": 0, "objects": [{"index": 0}]}, sort_keys=True), encoding="utf-8")
    return root


class WaitClassificationTest(unittest.TestCase):
    """The assertions `asm_loop_audit.py::_selftest` carries, restated against the
    restatement. Both spellings, and the ALU split kept out of the drain ratio."""

    def test_gfx9_spelling(self):
        self.assertEqual(S.classify_wait("s_waitcnt vmcnt(0) lgkmcnt(0)")[0], "drain")
        self.assertEqual(S.classify_wait("s_waitcnt vmcnt(0) lgkmcnt(3)")[0], "relaxed")
        self.assertEqual(S.classify_wait("s_waitcnt_vscnt null, 0x0")[0], "drain")

    def test_gfx11_12_spelling(self):
        self.assertEqual(S.classify_wait("s_wait_dscnt 0x0")[0], "drain")
        self.assertEqual(S.classify_wait("s_wait_dscnt 0x1")[0], "relaxed")
        self.assertEqual(S.classify_wait("s_wait_loadcnt 0x7")[0], "relaxed")
        self.assertEqual(S.classify_wait("s_wait_loadcnt 7")[0], "relaxed")
        self.assertEqual(S.classify_wait("s_wait_kmcnt 0x0")[0], "drain")
        self.assertEqual(S.classify_wait("s_wait_loadcnt_dscnt 0x0")[0], "drain")
        self.assertEqual(S.classify_wait("s_wait_dscnt 0xb")[1], [("dscnt", 11)])

    def test_alu_waits_are_their_own_kind_never_a_memory_drain(self):
        self.assertEqual(S.classify_wait("s_wait_alu depctr_va_vcc(0)")[0], "alu")
        self.assertEqual(S.classify_wait("s_delay_alu instid0(VALU_DEP_1)")[0], "alu")
        self.assertEqual(S.classify("s_wait_alu"), "alu_wait")
        self.assertEqual(S.classify("s_wait_dscnt"), "wait")

    def test_an_unknown_spelling_is_unknown_not_relaxed(self):
        """Counted as a drain by the caller, but reported, so a new spelling shows
        up as a parser gap rather than quietly improving the ratio."""
        self.assertEqual(S.classify_wait("s_waitcnt_future_thing")[0], "unknown")


class ClassificationTest(unittest.TestCase):
    def test_classes_are_disjoint_and_specific_families_win_over_valu(self):
        self.assertEqual(S.classify("v_cvt_f16_f32_e32"), "conversion")
        self.assertEqual(S.classify("v_accvgpr_read_b32"), "accvgpr_move")
        self.assertEqual(S.classify("v_mfma_f32_16x16x16_bf16"), "mfma")
        self.assertEqual(S.classify("v_wmma_f32_16x16x16_f16"), "mfma")
        self.assertEqual(S.classify("v_fmac_f32_e32"), "valu")
        self.assertEqual(S.classify("s_load_dwordx4"), "salu")
        self.assertEqual(S.classify("global_atomic_add"), "atomic")
        self.assertEqual(S.classify("ds_read_b128"), "lds_read")
        self.assertEqual(S.classify("global_load_dwordx4"), "global_load")

    def test_atomics_are_not_counted_as_loads_or_stores(self):
        """`global_atomic_add` starts with `global_` and would fall into the load
        or store bucket if the atomic rule came after them."""
        self.assertEqual(S.classify("global_atomic_add_f32"), "atomic")
        self.assertEqual(S.classify("ds_atomic_add_u32"), "atomic")


class AccessWidthTest(unittest.TestCase):
    def test_known_widths(self):
        self.assertEqual(S.access_bytes("global_load_dwordx4"), 16)
        self.assertEqual(S.access_bytes("global_load_dwordx2"), 8)
        self.assertEqual(S.access_bytes("global_load_dword"), 4)
        self.assertEqual(S.access_bytes("ds_read_b128"), 16)
        self.assertEqual(S.access_bytes("ds_read_b32"), 4)
        self.assertEqual(S.access_bytes("ds_read2_b64"), 8)
        self.assertEqual(S.access_bytes("global_load_ushort"), 2)
        self.assertEqual(S.access_bytes("global_load_ubyte_d16_hi"), 1)
        self.assertEqual(S.access_bytes("global_load_b128"), 16)

    def test_dwordx4_is_not_read_as_dword(self):
        """The ordered table exists for this one case: `..._dwordx4` contains
        `dword`, and an unordered scan reports a 16-byte load as 4."""
        self.assertEqual(S.access_bytes("buffer_load_dwordx4"), 16)

    def test_an_unrecognised_suffix_declines_instead_of_guessing(self):
        self.assertIsNone(S.access_bytes("buffer_load_format_xyzw"))
        self.assertIsNone(S.access_bytes("global_load_something_new"))


class KernelSpanTest(unittest.TestCase):
    def test_objdump_form(self):
        spans = S.kernel_spans(DISASM.splitlines())
        self.assertEqual([s[0] for s in spans], [KERNEL])

    def test_asm_form_and_basic_block_labels_are_not_kernels(self):
        asm = ["_Z3fooPf:", "\tv_mov_b32 v0, 0", ".LBB0_1:",
               "\tv_mov_b32 v1, 0", "\ts_endpgm", ".Lfunc_end0:"]
        spans = S.kernel_spans(asm)
        self.assertEqual([s[0] for s in spans], ["_Z3fooPf"])
        name, start, end = spans[0]
        body = S.analyze_body(asm, start, end)
        self.assertEqual(body["instructions"], 3, "the .LBB label must not split the body")

    def test_two_kernels_in_one_dump_are_two_spans(self):
        text = DISASM + DISASM.replace(KERNEL, "_Z12other_kernelPf")
        spans = S.kernel_spans(text.splitlines())
        self.assertEqual(sorted(s[0] for s in spans), ["_Z11gemm_kernelPfS_S_iii",
                                                       "_Z12other_kernelPf"])

    def test_preamble_outside_any_span_is_not_counted(self):
        """`llvm-objdump` prints `<file>:\\tfile format elf64-amdgpu` before the
        first label; it must not land in an instruction count."""
        signals = S.analyze_disasm(DISASM)
        self.assertNotIn("custom_gemm.hsaco:", signals[KERNEL]["opcodes"])


class BodySignalsTest(unittest.TestCase):
    def setUp(self):
        self.body = S.analyze_disasm(DISASM)[KERNEL]

    def test_counts_are_the_hand_countable_ones(self):
        b = self.body
        self.assertEqual(b["classes"]["mfma"], 1)
        self.assertEqual(b["barriers"], 1)
        self.assertEqual(b["conversions"], 1)
        self.assertEqual(b["accvgpr_moves"], 0)
        self.assertEqual(b["nops"], 1)
        self.assertEqual(b["nop_stall_cycles"], 7)
        self.assertEqual(b["mfma_shapes"], ["16x16x16"])

    def test_wait_quality(self):
        w = self.body["waits"]
        self.assertEqual((w["relaxed"], w["full_drain"]), (1, 1))
        self.assertAlmostEqual(w["drain_ratio"], 0.5)

    def test_access_widths(self):
        self.assertEqual(self.body["global_load_bytes"]["max"], 16)
        self.assertEqual(self.body["global_store_bytes"]["max"], 16)
        self.assertEqual(self.body["lds_access_bytes"]["max"], 16)
        self.assertEqual(self.body["lds_access_bytes"]["accesses"], 2)

    def test_drain_ratio_is_null_not_zero_when_there_are_no_waits(self):
        """A kernel with no memory waits has no drain quality. 0.0 would read as
        'perfectly pipelined', which is a confident wrong answer."""
        asm = ["_Z3fooPf:", "\tv_mov_b32 v0, 0", "\ts_endpgm"]
        body = S.analyze_body(asm, 1, 3)
        self.assertIsNone(body["waits"]["drain_ratio"])


class MetadataTest(unittest.TestCase):
    def test_reads_registers_scratch_and_lds(self):
        entries = S.parse_kernel_metadata(notes(scratch=512, vgpr=200, lds=16384))
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e["name"], KERNEL)
        self.assertEqual(e["vgpr_count"], 200)
        self.assertEqual(e["sgpr_count"], 24)
        self.assertEqual(e["agpr_count"], 4)
        self.assertEqual(e["scratch_bytes"], 512)
        self.assertEqual(e["lds_bytes"], 16384)

    def test_two_kernels_do_not_merge(self):
        text = notes(scratch=0, name="kernel_a") + notes(scratch=64, name="kernel_b")
        entries = {e["name"]: e for e in S.parse_kernel_metadata(text)}
        self.assertEqual(sorted(entries), ["kernel_a", "kernel_b"])
        self.assertEqual(entries["kernel_a"]["scratch_bytes"], 0)
        self.assertEqual(entries["kernel_b"]["scratch_bytes"], 64)

    def test_symbol_suffix_is_stripped_only_as_a_fallback_name(self):
        text = notes().replace(f"            .name:          {KERNEL}\n", "")
        entries = S.parse_kernel_metadata(text)
        self.assertEqual(entries[0]["name"], KERNEL)


class BuildSignalsTest(unittest.TestCase):
    def test_complete_archive_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = make_archive(Path(tmp) / "ir", DISASM, notes())
            payload = S.build_signals(archive)
        self.assertEqual(payload["exit_code"], S.EXIT_OK)
        self.assertEqual(payload["kernel_count"], 1)
        self.assertEqual(payload["arch"], "gfx942")
        self.assertEqual(payload["source_hash"], "a" * 64)
        self.assertTrue(payload["kernels"][0]["resources"]["available"])
        self.assertEqual(payload["kernels"][0]["resources"]["scratch_bytes"], 0)

    def test_no_kernel_is_a_hole_not_a_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = make_archive(Path(tmp) / "ir", "nothing here at all\n", notes())
            payload = S.build_signals(archive)
        self.assertEqual(payload["exit_code"], S.EXIT_HOLE)
        self.assertEqual(payload["kernel_count"], 0)

    def test_missing_metadata_is_partial_and_resources_are_unavailable_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = make_archive(Path(tmp) / "ir", DISASM, None)
            payload = S.build_signals(archive)
        self.assertEqual(payload["exit_code"], S.EXIT_PARTIAL)
        res = payload["kernels"][0]["resources"]
        self.assertFalse(res["available"])
        self.assertNotIn("scratch_bytes", res,
                         "an unread scratch size must be absent, never present as 0")

    def test_missing_manifest_is_reported_because_evidence_needs_an_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "ir"
            (archive / "objects").mkdir(parents=True)
            (archive / "objects" / "0.disasm.txt").write_text(DISASM, encoding="utf-8")
            (archive / "objects" / "0.notes.txt").write_text(notes(), encoding="utf-8")
            payload = S.build_signals(archive)
        self.assertEqual(payload["exit_code"], S.EXIT_PARTIAL)
        self.assertIsNone(payload["source_hash"])
        self.assertTrue(any(u.startswith("manifest:absent") for u in payload["unavailable"]))


class DiffTest(unittest.TestCase):
    def _pair(self, left_disasm, right_disasm, left_notes=None, right_notes=None, claims=()):
        with tempfile.TemporaryDirectory() as tmp:
            a = make_archive(Path(tmp) / "a", left_disasm, left_notes or notes(),
                             source_hash="a" * 64)
            b = make_archive(Path(tmp) / "b", right_disasm, right_notes or notes(),
                             source_hash="b" * 64)
            return S.diff_signals(S.build_signals(a), S.build_signals(b), list(claims))

    def test_identical_codegen_is_named_as_such(self):
        d = self._pair(DISASM, DISASM, claims=["widen_global_load"])
        self.assertTrue(d["unchanged_machine_code"])
        self.assertEqual(d["per_kernel"][0]["opcode_delta"], {})
        self.assertIs(d["mechanism_realized"], False,
                      "an observable claim over byte-identical codegen is refuted")

    def test_a_real_widening_is_realized(self):
        d = self._pair(DISASM_NARROW, DISASM,
                       claims=["widen_global_load", "widen_lds_access",
                               "introduce_matrix_core"])
        self.assertFalse(d["unchanged_machine_code"])
        verdicts = {c["claim"]: c["realized"] for c in d["claims"]}
        self.assertTrue(verdicts["widen_global_load"])
        self.assertTrue(verdicts["widen_lds_access"])
        self.assertTrue(verdicts["introduce_matrix_core"])
        self.assertIs(d["mechanism_realized"], True)
        self.assertEqual(d["claims_refuted"], [])

    def test_a_claim_the_codegen_contradicts_is_refuted(self):
        """The direction said it would widen the load; it narrowed it instead."""
        d = self._pair(DISASM, DISASM_NARROW, claims=["widen_global_load"])
        verdicts = {c["claim"]: c["realized"] for c in d["claims"]}
        self.assertIs(verdicts["widen_global_load"], False)
        self.assertEqual(d["claims_refuted"], ["widen_global_load"])
        self.assertIs(d["mechanism_realized"], False)

    def test_a_mechanism_in_a_NEW_symbol_is_indeterminate_not_refuted(self):
        """A symbol the parent does not have cannot be diffed, so a claim that
        landed there is unjudged rather than contradicted.

        This is the trap `learned_rules.md` records under "Two ISA-evidence
        validity traps": a verified, correctness-passing -4.72% patch came back
        `refuted` because the differ pairs only shared symbols and all of those
        were legitimately unchanged. Refuting on that basis costs three things at
        once -- `gate` rejects the candidate, `isaEvidenceDepth` sends the next
        round to the compiler role to explain a refusal that never happened, and
        the ledger records a mechanism as tested when it was not.
        """
        d = self._pair(DISASM_NARROW, DISASM_NARROW + RE_WAVED_SYMBOL,
                       claims=["widen_global_load"])
        self.assertEqual(d["only_in_to"], [f"{KERNEL}_8wave"])
        self.assertFalse(d["unchanged_machine_code"])
        self.assertIsNone(d["claims"][0]["realized"])
        self.assertIn("no parent to diff against", d["claims"][0]["evidence"])
        self.assertEqual(d["claims_refuted"], [])
        self.assertIsNone(d["mechanism_realized"])
        self.assertEqual(d["exit_code"], S.EXIT_PARTIAL)

    def test_a_new_symbol_does_not_mask_a_claim_the_shared_kernels_carry(self):
        """The branch above must not swallow a positive: a widening that landed in
        an existing kernel is still `realized` when the patch also adds a symbol."""
        d = self._pair(DISASM_NARROW, DISASM + RE_WAVED_SYMBOL,
                       claims=["widen_global_load"])
        self.assertEqual(d["only_in_to"], [f"{KERNEL}_8wave"])
        self.assertIs(d["claims"][0]["realized"], True)
        self.assertIs(d["mechanism_realized"], True)

    def test_remove_spill_is_indeterminate_without_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = make_archive(Path(tmp) / "a", DISASM, None)
            b = make_archive(Path(tmp) / "b", DISASM_NARROW, None)
            d = S.diff_signals(S.build_signals(a), S.build_signals(b), ["remove_spill"])
        self.assertIsNone(d["claims"][0]["realized"])
        self.assertEqual(d["claims_indeterminate"], ["remove_spill"])
        self.assertEqual(d["exit_code"], S.EXIT_PARTIAL)

    def test_remove_spill_realized_reads_the_metadata(self):
        d = self._pair(DISASM, DISASM_NARROW,
                       left_notes=notes(scratch=512), right_notes=notes(scratch=0),
                       claims=["remove_spill"])
        self.assertIs(d["claims"][0]["realized"], True)
        self.assertIn("512 -> 0", d["claims"][0]["evidence"])

    def test_remove_spill_is_indeterminate_when_the_parent_never_spilled(self):
        """Not False: there was nothing to remove, so the claim is ill-posed rather
        than refuted, and naming it a refutation would blame the engineer for the
        planner's mistake."""
        d = self._pair(DISASM, DISASM_NARROW,
                       left_notes=notes(scratch=0), right_notes=notes(scratch=0),
                       claims=["remove_spill"])
        self.assertIsNone(d["claims"][0]["realized"])

    def test_none_observable_never_decides_anything(self):
        d = self._pair(DISASM, DISASM, claims=["none_observable"])
        self.assertIsNone(d["mechanism_realized"],
                          "a host-side change over identical device code is not a refutation")
        self.assertEqual(d["claims_refuted"], [])

    def test_unmatched_kernels_are_not_called_unchanged(self):
        renamed = DISASM.replace(KERNEL, "_Z13gemm_kernel_v2PfS_S_iii")
        d = self._pair(DISASM, renamed, claims=["widen_global_load"])
        self.assertFalse(d["unchanged_machine_code"])
        self.assertEqual(d["only_in_from"], [KERNEL])
        self.assertEqual(d["exit_code"], S.EXIT_HOLE)

    def test_an_unknown_claim_id_is_never_a_pass(self):
        d = self._pair(DISASM_NARROW, DISASM, claims=["make_it_fast"])
        self.assertIsNone(d["claims"][0]["realized"])
        self.assertIn("unknown claim id", d["claims"][0]["evidence"])

    def test_source_hashes_travel_with_the_verdict(self):
        d = self._pair(DISASM_NARROW, DISASM, claims=["widen_global_load"])
        self.assertEqual(d["from"]["source_hash"], "a" * 64)
        self.assertEqual(d["to"]["source_hash"], "b" * 64)


class MechanismVerdictTest(unittest.TestCase):
    """Exhaustive, because collapsing the None is the one bug that inverts the
    purpose of the module."""

    @staticmethod
    def _v(*values):
        return [{"claim": f"c{i}", "realized": v} for i, v in enumerate(values)]

    def test_truth_table(self):
        cases = [
            ([], False, None),                    # nothing observable to judge
            ([], True, None),
            (self._v(True), False, True),
            (self._v(True, None), False, True),
            (self._v(None), False, None),         # evidence missing, not a refutation
            (self._v(None, None), False, None),
            (self._v(False), False, False),
            (self._v(True, False), False, False),  # any refutation dominates
            (self._v(None, False), False, False),
            (self._v(True), True, False),         # identical codegen dominates a pass
            (self._v(None), True, False),
        ]
        for observable, unchanged, expected in cases:
            with self.subTest(observable=[o["realized"] for o in observable],
                              unchanged=unchanged):
                self.assertIs(S.mechanism_verdict(observable, unchanged), expected)


class ChecksTest(unittest.TestCase):
    def _checks(self, disasm=DISASM, note_text=None):
        with tempfile.TemporaryDirectory() as tmp:
            archive = make_archive(Path(tmp) / "ir", disasm, note_text or notes())
            return S.run_checks(S.build_signals(archive))

    def test_spill_is_the_only_high_severity_rule(self):
        payload = self._checks(note_text=notes(scratch=512))
        high = [f for f in payload["findings"] if f["severity"] == "high"]
        self.assertEqual([f["rule"] for f in high], ["spill_to_scratch"])
        self.assertEqual(high[0]["observed"], 512)
        self.assertEqual(high[0]["expected"], 0)

    def test_a_clean_kernel_raises_no_high_finding(self):
        payload = self._checks()
        self.assertEqual(payload["high"], 0)

    def test_narrow_accesses_are_advisory_not_defects(self):
        payload = self._checks(disasm=NARROW_LDS_LOOP)
        rules = {f["rule"]: f["severity"] for f in payload["findings"]}
        self.assertEqual(rules.get("narrow_lds_access"), "advisory")
        self.assertEqual(rules.get("narrow_global_load"), "advisory")
        self.assertEqual(payload["high"], 0,
                         "a narrow access is a judgement call, never a high-severity defect")

    def test_the_access_rules_stay_quiet_below_their_evidence_threshold(self):
        """Two narrow accesses is not a pattern. The threshold exists so a kernel
        with one scalar tail load does not read as a vectorization failure."""
        payload = self._checks(disasm=DISASM_NARROW)
        rules = {f["rule"] for f in payload["findings"]}
        self.assertNotIn("narrow_lds_access", rules)

    def test_every_rule_names_a_reference_card(self):
        payload = self._checks(note_text=notes(scratch=512))
        for finding in payload["findings"]:
            self.assertIn(finding["rule"], S.CHECK_CARDS)
            self.assertTrue(finding["reference"].startswith("perf_knowledge/"))


class ObservedDescriptorTest(unittest.TestCase):
    """The axes the opcodes can decide, and -- more importantly -- the ones they
    cannot. Every `assertIsNone` here is the point of the feature: an axis filled
    with a plausible default is how a candidate gets filed under a mechanism it does
    not have, and a wrong classification with evidence attached is worse than none."""

    def _desc(self, disasm=DISASM, note_text=None):
        with tempfile.TemporaryDirectory() as tmp:
            archive = make_archive(Path(tmp) / "ir", disasm, note_text or notes())
            return S.build_descriptors(S.build_signals(archive))

    def test_matrix_core_is_read_off_the_opcodes(self):
        payload = self._desc()
        k = payload["kernels"][0]
        self.assertEqual(k["descriptor"]["compute_primitive"], "matrix_core")
        self.assertIn("v_mfma", k["evidence"]["compute_primitive"].replace("matrix-core", "v_mfma"))

    def test_it_refuses_to_split_rocwmma_from_native_mfma(self):
        """The rule `verify_engineer.md` states: those two emit the same opcodes on
        gfx942, so a finer value would be a guess dressed as a reading."""
        k = self._desc()["kernels"][0]
        self.assertEqual(k["descriptor"]["compute_primitive"], "matrix_core")
        self.assertNotIn(k["descriptor"]["compute_primitive"], ("rocwmma", "native_mfma"))
        self.assertIn("rocwmma", k["evidence"]["compute_primitive"])

    def test_valu_is_the_absence_of_matrix_ops_not_a_default(self):
        k = self._desc(disasm=DISASM_NARROW)["kernels"][0]
        self.assertEqual(k["descriptor"]["compute_primitive"], "valu")
        self.assertIn("no v_mfma", k["evidence"]["compute_primitive"])

    def test_direct_global_when_there_is_no_lds(self):
        asm = f"""0000000000000000 <{KERNEL}>:
\tglobal_load_dwordx4 v[2:5], v[0:1], off
\tv_fmac_f32_e32 v20, v2, v3
\tglobal_store_dwordx4 v[0:1], v[2:5], off
\ts_endpgm
"""
        k = self._desc(disasm=asm)["kernels"][0]
        self.assertEqual(k["descriptor"]["k_pipeline"], "direct_global")
        self.assertEqual(k["descriptor"]["output_path"], "direct_store")

    def test_atomics_name_the_output_path(self):
        asm = f"""0000000000000000 <{KERNEL}>:
\tglobal_load_dwordx4 v[2:5], v[0:1], off
\tglobal_atomic_add_f32 v[0:1], v2, off
\ts_endpgm
"""
        k = self._desc(disasm=asm)["kernels"][0]
        self.assertEqual(k["descriptor"]["output_path"], "atomic_fixup")

    def test_wave_schedule_is_always_undecided_and_says_why(self):
        """Every schedule in the vocabulary uses barriers, so a barrier count cannot
        pick between them. Mapping it onto one anyway is the failure this guards."""
        k = self._desc()["kernels"][0]
        self.assertIsNone(k["descriptor"]["wave_schedule"])
        self.assertIn("wave_schedule", k["undecided"])
        self.assertIn("not decidable", k["evidence"]["wave_schedule"])

    def test_an_ambiguous_output_path_is_undecided_rather_than_guessed(self):
        """DISASM has both ds_write and global_store, which is consistent with a
        staged store AND with LDS holding only operands."""
        k = self._desc()["kernels"][0]
        self.assertIsNone(k["descriptor"]["output_path"])
        self.assertIn("not decidable", k["evidence"]["output_path"])

    def test_a_partial_descriptor_exits_partial_not_ok(self):
        payload = self._desc()
        self.assertEqual(payload["exit_code"], S.EXIT_PARTIAL)

    def test_an_empty_archive_is_a_hole(self):
        payload = self._desc(disasm="nothing here\n")
        self.assertEqual(payload["exit_code"], S.EXIT_HOLE)
        self.assertEqual(payload["kernels"], [])

    def test_every_axis_carries_evidence(self):
        k = self._desc()["kernels"][0]
        for axis in S.DESCRIPTOR_AXES:
            with self.subTest(axis=axis):
                self.assertIn(axis, k["descriptor"])
                self.assertTrue(k["evidence"].get(axis), "an axis with no stated basis "
                                                         "is an assertion, not a reading")


class ClaimVocabularyTest(unittest.TestCase):
    def test_every_claim_has_a_checker_and_the_list_is_not_vacuous(self):
        self.assertGreaterEqual(len(S.CLAIMS), 10)
        for name, checker in S.CLAIMS.items():
            with self.subTest(claim=name):
                self.assertTrue(callable(checker))

    def test_claims_subcommand_prints_the_closed_vocabulary(self):
        payload = self._run(["claims"], expect=S.EXIT_OK)
        self.assertEqual(payload["claims"], sorted(S.CLAIMS))

    @staticmethod
    def _run(args, expect):
        proc = subprocess.run(
            [sys.executable, str(HERE / "isa_signals.py"), *args],
            capture_output=True, text=True, check=False)
        assert proc.returncode == expect, (proc.returncode, proc.stderr)
        return json.loads(proc.stdout)


class CliTest(unittest.TestCase):
    def _run(self, args, expect):
        proc = subprocess.run(
            [sys.executable, str(HERE / "isa_signals.py"), *args],
            capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, expect, proc.stderr)
        return proc.stdout

    def test_exit_code_matches_the_payload_so_a_caller_need_not_parse_to_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = make_archive(Path(tmp) / "ir", DISASM, notes())
            out = self._run(["signals", "--archive", str(archive)], S.EXIT_OK)
            self.assertEqual(json.loads(out)["exit_code"], S.EXIT_OK)
            empty = make_archive(Path(tmp) / "empty", "nothing\n", notes())
            self._run(["signals", "--archive", str(empty)], S.EXIT_HOLE)

    def test_output_is_canonical_so_receipts_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = make_archive(Path(tmp) / "ir", DISASM, notes())
            first = self._run(["signals", "--archive", str(archive)], S.EXIT_OK)
            second = self._run(["signals", "--archive", str(archive)], S.EXIT_OK)
        self.assertEqual(first, second)
        keys = list(json.loads(first).keys())
        self.assertEqual(keys, sorted(keys))

    def test_a_missing_archive_is_a_hole_not_a_traceback(self):
        self._run(["signals", "--archive", "/nonexistent/path/xyz"], S.EXIT_HOLE)

    def test_diff_cli_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = make_archive(Path(tmp) / "a", DISASM_NARROW, notes())
            b = make_archive(Path(tmp) / "b", DISASM, notes())
            out = self._run(["diff", "--from", str(a), "--to", str(b),
                             "--claim", "widen_global_load"], S.EXIT_OK)
        payload = json.loads(out)
        self.assertIs(payload["mechanism_realized"], True)
        self.assertEqual(payload["schema"], S.SCHEMA_DIFF)


if __name__ == "__main__":
    unittest.main(verbosity=2)
