#!/usr/bin/env python3
"""Tests for `ir_signals.py`, against hand-written IR and MIR.

Fixtures rather than captured dumps, because what is being checked is that the
readers count the right things and -- more important -- that they do not invent
changes. Every fabricated signal this module could emit becomes a pass name in
an L3 report and then a narrowed question for L4, so a parser that silently
stops matching does not degrade to "no finding": it degrades to a confident
attribution of a change that never happened. `PostRegisterAllocationTest` is
that failure, caught once and pinned here.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ir_signals  # noqa: E402


LL = """; ModuleID = 'p.hip'
define protected amdgpu_kernel void @_Z1kPfPKfi(ptr addrspace(1) %0) #5 {
  %4 = ptrtoint ptr addrspace(1) %0 to i64
  %7 = load i16, ptr addrspace(4) %6, align 4, !tbaa !6
  %17 = load <4 x float>, ptr addrspace(1) %16, align 16
  %18 = fmul contract float %17, 2.000000e+00
  store <4 x float> %19, ptr addrspace(1) %15, align 16
  %20 = tail call i32 @llvm.amdgcn.workitem.id.x()
  call void @llvm.amdgcn.s.barrier()
  %21 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(i32 %a)
  br label %22

22:                                               ; preds = %14, %3
  ret void
}
declare void @llvm.amdgcn.s.barrier()
!6 = !{!"omnipotent char"}
"""

MIR_PRE_RA = """# Machine code for function _Z1kPfPKfi: IsSSA, TracksLiveness
Function Live Ins: $vgpr0 in %3

bb.0 (%ir-block.3):
  successors: %bb.1(0x40000000)
  liveins: $vgpr0, $sgpr0_sgpr1
  %5:sgpr_32 = COPY $sgpr2
  %6:sreg_32_xm0_xexec = S_LOAD_DWORD_IMM %4:sgpr_64(p4), 16, 0 :: (load (s32))
  %9:vreg_128 = GLOBAL_LOAD_DWORDX4 %8:vreg_64, 0, 0 :: (load (s128))
  DS_WRITE_B64 %10:vgpr_32, %11:vreg_64, 0, 0
  S_BARRIER
  S_ENDPGM 0
"""

# The same function after register allocation. Every instruction now carries a
# `renamable $physreg` destination, which is the shape the first version of the
# reader could not see.
MIR_POST_RA = """# Machine code for function _Z1kPfPKfi: NoPHIs, NoVRegs, TracksLiveness

bb.0:
  liveins: $vgpr0, $sgpr0_sgpr1
  renamable $sgpr2 = COPY killed renamable $sgpr3
  renamable $sgpr4 = S_LOAD_DWORD_IMM killed renamable $sgpr0_sgpr1, 16, 0 :: (load (s32))
  renamable $vgpr0_vgpr1_vgpr2_vgpr3 = GLOBAL_LOAD_DWORDX4 renamable $vgpr8_vgpr9, 0, 0
  DS_WRITE_B64 killed renamable $vgpr4, killed renamable $vgpr6_vgpr7, 0, 0
  frame-setup S_MOV_B32 $sgpr33, 0
  dead early-clobber renamable $sgpr10 = S_MUL_I32 $sgpr2, 4
  S_BARRIER
  S_WAITCNT 0
  S_ENDPGM 0
"""


def make_archive(root: Path, stages: list[tuple[str, str, str]], **manifest_extra) -> Path:
    """`stages` is a list of (pass_id, extension, body)."""
    archive = root / "arch"
    (archive / "stages").mkdir(parents=True, exist_ok=True)
    entries = []
    for index, (pass_id, ext, body) in enumerate(stages):
        name = f"{index:03d}-{pass_id}{ext}"
        (archive / "stages" / name).write_text(body, encoding="utf-8")
        entries.append({"index": index, "pass_id": pass_id, "pass_name": pass_id,
                        "scope": "_Z1kPfPKfi", "file": f"stages/{name}",
                        "lines": body.count("\n")})
    manifest = {"schema": "geak.ir-archive/v1", "exit_code": 0, "backend": "hip",
                "source_hash": "abc123", "kernel_filter": "_Z1kPfPKfi",
                "provenance": {"checked": True, "ir_binary_equals_measured": True},
                "stages": entries, "holes": []}
    manifest.update(manifest_extra)
    (archive / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2),
                                           encoding="utf-8")
    return archive


class ParseLlTest(unittest.TestCase):
    def setUp(self):
        self.census = ir_signals.parse_ll(LL)

    def test_it_reads_llvm_ir(self):
        self.assertEqual(self.census["kind"], "llvm-ir")

    def test_metadata_declarations_and_headers_are_not_instructions(self):
        for noise in ("declare", "define", "ModuleID"):
            self.assertNotIn(noise, self.census["opcodes"])

    def test_a_leading_modifier_does_not_become_the_opcode(self):
        """`tail call` is the one place LLVM IR puts a word before the opcode."""
        self.assertNotIn("tail", self.census["opcodes"])
        self.assertEqual(self.census["opcodes"]["call"], 3)

    def test_a_vector_access_is_measured_from_its_type(self):
        self.assertEqual(self.census["load_widths_bytes"]["16"], 1)
        self.assertEqual(self.census["store_widths_bytes"]["16"], 1)

    def test_a_scalar_access_is_measured_from_its_element_type(self):
        """`load i16 ... align 4` is a two-byte access. Reading the alignment
        instead would report four and turn a real narrowing into a non-event."""
        self.assertEqual(self.census["load_widths_bytes"]["2"], 1)

    def test_address_spaces_are_counted(self):
        self.assertEqual(self.census["address_spaces"]["addrspace(1)"], 3)
        self.assertEqual(self.census["address_spaces"]["addrspace(4)"], 1)

    def test_structural_intrinsics_are_families_not_just_calls(self):
        """A barrier and an MFMA are structure; counting them only as `call`
        loses exactly the two numbers L3 exists to report."""
        self.assertEqual(self.census["families"]["sync"], 1)
        self.assertEqual(self.census["families"]["matrix"], 1)

    def test_families_cover_memory_and_control_flow(self):
        self.assertEqual(self.census["families"]["memory_load"], 2)
        self.assertEqual(self.census["families"]["memory_store"], 1)
        self.assertEqual(self.census["families"]["control_flow"], 2)


class ParseMirTest(unittest.TestCase):
    def setUp(self):
        self.census = ir_signals.parse_mir(MIR_PRE_RA)

    def test_it_reads_machine_ir(self):
        self.assertEqual(self.census["kind"], "mir")

    def test_block_and_operand_headers_are_not_instructions(self):
        for noise in ("Function", "Live", "IsSSA", "successors", "liveins"):
            self.assertNotIn(noise, self.census["opcodes"])

    def test_a_barrier_is_sync_not_scalar_alu(self):
        """`S_BARRIER` also starts with `S_`. If the catch-all wins, the sync
        count -- the number the paper's own L3 example turns on -- reads zero."""
        self.assertEqual(self.census["families"]["sync"], 1)

    def test_the_widest_suffix_wins(self):
        """`DWORDX4` contains `DWORD`. Matching the short one first would report
        a 16-byte load as 4 bytes and manufacture a narrowing."""
        self.assertEqual(ir_signals.mir_access_bytes("GLOBAL_LOAD_DWORDX4"), 16)
        self.assertEqual(ir_signals.mir_access_bytes("GLOBAL_LOAD_DWORD"), 4)
        self.assertEqual(ir_signals.mir_access_bytes("DS_WRITE_B64"), 8)

    def test_loads_and_stores_are_told_apart_by_the_opcode(self):
        self.assertEqual(self.census["load_widths_bytes"]["16"], 1)
        self.assertEqual(self.census["store_widths_bytes"]["8"], 1)

    def test_lds_and_global_are_separate_families(self):
        self.assertEqual(self.census["families"]["lds"], 1)
        self.assertEqual(self.census["families"]["global_memory"], 1)


class PostRegisterAllocationTest(unittest.TestCase):
    """The regression this module was shipped with once and must never regress to.

    After register allocation every destination is written `renamable $physreg`.
    A reader anchored on a `%vreg` destination sees none of them, so each post-RA
    pass appears to have deleted the instructions it merely renamed -- and the
    trajectory then reports a load narrowing at `virtregrewriter`, a pass that
    cannot narrow a load. That output is indistinguishable from a real finding.
    """

    def setUp(self):
        self.census = ir_signals.parse_mir(MIR_POST_RA)

    def test_post_ra_instructions_are_still_counted(self):
        self.assertGreaterEqual(self.census["instructions"], 9)
        self.assertEqual(self.census["opcodes"]["GLOBAL_LOAD_DWORDX4"], 1)
        self.assertEqual(self.census["opcodes"]["S_LOAD_DWORD_IMM"], 1)

    def test_flag_words_before_the_opcode_are_skipped(self):
        for flag in ("renamable", "killed", "frame-setup", "dead", "early-clobber"):
            self.assertNotIn(flag, self.census["opcodes"])
        self.assertEqual(self.census["opcodes"]["S_MOV_B32"], 1)
        self.assertEqual(self.census["opcodes"]["S_MUL_I32"], 1)

    def test_the_widest_load_survives_register_allocation(self):
        before = ir_signals.parse_mir(MIR_PRE_RA)
        self.assertEqual(ir_signals.widest(before["load_widths_bytes"]),
                         ir_signals.widest(self.census["load_widths_bytes"]))


class ResolveStageTest(unittest.TestCase):
    def setUp(self):
        self.entries = [
            {"index": 0, "pass_id": "SROAPass", "pass_name": "SROAPass", "file": "stages/000-SROAPass.ll"},
            {"index": 1, "pass_id": "InstCombinePass", "pass_name": "InstCombinePass",
             "file": "stages/001-InstCombinePass.ll"},
            {"index": 2, "pass_id": "InstCombinePass", "pass_name": "InstCombinePass",
             "file": "stages/002-InstCombinePass.ll"},
        ]

    def test_an_index_selects_exactly(self):
        self.assertEqual(ir_signals.resolve_stage(self.entries, "2")["index"], 2)

    def test_a_unique_pass_id_selects(self):
        self.assertEqual(ir_signals.resolve_stage(self.entries, "SROAPass")["index"], 0)

    def test_a_file_name_selects(self):
        self.assertEqual(
            ir_signals.resolve_stage(self.entries, "002-InstCombinePass.ll")["index"], 2)

    def test_a_repeated_pass_refuses_and_lists_the_candidates(self):
        """A pass that runs more than once is the normal case. Choosing one
        would produce a real census of a stage nobody asked for."""
        with self.assertRaises(LookupError) as caught:
            ir_signals.resolve_stage(self.entries, "InstCombinePass")
        self.assertIn("[1, 2]", str(caught.exception))

    def test_an_out_of_range_index_says_what_the_range_is(self):
        with self.assertRaises(LookupError) as caught:
            ir_signals.resolve_stage(self.entries, "9")
        self.assertIn("0..2", str(caught.exception))

    def test_an_unmatched_selector_raises(self):
        with self.assertRaises(LookupError):
            ir_signals.resolve_stage(self.entries, "NoSuchPass")


class DeltaTest(unittest.TestCase):
    def test_added_and_removed_are_split_not_netted(self):
        """A pass that adds two loads and removes two stores nets to zero, and
        the netted number hides both halves of what it did."""
        before = ir_signals.parse_mir(MIR_PRE_RA)
        after = ir_signals.parse_mir(MIR_POST_RA)
        change = ir_signals.delta(before, after)
        self.assertEqual(change["added"]["families"].get("sync"), 1)  # S_WAITCNT appears
        self.assertNotIn("sync", change.get("removed", {}).get("families", {}))

    def test_a_language_change_is_flagged(self):
        change = ir_signals.delta(ir_signals.parse_ll(LL), ir_signals.parse_mir(MIR_PRE_RA))
        self.assertTrue(change["crosses_ir_boundary"])

    def test_two_stages_in_one_language_are_not_flagged(self):
        change = ir_signals.delta(ir_signals.parse_mir(MIR_PRE_RA),
                                  ir_signals.parse_mir(MIR_POST_RA))
        self.assertFalse(change["crosses_ir_boundary"])


class RankingTest(unittest.TestCase):
    def test_the_language_boundary_ranks_behind_real_pass_effects(self):
        """Instruction selection rewrites every instruction, so by raw magnitude
        it always wins -- and it is never the answer to "which pass changed
        this"."""
        boundary = {"crosses_ir_boundary": True, "magnitude": 999, "to_index": 5}
        real = {"crosses_ir_boundary": False, "magnitude": 3, "to_index": 9}
        self.assertEqual(sorted([boundary, real], key=ir_signals.rank_key)[0], real)


class ObservationTest(unittest.TestCase):
    def test_no_width_drop_is_claimed_across_the_language_boundary(self):
        """An abstract `load <4 x float>` and a `GLOBAL_LOAD_DWORD` are counted by
        different readers. Comparing them would report a narrowing at whichever
        pass happens to sit on the boundary."""
        censuses = [ir_signals.parse_ll(LL), ir_signals.parse_mir(MIR_PRE_RA)]
        for index, census in enumerate(censuses):
            census["index"], census["pass_id"] = index, ("front", "amdgpu-isel")[index]
        notes = ir_signals.observations(censuses, ir_signals.transitions(censuses))
        self.assertTrue(any("IR language changes" in n for n in notes))
        self.assertFalse(any("widest load falls" in n for n in notes))

    def test_a_repeated_width_transition_is_folded_with_a_count(self):
        """`widest load` is a maximum over the whole function, so a pass that
        clones a block moves it transiently and the same transition reappears
        several times. Seen four times on the real gemm, where it crowded out
        every other line of the receipt."""
        wide = "  %1 = load <4 x float>, ptr %p, align 16\n"
        narrow = "  %1 = load float, ptr %p, align 4\n"
        bodies = [wide, narrow, wide, narrow, wide, narrow]
        censuses = []
        for index, body in enumerate(bodies):
            census = ir_signals.parse_ll(body)
            census["index"], census["pass_id"] = index, f"P{index}"
            censuses.append(census)
        notes = ir_signals.observations(censuses, ir_signals.transitions(censuses))
        drops = [n for n in notes if "widest load falls" in n]
        self.assertEqual(len(drops), 1)
        self.assertIn("3 stages, first at 1", drops[0])

    def test_the_observation_list_is_bounded_and_says_what_it_dropped(self):
        censuses = []
        for index in range(ir_signals.MAX_OBSERVATIONS + 20):
            body = "  call void @llvm.amdgcn.s.barrier()\n" * (index + 1)
            census = ir_signals.parse_ll(body)
            census["index"], census["pass_id"] = index, f"P{index}"
            censuses.append(census)
        notes = ir_signals.observations(censuses, ir_signals.transitions(censuses))
        self.assertEqual(len(notes), ir_signals.MAX_OBSERVATIONS + 1)
        self.assertIn("find-changes", notes[-1])

    def test_a_sync_increase_is_attributed_to_the_pass_that_made_it(self):
        censuses = [ir_signals.parse_mir(MIR_PRE_RA), ir_signals.parse_mir(MIR_POST_RA)]
        for index, census in enumerate(censuses):
            census["index"] = index
            census["pass_id"] = ("pre", "si-insert-waitcnts")[index]
        notes = ir_signals.observations(censuses, ir_signals.transitions(censuses))
        self.assertTrue(any("sync ops +1" in n and "si-insert-waitcnts" in n for n in notes))


class CliTest(unittest.TestCase):
    def _run(self, *args):
        proc = subprocess.run([sys.executable, str(HERE / "ir_signals.py"), *args],
                              capture_output=True, text=True)
        return proc.returncode, proc.stdout

    def test_find_changes_names_the_pass_and_the_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = make_archive(Path(tmp), [
                ("pre", ".mir", MIR_PRE_RA),
                ("si-insert-waitcnts", ".mir", MIR_POST_RA),
            ])
            code, out = self._run("find-changes", "--archive", str(archive))
            self.assertEqual(code, 0)
            payload = json.loads(out)
            change = payload["changes"][0]
            self.assertEqual(change["pass_id"], "si-insert-waitcnts")
            self.assertEqual(change["added"]["families"]["sync"], 1)

    def test_receipts_are_written_with_sorted_keys(self):
        """`test_script_conventions.py` enforces this at the source level; this
        checks the emitted bytes, which is what actually gets diffed."""
        with tempfile.TemporaryDirectory() as tmp:
            archive = make_archive(Path(tmp), [("a", ".ll", LL), ("b", ".ll", LL)])
            _code, out = self._run("performance-signals", "--archive", str(archive))
            payload = json.loads(out)
            self.assertEqual(out.strip(),
                             json.dumps(payload, sort_keys=True, indent=2).strip())

    def test_an_archive_with_no_stages_is_a_hole(self):
        """Not an empty success. A capture that produced nothing and a kernel
        whose passes changed nothing must not exit the same way."""
        with tempfile.TemporaryDirectory() as tmp:
            archive = make_archive(Path(tmp), [])
            code, out = self._run("list-stages", "--archive", str(archive))
            self.assertEqual(code, ir_signals.EXIT_HOLE)
            self.assertIn("HOLE", json.loads(out)["note"])

    def test_provenance_travels_with_every_receipt(self):
        """A reader must never have to go find out separately whether this
        trajectory was tied to the measured binary."""
        with tempfile.TemporaryDirectory() as tmp:
            archive = make_archive(
                Path(tmp), [("a", ".ll", LL)],
                provenance={"checked": True, "ir_binary_equals_measured": False,
                            "reason": "drifted"})
            _code, out = self._run("stage-summary", "--archive", str(archive), "--stage", "0")
            payload = json.loads(out)
            self.assertFalse(payload["provenance"]["ir_binary_equals_measured"])

    def test_an_ambiguous_selector_exits_nonzero_with_the_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = make_archive(Path(tmp), [("dup", ".ll", LL), ("dup", ".ll", LL)])
            code, out = self._run("stage-summary", "--archive", str(archive), "--stage", "dup")
            self.assertEqual(code, ir_signals.EXIT_ERROR)
            self.assertIn("[0, 1]", json.loads(out)["error"])

    def test_a_missing_archive_is_an_error_not_a_traceback(self):
        code, out = self._run("list-stages", "--archive", "/nonexistent/archive")
        self.assertEqual(code, ir_signals.EXIT_ERROR)
        self.assertIn("manifest.json", json.loads(out)["error"])


if __name__ == "__main__":
    unittest.main()
