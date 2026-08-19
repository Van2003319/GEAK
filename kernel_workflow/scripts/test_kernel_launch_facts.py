#!/usr/bin/env python3
"""Tests for `kernel_launch_facts.py`.

The load-bearing test here is `test_the_lds_figure_tracks_the_source`. This tool
exists because remembered numbers went stale (144); a version of it that quietly
reproduced its own constants would be the disease wearing the cure. So the
central test mutates the source and requires the output to move with it.

Second in weight is `test_the_two_kernels_get_different_footprints`. Both
kernels in this family declare `as`, `bs` and `out`, and a file-wide scan
collides them into one plausible, tidy, wrong table -- which already happened
once this round with the codegen metadata parser and was caught by hand
arithmetic, not by review.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import kernel_launch_facts as KLF  # noqa: E402

LIVE = REPO_ROOT / "examples" / "tasks" / "dense_bf16_gemm_fused" / "src" / "custom_gemm.hip"

# Hand-computed from the declarations, independently of the tool:
#   tall:    as[32][68]*2 + bs[4][16][68]*2 + out[4][32][16]*4
#          = 4352 + 8704 + 8192 = 21248
#   generic: as[16][68]*2 + bs[4][16][68]*2 + out[4][16][16]*4
#          = 2176 + 8704 + 4096 = 14976
TALL_LDS = 32 * 68 * 2 + 4 * 16 * 68 * 2 + 4 * 32 * 16 * 4
GENERIC_LDS = 16 * 68 * 2 + 4 * 16 * 68 * 2 + 4 * 16 * 16 * 4


def source(body: str) -> Path:
    path = Path(tempfile.mkdtemp(prefix="klf_")) / "k.hip"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def copy_of_live() -> Path:
    path = Path(tempfile.mkdtemp(prefix="klf_live_")) / "custom_gemm.hip"
    shutil.copy(LIVE, path)
    return path


class LiveSightingTest(unittest.TestCase):
    """(142). Proven against the real kernel before it is trusted about any."""

    def test_the_live_kernel_is_still_there(self):
        self.assertTrue(LIVE.is_file(), f"{LIVE} is gone; this file is now vacuous")

    def test_it_derives_the_hand_computed_footprints(self):
        src = KLF.read_source(LIVE)
        self.assertEqual(TALL_LDS, src["lds"]["tall"])
        self.assertEqual(GENERIC_LDS, src["lds"]["generic"])
        self.assertEqual(21248, TALL_LDS, "the hand arithmetic in this file drifted")
        self.assertEqual(14976, GENERIC_LDS, "the hand arithmetic in this file drifted")

    def test_the_two_kernels_get_different_footprints(self):
        """Both declare `as`, `bs`, `out`. Equal footprints would mean the split
        collapsed and one kernel's figures are being reported for both."""
        src = KLF.read_source(LIVE)
        self.assertNotEqual(src["lds"]["tall"], src["lds"]["generic"])

    def test_launch_bounds_does_not_become_the_kernel_name(self):
        """The first regex written for this named every kernel
        `__launch_bounds__`, which failed loudly. It could just as easily have
        matched something plausible, so the trap is pinned here."""
        names = set(KLF.kernel_regions(LIVE.read_text(encoding="utf-8")))
        self.assertNotIn("__launch_bounds__", names)
        self.assertTrue(any("tall" in n for n in names), names)
        self.assertTrue(any("generic" in n for n in names), names)

    def test_the_constants_come_out_as_the_source_states_them(self):
        c = KLF.read_source(LIVE)["consts"]
        self.assertEqual(16, c["kTile"])
        self.assertEqual(4, c["kWaves"])
        self.assertEqual(32, c["kTallM"], "kTallM = 2 * kTile must resolve, not be skipped")
        self.assertEqual(64, c["kStageK"])


class DerivedNotRememberedTest(unittest.TestCase):
    """The whole reason this file exists (144)."""

    def test_the_lds_figure_tracks_the_source(self):
        path = copy_of_live()
        text = path.read_text(encoding="utf-8")
        out = text.replace("constexpr int kLdsStride = 68;",
                           "constexpr int kLdsStride = 72;")
        self.assertNotEqual(text, out, "kLdsStride is no longer declared this way; "
                                       "this test's mutation is a no-op")
        path.write_text(out, encoding="utf-8")
        src = KLF.read_source(path)
        self.assertEqual(32 * 72 * 2 + 4 * 16 * 72 * 2 + 4 * 32 * 16 * 4,
                         src["lds"]["tall"],
                         "the footprint did not move with the source; this tool is "
                         "reciting a remembered number")
        self.assertNotEqual(TALL_LDS, src["lds"]["tall"])

    def test_the_grid_tracks_the_tile_constants(self):
        path = copy_of_live()
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("constexpr int kWaves = 4;",
                                     "constexpr int kWaves = 2;"), encoding="utf-8")
        before = {r["case"]: r["ctas"] for r in KLF.rows(KLF.read_source(LIVE))}
        after = {r["case"]: r["ctas"] for r in KLF.rows(KLF.read_source(path))}
        self.assertNotEqual(before, after, "halving kWaves changed no grid")
        # Half the columns per workgroup means twice the workgroups.
        self.assertEqual(2 * before["prefill_m2048_square"],
                         after["prefill_m2048_square"])

    def test_occupancy_tracks_the_footprint(self):
        """A bigger tile admits fewer CTAs per CU. If the slot count is frozen,
        every occupancy figure downstream is fiction."""
        path = copy_of_live()
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("constexpr int kLdsStride = 68;",
                                     "constexpr int kLdsStride = 200;"),
                        encoding="utf-8")
        tall_rows = [r for r in KLF.rows(KLF.read_source(path)) if r["kernel"] == "tall"]
        self.assertTrue(tall_rows)
        for r in tall_rows:
            with self.subTest(case=r["case"]):
                self.assertEqual(1, r["ctas_per_cu_cap"])


class TemplatedSourceTest(unittest.TestCase):
    """The round-17 D1 variant templates both kernels. It must still parse --
    an UNREADABLE there would silently push the candidate off this analysis."""

    D1 = (REPO_ROOT / "exp" / "round17_d1_awide_20260816_233414" / "src"
          / "custom_gemm_awide.hip")

    def test_the_d1_variant_parses_and_matches_the_baseline_footprint(self):
        if not self.D1.is_file():
            self.skipTest(f"{self.D1} not present")
        src = KLF.read_source(self.D1)
        self.assertEqual(TALL_LDS, src["lds"]["tall"],
                         "D1 was supposed to leave LDS untouched; if this fires, "
                         "occupancy and the rounds arithmetic both moved")
        self.assertEqual(GENERIC_LDS, src["lds"]["generic"])


class FailsLoudTest(unittest.TestCase):
    """(141). What it cannot read must not come back as a number."""

    def test_an_unknown_element_type_is_unreadable(self):
        path = source("""
            constexpr int kTile = 16;
            constexpr int kWaves = 4;
            constexpr int kTallM = 32;
            constexpr int kStageK = 64;
            __global__ void tall_k() { __shared__ Mystery as[32][68]; }
            __global__ void generic_k() { __shared__ Bf16 as[16][68]; }
        """)
        with self.assertRaises(KLF.Unreadable):
            KLF.read_source(path)
        self.assertEqual(2, KLF.main([str(path)]))

    def test_an_unresolvable_dimension_is_unreadable(self):
        path = source("""
            constexpr int kTile = 16;
            constexpr int kWaves = 4;
            constexpr int kTallM = 32;
            constexpr int kStageK = 64;
            __global__ void tall_k() { __shared__ Bf16 as[32][kFromAHeader]; }
            __global__ void generic_k() { __shared__ Bf16 as[16][68]; }
        """)
        self.assertEqual(2, KLF.main([str(path)]))

    def test_a_missing_geometry_constant_is_unreadable(self):
        path = source("""
            constexpr int kTile = 16;
            constexpr int kWaves = 4;
            __global__ void tall_k() { __shared__ Bf16 as[32][68]; }
            __global__ void generic_k() { __shared__ Bf16 as[16][68]; }
        """)
        with self.assertRaisesRegex(KLF.Unreadable, "kTallM"):
            KLF.read_source(path)

    def test_a_kernel_with_no_shared_memory_is_unreadable_not_zero(self):
        """Zero LDS would divide into an enormous slot count and produce the
        most optimistic table in the file. Refusing is the only safe answer."""
        path = source("""
            constexpr int kTile = 16;
            constexpr int kWaves = 4;
            constexpr int kTallM = 32;
            constexpr int kStageK = 64;
            __global__ void tall_k() { int x = 0; }
            __global__ void generic_k() { __shared__ Bf16 as[16][68]; }
        """)
        with self.assertRaisesRegex(KLF.Unreadable, "no __shared__"):
            KLF.read_source(path)

    def test_an_unexpected_kernel_count_is_unreadable(self):
        path = source("""
            constexpr int kTile = 16;
            constexpr int kWaves = 4;
            constexpr int kTallM = 32;
            constexpr int kStageK = 64;
            __global__ void only_one() { __shared__ Bf16 as[16][68]; }
        """)
        with self.assertRaisesRegex(KLF.Unreadable, "tall="):
            KLF.read_source(path)


class TableShapeTest(unittest.TestCase):
    def test_every_harness_case_appears_exactly_once(self):
        table = KLF.rows(KLF.read_source(LIVE))
        self.assertEqual(11, len(table))
        self.assertEqual(len({r["case"] for r in table}), len(table))

    def test_the_kernel_split_follows_ktallm(self):
        for r in KLF.rows(KLF.read_source(LIVE)):
            with self.subTest(case=r["case"]):
                self.assertEqual("generic" if r["M"] < 32 else "tall", r["kernel"])

    def test_tile_rows_useful_is_a_percentage_of_a_full_tile(self):
        table = {r["case"]: r for r in KLF.rows(KLF.read_source(LIVE))}
        # M=2 in a 16-row generic tile.
        self.assertAlmostEqual(12.5, table["decode_m2_square"]["tile_rows_useful_pct"])
        # M=16 fills a generic tile exactly.
        self.assertAlmostEqual(100.0, table["decode_m16_square"]["tile_rows_useful_pct"])

    def test_the_warnings_are_in_the_tool_not_only_in_the_notes(self):
        """Both misreadings this table invites have already been made once. The
        warning has to travel with the output, not live in a progress file."""
        text = (HERE / "kernel_launch_facts.py").read_text(encoding="utf-8")
        self.assertIn("is NOT the ROUNDS LAW", text)
        self.assertIn("NOT recoverable time", text)


class DefinedIsNotLaunchedTest(unittest.TestCase):
    """The load-bearing tests here. (142): a checker must be shown catching the
    real thing in real data. The real thing is the v98 ship point, which carries
    `tall` and `generic` verbatim from the seed and launches NEITHER -- it
    dispatches `tiled_bf16_gemm_kernel` through a slice planner. Before the
    launch-site check, this tool produced a full, plausible, authoritative grid
    and occupancy table for v98 describing code that cannot execute, and a
    tail-quantisation experiment was very nearly pre-registered on it.
    """

    V98 = REPO_ROOT / "exp" / "v98_base_20260817_epochR" / "src" / "custom_gemm.hip"

    def test_the_v98_ship_point_is_refused_not_described(self):
        if not self.V98.is_file():
            self.skipTest(f"{self.V98} absent")
        with self.assertRaises(KLF.Unreadable) as cm:
            KLF.read_source(self.V98)
        self.assertIn("never launched", str(cm.exception))

    def test_the_refusal_names_the_kernels_that_ARE_launched(self):
        # A refusal that does not say what the file actually runs sends the
        # reader back to grep. Both real kernels must be named.
        if not self.V98.is_file():
            self.skipTest(f"{self.V98} absent")
        with self.assertRaises(KLF.Unreadable) as cm:
            KLF.read_source(self.V98)
        self.assertIn("tiled_bf16_gemm_kernel", str(cm.exception))
        self.assertIn("splitk_reduce_kernel", str(cm.exception))

    def test_the_seed_still_reads_because_it_really_does_launch_them(self):
        # The check must not fire on its own intended input.
        self.assertTrue(KLF.rows(KLF.read_source(LIVE)))

    def test_a_templated_multiline_launch_is_recognised(self):
        # v98's real launch puts the name on the previous line behind template
        # arguments that themselves contain `<`. A plain `(\w+)\s*<<<` regex
        # misses it and the refusal then under-reports what the file launches.
        text = """
            tiled_bf16_gemm_kernel<WAVES_M, WAVES_N, (A < B)>
                <<<grid, 256, 0, stream>>>(m, n, k);
        """
        self.assertEqual({"tiled_bf16_gemm_kernel"}, KLF.launch_targets(text))

    def test_both_hipify_spellings_are_recognised(self):
        plain = "foo_kernel<<<g, b, 0, s>>>(x);"
        hipify = "hipLaunchKernelGGL(( foo_kernel), dim3(g), dim3(b), 0, s, x);"
        self.assertEqual({"foo_kernel"}, KLF.launch_targets(plain))
        self.assertEqual({"foo_kernel"}, KLF.launch_targets(hipify))

    def test_a_file_with_no_recognised_launch_is_not_condemned(self):
        # Fail-open ONLY here, deliberately: if the launch parser stops
        # recognising anything, the tool must not start declaring every file
        # dead. "Cannot tell" and "not launched" are different verdicts.
        self.assertEqual(set(), KLF.launch_targets("int main() { return 0; }"))
        text = LIVE.read_text(encoding="utf-8").replace("<<<", "/*x*/")
        import re as _re
        text = _re.sub(r"hipLaunchKernelGGL", "nope", text)
        self.assertEqual(set(), KLF.launch_targets(text))


if __name__ == "__main__":
    unittest.main(verbosity=2)
