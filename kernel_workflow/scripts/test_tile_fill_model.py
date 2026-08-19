#!/usr/bin/env python3
"""Tests for the machine-fill model.

This model's whole value is that it reproduces the SHIPPED dispatcher's
arithmetic. A copy that has silently drifted from custom_gemm.hip would predict
a launch nobody performs, so the first test here reads the real source and
compares every copied constant against it.
"""

import re
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tile_fill_model as F  # noqa: E402

SRC = Path("/home/yxh/GEAK/exp/state_greedy_coldstart_20260817/best/src/"
           "custom_gemm.hip")


class ConstantSyncTest(unittest.TestCase):
    """The copies must still equal the shipped ones."""

    @classmethod
    def setUpClass(cls):
        if not SRC.exists():
            raise unittest.SkipTest(f"{SRC} not present")
        cls.text = SRC.read_text()

    def _cpp(self, name: str) -> int:
        m = re.search(rf"\b{name}\s*=\s*([^;]+);", self.text)
        self.assertIsNotNone(m, f"{name} not found in {SRC}")
        expr = m.group(1).strip()
        expr = expr.replace("ull", "").replace("LL", "").replace("ll", "")
        return int(eval(expr, {"__builtins__": {}}))   # noqa: S307 -- literals

    def test_every_copied_constant_matches_the_shipped_source(self):
        for py, cpp in (
            (F.K_FILL_TARGET, "kFillTarget"),
            (F.K_SPLIT_FIXUP_BYTES, "kSplitFixupBytes"),
            (F.K_MAX_SPLITS, "kMaxSplits"),
            (F.K_SPLIT_TILE_GATE, "kSplitTileGate"),
            (F.K_NARROW_TILE_GATE, "kNarrowTileGate"),
            (F.K_WIDE_TILE_GATE, "kWideTileGate"),
            (F.K_WORKSPACE_BYTES, "kWorkspaceBytes"),
        ):
            self.assertEqual(py, self._cpp(cpp), cpp)


class SplitsTest(unittest.TestCase):
    def test_split_k_is_off_above_the_tile_gate(self):
        # prefill_m2048_square: 512 tiles > 304, so no split-K refills it.
        self.assertEqual(F.splits_for(2048, 4096, 4096, 512, 32), 1)

    def test_the_fixup_budget_binds_on_the_largest_outputs(self):
        # m1024_down: 16 MiB of fp32 output against a 48 MiB budget -> at most
        # 3 slices. m2048_square doubles the output to 32 MiB and gets 1, i.e.
        # none. The cap tightens as the output grows, which is why split-K
        # cannot refill the machine on precisely the largest routes.
        self.assertEqual(F.K_SPLIT_FIXUP_BYTES // (1024 * 4096 * 4), 3)
        self.assertEqual(F.K_SPLIT_FIXUP_BYTES // (2048 * 4096 * 4), 1)

    def test_splits_never_drop_below_one(self):
        for m, n, k in F.ROUTES.values():
            for bn in (64, 128, 256):
                tiles = -(-n // bn) * -(-m // 128)
                self.assertGreaterEqual(F.splits_for(m, n, k, tiles, 32), 1)


class FillTest(unittest.TestCase):
    def test_bigger_tiles_leave_cus_with_no_work_at_all(self):
        # The finding that refuted the tile-enlargement direction: 256x256 on
        # m2048_square produces 128 workgroups for 304 CUs. This is not
        # "under-occupied", it is 176 CUs that never receive a block.
        f = F.fill(2048, 4096, 4096, 256, 256, 32, 1)
        self.assertEqual(f["wgs"], 128)
        self.assertLess(f["wgs"], F.CUS)

    def test_the_shipped_tile_is_itself_under_filled_on_m2048(self):
        # 512 workgroups against a 912 capacity. This is what sent the search
        # toward SMALLER tiles rather than larger ones.
        f = F.fill(2048, 4096, 4096, 128, 128, 32, 3)
        self.assertEqual(f["wgs"], 512)
        self.assertEqual(f["capacity"], 912)
        self.assertLess(f["tail_utilisation"], 0.6)

    def test_bn64_fills_m2048_far_better_in_a_single_wave(self):
        # The round-9 hypothesis, stated as arithmetic: 1024 workgroups into a
        # 1216 capacity, one wave, 84% full.
        f = F.fill(2048, 4096, 4096, 128, 64, 32, 4)
        self.assertEqual(f["wgs"], 1024)
        self.assertEqual(f["waves"], 1)
        self.assertGreater(f["tail_utilisation"], 0.8)

    def test_bn64_does_NOT_help_m1024_down(self):
        # Guards against over-generalising the hypothesis to both routes. The
        # dispatcher already measured BN=64 as a 12% loss here, and the fill
        # model agrees: split-K has already filled it to 84%.
        shipped = F.fill(1024, 4096, 11008, 128, 128, 32, 3)
        narrow = F.fill(1024, 4096, 11008, 128, 64, 32, 4)
        self.assertGreater(shipped["tail_utilisation"],
                           narrow["tail_utilisation"])

    def test_wave_count_and_tail_are_self_consistent(self):
        # The invariant rather than one hand-picked shape: the launch needs
        # exactly enough waves to hold it, the last one is the only partial
        # one, and the reported tail is that fraction.
        for m, n, k in F.ROUTES.values():
            for _label, bm, bn, bk, cta in F.CANDIDATES:
                f = F.fill(m, n, k, bm, bn, bk, cta)
                cap, wgs, waves = f["capacity"], f["wgs"], f["waves"]
                self.assertLessEqual(wgs, waves * cap)
                self.assertGreater(wgs, (waves - 1) * cap)
                self.assertAlmostEqual(f["tail_utilisation"],
                                       wgs / (waves * cap))
                self.assertEqual(f["cus_idle_in_tail"], waves * cap - wgs)


class GateChangeTest(unittest.TestCase):
    """Raising kWideTileGate must move exactly one case."""

    CASES = {
        "decode_m2_square": (2, 4096, 4096),
        "decode_m8_up": (8, 11008, 4096),
        "decode_m16_square": (16, 4096, 4096),
        "decode_m32_down": (32, 4096, 11008),
        "decode_m64_square": (64, 8192, 8192),
        "decode_m96_up": (96, 11008, 4096),
        "prefill_m128_square": (128, 4096, 4096),
        "prefill_m256_down": (256, 4096, 11008),
        "prefill_m512_up": (512, 11008, 4096),
        "prefill_m1024_down": (1024, 4096, 11008),
        "prefill_m2048_square": (2048, 4096, 4096),
    }

    @staticmethod
    def _bn(m, n, gate_wide):
        bm = 128 if m >= 128 else (64 if m >= 64 else 32)
        t128 = -(-n // 128) * -(-m // bm)
        narrow = (t128 <= F.K_NARROW_TILE_GATE
                  or (t128 > F.K_SPLIT_TILE_GATE and t128 <= gate_wide))
        return 64 if narrow else 128

    def test_only_m2048_square_changes_when_the_gate_widens(self):
        moved = [c for c, (m, n, _k) in self.CASES.items()
                 if self._bn(m, n, 400) != self._bn(m, n, 512)]
        self.assertEqual(moved, ["prefill_m2048_square"])

    def test_it_stays_the_only_one_at_a_much_wider_gate(self):
        # If a larger widening moved more cases, the change would not be
        # surgical and the round would confound two effects.
        moved = [c for c, (m, n, _k) in self.CASES.items()
                 if self._bn(m, n, 400) != self._bn(m, n, 1024)]
        self.assertEqual(moved, ["prefill_m2048_square"])

    def test_m2048_square_sits_just_outside_the_current_bound(self):
        # 512 tiles against a bound of 400. The route meets every stated
        # condition of the BN=64 rule -- split-K off, 1-2 workgroups per CU --
        # and is excluded only by that upper number.
        self.assertEqual(self._bn(2048, 4096, 400), 128)
        self.assertGreater(512, F.K_WIDE_TILE_GATE)
        self.assertGreater(512, F.K_SPLIT_TILE_GATE)


if __name__ == "__main__":
    unittest.main()
