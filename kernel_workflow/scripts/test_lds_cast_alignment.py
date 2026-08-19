#!/usr/bin/env python3
"""Tests for `lds_cast_alignment.py`.

The load-bearing test in this file is `test_it_sees_the_live_instance`. Finding
142: a backstop scanner has to be shown catching the thing it guards, in the
real source, before any claim that a file is clean means anything. The shipped
fused-GEMM kernel contains a genuine instance -- `Bf16[4][16][68]` cast to
`uint4*`, 136-byte rows -- so the scanner cannot go blind without that test
going red. A synthetic fixture would not have that property: it would keep
passing after the regex stopped matching anything anyone writes.
"""
from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import lds_cast_alignment as LCA  # noqa: E402

LIVE = REPO_ROOT / "examples" / "tasks" / "dense_bf16_gemm_fused" / "src" / "custom_gemm.hip"


def source(body: str) -> Path:
    path = Path(tempfile.mkdtemp(prefix="lca_")) / "k.hip"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


class LiveSightingTest(unittest.TestCase):
    """(142). Proven against real code, before it is trusted about any code."""

    def test_the_live_kernel_is_still_there(self):
        """(55). If the task moves, every assertion below starts passing by
        reading a file that does not exist -- so say so here instead."""
        self.assertTrue(LIVE.is_file(), f"{LIVE} is gone; this file is now vacuous")

    def test_it_sees_the_live_instance(self):
        findings, unknown = LCA.scan(LIVE)
        self.assertEqual([], unknown, f"the live kernel did not fully parse: {unknown}")
        self.assertTrue(findings, "the shipped kernel casts Bf16[4][16][68] to uint4* "
                                  "and the scanner found nothing; it is blind")
        for f in findings:
            with self.subTest(line=f["line"]):
                self.assertEqual("bs", f["array"])
                self.assertEqual(136, f["stride_bytes"])
                self.assertEqual(8, f["misaligned_by"])
                self.assertEqual(16, f["cast_width"])

    def test_the_live_kernel_exits_one_and_not_two(self):
        """Two different non-zero exits mean two different things, and only one
        of them is a finding. An UNPARSEABLE verdict on the one file whose
        answer we know by hand would mean the type tables are wrong -- and then
        every clean exit anywhere else is a guess wearing a pass."""
        self.assertEqual(1, LCA.main([str(LIVE)]))


class AlignedIsCleanTest(unittest.TestCase):
    def test_a_stride_that_is_a_multiple_of_the_cast_is_not_flagged(self):
        path = source("""
            constexpr int kStride = 72;
            __global__ void k() {
              __shared__ __align__(16) Bf16 as[32][kStride];
              *reinterpret_cast<uint4*>(&as[r][c]) = v;
            }
        """)
        self.assertEqual(([], []), LCA.scan(path), "72 * 2 = 144, a multiple of 16")

    def test_a_narrower_cast_on_the_same_array_is_fine(self):
        """The fix applied in round 17: 136 % 8 == 0, so uint2 is honest where
        uint4 is not. A checker that flagged the array rather than the cast
        would forbid the correct code along with the broken code."""
        path = source("""
            constexpr int kStride = 68;
            __global__ void k() {
              __shared__ __align__(16) Bf16 as[32][kStride];
              *reinterpret_cast<uint2*>(&as[r][c]) = v;
            }
        """)
        self.assertEqual(([], []), LCA.scan(path))

    def test_the_flag_appears_when_the_cast_widens(self):
        """The pair to the test above: same array, same indices, only the cast
        width changes. This is the entire property, isolated."""
        path = source("""
            constexpr int kStride = 68;
            __global__ void k() {
              __shared__ __align__(16) Bf16 as[32][kStride];
              *reinterpret_cast<uint4*>(&as[r][c]) = v;
            }
        """)
        findings, unknown = LCA.scan(path)
        self.assertEqual([], unknown)
        self.assertEqual(1, len(findings))
        self.assertEqual(136, findings[0]["stride_bytes"])


class InnermostIsNotCheckedTest(unittest.TestCase):
    def test_a_one_dimensional_array_is_never_flagged(self):
        """There is no outer stride, so there is nothing decidable. Reporting it
        clean is correct; reporting it checked would not be, which is what the
        docstring and the clean-exit message both say out loud."""
        path = source("""
            __global__ void k() {
              __shared__ Bf16 as[512];
              *reinterpret_cast<uint4*>(&as[c]) = v;
            }
        """)
        self.assertEqual(([], []), LCA.scan(path))

    def test_the_clean_message_does_not_claim_full_coverage(self):
        src = (HERE / "lds_cast_alignment.py").read_text(encoding="utf-8")
        self.assertIn("innermost index granularity is not decidable here", src)


class RefusesToGuessTest(unittest.TestCase):
    """(141). Input the tool cannot read must not come back as a pass."""

    def test_an_unknown_element_type_is_reported_not_skipped(self):
        path = source("""
            __global__ void k() {
              __shared__ MysteryType as[32][68];
              *reinterpret_cast<uint4*>(&as[r][c]) = v;
            }
        """)
        findings, unknown = LCA.scan(path)
        self.assertEqual([], findings)
        self.assertEqual(1, len(unknown))
        self.assertIn("ELEM_SIZE", unknown[0]["reason"])
        self.assertEqual(2, LCA.main([str(path)]), "an unreadable file exited 0")

    def test_an_unresolvable_dimension_is_reported_not_skipped(self):
        path = source("""
            __global__ void k() {
              __shared__ Bf16 as[32][kStrideFromAnotherHeader];
              *reinterpret_cast<uint4*>(&as[r][c]) = v;
            }
        """)
        findings, unknown = LCA.scan(path)
        self.assertEqual([], findings)
        self.assertEqual(1, len(unknown))
        self.assertEqual(2, LCA.main([str(path)]))

    def test_an_unknown_cast_type_is_reported_not_skipped(self):
        path = source("""
            __global__ void k() {
              __shared__ Bf16 as[32][68];
              *reinterpret_cast<my_vec_t*>(&as[r][c]) = v;
            }
        """)
        findings, unknown = LCA.scan(path)
        self.assertEqual([], findings)
        self.assertEqual(1, len(unknown))
        self.assertIn("CAST_WIDTH", unknown[0]["reason"])

    def test_a_cast_on_something_this_file_does_not_declare_is_not_a_finding(self):
        """A pointer that came in as a parameter has no declaration to reason
        from. Silence is the honest answer; inventing a stride would not be."""
        path = source("""
            __global__ void k(const Bf16* a) {
              *reinterpret_cast<const uint4*>(&a[i]) = v;
            }
        """)
        self.assertEqual(([], []), LCA.scan(path))


class StrideArithmeticTest(unittest.TestCase):
    def test_strides_are_outermost_first_and_end_at_the_element(self):
        self.assertEqual([2176, 136, 2], LCA.strides([4, 16, 68], 2))

    def test_a_single_dimension_has_only_the_element_stride(self):
        self.assertEqual([4], LCA.strides([512], 4))

    def test_constants_resolve_against_earlier_constants(self):
        known = LCA.constants("constexpr int kTile = 16;\n"
                              "constexpr int kStageK = 4 * kTile;\n")
        self.assertEqual({"kTile": 16, "kStageK": 64}, known)

    def test_a_constant_it_cannot_evaluate_is_left_out(self):
        """Left out, so the dimension using it becomes Unresolved and gets
        reported -- rather than resolving to something plausible."""
        self.assertEqual({}, LCA.constants("constexpr int k = some_fn(3);\n"))


class GateModeTest(unittest.TestCase):
    """Gate mode has to pass the baseline and fail the mutation of it.

    Both halves matter. A gate that fails the unmodified shipped kernel is
    permanently red and gets bypassed; a gate that passes the widened A tile is
    the gate not existing. The mutation used here is not invented -- it is the
    round-17 D1 edit, reproduced by substitution on the live source, and it is
    the bug that reached a compiled object before anyone caught it.
    """

    def widened(self) -> Path:
        """The shipped kernel with the A-tile staging cast widened to uint4."""
        text = LIVE.read_text(encoding="utf-8")
        out = text.replace("reinterpret_cast<unsigned*>(&as[",
                           "reinterpret_cast<uint4*>(&as[")
        self.assertNotEqual(text, out, "the A-tile cast no longer looks like this; "
                                       "the mutation this test applies is a no-op")
        path = Path(tempfile.mkdtemp(prefix="lca_mut_")) / "custom_gemm.hip"
        path.write_text(out, encoding="utf-8")
        return path

    def test_the_baseline_against_itself_is_clean(self):
        self.assertEqual(0, LCA.main([str(LIVE), "--baseline", str(LIVE)]),
                         "the unmodified shipped kernel fails its own gate; this "
                         "gate would be red on every candidate and get bypassed")

    def test_the_widened_a_tile_is_a_new_finding(self):
        self.assertEqual(1, LCA.main([str(self.widened()), "--baseline", str(LIVE)]))

    def test_the_new_finding_is_the_a_tile_and_the_b_tile_is_inherited(self):
        findings, _ = LCA.scan(self.widened())
        base, _ = LCA.scan(LIVE)
        known = {LCA.signature(f) for f in base}
        new = [f for f in findings if LCA.signature(f) not in known]
        self.assertTrue(new, "the mutation produced no new signature")
        for f in new:
            with self.subTest(line=f["line"]):
                self.assertEqual("as", f["array"], "the B tile should be inherited, "
                                                   "not reported as introduced")

    def test_signatures_ignore_line_numbers(self):
        """A candidate that only moves code has introduced nothing."""
        text = LIVE.read_text(encoding="utf-8")
        moved = Path(tempfile.mkdtemp(prefix="lca_mv_")) / "custom_gemm.hip"
        moved.write_text("// a new header comment\n" * 40 + text, encoding="utf-8")
        self.assertEqual(0, LCA.main([str(moved), "--baseline", str(LIVE)]))

    def test_an_unreadable_baseline_does_not_yield_a_pass(self):
        """"New" is undecidable when the baseline did not parse. Answering
        `clean` there is a verdict derived from a file nobody could read."""
        murky = source("""
            __global__ void k() { __shared__ MysteryType as[32][68];
              *reinterpret_cast<uint4*>(&as[r][c]) = v; }
        """)
        clean = source("""
            __global__ void k() { __shared__ Bf16 as[32][72];
              *reinterpret_cast<uint4*>(&as[r][c]) = v; }
        """)
        self.assertEqual(2, LCA.main([str(clean), "--baseline", str(murky)]))


class ExitCodeTest(unittest.TestCase):
    def test_clean_source_exits_zero(self):
        path = source("""
            __global__ void k() {
              __shared__ Bf16 as[32][72];
              *reinterpret_cast<uint4*>(&as[r][c]) = v;
            }
        """)
        self.assertEqual(0, LCA.main([str(path)]))

    def test_a_finding_exits_one(self):
        path = source("""
            __global__ void k() {
              __shared__ Bf16 as[32][68];
              *reinterpret_cast<uint4*>(&as[r][c]) = v;
            }
        """)
        self.assertEqual(1, LCA.main([str(path)]))

    def test_unparseable_outranks_a_clean_scan(self):
        """Two files, one clean and one unreadable. Exiting 0 because the first
        one passed is the failure this ordering exists to prevent."""
        clean = source("""
            __global__ void k() { __shared__ Bf16 as[32][72];
              *reinterpret_cast<uint4*>(&as[r][c]) = v; }
        """)
        murky = source("""
            __global__ void k() { __shared__ MysteryType as[32][68];
              *reinterpret_cast<uint4*>(&as[r][c]) = v; }
        """)
        self.assertEqual(2, LCA.main([str(clean), str(murky)]))


class OutermostExtentIsNotLoadBearing(unittest.TestCase):
    """An unresolvable *leading* dimension must not fail the array closed.

    `strides` reads `extents[1:]`, so extent 0 feeds no stride and cannot
    change a verdict. Template-parameterised leading dims are how these
    kernels are actually written (`Bf16 as[TM][kLdsStride]`, `TM = NM*kTile`);
    treating them as unparseable stalled a real candidate on a question the
    checker was never asking.
    """

    def test_template_leading_dim_is_decidable(self):
        path = source("""
            constexpr int kLdsStride = 72;
            template <int NM> __global__ void k() {
              constexpr int TM = NM * 16;
              __shared__ Bf16 as[TM][kLdsStride];
              *reinterpret_cast<uint4*>(&as[r][c]) = v;
            }
        """)
        self.assertEqual(0, LCA.main([str(path)]))

    def test_template_leading_dim_still_reports_a_real_finding(self):
        """Deciding the array is only worth anything if the verdict survives:
        68 * 2 = 136 bytes is still not a multiple of 16."""
        path = source("""
            constexpr int kLdsStride = 68;
            template <int NM> __global__ void k() {
              constexpr int TM = NM * 16;
              __shared__ Bf16 as[TM][kLdsStride];
              *reinterpret_cast<uint4*>(&as[r][c]) = v;
            }
        """)
        self.assertEqual(1, LCA.main([str(path)]))

    def test_unresolvable_inner_dim_still_fails_closed(self):
        """The narrow exemption must stay narrow. An inner unknown does land
        in a stride, so guessing it would be guessing the answer."""
        path = source("""
            constexpr int kLdsStride = 72;
            template <int NM> __global__ void k() {
              __shared__ Bf16 as[4][NM][kLdsStride];
              *reinterpret_cast<uint4*>(&as[w][r][c]) = v;
            }
        """)
        self.assertEqual(2, LCA.main([str(path)]))

    def test_finding_shows_the_symbol_not_none(self):
        path = source("""
            constexpr int kLdsStride = 68;
            template <int NM> __global__ void k() {
              __shared__ Bf16 as[TM][kLdsStride];
              *reinterpret_cast<uint4*>(&as[r][c]) = v;
            }
        """)
        findings, unknown = LCA.scan(Path(path))
        self.assertEqual([], unknown)
        self.assertEqual("Bf16[TM][68]", findings[0]["declared"])


if __name__ == "__main__":
    unittest.main(verbosity=2)


class ByteArenaTest(unittest.TestCase):
    """A raw `char` LDS arena must be READ, not refused unread.

    Regression for the round-1 sighting on dense_bf16_gemm_fused: `char` was
    absent from ELEM_SIZE, so a `__shared__ __align__(16) char smem[N];` arena
    produced findings=[] with four `unparseable` rows and passed=false --
    stopping a candidate before its build, correctness run, or any timing,
    while the scan had in fact decided nothing about it. `char` is 1 byte by
    definition; refusing to resolve it was not caution, it was a false refusal.
    """

    def test_a_char_arena_is_resolved_and_clean(self):
        path = source("""
            __global__ void k() {
              __shared__ __align__(16) char smem[16384];
              *reinterpret_cast<uint4*>(&smem[off]) = v;
            }
        """)
        findings, unknown = LCA.scan(path)
        self.assertEqual([], findings)
        self.assertEqual([], unknown, "a char arena came back unparseable")
        self.assertEqual(0, LCA.main([str(path)]), "a decidable clean file exited nonzero")

    def test_a_one_dimensional_arena_has_no_outer_stride_to_judge(self):
        # strides[:-1] is empty for a 1-D array, so no verdict is reachable
        # here for ANY element type. The gate must therefore not fail on it.
        path = source("""
            __global__ void k() {
              __shared__ char smem[1000];
              *reinterpret_cast<uint4*>(&smem[i]) = v;
            }
        """)
        findings, unknown = LCA.scan(path)
        self.assertEqual(([], []), (findings, unknown))

    def test_a_misaligned_char_backed_2d_view_is_still_caught(self):
        # The fix resolves the size; it must not switch the check off. A 2-D
        # char array with an odd row stride is still a real finding.
        path = source("""
            __global__ void k() {
              __shared__ char as[32][136];
              *reinterpret_cast<uint4*>(&as[r][c]) = v;
            }
        """)
        findings, unknown = LCA.scan(path)
        self.assertEqual([], unknown)
        self.assertEqual(1, len(findings), "a 136-byte row stride passed a uint4 cast")
        # 1 = a real finding, 2 = could not read. Resolving `char` moves this
        # file from the second bucket into the first, which is the whole point.
        self.assertEqual(1, LCA.main([str(path)]))


class VectorTypedefTest(unittest.TestCase):
    """A vector typedef declared in the scanned file is READ, not refused.

    Regression for the round-1 sighting on dense_bf16_gemm_fused: the candidate
    declared `typedef __attribute__((__vector_size__(4 * sizeof(short)))) short
    shortx4_t;` a hundred lines above the casts that used it, and the scan
    refused the whole candidate -- findings [], four unparseable rows, no build,
    no correctness run, no timing. The width was written in the source the tool
    was already reading. This resolves it the same way `constants()` resolves a
    `constexpr` dimension; a typedef it cannot evaluate is still reported.
    """

    def test_vector_size_with_sizeof_product(self):
        self.assertEqual(
            {"shortx4_t": 8},
            LCA.vector_typedefs(
                "typedef __attribute__((__vector_size__(4 * sizeof(short)))) short shortx4_t;"))

    def test_vector_size_with_integer_literal(self):
        self.assertEqual(
            {"v16_t": 16},
            LCA.vector_typedefs("typedef __attribute__((__vector_size__(16))) float v16_t;"))

    def test_ext_vector_type_spelling(self):
        self.assertEqual(
            {"h8_t": 16},
            LCA.vector_typedefs(
                "typedef _Float16 h8_t __attribute__((ext_vector_type(8)));"))

    def test_an_unevaluable_width_stays_unresolved(self):
        # A named constant is not something this may guess at.
        self.assertEqual(
            {}, LCA.vector_typedefs(
                "typedef __attribute__((__vector_size__(kWidth))) short v_t;"))

    def test_an_unknown_base_stays_unresolved(self):
        self.assertEqual(
            {}, LCA.vector_typedefs(
                "typedef MysteryScalar v_t __attribute__((ext_vector_type(4)));"))

    def test_a_file_local_typedef_makes_a_cast_decidable(self):
        path = source("""
            typedef __attribute__((__vector_size__(4 * sizeof(short)))) short shortx4_t;
            __global__ void k() {
              __shared__ Bf16 as[32][68];
              *reinterpret_cast<shortx4_t*>(&as[r][c]) = v;
            }
        """)
        findings, unknown = LCA.scan(path)
        self.assertEqual([], unknown, "a file-local typedef came back unparseable")
        # 68 Bf16 = 136 bytes per row; 136 % 8 == 0, so an 8-byte cast is clean
        # where the 16-byte one in LiveSightingTest is not.
        self.assertEqual([], findings)
        self.assertEqual(0, LCA.main([str(path)]))

    def test_a_file_local_typedef_still_catches_a_real_hazard(self):
        path = source("""
            typedef __attribute__((__vector_size__(8 * sizeof(short)))) short shortx8_t;
            __global__ void k() {
              __shared__ Bf16 as[32][68];
              *reinterpret_cast<shortx8_t*>(&as[r][c]) = v;
            }
        """)
        findings, unknown = LCA.scan(path)
        self.assertEqual([], unknown)
        self.assertEqual(1, len(findings), "136-byte row stride passed a 16-byte cast")
        self.assertEqual(1, LCA.main([str(path)]))
