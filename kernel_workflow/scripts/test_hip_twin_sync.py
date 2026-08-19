"""Tests for hip_twin_sync.py (finding 87).

The checker's job is to tell apart three states that all look alike on disk:
an untouched pair, a pair where the edit reached only the primary, and a pair
where the edit reached both. Only the middle one is a defect. A checker that
flags any modified file would pass a two-case test and be useless, so every
behavioural test here carries the "edited both" control alongside.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import hip_twin_sync as hts


# hipify's actual output shape: the kernel name moves onto the rewritten line,
# so the primary's first launch line carries no launch token at all. This is
# what broke the first version of the checker, so it is the fixture.
PRIMARY = textwrap.dedent("""\
    #include "x.h"

    constexpr int kLdsStride = 68;

    __global__ void k(int m, float* c) {
    #if defined(GEAK_MFMA_ARCH)
        const int tid = threadIdx.x;
        c[tid] = m;
    #else
        (void)m;
        (void)c;
    #endif
    }

    void launch(int m, float* c, hipStream_t stream) {
        k<CTA_M, CTA_N>
            <<<grid, 256, 0, stream>>>(m, c);
    }
    """)

TWIN = textwrap.dedent("""\
    #include "x.h"

    constexpr int kLdsStride = 68;

    __global__ void k(int m, float* c) {
    #if defined(GEAK_MFMA_ARCH)
        const int tid = threadIdx.x;
        c[tid] = m;
    #else
        (void)m;
        (void)c;
    #endif
    }

    void launch(int m, float* c, hipStream_t stream) {
       hipLaunchKernelGGL(( k<CTA_M, CTA_N>)
            , dim3(grid), dim3(256), 0, stream, m, c);
    }
    """)


def write_pair(d, stem, primary, twin):
    (d / f"{stem}.hip").write_text(primary)
    (d / f"{stem}_hip.hip").write_text(twin)
    return d / f"{stem}.hip", d / f"{stem}_hip.hip"


def test_launch_syntax_alone_is_not_drift(tmp_path):
    """The whole point: these two files are equivalent and must not be flagged.

    This is the regression for the false positive the first version produced on
    a workspace that was genuinely in lockstep.
    """
    p, t = write_pair(tmp_path, "a", PRIMARY, TWIN)
    ok, detail = hts.check_pair(p, t)
    assert ok, detail


def test_edit_to_primary_only_is_drift(tmp_path):
    p, t = write_pair(tmp_path, "b", PRIMARY.replace("= 68", "= 72"), TWIN)
    ok, detail = hts.check_pair(p, t)
    assert not ok
    assert "72" in detail and "68" in detail, detail


def test_same_edit_to_both_is_not_drift(tmp_path):
    """The control that gives the previous test its meaning."""
    p, t = write_pair(tmp_path, "c",
                      PRIMARY.replace("= 68", "= 72"),
                      TWIN.replace("= 68", "= 72"))
    ok, detail = hts.check_pair(p, t)
    assert ok, detail


def test_edit_to_twin_only_is_drift(tmp_path):
    """The observed historical case (88) ran this direction: the twin was the
    file carrying the newer value."""
    p, t = write_pair(tmp_path, "d", PRIMARY, TWIN.replace("= 68", "= 72"))
    ok, detail = hts.check_pair(p, t)
    assert not ok, detail


def test_arch_guard_divergence_is_drift(tmp_path):
    """(85)'s repair must land in both files or it does not land at all."""
    p, t = write_pair(tmp_path, "e", PRIMARY,
                      TWIN.replace("defined(GEAK_MFMA_ARCH)", "defined(__gfx90a__)"))
    ok, detail = hts.check_pair(p, t)
    assert not ok, detail


def test_added_line_in_primary_only_is_drift(tmp_path):
    """Length changes, not just substitutions."""
    p, t = write_pair(tmp_path, "f",
                      PRIMARY.replace("    c[tid] = m;", "    c[tid] = m;\n    __syncthreads();"),
                      TWIN)
    ok, detail = hts.check_pair(p, t)
    assert not ok, detail


def test_reindentation_is_not_drift(tmp_path):
    """hipify reindents around what it rewrites; trailing whitespace is noise."""
    p, t = write_pair(tmp_path, "g", PRIMARY, TWIN.replace("    c[tid] = m;", "    c[tid] = m;   "))
    ok, detail = hts.check_pair(p, t)
    assert ok, detail


def test_launch_body_edit_is_still_outside_the_line_comparison(tmp_path):
    """The hole this used to pin, now recorded as a division of labour.

    Launch statements are excluded from the line comparison by construction --
    they are the one place the two files are *supposed* to differ. That has not
    changed and must not: `check_pair` staying blind here is what lets it treat
    hipify's rewrite as equivalence. What changed is that the excluded half is
    no longer unexamined; `check_launches` reads it, and the test below is the
    one that fails if that regresses.
    """
    p, t = write_pair(tmp_path, "h", PRIMARY.replace("dim3(256)", "dim3(512)")
                                            .replace("<<<grid, 256,", "<<<grid, 512,"), TWIN)
    ok, _ = hts.check_pair(p, t)
    assert ok, ("the line comparison started reading launch statements -- it must not, "
                "or hipify's own rewrite becomes drift")


# --- the launch half (finding 87, second round) -----------------------------
#
# Every test here carries its "edited both" control, for the same reason the
# line tests do: a launch checker that flags any launch at all would pass the
# single-direction half and be useless.

def test_the_two_dialects_normalize_to_the_same_tuple(tmp_path):
    """The premise everything else rests on. If hipify's rewrite did not
    round-trip, every pair in the tree would read as drift (53)."""
    p, t = write_pair(tmp_path, "l0", PRIMARY, TWIN)
    status, detail = hts.check_launches(p, t)
    assert status == "ok", detail


def test_a_block_size_edit_to_the_primary_only_is_caught(tmp_path):
    """The concrete case the old hole let through: a tuning move whose entire
    effect lives on the launch line, applied to the file ninja does not
    compile."""
    p, t = write_pair(tmp_path, "l1", PRIMARY.replace("<<<grid, 256,", "<<<grid, 512,"), TWIN)
    status, detail = hts.check_launches(p, t)
    assert status == "drift", detail
    assert "block" in detail, detail


def test_the_same_block_size_edit_applied_to_both_is_not_drift(tmp_path):
    p, t = write_pair(tmp_path, "l2",
                      PRIMARY.replace("<<<grid, 256,", "<<<grid, 512,"),
                      TWIN.replace("dim3(256)", "dim3(512)"))
    status, detail = hts.check_launches(p, t)
    assert status == "ok", detail


def test_a_grid_edit_to_the_twin_only_is_caught(tmp_path):
    p, t = write_pair(tmp_path, "l3", PRIMARY, TWIN.replace("dim3(grid)", "dim3(grid * 2)"))
    status, detail = hts.check_launches(p, t)
    assert status == "drift", detail
    assert "grid" in detail, detail


def test_a_stream_edit_is_caught(tmp_path):
    p, t = write_pair(tmp_path, "l4", PRIMARY.replace(", 0, stream>>>", ", 0, other>>>"), TWIN)
    status, detail = hts.check_launches(p, t)
    assert status == "drift", detail
    assert "stream" in detail, detail


def test_an_argument_edit_is_caught(tmp_path):
    p, t = write_pair(tmp_path, "l5", PRIMARY.replace(">>>(m, c)", ">>>(m + 1, c)"), TWIN)
    status, detail = hts.check_launches(p, t)
    assert status == "drift", detail
    assert "arguments" in detail, detail


def test_a_template_argument_edit_is_caught(tmp_path):
    """Template parameters on the launch travel inside the kernel name, which is
    the field most likely to be mis-parsed: it is the one carrying commas."""
    p, t = write_pair(tmp_path, "l6", PRIMARY.replace("k<CTA_M, CTA_N>", "k<CTA_M, CTA_N * 2>"),
                      TWIN)
    status, detail = hts.check_launches(p, t)
    assert status == "drift", detail
    assert "kernel" in detail, detail


def test_a_launch_present_in_only_one_file_is_caught(tmp_path):
    second = "void launch2" + PRIMARY.split("void launch")[1]
    p, t = write_pair(tmp_path, "l7", PRIMARY + second, TWIN)
    status, detail = hts.check_launches(p, t)
    assert status == "drift", detail
    assert "launches something the other does not" in detail, detail


def test_omitted_shmem_and_stream_default_to_zero_rather_than_reading_as_drift(tmp_path):
    """`<<<grid, block>>>` is legal and hipify always writes the twin's four
    arguments out. Without the defaults every two-argument launch in the tree
    would be reported, which is the crying-wolf failure (53)."""
    primary = PRIMARY.replace("<<<grid, 256, 0, stream>>>", "<<<grid, 256>>>")
    twin = TWIN.replace(", dim3(grid), dim3(256), 0, stream, m, c)",
                        ", dim3(grid), dim3(256), 0, 0, m, c)")
    p, t = write_pair(tmp_path, "l8", primary, twin)
    status, detail = hts.check_launches(p, t)
    assert status == "ok", detail


def test_an_unreadable_launch_is_a_hole_not_a_pass(tmp_path):
    """The residual hole, made loud. A statement this parser cannot normalize
    must not return 'ok': that would be the exact shape of the bug being fixed,
    a check reporting agreement about something it never read (54)."""
    p, t = write_pair(tmp_path, "l9",
                      PRIMARY.replace("<<<grid, 256, 0, stream>>>(m, c);",
                                      "<<<grid>>>(m, c);"), TWIN)
    status, detail = hts.check_launches(p, t)
    assert status == "hole", detail
    assert "UNCHECKED" in detail, detail


def test_a_hole_exits_3_and_a_drift_still_exits_1(tmp_path):
    """Distinct codes, because a caller has to be able to tell "the launch half
    was not read" from "the launch half disagreed". Collapsing them into 1 would
    make an unparseable launch look like a found defect; collapsing into 0 would
    make it look like a pass."""
    hole, drift = tmp_path / "hole", tmp_path / "drift"
    hole.mkdir(); drift.mkdir()
    write_pair(hole, "a", PRIMARY.replace("<<<grid, 256, 0, stream>>>(m, c);",
                                          "<<<grid>>>(m, c);"), TWIN)
    write_pair(drift, "a", PRIMARY.replace("<<<grid, 256,", "<<<grid, 512,"), TWIN)
    script = str(Path(hts.__file__))
    assert subprocess.run([sys.executable, script, str(hole)]).returncode == 3
    assert subprocess.run([sys.executable, script, str(drift)]).returncode == 1


def test_a_pair_that_drifts_in_both_halves_is_counted_once(tmp_path):
    """Per-pair, not per-check: one stale twin is one problem, and reporting it
    as two would overstate the size of the finding."""
    p, t = write_pair(tmp_path, "l10",
                      PRIMARY.replace("= 68", "= 72").replace("<<<grid, 256,", "<<<grid, 512,"),
                      TWIN)
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = hts.main(["hip_twin_sync.py", str(tmp_path)])
    assert rc == 1
    assert "0 pair(s) in lockstep, 1 drifted" in buf.getvalue(), buf.getvalue()


def test_twin_of_a_twin_is_not_paired(tmp_path):
    """`a_hip.hip` must not pair with `a_hip_hip.hip`."""
    write_pair(tmp_path, "i", PRIMARY, TWIN)
    (tmp_path / "i_hip_hip.hip").write_text(TWIN)
    pairs = hts.find_pairs(tmp_path)
    assert [p.name for p, _ in pairs] == ["i.hip"]


def test_unpaired_primary_is_skipped(tmp_path):
    (tmp_path / "lonely.hip").write_text(PRIMARY)
    assert hts.find_pairs(tmp_path) == []


def test_no_pairs_exits_2_not_0(tmp_path):
    """A tree with nothing to check is a hole, not a pass. Exiting 0 here would
    let a caller that only tests `rc == 0` treat "checked nothing" as "clean"."""
    (tmp_path / "lonely.hip").write_text(PRIMARY)
    assert hts.main(["hip_twin_sync.py", str(tmp_path)]) == 2


def test_exit_codes_end_to_end(tmp_path):
    clean, dirty = tmp_path / "clean", tmp_path / "dirty"
    clean.mkdir(); dirty.mkdir()
    write_pair(clean, "a", PRIMARY, TWIN)
    write_pair(dirty, "a", PRIMARY.replace("= 68", "= 72"), TWIN)
    script = str(Path(hts.__file__))
    assert subprocess.run([sys.executable, script, str(clean)]).returncode == 0
    assert subprocess.run([sys.executable, script, str(dirty)]).returncode == 1


def test_missing_directory_fails_closed(tmp_path):
    assert hts.main(["hip_twin_sync.py", str(tmp_path / "nope")]) == 1


@pytest.mark.parametrize("tree", [
    "exp/opt_bf16_20260814/ws_a/src",
    "examples/tasks/dense_bf16_gemm_fused/src",
])
def test_known_good_real_trees_stay_clean(tree):
    """Guards against a future tightening that starts crying wolf (53) on trees
    that are actually in lockstep. Skips rather than fails if the tree is gone,
    since these live outside the package."""
    root = Path(__file__).resolve().parents[2] / tree
    if not root.is_dir():
        pytest.skip(f"{tree} not present")
    pairs = hts.find_pairs(root)
    assert pairs, f"{tree} has no pairs -- this test would pass vacuously"
    for primary, twin in pairs:
        ok, detail = hts.check_pair(primary, twin)
        assert ok, f"{primary.name}: {detail}"
        # The launch half on real hipify output, which is the only place the
        # parser meets the dialect it was written against rather than a fixture
        # I wrote to match it.
        status, ldetail = hts.check_launches(primary, twin)
        assert status == "ok", f"{primary.name} launches: {ldetail}"
