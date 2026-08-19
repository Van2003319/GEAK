"""Proof that the two-sentinel write probe pins the defect it claims to pin (82).

This file is BOTH a pytest test and a standalone script, and the two modes must not interfere:

  * As a script (`python3 test_write_probe_coincidence.py`) it prints a per-case table and exits
    non-zero on any failure.
  * Under pytest it is a single test that SKIPS unless `GEAK_GPU_TESTS=1`.

The skip is not squeamishness about slow tests. The probe allocates on the GPU, and every piece of
GPU work in this pipeline goes through `gpu_lock.sh`; a test that grabs a device the moment someone
types `pytest -q` would run unfenced next to a live measurement and quietly widen its noise. So the
deterministic suite stays CPU-only by default and this proof is opt-in.

The earlier version of this file did its work at import time and ended in `sys.exit(0)`. pytest's
collector imports every candidate module, so that `SystemExit` aborted collection with
INTERNALERROR and *the entire suite reported "no tests ran"* -- a green-looking outcome that had
executed nothing. A test file that disables the test runner is worse than a missing test: the
missing one is visibly absent. Hence the `__main__` guard.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SHAPE, DEV, DT_NAME = (64, 512), "cuda", "bfloat16"


def _cases(torch, H):
    def mk(fn):
        def call(args):
            out = torch.empty(SHAPE, dtype=getattr(torch, DT_NAME), device=DEV)
            fn(out)
            return out
        return call

    return [
        # A kernel that returns an untouched buffer must be caught...
        ("never_writes",      mk(lambda o: None),                       "never_written"),
        # ...while a kernel whose OUTPUT HAPPENS TO EQUAL a sentinel everywhere must NOT be. These
        # two rows are the whole reason the probe uses two sentinels: under the old single-sentinel
        # probe, `writes_poison_A` was indistinguishable from `never_writes`, and that is not a
        # hypothetical -- the rocBLAS oracle itself hit the coincidence on prefill_m512_up.
        ("writes_poison_A",   mk(lambda o: o.fill_(H.POISON_A)),        "ok"),
        ("writes_poison_B",   mk(lambda o: o.fill_(H.POISON_B)),        "ok"),
        ("normal_write",      mk(lambda o: o.fill_(0.25)),              "ok"),
        # Partial coverage: one element and one row. A per-shape masked-epilogue or tile-bound bug
        # looks like these, not like never_written.
        ("skips_one_element", mk(lambda o: o.view(-1)[1:].fill_(0.25)), "partially_written"),
        ("skips_last_row",    mk(lambda o: o[:-1].fill_(0.25)),         "partially_written"),
    ]


def _evaluate():
    """Returns a list of (name, got_status, want_status, frac)."""
    import torch
    import harness_lib as H

    rows = []
    for name, call, want in _cases(torch, H):
        _ok, status, frac = H.assert_writes_output(call, None)
        rows.append((name, status, want, frac))
    return rows


@pytest.mark.skipif(os.environ.get("GEAK_GPU_TESTS") != "1",
                    reason="needs a GPU; run under gpu_lock.sh with GEAK_GPU_TESTS=1")
def test_two_sentinel_probe_distinguishes_coincidence_from_absence():
    rows = _evaluate()
    bad = [(n, got, want) for n, got, want, _ in rows if got != want]
    assert not bad, f"probe misclassified: {bad}"


if __name__ == "__main__":
    fails = 0
    for name, got, want, frac in _evaluate():
        good = got == want
        fails += not good
        print(f"{'PASS' if good else 'FAIL'}  {name:20s} -> {got:18s} frac={frac} (want {want})")
    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
