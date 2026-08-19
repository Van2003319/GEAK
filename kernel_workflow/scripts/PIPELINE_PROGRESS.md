
## Finding (111): the oracle digest over-covers, so it cannot do the job it exists for

The relaunch bootstrapped clean — `baseline_frozen: true`, `baseline_callable:
oracle_wrapper:rocblas_bf16_gemm`, and the note *"KERNEL_PATH_ORIG was never
written to"*. But it reported

```
oracle_digest: 8a697c649ded3c8f... over 18 files
```

against the failed run's

```
oracle_digest: 7bd6dba52b41f64c... over 71 files
```

A changed scoring oracle would invalidate every comparison to prior runs, so I
checked instead of assuming. All six oracle-relevant files —
`src/rocblas_baseline.{cpp,h}`, `src/rocblas_bindings.cpp`, `oracle_wrapper.py`,
`oracle_loader.py`, `harness_lib.py` — are **byte-identical** by `cmp` between
the two frozen baselines. The scoring oracle did not change.

The digest differs because it is taken over the **whole frozen tree** (61 files
vs 18). The 43 extra files in the old baseline are `results/`, `research/` and
`build/` leftovers copied from the original task — candidate JSON receipts and
rejected `.hip` experiments that cannot affect what the oracle computes.

So the check is **sound but not useful**: it can never wrongly report two
different oracles as the same, but it reports two *identical* oracles as
different whenever anything else in the tree moved. A digest that changes for
reasons unrelated to its subject gets learned as noise and then ignored — which
is (55) arriving by a different road: not a gate whose result is fixed, but a
gate whose result is uninformative.

This is one of the three auditor HOLEs already on the list ("oracle digest"), now
with a concrete diagnosis: **the digest must cover the oracle's own inputs — the
baseline sources, bindings, wrapper, loader and harness — and nothing else.**
Scoped that way it would have read identical across both runs, which is the true
answer. Not fixed yet; the run owns the tree right now.

Incidental confirmation: the pristine tree is doing what it was built for. 18
files against 61, with the 43 dropped ones all scratch.
