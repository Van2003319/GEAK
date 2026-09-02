# Canonical asm_tuner integration

`asm_tuner` is executable gfx942 infrastructure for the canonical BF16 and FP8 rowwise GEMM tasks.
The task contracts bind `CASES`, input construction, correctness, measurement-protocol/v2, a
hashed `geak.score-metric/v1`, the frozen oracle, route data, and the generated search-space manifest
into one task digest. The canonical BF16/FP8 manifests select `sustained_throughput` (`inner=16`).

## Generated bounded search

The v2 search-space manifests declare domains plus a dependent generator; vendor database points
are not copied into the executable set. The generator forms legal MFMA wave-grid factor pairs and
applies three profile sweeps (pipeline/LDS, global+epilogue I/O, and mapping/schedule). Static
configuration and route dependencies run before compilation. Grid coverage, edge waste, and tile
reuse are soft ordering signals only.

- BF16 domains contain BM/BN `{64,96,112,128,160,176,192,224,256}`, BK `{32,64,128}`, 2/4/8-wave
  grids, PGR1/PGR2, independent A/B pad and buffer choices, vector/store, WGM, and stagger. They
  produce 61,020 unique points; 38,192 survive the precompile source-level resource gate.
- FP8 domains produce 50,880 points (45,342 static survivors) and naturally contain the
  128x128x128 2x2-wave point and neighbouring tiles/depths.
- A full route starts at top-12 and halves 12 -> 6 -> 3, retaining top-2. It has a 24 arm-evaluation
  cap and a 160-candidate union compile cap. The current eleven-route plan deduplicates to 85 kernels
  for either task and balances them as five 11-kernel plus three 10-kernel shards.
- Bootstrap starts at top-8, measures 8 -> 4, and retains top-2 with a 12 arm-evaluation cap and one
  8-kernel shard. BF16
  bootstraps `prefill_m2048_square`; FP8 bootstraps `prefill_m1024_down`.

Every halving round uses the score metric from the task contract: for the fresh tasks this is
`sustained_throughput`, continuous `inner=16` launch groups over canonical protocol-v2
semantic-weight slots and paired arm ordering. Round receipts bind metric id/inner as well as
artifact/ISA fingerprints and are reusable after interruption. A changed task, protocol, score
metric, candidate set, binary, or ISA invalidates the cache. Ranking never falls back to the other
metric.

## PGR2 and resource gates

PGR2 preloads the next global K stage into VGPR vector packs before consuming the current LDS stage.
The packs stay live across MFMA and are written into LDS on the following iteration. PGR does not
multiply the LDS footprint, so one A and one B LDS buffer are valid. PGR1 and PGR2 semantic pairs
must compile to different ISA hashes.

Compilation is accepted only when amdhsa metadata reports exact LDS, no private-segment spill,
architectural register counts, parsed instructions, and nonzero occupancy. PLR and XCC remain
inactive at `0` and `1`; GSU and StreamK remain unsupported at `1` and `0`. All four fail closed.

## Canonical bridge and publication

`kernel_workflow/scripts/config_search.py search-asm` expands and prunes domains, compiles bounded
shards, then performs per-route successive halving under `gpu_lock.sh` and `gpu_fence_run.sh`.
`prepare`/`materialize` install immutable content-addressed artifacts below the task's ignored
`.geak_tuner/` directory, then atomically publish the complete suite to three stable task-root files:
`asm_tuner_registry.json`, `asm_tuner_runtime_table.json`, and `asm_tuner_selection.json`. The
registry's versioned `launch_contract` cross-binds the task/source/oracle digests, protocol and score
metric, tracked table and selection hashes, workflow consumer, task-local loader, and normalized
artifact-set digest. Publication also writes a content-addressed receipt below
`.geak_tuner/receipts/`. Each measured route carries its ranked top-K; the task-local bridge consumes
the tracked table, validates contiguous ranks and task/protocol/score-metric bindings, and dispatches
rank zero. Shapes outside the eleven-case suite use the declared canonical fallback.

The committed oracle files still describe the earlier latency bake and are deliberately incompatible
with these task contracts. Before `search-asm`, run each task's full-suite `oracle-bakeoff
--score-metric sustained_throughput --confirm-oracle-bakeoff` under the GPU lock/fence and commit the
reviewed manifests. Do not edit `selection_metric` by hand.

`verify-materialization --rebuild` independently copies the task, validates every ranked artifact,
rebuilds source, compares ISA/symbols, and loads the libraries. The bounded BF16/FP8 correctness and
PGR ISA smoke is recorded in `evidence/gfx942_pgr2_smoke_20260902.json`.
