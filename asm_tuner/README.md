# Canonical asm_tuner integration

`asm_tuner` is executable gfx942 infrastructure for the canonical BF16 and FP8 rowwise GEMM tasks.
The task contracts bind `CASES`, input construction, correctness, measurement-protocol/v2, the
frozen oracle, route data, and the generated search-space manifest into one task digest.

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

Every halving round uses the canonical protocol-v2 semantic-weight slots and paired arm ordering.
Round receipts include artifact/ISA fingerprints and are reusable after interruption; compilation is
content-addressed. A changed task, protocol, candidate set, binary, or ISA invalidates the cache.

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
`materialize` installs immutable artifacts and publishes one complete `geak.runtime_table/v2` with a
single atomic replace. Each measured route carries its ranked top-K; the task-local bridge validates
contiguous ranks and task/protocol bindings and dispatches rank zero. Shapes outside the eleven-case
suite use the declared canonical fallback.

`verify-materialization --rebuild` independently copies the task, validates every ranked artifact,
rebuilds source, compares ISA/symbols, and loads the libraries. The bounded BF16/FP8 correctness and
PGR ISA smoke is recorded in `evidence/gfx942_pgr2_smoke_20260902.json`.
