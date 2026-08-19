#!/usr/bin/env python3
"""Deterministic geak-qd-v2 descriptor vocabulary and gfx90a eligibility rules.

kernel_lane.js's QD v2 archive classifies a candidate as one observed workload
CONTEXT (a benchmark case id) paired with one AMD mechanism DESCRIPTOR -- a
five-field named tuple, never a dense integer grid. This module is a
standalone, GPU-free, filesystem-free mirror of that same vocabulary and
legality/coverage-eligibility contract (kept intentionally identical to
kernel_lane.js's QD_VOCAB / qdDescriptorValid / qdCoverageEligible / qdCellId
so a Python-side caller and the JS orchestrator agree on what a valid cell is),
plus one thing kernel_lane.js does not compute: ADJACENCY -- the neighbor
descriptors reachable by moving exactly one axis to the next/previous position
in that axis's fixed, meaningful order. Adjacency is what a "directed
transition" QD operator needs to propose a nearby, still-legal mechanism
without re-deriving the vocabulary's ordering by hand.

Two architectures are modeled, gfx90a (CDNA2 / MI200-class) and gfx942
(CDNA3 / MI300-class). Any other is rejected outright rather than silently
approximated, because the legality rules below (wavefront-locked wave
scheduling, LDS-mediated K pipelines, die-count-dependent rasterization) are
per-architecture engineering judgments, not portable physics.

WHAT EARNS AN AXIS. An axis is admitted only if moving along it changes at
least one of the three quantities a kernel actually trades -- effective
arithmetic intensity, occupancy, or memory traffic -- or the binding time of
the plan. An axis whose moves leave all of those invariant is a *renaming* of
the search space, not an enlargement of it, and adding it makes the archive
look better covered while searching exactly the same points. This is not a
style rule; it was learned by measurement. A sixth axis
`mfma_shape: {16x16x16, 32x32x8}` was proposed on the strength of rocBLAS's
kernel name, implemented, and measured at 0.969 against a 1.062 baseline --
and the reason is arithmetic: effective intensity is
`(kFM*kFN/(kFM+kFN)) * MACs-per-LDS-element`, and doubling the fragment edge
doubles the second factor while halving both kFM and kFN, so the product is
invariant by construction. `AXIS_EFFECTS` below records, per axis, which of
the three quantities it moves; `dead_axes()` is the check.

That rejection was re-litigated on gfx942 and it survives, with a sharper
reason -- see `MFMA_SHAPE_IS_NOT_AN_AXIS` below. The short version: the
invariance argument above explains why the large fragment cannot *win*, and
the gfx942 measurement adds why it actively *loses* 13-16%. Both halves matter,
because a reader who only has the invariance argument will conclude the axis is
merely useless and might add it for coverage's sake.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace as _replace
from typing import Mapping, Sequence

CLASSIFIER_VERSION = "geak-qd-v2"
SUPPORTED_ARCH = "gfx90a"
SUPPORTED_ARCHES: tuple[str, ...] = ("gfx90a", "gfx942")
# Architectures whose dispatcher hands consecutive workgroup ids to different
# dies round-robin, so that blockIdx adjacency is NOT last-level-cache
# adjacency and un-shuffling it before grouping is a real traffic mechanism.
# gfx942 has 8 XCDs behind 8 L2s; gfx90a's two GCDs are separate HIP devices
# and a single kernel never spans them, so the mechanism has nothing to undo.
MULTI_DIE_ARCHES: frozenset[str] = frozenset({"gfx942"})
SUPPORTED_DTYPES: tuple[str, ...] = ("bf16", "fp16", "fp32", "fp64", "int8")
# Tombstone. `RouteFacts.occupancy_fill` -- `min(1, ctas/residency_slots)` --
# existed briefly as the term finding (23) thought had survived. Finding (24)
# removed it: it reads 1.000 for BOTH configurations that crossed a round
# boundary (m2048 sk32 double, -26.7%; m512 sk32 double, -43.4%) and 0.56-0.95
# for every configuration that won, because it saturates exactly where the grid
# overflows into a second pass. It points at the cliff. Use `rounds` and
# `round_slack`. Do not reintroduce it under any name.
OCCUPANCY_FILL_REMOVED_BY_FINDING = 24

# Tombstone. `mfma_shape: {16x16x16, 32x32x8}` is REFUSED as an axis twice over,
# on both modeled architectures, for two different reasons that a future reader
# needs together:
#
#   gfx90a, the intensity argument (module docstring): the product
#   `(kFM*kFN/(kFM+kFN)) * MACs-per-LDS-element` is invariant when the fragment
#   edge doubles, so the move cannot raise intensity. Measured 0.969 vs 1.062.
#   This says the axis cannot win.
#
#   gfx942, the ILP argument (findings 41-43): it also loses, 15.6% on
#   `prefill_m2048_square` and 13.4% on `prefill_m512_up`, byte-identical
#   control flat, ABBA, 4 reps/arm, runs 1910-1933. At a FIXED wave tile a
#   32x32 fragment covers 4x the area of a 16x16 one, so it buys half the MFMA
#   instructions by QUARTERING the accumulator count -- 8 independent dependency
#   chains of 16-cycle ops become 2 chains of 32-cycle ops. Counters confirm the
#   trade is not even clean: `SQ_VALU_MFMA_BUSY_CYCLES`, `SQ_LDS_IDX_ACTIVE` and
#   `SQ_LDS_BANK_CONFLICT` all bit-identical across the two shapes, while
#   `SQ_INSTS_LDS` rose 1.75x -- the large fragment pins K granularity at 8 and
#   spends MORE LDS issue slots to move the SAME bytes.
#
# The gfx942 half is the one with teeth for the archive design, because it is a
# COUPLING and not a dead axis: the shape is only bad at a wave tile chosen for
# the other shape. Restoring the 8 chains needs a 128x64 wave tile, whose
# accumulators alone cost 128 VGPR and land the kernel near 160, crossing
# finding (24)'s `rounds` boundary at a 27-43% price. So (shape, wave_tile) is
# ONE mechanism, and offering `mfma_shape` as a free axis would let the search
# mutate half of it -- generating exactly the measured 13-16% loss on every route
# it touched, while `dead_axes()` stayed happy because the axis does move
# occupancy. **A dead-axis check cannot catch a coupled axis.** That is the
# transferable lesson and the reason this tombstone is longer than the others.
#
# Reaching the axis at all required leaving rocWMMA, which silently emulates the
# 32x32x8 bf16 fragment on gfx942 by lowering it to `v_mfma_f32_16x16x16_bf16`
# (finding 42). Any future claim that some axis "was tried and did not help"
# through a library abstraction should be checked at the opcode level first.
MFMA_SHAPE_IS_NOT_AN_AXIS = "findings 41-43; coupled to wave tile, not dead"
_TOMBSTONED_AXES: frozenset[str] = frozenset({"mfma_shape", "occupancy_fill"})

# LDS per CU on the modeled architectures. Both gfx90a and gfx942 have 64 KB.
_LDS_BYTES_PER_CU = 65536

# The unified VGPR/AGPR file, per lane per SIMD, on gfx90a and gfx942, its
# allocation granule, and the hardware ceiling on resident wave64s per SIMD.
# These only ever apply to counts read back from a built object -- see
# `RouteFacts.ctas_by_vgpr` for why they are never inferred from source.
_VGPR_FILE = 512
_VGPR_GRANULE = 8
_MAX_WAVES_PER_SIMD = 8

# gfx942 (MI300X) last-level cache -- the Infinity Cache / MALL, shared by all
# 8 XCDs behind their private 4 MB L2s. Finding (17): a workspace smaller than
# this is not an HBM term, and pricing it as one produced a 10x-fewer-bytes
# design that measured 8% slower.
_MALL_BYTES = 256 * 1024 * 1024

# Order matters: each tuple is the axis's canonical progression from the
# simplest mechanism to the most sophisticated one. Adjacency only ever steps
# to the immediate neighbor in this order -- it never jumps across it.
AXIS_ORDER: Mapping[str, tuple[str, ...]] = {
    "compute_primitive": ("valu", "rocwmma", "native_mfma"),
    "wave_schedule": ("independent", "symmetric_interleave", "symmetric_pingpong",
                       "asymmetric_producer_consumer"),
    # lds_deep_single sits between pingpong and multistage on purpose, and it is
    # not a weaker pingpong: past a certain stage depth the double buffer stops
    # fitting in LDS at all, so "deeper stage, one buffer" is the only way
    # further along this axis. It is a real mechanism with a measured win --
    # a 32x64 tile at a 256-deep stage, single-buffered, one CTA per CU, moved
    # its shape from 0.76 to 0.88 -- and it is reachable from lds_pingpong by
    # exactly one step, which is what makes it a directed transition rather
    # than a redesign.
    "k_pipeline": ("direct_global", "lds_single", "lds_reg_prefetch", "lds_pingpong",
                   "lds_deep_single", "lds_multistage"),
    "decomposition": ("tile_grid", "persistent_output", "split_k", "stream_k"),
    "output_path": ("direct_store", "lds_staged_store", "atomic_fixup", "workspace_fixup"),
    # Which output tile a workgroup id maps to. Pure traffic: the arithmetic is
    # identical and the occupancy is identical, but the set of A and B panels a
    # single last-level cache sees is not. Measured on gfx942 at a 4x86 grid:
    # 4 A + 43 B panels per XCD under the plain mapping, 4 A + 11 B after
    # un-shuffling the round-robin and then grouping.
    "rasterization": ("linear", "grouped_m", "xcd_remapped_grouped"),
    # When the launch configuration is chosen. `static` decides on the host
    # from shape and CU count; `runtime_tuned` measures the candidate
    # configurations on the real stream and keeps the winner. The resource
    # traded is not intensity, occupancy or traffic -- it is host time and
    # per-call variance against per-shape fit -- which is why it is a separate
    # axis and not a value of another one. Measured worth +0.012 geomean once
    # the tuner was made to measure in the same cache regime as the harness.
    "plan_binding": ("static", "runtime_tuned"),
}
QD_VOCAB: Mapping[str, frozenset[str]] = {
    axis: frozenset(values) for axis, values in AXIS_ORDER.items()
}
AXES: tuple[str, ...] = tuple(AXIS_ORDER)
_REDUCTION = frozenset({"split_k", "stream_k"})
_FIXUP = frozenset({"atomic_fixup", "workspace_fixup"})
_PINGPONG_FAMILY = frozenset({"symmetric_interleave", "symmetric_pingpong"})


def descriptor_reject(descriptor: Mapping[str, object] | None, *,
                      arch: str = SUPPORTED_ARCH, dtype: str = "bf16") -> str | None:
    """None if legal, else a token naming the single rule that refused.

    Mirrors kernel_lane.js's `qdDescriptorReject` token for token, deliberately:
    the two sides are one policy, and a refusal an agent reads in one wording
    and the orchestrator logs in another is two policies that happen to agree
    today. `test_qd_lane_parity.py` asserts the vocabularies stay identical.

    This exists for the reason finding (48) gives: `descriptor_valid` returned a
    bare `False` for six different rules, so an agent whose descriptor was
    rejected learned only that something was wrong with it. Finding (44) is what
    that costs -- every descriptor from every agent was refused for weeks over a
    two-axis vocabulary mismatch, and the only symptom was an empty archive.

    The ONLY copy of the rules. `descriptor_valid` delegates rather than
    restating them; a second explain-only copy is precisely how (44) happened.
    """
    if not isinstance(descriptor, Mapping):
        return "descriptor:absent"
    for axis, values in QD_VOCAB.items():
        if descriptor.get(axis) not in values:
            got = "<missing>" if axis not in descriptor else json.dumps(descriptor.get(axis))
            return f"axis:{axis}={got}"
    reduction = descriptor["decomposition"] in _REDUCTION
    fixup = descriptor["output_path"] in _FIXUP
    # Finding (61): one rule, two violations needing opposite corrections. The
    # planner reading the token is what has to choose, so the token says which.
    if reduction and not fixup:
        return "rule:reduction_without_fixup"
    if fixup and not reduction:
        return "rule:fixup_without_reduction"
    if descriptor["wave_schedule"] in _PINGPONG_FAMILY and descriptor["compute_primitive"] == "valu":
        return "rule:pingpong_requires_matrix_core"
    if arch not in SUPPORTED_ARCHES or dtype not in SUPPORTED_DTYPES:
        return "rule:unsupported_arch_or_dtype"
    if (descriptor["rasterization"] == "xcd_remapped_grouped"
            and arch not in MULTI_DIE_ARCHES):
        return "rule:xcd_remap_requires_multi_die"
    if descriptor["plan_binding"] == "runtime_tuned" and not reduction:
        return "rule:runtime_tuned_requires_reduction"
    return None


def descriptor_valid(descriptor: Mapping[str, object] | None, *,
                      arch: str = SUPPORTED_ARCH, dtype: str = "bf16") -> bool:
    """True iff `descriptor` is a legal geak-qd-v2 mechanism tuple on `arch`/`dtype`.

    Thin predicate over `descriptor_reject`, which holds the rules and names
    them. Mirrors kernel_lane.js's qdDescriptorValid exactly:
      1. every axis is present and takes one of its named values;
      2. a K-reduction decomposition (split_k/stream_k) must be paired with a
         fixup output_path (atomic_fixup/workspace_fixup), and vice versa --
         a reduction with no fixup silently drops partial sums, and a fixup
         with no reduction has nothing to fix up;
      3. symmetric wave pairing (interleave/pingpong) requires a matrix-core
         compute_primitive: bare VALU issue has no producer/consumer wave
         pairing to synchronize against gfx90a's fixed 64-wide wavefront;
      4. only a modeled architecture, and only the dtypes GEAK's harness can
         currently verify;
      5. xcd_remapped_grouped only where the dispatcher actually round-robins
         workgroups across dies (gfx942, not gfx90a);
      6. runtime_tuned only where a K reduction gives the tuner a parameter.

    Rules 5 and 6 above are worth keeping in words next to the code, since the
    tokens alone do not carry the argument: un-shuffling a die round-robin is
    only a mechanism where the hardware shuffles -- on a single-die dispatch it
    is the identity plus a scalar prologue, i.e. a pure cost. And runtime tuning
    needs something to tune; the only launch parameter chosen per call rather
    than per route is the K-reduction slice count, so `runtime_tuned` without a
    reduction claims a mechanism the kernel does not have.
    """
    return descriptor_reject(descriptor, arch=arch, dtype=dtype) is None


def coverage_eligible(descriptor: Mapping[str, object] | None, *,
                       arch: str = SUPPORTED_ARCH, dtype: str = "bf16") -> bool:
    """Legal cells worth directed QD credit: excludes asymmetric_producer_consumer.

    That wave_schedule is a legal mechanism (kernels using it can still occupy
    an archive cell), but it is not a directed-transition TARGET: a
    producer/consumer split is a whole-kernel restructuring decision, not a
    local knob a directed transition should aim at from a neighboring cell.

    Written as a design argument, now also a measured one -- see finding (8).
    It was built and run on gfx942/bf16 at last (runs 519-560) and is 1.5%
    SLOWER on the shipped path, so the exclusion costs nothing on the one route
    where the value has now been tried.
    """
    return (descriptor_valid(descriptor, arch=arch, dtype=dtype)
            and descriptor["wave_schedule"] != "asymmetric_producer_consumer")


def cell_id(context_id: str, descriptor: Mapping[str, object] | None, *,
            known_contexts: Sequence[str] | None = None,
            arch: str = SUPPORTED_ARCH, dtype: str = "bf16") -> str | None:
    """The archive cell key for (context_id, descriptor), or None if either is invalid.

    `known_contexts`, when given, restricts context_id to that allow-list
    (mirrors kernel_lane.js checking membership in QD_CONTEXT_IDS, derived
    from the baseline's own per-case ids -- a context that was never
    benchmarked cannot be a cell).
    """
    if not context_id or not isinstance(context_id, str):
        return None
    if known_contexts is not None and context_id not in known_contexts:
        return None
    if not descriptor_valid(descriptor, arch=arch, dtype=dtype):
        return None
    return "|".join([context_id, *(str(descriptor[axis]) for axis in AXES)])


@dataclass(frozen=True)
class Neighbor:
    descriptor: dict[str, str]
    axes: tuple[str, ...]
    direction: str  # "prev" | "next" | "coupled"

    @property
    def axis(self) -> str:
        """Backward-compatible label for ordinary one-axis transitions."""
        return self.axes[0] if len(self.axes) == 1 else "+".join(self.axes)

    def object(self) -> dict[str, object]:
        return {"descriptor": dict(self.descriptor), "axes": list(self.axes),
                "direction": self.direction}


def adjacency(descriptor: Mapping[str, object], *, arch: str = SUPPORTED_ARCH,
              dtype: str = "bf16") -> list[Neighbor]:
    """Legal named neighbors, including coupled reduction/fixup transitions.

    Ordinary edges change one axis by one ordered step. The legality invariant
    that reductions and fixups appear together requires explicit two-axis
    edges at that boundary: persistent_output/non-fixup ↔ split_k/fixup and
    split_k ↔ stream_k while preserving the selected fixup. This avoids the
    dead illegal intermediate state produced by a Cartesian one-axis walk.
    The split_k -> persistent_output direction also has to release
    plan_binding=runtime_tuned, for the same reason.
    """
    if not isinstance(descriptor, Mapping) or not descriptor_valid(
            descriptor, arch=arch, dtype=dtype):
        return []
    out: list[Neighbor] = []
    seen: set[tuple[tuple[str, str], ...]] = set()

    def add(candidate: Mapping[str, object], axes: tuple[str, ...], direction: str) -> None:
        normalized = {axis: str(candidate[axis]) for axis in AXES}
        key = tuple((axis, normalized[axis]) for axis in AXES)
        if key in seen or not descriptor_valid(normalized, arch=arch, dtype=dtype):
            return
        seen.add(key)
        out.append(Neighbor(descriptor=normalized, axes=axes, direction=direction))

    for axis in AXES:
        order = AXIS_ORDER[axis]
        current = descriptor[axis]
        idx = order.index(current)
        for step, direction in ((-1, "prev"), (1, "next")):
            j = idx + step
            if j < 0 or j >= len(order):
                continue
            candidate = dict(descriptor)
            candidate[axis] = order[j]
            add(candidate, (axis,), direction)

    decomposition = str(descriptor["decomposition"])
    output_path = str(descriptor["output_path"])
    if decomposition == "persistent_output" and output_path not in _FIXUP:
        for fixup in AXIS_ORDER["output_path"]:
            if fixup in _FIXUP:
                candidate = dict(descriptor, decomposition="split_k", output_path=fixup)
                add(candidate, ("decomposition", "output_path"), "coupled")
    elif decomposition == "split_k":
        for non_fixup in AXIS_ORDER["output_path"]:
            if non_fixup not in _FIXUP:
                # Dropping the reduction also drops the only parameter the
                # runtime tuner binds, so plan_binding has to come back to
                # static in the same step or the edge lands on an illegal
                # descriptor and silently disappears.
                axes = ("decomposition", "output_path")
                candidate = dict(descriptor, decomposition="persistent_output",
                                 output_path=non_fixup)
                if candidate["plan_binding"] == "runtime_tuned":
                    candidate["plan_binding"] = "static"
                    axes = ("decomposition", "output_path", "plan_binding")
                add(candidate, axes, "coupled")

    return out


def all_legal_descriptors(*, arch: str = SUPPORTED_ARCH, dtype: str = "bf16") -> list[dict[str, str]]:
    """Every legal descriptor tuple on `arch`/`dtype`, in deterministic AXES order.

    Small by construction (<= 3*4*6*4*4*3*2 = 6912 combinations before filtering),
    so an exhaustive scan is cheap and gives tests/tools a ground truth to
    check `descriptor_valid`/`adjacency` against instead of hand enumeration.
    """
    out: list[dict[str, str]] = []

    def rec(i: int, acc: dict[str, str]) -> None:
        if i == len(AXES):
            if descriptor_valid(acc, arch=arch, dtype=dtype):
                out.append(dict(acc))
            return
        axis = AXES[i]
        for value in AXIS_ORDER[axis]:
            acc[axis] = value
            rec(i + 1, acc)

    rec(0, {})
    return out


# Which of the three tradeable quantities each axis actually moves. An axis
# with an empty entry is a dead axis: see the module docstring. `binding` is
# listed separately from the physical three because plan_binding trades host
# time and decision variance rather than device resources, and collapsing it
# into "traffic" would make the dead-axis check unfalsifiable.
AXIS_EFFECTS: Mapping[str, frozenset[str]] = {
    "compute_primitive": frozenset({"intensity", "occupancy"}),
    "wave_schedule": frozenset({"occupancy"}),
    "k_pipeline": frozenset({"occupancy", "traffic"}),
    "decomposition": frozenset({"occupancy", "traffic"}),
    "output_path": frozenset({"traffic"}),
    "rasterization": frozenset({"traffic"}),
    "plan_binding": frozenset({"binding"}),
}
TRADEABLE: frozenset[str] = frozenset({"intensity", "occupancy", "traffic", "binding"})


def dead_axes() -> list[str]:
    """Axes that move nothing tradeable, in AXES order. Must always be empty.

    CORRECTION, findings (41)-(43). This docstring used to claim it was "the
    check that would have rejected the `mfma_shape` axis". It would not have.
    `mfma_shape` moves occupancy -- the gfx942 build measurably dropped VGPR
    from 76 to 64 -- so it is tradeable by this test's own definition and would
    have passed cleanly. The axis is refused for being COUPLED to wave tile,
    not for being dead, and no per-axis predicate can see a coupling. Keep the
    two failure modes separate: `dead_axes()` catches renamings of the search
    space, `tombstoned_axes()` catches the ones that were measured and refused.
    """
    return [axis for axis in AXES
            if not (AXIS_EFFECTS.get(axis, frozenset()) & TRADEABLE)]


def tombstoned_axes() -> list[str]:
    """Axes that were proposed, measured, refused, and must not come back.

    Must always be empty. Unlike `dead_axes()` this is not a derivation from
    axis properties -- it cannot be, because the reasons are measurements rather
    than definitions -- it is a re-adoption guard. An axis lands here when
    building it produced a run-identified loss, so the cost of rediscovering it
    is a build plus a sweep. See `MFMA_SHAPE_IS_NOT_AN_AXIS` and
    `OCCUPANCY_FILL_REMOVED_BY_FINDING` for the two entries and their evidence.
    """
    return [axis for axis in AXES if axis in _TOMBSTONED_AXES]


# ---------------------------------------------------------------------------
# Mechanism records: what a cell should actually store.
#
# A cell holding a parameter vector ("tile=128x128, slices=8") can only ever be
# mutated into another point of the space that vector lives in. That is what
# the v40..v67 line did for twenty-odd variants, and its ceiling was measured.
# What transferred instead was a triple: a PRECONDITION that is checkable on a
# route the archive has never seen, a MECHANISM that is one named move along
# one axis, and the RESOURCE the precondition proves is safe to spend. Stored
# that way, "deepen the K stage and give up the second LDS buffer" is reusable
# on any starved shape; stored as "tile=32x64, STAGE_K=256" it is reusable on
# nothing.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RouteFacts:
    """The launch-shape facts a precondition is allowed to look at.

    Deliberately small and device-independent given `cu_count`: anything a
    precondition needs must be derivable on the host before the kernel runs,
    or it cannot gate a dispatch decision.
    """
    tiles: int          # output tiles in the plan (grid x * grid y)
    slices: int         # planned K-reduction slices (grid z)
    cu_count: int
    tile_rows: int = 1  # grid y, i.e. how many tile rows the grid has
    ctas_per_cu_cap: int = 1   # occupancy the tile's LDS/VGPR budget allows
    k: int = 0          # reduction extent; 0 means "not supplied"
    m: int = 0          # output rows; 0 means "not supplied"
    n: int = 0          # output cols; 0 means "not supplied"
    cta_m: int = 0      # tile rows; 0 means "not supplied"
    cta_n: int = 0      # tile cols; 0 means "not supplied"
    waves_m: int = 0    # WAVES_M; 0 means "not supplied"
    waves_n: int = 0    # WAVES_N; 0 means "not supplied"
    stage_k: int = 0    # K staged in LDS per barrier round; 0 = not supplied
    frag_edge: int = 16   # MFMA fragment edge; 16x16x16 on both modeled arches
    lds_bytes: int = 0  # the tile's LDS footprint; 0 means "not supplied"
    waves_per_cta: int = 0     # WAVES_M * WAVES_N; 0 means "not supplied"
    simds_per_cu: int = 4
    # Read back from the BUILT object, never estimated from source. 0 means
    # "not measured", which is the honest default: a route whose registers have
    # not been read must fall back to the LDS bound alone.
    vgpr_count: int = 0
    agpr_count: int = 0
    # Which `ctas_per_cu` argument this route's plan site passes to
    # `plan_slices`. 2 is the shipped default and every ordinary call site
    # takes it; the sole exception is the 64x128@128 deepening, which derives
    # 1 from the tile's own LDS. See `planner_slices`.
    planner_ctas_per_cu: int = 2

    @property
    def ctas(self) -> int:
        return self.tiles * self.slices

    @property
    def ctas_by_lds(self) -> int:
        """How many CTAs the LDS budget alone admits. 0 if not supplied.

        LDS is the only occupancy bound that is exact on the host. Registers
        are not: `__launch_bounds__(N, 1)` is a floor on what the compiler must
        fit, not a ceiling on what the hardware runs, and v80 was built on
        mistaking one for the other.
        """
        if self.lds_bytes <= 0:
            return 0
        return _LDS_BYTES_PER_CU // self.lds_bytes

    @property
    def ctas_by_vgpr(self) -> int:
        """How many CTAs the register file admits. 0 if registers unmeasured.

        The counts this reads must come from the built object -- `objmeta.py`
        parses them out of the `.hip_fatbin` bundle -- never from reading the
        source. That distinction is the whole point: `__launch_bounds__(N, 1)`
        is a floor on what the compiler must fit, not a ceiling on what the
        hardware runs, and v80 was built on mistaking one for the other. An
        *estimate* off the source is worse still: the m128 route was carried
        for two stages at an estimated "~148 VGPR against a 170 ceiling", and
        the object says 92 against 256.

        AGPRs count. On gfx90a/gfx942 the file is unified, so an accumulator
        parked in an AGPR costs no memory traffic and is not a spill -- but it
        consumes the same 512, so `vgpr + agpr` is the quantity that gates
        occupancy. Dropping the AGPR term makes four of the eleven measured
        routes read as twice as resident as they are.
        """
        used = self.vgpr_count + self.agpr_count
        if used <= 0 or self.waves_per_cta <= 0:
            return 0
        granule = max(_VGPR_GRANULE,
                      -(-used // _VGPR_GRANULE) * _VGPR_GRANULE)
        waves = min(_MAX_WAVES_PER_SIMD, _VGPR_FILE // granule)
        return waves * self.simds_per_cu // self.waves_per_cta

    @property
    def ctas_by_occupancy(self) -> int:
        """The binding residency bound: LDS, and registers where measured.

        `ctas_per_cu_cap` as supplied by callers has historically been the LDS
        number. Where registers have been read back it can be strictly smaller
        -- `64x128@32` is 4 by LDS and 2 by registers -- so any rule keyed on
        residency, `s_opt = floor(cu_count * cap / tiles)` above all, must use
        this and not `ctas_by_lds`.
        """
        bounds = [b for b in (self.ctas_by_lds, self.ctas_by_vgpr) if b > 0]
        return min(bounds) if bounds else 0

    @property
    def residency_slots(self) -> int:
        """CTAs the machine holds at once, at the binding occupancy bound."""
        return self.cu_count * self.ctas_by_occupancy

    @property
    def grid_asks_per_cu(self) -> int:
        """CTAs each CU is actually asked to hold: `ceil(ctas / cu_count)`.

        Distinct from `ctas_by_occupancy`, which is what the CU *could* hold,
        and the difference between the two is a resource. A grid of 512 CTAs on
        304 CUs asks for 2 -- 208 CUs take two and 96 take one -- so a route
        compiled to fit 3 has a slot per CU that nothing will ever occupy.

        This is a ceiling, not the mean 1.684, and deliberately so: makespan is
        set by the CU holding the most CTAs, not by the average one. The same
        rounding is the grid-quantisation waste term, `ceil(tiles/cu)*cu/tiles`,
        measured at 18.7% on that grid and shown to be uncollectible by tile
        choice.
        """
        if self.cu_count <= 0 or self.ctas <= 0:
            return 0
        return -(-self.ctas // self.cu_count)

    @property
    def grid_utilisation(self) -> float:
        """Fraction of the launched rounds that is real work.

        `ctas / (cu_count * grid_asks_per_cu)`. 0.0 when undetermined.

        **DO NOT TREAT THE RECIPROCAL AS RECOVERABLE TIME (finding 23).** It
        reads like a waste factor and it is not one. Raising it was measured
        twice, on the two worst-utilisation routes in the suite, and returned
        nothing:

            prefill_m512_up   split-K s=1 -> 2   util 0.754 -> 0.905   -21.4%
            prefill_m256_down split-K 12 -> 16   util 0.684 -> 0.912   -11.3%

        On the m256 row the model demanded -29.1 us and the extra reduction
        only accounts for 8.4 us of the 13.2 us actually lost, so the balance
        term returned at most a few microseconds against a predicted 25%. The
        makespan-is-set-by-the-busiest-CU argument assumes a throughput-bound
        CU; these are not, so the third CTA fills issue slots the other two
        were leaving idle. Use `rounds` instead (finding 24): the residency term
        that costs is whether the grid fits in one pass at all, not how evenly
        a grid that does fit is spread. See `tile_mutation_verdict` for why a
        rule keyed on this field still works despite the mechanism being wrong.
        """
        if self.cu_count <= 0 or self.ctas <= 0:
            return 0.0
        return self.ctas / (self.cu_count * self.grid_asks_per_cu)

    @property
    def rounds(self) -> int:
        """Sequential passes the grid needs: `ceil(ctas / residency_slots)`.
        0 when undetermined.

        THIS IS THE RESIDENCY TERM THAT SURVIVED (finding 24), and it replaces
        `occupancy_fill`, which was removed for having the wrong sign. Both
        configurations ever measured that crossed `rounds` 1 -> 2 lost
        catastrophically, and nothing that stayed at 1 lost more than a
        separately-priced named term explains:

            m2048 sk32 double  512 ctas, cap 3 -> 1   rounds 1 -> 2   -26.7%
            m512  sk32 double  688 ctas, cap 3 -> 2   rounds 1 -> 2   -43.4%

        The mechanism is not tail imbalance -- finding (23) measured that and
        it is free, because a third co-resident CTA fills issue slots the other
        two leave idle. It is serialisation: CTAs that do not fit run in a
        whole additional pass, and during that pass most CUs sit empty. m512's
        second round holds 80 CTAs on 304 CUs and still costs a full
        CTA-duration of makespan.

        Every shipped route in the suite is at `rounds == 1`. Treat leaving 1
        as disqualifying, not as a cost to weigh.
        """
        slots = self.residency_slots
        if slots <= 0 or self.ctas <= 0:
            return 0
        return -(-self.ctas // slots)

    @property
    def round_slack(self) -> int:
        """CTAs that could still be added before `rounds` increments. 0 when
        undetermined or already exactly on a boundary.

        This is the headroom a residency-affecting mutation gets to spend.
        `prefill_m64_square` is the cautionary route: 576 CTAs against 608
        slots is 94.7% of a round, so its slack is 32 and any mutation that
        costs it one CTA of occupancy doubles its makespan. Occupancy-frozen.
        """
        slots = self.residency_slots
        if slots <= 0 or self.ctas <= 0:
            return 0
        return max(0, self.rounds * slots - self.ctas)

    @property
    def padding_factor(self) -> float:
        """Launched tile area divided by real area: the cost of a tile edge
        that does not divide the shape. 1.0 when it divides exactly, 0.0 when
        the shape is undetermined.
        """
        if self.m <= 0 or self.n <= 0 or self.cta_m <= 0 or self.cta_n <= 0:
            return 0.0
        mt = -(-self.m // self.cta_m)
        nt = -(-self.n // self.cta_n)
        return (mt * self.cta_m) * (nt * self.cta_n) / (self.m * self.n)

    @property
    def grid_efficiency(self) -> float:
        """`grid_utilisation / padding_factor` -- the fraction of the machine's
        launched capacity that does real work.

        Utilisation alone is not the figure of merit for a tile choice, and
        using it alone is what made the first version of finding (18) wrong. A
        tile can reach utilisation 1.000 by having an edge that divides the CU
        count nicely while badly mis-dividing the shape, and then pays the
        difference back in padded rows or columns nobody needs. Only the ratio
        is comparable across tiles.
        """
        if self.padding_factor <= 0.0:
            return 0.0
        return self.grid_utilisation / self.padding_factor

    def legal_tiles(self, *, waves_per_cta: int = 0,
                    stage_k: int = 0) -> list[tuple[int, int, int, int]]:
        """Every `(cta_m, cta_n, waves_m, waves_n)` this kernel can instantiate
        for the route's shape at the given wave count, subject to LDS holding
        as many CTAs as the resulting grid actually asks each CU to hold.

        A tile edge must be `16 * waves * frags` because 16 is the MFMA
        fragment edge -- which admits 112, 160 and 224 just as much as it
        admits 128, and that is the whole reason this enumeration exists
        rather than a closed form.
        """
        w = waves_per_cta or self.waves_per_cta
        sk = stage_k or self.stage_k
        if self.m <= 0 or self.n <= 0 or w <= 0 or sk <= 0 or self.cu_count <= 0:
            return []
        out: list[tuple[int, int, int, int]] = []
        for wm in (1, 2, 4, 8):
            wn, rem = divmod(w, wm)
            if rem or wn not in (1, 2, 4, 8):
                continue
            for fm in range(1, 17):
                for fn in range(1, 17):
                    tm, tn = 16 * wm * fm, 16 * wn * fn
                    if tm > 256 or tn > 256:
                        continue
                    tiles = (-(-self.m // tm)) * (-(-self.n // tn))
                    ctas = -(-tiles // self.cu_count)
                    if 2 * (tm + tn) * sk * ctas > _LDS_BYTES_PER_CU:
                        continue
                    out.append((tm, tn, wm, wn))
        return out

    @property
    def occupancy_waves_per_simd(self) -> float:
        """Waves per SIMD the route ACTUALLY runs at: the grid's ask and the
        LDS capacity, whichever binds, times waves per CTA over four SIMDs.

        Distinct from `waves_per_simd`, which is capacity alone. Using capacity
        here would be a bug in the permissive direction -- two tiles with room
        for eight waves each look identical even when one of them launches a
        grid that only ever asks for one CTA per CU. LDS is the bound used
        rather than registers because an unbuilt tile has no register count;
        see `wave_grid_occupancy_gain` for why that number costs a build.
        """
        if self.waves_per_cta <= 0:
            return 0.0
        asks = self.grid_asks_per_cu
        cap = self.ctas_by_lds
        if asks <= 0 or cap <= 0 or self.simds_per_cu <= 0:
            return 0.0
        return min(asks, cap) * self.waves_per_cta / self.simds_per_cu

    @property
    def ctas_per_cu(self) -> int:
        """Co-resident CTAs per CU: the grid's ask and the LDS capacity,
        whichever binds. 0 if either is undetermined.

        This -- not `occupancy_waves_per_simd` -- is the occupancy variable that
        predicts this route, and finding (19) is the measurement that says so.
        Two CTAs of four waves and one CTA of eight waves are the SAME
        `occupancy_waves_per_simd`, put the SAME eight waves on the CU, and have
        the SAME makespan in work units; on the machine they are 26.7% apart
        (runs 1549-1551 against 1561-1570). Pricing the CTAs-per-CU axis with a
        waves-per-CTA measurement is exactly the error that produced v102.

        THE VARIABLE IS MEASURED; THE MECHANISM IS NOT KNOWN. The first
        explanation was barrier coverage -- `__syncthreads()` is per-CTA, this
        route runs 128 stages, the kernel is single-buffered at every tile that
        matters, so 256 barriers go uncovered when a CTA is alone on its CU.
        That was tested and REFUTED: v104 double-buffered the 256x112 tile,
        which costs nothing because the grid already pinned it to one CTA,
        halving the barriers to 128 and decoupling the ds_write from the
        ds_reads. It recovered 0.5% of the 26.7% (0.24420 against 0.24560).
        Register pressure is also ruled out -- objmeta reports 126 VGPR and
        zero spills on the instantiation.

        So keep the guard, which is empirical and holds, and do not trust any
        story about why until it is profiled. The live candidates are global
        latency coverage and L2 locality (the XCD swizzle gives the 8x37 grid
        21.4 MB per XCD against the 16x37 grid's 17.6 MB), and separating them
        needs counters rather than another build.
        """
        asks = self.grid_asks_per_cu
        cap = self.ctas_by_lds
        if asks <= 0 or cap <= 0:
            return 0
        return min(asks, cap)

    @property
    def operand_bytes_per_output(self) -> float:
        """Operand elements loaded from global per output element, `(cta_m +
        cta_n) / (cta_m * cta_n)`. 0.0 if the tile is not supplied.

        Surface-to-volume. A CTA reads `(cta_m + cta_n) * k` operand elements to
        produce `cta_m * cta_n` outputs, so for a fixed perimeter budget the
        biggest square tile moves the least, and any tile smaller in area than
        the incumbent moves strictly more per output.

        Finding (19) is why this is priced at all. It is the term that makes the
        grid axis unreachable on a power-of-two shape: buying the factor 19 the
        machine has and the shape does not requires an edge of 112 or 224, both
        of which shrink the tile below 128x128's 16384, and the traffic tax on
        the smaller tile arrives before the grid gain does. Measured on this
        route: +7.1% traffic cost +4.45% time, +28.6% cost +45.8%.
        """
        if self.cta_m <= 0 or self.cta_n <= 0:
            return 0.0
        return (self.cta_m + self.cta_n) / (self.cta_m * self.cta_n)

    def xcd_panel_balance(self, group_m: int = 8, xcds: int = 8) -> float:
        """How lopsided one XCD's rectangle of operand panels is, as the ratio
        of its larger side in bytes to its smaller. 1.0 is square; bigger is
        worse; 0.0 if the tile or shape is not supplied.

        This is the *mechanism* behind finding (19)'s co-residency term, and it
        is priced separately because `ctas_per_cu` was only ever a proxy for it.
        v59's grouped rasterisation hands XCD x a `group_m`-tall by
        `chunk / group_m`-wide rectangle of tiles, which is `group_m` A panels
        against `chunk / group_m` B panels; the L2 that has to hold them is 4
        MiB per XCD, so what matters is not the rectangle's area but whether one
        operand crowds out the other. v59 argued exactly this and then hardcoded
        `kGroupM = 8`, which degenerates whenever `tiles_m <= group_m`: the group
        covers the whole grid, `first_m` pins at 0, and the XCD owns every A
        panel against a sliver of B.

        Measured, `prefill_m2048_square` at `256x112` (finding 19d), L2 hit rate
        against this ratio at fixed everything else -- group 8 gives 20.4 MiB at
        4.0:1 and 76.0%, group 4 gives 16.1 MiB at 1.01:1 and 86.8%, group 2
        gives 20.2 MiB at 4.6:1 and 79.5%. Area does not order those; balance
        does. On `prefill_m1024_down` the same sweep moves hit rate only 85.3 ->
        86.3 because that route ships `slices = 4` and the swizzle ignores
        `blockIdx.z`, so its panels are already four-way shared -- which is why
        this term must be read together with `slices` and not alone.
        """
        if self.cta_m <= 0 or self.cta_n <= 0 or self.m <= 0 or self.n <= 0:
            return 0.0
        if group_m <= 0 or xcds <= 0:
            return 0.0
        tiles_m = -(-self.m // self.cta_m)
        tiles_n = -(-self.n // self.cta_n)
        chunk = (tiles_m * tiles_n) // xcds
        if chunk <= 0:
            return 0.0
        # The group can never be taller than the grid; this clamp is the kernel's
        # own `min(kGroupM, tiles_m - first_m)` and it is why m512 (tiles_m = 4)
        # never saw the degeneracy that m1024 (tiles_m = 8) did.
        rows = min(group_m, tiles_m)
        cols = max(1, chunk // rows)
        a_bytes = rows * self.cta_m
        b_bytes = cols * self.cta_n
        hi, lo = max(a_bytes, b_bytes), min(a_bytes, b_bytes)
        return hi / lo

    def best_group_m(self, xcds: int = 8) -> int:
        """The group height in {1, 2, 4, 8} that squares the XCD rectangle.

        Returns 8 -- v59's shipped constant -- when the tile or shape is not
        supplied, so this is never a silent behaviour change. It picked 4 for
        `256x112` and 4 for `prefill_m1024_down`'s `128x160`, and the counters
        agreed with it on both; on `128x128` and on m512 it returns what is
        already shipped. Predicting the winner is not the same as the winner
        being worth shipping -- see finding (19d), where it was worth 45% of the
        misses on one route and 0.55% of cycles on the other.
        """
        if self.cta_m <= 0 or self.cta_n <= 0 or self.m <= 0 or self.n <= 0:
            return 8
        scored = [(self.xcd_panel_balance(g, xcds), g) for g in (8, 4, 2, 1)]
        scored = [s for s in scored if s[0] > 0.0]
        if not scored:
            return 8
        return min(scored)[1]

    @property
    def tile_axis_can_collect_grid_waste(self) -> bool:
        """Whether any legal tile raises `grid_efficiency` without paying for
        it in arithmetic intensity, in co-residency, or in operand traffic.

        Finding (18), CORRECTED. The original argument ran: 304 = 2**4 * 19,
        every shape here has power-of-two M and N, and 2 is a primitive root
        mod 19, so a power-of-two tile count never lands on a multiple of 304 --
        and since doubling the count doubles the residue too, `T`, `2T` and
        `4T` share a utilisation exactly. All of that is true, and it is why
        128x128, 128x64, 64x128 and 64x64 all read 0.8421 on
        `prefill_m1024_down`, and why 1.1875 recurred across every tile on
        `prefill_m2048_square`.

        THE PREMISE WAS TOO NARROW. A tile edge only has to be a multiple of
        `16 * waves`, so 112, 160 and 224 are all legal -- this kernel already
        ships a 160 -- and those edges are NOT powers of two. 4096/224 needs 19
        tiles, so 64x224 puts exactly 304 tiles on `prefill_m1024_down` and
        exactly 608 on `prefill_m2048_square`: utilisation 1.0000, the thing
        the original argument said was unreachable. The conclusion survived,
        but not for the stated reason, and a proof with a false premise is not
        a proof.

        THE REAL REASON is that utilisation is the wrong figure of merit.
        Landing on a multiple of 304 requires a factor of 19 in the tile count,
        which on power-of-two shapes has to come from an edge that mis-divides
        the shape -- 4096 = 18.29 * 224, so the 19th column is 4/5 padding.
        Netting that out, 64x224 has `grid_efficiency` 0.9624, not 1.0. And
        0.9624 is exactly what the crude 32x64 option scored, because both are
        the same 19 showing up in different places.

        So the test enumerates the legal tiles and asks for a move that raises
        efficiency while giving up nothing else. The guards were then WRONG A
        SECOND TIME, and finding (19) is the measurement that corrected them:
        the search was run with `occupancy_waves_per_simd` and no traffic term,
        it named `256x112/8x1` on `prefill_m2048_square` as the closest miss
        (+14.3% efficiency, +16.7% intensity, refused only on waves/SIMD), and
        that tile was built as v102 and measured **32.3% worse** against a
        prediction of -2% to +6%. Both missing terms are now priced --
        `ctas_per_cu` and `operand_bytes_per_output` -- and each has its own
        docstring for why.

        WITH THEM PRICED, the axis closes by measurement rather than by search.
        Every tile reaching utilisation >= 0.99 on `prefill_m2048_square` is
        either big enough to have good surface-to-volume and therefore lands at
        304-ish tiles and one CTA per CU (128x224, 112x256, 256x112 -- the
        barrier cliff, measured +32.3%), or holds two CTAs per CU and is
        therefore smaller than 16384 in area and pays traffic (112x128 and
        128x112 at +7.1%, measured +4.45%; 64x224 at +28.6%, measured +45.8%).
        There is no third option, because 19 is prime: the count reaches a
        multiple of 304 only if one of the two tile-edge divisions supplies the
        whole factor, and on 2048x4096 that forces an edge of 112 or 224.

        That is a trilemma among grid, co-residency and traffic, and no tile
        escapes it. Only a decomposition that stops choosing tile *counts* --
        stream-K -- reaches the term, and it reaches it precisely because it
        keeps the 128x128 tile's traffic and co-residency and fills the idle
        CUs with partial-K work instead of with a different tile.
        """
        if self.cu_count <= 0 or self.ctas <= 0 or self.intensity <= 0:
            return False
        here_eff = self.grid_efficiency
        here_int = self.intensity
        here_ctas = self.ctas_per_cu
        here_traffic = self.operand_bytes_per_output
        if here_eff <= 0.0 or here_ctas <= 0 or here_traffic <= 0.0:
            return False
        for tm, tn, wm, wn in self.legal_tiles():
            cand = _replace(
                self, cta_m=tm, cta_n=tn, waves_m=wm, waves_n=wn,
                tiles=(-(-self.m // tm)) * (-(-self.n // tn)))
            # Beat the incumbent by more than the ~1.5% run-to-run drift of the
            # suite geomean, and give up nothing on any of the other three.
            if (cand.grid_efficiency > here_eff * 1.015
                    and cand.intensity >= here_int
                    and cand.ctas_per_cu >= here_ctas
                    and cand.operand_bytes_per_output <= here_traffic):
                return True
        return False

    @property
    def deepest_stage_k_in_lds(self) -> int:
        """The deepest single-buffered stage this tile's LDS admits, rounded
        down to a power of two. 0 if the tile is not supplied.

        The stage panel is `(cta_m + cta_n) * (stage_k + pad) * 2` bytes; the
        pad is 4 on the measured kernel and is included so this is an
        achievable depth rather than an optimistic one. The power-of-two floor
        is not cosmetic -- see `deepen_underfilled_grid`'s evidence, where a
        non-power-of-two stage passed every static check and then blew the
        register file.

        This exists so the *size* of the depth prize is host-checkable before
        anything is built. `deepest_stage_k_in_lds / stage_k` is the cover
        multiplier a full deepening would buy, and finding (15) prices the
        move in exactly that currency.
        """
        if self.cta_m <= 0 or self.cta_n <= 0:
            return 0
        per_k = (self.cta_m + self.cta_n) * 2
        if per_k <= 0:
            return 0
        depth = _LDS_BYTES_PER_CU // per_k - 4   # the +4 stride pad
        if depth < 1:
            return 0
        p = 1
        while p * 2 <= depth:
            p *= 2
        return p

    def stage_k_for_double_buffer(self, ctas: int = 3) -> int:
        """Deepest stage, power-of-two, that still double-buffers at `ctas`
        CTAs/CU. 0 if the tile is not supplied or no depth fits.

        Same panel arithmetic as `deepest_stage_k_in_lds`, with the factor of
        two the second buffer costs and the co-residency the caller refuses to
        give up: `2 * (cta_m + cta_n) * (stage_k + 4) * 2 * ctas <= 65536`.
        This mirrors the shipped kernel's own compile-time gate, which reads
        `kDoubleBuffer = 2 * kPanelBytes * 3 <= 65536` -- so the host can
        predict whether a build will double-buffer without building it.
        """
        if self.cta_m <= 0 or self.cta_n <= 0 or ctas <= 0:
            return 0
        per_k = 2 * (self.cta_m + self.cta_n) * 2 * ctas
        if per_k <= 0:
            return 0
        depth = _LDS_BYTES_PER_CU // per_k - 4   # the +4 stride pad
        if depth < 1:
            return 0
        p = 1
        while p * 2 <= depth:
            p *= 2
        return p

    def lds_budget_corner(self, ctas: int = 3) -> str:
        """Which two of {cover, overlap, co-residency} this route can hold.

        THIS IS A MECHANISM, NOT A PARAMETER. Within a fixed LDS budget at a
        fixed tile you can buy at most two of: a deep stage (latency cover for
        the global load), a second LDS buffer (so the `ds_write` for stage s+1
        overlaps the MFMA of stage s instead of sitting between two barriers),
        and enough CTAs per CU to cover what is left exposed. It is a resource
        trade, so the search should price it once per tile and never re-derive
        it by sweeping stage_k per route.

        Returns one of "cover+coresidency", "overlap+coresidency",
        "cover+overlap", or "" when the tile is not supplied.

        Measured exhaustively on `prefill_m2048_square` at 128x128/2x4, gfx942,
        64 KB LDS, wanting 3 CTAs/CU (finding 20g, runs 1620-1623 and the
        finding 19 arithmetic):

            sk32 single, 3 CTAs   0.18390 ms   SHIPPED, best
            sk32 double, 1 CTA    -26.7%       co-residency cliff
            sk16 double, 3 CTAs   -34.3%       cover halved 16 -> 8 MFMA
            sk48/sk64 single      lost         v97, v99

        Every corner other than the shipped one is worse, so the exposed
        barrier window that finding (20c) read out of the ISA is real, is
        correctly diagnosed, and is NOT AFFORDABLE. That is a stronger claim
        than the axis being closed, and it is the reason a mutation proposing
        "double-buffer it" or "shrink the stage" on a route already at
        cover+coresidency should be refused by the archive rather than built.
        """
        if self.cta_m <= 0 or self.cta_n <= 0 or self.stage_k <= 0:
            return ""
        if self.stage_k <= self.stage_k_for_double_buffer(ctas):
            return "overlap+coresidency"
        if self.ctas_per_cu_cap >= ctas:
            return "cover+coresidency"
        return "cover+overlap"

    def cover_cost_of_double_buffering(self, ctas: int = 3) -> float:
        """What fraction of `prefetch_cover` a second buffer would cost, if
        co-residency is held at `ctas`. 0.0 if not supplied or nothing to pay.

        1.0 means the whole stage would have to go; 0.5 means half of it. This
        is the honest price tag on "just double-buffer it": on the shipped
        128x128 tile at sk32 wanting 3 CTAs it returns 0.5, and that half-stage
        measured -34.3%, which is far more than the overlap returns. Use it to
        refuse the mutation on the host, before a build.
        """
        if self.stage_k <= 0:
            return 0.0
        fits = self.stage_k_for_double_buffer(ctas)
        if fits <= 0:
            return 1.0
        if fits >= self.stage_k:
            return 0.0
        return 1.0 - fits / self.stage_k

    def tile_mutation_verdict(self, other: "RouteFacts") -> str:
        """Should a tile mutation from this route to `other` be built at all?
        `""` means allow; anything else is the refusal and its reason.

        THIS IS A MECHANISM, NOT A PARAMETER, AND IT DELIBERATELY FITS NO
        CONSTANT. `grid_utilisation` and `operand_bytes_per_output` move in
        OPPOSITE directions under any tile change -- a bigger tile always cuts
        operand traffic and always coarsens the grid -- so a mutation scored on
        either field alone is a coin flip. Both directions have now been
        measured, on two routes, through the same forced-tile hook with the
        shipped tile serving as its own control:

            prefill_m2048_square   util 0.842, traffic 0.0156
              tile HALVED   traffic 0.0156 -> 0.0234 (+50%)     -26 to -27%
              (runs 1610-1615, finding 20f)

            prefill_m512_up        util 0.754, traffic 0.0234
              tile GROWN    util 0.754 -> 0.566 (-25%)
                            traffic 0.0234 -> 0.0156 (-33%)     -8.8%
              (runs 1630-1633, finding 21b)

        Note what the second row rules out: a 33% traffic WIN did not pay for a
        25% utilisation loss. So the two terms are not interchangeable and no
        weighted sum of them is justified by two points -- an earlier draft of
        this method used a utilisation floor and had to be withdrawn, because
        both measured routes sit below any floor that would separate them
        (0.842 and 0.754). What the data does support is weaker and sound: on
        both routes, in both directions, every tile change that degraded either
        term lost, and none improved both. So refuse unless the mutation is a
        Pareto improvement, and let a measurement -- not a curve -- be what
        ever relaxes this.

        CORRECTION (finding 23): this rule survives, its rationale does not.
        `grid_utilisation` measures tail balance, and tail balance was
        subsequently measured to be worth nothing -- raising it on the two
        worst-balanced routes lost 21.4% and 11.3%. The reason the rule still
        predicts correctly is that on BOTH measured tile mutations, falling
        `grid_utilisation` came with falling residency headroom, which is a real
        latency term (m512's grown tile left most CUs holding one CTA, 38% of
        cap). So this is a correct predictor riding a correlated variable. It
        is kept because it is measured and flagged because it is not
        mechanistic: a route where the two come apart will break it, and that
        route is worth going looking for.

        A route where no tile is a Pareto improvement has its tile axis closed.
        On `prefill_m512_up` that is permanent -- 304 = 2^4 * 19, `n = 11008 =
        2^8 * 43`, `m = 512 = 2^9`, so every reachable tile count is `43 * 2^a`
        and none is divisible by 19 (finding 21c). An earlier draft of this
        docstring concluded from that that route-gated stream-K was the right
        mutation. It is NOT: see `slice_mutation_verdict` and finding (23).
        Stream-K's only advantage over split-K is reaching a balanced grid more
        cheaply, and split-K has now reached that balance twice for no gain.
        """
        if self.cta_m <= 0 or self.cta_n <= 0 or self.tiles <= 0:
            return ""
        if other.cta_m <= 0 or other.cta_n <= 0 or other.tiles <= 0:
            return ""
        if (other.cta_m, other.cta_n) == (self.cta_m, self.cta_n):
            # Not a tile mutation, so this rule has no opinion. Without this
            # the "no change to either term" arm below fires on every mutation
            # of some *other* axis -- it did exactly that when the rules were
            # first composed in `mutation_verdict`, refusing the m512 double
            # buffer twice, once for a correct reason and once for a reason
            # about a tile nobody had proposed changing. A refusal gate that
            # cries wolf gets switched off, so each rule must abstain outside
            # its own axis.
            return ""
        if other.grid_utilisation < self.grid_utilisation:
            return "refuse: degrades grid_utilisation"
        if other.operand_bytes_per_output > self.operand_bytes_per_output:
            return "refuse: degrades operand_bytes_per_output"
        if (other.grid_utilisation == self.grid_utilisation
                and other.operand_bytes_per_output
                == self.operand_bytes_per_output):
            return "refuse: no change to either term"
        return ""

    def slice_mutation_verdict(self, other: "RouteFacts") -> str:
        """Should a change to the slice count be built? `""` allows.

        Refuses any increase in `slices`, because every measured one lost:

            prefill_m512_up    1 -> 2   -21.4%    1 -> 3   -7.7%
            prefill_m256_down 12 -> 14   -9.7%   12 -> 16  -11.3%

        Runs 1640-1645 and 1651-1656, both palindromic, both with the hook
        verified live in the plan trace and the control reproducing the shipped
        number (to 1.9% and 0.07% respectively). These four points span the two
        regimes that could have differed -- m512 turns a reduction ON from
        nothing, m256 adds a third to one already running -- and both lost, so
        the loss is not an artifact of the first slice being unusually
        expensive.

        More slices is the canonical "improve grid_utilisation" mutation, and
        finding (23) is that utilisation is not a recoverable cost. Every extra
        slice writes another full `m x n` FP32 partial plane and reads it back;
        that traffic is real and the balance it buys is not. The shipped
        `plan_slices` plus its runtime tuner already sit at or below the
        optimum on both routes.

        DECREASES are allowed through, not because one has been measured to win
        but because none has been measured at all -- the tuner searches
        downward already, so an archive proposing one is duplicating the tuner
        rather than proposing something new.
        """
        if self.slices <= 0 or other.slices <= 0:
            return ""
        if other.slices > self.slices:
            return "refuse: raises slices, and grid_utilisation is not a cost"
        return ""

    def residency_mutation_verdict(self, other: "RouteFacts") -> str:
        """Should a mutation that changes CTA residency be built? `""` allows.

        This is the strongest refusal in the descriptor, because the term it
        guards produced the two largest losses in the ledger (finding 24):

            m2048 sk32 double  cap 3 -> 1   rounds 1 -> 2   -26.7%
            m512  sk32 double  cap 3 -> 2   rounds 1 -> 2   -43.4%

        Any mutation that raises `rounds` is refused outright. A grid that does
        not fit in one residency pass serialises, and the second pass costs a
        whole CTA-duration of makespan no matter how few CTAs are in it -- m512
        put 80 CTAs on 304 CUs in its second round and paid 43%.

        Note this is NOT the tail-balance term finding (23) refuted. (23)
        measured imbalance among CTAs that were all simultaneously resident and
        found it free. Crossing a round boundary means they are not all
        resident. The two are opposite facts about the same hardware, and
        conflating them is what caused the v107 build; see finding (24e).

        A mutation that keeps `rounds` but eats more than half of
        `round_slack` gets a warning rather than a refusal: it has not crossed
        the boundary but it has spent the headroom that protects the route from
        the next mutation. `prefill_m64_square` (slack 32 of 608) is already
        there and should be treated as occupancy-frozen.
        """
        if self.rounds <= 0 or other.rounds <= 0:
            return ""
        if other.rounds > self.rounds:
            return (
                "refuse: raises rounds %d -> %d; the grid no longer fits in "
                "one residency pass" % (self.rounds, other.rounds)
            )
        if other.rounds < self.rounds:
            return ""
        if self.round_slack > 0 and other.round_slack * 2 < self.round_slack:
            return (
                "warn: consumes round_slack %d -> %d, leaving the route one "
                "small mutation from a second round"
                % (self.round_slack, other.round_slack)
            )
        return ""

    def mutation_verdict(self, other: "RouteFacts") -> dict[str, object]:
        """THE ENTRY POINT. Every measured refusal, applied to one proposed
        mutation, as `{"allow": bool, "refusals": [...], "warnings": [...]}`.

        This exists because the individual `*_mutation_verdict` rules had zero
        callers outside their own tests. Findings (19) through (24) each closed
        an axis by measurement and each was encoded here, and none of it was
        reachable by the search: `qd_v2.py` is handed to agents as
        `QD_EVIDENCE_HELPER` but the only subcommand the lane names is
        `hash-tree`, and no role prompt mentions residency or the descriptor at
        all. So the workflow re-proposed refuted mutations every run. That is
        not an iteration-count problem and no amount of budget fixes it; it is
        a missing edge between what was measured and what the search can read.

        Refusals compose by union, not by vote. Each rule was established by an
        independent measurement on a different axis, so a mutation that trips
        two of them is worse, not ambiguous. A warning does not block.

        Callers should treat `allow == False` as "do not build this" and spend
        the budget elsewhere. Reaching for it via the CLI:

            python3 qd_v2.py mutation-verdict --current <json> --candidate <json>
        """
        if not isinstance(other, RouteFacts):
            return {"allow": True, "refusals": [], "warnings": []}
        refusals: list[str] = []
        if other == self:
            # A proposal identical to the incumbent spends a build and a verify
            # to re-measure noise. `tile_mutation_verdict` used to catch this,
            # but it had to stop -- it was firing on mutations of every *other*
            # axis too -- so the check moves up here, where "nothing changed"
            # is actually knowable. Refusing outright rather than warning:
            # there is no reading under which building it is the right call.
            return {"allow": False,
                    "refusals": ["mutation_verdict: no change on any axis"],
                    "warnings": []}
        warnings: list[str] = []
        for rule in (self.residency_mutation_verdict,
                     self.tile_mutation_verdict,
                     self.slice_mutation_verdict):
            verdict = rule(other)
            if not verdict:
                continue
            if verdict.startswith("warn"):
                warnings.append("%s: %s" % (rule.__name__, verdict))
            else:
                refusals.append("%s: %s" % (rule.__name__, verdict))
        return {"allow": not refusals,
                "refusals": refusals,
                "warnings": warnings}

    @property
    def planner_slices(self) -> int:
        """What the shipped `plan_slices` would choose. 0 if `k` not supplied.

        A faithful re-implementation of the kernel's host planner, kept here so
        the descriptor can say *how wrong* the shipped analytic plan is on a
        route rather than merely what the right answer would be. It mirrors
        `plan_slices(tiles, k, cu_count, ctas_per_cu = 2)`, and the default is
        the point: **every ordinary call site takes it** -- lines 1138, 1157,
        1289, 1320, 1797, 1825.

        Exactly one site passes anything else, the 64x128@128 deepening at line
        1267, and it is worth being precise about because it looks like a
        general fix and is not. That site derives its argument from the tile it
        is about to launch, `65536 / ((64+128)*(128+4)*2)` = 1, which selects
        the `else` arm -- floor to a single CTA wave, `cu_count / tiles`. So the
        v75 tail-wave correction exists in the code, is *right*, and reaches one
        hard-coded tile. `planner_ctas_per_cu` records which arm a route takes;
        `prefill_m128_square` is the only traced route on the 1 arm, where the
        planner returns 4 instead of the 8 the default arm would give.

        Note what is absent from both arms: there is no term for
        `ctas_by_occupancy`. The default arm targets 1.5 CTAs per **CU** whether
        the route holds one CTA or four, and the one-wave arm assumes the
        residency it was written for rather than reading it. Finding (18).
        """
        if self.k <= 0 or self.cu_count <= 0 or self.tiles <= 0:
            return 0
        if self.tiles >= 2 * self.cu_count:
            return 1
        if self.planner_ctas_per_cu >= 2:
            target = self.cu_count + self.cu_count // 2
            slices = -(-target // self.tiles)      # ceil
        else:
            slices = max(1, self.cu_count // self.tiles)   # floor: one wave
        slices = min(slices, 16)
        while slices > 1 and self.k // slices < 256:
            slices -= 1
        return slices

    @property
    def fill_limited_slices(self) -> int:
        """`floor(cu_count * ctas_by_occupancy / tiles)`. 0 if unmeasured.

        The residency term the shipped planner is missing. Finding (18) scored
        it against what the cold autotune actually chooses on five traced
        routes and it is exact on four, including both routes where the shipped
        planner is wrong -- `decode_m96_up` most sharply, where the planner asks
        for 6 against a one-CTA-per-CU route and forcing that answer instead of
        the formula's 3 costs **+19.9%** (0.06700 -> 0.08032 ms, medians of
        three, fully rank-separated, runs 1416-1421).

        It is an **upper bound, not the answer**, and the fifth route says so.
        On `prefill_m256_down` (`128x160/2x2/sk32`, 3 CTAs/CU, 52 tiles) this
        returns 17 and the measured optimum is 12; forcing 17 costs +6.7%
        (runs 1410-1415) while forcing the shipped planner's 9 costs +9.3%
        (runs 1430-1435). The two analytic answers straddle the optimum: the
        planner under-splits by 25%, this term over-splits by 42%.

        The missing half is the reduction plane, `2 * 4 * m * n * slices`, which
        charges for slices the fill term hands out for free -- 100.7 MB at s=12
        against 142.6 MB at s=17 here. It is deliberately **not** modelled. Two
        obvious closed forms were checked against this point and both are
        refuted: 142.6 MB is under the 256 MB MALL, so a last-level-fit
        threshold did not fire, and 17 slices is 2.91 CU-waves of CTAs, under 3,
        so a tail-wave argument did not fire either. Finding (17) is the reason
        not to guess a third: on a different route, deleting the plane outright
        did not pay, so the plane's sign is not even stable across routes. Until
        an arm exists that moves the plane at fixed tile and fixed occupancy,
        this stays a bound with a measured failure case attached.
        """
        slots = self.residency_slots
        if slots <= 0 or self.tiles <= 0:
            return 0
        return max(1, slots // self.tiles)

    @property
    def waves_per_simd(self) -> float:
        """Waves per SIMD as the LDS budget forces them. 0.0 if undetermined.

        Deliberately computed from `ctas_by_lds` alone -- not from
        `ctas_per_cu_cap`, and not from `ctas_by_occupancy` either, even now
        that registers can be measured. Folding the register bound in would
        let a route that merely *may* run one CTA read as one that *must*,
        which is precisely the v80 error, and it would silently change the
        answer for a route the moment somebody got round to building it.
        Reading LDS alone makes this an upper bound on residency and therefore
        an upper bound on waves per SIMD. The register bound can only lower
        residency, so it can only lower the true figure below this one, and a
        precondition written as `== 1.0` therefore fires only where one wave
        per SIMD is forced by LDS alone -- never where it is merely possible.
        The gate stays conservative in the direction that matters.

        (The docstring previously argued this was a *lower* bound and that the
        register bound could only raise the number. That has the sign
        backwards. The conclusion -- that `== 1.0` never over-fires -- is
        unaffected, but the cost of the error is real and is now visible:
        see `waves_per_simd_measured`.)
        """
        if self.waves_per_cta <= 0 or self.ctas_by_lds <= 0:
            return 0.0
        return self.ctas_by_lds * self.waves_per_cta / self.simds_per_cu

    def wave_grid_occupancy_gain(self, widened: "RouteFacts") -> float:
        """How much `widened`'s waves-per-SIMD beat this route's. 0.0 if either
        side has unread registers.

        The prize term for the wave-grid mechanism -- widening `waves_m x
        waves_n` at a fixed tile, so each wave owns fewer accumulators, the
        register footprint per CTA falls, and more waves go resident. The
        mechanism needs a prize term because it has a measured break-even.
        `grid_asks_per_cu < ctas_by_occupancy` -- the surplus slot -- licenses
        *considering* the move, and two routes that both satisfied it came out
        opposite:

          128x128 2x2 -> 2x4, gain x2.00, intensity -33%: **-5.58%** (v98)
          128x160 2x2 -> 4x2, gain x1.33, intensity -36%: **+2.04%** (v100)

        The intensity costs are within three points, so the gain is what
        separates them, and **the break-even is between x1.33 and x2.00**.
        v100 is the strong form of the negative: its object came in at 90 VGPR
        against a predicted 112, no spills, LDS untouched, CTAs landing exactly
        on what the grid asks, waves per SIMD 3.00 -> 4.00 as forecast. Every
        intermediate quantity behaved and the kernel still got slower. So this
        is not a modelling failure to be patched -- it is a real threshold, and
        a route offering less than a doubling should not attempt the move.

        Same shape as the cover prize on the deepening mechanism
        (`deepest_stage_k_in_lds`, break-even between x2 and x4): a surplus is
        necessary for both mechanisms and sufficient for neither.

        **The evidence does not separate two readings of why, and this returns
        the right answer under both.** Occupancy here is capacity, and capacity
        is not what runs: the grid asks for `grid_asks_per_cu` CTAs, which on
        both routes is 2, so *actual* waves per SIMD goes 2.00 -> 4.00 on the
        winner and 2.00 -> 4.00 on the loser as well. Actual occupancy
        discriminates nothing. Capacity does, and the reason it does can be
        read either as "the prize must be a doubling" or as "the widening must
        not spend the surplus slot that licensed it" -- v98 kept 3 CTAs against
        an ask of 2 and v100 fell to exactly 2, leaving no slack to absorb the
        grid's unevenness. On this evidence those are the same statement (the
        ratio is low precisely because capacity fell), and no third route has
        been built that would tell them apart. Recorded, not resolved.

        **This one costs a build, and the cover prize does not.** That
        asymmetry is the point of the signature. Deepening's prize is fixed by
        LDS arithmetic, which is exact on the host. This prize is fixed by
        where the CTA count lands after the widening, and that is a register
        question: v98 kept 3 CTAs and doubled, v100 dropped to 2 and gained a
        third. Registers are never inferred from source here -- see
        `ctas_by_vgpr` and the v80 error -- so the honest form takes the built
        widened route and reads it, rather than estimating and being wrong in
        the direction that approves the build. `wave_grid_gain_ceiling` below
        is the host-side bound, and it is weak.
        """
        if not isinstance(widened, RouteFacts):
            return 0.0
        here = self.waves_per_simd_measured
        there = widened.waves_per_simd_measured
        if here <= 0.0 or there <= 0.0:
            return 0.0
        return there / here

    def wave_grid_gain_ceiling(self, waves_m: int, waves_n: int) -> float:
        """Host-side upper bound on `wave_grid_occupancy_gain` for a proposed
        `waves_m x waves_n`. 0.0 if not computable.

        Widening does not touch LDS -- same tile, same stage, same buffers --
        so `ctas_by_lds` is unchanged and gives the most CTAs the widened route
        could possibly hold. Registers can only take that down. So this is a
        true ceiling, and a route whose *ceiling* is under the x2 break-even
        can be refused before anything is built.

        Stated plainly: **it would not have saved the v100 build.** Both
        measured routes have `ctas_by_lds == 3`, so both read a ceiling of
        exactly 2.00 here -- the one that won and the one that lost. The bound
        only bites where LDS itself caps the widened residency, and neither of
        the two routes in the bracket is that case. It is recorded as a cheap
        negative that has never yet fired, not as the discriminator; the
        discriminator is measured, above.
        """
        target = max(1, waves_m) * max(1, waves_n)
        here = self.waves_per_simd_measured
        if here <= 0.0 or self.ctas_by_lds <= 0 or self.simds_per_cu <= 0:
            return 0.0
        if target <= self.waves_per_cta:
            return 0.0
        best = min(float(_MAX_WAVES_PER_SIMD),
                   self.ctas_by_lds * target / self.simds_per_cu)
        return best / here

    @property
    def fits_in_mall(self) -> bool:
        """True iff this route's split-K workspace fits in the gfx942 MALL.

        Finding (17). A workspace term is only an HBM term if it misses the
        last-level cache, and on gfx942 that cache is 256 MB -- large enough
        that most of this project's split-K planes never reach memory at all.
        Pricing 50 MB of plane traffic at HBM bandwidth said 28.8% of
        `prefill_m256_down`'s runtime was fixup and motivated an atomic-fixup
        build that lost 8.05%. The bytes were real; the cost attributed to them
        was not.

        Returns False when the workspace is unknown (0 slices or no shape), so
        an unmeasured route is never waved through as cache-resident.
        """
        b = self.splitk_workspace_bytes
        return 0 < b <= _MALL_BYTES

    @property
    def waves_per_simd_measured(self) -> float:
        """Waves per SIMD at the binding occupancy bound. 0.0 if registers are
        unread.

        The honest figure, and it needs both bounds. `waves_per_simd` above is
        LDS-only and conservative by construction, which is right for a gate
        that must not over-fire -- but it is an over-estimate on any
        register-bound route, and finding (14) is a statement about how many
        waves are *actually* resident, so it cannot use it. The gap is not
        hypothetical: 96x128/2x2/sk64 reads 2.0 from LDS (30464 B admits two
        CTAs) and objmeta measures 1.00, because 196 VGPR + 72 AGPR admits one.
        Prefer this wherever registers have been read; keep the other for the
        gates that were calibrated on it.
        """
        if self.waves_per_cta <= 0 or self.ctas_by_occupancy <= 0:
            return 0.0
        if self.vgpr_count <= 0 and self.agpr_count <= 0:
            return 0.0
        return self.ctas_by_occupancy * self.waves_per_cta / self.simds_per_cu

    @property
    def frags_m(self) -> int:
        """kFM: MFMA fragments down a wave tile. 0 if not determined."""
        if self.cta_m <= 0 or self.waves_m <= 0:
            return 0
        return self.cta_m // self.waves_m // self.frag_edge

    @property
    def frags_n(self) -> int:
        """kFN: MFMA fragments across a wave tile. 0 if not determined."""
        if self.cta_n <= 0 or self.waves_n <= 0:
            return 0
        return self.cta_n // self.waves_n // self.frag_edge

    @property
    def intensity(self) -> float:
        """MFMA issued per LDS fragment load: kFM*kFN/(kFM+kFN). 0.0 if unknown.

        The wave-tile law. Note it is only half of effective intensity -- the
        other factor is MACs per LDS element -- and that MFMA SHAPE IS A
        REPARAMETERISATION OF IT, not an axis: doubling the fragment edge
        doubles MACs per element while halving both kFM and kFN, so the product
        is invariant. See the module docstring.
        """
        fm, fn = self.frags_m, self.frags_n
        if fm <= 0 or fn <= 0:
            return 0.0
        return fm * fn / (fm + fn)

    @property
    def prefetch_cover(self) -> int:
        """MFMA a wave issues inside one global-load latency window. 0 if unknown.

        `(STAGE_K / frag_edge) * kFM * kFN`. The staged loop issues stage s+2's
        global loads at iteration s and stores them at s+1, so the window is one
        stage of MFMA, and this counts it. Measured on gfx942/bf16, halving it
        costs about 15% on the m128 route -- v82 halved it 32 -> 16 with the
        wave count held and lost 16.2% -- but `waves_per_simd` SUBSTITUTES for
        it: v83 halved it the same way while doubling waves per SIMD 2 -> 4 and
        lost only ~2%, and v79 halved it 64 -> 32 while going 1 -> 2 and GAINED
        7.5%. Read the two together, never one alone. Runs 492-495, 507-518.
        """
        fm, fn = self.frags_m, self.frags_n
        if self.stage_k <= 0 or fm <= 0 or fn <= 0:
            return 0
        return (self.stage_k // self.frag_edge) * fm * fn

    @property
    def splitk_plane_bytes(self) -> int:
        """HBM traffic the split-K workspace costs: written once, read once.

        The FP32 partial plane is `m * n * slices` elements REGARDLESS OF TILE
        -- every slice writes a full output-shaped plane -- so this is the one
        cost that a tile change moves without appearing anywhere in the tile's
        own arithmetic. 0 if the facts do not determine it.
        """
        if self.m <= 0 or self.n <= 0 or self.slices <= 1:
            return 0
        return 2 * self.splitk_workspace_bytes

    @property
    def splitk_workspace_bytes(self) -> int:
        """FOOTPRINT of the split-K workspace, `4 * m * n * slices`. 0 if not
        determined.

        Distinct from `splitk_plane_bytes` above, which is the round-trip
        traffic and therefore exactly twice this. The distinction matters
        because cache residency is a question about footprint while bandwidth
        is a question about traffic: comparing the traffic figure against a
        cache size would be wrong by a factor of two, in the direction that
        calls a resident workspace non-resident. Finding (17).
        """
        if self.m <= 0 or self.n <= 0 or self.slices <= 1:
            return 0
        return 4 * self.m * self.n * self.slices


def _tile_grid_is_one_row_tall(f: RouteFacts) -> bool:
    """The output is at most one tile tall, so widening the tile costs slices.

    A NEGATIVE precondition: where it holds, widening `cta_m` is not free and
    must be priced before it is proposed. With `m <= cta_m` the grid has a
    single tile row, so doubling `cta_m` cannot reduce it further -- it only
    halves the tile count, and filling the machine then requires doubling
    `slices`, which doubles `splitk_plane_bytes`. Measured on gfx942/bf16:
    `prefill_m128_square` moving 64x128 -> 128x128 went 4 slices -> 8 and paid
    +16.78 MB of HBM traffic = +3.17 us at 5.3 TB/s against a measured +3.34 us
    total loss, i.e. the traffic accounts for 95% of it, while the move's two
    intensity improvements (25% fewer LDS reads per MFMA, 1.5x fewer
    global->LDS elements per MAC) were together worth the remaining 5%.
    Runs 501-506.
    """
    return f.m > 0 and f.cta_m > 0 and f.m <= f.cta_m


def _simd_holds_one_wave(f: RouteFacts) -> bool:
    """Each SIMD holds exactly one wave, so nothing can issue under its latency.

    The precondition behind splitting a CTA into more, smaller waves. It is
    checkable from the tile alone, and it is deliberately phrased on LDS rather
    than on launch bounds: measured on gfx942/bf16, the only route in the suite
    whose LDS forces one CTA per CU (64x128 at a 128-deep stage, 50688 of 65536
    bytes) is the only one where the move paid -- +4.6% in suite and +7.5%
    isolated on 128x4096x4096. The route whose *launch bounds* ask for one CTA
    but whose LDS admits four (128x64, 13824 bytes) already had its second wave
    and only paid the bill: -1.9%, complete separation, run 497-500.

    Tested on both sides, which no other precondition here can claim. Violated
    at four waves per SIMD (96x128@32, 16128 bytes, four CTAs of 256 threads)
    the move loses 3.5% on the shipped path and 4.5% with the launch-bound
    min-CTA relaxed -- runs 721-732, finding (9). So this is a gate, not a
    heuristic: `waves_per_simd >= 2` makes `wave_tile: split_n`
    predicted-negative without a run.

    Finding (17) is why the split-K partial plane is *not* a descriptor
    dimension, and this gate is what decides it. On the same 64x128@128 route a
    narrowing to 32x64/1x4/sk256 was measured against the shipped tile with
    cover held equal at 32 and the plane deleted outright -- 256 tiles means one
    slice, so `2*4*128*4096*4 = 16.8 MB` of fp32 write-then-read-back
    disappears, roughly a third of the route's compulsory traffic. It still lost
    6.8% (0.03564 -> 0.03808 ms, medians of three, disjoint groups, runs
    1350-1358).

    It bought that plane saving with two things, not one, and the run cannot
    separate them: intensity 1.00 -> 0.67 (50% more `ds_read` per MFMA), and a
    wave per SIMD, 2.0 -> 1.0, because the shipped route has been 2x4 since v79
    -- this gate fires on the narrow arm and NOT on the shipped one. What makes
    the result usable anyway is that finding (7) already priced the wave term in
    the opposite direction on this exact route: v79 went 1 wave at cover 64 to 2
    waves at cover 32 for +7.5% isolated. Running it backwards at matched cover
    should therefore cost about 7.5%, and the measured total is 6.8%, so the
    plane saving and the intensity loss cancel to within the noise.

    The transferable conclusion is the negative one: **a third of a route's
    compulsory traffic, deleted, did not buy back one wave per SIMD.** Promoting
    the plane to a first-class term would have made this descriptor recommend
    exactly the arm that lost, so it is recorded as a bounded second-order term
    with no gate of its own. Bounded, not valued: a number for the plane needs
    an arm that moves it at fixed tile, which the plan interface cannot express
    (slices are chosen, not forced). The comparable saving on decode_m96_up was
    4.3%, but that is another route and must not be quoted as this one's.
    """
    return f.waves_per_simd == 1.0


def _lds_pins_route_to_one_cta(f: RouteFacts) -> bool:
    """The route holds one CTA, and no tile that keeps its cover would hold two.

    Finding (16), and the predicate that retired the last open axis. It is a
    *terminal* gate: unlike every other precondition here it does not license a
    move, it forbids the whole cover/waves plane at this depth. A route it fires
    on must be handed to a new mechanism, not to another parameter search.

    The arithmetic is exact and needs nothing but LDS. At a given depth a tile
    costs `(cta_m + cta_n) * (stage_k + 4) * 2` bytes, so holding two CTAs means
    costing at most half the 65536-byte budget. Cover is
    `(stage_k / frag_edge) * kFM * kFN` and `kFM * kFN` is the tile area divided
    by `waves_per_cta` and the fragment area -- so at a fixed wave count,
    shrinking the tile to fit under half the budget *lowers* cover by exactly
    the factor it shrinks the area. There is no way to buy cover back except by
    taking waves off the SIMDs, and finding (9) already gates that.

    Measured on `prefill_m128_square`, `64x128/2x4/sk128`: `objmeta` reads 92
    VGPR, **0 AGPR**, LDS 50688, ctaV 2, ctaL 1 -- it binds LDS with register
    headroom going to waste, which looked like an opening and is not. 50688 is
    well over the 32768 half-budget, and the full depth-128 enumeration for
    `m=128 n=4096 k=4096` says why nothing recovers it: every configuration with
    two or more waves per SIMD has cover <= 32, every configuration with cover
    above 32 sits at one wave per SIMD, and both sides of that trade are already
    measured negative -- the pre-v79 `2x2` at cover 64 lost 7.5% and the v83
    `4x4` at cover 16 lost 2.3%. The only tile at this depth that holds two CTAs
    at all is `32x64` (perimeter 96, cover 16). The higher-cover rows of the
    plane (`96x128` at cover 96, `160x64` at cover 80) are unreachable for an
    unrelated reason: 128 is divisible by neither 96 nor 160, so they buy their
    cover with 25-33% padded MFMA work.

    Note what this does *not* say. It is not `simd_holds_one_wave` -- that one
    fires on a route with room to split and licenses splitting it. This fires on
    a route with no room in either direction, and the correct response is to
    change what a stage costs, not how it is divided.
    """
    if f.lds_bytes <= 0 or f.stage_k <= 0:
        return False
    if f.ctas_by_lds != 1:
        return False
    return f.lds_bytes > _LDS_BYTES_PER_CU // 2


def _grid_cannot_fill_machine(f: RouteFacts) -> bool:
    """Every CTA the plan can produce still leaves CUs idle.

    The precondition behind the deep-stage mechanism. If it holds, the second
    LDS buffer is buying residency that does not exist, so its capacity can be
    spent on stage depth instead at no occupancy cost.
    """
    return f.ctas <= f.cu_count


def _grid_has_multiple_tile_rows(f: RouteFacts) -> bool:
    """Rasterization can move traffic only if there is more than one tile row.

    With a single row the remap is the identity and only its scalar prologue
    survives -- measured as a 0.03..0.09 loss on exactly the one-row shapes.
    """
    return f.tile_rows > 1


def _grid_oversubscribes_machine(f: RouteFacts) -> bool:
    """At least two CTAs per CU, so residency-buying mechanisms have something
    to overlap with."""
    return f.ctas >= 2 * f.cu_count


def _grid_overflows_residency(f: RouteFacts) -> bool:
    """The grid supplies more CTAs than the machine can hold at once, so buying
    residency has something to hold.

    This is the gate on *raising occupancy*, and it is the exact converse of
    the law that filling the machine with tiles is free. Residency you have no
    CTAs for is worth nothing: you cannot overlap a slot that stays empty.

    Tested on both sides in the same v86 run, which is what makes it a gate
    rather than a plausible story -- and the prediction was made from the tile
    arithmetic before either number was read:

    * `prefill_m512_up`, tile 128x64, 4 x 172 = 688 tiles at one slice = 688
      CTAs against 304 x 2 = 608 slots. It overflows, so doubling the cap to
      four CTAs per CU (1216 slots) lets the whole grid sit resident, and the
      shape moves -1.6% in suite and -1.25% isolated.
    * `decode_m64_square`, tile 64x128, 1 x 64 = 64 tiles at eight slices = 512
      CTAs against the same 608 slots. It does NOT overflow, so the identical
      register change on the identical wave layout buys **+0.08%, i.e. nothing
      measurable**, on a route whose object changed exactly as much.

    Note this must read `ctas_by_occupancy` and not `ctas_by_lds`: on the two
    routes v86 touched, the LDS bound said four CTAs and the register bound
    said two, and it was the register bound that was binding and that the fix
    moved. See finding (10).
    """
    slots = f.cu_count * f.ctas_by_occupancy
    return slots > 0 and f.ctas > slots


def _grid_underfills_residency(f: RouteFacts) -> bool:
    """The grid cannot supply enough CTAs to occupy the residency the tile
    already has, so that residency can be spent on something else for free.

    The exact converse of `grid_overflows_residency`, finding (13), and it had
    to be written because `grid_cannot_fill_machine` is strictly stronger and
    misses the case that mattered. That one asks `ctas <= cu_count`: one CTA
    per CU. This one asks `ctas <= cu_count * ctas_by_occupancy`: enough CTAs to
    use every *slot* the occupancy bound admits. On `decode_m96_up` the numbers
    are 86 tiles x 7 slices = 602 CTAs against 304 CUs -- so the machine is
    oversubscribed on CUs and `grid_cannot_fill_machine` is false -- but the
    shallow tile holds four CTAs per CU, 1216 slots, so **three of every four
    slots are empty no matter what the kernel does.** Occupancy the grid cannot
    reach is not a resource in use.

    That makes it the licence to spend residency, and the price is charged
    against slots that were already idle. Measured on `decode_m96_up`, 96x128,
    where the move 32-deep stage -> 128-deep stage takes the tile from 4
    resident CTAs to 1: 0.07728 -> 0.06844 ms, -11.44%, speedup 0.9469 ->
    1.0524, confirmed with no debug hook set (run 1080) and again in suite
    (runs 1090-1095).

    It must not overrule finding (11) and it cannot: the two predicates are
    disjoint by construction. A grid that fills its slots is buying something
    real with them and keeps them.

    Guarded like the others -- an unmeasured occupancy leaves
    `ctas_by_occupancy` at 0 and must not read as "everything is idle".
    """
    slots = f.cu_count * f.ctas_by_occupancy
    return slots > 0 and 0 < f.ctas <= slots


def _split_k_tail_wave_unhidable(f: RouteFacts) -> bool:
    """The plan buys a second CTA wave that no residency can hide, on a split
    whose reduction is already an expensive fraction of the work.

    Three conditions, and all three were measured rather than assumed:

    * `ctas_per_cu_cap <= 1` -- one CTA per CU, so the tail of wave one has
      nothing on its own CU to overlap with and the extra wave is fully serial.
    * `ctas > cu_count` -- the plan actually spills past one wave. If it fits,
      there is nothing to give back.
    * `slices**2 * 100 >= k` -- the reduction overhead fraction is high. This
      is the term that decides it, and it is the one an occupancy-only rule
      would miss. Reduction cost scales with `slices` and is independent of
      `k`, while compute per slice scales as `k / slices`, so the overhead
      goes as `slices**2 / k`.

    The third test is calibrated on gfx942/bf16, not derived: it separates the
    one measured shape that gives back the wave (128x4096x4096, 8**2/4096 =
    0.0156) from the two that must keep it (256x4096x11008, 0.0058, and
    1024x4096x11008, 0.0015). Three points fix a threshold but not its slope
    -- treat the 0.01 boundary as the edge of what has been measured, and
    widen it only with more shapes, never to make a candidate pass.
    """
    if f.ctas_per_cu_cap > 1 or f.slices <= 1 or f.k <= 0:
        return False
    return f.ctas > f.cu_count and f.slices * f.slices * 100 >= f.k


def _barrier_is_cheap_enough_to_spend(f: RouteFacts) -> bool:
    """A move that buys something by *adding* barriers has a chance of paying.

    Finding (12). Every restructuring that subdivides a stage -- a ramped
    prologue, a chunked fill, any producer/consumer split within one stage --
    pays in barriers, and a barrier is priced by how many waves must rendezvous
    at it and by how little else the CU has to run meanwhile. This is the same
    coin v49 collected +0.042 from by *halving* the barrier count, spent in the
    other direction.

    Measured on `64x128/2x4/sk128`, eight waves per CTA at one CTA per CU --
    the most expensive rendezvous in the plan space:

    * v87 ramped stage 0 into four chunks: +5.85% on `prefill_m128_square`,
      rank sum 57 of a maximum 57, i.e. perfect separation.
    * v88 peeled stage 0 into *one* chunk -- barrier count identical to the
      uniform prologue, first MFMA no earlier, only the stage-0 schedule
      changed: +1.56%, also 57 of 57.
    * The residual, 4.29%, is three extra barriers **minus** whatever earlier
      MFMA issue was worth. At ~1.4% per barrier the barriers account for all of
      it, so the head latency the ramp was built to uncover is worth ~0.

    Hence the gate is not "is the ramp deep enough" but "does this route have a
    cheap barrier at all". Eight waves per CTA with one CTA per CU has the
    dearest possible one and nothing else resident to hide it, so it must never
    spend more; a route with several CTAs per CU can overlap another CTA's work
    across the rendezvous and may. `waves_per_cta` alone is not the test --
    residency is what supplies something to run during the wait.

    Deliberately conservative: unmeasured registers leave `ctas_by_occupancy`
    at 0, and an unknown residency must not read as a licence to spend.
    """
    if f.ctas_by_occupancy <= 0 or f.waves_per_cta <= 0:
        return False
    return f.ctas_by_occupancy > 1


def _plan_slices_ignores_residency(f: RouteFacts) -> bool:
    """The shipped analytic plan disagrees with the residency-corrected one.

    Finding (18). This is a **diagnostic** gate, not a licence: it fires on
    routes where `plan_slices` and `fill_limited_slices` differ, meaning the
    tuner is the only thing standing between the plan and a wrong slice count.
    No mechanism is keyed on it, and none should be, for two independent
    reasons.

    First, it names no descriptor edge. Slices are not a descriptor axis --
    they are chosen per launch by the planner and then re-searched by the
    tuner -- so "change the slice count" is not a move in this vocabulary and
    cannot be an adjacency edge. Registering it as one would put a knob in the
    archive that the archive cannot turn.

    Second, and this is the standing rule rather than a property of this
    finding: the tuner already searches `[planned/2, planned*2]` cold, and both
    measured optima -- 12 on `prefill_m256_down` against a `[4,18]` ladder, 3 on
    `decode_m96_up` against `[3,12]` -- are inside it. **A slice change the
    tuner already reaches is worth zero to ship.** Nothing was built off this
    finding and nothing should be.

    What it is worth is exactly what the QD redesign is for. The archive is
    supposed to store route-level mechanisms rather than re-run local parameter
    search in every cell, and "the analytic plan is missing its residency term,
    the correction is `floor(cu_count * ctas_by_occupancy / tiles)`, and that
    correction is an upper bound because the reduction plane charges for the
    slices it hands out" is a mechanism. "The tuner found 12" is not. This gate
    marks the cells where that distinction has teeth.

    Requires measured occupancy: an unread register file leaves
    `fill_limited_slices` at 0, and an unknown correction must not read as a
    disagreement.
    """
    corrected = f.fill_limited_slices
    planned = f.planner_slices
    return corrected > 0 and planned > 0 and corrected != planned


PRECONDITIONS: Mapping[str, object] = {
    "plan_slices_ignores_residency": _plan_slices_ignores_residency,
    "barrier_is_cheap_enough_to_spend": _barrier_is_cheap_enough_to_spend,
    "grid_cannot_fill_machine": _grid_cannot_fill_machine,
    "simd_holds_one_wave": _simd_holds_one_wave,
    "tile_grid_is_one_row_tall": _tile_grid_is_one_row_tall,
    "grid_has_multiple_tile_rows": _grid_has_multiple_tile_rows,
    "grid_oversubscribes_machine": _grid_oversubscribes_machine,
    "grid_overflows_residency": _grid_overflows_residency,
    "grid_underfills_residency": _grid_underfills_residency,
    "lds_pins_route_to_one_cta": _lds_pins_route_to_one_cta,
    "split_k_tail_wave_unhidable": _split_k_tail_wave_unhidable,
}


@dataclass(frozen=True)
class Mechanism:
    """One transferable route-level finding.

    `axis`/`from_value`/`to_value` name the move in the descriptor vocabulary,
    so a mechanism is always an edge of the adjacency graph and never a free
    text suggestion. `spends` names the resource the precondition licenses
    giving up. `evidence` is the measured delta that earned it, with the run
    ids, because an unmeasured mechanism is a hypothesis and must not be
    selectable as a parent.
    """
    name: str
    precondition: str
    axis: str
    from_value: str
    to_value: str
    spends: str
    evidence: str

    def applies_to(self, facts: RouteFacts) -> bool:
        predicate = PRECONDITIONS.get(self.precondition)
        return bool(predicate) and bool(predicate(facts))  # type: ignore[operator]

    def valid(self) -> bool:
        order = AXIS_ORDER.get(self.axis)
        if order is None or self.precondition not in PRECONDITIONS:
            return False
        if self.from_value not in order or self.to_value not in order:
            return False
        if self.spends not in TRADEABLE:
            return False
        return abs(order.index(self.to_value) - order.index(self.from_value)) == 1

    def object(self) -> dict[str, str]:
        return {"name": self.name, "precondition": self.precondition,
                "axis": self.axis, "from": self.from_value, "to": self.to_value,
                "spends": self.spends, "evidence": self.evidence}


# The mechanisms this line actually measured, in the form a cell should store
# them. Every one is a single legal adjacency step with a host-checkable
# precondition and a run-identified delta.
MECHANISMS: tuple[Mechanism, ...] = (
    Mechanism(
        name="deepen_starved_grid",
        precondition="grid_cannot_fill_machine",
        axis="k_pipeline",
        from_value="lds_pingpong",
        to_value="lds_deep_single",
        spends="occupancy",
        evidence="gfx942/bf16 128x4096x4096: 0.763 -> 0.876, three interleaved "
                 "pairs, paired deltas +0.0186/+0.0192/+0.0181 (runs 184-189); "
                 "then, when the deep route's wave tile was also widened "
                 "32x64/1x4 waves -> 64x128/2x2 (intensity 2/3 -> 4/3), the "
                 "kernel's own device time on this shape fell 0.039800 -> "
                 "0.039000 ms, -2.01%, Mann-Whitney z=+3.35 over eleven "
                 "interleaved pairs (runs 208-230). Depth alone and intensity "
                 "alone were both measured and both lose: 64x128 at depth 64 "
                 "is 0.794 speedup, 128x128 at depth 64 is 0.828. "
                 "SCOPE: this is a per-shape mechanism only. Measured on "
                 "candidate device time, the same change is -0.20% on the "
                 "geomean of all eleven contexts, so it must NOT be recorded "
                 "as a cross-shape improvement. Speedup-denominated evidence "
                 "was withdrawn: the rocBLAS oracle on this box is bistable "
                 "at 4.4% in blocks of 5-8 runs, which an alternating "
                 "interleave cannot cancel, and a negative control on an "
                 "untouched route moved +0.0396 speedup at p=0.025 while its "
                 "candidate time was flat to -0.06%.",
        # The wave tile is deliberately NOT an axis. It changes intensity, so it
        # passes the AXIS_EFFECTS admission test on its own -- but the move is a
        # choice of *elite* inside the lds_deep_single cell, not a transition
        # between cells: both the 32x64 and the 64x128 route carry the identical
        # descriptor. Recording it here is the point of the mechanism table.
        # An archive that only stored coordinates would have had to rediscover
        # it by local parameter search at that cell, which is the failure this
        # whole representation exists to remove.
        #
        # v75, same cell again, and the same lesson a second time. The slice
        # count is likewise not an axis -- it is an elite choice inside this
        # cell -- and the default heuristic aims at 1.5 CTA waves, which is a
        # bet that the half wave past the first is a partially hidden tail.
        # This route's tile is 50688 B of LDS, so exactly one CTA is resident
        # and there is nothing to hide behind: the second wave is fully
        # serial. Cutting 8 slices to 4 on gfx942/bf16 128x4096x4096 is
        # 0.04686 -> 0.03864 ms isolated, +17.5%, z=+2.31 with the two arms
        # completely separated, and +16.4% (z=+2.88) in-suite; the untouched
        # negative control is flat at -0.36%, z=-0.58. The forced-slice sweep
        # puts the cliff exactly on the wave boundary -- 4 slices is 256 CTAs
        # on 304 CUs at 0.03828, 5 slices is 320 CTAs at 0.04312, +12.6% for
        # 25% more parallelism -- which is why `split_k_tail_wave_unhidable`
        # is keyed to residency and reduction fraction rather than to a shape.
        # Both other split-K shapes were measured and both REJECT the move
        # (K=11008 amortises the reduction), so the precondition's third term
        # is not decoration. See `_split_k_tail_wave_unhidable`.
    ),
    Mechanism(
        name="deepen_underfilled_grid",
        precondition="grid_underfills_residency",
        axis="k_pipeline",
        from_value="lds_pingpong",
        to_value="lds_deep_single",
        spends="occupancy",
        evidence="gfx942/bf16 96x11008x4096 (decode_m96_up), 96x128 tile: "
                 "stage 32 -> stage 128, 4 resident CTAs -> 1, 0.07728 -> "
                 "0.06844 ms, -11.44%, speedup 0.9469 -> 1.0524; forced-tile "
                 "three-point sweep 0.07862 (@32) / 0.07188 (@64) / 0.06824 "
                 "(@128) is monotone in depth and @256 does not fit LDS "
                 "(116480 B > 65536), so 128 is the end of the axis; shipped "
                 "planner reproduces it with no hook set (run 1080) and the "
                 "suite agrees (runs 1090-1095). The same move on the narrowed "
                 "96x64 tile is the control and it loses: 0.08132 at @128, "
                 "matched to the winner on cover (48), CTAs (1) and waves per "
                 "SIMD (1.00), differing only in intensity (1.20 vs 1.71) -- "
                 "so spend the idle residency on depth, never on width. "
                 "SECOND TERM, added after two counterexamples on "
                 "prefill_m2048_square (128x128, 512 CTAs on 304 CUs, also "
                 "underfilled): an underfilled grid is NECESSARY BUT NOT "
                 "SUFFICIENT, and the size of the cover prize decides it. "
                 "`deepest_stage_k_in_lds / stage_k` is x4 here and x2 there. "
                 "x1.5 loses (v97, stage 32 -> 48, +12.7%, and it also blew "
                 "the register file to 228 VGPR + 96 AGPR because the stage "
                 "was not a power of two); x2 loses (v99, stage 32 -> 64, the "
                 "whole prize that tile has, 0.18392 -> 0.22636 ms, +23.1%, "
                 "runs 1490-1495 rotated three each). Both end at one CTA per "
                 "CU exactly as this route does, so the price is the same and "
                 "only the prize differs. **The break-even is between x2 and "
                 "x4: do not attempt the move on a tile whose LDS admits only "
                 "a doubling.** A gate keyed instead on which resource binds "
                 "(`ctas_by_vgpr <= ctas_by_lds`) was written and deleted -- "
                 "it fires on THIS route, where the move wins, because both "
                 "routes are equal-bound at 4/4 and 3/3.",
    ),
    Mechanism(
        name="undo_die_round_robin_then_group",
        precondition="grid_has_multiple_tile_rows",
        axis="rasterization",
        from_value="grouped_m",
        to_value="xcd_remapped_grouped",
        spends="traffic",
        evidence="gfx942/bf16 512x11008x4096: per-XCD distinct panels 47 MB -> "
                 "15 MB, +0.02 paired",
    ),
    Mechanism(
        name="tune_slice_count_in_the_measured_cache_regime",
        precondition="grid_cannot_fill_machine",
        axis="plan_binding",
        from_value="static",
        to_value="runtime_tuned",
        spends="binding",
        evidence="gfx942/bf16 suite: +0.0117 geomean, and it halves the tuner's "
                 "own decision variance (v61). Re-measured 2026-08-15 on "
                 "candidate device time, tuner on vs off, interleaved: "
                 "geomean +2.15%, and the wins are where the static formula "
                 "is wrong -- 1024x4096x11008 +12.46%, 256x4096x11008 +6.10%, "
                 "64x8192x8192 +9.18%, all with complete run separation. It "
                 "is the strongest single mechanism in the archive. ONE "
                 "measured regression: 16x4096x4096 -6.73%, also separated. "
                 "The ladder is [planned/2, 2*planned] and at 0.026 ms it is "
                 "deciding between candidates microseconds apart, which is "
                 "the cold/warm regime hazard the implementation already "
                 "documents, at the shape with the thinnest margin. Record it "
                 "as a bounded defect of this mechanism, not as noise.",
    ),
)

# 2026-08-15. Two findings from re-measuring gfx942/bf16 with the tuner ON --
# i.e. in the configuration the task actually scores, which is NOT the one the
# preceding A/Bs used -- and both are about what may be called a mechanism at
# all.
#
# (1) A better STATIC slice count is worth nothing on top of this axis.
#     v75 made the analytic plan compute 4 slices where it used to compute 8
#     on 128x4096x4096, worth +17.5% with the tuner off. With the tuner on,
#     the tune census shows it was already overruling 8 -> 4, the two builds
#     launch the same kernel, and the measured difference is +0.32% with no
#     separation on any of eleven cases. Anything that only moves `slices`
#     inside [planned/2, 2*planned] is free to `runtime_tuned` and must not be
#     recorded as a mechanism. Check the tune census before proposing one.
#
# (2) A starvation precondition must count CTAs, not tiles -- and the
#     descriptor was already right where the kernel was wrong.
#     `_grid_cannot_fill_machine` reads `f.ctas`, i.e. tiles * slices. The
#     kernel's own tile-widening gate read the tile count, which prices a wide
#     grid as if it never split K: on 256x4096x11008 that is 52 tiles read as
#     104 CTAs against 304 CUs, refusing the widening, while the plan launched
#     splits K eight ways for 416. Correcting it to the CTA count is worth
#     +2.82% on that shape (U=61/64, z=+3.05 over sixteen runs in two blocks
#     whose negative-control drift ran +1.33% and -0.82%, i.e. in opposite
#     directions, and +3.93% again with the shape isolated in its own
#     process). No new axis, no new precondition: the arithmetic in
#     `RouteFacts.ctas` is the whole content, and it is worth stating that the
#     descriptor caught this before the kernel did.
#
# (3) `workspace_fixup -> atomic_fixup` is a MEASURED LOSS on this arch when the
#     "atomic" is a per-tile arrival counter, and it is a loss for a reason that
#     generalizes. v78 moved the split-K reduction inside the GEMM kernel: every
#     CTA publishes its FP32 tile, bumps a per-tile counter, and the last
#     arrival reduces the tile it already has addresses for. Correctness passed
#     and the arithmetic is bit-identical (the atomic is on the counter, never
#     on the data, so partials are still summed once in slice order). It lost on
#     every one of the nine split-capable shapes, -43.6% geomean, while the two
#     shapes that never split moved -0.80% and +0.29% -- opposite directions,
#     both inside drift, which is what makes one run enough to reject.
#
#     The cost is not the dispatch it removed (~4.7 us) and not bandwidth: the
#     added time tracks the size of the FP32 partial plane (m256 +157 us, m1024
#     +169 us, m2 +25 us) at roughly 20x what that traffic costs at HBM speed.
#     Two things do it, and both are properties of the move rather than of this
#     implementation. First, winner-take-all collapses reduction parallelism
#     from the whole grid to one CTA per tile -- a factor of `slices` -- and it
#     spends that one CTA at the tail, when the machine is otherwise empty.
#     Second, the release/acquire pair the handoff requires is device-scope on
#     an 8-XCD part, so every partial is pushed past L2 and every read of it is
#     guaranteed to miss. The separate reduce kernel pays a dispatch to buy back
#     both. Do not price this edge as "one fewer launch"; price it as trading
#     `slices`-way reduction parallelism and L2 residency for that launch.
#     Runs 441-457 (baseline) vs 480.
#
# (4) The wave count is a route fact this axis grid does not contain, and the
#     first thing it bought was a mechanism the grid had been calling "tried".
#     Every variant from v1 to v78 launched `constexpr int kThreads = 256`,
#     four waves, from a file-global. On a route whose LDS admits exactly one
#     CTA per CU that is one wave per SIMD, and one wave per SIMD is a SIMD
#     with no second instruction stream to issue under its own ds_read and
#     MFMA latency. So `wave_schedule: symmetric_interleave` and
#     `symmetric_pingpong` had nothing to interleave and `s_setprio` was
#     biasing an arbiter holding one candidate. The archive could not tell
#     that apart from an axis whose values were tried and lost, and the
#     general form is worth stating: AN AXIS IS ONLY REAL IF THE CODE IT
#     INDEXES IS PARAMETERISED. A constant one level below the search space
#     is indistinguishable, from inside the search, from a measured negative.
#     Before recording an axis value as explored, check that the build can
#     express it.
#
#     v79 parameterised it (`Panel`/`load_panel`/`store_panel` take THREADS;
#     the block size derives from WAVES_M*WAVES_N) and split the 64x128 stage
#     -128 route into eight waves, 2x4, so each SIMD holds two. +4.55% in
#     suite, +7.5% with the shape isolated, complete separation both ways,
#     runs 492-495. Priced exactly: intensity kFM*kFN/(kFM+kFN) falls 1.33 ->
#     1.00, i.e. 33% more LDS reads per MFMA, accumulators 32 -> 16 VGPR per
#     lane, and the LDS footprint, stage depth and barrier count are all
#     unchanged. That last clause is what separates it from forcing a
#     shallower stage, which buys CTAs by multiplying barriers and lost ~20%
#     (runs 481-484).
#
#     v80 ran the same move on 128x64 and lost 1.89% with complete separation
#     (runs 497-500), and the rejection is what fixed the precondition. I had
#     selected that route because its `__launch_bounds__(256, 1)` asks for one
#     CTA per CU -- but a launch bound is a FLOOR on what the compiler must
#     fit, not a ceiling on what the hardware runs, and 13824 bytes of LDS
#     admits four. The route already had its second wave and only paid the
#     bill. Hence `_simd_holds_one_wave`, which reads LDS alone: it is exact
#     on the host, needs no profiler, and would have declined v80 before it
#     was built. It also stops firing once the move is made (8 waves at one
#     CTA/CU reads 2.0), so the search cannot take the step twice.
#
# (5) A TILE WIDENING ON A ONE-TILE-TALL GRID IS PRICED IN SPLIT-K TRAFFIC, and
#     the price appears nowhere in the tile's own arithmetic. v81 widened the
#     m128 route 64x128 -> 128x128 (at STAGE_K 64 so the LDS still forces one
#     CTA and the eight waves still give two per SIMD). Both intensity terms
#     improved -- kFM*kFN/(kFM+kFN) 1.00 -> 1.33, and MACs per LDS element
#     CTA_M*CTA_N/(CTA_M+CTA_N) 42.67 -> 64 -- and it lost 8.4% with complete
#     separation, runs 501-504.
#
#     The reason is `_tile_grid_is_one_row_tall`. With m=128 and cta_m=128 the
#     grid is one tile row, so doubling cta_m cannot shorten it; it only halves
#     the tile count, 64 -> 32, and 32 tiles cannot fill 304 CUs without
#     doubling `slices`, 4 -> 8 (confirmed by plan census, runs 505/506: both
#     arms plan 8 per their own grid). The split-K partial plane is m*n*slices
#     FP32 INDEPENDENT OF TILE, written once and read once, so the doubling
#     costs 2*4*128*4096*4 = 16.78 MB extra = 3.17 us at 5.3 TB/s against a
#     measured 3.34 us loss. THE TRAFFIC IS 95% OF IT. Two consequences. First,
#     price a widening on such a shape before proposing it: `splitk_plane_bytes`
#     is host-checkable and on this route one doubling of slices is 8.8% of
#     runtime, which no tile-shape improvement repays. Second, and this is the
#     part that transfers: the two intensity improvements were together worth
#     the residual 5%, i.e. nothing -- so on this route LDS bandwidth does not
#     bind. v79 paid 33% MORE LDS reads and won; v81 collected 25% fewer and did
#     not. Read those two together before proposing anything on intensity here.
#
# (6) THE DEEP SINGLE STAGE IS LOAD-BEARING, AND NOT THROUGH THE BARRIER COUNT.
#     v82 double-buffered the m128 route at STAGE_K 64 -- same tile, same tile
#     count, same slices, same split-K plane, same eight waves, same 32x32 wave
#     tile and intensity, same one CTA per CU and two waves per SIMD, and by
#     construction the same barriers per unit K (one per 64-deep stage against
#     two per 128-deep stage, both k/64). It lost 16.2% on m128 and 14.2% on the
#     m64 shape that shares the tile, both completely separated, runs 507-510.
#     Building it required relaxing `kDoubleBuffer`'s "room for three CTAs"
#     residency gate, whose premise -- "at one CTA there is nothing to overlap
#     with" -- is another four-wave assumption that an eight-wave CTA falsifies;
#     the relaxation is correct and is inert on every route reachable today.
#
#     Every attempt to trade STAGE_K 128 away has now lost: 144 (-4.7%) and 160
#     (-0.9%) with FEWER barriers, 64 and 32 at four waves (~-20% each) with
#     more, 128x128@64 with the same barriers per MAC (-8.4%, all traffic), and
#     now 64 double-buffered with barriers held exactly equal (-16.2%). The
#     incumbent is a maximum in both directions and the barrier count is
#     falsified as the explanation. What replaces it is prefetch cover in TIME
#     -- see (7), where v83 turns it from a hypothesis into a two-term model.
#
# (7) PREFETCH COVER AND WAVES PER SIMD ARE SUBSTITUTES, and the m128 route is
#     standing on the interior optimum of the trade. The staged loop issues
#     stage s+2's global loads at iteration s and stores them at s+1, so the
#     latency window a wave has to fill is exactly one stage of its own MFMA:
#     `prefetch_cover = (STAGE_K/frag_edge) * kFM * kFN`, the property above.
#     v82 and v83 both halve it 32 -> 16 and differ ONLY in the wave count, and
#     that difference is worth fourteen points:
#
#         probe                    cover  waves/SIMD  result
#         pre-v79  64x128 2x2 @128    64      1       baseline of its day
#         v79      64x128 2x4 @128    32      2       +7.5%   ACCEPTED
#         v82      64x128 2x4 @64     16      2       -16.2%  (double-buffered)
#         v83      64x128 4x4 @128    16      4       -2.3% / neutral
#
#     So cover is real at roughly 15% per halving, and another resident wave
#     SUBSTITUTES for it -- it supplies the instruction stream the shortened
#     window stopped supplying. This re-reads v79 correctly: it was never "buy a
#     second wave and pay in intensity", it was "halve the cover and pay for it
#     with a wave", which nets +7.5% going 1 -> 2 waves per SIMD and -2.3% going
#     2 -> 4. Neither term may be read alone, which is why `prefetch_cover`'s
#     docstring refuses to be a precondition on its own.
#
#     This CLOSES the wave-count axis on the m128 route: four waves is -7.5%,
#     eight is the optimum, sixteen is not better, and thirty-two is not
#     expressible (2048 threads exceeds the block limit). Runs 492-495, 511-518.
#
#     (6)'s counterexample is retired as bounded rather than explained: run 482
#     (64x128@64 at 2x2) has cover 32 and two waves per SIMD, both equal to the
#     incumbent, and still lost ~20%. The one thing left that differs is that it
#     pays TWO barriers per MAC -- the only configuration in the entire ledger
#     that does -- so the honest reading of the barrier count is not "irrelevant"
#     but "cheap enough that holding it equal while halving the cover still
#     loses 16%".
#
# (8) `wave_schedule: asymmetric_producer_consumer` IS MEASURED AND IT LOSES,
#     and the m128 route now has no untried axis value left. Four producer waves
#     were added on top of the accepted eight-wave 64x128@128 consumer geometry,
#     holding tile, wave tile, intensity, cover, barriers, LDS, CTAs per CU and
#     slices all fixed, so the only thing that changes is who issues the global
#     loads. On the shipped path (in-suite, autotuner on) it is +1.53% slower,
#     four of four rotated blocks positive with the controls passing rule 4.
#
#     The result is REGIME-DEPENDENT, and that is the transferable part: the
#     same binary is -1.49% (i.e. faster) in-suite with the autotuner OFF, and a
#     `[tunepick]` census proves the tuner reaches an IDENTICAL decision on all
#     eleven cases for both variants. So a tuner can move a verdict by three
#     points through the GPU work its ladder performs before the timed region,
#     without changing a single plan it emits. An A/B that reads one regime and
#     reports it as the mechanism's value is, on this evidence, making a claim
#     it has not measured -- and the shipped regime is the only one that decides.
#
#     With this value run, every axis value that is legal AND expressible on
#     `prefill_m128_square` has been measured: tile (420-425), stage depth
#     (470-473), wave count and occupancy (481-484, 492-495, 511-518), slices
#     (321-336, now owned by the tuner), output_path (480), k_pipeline's double
#     buffer (507-510), wave_schedule interleave (v50-v54) and now the
#     asymmetric split. The only unexplored route move left on this shape -- on
#     ANY shape -- is `decomposition: split_k -> stream_k`.
#
# (9) `simd_holds_one_wave` IS NOW A MEASURED GATE, NOT A DESIGN ARGUMENT, AND
#     TWO MORE ROUTE MOVES ARE PRICED WITHOUT BEING SHIPPED.
#
#     The v79 move was applied to `decode_m96_up` -- 96x11008x4096, route
#     96x128@32, one of the two shapes still behind rocBLAS at 0.91 -- whose plan
#     space had been declared exhausted three hundred runs earlier, before the
#     wave count was known to be an axis. 2x2 -> 2x4 takes the wave tile from
#     48x64 to 48x32: kFM 3 both ways, kFN 4 -> 2, intensity 1.71 -> 1.20, cover
#     24 -> 12. It LOSES 3.5% on the shipped path (4.5% with the launch-bound
#     min-CTA relaxed, so register pressure is not the cause), rule 4 passing on
#     ten exact negative controls in both blocks. Runs 721-732.
#
#     Finding (7) ordered that sign in advance. m96 already runs FOUR waves per
#     SIMD (16128 B of LDS admits four CTAs of 256 threads), so the wave the move
#     buys is the fifth through eighth on a saturated SIMD and is worth nothing,
#     while the halved cover is charged in full at the ledger's ~15%. That is
#     exactly what `_simd_holds_one_wave` says, and it is now the only
#     precondition in this file that has been tested on BOTH sides: satisfied and
#     the move wins (+7.5% isolated, m128), violated and the move loses (-3.5%,
#     m96). Treat `waves_per_simd >= 2` as predicted-negative for `wave_tile:
#     split_n` without a run.
#
#     Two further moves on `prefill_m128_square` are priced but unbuilt, and the
#     arithmetic is recorded so a later search does not re-derive it:
#
#     * `tile: 64x128 -> 128x128` at eight waves. Measured and REJECTED, -19%
#       (runs 704-712). Halving the tile count halves the CTAs, so the tile needs
#       twice the slices to reach the same occupancy, and `splitk_plane_bytes` is
#       INDEPENDENT OF TILE -- 16.8 -> 33.5 MB, +33% of total HBM traffic for +0%
#       of output. Filling the machine with tiles is free; filling it with slices
#       is not. Corollary: at intensity 1.00 with a 32x32 wave tile and eight
#       waves the CTA is pinned at 8192 output elements, and every factorisation
#       of 8192 that divides this shape gives exactly 64 tiles, so 304/64 = 4.75
#       is unreachable by any integer slice count.
#     * `decomposition: split_k -> stream_k`. PREDICTED NEGATIVE, ~ -10%, not
#       built. 2048 stages over 304 CTAs is 6.74, so the critical path falls 8 ->
#       7 stages (+12.5%), but 4.75 CTAs per tile means most CTAs straddle a tile
#       boundary and run the staged loop twice: 2 + x + 2 + (7-x) = 11 against
#       today's 2 + 8 = 10. The prologue count, not its placement, is the cost.
#       It becomes positive only behind `prologue_depth: uniform -> ramped`,
#       which is blocked on ~+24 VGPR per lane against a 170 ceiling.
#       [SUPERSEDED by finding (10): the ramp is NOT register-blocked. Measured
#       92 VGPR against a 256 ceiling, so +24 lands at 120. Stream-K's own
#       -10% price is unchanged; only its enabler's blocker is gone.]
#       [CLOSED by finding (12): the enabler was then BUILT and lost, 5.85% with
#       perfect rank separation, and its own control priced the head latency it
#       was supposed to recover at ~0. So stream-K keeps its -10% and no longer
#       has an enabler to sit behind. Both are closed on this route.]
#
#     Separately, the m128 slice optimum is GRID TAIL QUANTISATION, not pipeline
#     fill: `s_opt = floor(cu_count * ctas_per_cu_cap / tiles)`, and exceeding it
#     is expensive only when `ctas_per_cu_cap == 1`. Measured both ways -- at
#     64x128@128 (1 CTA/CU) crossing 304 CTAs costs 13% between s=4 and s=5; at
#     64x128@64 (2 CTAs/CU) crossing 608 between s=9 and s=10 costs nothing,
#     0.04282 vs 0.04278. Runs 601-712.
#
# (10) REGISTERS ARE MEASURABLE, AND ONCE MEASURED THEY MOVE TWO THINGS. Read
#     back from the built object -- not the source -- for all eleven gfx942
#     instantiations of `.v79_waves8`. Zero spills anywhere.
#
#     * The ramp is unblocked. `64x128/2x4/sk128` uses 92 VGPR and 0 AGPR, and
#       its ceiling is 256 per lane, not 170: LDS 50688 pins it to one CTA per
#       CU, hence 2 waves per SIMD, hence 512/2 each. Granule(92)=96, so the
#       ramp's +24 lands at 120 with 136 to spare -- it cannot spill and cannot
#       cost a wave, because residency here is set by LDS and stays at one CTA
#       until registers pass 256. The estimate that blocked it ("~148 against
#       170") was read off the source and was wrong at both ends.
#     * Four routes are register-limited BELOW their LDS allowance, via AGPRs
#       the earlier tool never even parsed. `64x128@32` and `128x64@32` are
#       124+48 = 172 -> granule 176 -> 2 CTAs where LDS says 4; `96x64@32` gets
#       3 where LDS says 5; `32x128@32` gets 4 where LDS says 5. So `s_opt`
#       above must key on `ctas_by_occupancy`, not `ctas_by_lds`. The two
#       routes carrying the shapes still behind rocBLAS are unaffected --
#       `96x128@32` is bound equally at 4, `64x128@128` at 1 by LDS -- so
#       findings (7) and (9) and the tail-quantisation measurements stand.
#
#     Finding (9) also gains an independent confirmation that involves no clock.
#     `.v85_m96w8` was built compile-only: `96x128/2x4/sk32` measures 68 VGPR,
#     0 AGPR, no spills -- registers FELL from 122 because splitting n halves
#     the accumulators -- giving 3 CTAs and 6.00 waves/SIMD against v79's 4.00.
#     Occupancy went UP. So v85's +3.5% is neither a register nor an occupancy
#     effect; it is only the halved cover, exactly as (9) says. The competing
#     "it ran out of registers" explanation is closed.
#
# (11) BUYING RESIDENCY PAYS ONLY WHERE THE GRID OVERFLOWS IT. See
#     `_grid_overflows_residency` for the both-sides measurement. The corollary
#     that matters for search: over the seven tiled shipped routes, exactly one
#     -- `prefill_m512_up`, 688 CTAs against 608 slots -- was ever eligible, and
#     v86 fixed it. The mechanism is exhausted on this suite, so a proposal to
#     raise occupancy anywhere else here is predicted-negative without a run.
#
# (12) A STAGE SUBDIVISION IS PAID FOR IN BARRIERS, AND SOME ROUTES CANNOT
#     AFFORD ONE. `prologue_depth: uniform -> ramped` was the top build item for
#     two stages, unblocked by finding (10). It was built and it lost: +5.85% on
#     `prefill_m128_square`, rank sum 57 of a maximum 57.
#
#     The number that matters is not that one, though, because v87 changed three
#     things at once -- earlier first MFMA, barrier count 1 -> 4, and stage 0
#     demoted to the v60 one-step MFMA schedule. The v88 control peeled stage 0
#     into a SINGLE chunk: same barriers, same MFMA timing, only the schedule.
#     It lost 1.56%, also 57 of 57. So the split is 1.56% schedule + 4.29% for
#     "three barriers minus the head latency saved", and at ~1.4% a barrier on
#     this route the barriers are the whole of it. THE PROLOGUE HEAD LATENCY IS
#     WORTH ~0. It was already covered, and two stages of planning had assumed
#     otherwise.
#
#     Generalised in `_barrier_is_cheap_enough_to_spend`: the gate is not the
#     ramp's depth but whether the route owns a cheap barrier at all. Eight waves
#     per CTA at one CTA per CU is the dearest rendezvous in the plan space and
#     has nothing else resident to run during the wait, so it can never spend
#     more; residency, not wave count, is what supplies the cover. This closes
#     the whole axis at every kRamp -- kRamp = 2 prices at ~+3% -- and, with it,
#     stream-K, which was only ever positive behind this enabler.
#
# (13) A GRID THAT UNDERFILLS ITS RESIDENCY IS HOLDING A RESOURCE IT CANNOT USE,
#     and the converse of finding (11) is worth more than finding (11) was. See
#     `_grid_underfills_residency`. The reason this took until run 1080 to find
#     is that the existing predicate for "there is idle machine here" was
#     `grid_cannot_fill_machine`, `ctas <= cu_count`, and `decode_m96_up` fails
#     it by a factor of two: 602 CTAs on 304 CUs. Counting CUs said the machine
#     was oversubscribed. Counting *slots* -- 1216 at four CTAs per CU -- said
#     three quarters of it was empty. The deep stage that the first count
#     forbade was worth -11.44% under the second.
#
#     Read (11) and (13) together as one law with a sign: residency is worth
#     buying exactly when the grid overflows it and worth selling exactly when
#     the grid underfills it, and `ctas` against `cu_count * ctas_by_occupancy`
#     decides which. Neither side is about the shape.
#
# (14) INTENSITY STOPS BEING INTERCHANGEABLE WITH COVER AT ONE WAVE PER SIMD,
#     which is the correction finding (7) needed and did not get until the
#     controls in runs 1090-... Cover is MFMA per *global* fetch; intensity is
#     MFMA per *LDS* read. Finding (7) measured them as substitutes, but every
#     point in that ledger had two or more waves per SIMD -- and with a second
#     wave resident the LDS latency hides underneath it, so only cover is
#     visible. Spend the residency down to one wave and there is nothing left to
#     hide behind: reads-per-MFMA binds directly and intensity becomes a
#     first-class term.
#
#     Measured, and it refuted an additive prediction made from finding (7)
#     before the run. `96x64@128` (cover 48, intensity 1.20) was predicted best
#     by adding the cover and depth terms; it was the *worst* deep arm at
#     0.08132, while `96x128@64` -- matched to it on cover (48), resident CTAs
#     (1) and waves per SIMD (1.00), differing in nothing but intensity (1.71)
#     -- ran 0.07188, 13% faster. One axis differed and it was worth 13%.
#
#     Operationally: **depth is free on an underfilled grid; width is not.**
#     When (13) licenses spending residency, spend it on stage depth, and do not
#     narrow the tile to pay for it -- narrowing buys cover at the cost of
#     intensity, and at one wave per SIMD that trade is a loss.
#
# (15) HOW MUCH RESIDENCY (13) LICENSES SPENDING DEPENDS ON THE SIZE OF THE
#     PRIZE, and the general rule is NOT the rule v93 shipped. v94 measured the
#     same trade -- stage 32 -> 64, cover 16 -> 32, occupancy 4 -> 2 -- on two
#     routes and called both signs in advance:
#
#       decode_m64_square  64x128  512 CTAs, 1216 -> 608 slots (0.84w)  -3.33%
#       prefill_m512_up    128x64  688 CTAs, 1216 -> 608 slots (1.13w)  +26.4%
#
#     Both underfill before. Only one still fits after, and the one that spills
#     13% into a second CTA wave loses 26% to the tail. So for a 2x-cover
#     deepening the law is `ctas <= cu_count * ctas_when_deep`: the *deepened*
#     grid must fit, not the shallow one.
#
#     But `decode_m96_up` deepens 32 -> 128, ends at 1.70 CTA waves -- spilling
#     harder than m512 did -- and wins 11.4%. Its cover *quadruples*, 24 -> 96.
#     So the fit test is not a law about grids at all; it is the break-even of a
#     trade whose two sides both scale. A 4x prize buys a second wave that a 2x
#     prize cannot. Encoding the general rule as the conservative one and
#     keeping the 4x case as a measured exception is the honest reading of three
#     points, and `deepen_underfilled_grid` in the kernel carries both rules
#     side by side with a comment forbidding their unification. Do not collapse
#     them here either: the permissive rule loses 26% on m512, the conservative
#     one gives back 11.4% on m96.
#
#     What is NOT yet known is where the break-even actually sits -- 3x cover at
#     1.3 waves has never been measured, and two of the three points are the
#     extremes. Until it is, treat the 4x exception as one measured route and
#     not as a second rule.
#
# (16) THE COVER/WAVES PLANE IS CLOSED WHEN LDS PINS THE ROUTE TO ONE CTA, and
#     with it the whole axis space this line has been searching. See
#     `_lds_pins_route_to_one_cta` for the arithmetic and the depth-128
#     enumeration. The reason this belongs in the ledger and not just in a
#     predicate is what it implies about the *search*, not about the route:
#
#       decode_m96_up         deepened, -11.44%          banked
#       decode_m64_square     deepened, -3.3%            banked
#       prefill_m512_up       deepening measured +26.4%  closed by measurement
#       prefill_m1024_down    2x prize, 2.21 CTA waves   closed by (15), no run
#       prefill_m256_down     2x prize, 1.54 CTA waves   closed by (15), no run
#       prefill_m2048_square  2x prize, 1.68 CTA waves   closed by (15), no run
#       prefill_m128_square   pinned at one CTA          closed by (16), no run
#
#     Seven of seven. Four of them closed without a run, which is the entire
#     argument for storing mechanisms instead of parameters -- a local search
#     would have had to pay for four more A/Bs to learn the same thing, and on
#     the m512 evidence three of them would have been large losses.
#
#     So the answer to "why can't more iterations close the gap to rocBLAS" is
#     that iterations are not the binding constraint: there are no unexplored
#     affordable points left in this space. A new axis is required, and (16)
#     states its specification exactly -- it must raise cover or intensity
#     WITHOUT spending waves per SIMD, which on gfx942 means reducing the LDS
#     cost per unit of stage depth. Direct-to-register global loads for one
#     operand, or an async-copy pipeline that does not double-allocate the
#     stage, both satisfy it; neither exists in the candidate today. That is a
#     code-generation change, and it is where the next effort belongs.
#
# (17) A WORKSPACE THAT FITS IN MALL IS NOT PRICED AT HBM BANDWIDTH, AND
#     ATOMICS DO NOT STREAM. `output_path: workspace_fixup -> atomic_fixup`
#     was built on `prefill_m256_down` and REJECTED at +8.05% -- isolated A/B,
#     three each rotated, fully rank-separated with disjoint +/-2 MAD bands
#     (runs 1520-1525).
#
#     The route is the most favourable case for the move that exists in this
#     suite. 256x4096x11008 on the 128x160 tile is 52 tiles, 17% of the
#     machine, so split-K is mandatory and the shipped plan runs twelve slices;
#     the kernel then writes a full m x n FP32 plane PER SLICE and reads them
#     all back. By byte count that is 102.8 MB against a 0.11876 ms kernel and
#     the atomic form cuts it to 10.5 MB -- a 10x reduction that lost anyway.
#
#     There is no confound. The built object's `128x160/2x2/sk32` line is
#     byte-identical across the two builds -- 152 VGPR, 0 AGPR, 0 spills,
#     20736 B LDS, 3 CTAs, 3.00 waves/SIMD -- and the device code contains
#     `global_atomic_add_f32` and no cmpswap, so a CAS-loop fallback is
#     excluded. Only the output path changed.
#
#     TWO RULES COME OUT OF IT.
#
#     * Atomics trade bytes for throughput and the trade is bad here. Twelve
#       streaming plane writes become ~12.6M read-modify-writes on one 4.19 MB
#       plane: 16 floats to a 64 B line, twelve RMWs per line, issued from CTAs
#       spread over all 8 XCDs, so each one is cross-die coherence traffic. The
#       axis is closed on every split-K route in this suite, because none of
#       them has more to gain than this one did.
#     * The 28.8% fixup estimate that motivated the build was too high, and
#       that is the more portable lesson. Twelve 4.19 MB planes are 50 MB --
#       past L2's 32 MB but well inside the 256 MB MALL -- so the round trip
#       was already being served from cache, not from HBM. `splitk_plane_bytes`
#       and every other workspace term in this file must be checked against
#       `fits_in_mall` before being priced at HBM bandwidth. Finding (9)'s
#       companion arithmetic on `prefill_m128_square` ("+33% of total HBM
#       traffic") is one of the estimates this calls into question: 33.5 MB
#       also fits in MALL, and that move was separately measured, so the
#       verdict stands even though the stated reason may not.
#
# (18) NO LEGAL TILE BUYS GRID EFFICIENCY FOR FREE, AND SPLIT-K REACHES THE
#      TERM ONLY PAST THE POINT WHERE IT PAYS.
#
#      CORRECTED. This finding was first written with a false premise and the
#      correction is recorded rather than overwritten, because the false
#      version was persuasive and the way it failed is the reusable part.
#
#      AS FIRST WRITTEN: this machine has 304 = 2**4 * 19 CUs and every shape
#      in this suite has power-of-two M and N, so a power-of-two tile gives a
#      power-of-two tile count; 2 is a primitive root mod 19, so that count is
#      never divisible by 19 and never lands on a multiple of 304. The residue
#      is not merely nonzero but pinned -- doubling the count doubles it mod 19
#      too -- so T, 2T and 4T share a utilisation exactly. All of that is true.
#      It is why 128x128, 128x64, 64x128 and 64x64 all read 0.8421 on
#      `prefill_m1024_down`, and why 1.1875 recurred across every tile on
#      `prefill_m2048_square` rather than being a coincidence of that shape.
#
#      WHERE IT WAS WRONG: a tile edge does not have to be a power of two. It
#      has to be a multiple of `16 * waves`, which admits 112, 160 and 224 --
#      and this kernel already ships a 160, so the counterexample was in the
#      source the whole time. 4096 needs 19 columns of 224, so 64x224 puts
#      exactly 304 tiles on `prefill_m1024_down` and 608 on
#      `prefill_m2048_square`: utilisation 1.0000, precisely the thing the
#      argument said could not exist.
#
#      WHY THE CONCLUSION SURVIVED ANYWAY: utilisation was the wrong figure of
#      merit. Landing on a multiple of 304 requires a factor of 19 in the tile
#      count, and on a power-of-two shape that factor has to come from an edge
#      that mis-divides the shape -- 4096 = 18.29 * 224, so the 19th column is
#      four-fifths padding. Netting it out, 64x224 scores `grid_efficiency`
#      0.9624, which is exactly what the crude 32x64 option scores. Both are
#      the same 19 surfacing in different places, one as idle CUs and one as
#      padded columns. A tile cannot spend its way out of a factor the machine
#      has and the shape does not.
#
#      SO THE TEST IS NOW AN EXHAUSTIVE SEARCH, not an argument about one
#      family of tiles: `legal_tiles` enumerates every (cta_m, cta_n, waves)
#      the kernel can instantiate at the route's wave count with LDS holding
#      what the resulting grid asks, and `tile_axis_can_collect_grid_waste`
#      accepts a move only if it raises `grid_efficiency` by more than the
#      suite's 1.5% drift while giving up neither `intensity` nor
#      `occupancy_waves_per_simd`. Over 112 legal tiles on
#      `prefill_m2048_square` and 126 on `prefill_m1024_down`, nothing
#      dominates what is already shipped -- so this is now also a positive
#      statement about choose_plan's tile order, not only a negative one.
#
#      THE CLOSEST MISS IS WORTH NAMING because it is the next thing anyone
#      will propose: `256x112/8x1` on `prefill_m2048_square` beats the shipped
#      tile on BOTH grid efficiency (+14.3%) and intensity (+16.7%), and is
#      refused only because 296 tiles ask one CTA per CU where 512 asked two,
#      halving waves/SIMD from 4 to 2. v98 measured that same occupancy step at
#      5.58% going the other way. It is a real candidate, not a nonsense one,
#      and if the occupancy term is ever shown to be smaller than 14% on this
#      route it should be built.
#
#      TWO PROPERTIES CHANGED HANDS in the correction and both were bugs in the
#      permissive direction. `grid_utilisation` alone called a 4% padding cost
#      a free win. And the occupancy guard first read `waves_per_simd`, which
#      is LDS/register CAPACITY, so two tiles with room for eight waves looked
#      identical even when one launched a grid that only ever asks for one CTA
#      per CU -- that is what let `256x112` through. `occupancy_waves_per_simd`
#      is the actual figure, `min(grid ask, LDS capacity) * waves_per_cta / 4`.
#
#      THE SLICE AXIS WAS THEN MEASURED, and the experiment was built to
#      separate two terms rather than to A/B one. `prefill_m1024_down` plans 2
#      slices and the tuner searches [1, 4], so every count that could collect
#      the term is outside its ladder; `GEAK_DEBUG_FORCE_SLICES` prices the
#      cold curve without a build. Runs 1536-1545, machine I, rotated:
#
#          s=2 (tuner) util 0.8421  0.26724 ms  +/-0.00336   --
#          s=4         util 0.8421  0.26742 ms  +/-0.00460   +0.07%
#          s=7         util 0.9825  0.30060 ms  +/-0.00320  +12.48%
#          s=13        util 0.9952  0.34418 ms  +/-0.01364  +28.79%
#
#      s=4 is a null control by construction: identical utilisation, twice the
#      fixup planes. It came back flat, which validates the instrument and
#      independently re-confirms finding (17) from the other direction -- a
#      64 MB MALL-resident workspace is very nearly free, and the per-plane
#      figure this file had been carrying was at least 3x too high.
#
#      So the loss at s=7 is not fixup. K per slice falls 5504 -> 1572, i.e.
#      86 stages -> 24, and the cost is a cliff rather than a slope: 86 -> 43
#      costs nothing measurable, 43 -> 24 costs ~27% once the 14.3% grid win
#      being collected at the same time is added back.
#
#      CONSEQUENCE. Both routes carrying this term now have the tile axis
#      closed by proof and the slice axis closed by measurement or arithmetic:
#      `prefill_m2048_square` (512 tiles overflowing into 2 rounds) and
#      `prefill_m1024_down` (256 tiles underfilling 1). They are one term seen
#      from either side of a round boundary. Stream-K is the only remaining
#      mechanism that can reach it, and the ~0.071 ms the two routes hold
#      between them is the largest coherent block of headroom left here.


def mechanisms_for(facts: RouteFacts, descriptor: Mapping[str, object] | None = None,
                   *, arch: str = SUPPORTED_ARCH, dtype: str = "bf16") -> list[Mechanism]:
    """Measured mechanisms whose precondition holds on `facts`.

    When `descriptor` is given, also requires that the mechanism starts where
    the descriptor currently sits and that the destination is legal -- i.e.
    that the move is an edge this cell can actually take.
    """
    out: list[Mechanism] = []
    for mech in MECHANISMS:
        if not mech.valid() or not mech.applies_to(facts):
            continue
        if descriptor is not None:
            if descriptor.get(mech.axis) != mech.from_value:
                continue
            moved = {axis: str(descriptor[axis]) for axis in AXES
                     } if descriptor_valid(descriptor, arch=arch, dtype=dtype) else None
            if moved is None:
                continue
            moved[mech.axis] = mech.to_value
            if not descriptor_valid(moved, arch=arch, dtype=dtype):
                continue
        out.append(mech)
    return out


def _parser():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list-legal", action="store_true", help="print every legal descriptor for --arch/--dtype")
    p.add_argument("--list-mechanisms", action="store_true", help="print the measured mechanism records")
    p.add_argument("--check-axes", action="store_true", help="fail if any axis moves nothing tradeable")
    # No default: see `qd_v2._arch_argument`. `--list-legal` is the surface
    # where a silent gfx90a would be most misleading, because it enumerates
    # the search space itself and would omit every xcd_remapped_grouped tuple
    # without saying that an arch assumption is why.
    p.add_argument("--arch", required=True,
                   help=f"target gfx arch, one of {', '.join(sorted(SUPPORTED_ARCHES))}")
    p.add_argument("--dtype", default="bf16")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    import json
    import sys
    args = _parser().parse_args(argv)
    if args.list_legal:
        payload = json.dumps(all_legal_descriptors(arch=args.arch, dtype=args.dtype),
                              sort_keys=True, separators=(",", ":")) + "\n"
        sys.stdout.write(payload)
        return 0
    if args.list_mechanisms:
        payload = json.dumps([m.object() for m in MECHANISMS],
                              sort_keys=True, separators=(",", ":")) + "\n"
        sys.stdout.write(payload)
        return 0
    if args.check_axes:
        dead = dead_axes()
        if dead:
            sys.stderr.write("dead axes (move nothing tradeable): "
                             + ", ".join(dead) + "\n")
            return 1
        # Separate exit code: a tombstoned axis is not a design slip, it is a
        # measured refusal being undone, and the operator needs to be told which
        # finding to read rather than merely that a check failed.
        undead = tombstoned_axes()
        if undead:
            sys.stderr.write("readopted tombstoned axes: " + ", ".join(undead)
                             + "\n  mfma_shape: " + MFMA_SHAPE_IS_NOT_AN_AXIS
                             + "\n  occupancy_fill: removed by finding "
                             + str(OCCUPANCY_FILL_REMOVED_BY_FINDING) + "\n")
            return 3
        sys.stdout.write("every axis moves at least one tradeable quantity; "
                         "no tombstoned axis readopted\n")
        return 0
    _parser().error("nothing to do (pass --list-legal, --list-mechanisms or --check-axes)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
