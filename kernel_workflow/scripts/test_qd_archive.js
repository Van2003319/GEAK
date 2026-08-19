#!/usr/bin/env node
// Deterministic regression guard for GEAK's route-aware QD v2 archive (no GPU/model/network).
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const src = fs.readFileSync(path.join(ROOT, 'kernel_workflow', 'kernel_lane.js'), 'utf8');
// Finding (65): the lane produces a validation verdict and kernel_workflow.js consumes it to pick the
// bake-off winner. A guard that reads only the producer cannot see the consumer drop it on the floor.
const wfSrc = fs.readFileSync(path.join(ROOT, 'kernel_workflow', 'kernel_workflow.js'), 'utf8');
let failures = 0;
const ok = (cond, msg, detail) => {
  if (!cond) { console.error('  FAIL:', msg, detail == null ? '' : `-- ${detail}`); failures++; }
  else console.log('  ok:', msg);
};
const grab = (re, name) => {
  const m = src.match(re);
  if (!m) throw new Error(`could not extract ${name} from kernel_lane.js`);
  return m[0];
};
// The noise-floor declarations, as one runnable block: the by-machine table,
// the epoch pin, the selected table, the cross-machine default, and the lookup.
// Three separate sites used to spell this out as three or four `grab` calls
// each, so making the floor machine-keyed broke all of them independently --
// (57), every site. One constant now, used everywhere.
const FLOOR_BLOCK_RE =
  /const QD_NOISE_FLOOR_BY_MACHINE = new Map\([\s\S]*?const qdNoiseFloor = [^;]*;\n/;
// Same, but returns the first capture group. `grab` deliberately returns the
// whole match because most callers want a runnable block of source; the arch
// lists want just the array literal.
const grabGroup = (re, name) => {
  const m = src.match(re);
  if (!m || m[1] === undefined) throw new Error(`could not extract ${name} from kernel_lane.js`);
  return m[1];
};

console.log('\n# strategy routing remains opt-in');
const strategyBody = grab(/const SEARCH_STRATEGY = [\s\S]*?;\nconst QD_ENABLED[^\n]*\n/, 'SEARCH_STRATEGY');
const strategy = (A) => new Function('A', `${strategyBody}\nreturn {SEARCH_STRATEGY,QD_ENABLED};`)(A);
ok(strategy({}).SEARCH_STRATEGY === 'greedy', 'default is greedy');
ok(strategy({ search_strategy: ' qD_ArChIvE ' }).QD_ENABLED, 'qd_archive normalizes case/whitespace');
ok(strategy({ search_strategy: 'unknown' }).SEARCH_STRATEGY === 'greedy', 'unknown strategy safely falls back');
ok(/const QD_CLASSIFIER_VERSION = 'geak-qd-v2'/.test(src), 'classifier version is v2');

console.log('\n# sparse named descriptors and exact contexts');
const descriptorBlock = grab(/const QD_VOCAB = \{[\s\S]*?const qdCellId = \(contextId, d\) => \{[\s\S]*?\n\};\n/, 'QD descriptor helpers');
// QD_SUPPORTED_ARCHES/QD_MULTI_DIE_ARCHES are declared above the grabbed block,
// so they have to be injected or the sandbox throws ReferenceError. They are read
// back out of the source rather than restated, so the test cannot drift from it.
const archList = (name) => JSON.parse(grabGroup(
  new RegExp(`const ${name} = (\\[[^\\]]*\\]);`), name).replace(/'/g, '"'));
const SUPPORTED = archList('QD_SUPPORTED_ARCHES'), MULTI_DIE = archList('QD_MULTI_DIE_ARCHES');
ok(SUPPORTED.includes('gfx90a') && SUPPORTED.includes('gfx942') && MULTI_DIE.join() === 'gfx942',
  'v2 models exactly gfx90a and gfx942, and only gfx942 is multi-die');
const mkQd = (arch) => new Function('QD_ARCH', 'QD_DTYPE', 'QD_CONTEXT_IDS',
  'QD_SUPPORTED_ARCHES', 'QD_MULTI_DIE_ARCHES',
  `${descriptorBlock}\nreturn {qdDescriptorValid,qdCoverageEligible,qdHashValid,qdAdjacency,qdCellId,` +
  `qdDescriptorKey,qdDescriptorSame};`
)(arch, 'bf16', new Set(['decode_m8', 'prefill_m64']), SUPPORTED, MULTI_DIE);
const qd = mkQd('gfx90a');
const base = { compute_primitive: 'native_mfma', wave_schedule: 'independent',
  k_pipeline: 'lds_single', decomposition: 'tile_grid', output_path: 'direct_store',
  rasterization: 'grouped_m', plan_binding: 'static' };
ok(qd.qdDescriptorValid(base), 'legal gfx90a mechanism tuple is accepted');
ok(qd.qdCellId('decode_m8', base) ===
  'decode_m8|native_mfma|independent|lds_single|tile_grid|direct_store|grouped_m|static',
  'cell key is exact context plus named mechanisms');
ok(!qd.qdDescriptorValid({ ...base, rasterization: 'xcd_remapped_grouped' }),
  'single-die gfx90a cannot claim an XCD remap');
ok(!qd.qdDescriptorValid({ ...base, plan_binding: 'runtime_tuned' }),
  'runtime plan tuning needs a reduction whose plan there is something to tune');
ok(qd.qdCellId('M<64', base) === null, 'LLM-invented workload bucket is rejected');
ok(qd.qdCellId('decode_m8', { ...base, decomposition: 'split_k' }) === null,
  'reduction without fixup is rejected');
ok(qd.qdDescriptorValid({ ...base, decomposition: 'stream_k', output_path: 'workspace_fixup' }),
  'reduction with fixup is legal');
ok(!qd.qdDescriptorValid({ ...base, compute_primitive: 'valu', wave_schedule: 'symmetric_pingpong' }),
  'VALU cannot claim symmetric matrix-wave ping-pong');
ok(!qd.qdCoverageEligible({ ...base, wave_schedule: 'asymmetric_producer_consumer' }),
  'gfx90a producer/consumer is recorded but receives no coverage credit');
ok(qd.qdHashValid('a'.repeat(64)) && !qd.qdHashValid('') && !qd.qdHashValid('x'.repeat(64)),
  'source identities are canonical sha256 strings');
ok(/route\.classification_status !== 'classified'/.test(src),
  'only explicit verifier-classified routes can occupy cells');

// Finding (59). These execute qdRouteCells and qdSolCaseValid rather than
// grepping for them. Both defects below were invisible to every lexical check
// in this file, because the code that contained them read exactly right.
console.log('\n# route -> cell mapping and SOL card sanity, executed');
const cellFns = new Function('QD_ARCH', 'QD_DTYPE', 'QD_CONTEXT_IDS',
  'QD_SUPPORTED_ARCHES', 'QD_MULTI_DIE_ARCHES',
  `${descriptorBlock}\n` +
  grab(/const qdSolCaseValid = [\s\S]*?\n\};\n/, 'qdSolCaseValid') +
  grab(/const qdRouteCells = \(ver, rejects\) => \{[\s\S]*?\n\};\n/, 'qdRouteCells') +
  '\nreturn {qdRouteCells, qdSolCaseValid};'
)('gfx90a', 'bf16', new Set(['decode_m8', 'prefill_m64']), SUPPORTED, MULTI_DIE);
const route = (route_id, case_ids, descriptor) => ({ route_id, case_ids, descriptor,
  classification_status: 'classified' });
const cellsFor = (routes) => {
  const rejects = [];
  return { cells: cellFns.qdRouteCells({ route_descriptors: routes }, rejects), rejects };
};
const dupInOne = cellsFor([route('r1', ['decode_m8', 'decode_m8'], base)]);
ok(dupInOne.cells.length === 1 && dupInOne.rejects.length === 1 &&
   /already_claimed_by_r1/.test(dupInOne.rejects[0].reason),
  'one context listed twice yields one cell, and says so',
  JSON.stringify(dupInOne));
const dupAcross = cellsFor([route('r1', ['decode_m8'], base), route('r2', ['decode_m8'], base)]);
ok(dupAcross.cells.length === 1 && dupAcross.cells[0].route.route_id === 'r1' &&
   /already_claimed_by_r1/.test((dupAcross.rejects[0] || {}).reason),
  'two routes with the same mechanism on one context collapse to one cell, first wins',
  JSON.stringify(dupAcross));
ok(cellsFor([route('r1', ['decode_m8'], base), route('r2', ['prefill_m64'], base)]).cells.length === 2,
  'the same mechanism on two different contexts is two cells, not a collision');
ok(cellsFor([route('r1', ['decode_m8'], base),
             route('r2', ['decode_m8'], { ...base, k_pipeline: 'lds_multistage' })]).cells.length === 2,
  'two different mechanisms on one context are two cells');
// The dedupe must not become a way to lose a route to a typo'd axis value: an
// illegal descriptor is refused by the legality rules, with its own reason.
const illegal = cellsFor([route('r1', ['decode_m8'], { ...base, k_pipeline: 'lds_double' })]);
ok(illegal.cells.length === 0 && illegal.rejects.length === 1 &&
   !/already_claimed/.test(illegal.rejects[0].reason),
  'an axis value outside the vocabulary is rejected as illegal, not as a collision',
  JSON.stringify(illegal.rejects));
// sol_ms is a lower bound on time, so a gap below 1 is a broken model. The
// formula check cannot see it: such a card agrees with itself perfectly.
ok(cellFns.qdSolCaseValid({ name: 'decode_m8', measured_ms: 6, sol_ms: 2, sol_gap: 3,
  remaining_headroom: 1 - 1 / 3 }, 'decode_m8'), 'a normal SOL card is accepted');
ok(cellFns.qdSolCaseValid({ name: 'decode_m8', measured_ms: 2, sol_ms: 2, sol_gap: 1,
  remaining_headroom: 0 }, 'decode_m8'), 'a kernel exactly at SOL is accepted, headroom 0');
ok(!cellFns.qdSolCaseValid({ name: 'decode_m8', measured_ms: 1, sol_ms: 2, sol_gap: 0.5,
  remaining_headroom: -1 }, 'decode_m8'),
  'a self-consistent card claiming the kernel beat its own speed of light is refused');

// qd_score: per-context mean of per-cell means. The quantity summed must be the
// elite's speedup ON THAT CONTEXT and nothing else -- summing a suite-wide
// geomean into a per-context total is the (55) failure with a plausible number.
const scoreFn = new Function(grab(/const qdContextScore = \(cells\) => \{[\s\S]*?\n\};\n/,
  'qdContextScore') + '\nreturn qdContextScore;')();
const cell = (context_id, median, extra) => ({ cell: `${context_id}|c`, context_id,
  elite_id: `e_${context_id}_${median}`, robust: { median, score: median }, ...extra });
ok(Math.abs(scoreFn({ a: cell('decode_m8', 2), b: cell('prefill_m64', 4) }) - 3) < 1e-12,
  'two contexts with one cell each average to the mean of their medians');
ok(Math.abs(scoreFn({ a: { ...cell('decode_m8', 2), cell: 'x' },
                      b: { ...cell('decode_m8', 4), cell: 'y' },
                      c: cell('prefill_m64', 9) }) - 6) < 1e-12,  // (2+4)/2 and 9, then /2
  'two cells on one context are averaged first, so a well-covered context is not double-counted');
let threw = null;
try { scoreFn({ a: { cell: 'k', elite_id: 'e1', context_id: 'decode_m8', geomean: 100 } }); }
catch (e) { threw = String(e.message || e); }
ok(threw !== null && /robust\.median/.test(threw) && /admission guard/.test(threw),
  'a cell entry with no robust.median is refused loudly, not scored from its suite geomean',
  threw);

// (108). `structural_coverage` counts cells; this counts work. The real archive
// that motivated it read as 12 cells and was 2 variants, and nothing the
// planner could see distinguished that from 12 independent mechanisms.
const spreadFn = new Function(grab(/const qdVariantSpread = \(cells\) => \{[\s\S]*?\n\};\n/,
  'qdVariantSpread') + '\nreturn qdVariantSpread;')();
const vcell = (context_id, source_hash) => ({ cell: `${context_id}|c`, context_id,
  elite_id: `e_${context_id}`, source_hash });
{
  // Twelve cells, two variants, split 8/4 -- the shape of the real archive.
  const cells = {};
  for (let i = 0; i < 8; i++) cells[`a${i}`] = vcell(`ctx${i}`, 'cf4e8cceeb51');
  for (let i = 0; i < 4; i++) cells[`b${i}`] = vcell(`ctx${i}`, '80544dd4afe9');
  const s = spreadFn(cells);
  ok(s.cells_filled === 12 && s.distinct_variants === 2,
     'twelve filled cells holding two variants report both numbers, not just the twelve',
     JSON.stringify(s));
  ok(Math.abs(s.top_variant_share - 8 / 12) < 1e-12 &&
     Math.abs(s.cells_per_variant - 6) < 1e-12,
     'concentration is the top variant\'s share of filled cells', JSON.stringify(s));
}
{
  // The state the metric exists to distinguish FROM the one above. Same cell
  // count, opposite diversity -- if these two agreed the field would be a
  // comment (55).
  const cells = {};
  for (let i = 0; i < 12; i++) cells[`a${i}`] = vcell(`ctx${i}`, `hash${i}`);
  const s = spreadFn(cells);
  ok(s.cells_filled === 12 && s.distinct_variants === 12 &&
     Math.abs(s.top_variant_share - 1 / 12) < 1e-12,
     'twelve cells from twelve variants score identically on coverage and oppositely here',
     JSON.stringify(s));
}
{
  const s = spreadFn({});
  ok(s.distinct_variants === 0 && s.cells_filled === 0 && s.top_variant_share === 0 &&
     s.cells_per_variant === 0,
     'an empty archive divides by zero nowhere and reports zeros, not NaN',
     JSON.stringify(s));
}
{
  // Seed import is the legitimate 1.0 case: one artifact deliberately
  // referenced by every route-context cell. The metric must report it plainly
  // rather than treat it as a defect -- it is a gate on nothing.
  const cells = {};
  for (let i = 0; i < 11; i++) cells[`s${i}`] = vcell(`ctx${i}`, 'seedhash');
  const s = spreadFn(cells);
  ok(s.distinct_variants === 1 && s.top_variant_share === 1,
     'a freshly seeded archive reports one variant at full concentration',
     JSON.stringify(s));
}
{
  // A cell with no source_hash must not silently become its own variant --
  // that would make a broken archive look like a diverse one, which is the
  // inflation this finding is about, inverted.
  const s = spreadFn({ a: vcell('ctx0', 'h1'), b: { cell: 'b', context_id: 'ctx1' } });
  ok(s.distinct_variants === 1 && s.cells_filled === 2,
     'a hashless cell counts toward occupancy but never invents a variant',
     JSON.stringify(s));
}

console.log('\n# adjacency is legal and crosses reduction boundary atomically');
const persistent = { ...base, decomposition: 'persistent_output' };
const neighbors = qd.qdAdjacency(persistent);
const coupled = neighbors.filter(n => n.axes.length === 2);
ok(coupled.length === 2, 'persistent output has two coupled split-K/fixup neighbors', coupled.length);
ok(coupled.every(n => n.descriptor.decomposition === 'split_k' &&
  ['atomic_fixup', 'workspace_fixup'].includes(n.descriptor.output_path)),
  'coupled edges never create illegal reduction intermediates');
ok(neighbors.every(n => qd.qdDescriptorValid(n.descriptor)), 'every emitted neighbor is a legal cell');

const tuned = { ...base, decomposition: 'split_k', output_path: 'workspace_fixup',
  plan_binding: 'runtime_tuned' };
ok(qd.qdDescriptorValid(tuned), 'a reduction may bind its plan at runtime');
const released = qd.qdAdjacency(tuned).filter(n => n.direction === 'coupled');
ok(released.length === 2 && released.every(n => n.descriptor.plan_binding === 'static' &&
  n.axes.length === 3),
  'dropping the reduction releases runtime tuning in the same coupled step', released.length);

const qd942 = mkQd('gfx942');
ok(qd942.qdDescriptorValid({ ...base, rasterization: 'xcd_remapped_grouped' }),
  'a multi-die dispatcher may undo the die round-robin');

console.log('\n# (80) descriptor identity is a VALUE, not a key order');
// The defect this covers dispatched zero rounds on the 2026-08-16 smoke run while
// reporting a clean `rounds: 0`. Every descriptor comparison was
// `JSON.stringify(a) === JSON.stringify(b)`, which serialises keys in insertion
// order. `qdAdjacency` builds neighbours as `{...parent, [axis]: v}` so they carry
// the parent's (alphabetical, as persisted) order; the planner is an agent writing
// JSON and used the order it reasoned in. Every legal single-axis edge compared
// unequal to itself and the legality gate rejected all of them.
//
// `shuffled` is the SAME descriptor with the SAME values, written in the exact key
// order the planner emitted on that run.
const AXES = ['compute_primitive', 'wave_schedule', 'k_pipeline', 'decomposition',
  'output_path', 'rasterization', 'plan_binding'];
const reorder = (d, order) => { const o = {}; for (const k of order) o[k] = d[k]; return o; };
const alpha = reorder(base, [...AXES].sort());
const shuffled = reorder(base, AXES);
ok(JSON.stringify(alpha) !== JSON.stringify(shuffled),
  'the two orderings really are different byte strings -- i.e. this test can fail');
ok(qd.qdDescriptorSame(alpha, shuffled),
  'the same descriptor written in two key orders is ONE descriptor');
ok(qd.qdDescriptorKey(alpha) === qd.qdDescriptorKey(shuffled),
  'the canonical key is independent of key order');
// The regression proper: the planner's edge, in the planner's key order, against
// neighbours built in the parent's key order. This is the exact comparison at
// kernel_lane.js:2056 and it is what returned false for both directions.
const parentDesc = reorder({ ...base, k_pipeline: 'lds_single' }, [...AXES].sort());
const plannedEdge = reorder({ ...base, k_pipeline: 'lds_reg_prefetch' }, AXES);
const legalTargets = qd.qdAdjacency(parentDesc);
ok(legalTargets.some(n => qd.qdDescriptorSame(n.descriptor, plannedEdge)),
  'a legal single-axis edge is recognised as legal when the planner writes its keys in another order');
ok(!legalTargets.some(n => JSON.stringify(n.descriptor) === JSON.stringify(plannedEdge)),
  'and the old JSON.stringify comparison still rejects it -- this is the bug, pinned');
// The sibling gate failed the other way: a no-op edge written in a different key
// order was not recognised as a no-op and would have bought a build-and-verify
// cycle to re-measure the parent.
ok(qd.qdDescriptorSame(parentDesc, reorder(parentDesc, AXES)),
  'a target identical to the parent is caught as a no-op regardless of key order');
// Values, not just ordering. Reordering must not make two DIFFERENT descriptors equal.
ok(!qd.qdDescriptorSame(alpha, reorder({ ...base, k_pipeline: 'lds_pingpong' }, AXES)),
  'canonicalisation does not collapse genuinely different mechanisms');
// A non-descriptor has no identity, and two of them are not equal to each other:
// `===` on two nulls would have made all garbage identical.
ok(qd.qdDescriptorKey({ compute_primitive: 'native_mfma' }) === null &&
   qd.qdDescriptorKey(null) === null,
  'an incomplete or absent descriptor has no canonical key');
ok(!qd.qdDescriptorSame(null, null) && !qd.qdDescriptorSame({}, {}),
  'two non-descriptors are not equal -- unequal is the fail-closed answer');
ok(!qd.qdDescriptorSame(base, { ...base, k_pipeline: 'not_a_real_stage' }),
  'an out-of-vocabulary axis value has no key and matches nothing');
// Extra keys are ignored on purpose: the descriptor IS its seven axes, and an agent
// decorating one with a note must not thereby invent a new mechanism (which, since
// the capsule ledger is keyed on it, would file one direction under two names and
// defeat the "already failed twice on this route" block from (76)).
ok(qd.qdDescriptorSame(base, { ...base, note: 'why I picked this' }),
  'a decorated descriptor is the same mechanism, so the capsule ledger keys it once');

console.log('\n# per-context robust admission is recomputed from raw timing samples');
const robustBlock = grab(/const qdMedian = \(xs\) => \{[\s\S]*?const qdCaseRobust = \(ver, contextId\) => \{[\s\S]*?\n\};\n/, 'QD robust helpers');
// The noise floor (finding 58) is declared above the grabbed block, like the
// arch lists. Pulled from the source rather than restated so the sandbox cannot
// silently run a different floor than the lane does.
const floorBlock = grab(FLOOR_BLOCK_RE, 'QD noise floor table');
const robust = new Function('QD_BASELINE_MS',
  `${floorBlock}\n${robustBlock}\nreturn {qdCaseRobust, qdNoiseFloor, QD_DEFAULT_NOISE_FLOOR,`
  + ` QD_NOISE_FLOOR_BY_MACHINE, QD_CURRENT_MACHINE, QD_PROVISIONAL_MACHINES,`
  + ` qdFloorIsMeasured};`
)(new Map([['decode_m8', 10], ['prefill_m64', 20], ['decode_m2_square', 10]]));
const measurement = {
  per_case: [
    // Deliberately malicious verifier-returned baseline/speedup: admission must ignore both.
    { name: 'decode_m8', baseline_ms: 1000, optimized_ms: 5, speedup: 999 },
    { name: 'prefill_m64', baseline_ms: 2000, optimized_ms: 10, speedup: 999 },
  ],
  case_measurement_samples: [
    { name: 'decode_m8', samples: [5, 4, 10], median: 999, mad: 0, lower: 999, upper: 999 },
    { name: 'prefill_m64', samples: [10, 10, 10], median: 999, mad: 0, lower: 999, upper: 999 },
  ],
};
const decodeStats = robust.qdCaseRobust(measurement, 'decode_m8');
ok(decodeStats.score === 2 && decodeStats.median === 2,
  'score and median derive from raw latencies plus authoritative baseline');
ok(decodeStats.mad === 0.5 && decodeStats.lower === 1 && decodeStats.upper === 3,
  'bounds use median ± 2 raw MAD and ignore all model-supplied statistics', JSON.stringify(decodeStats));
ok(robust.qdCaseRobust(measurement, 'unknown') === null,
  'missing authoritative baseline cannot silently become a point interval');
ok(robust.qdCaseRobust({ case_measurement_samples: [{ name: 'decode_m8', samples: [5, 5] }] },
  'decode_m8') === null, 'fewer than three repeated samples fail closed');

// Finding (58). `median +- 2*MAD` is not a 95% interval; at n=3 MAD is the
// smaller of the two gaps, so three samples that happen to agree present a
// bound far tighter than the route's own run-to-run spread -- and admission
// compares bounds.
// This file deliberately does NOT pin the epoch's concrete floor values. It used
// to assert `FLOOR_M2 === 0.0378 && FLOOR_M256 === 0.0137`, machine N's numbers,
// and the moment the container moved to machine O those three literals became a
// THIRD copy of a table that already exists twice -- and the copy that fails is
// the one nobody thinks to update, so the epoch bump reads as "the JS suite is
// broken" rather than "the floors moved". `test_qd_lane_parity.py` is what pins
// the values, by comparing the lane's table to `qd_robust_stats.py` entry by
// entry and asserting the two CURRENT_MACHINE constants agree. What is left for
// this file is the part parity cannot see: that the lookup is per-route at all.
//
// Machine Q added the case this file could not previously express: a
// PROVISIONAL epoch, whose table is a fail-closed placeholder at one value on
// every route. "Not the same number" is then false for a correct table, so it
// is asked only of an epoch that was actually measured; on a provisional one
// the requirement is the opposite and just as checkable -- every route sits at
// the cross-machine default, and nothing claims to be measured.
const PROVISIONAL = robust.QD_PROVISIONAL_MACHINES.has(robust.QD_CURRENT_MACHINE);
const FLOOR_M2 = robust.qdNoiseFloor('decode_m2_square');
const FLOOR_M256 = robust.qdNoiseFloor('prefill_m256_down');
ok(Number.isFinite(FLOOR_M2) && FLOOR_M2 > 0 &&
   Number.isFinite(FLOOR_M256) && FLOOR_M256 > 0 &&
   (PROVISIONAL ? FLOOR_M2 === robust.QD_DEFAULT_NOISE_FLOOR &&
                  FLOOR_M256 === robust.QD_DEFAULT_NOISE_FLOOR
                : FLOOR_M2 !== FLOOR_M256),
  'each route carries its own noise floor -- a positive number, and on a measured '
  + 'epoch not the same number');
ok(PROVISIONAL
   ? !robust.qdFloorIsMeasured('decode_m2_square') &&
     !robust.qdFloorIsMeasured('prefill_m256_down')
   : robust.qdFloorIsMeasured('decode_m2_square') &&
     !robust.qdFloorIsMeasured('a_route_nobody_measured'),
  'a floor reports itself measured only when the epoch it came from was measured -- '
  + 'table membership cannot answer this, because a provisional table is complete');
// Widest ANYWHERE, not widest on this machine: an unmeasured route belongs to no
// epoch, so it has no claim on this epoch's spread. Computed from the lane's own
// table rather than restated as a literal, so it survives an epoch change --
// which is exactly when an unmeasured route is most likely to appear.
const allFloors = [...robust.QD_NOISE_FLOOR_BY_MACHINE.values()]
  .flatMap(t => [...t.values()]);
const widestAnywhere = Math.max(...allFloors);
const widestHere = Math.max(
  ...[...robust.QD_NOISE_FLOOR_BY_MACHINE.get(robust.QD_CURRENT_MACHINE).values()]);
// Without this the check below cannot tell "widest anywhere" from "widest here":
// if some future epoch is the widest on every route the two rules coincide and
// the assertion passes for the wrong reason. Fail loudly at that point instead.
//
// That is a question about the RULE, and asking it of the current epoch made it
// a question about which host we are on: it is false by construction on a
// provisional epoch (flat at the default) and on whichever epoch supplies the
// max. Ask whether SOME epoch separates the two rules; where the current one
// does, ask it live as well.
ok([...robust.QD_NOISE_FLOOR_BY_MACHINE.values()]
     .some(t => Math.max(...t.values()) < widestAnywhere),
  'some epoch is narrower than the cross-machine default, so "widest anywhere" and '
  + '"widest here" are distinguishable rules at all');
ok(widestHere <= widestAnywhere,
  'this epoch\'s widest floor is not wider than the cross-machine maximum -- if it '
  + 'were, the default would be narrower than a floor actually in use');
ok(robust.qdNoiseFloor('a_route_nobody_measured') === robust.QD_DEFAULT_NOISE_FLOOR &&
   robust.QD_DEFAULT_NOISE_FLOOR === widestAnywhere,
  'an unmeasured context gets the widest floor measured on any machine, which is the fail-closed direction');
// And lexically, because the check above cannot see the difference on every
// epoch: when the CURRENT epoch happens to be the widest -- which a provisional
// one always is, by construction -- "widest anywhere" and "widest here" produce
// the same number and narrowing the rule to this epoch's own table is
// undetectable by execution. It is not undetectable by reading.
ok(/const QD_DEFAULT_NOISE_FLOOR = Math\.max\(\s*\n?\s*\.\.\.\[\.\.\.QD_NOISE_FLOOR_BY_MACHINE\.values\(\)\]/
     .test(src),
  'the default floor is still computed across every machine\'s table, not from the '
  + 'current epoch\'s -- a narrower default fails OPEN and admits noise as an elite');
// Three identical samples: MAD is exactly 0, which used to collapse the
// interval to a point and make replacement a coin flip.
const collapsed = robust.qdCaseRobust(
  { case_measurement_samples: [{ name: 'decode_m2_square', samples: [5, 5, 5] }] },
  'decode_m2_square');
ok(collapsed.mad === 0 && collapsed.lower < collapsed.median && collapsed.upper > collapsed.median,
  'a zero-MAD sample set still carries a non-degenerate interval', JSON.stringify(collapsed));
ok(Math.abs(collapsed.upper - collapsed.median * (1 + FLOOR_M2)) < 1e-12 &&
   Math.abs(collapsed.median - collapsed.lower - collapsed.median * FLOOR_M2) < 1e-12,
  'the collapsed interval is exactly the route floor wide');
// ...and where 2*MAD is genuinely wider than the floor, MAD still wins.
const wide = robust.qdCaseRobust(
  { case_measurement_samples: [{ name: 'decode_m8', samples: [5, 4, 10] }] }, 'decode_m8');
ok(wide.median === 2 && wide.mad === 0.5 && wide.lower === 1 && wide.upper === 3,
  'the floor raises a narrow interval and never shrinks a wide one', JSON.stringify(wide));

console.log('\n# opening an empty cell is a distinct outcome from beating an incumbent (99)');
// `improved` is `replacement && !!incumbent && score > incumbent.score`. It is
// correct, and it structurally cannot be true for a mechanism that fills cells
// nothing had reached -- which is what round 1's best mechanism did, eight
// times, at 1.72x-2.29x. `improved` is the field the next planner reads (76).
ok(/opened_empty_cell: replacement && !incumbent/.test(src),
  'the lane records cell openings as their own outcome');
ok(/improved: replacement && !!incumbent && robust\.score > incumbent\.robust\.score/.test(src),
  'and `improved` still means what it said -- the repair is a second flag, not a wider first one');
{
  // The two flags evaluated the way the lane evaluates them, over the three
  // cases that matter.
  const flags = (replacement, incumbent, score) => ({
    improved: replacement && !!incumbent && score > (incumbent ? incumbent.robust.score : 0),
    opened_empty_cell: replacement && !incumbent,
  });
  const opening = flags(true, null, 2.29);
  ok(opening.opened_empty_cell === true && opening.improved === false,
    'a 2.29x opening is flagged as an opening and not as an improvement');
  const beating = flags(true, { robust: { score: 1.2 } }, 1.8);
  ok(beating.improved === true && beating.opened_empty_cell === false,
    'beating an incumbent is an improvement and not an opening');
  const refused = flags(false, { robust: { score: 1.2 } }, 1.9);
  ok(refused.improved === false && refused.opened_empty_cell === false,
    'a refused candidate is neither, however good its median looked');
}
// The planner side -- that the tech lead prompt names the flag and says which
// way to read it -- is checked in test_qd_lane_parity.py. This runner is fenced
// to the two lane files and cannot read a role prompt, which is the right fence
// to keep: a suite that can read anything drifts into testing everything.

console.log('\n# the seed carries a real interval, not a point (95)');
// The seed's speedup is 1 by construction -- it IS the baseline -- and the lane
// used to encode that as `{median:1, mad:0, lower:1, upper:1}`. Admission
// compares intervals, so a zero-width one is one-sided: any candidate over 1.0
// displaces the seed. That is how a 1.0091 median was admitted on a route whose
// own noise floor is 0.007. (That 0.007 is a machine-L number, and the epoch has
// since moved: the same route measures 0.0198 on machine N. The regression is
// re-derived from whatever the current table says, below, so it keeps testing
// the arithmetic rather than a frozen pair of digits -- but note the direction,
// because it matters: a WIDER floor makes the seed harder to displace, so if
// this check ever starts passing for the wrong reason it will be because the
// floor shrank, not because the rule got stricter.)
const seedBlock = grab(/const qdSeedRobust = \(contextId\) => \{[\s\S]*?const qdSeedSuiteRobust = \(\) => \{[\s\S]*?\n\};\n/, 'qdSeedRobust');
const mkSeed = (baselineMs, samples, contexts) => new Function(
  'QD_BASELINE_MS', 'QD_BASELINE_SAMPLES', 'QD_CONTEXT_IDS',
  `${floorBlock}\n${robustBlock}\n${seedBlock}\nreturn {qdSeedRobust, qdSeedSuiteRobust};`
)(new Map(baselineMs), new Map(samples), new Set(contexts));

// (b) no repeats recorded: the interval is floored at the route's own spread.
const floored = mkSeed([['decode_m64_square', 10]], [], ['decode_m64_square'])
  .qdSeedRobust('decode_m64_square');
const FLOOR_M64 = robust.qdNoiseFloor('decode_m64_square');
ok(floored.median === 1 && floored.mad === 0 && FLOOR_M64 > 0 &&
   Math.abs(floored.upper - (1 + FLOOR_M64)) < 1e-12 &&
   Math.abs(floored.lower - (1 - FLOOR_M64)) < 1e-12,
  'a seed with no repeats gets an interval exactly the route floor wide', JSON.stringify(floored));
ok(/^floored_at_route_noise_floor/.test(floored.interval_provenance),
  'and says so, so an audit can tell a floored interval from a measured one');

// The concrete regression: r1_s1 was admitted at a 1.0091 median on this route.
// The rule it violated is "a median inside the route's own band is not a win",
// and that is what gets asserted here -- with a challenger built FROM the
// ambient floor rather than from the frozen 1.0091. The frozen number stopped
// being an in-band example the moment the epoch moved to a machine whose
// decode_m64_square floor is tighter than 0.9%: on epoch W the floor is 0.0042,
// so 1.0091 really is above the noise there and the lane is right to admit it.
// Asserting the refusal of a fixed median therefore tested the epoch, not the
// arithmetic. Same defect family as a test that hardcodes a real hostname as
// its example of an unregistered host: never pin a literal that the registry
// the restores mutate can move out from under.
const inBand = 1 + FLOOR_M64 * 0.9;
const r1s1 = { median: inBand, lower: inBand * (1 - FLOOR_M64), upper: inBand * (1 + FLOOR_M64) };
ok(!(r1s1.lower > floored.upper),
  'a median inside the route band -- what (95) admitted -- does not clear the seed',
  `challenger.median=${inBand} lower=${r1s1.lower} vs seed.upper=${floored.upper}`);
// And the historical number itself, checked only where it is still in-band.
if (1.0091 <= 1 + FLOOR_M64) {
  ok(!((1.0091 * (1 - FLOOR_M64)) > floored.upper),
    'the 1.0091 admission that (95) produced no longer clears the seed either',
    `floor=${FLOOR_M64}`);
}
// ...and a mechanism that really moves the route still gets in.
ok((1.81 * (1 - FLOOR_M64)) > floored.upper,
  'a 1.81x direct_store result still clears it, so the floor is not a wall');

// (a) repeats recorded: the interval is measured, through the same arithmetic.
const measured = mkSeed([['decode_m8', 10]],
  [['decode_m8', [10, 9, 11]]], ['decode_m8']).qdSeedRobust('decode_m8');
ok(Math.abs(measured.median - 1) < 1e-12 && measured.mad > 0,
  'baseline repeats give the seed a measured spread', JSON.stringify(measured));
ok(/^measured: 3 primed baseline repeats/.test(measured.interval_provenance),
  'and the provenance names how many repeats it rests on');
// ...and WHERE. A floor is a reading of a box (26)/(58), so an interval is too:
// two epochs' seed intervals are not comparable, and one quoted out of an
// archive with no epoch on it has to be dated by hand from the run id. Both
// outcomes carry it, because the one that gets quoted is the one that does not.
const EPOCH_RE = new RegExp(`\\[epoch ${robust.QD_CURRENT_MACHINE} \\(`);
ok(EPOCH_RE.test(measured.interval_provenance),
  'a measured interval names the epoch it was measured on',
  measured.interval_provenance);
ok(EPOCH_RE.test(floored.interval_provenance),
  'and so does a floored one -- the floor is epoch-specific too',
  floored.interval_provenance);
ok(robust.QD_PROVISIONAL_MACHINES.has(robust.QD_CURRENT_MACHINE)
   === /floor table PROVISIONAL/.test(measured.interval_provenance),
  'the stamp says whether the floor table behind the interval was ever measured (126)',
  measured.interval_provenance);
// A measured spread wider than the table wins; the floor never shrinks it.
ok(measured.upper > 1.072,
  'a genuinely noisy baseline is not narrowed to the tabulated floor',
  JSON.stringify(measured));

// Fewer than three repeats is not a measurement. It falls back rather than
// computing a MAD from two points, which is the same fail-closed rule
// `qdCaseRobust` applies to candidates.
const twoOnly = mkSeed([['decode_m8', 10]], [['decode_m8', [10, 9]]], ['decode_m8'])
  .qdSeedRobust('decode_m8');
ok(twoOnly.mad === 0 && /^floored/.test(twoOnly.interval_provenance),
  'two repeats fall back to the floor instead of inventing a spread');

// The suite roll-up averages the routes rather than taking the worst.
const suite = mkSeed([['decode_m2_square', 10], ['prefill_m256_down', 10]], [],
  ['decode_m2_square', 'prefill_m256_down']).qdSeedSuiteRobust();
// The `FLOOR_M2 !== FLOOR_M256` half is what makes "mean" distinguishable from
// "widest"; with equal floors the two rules give the same number and the check
// would pass either way. It can only be asked of a measured epoch, so on a
// provisional one the mean is still verified and the distinguishing case is
// made explicitly, out of the two floors' own epoch-independent arithmetic.
ok(Math.abs(suite.upper - ((1 + FLOOR_M2) + (1 + FLOOR_M256)) / 2) < 1e-12 &&
   (PROVISIONAL || FLOOR_M2 !== FLOOR_M256),
  'the seed suite interval is the mean of its routes, not the widest', JSON.stringify(suite));
{
  // Same roll-up, on two routes with deliberately different floors: an
  // unmeasured route (the cross-machine default) and a measured one. Their
  // floors differ on every epoch, provisional or not, unless the measured
  // route IS at the default -- which is exactly when the check cannot be made
  // and says so.
  const known = 'decode_m8_up';
  const wide = robust.qdNoiseFloor('a_route_nobody_measured');
  const narrow = robust.qdNoiseFloor(known);
  const mixed = mkSeed([[known, 10], ['a_route_nobody_measured', 10]], [],
    [known, 'a_route_nobody_measured']).qdSeedSuiteRobust();
  ok(narrow === wide ||
     (Math.abs(mixed.upper - ((1 + narrow) + (1 + wide)) / 2) < 1e-12 &&
      mixed.upper < 1 + wide),
    'a roll-up over two DIFFERENT floors lands on their mean, strictly inside the '
    + 'wider of the two', JSON.stringify(mixed));
}
ok(/floored_at_route_noise_floor on at least one context/.test(suite.interval_provenance),
  'one unmeasured route is enough to mark the whole roll-up floored');
ok(EPOCH_RE.test(suite.interval_provenance),
  'and the roll-up carries the epoch too', suite.interval_provenance);
{
  // The measured branch of the roll-up. Both outcomes are separate string
  // literals in the lane, so a stamp on one of them is not a stamp on the
  // other, and the unstamped one is the one that eventually gets quoted.
  const allMeasured = mkSeed([['decode_m8', 10], ['prefill_m64', 20]],
    [['decode_m8', [10, 9, 11]], ['prefill_m64', [20, 19, 21]]],
    ['decode_m8', 'prefill_m64']).qdSeedSuiteRobust();
  ok(/^measured: primed baseline repeats on every context/
       .test(allMeasured.interval_provenance)
     && EPOCH_RE.test(allMeasured.interval_provenance),
    'a fully measured roll-up says so AND names the epoch',
    allMeasured.interval_provenance);
}

// The literal must not come back.
ok(!/robust:\s*\{\s*score:\s*1,\s*median:\s*1,\s*mad:\s*0,\s*lower:\s*1,\s*upper:\s*1\s*\}/.test(src),
  'the hardcoded zero-width seed interval is gone from the lane');

console.log('\n# one artifact may back many route/context cells');
ok(/archive_dir: `\$\{EVAL_DIR\}\/qd_archive`/.test(src) &&
   /artifacts\/\$\{sourceHash\}\/workspace/.test(src), 'artifacts are content-addressed by source hash');
ok(/for \(const rc of routeCells\)/.test(src) && /artifact: sourceHash/.test(src),
  'one verified source expands into route/context cell references');
ok(/const routeCells = qdRouteCells\(r\.ver, rejects\)/.test(src) &&
   /const routes = Array\.isArray\(ver && ver\.route_descriptors\) \? ver\.route_descriptors : \[\]/.test(src),
  'main admission trusts verifier routes only with no fallback path');
// The emptiness guard is an early return rather than a conjunct, so it is
// asserted separately from the sample requirement. Matching them as one
// expression is what made this check stale the first time.
ok(/if \(!routeCells\.length\) \{[\s\S]{0,600}?return false;/.test(src),
  'a candidate that classifies into no cell is not admitted');
// Finding (60). This was a regex over the old inline conjunction; it is now an
// executed check on the shared helper, and it asserts the REASONS as well as the
// refusals. Both admission paths must call it, so the site count is asserted too.
{
  const CTX = ['decode_m8', 'prefill_m64'];
  // ORACLE_DIGEST is a run-level pin (67); these cases are about the clauses behind it, so the
  // sandbox runs pinned and every `ver` below carries the matching digest.
  const PINNED = 'c'.repeat(64);
  const mkAdm = (hasWorkload) => new Function('QD_CONTEXT_IDS', 'QD_BASELINE_MS', 'QD_CELL_GUARDRAIL',
    'HAS_WORKLOAD', 'ORACLE_DIGEST',
    // (81): oracleDrift now names the empty-file-set constant as its own refusal,
    // so the sandbox needs the table it consults.
    grab(/const ORACLE_DEGENERATE = new Set\(\[[\s\S]*?\n\]\);\n/, 'ORACLE_DEGENERATE') +
    grab(/const oracleDrift = \(o\) => \{[\s\S]*?\n\};\n/, 'oracleDrift') +
    grab(/const primWeighted = \(o\) => \{[\s\S]*?\n\};\n/, 'primWeighted') +
    grab(/const primMetricReason = \(o\) =>[\s\S]*?: null\);\n/, 'primMetricReason') +
    grab(FLOOR_BLOCK_RE, 'QD noise floor table') +
    grab(/const qdMedian = \(xs\) => \{[\s\S]*?const qdCaseRobust = \(ver, contextId\) => \{[\s\S]*?\n\};\n/,
      'qdCaseRobust') +
    grab(/const qdRobust = \(ver\) => \{[\s\S]*?\n\};\n/, 'qdRobust') +
    grab(/const qdMinCase = \(ver\) => \{[\s\S]*?\n\};\n/, 'qdMinCase') +
    grab(/const qdAdmissionCheck = \(ver, routeCells\) => \{[\s\S]*?\n\};\n/, 'qdAdmissionCheck') +
    '\nreturn qdAdmissionCheck;'
  )(new Set(CTX), new Map([['decode_m8', 10], ['prefill_m64', 20]]), 0.80, hasWorkload, PINNED);
  const adm = mkAdm(false);
  // baseline/candidate = speedup, so equal times are 1.00 and a 4x slowdown is 0.25.
  const ver = (perCtx) => ({ oracle_digest: PINNED,
    case_measurement_samples: CTX.map(name => ({ name, samples: perCtx[name] })) });
  const cells = CTX.map(c => ({ context_id: c }));
  const three = { decode_m8: [10, 10, 10], prefill_m64: [20, 20, 20] };
  ok(adm(ver(three), cells).suiteRobust && !adm(ver(three), cells).reason,
    'a candidate measured three times on every context is admitted');
  const thin = adm(ver({ ...three, prefill_m64: [20, 20] }), cells);
  ok(!thin.suiteRobust && /^measurement:/.test(thin.reason || '') && /3 usable samples/.test(thin.reason),
    'two samples on one context is refused as a MEASUREMENT fault, not a slow kernel',
    JSON.stringify(thin));
  const slow = adm(ver({ ...three, prefill_m64: [80, 80, 80] }), cells);
  ok(!slow.suiteRobust && /^performance:/.test(slow.reason || '') &&
     /0\.2500/.test(slow.reason) && /0\.8/.test(slow.reason),
    'a real regression is refused as a PERFORMANCE verdict, naming the worst speedup and the guardrail',
    JSON.stringify(slow));
  ok(thin.reason !== slow.reason && !/^measurement:/.test(slow.reason),
    'the two refusals are distinguishable in the round log, which is the whole point');
  // Finding (62): a third kind of refusal, and a third distinct prefix. A perfectly
  // measured, perfectly fast candidate is still not admissible if the number it would
  // be filed under is the wrong quantity.
  const admW = mkAdm(true);
  const noWeight = admW(ver(three), cells);
  ok(!noWeight.suiteRobust && /^metric:/.test(noWeight.reason || ''),
    'on a workload-aligned run a report with no weighted speedup is refused as a METRIC fault',
    JSON.stringify(noWeight));
  // Finding (67): a fourth, at the same gate. The archive outlives the run, so an elite whose
  // denominator cannot be tied to the pinned oracle would export this run's doubt into every
  // future warm start.
  const noOracle = adm({ ...ver(three), oracle_digest: undefined }, cells);
  ok(!noOracle.suiteRobust && /^oracle:digest_missing/.test(noOracle.reason || ''),
    'a perfectly measured, perfectly fast candidate is refused if its denominator is unpinnable',
    JSON.stringify(noOracle));
  ok(new Set([thin.reason, slow.reason, noWeight.reason, noOracle.reason]).size === 4,
    'measurement, performance, metric and oracle faults are four distinguishable refusals');
  // Finding (81). The prescribed digest command named four paths from the generic
  // task layout; on a task with none of them `find` matched nothing, `xargs` ran
  // `sha256sum` with no arguments, and the pipeline returned a well-formed constant
  // covering zero bytes. Pinned and compared to itself it certifies any oracle as
  // immutable. These two values are the hash of nothing and the hash of the line
  // sha256sum prints for nothing.
  const EMPTY_STDIN = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855';
  const EMPTY_FILESET = 'abcfa6a9d4df344d1781bc2560b5e4cdcae08b39ed303063535e7e1e926a304a';
  for (const [name, digest] of [['empty stdin', EMPTY_STDIN], ['empty file set', EMPTY_FILESET]]) {
    const degen = adm({ ...ver(three), oracle_digest: digest }, cells);
    ok(!degen.suiteRobust && /^oracle:digest_empty_fileset/.test(degen.reason || ''),
      `a digest that is the hash of nothing (${name}) is refused as an empty file set`,
      JSON.stringify(degen));
    // The whole point of the separate refusal: the run did NOT observe drift, and
    // saying so would be a false and maximally alarming diagnosis of the oracle.
    ok(!/digest_drift/.test(degen.reason || ''),
      `a degenerate digest (${name}) is not misreported as the oracle having changed`);
  }
  // And the case that motivated the constant: had BOTH sides returned it, the old
  // code would have compared it to itself and passed. It must not.
  const selfConsistentEmpty = new Function('QD_CONTEXT_IDS', 'QD_BASELINE_MS', 'QD_CELL_GUARDRAIL',
    'HAS_WORKLOAD', 'ORACLE_DIGEST',
    grab(/const ORACLE_DEGENERATE = new Set\(\[[\s\S]*?\n\]\);\n/, 'ORACLE_DEGENERATE') +
    grab(/const oracleDrift = \(o\) => \{[\s\S]*?\n\};\n/, 'oracleDrift') +
    '\nreturn oracleDrift;'
  )(new Set(CTX), new Map(), 0.80, false, EMPTY_FILESET)({ oracle_digest: EMPTY_FILESET });
  ok(/^oracle:digest_empty_fileset/.test(selfConsistentEmpty || ''),
    'an empty-file-set digest that MATCHES the pin is still refused -- agreement between two ' +
    'measurements of nothing is not evidence that the oracle is intact');
  const weighted = admW({ ...ver(three), verified_weighted: 1.3 }, cells);
  ok(!weighted.reason && weighted.suiteRobust,
    'the same candidate is admitted once it reports the weighted number');
  // Finding (63): the value the archive stores as `min_case` is the value the gate
  // ran on, returned, not re-derived at the storing site.
  const admitted = adm(ver(three), cells);
  ok(Math.abs(admitted.minCase - 1) < 1e-12 &&
     Math.abs(adm(ver({ ...three, prefill_m64: [25, 25, 25] }), cells).minCase - 0.8) < 1e-12,
    'the admission result carries the worst-context speedup it gated on',
    JSON.stringify(admitted));
}
// Round 2's `r2_s0` is the case the two-context sandbox above cannot express. It held
// nine of eleven routes inside +-2.6% of the parent and cratered the two down-projection
// prefills to 0.2336 and 0.5355 -- so the SUITE mean of the per-context medians was
// 0.887, and had the guardrail been read off the suite it would have been a near miss
// rather than a refusal. With only two contexts one route at 0.25 drags the mean to
// 0.625 and both readings agree, which means the existing cases would keep passing
// against a gate that had quietly moved from the worst context to the average of them.
// The worst-context rule exists precisely because a suite average is an average: it pays
// for a killed route out of the ten that still work. This is the cross-shape geomean
// target's whole reason for having a per-route floor underneath it.
{
  const CTX = Array.from({ length: 11 }, (_, i) => `ctx_${i}`);
  const PINNED = 'c'.repeat(64);
  const adm = new Function('QD_CONTEXT_IDS', 'QD_BASELINE_MS', 'QD_CELL_GUARDRAIL',
    'HAS_WORKLOAD', 'ORACLE_DIGEST',
    grab(/const ORACLE_DEGENERATE = new Set\(\[[\s\S]*?\n\]\);\n/, 'ORACLE_DEGENERATE') +
    grab(/const oracleDrift = \(o\) => \{[\s\S]*?\n\};\n/, 'oracleDrift') +
    grab(/const primWeighted = \(o\) => \{[\s\S]*?\n\};\n/, 'primWeighted') +
    grab(/const primMetricReason = \(o\) =>[\s\S]*?: null\);\n/, 'primMetricReason') +
    grab(FLOOR_BLOCK_RE, 'QD noise floor table') +
    grab(/const qdMedian = \(xs\) => \{[\s\S]*?const qdCaseRobust = \(ver, contextId\) => \{[\s\S]*?\n\};\n/,
      'qdCaseRobust') +
    grab(/const qdRobust = \(ver\) => \{[\s\S]*?\n\};\n/, 'qdRobust') +
    grab(/const qdMinCase = \(ver\) => \{[\s\S]*?\n\};\n/, 'qdMinCase') +
    grab(/const qdAdmissionCheck = \(ver, routeCells\) => \{[\s\S]*?\n\};\n/, 'qdAdmissionCheck') +
    '\nreturn qdAdmissionCheck;'
  )(new Set(CTX), new Map(CTX.map(c => [c, 100])), 0.80, false, PINNED);
  const cells = CTX.map(c => ({ context_id: c }));
  // Ten routes at 1.05 and one at 0.30. Mean of the medians = 0.9955, which clears the
  // 0.95 canonical guardrail; worst context 0.30, which does not clear 0.80.
  const times = Object.fromEntries(CTX.map(c => [c, [100 / 1.05, 100 / 1.05, 100 / 1.05]]));
  times[CTX[7]] = [100 / 0.30, 100 / 0.30, 100 / 0.30];
  const ver = { oracle_digest: PINNED,
    case_measurement_samples: CTX.map(name => ({ name, samples: times[name] })) };
  const verdict = adm(ver, cells);
  // Positive control on the arithmetic first: an assertion that the suite mean is high
  // proves nothing if the sandbox is silently returning nulls, and the refusal below
  // would then be passing for the wrong reason.
  const healthy = adm({ oracle_digest: PINNED, case_measurement_samples:
    CTX.map(name => ({ name, samples: [100 / 1.05, 100 / 1.05, 100 / 1.05] })) }, cells);
  ok(healthy.suiteRobust && !healthy.reason && Math.abs(healthy.suiteRobust.median - 1.05) < 1e-9,
    'the eleven-context sandbox measures a uniformly healthy candidate at 1.05',
    JSON.stringify(healthy));
  ok(!verdict.suiteRobust && /^performance:/.test(verdict.reason || '') &&
     /0\.3000/.test(verdict.reason),
    'one cratered route out of eleven is refused on the worst context even though the ' +
    'suite mean clears every suite-level guardrail',
    JSON.stringify(verdict));
  // The number the refusal must NOT be reporting. If this ever appears in the reason the
  // gate has been rewritten to average, and the message would be actively misleading:
  // it would name a value no single route ever measured.
  ok(!/0\.99[0-9]/.test(verdict.reason || ''),
    'the refusal names the route that failed, not the average that hid it',
    JSON.stringify(verdict));
  // On the ADMIT path the worst-context value is carried out as a field (63, asserted
  // above). On the refuse path it is not -- there is no admission result to attach it
  // to -- so the reason string is the only channel it reaches the round log through,
  // and that is why the check above matches the value inside the message rather than
  // reading a field. Pinned here so a later refactor that drops the number from the
  // string does not quietly leave the refusal unquantified.
  ok(verdict.minCase === undefined && /0\.3000/.test(verdict.reason || ''),
    'a refusal carries its worst-context value only in the reason text, so the text must keep it',
    JSON.stringify(verdict));
}
// Finding (63). (60) unified the two sites spelled `qdAdmissionCheck` and missed a
// third that was not: the archive-writing loop re-derived suiteRobust/routeCells/
// minCase from the same `ver` and carried its own bare `continue`. Two copies of one
// rule is one copy too many (53), and the silent one is the one that bites.
ok(/const \{ suiteRobust, routeCells, minCase \} = r\.qd_admission;/.test(src) &&
   /if \(!r\.qd_admission\) \{/.test(src) &&
   !/const suiteRobust = qdRobust\(r\.ver\);/.test(src),
  'archive admission consumes the filter decision instead of re-deriving it');
ok((src.match(/qdMinCase\(/g) || []).length === 1 &&
   (src.match(/qdRobust\(r?\.?ver\)/g) || []).length === 1,
  'the worst-case and suite-robust statistics each have exactly one call site, inside the gate');
ok((src.match(/const admission = qdAdmissionCheck\(/g) || []).length === 2 &&
   !/qdMinCase\((ver|r\.ver)\) [<>]/.test(src),
  'both admission paths refuse through the shared named-reason check, with no open-coded guardrail left');
// There are two admission paths -- warm import and main round admission -- and
// they are two views of one policy. `.test` only proves at least one site has
// the rule, which is how a regression in exactly one of them stays invisible;
// a mutation probe confirmed that hole. Count the sites instead.
const count = (re) => (src.match(re) || []).length;
const ADMISSION_PATHS = 2;
ok(count(/const replacement = !incumbent \|\| robust\.lower > incumbent\.robust\.upper;/g) === ADMISSION_PATHS,
  'replacement requires non-overlapping robust intervals on every admission path',
  count(/const replacement = !incumbent \|\| robust\.lower > incumbent\.robust\.upper;/g));
ok(count(/const nearBoundary = !!\(incumbent && !replacement && robust\.upper > incumbent\.robust\.lower\);/g)
     === ADMISSION_PATHS &&
   count(/if \(!current \|\| robust\.upper > current\.robust\.upper\) qdArchive\.challengers\[rc\.cell\] = entry;/g)
     === ADMISSION_PATHS,
  'overlapping intervals remain visible as challengers on every admission path');
ok(/e\.min_case < QD_CANONICAL_GUARDRAIL\) continue/.test(src),
  'full-suite canonical promotion keeps the stricter guardrail');

console.log('\n# QD selection precedes and is isolated from SOL');
const selectPos = src.indexOf("roleAgent('tech_lead', 'select_qd_parents'");
const solPos = src.indexOf("roleAgent('profile_engineer', 'selected_cell_sol'");
const mutatePos = src.indexOf("roleAgent('tech_lead', 'plan_qd_mutations'");
ok(selectPos >= 0 && selectPos < solPos && solPos < mutatePos,
  'control order is QD select -> selected-cell SOL -> mutation plan');
const selectWindow = src.slice(selectPos, solPos);
ok(!/SOL_CARD|sol_card|CELL_SOL/.test(selectWindow), 'parent selection receives no SOL card');
ok(/card\.context_id !== s\.context_id/.test(src) &&
   /qdSolCaseValid\(c, s\.context_id\)/.test(src),
  'SOL card is bound to selected source/cell/context and formula-checked');
// (84). This pin used to be `const beforeCase = [\s\S]*?c && c.name === rc.context_id`,
// and the lazy wildcard swallowed anything at all in between -- including a
// `(cases || [])[0]` index, which is the single wrong form the assertion names.
// A mutation probe introducing exactly that survived. The `.find(` is now
// required in the match and the indexed form is refused by name, which is the
// converse-absence shape (84) recommends and this site had not got.
ok(/const beforeCase = r\.d\.sol_card && \(r\.d\.sol_card\.cases \|\| \[\]\)\.find\(c =>\s*\n?\s*c && c\.name === rc\.context_id\)/.test(src) &&
   !/sol_card\.cases \|\| \[\]\)\[0\]/.test(src),
  'transition feedback reads the matching context rather than cases[0]');

console.log('\n# the parent-provenance gate, executed');
// This block used to be three regexes against the inline condition in the
// direction map, and `audit_pin_coverage.py` is what showed that was not
// coverage: the log-message pin was flipped by no mutant in the corpus, and the
// section was one of the few not marked `executed`. The residency and
// route-priority gates in the same map had already been extracted into named
// functions so they could be run against fabricated receipts; the oldest and
// most load-bearing of the three had not. Two of three is (57).
{
  const parentBlock = grab(
    /const qdParentReject = \(d, selections, solCards, cells\) => \{[\s\S]*?\n\};\n/,
    'qdParentReject');
  const g = new Function('QD_ARCH', 'QD_DTYPE', 'QD_CONTEXT_IDS',
    'QD_SUPPORTED_ARCHES', 'QD_MULTI_DIE_ARCHES',
    `${descriptorBlock}\n${parentBlock}\nreturn {qdParentReject, qdAdjacency};`
  )('gfx90a', 'bf16', new Set(['decode_m8']), SUPPORTED, MULTI_DIE);

  const H = (c) => c.repeat(64);
  const desc = { compute_primitive: 'native_mfma', wave_schedule: 'independent',
    k_pipeline: 'lds_single', decomposition: 'tile_grid', output_path: 'direct_store',
    rasterization: 'grouped_m', plan_binding: 'static' };
  const parent = { elite_id: 'E1', source_hash: H('a'), descriptor: desc,
    snapshot: '/ws/E1', artifact: 'art1', context_id: 'decode_m8' };
  const donor = { elite_id: 'E2', source_hash: H('b'), descriptor: desc };
  const sel = { id: 'd1', parent_elite_id: 'E1', selected_cell: 'CELL',
    context_id: 'decode_m8', parent_source_hash: parent.source_hash, parent };
  const card = { selected_cell: 'CELL', context_id: 'decode_m8',
    parent_source_hash: parent.source_hash };
  const dir = { id: 'd1', parent_elite_id: 'E1', selected_cell: 'CELL',
    parent_source_hash: parent.source_hash, operator: 'local_mutation' };
  const cells = { CELL: parent, OTHER: donor };
  const f = (d, s, c, cl) => g.qdParentReject(d, s || [sel], c || [card], cl || cells);
  const why = (...a) => f(...a).reason || '';

  // (55) first: if the sound direction did not pass, every refusal below would
  // be green for the wrong reason and the block would assert nothing.
  const sound = f(dir);
  ok(!sound.reason && sound.parent === parent && sound.card === card &&
     sound.selection === sel && Array.isArray(sound.donors) && sound.donors.length === 0,
    'a direction whose parent, cell and card all resolve is accepted, and carries them out',
    JSON.stringify(sound.reason || Object.keys(sound)));

  ok(/^parent:no_selection_for_direction\(d1\)/.test(why(dir, [])),
    'a direction with no matching selection is refused -- there is no canonical fallback to take',
    why(dir, []));
  ok(/^parent:selection_carries_no_parent/.test(why(dir, [{ ...sel, parent: null }])),
    'a selection that resolved to no live elite is refused rather than treated as unparented',
    why(dir, [{ ...sel, parent: null }]));
  ok(/^parent:no_sol_card\(CELL\|decode_m8\|a{64}\)/.test(why(dir, undefined, [])),
    'a direction with no SOL card for its own cell+context+parent is refused',
    why(dir, undefined, []));
  // The lookup, not the presence. A card for the right cell but a different
  // parent build describes a different measurement, and matching on cell alone
  // is the plausible way to get this wrong.
  ok(/^parent:no_sol_card/.test(why(dir, undefined, [{ ...card, parent_source_hash: H('c') }])),
    'a SOL card for the same cell but a different parent build does not satisfy the requirement',
    why(dir, undefined, [{ ...card, parent_source_hash: H('c') }]));
  ok(/^parent:no_sol_card/.test(why(dir, undefined, [{ ...card, context_id: 'prefill_m64' }])),
    'nor does a card for the same cell and parent but another context');

  // The one that invalidates results already collected: the archive moved under
  // the round, so the direction is aimed at a build that is no longer the elite.
  const moved = { ...dir, parent_source_hash: H('9') };
  ok(/^parent:source_hash_moved\(direction=9{64}, live cell=a{64}\)/.test(why(moved)),
    'a parent hash that no longer matches the live cell is refused, naming both hashes',
    why(moved));

  const legal = g.qdAdjacency(desc);
  ok(legal.length > 0, 'the fixture descriptor has legal neighbours, so the two transition '
    + 'checks below are distinguishable rather than both failing for lack of any edge');
  const step = { ...dir, operator: 'directed_transition', target_descriptor: legal[0].descriptor };
  ok(!f(step).reason, 'a directed transition to a legal adjacent descriptor is accepted',
    why(step));
  ok(why({ ...step, target_descriptor: { ...desc } }) ===
       'parent:transition_target_is_the_parent_itself',
    'a transition whose target is the parent descriptor spends a build to arrive where it started',
    why({ ...step, target_descriptor: { ...desc } }));
  ok(/^parent:transition_illegal/.test(
       why({ ...step, target_descriptor: { ...desc, k_pipeline: 'not_a_real_stage' } })),
    'a transition to a descriptor not in the adjacency is refused by value');
  ok(/^parent:transition_illegal/.test(why({ ...step, target_descriptor: undefined })),
    'a directed transition with no target at all is illegal, not unchecked');
  // (80) again, from the other side: adjacency is compared by value, so a
  // legal target with its keys in another order must still be accepted.
  const shuffled = Object.fromEntries(Object.entries(legal[0].descriptor).reverse());
  ok(!f({ ...step, target_descriptor: shuffled }).reason,
    'the legal target still passes with its keys in reverse order -- value equality, not key order',
    why({ ...step, target_descriptor: shuffled }));
  // And the converse: a non-transition operator is not silently held to the
  // adjacency rule, or every local mutation would need a target descriptor.
  ok(!f({ ...dir, target_descriptor: { ...desc, k_pipeline: 'not_a_real_stage' } }).reason,
    'a local mutation is not judged against the adjacency -- only directed transitions are');

  const cross = { ...dir, operator: 'semantic_crossover', parent_elite_ids: ['E2'] };
  const crossed = f(cross);
  ok(!crossed.reason && crossed.donors.length === 1 && crossed.donors[0] === donor,
    'a crossover with one distinct donor is accepted and the resolved donor is carried out',
    why(cross));
  ok(/^parent:crossover_sources_not_distinct\(1 donor id\(s\), 0 resolved\)/.test(
       why({ ...cross, parent_elite_ids: ['E_nonexistent'] })),
    'a donor id that resolves to nothing is refused, and the count says it resolved to nothing',
    why({ ...cross, parent_elite_ids: ['E_nonexistent'] }));
  ok(/^parent:crossover_sources_not_distinct/.test(
       why(cross, undefined, undefined, { CELL: parent, TWIN: { ...donor, source_hash: H('a') } })),
    'a donor sharing the parent\'s source artifact is a no-op dressed as a crossover');
  ok(/^parent:crossover_sources_not_distinct/.test(
       why({ ...cross, parent_elite_ids: ['E2', 'E2'] })),
    'the same donor twice is two ids and one artifact, which is not two sources');
  ok(!f({ ...dir, parent_elite_ids: ['E_nonexistent'] }).reason,
    'donor ids on a non-crossover direction are not held to distinctness -- the rule belongs '
    + 'to the operator that has sources to keep apart');

  // Matching a selection by (parent_elite_id, selected_cell) when the ids differ
  // is the other half of the lookup, and it is what lets a planner rename its
  // directions between rounds without losing its parent.
  ok(!f({ ...dir, id: 'renamed' }).reason,
    'a direction renamed since selection still resolves via parent_elite_id + selected_cell',
    why({ ...dir, id: 'renamed' }));
  ok(/^parent:no_selection_for_direction\(renamed\)/.test(
       why({ ...dir, id: 'renamed', selected_cell: 'ELSEWHERE' })),
    'but a different cell is a different selection, not a rename');

  const reasons = [why(dir, []), why(dir, [{ ...sel, parent: null }]), why(dir, undefined, []),
    why(moved), why({ ...step, target_descriptor: { ...desc } }),
    why({ ...step, target_descriptor: { ...desc, k_pipeline: 'not_a_real_stage' } }),
    why({ ...cross, parent_elite_ids: ['E_nonexistent'] })];
  ok(new Set(reasons.map(r => r.split('(')[0])).size === reasons.length,
    'the seven refusals are seven distinguishable tokens -- the single lumped message named '
    + 'four possibilities and committed to none', JSON.stringify(reasons));
}

console.log('\n# provenance and transitions fail closed');
ok(/parent\.source_hash === s\.parent_source_hash/.test(src), 'selected parent hash must match live cell');
ok(/rejected stale\/missing\/same-source\/illegal-transition\s*`?\s*\+?\s*`?QD parent; no canonical fallback -- \$\{gate\.reason\}/.test(src),
  'the refusal reaches the log with the specific reason appended, not the lumped four');
// And the converse, which is the assertion that would actually have caught the bug:
// no descriptor comparison anywhere may go back to stringify equality.
const stringifyCmp = src.match(/JSON\.stringify\([^)]*descriptor[^)]*\)\s*===\s*JSON\.stringify/gi) || [];
ok(stringifyCmp.length === 0,
  'no descriptor is compared by JSON.stringify equality -- that is an ordering test, not a value test',
  JSON.stringify(stringifyCmp));
// Written as a negated early-return in the qdVerified filter: a candidate whose
// seed is not the parent it was directed from, or whose final source hash is not
// canonical, is dropped before it can reach a cell.
ok(/r\.ver\.seed_source_hash !== r\.d\.parent_source_hash \|\|\s*!qdHashValid\(r\.ver\.source_hash\)\) \{/.test(src),
  'verified candidate seed and final source hashes are enforced');
// Matched as the whole guard, not as the bare `priorLocal >= 2` this used to
// look for. A bare fragment is matchable by PROSE: adding a comment elsewhere
// in the lane that mentioned the threshold in backticks was enough to keep this
// check green while the real threshold was mutated to 3, and the mutation probe
// in test_js_suite.py caught exactly that. A lexical assertion has to name
// enough of the statement that no comment about the statement can satisfy it.
ok(/if \(QD_ENABLED && priorLocal >= 2 &&/.test(src) && /blocked repeated local mutation/.test(src),
  'two failed local attempts force a route strategy change');

console.log('\n# warm import is source-only and supports explicit reclassification');
ok(/const reclassifyMode = !!\(importSourceOK && QD_RECLASSIFY\)/.test(src),
  'explicit reclassification is a separate import mode');
ok(/QD_RECLASSIFY: reclassifyMode \? '1' : '0'/.test(src),
  'reclassification intent is passed to current-run verifier');
// `rejects` is an out-param for diagnostics only; the point of the check is that
// the cells come from the CURRENT run's verifier output `ver`, never from the
// historical descriptors carried in the imported manifest.
ok(/const routeCells = qdRouteCells\(ver(, rejects)?\);/.test(src) &&
   !/qdRouteCells\((proposed|imported|candidate|entry)\b/.test(src),
  'warm import never falls back to historical route descriptors');
ok(/label: 'persist verified QD imports'/.test(src) &&
   /entry\.snapshot = `\$\{qdArchive\.archive_dir\}\/artifacts\/\$\{entry\.source_hash\}\/workspace`/.test(src),
  'verified imports become parents only through this run durable content-addressed artifact');
ok(/HISTORICAL_SCORE_TRUSTED: 'false'/.test(src) && /REQUIRE_CURRENT_HARNESS_REVERIFY: '1'/.test(src),
  'historical fitness is untrusted and current harness re-verifies');
ok(!/qdArchive\.cells = setup\.qd_import_manifest/.test(src),
  'historical cells are never deserialized into the live archive');

console.log('\n# source-level budget and greedy compatibility invariants');
ok(/plannedCost \+ d\.cost > remaining/.test(src), 'QD planner cannot overspend remaining budget');
ok(/op === 'deep_mutation' \|\| op === 'semantic_crossover'/.test(src),
  'deep and crossover share heavyweight cost');
// (96). `REQUIRE_BASELINE_RELATIVE_BINARY_PATCH` used to be a flag in a prompt
// asking an agent to produce this file. The agent is no longer asked: the
// promotion path still reads `artifacts/<hash>/baseline.patch`, and
// qd_persist_manifest.py writes it -- deriving and reapplying one even for the
// patch-less bootstrap/import artifacts, which is the case the flag never
// covered. Checked here as the path the lane actually consumes, because that is
// the half of the contract this file can see.
ok(/artifacts\/\$\{e\.source_hash\}\/baseline\.patch/.test(src),
  'canonical promotion consumes baseline-relative artifact patches');
ok(/scripts\/qd_persist_manifest\.py/.test(src) && !/REQUIRE_ATOMIC_MANIFEST/.test(src),
  'the archive is written by the persister script, not described to an agent');
// Counted, not merely present. The name appears at four EXECUTABLE sites --
// the payload runner and the three persist calls -- and `.test()` stays true
// while two of them point at one script and the third at another. Two
// persisters is precisely the state the atomicity guarantee cannot survive,
// and it is invisible to a presence check.
ok((src.match(/scripts\/qd_persist_manifest\.py/g) || []).length === 4,
  'all four executable references name the SAME persister; a renamed one is a '
  + 'second writer, not a typo',
  String((src.match(/scripts\/qd_persist_manifest\.py/g) || []).length));
// The threshold is unchanged and is still the DEFAULT: a run that passes no measured
// `route_bands` table evaluates exactly this expression and commits on exactly this test. What
// changed is that a run WITH a band table can overrule it per-route, so the fallback is pinned
// separately below -- a lexical check on the threshold alone would keep passing if the fallback
// were deleted and the per-route verdict became unconditional.
ok(/const legacyImproved = !!\(winner && winner\.geomean > cumulative \* \(1 \+ MIN_IMPROVE\)\)/.test(src),
  'greedy canonical promotion threshold remains unchanged');
ok(/let improved = legacyImproved;/.test(src),
  'the legacy threshold is the default commit decision, not one branch of two');
ok(/if \(winner && ROUTE_BANDS\) \{/.test(src) &&
   /if \(routeVerdict\.applicable\) \{\s*improved = routeVerdict\.accepted;/.test(src),
  'the per-route gate overrules the legacy threshold only when bands were supplied AND it is applicable');
// Finding (62) split this from one conjunction into a gate plus a named metric
// refusal, so the shape changed; the threshold it enforces did not.
ok(/says\(r\.ver\.correctness, 'pass'\)\)\) return false;/.test(src) &&
   /return primSpeedup\(r\.ver\) > CANDIDATE_FLOOR;/.test(src),
  'greedy verified filter still gates on correctness and the candidate floor');

// Finding (61). These four helpers gated real decisions and had never been
// executed by any test -- only matched by regex. Executing them found them
// sound, which is a result worth pinning rather than re-deriving: the rounds
// law (24) is the single most valuable durable rule in this project and this
// is its only enforcement.
console.log('\n# the rounds law and the legality rules, executed');
{
  const gates = new Function('QD_ARCH', 'QD_DTYPE', 'QD_CONTEXT_IDS',
    'QD_SUPPORTED_ARCHES', 'QD_MULTI_DIE_ARCHES',
    `${descriptorBlock}\nreturn {qdDescriptorReject, qdResidencyReject, setArch: (a, d) => ` +
    '{ QD_ARCH = a; QD_DTYPE = d; }};'
  )('gfx942', 'bf16', new Set(['decode_m8', 'prefill_m64']), SUPPORTED, MULTI_DIE);
  const rr = (cur, cand, extra) => gates.qdResidencyReject({ allow: true, current: cur,
    candidate: cand, ...extra });
  const at1 = { ctas: 608, residency_slots: 608, rounds: 1 };
  const at2 = { ctas: 1216, residency_slots: 608, rounds: 2 };
  ok(rr(at1, at1) === null, 'a receipt that stays at one round is accepted');
  ok(/rounds_raised\(1 -> 2\)/.test(rr(at1, at2) || ''),
    'crossing from one round to two is refused, naming the crossing', rr(at1, at2));
  ok(rr(at2, at2) === null,
    'a route already at two rounds is held at two, not declared unfixable -- max(1, current)');
  ok(rr(at2, at1) === null, 'dropping from two rounds to one is allowed');
  // The one thing this gate can really do: refuse a receipt whose own numbers
  // do not agree. A planner faking a pass has to fake three per side.
  ok(/candidate_arithmetic\(rounds=1, ceil\(1216\/608\)=2\)/.test(
       rr(at1, { ctas: 1216, residency_slots: 608, rounds: 1 }) || ''),
    'a receipt whose rounds contradict its own ctas/slots is refused, showing the arithmetic');
  ok(gates.qdResidencyReject(null) === 'residency:receipt_absent', 'an absent receipt is refused');
  ok(/gate_refused/.test(gates.qdResidencyReject({ current: at1, candidate: at1 }) || ''),
    'a receipt that omits allow is refused: the gate is opt-in, not opt-out');
  ok(/gate_refused/.test(gates.qdResidencyReject(
       { allow: 'true', current: at1, candidate: at1 }) || ''),
    'allow must be the boolean true, not a truthy string');
  ok(/candidate\.residency_slots=0/.test(rr(at1, { ...at1, residency_slots: 0 }) || '') &&
     /candidate\.ctas=608\.5/.test(rr(at1, { ...at1, ctas: 608.5 }) || ''),
    'non-positive or non-integer facts are refused, naming the side and the field');
  ok(gates.qdResidencyReject({ allow: true, current: at1 }) === 'residency:candidate_facts_absent',
    'a receipt missing a whole side is refused, naming the side');

  const rej = (d, arch, dtype) => { gates.setArch(arch || 'gfx942', dtype || 'bf16');
    const out = gates.qdDescriptorReject(d); gates.setArch('gfx942', 'bf16'); return out; };
  ok(rej(base) === null && rej(base, 'gfx90a') === null, 'the legal tuple is legal on both arches');
  ok(gates.qdDescriptorReject(null) === 'descriptor:absent', 'an absent descriptor says so');
  const { k_pipeline, ...noK } = base;
  ok(rej(noK) === 'axis:k_pipeline=<missing>' &&
     rej({ ...base, k_pipeline: 'lds_double' }) === 'axis:k_pipeline="lds_double"',
    'a missing axis and an out-of-vocabulary value are distinguishable');
  // Finding (61): these were one token. They need opposite corrections.
  ok(rej({ ...base, decomposition: 'split_k' }) === 'rule:reduction_without_fixup' &&
     rej({ ...base, output_path: 'atomic_fixup' }) === 'rule:fixup_without_reduction',
    'the two directions of the reduction/fixup coupling are named separately');
  ok(rej({ ...base, compute_primitive: 'valu', wave_schedule: 'symmetric_pingpong' })
       === 'rule:pingpong_requires_matrix_core' &&
     rej({ ...base, plan_binding: 'runtime_tuned' }) === 'rule:runtime_tuned_requires_reduction',
    'the ping-pong and runtime-tuning rules each fire with their own token');
  const xcd = { ...base, rasterization: 'xcd_remapped_grouped' };
  ok(rej(xcd) === null && rej(xcd, 'gfx90a') === 'rule:xcd_remap_requires_multi_die',
    'the XCD remap is legal on gfx942 and refused on gfx90a -- the mechanism (55) re-legalized');
  ok(rej(base, 'gfx1100') === 'rule:unsupported_arch_or_dtype' &&
     rej(base, 'gfx942', 'fp8') === 'rule:unsupported_arch_or_dtype',
    'an unsupported arch or dtype is refused');
  // Every rule branch above returned a DIFFERENT token. (60): refusals of
  // different kinds must be distinguishable, or they are one refusal.
  const tokens = [rej(noK), rej({ ...base, decomposition: 'split_k' }),
    rej({ ...base, output_path: 'atomic_fixup' }),
    rej({ ...base, compute_primitive: 'valu', wave_schedule: 'symmetric_pingpong' }),
    rej({ ...base, plan_binding: 'runtime_tuned' }), rej(xcd, 'gfx90a'), rej(base, 'gfx1100')];
  ok(new Set(tokens).size === tokens.length, 'no two legality violations share a token',
    JSON.stringify(tokens));
}

console.log('\n# the persistence receipt, executed');
{
  // Finding (64). Durability is asserted by an agent receipt the lane cannot verify
  // against the filesystem. It can verify the receipt against the list it handed out,
  // and it must say what a rollback cost -- silently un-writing a won cell is exactly
  // what makes an archive look like it never filled (44).
  const lines = [];
  const recv = new Function('log',
    grab(/const qdPersistenceReceipt = \(persisted, admissions, where\) => \{[\s\S]*?\n\};\n/,
      'qdPersistenceReceipt') +
    grab(/const qdLogRollback = \(entry, where\) =>[\s\S]*?`\);\n/, 'qdLogRollback') +
    '\nreturn {qdPersistenceReceipt, qdLogRollback};')((m) => lines.push(String(m)));
  const adms = [{ elite_id: 'e1', cell: 'c1' }, { elite_id: 'e2', cell: 'c2', previous_elite: {} }];
  lines.length = 0;
  ok(JSON.stringify(recv.qdPersistenceReceipt({ persisted_elite_ids: ['e1', 'e2'] }, adms, 'r1'))
       === '["e1","e2"]' && lines.length === 0,
    'a receipt that matches the admissions is honoured in full and says nothing');
  lines.length = 0;
  const none = recv.qdPersistenceReceipt(null, adms, 'r1');
  ok(none.length === 0 && lines.length === 1 && /receipt MISSING/.test(lines[0]) &&
     /not a search result/.test(lines[0]),
    'a missing receipt is named as a harness fault, not left to read as "found nothing"',
    JSON.stringify(lines));
  lines.length = 0;
  const partial = recv.qdPersistenceReceipt({ persisted_elite_ids: ['e2', 'ghost'] }, adms, 'r1');
  ok(JSON.stringify(partial) === '["e2"]' && lines.length === 1 &&
     /never admitted/.test(lines[0]) && /ghost/.test(lines[0]),
    'an id that was never admitted is refused by name and the intersection is honoured',
    JSON.stringify(lines));
  // The receipt is an agent self-report: an empty one must not be reported as success.
  lines.length = 0;
  ok(recv.qdPersistenceReceipt({ persisted_elite_ids: [] }, adms, 'r1').length === 0 &&
     lines.length === 0,
    'an explicitly empty receipt is honoured as empty, distinct from a missing one');
  lines.length = 0;
  recv.qdLogRollback(adms[0], 'r1'); recv.qdLogRollback(adms[1], 'r1');
  ok(lines.length === 2 && lines.every(l => /persistence:not_in_receipt/.test(l)) &&
     /left empty/.test(lines[0]) && /restored to its previous elite/.test(lines[1]),
    'every rolled-back cell is logged, and says whether an incumbent came back');
}
// Both rollback loops go through the shared logger, and the bootstrap path still throws
// rather than rolling back -- a run with no durable seed cannot proceed.
ok((src.match(/qdPersistenceReceipt\(/g) || []).length === 2 &&
   (src.match(/qdLogRollback\(entry, /g) || []).length === 2 &&
   /QD bootstrap failed closed: canonical artifact\/cell manifest was not durably persisted/.test(src),
  'both persistence paths validate and log through one helper; bootstrap still fails closed');
// (124b). The round persist site must NOT be gated on the admission list. It
// was, for sixteen runs, and the cost is invisible from inside a run: the
// capsule ledger, the transition edges, the stall counters and the generation
// number all ride in the same payload's top-level fields, so a round that
// refuted everything -- the cheapest and most transferable result a QD search
// produces -- wrote nothing at all and was gone at process exit. Two lexical
// halves, because either alone is satisfiable by the wrong code: the guard is
// absent, AND the ledger-only branch that reports the write is present.
ok(!/let persistedIds = \[\];\s*\n\s*if \(qdAdmissions\.length\)/.test(src),
  'the round persist site is not gated on `qdAdmissions.length` -- a round that admitted '
  + 'nothing still has its refutations to persist');
ok(/const ledgerOnly = !qdAdmissions\.length;/.test(src) &&
   /admitted nothing AND its ledger write failed/.test(src),
  'a ledger-only round is persisted, and a FAILED ledger write is announced rather than '
  + 'being indistinguishable from a round with nothing to say');

console.log('\n# the correctness gate and the primary-metric selector, executed');
{
  // Finding (62). Both of these were lexical-only until now, and both are gates:
  // `says` decides whether a candidate counts as correct, `primSpeedup` decides
  // what number the round gates on and what the archive stores as an elite score.
  const gateBlock = grab(/const saysContradicted = \(v\) => \{[\s\S]*?\nconst says = [^;]*;\n/, 'says');
  const says = new Function(`${gateBlock}\nreturn {says, saysContradicted};`)();

  // The gate opens on genuine successes, including the phrasing it exists for.
  for (const s of ['PASS', 'pass', ' passed ', 'PASS - 15/15 draws', 'passes all 11 cases',
                   'PASS_WITH_WARNINGS']) {
    ok(says.says(s, 'pass') === true, `the correctness gate opens on ${JSON.stringify(s)}`);
  }
  // ...and stays shut on failures, INCLUDING the ones that used to open it.
  for (const s of ['FAIL', 'failed', 'did not pass', 'partially passes', '', null, undefined,
                   'passes 10/11 cases', 'pass rate 10/11', 'PASS except decode_m2',
                   'pass with 1 mismatch', 'passed but 2 cases are incorrect']) {
    ok(says.says(s, 'pass') === false, `the correctness gate stays shut on ${JSON.stringify(s)}`);
  }
  ok(says.says('verified', 'verified') === true &&
     says.says('verified: FAILED correctness', 'verified') === false &&
     says.says('not verified', 'verified') === false,
    'the status gate stays shut on a report that verifies its own failure');
  // The veto is what does the work -- without it the prefix alone says true.
  ok('passes 10/11 cases'.trim().toLowerCase().startsWith('pass') &&
     says.saysContradicted('passes 10/11 cases') === true &&
     says.saysContradicted('PASS - 15/15 draws') === false,
    'an n-of-m fraction is a contradiction only when it is not n/n');

  const metricBlock = grab(/const primWeighted = \(o\) => \{[\s\S]*?\n\};\n/, 'primWeighted') +
    grab(/const primMetricReason = \(o\) =>[\s\S]*?: null\);\n/, 'primMetricReason') +
    grab(/const primSpeedup = \(o\) => \{[\s\S]*?\n\};\n/, 'primSpeedup');
  const mk = (hasWorkload) => new Function('HAS_WORKLOAD',
    `${metricBlock}\nreturn {primSpeedup, primMetricReason};`)(hasWorkload);
  const off = mk(false), on = mk(true);
  const both = { verified_geomean: 1.1, verified_weighted: 1.3 };
  const onlyGeo = { verified_geomean: 1.1 };
  ok(off.primSpeedup(both) === 1.1 && off.primMetricReason(onlyGeo) === null,
    'an unweighted run is unchanged: the geomean is the primary metric and nothing is refused');
  ok(on.primSpeedup(both) === 1.3, 'a workload-aligned run gates on the weighted number');
  // The defect: the same call used to return 1.1 and say nothing.
  ok(/^metric:weighted_speedup_missing/.test(String(on.primMetricReason(onlyGeo))) &&
     /^metric:weighted_speedup_missing/.test(String(on.primMetricReason(
       { verified_geomean: 1.1, verified_weighted: null }))),
    'a workload-aligned run refuses, by name, a report with no weighted number');
  ok(on.primMetricReason({ verified_geomean: 1.1, speedup_weighted: 1.4 }) === null &&
     on.primSpeedup({ verified_geomean: 1.1, speedup_weighted: 1.4 }) === 1.4,
    'the legacy speedup_weighted field still satisfies the requirement');
  // (60): a metric fault must not be reported as, or silently share a path with,
  // a slow kernel. Both admission paths route it through the named check.
  ok(/const metricReason = primMetricReason\(ver\);/.test(src) &&
     (src.match(/primMetricReason\(/g) || []).length >= 3,
    'the authoritative sites -- both archive paths, the greedy filter and integrate -- '
    + 'route an ambiguous primary metric through the named refusal');
}

console.log('\n# the oracle pin, executed');
{
  // Finding (67). Candidates carry `source_hash` provenance; the oracle -- the denominator of
  // every speedup, the correctness reference, and the yardstick the persisted archive is scored
  // on -- carried none, though four role prompts say "NEVER modify" it and the golden reaches
  // each workspace as an absolute symlink that writes through to the shared original.
  const driftBlock = grab(/const ORACLE_DEGENERATE = new Set\(\[[\s\S]*?\n\]\);\n/, 'ORACLE_DEGENERATE') +
    grab(/const oracleDrift = \(o\) => \{[\s\S]*?\n\};\n/, 'oracleDrift');
  const mkDrift = (pinned) => new Function('ORACLE_DIGEST',
    `${driftBlock}\nreturn oracleDrift;`)(pinned);
  const PIN = 'a'.repeat(64), OTHER = 'b'.repeat(64);
  const pinned = mkDrift(PIN), unpinned = mkDrift(null);

  ok(pinned({ oracle_digest: PIN }) === null && pinned({ oracle_digest: ` ${PIN} ` }) === null,
    'a report measuring against the pinned oracle passes, whitespace and all');
  ok(/^oracle:digest_missing/.test(String(pinned({ verified_geomean: 1.4 }))) &&
     /^oracle:digest_missing/.test(String(pinned({ oracle_digest: '' }))) &&
     /^oracle:digest_missing/.test(String(pinned(null))),
    'a report with no digest is refused by name, not read as a result');
  // The drift case is the one that must NOT be a per-candidate skip: results already admitted
  // to a persisted archive were scored against the denominator that just moved.
  let threw = null;
  try { pinned({ oracle_digest: OTHER }); } catch (e) { threw = e; }
  ok(threw && /oracle:digest_drift/.test(threw.message) &&
     /already admitted to the archive/.test(threw.message) &&
     /Refusing to continue or to persist/.test(threw.message),
    'a moved oracle fails the run closed and says why, rather than dropping one candidate');
  ok(unpinned({ oracle_digest: OTHER }) === null && unpinned({}) === null,
    'an unpinned run does not re-litigate every candidate -- it is named once, at setup');

  // The pin must reach all three gates, and the unpinned run must not be publishable.
  ok(/const oracleReason = oracleDrift\(ver\);/.test(src) &&
     /primMetricReason\(r\.ver\) \|\| oracleDrift\(r\.ver\)/.test(src) &&
     /const oracleFinal = validation \? oracleDrift\(validation\) : null;/.test(src),
    'archive admission, the greedy filter and the final verdict all consult the pin');
  ok(/oracle_digest: \{ type: 'string' \}/.test(src) &&
     (src.match(/oracle_digest: \{ type: 'string' \}/g) || []).length === 3,
    'setup, verify and validate all declare the digest -- a field only one side reports is not a check');
  ok(/if \(!oraclePinned\) \{/.test(src) && /oracle:unpinned\(setup recorded no oracle digest/.test(src),
    'a run that never pinned the oracle is flagged, not quietly trusted');
}

console.log('\n# the final validation verdict, executed');
{
  // Finding (65). Every field of the run's only independent arbitration was decorative:
  // `correctness` was in the schema and read by nothing, `validation_status` was passed
  // through with no gate, `timing_basis` was required by director.md and absent from both
  // the schema and every consumer, and a null verdict published the TechLead's self-report.
  const gateBlock = grab(/const saysContradicted = \(v\) => \{[\s\S]*?\nconst says = [^;]*;\n/, 'says');
  const verdictBlock = grab(/const validationVerdict = \(v, oraclePinned\) => \{[\s\S]*?\n\};\n/, 'validationVerdict');
  const { validationVerdict: raw } =
    new Function(`${gateBlock}\n${verdictBlock}\nreturn {validationVerdict};`)();
  // (67) added the oracle pin ahead of every other clause; these cases are about the clauses
  // behind it, so they run pinned. The unpinned case is asserted separately just below.
  const vv = (v) => raw(v, true);

  const clean = { director_verified_speedup_geomean: 1.3, validation_status: 'accepted',
                  correctness: 'PASS - 11/11', timing_basis: 'device' };
  ok(vv(clean).trust === 'verified' && vv(clean).reason === null && vv(clean).basis === 'device',
    'a clean, accepted, device-timed, correct verdict is verified and needs no reason');

  // The four refusals, and -- (60) -- the requirement that they be four DIFFERENT refusals.
  const none = vv(null);
  const wrong = vv({ ...clean, correctness: 'FAIL: decode_m2 mismatch' });
  const notAccepted = vv({ ...clean, validation_status: 'flagged' });
  const noBasis = vv({ ...clean, timing_basis: undefined });
  ok(none.trust === 'unverified' && /^provenance:director_validate_returned_nothing/.test(none.reason),
    'a dead or hung arbiter is named as absent arbitration, not as a result');
  ok(wrong.trust === 'refused' && /^correctness:/.test(wrong.reason),
    'a speedup over a kernel that computes the wrong answer is refused by name');
  ok(notAccepted.trust === 'flagged' && /^validation:director_status_"flagged"/.test(notAccepted.reason),
    'a status the arbiter did not accept is flagged by name');
  ok(noBasis.trust === 'flagged' && noBasis.basis === 'unknown' && /^timing:basis_"unknown"/.test(noBasis.reason),
    'a missing timing basis defaults to unknown and is flagged -- absence is not evidence of priming');
  ok(vv({ ...clean, timing_basis: 'host_bound' }).trust === 'flagged' &&
     vv({ ...clean, timing_basis: 'unprimed' }).trust === 'flagged',
    'a host-bound or unprimed ratio is not a device-time speedup');
  // (67): a flawless arbiter over an unknown denominator is still an unknown result, so the pin
  // is checked ahead of every other clause and cannot be argued past by a clean verdict.
  const unpinned = raw(clean, false);
  ok(unpinned.trust === 'flagged' && /^oracle:unpinned/.test(unpinned.reason),
    'an unpinned oracle flags even an otherwise perfect verdict');
  ok(new Set([none.reason, wrong.reason, notAccepted.reason, noBasis.reason, unpinned.reason]).size === 5,
    'the five refusals are five distinguishable strings, not one');
  // The gate must not be so eager it refuses honest arbitration.
  ok(vv({ ...clean, validation_status: 'accept' }).trust === 'verified' &&
     vv({ ...clean, validation_status: 'validated' }).trust === 'verified' &&
     vv({ ...clean, correctness: '' }).trust === 'verified' &&
     vv({ ...clean, timing_basis: ' Device ' }).trust === 'verified',
    'accepted/validated, an unstated correctness, and a padded basis all still pass');
  // (62)'s veto reaches this gate too: a correctness string that contradicts its own word.
  ok(vv({ ...clean, correctness: 'passes 10/11 cases' }).trust === 'refused',
    'a partial pass is refused here as well, not read as a pass');

  // Structural: the producer must publish no number it refused, and the consumer must gate.
  ok(/const trusted = verdict\.trust !== 'unverified' && verdict\.trust !== 'refused';/.test(src) &&
     /const finalGeomean = trusted \? validation\.director_verified_speedup_geomean : null;/.test(src) &&
     !/const finalGeomean = validation \? validation\.director_verified_speedup_geomean : cumulative;/.test(src),
    'a refused verdict publishes no geomean -- the self-report is not resurrected as a verified number');
  ok(/validation_trust: verdict\.trust,/.test(src) && /timing_basis: verdict\.basis,/.test(src),
    'the lane hands the consumer its verdict and its timing basis, not just a free-text status');
  ok(/const rankable = \(c\) => c\.kind !== 'lane' \|\| c\.validation_trust === 'verified';/.test(wfSrc) &&
     /cands\.filter\(c => c\.speedup > 1\.0 && rankable\(c\)\)/.test(wfSrc) &&
     !/cands\.filter\(c => c\.speedup > 1\.0\)\.sort/.test(wfSrc),
    'the bake-off ranks only lanes whose own arbiter stands behind the number');
  ok(/cannot win: /.test(wfSrc),
    'a fast-but-unverified lane is excluded out loud, which is what makes it not "found nothing"');
  ok(/timing_basis: \{ type: 'string' \}/.test(src),
    'timing_basis is in the validate schema, so a Director that reports it is not dropped');
}

// Finding (68). The third mandatory planner gate had no orchestrator half at
// all, so unlike (61)'s helpers there was nothing here to execute. Executed
// from the start: every branch that can cost a direction, plus the mixed-list
// case, which is the one the prompt's rule is actually about.
console.log('\n# the route-priority gate, executed');
{
  const floorBlock = grab(FLOOR_BLOCK_RE, 'QD noise floor table');
  const priorityBlock = grab(/const QD_CLOSED_RATIO = [\s\S]*?const qdPriorityFilter = \(receipt, targetCases\) => \{[\s\S]*?\n\};\n/,
    'qdPriorityFilter');
  const g = new Function(`${floorBlock}\n${priorityBlock}\n` +
    'return {qdPriorityFilter, qdPriorityVerdict, qdNoiseFloor, QD_CLOSED_RATIO, '
    + 'QD_MARGINAL_RATIO, QD_NOISE_FLOOR_BY_MACHINE, qdFloorIsMeasured, '
    + 'QD_CURRENT_MACHINE, QD_PROVISIONAL_MACHINES};')();
  // Real rows, against the CURRENT epoch's table. Using the real table is the
  // point -- the gate cross-checks against it, so a fixture floor would not
  // exercise it -- but that means these headrooms are epoch-specific and must be
  // re-derived when QD_CURRENT_MACHINE moves. They were, once: on machine L the
  // closed fixture was headroom 0.05 against a 0.072 floor (ratio 0.69); on
  // machine N that same pair is 0.05/0.0378 = 1.32, which is MARGINAL, so the
  // fixture would have stopped testing the branch it is named for. The `ok`
  // immediately below is what caught it and is the only reason this comment is
  // not describing a silent hole -- keep it ahead of every use of these rows.
  //
  // Machine N: decode_m96_up is the tightest floor (0.0108), decode_m2_square
  // still the widest (0.0378), and the spread is 3.5x rather than L's 14x.
  const row = (context, headroom) => ({ context, remaining_headroom: headroom,
    noise_floor: g.qdNoiseFloor(context),
    slack_to_floor: headroom / g.qdNoiseFloor(context),
    verdict: g.qdPriorityVerdict(headroom, g.qdNoiseFloor(context)) });
  const rc = (...rows) => ({ per_context: rows });
  //
  // Machine P moved it a second time: decode_m2_square's floor narrowed to
  // 0.0224, so the hard-coded 0.03 headroom became ratio 1.34 -- MARGINAL --
  // and the `closed` fixture stopped being closed. The `ok` below caught it
  // again, which is twice out of two machine changes. Hard-coding the
  // headrooms is what keeps re-breaking, so they are now DERIVED from each
  // route's own floor at a fixed ratio. The floors are still the real ones, so
  // the cross-check against the table is still exercised; only the numerator
  // follows the epoch instead of having to be re-typed each time.
  const atRatio = (context, ratio) => row(context,
    Number((g.qdNoiseFloor(context) * ratio).toFixed(9)));
  const open = atRatio('prefill_m256_down', 10.0);      // >= 3.0 -> open
  const closed = atRatio('decode_m2_square', 0.5);      // <  1.0 -> closed
  const marginal = atRatio('prefill_m128_square', 1.75); // in between -> marginal

  ok(open.verdict === 'open' && closed.verdict === 'closed' && marginal.verdict === 'marginal',
    'the three fixtures really are open/closed/marginal against the shipped floor table',
    `${open.verdict}/${closed.verdict}/${marginal.verdict}`);
  ok(g.QD_CLOSED_RATIO === 1.0 && g.QD_MARGINAL_RATIO === 3.0,
    'the thresholds match qd_route_priority.py: 1.0 is the definition, 3.0 the same order of magnitude');

  const f = (receipt, cases) => g.qdPriorityFilter(receipt, cases);
  ok(JSON.stringify(f(rc(open), ['prefill_m256_down']).cases) === '["prefill_m256_down"]',
    'an open route survives the gate');
  ok(JSON.stringify(f(rc(open, marginal), ['prefill_m256_down', 'prefill_m128_square']).cases)
       === '["prefill_m256_down","prefill_m128_square"]',
    'marginal is not closed: it costs nothing');
  // THE case the prompt's rule is about, and the one a boolean gate would get
  // wrong in both directions.
  //
  // Finding (126). A "closed" verdict on a PROVISIONAL epoch rests on a
  // placeholder floor, not on a measurement taken here, so the gate keeps the
  // route rather than dropping it -- the drop would be a claim about a
  // measurement nobody took, and it is precisely the claim that makes the
  // mistake permanent. Every check below that turns on the drop therefore has
  // two correct outcomes, one per kind of epoch, and both are asserted.
  const DROPS = g.qdFloorIsMeasured('decode_m2_square');
  //
  // ...and both outcomes being correct is exactly how a drop-branch mutation
  // survives: on a provisional epoch the branch is dead code, so every
  // assertion about it is vacuous and the suite reports PASS on a lane that
  // drops routes it must keep. The mechanism does not depend on which epoch is
  // current, so it is also exercised against a second instance of the SAME
  // source pinned to an epoch that was measured, whichever box this runs on.
  const measuredEpoch = [...g.QD_NOISE_FLOOR_BY_MACHINE.keys()]
    .find(m => !g.QD_PROVISIONAL_MACHINES.has(m));
  ok(measuredEpoch !== undefined,
    'at least one epoch has measured floors, so the drop branch can be exercised');
  const gM = measuredEpoch === g.QD_CURRENT_MACHINE ? g : new Function(
    `${floorBlock.replace(`const QD_CURRENT_MACHINE = '${g.QD_CURRENT_MACHINE}';`,
                          `const QD_CURRENT_MACHINE = '${measuredEpoch}';`)}\n`
    + `${priorityBlock}\n`
    + 'return {qdPriorityFilter, qdPriorityVerdict, qdNoiseFloor, qdFloorIsMeasured};')();
  ok(gM.qdFloorIsMeasured('decode_m2_square') &&
     !gM.qdFloorIsMeasured('a_route_nobody_measured'),
    `the measured-epoch instance (${measuredEpoch}) really does read as measured, or `
    + 'the repin silently failed and the checks below are the vacuous ones again');
  {
    const fM = gM.qdPriorityFilter;
    const rowM = (context, ratio) => {
      const headroom = Number((gM.qdNoiseFloor(context) * ratio).toFixed(9));
      return { context, remaining_headroom: headroom,
        noise_floor: gM.qdNoiseFloor(context),
        slack_to_floor: headroom / gM.qdNoiseFloor(context),
        verdict: gM.qdPriorityVerdict(headroom, gM.qdNoiseFloor(context)) };
    };
    const openM = rowM('prefill_m256_down', 10.0);
    const closedM = rowM('decode_m2_square', 0.5);
    ok(openM.verdict === 'open' && closedM.verdict === 'closed',
      'the measured-epoch fixtures really are open and closed', JSON.stringify([openM, closedM]));
    const mixedM = fM({ per_context: [openM, closedM] },
                      ['prefill_m256_down', 'decode_m2_square']);
    ok(!mixedM.reason && JSON.stringify(mixedM.cases) === '["prefill_m256_down"]',
      'on a measured epoch a mixed list is filtered: the closed entry is dropped and '
      + 'the rest kept', JSON.stringify(mixedM));
    ok(/all_targets_closed\(decode_m2_square\)/.test(
         fM({ per_context: [closedM] }, ['decode_m2_square']).reason || ''),
      'on a measured epoch a wholly-closed target list is exit 3');
    const uClosedM = { ...closedM, verdict: 'needs_fresh_elapsed',
      verdict_if_elapsed_confirmed: closedM.verdict, elapsed_is_default: true };
    const keptM = fM({ per_context: [uClosedM] }, ['decode_m2_square']);
    ok(!keptM.reason && JSON.stringify(keptM.cases) === '["decode_m2_square"]',
      'and (92) still holds there: an UNMEASURED-elapsed route whose conditional '
      + 'verdict is closed is kept', JSON.stringify(keptM));
    // Finding (135), applied to (126)'s own guard. The mutation "a placeholder
    // floor is treated as a measurement and closes the route" -- deleting
    // `!qdFloorIsMeasured(context)` from the keep-condition -- SURVIVED this
    // suite, and every check above is the reason why: they all use routes that
    // are IN the floor table, and for those the deleted term is already false,
    // so removing it is a no-op. The term's only cover was the current epoch
    // happening to be provisional, which is a property of which box the
    // container was restored onto, not a property of the code. That is (135)
    // exactly: the exposure is never "this file", it is "a guard nothing
    // executes".
    //
    // A route ABSENT from the table is the shape that separates them on every
    // epoch. Its floor is the fail-closed default rather than a number measured
    // here, so a `closed` verdict on it rests on no measurement at all, and the
    // gate must keep it -- dropping it would spend the route on a claim nobody
    // took a reading for.
    const OFF_TABLE = 'a_route_nobody_measured';
    ok(!gM.qdFloorIsMeasured(OFF_TABLE) && Number.isFinite(gM.qdNoiseFloor(OFF_TABLE)),
      'the off-table route reads as unmeasured yet still resolves to a usable '
      + 'fail-closed floor, so a closed verdict can actually be built on it -- '
      + 'without both halves the two checks below are vacuous',
      `measured=${gM.qdFloorIsMeasured(OFF_TABLE)} floor=${gM.qdNoiseFloor(OFF_TABLE)}`);
    const offClosed = rowM(OFF_TABLE, 0.5);
    ok(offClosed.verdict === 'closed',
      'the off-table fixture really is closed, or the keep below proves nothing',
      JSON.stringify(offClosed));
    const offKept = fM({ per_context: [offClosed] }, [OFF_TABLE]);
    ok(!offKept.reason && JSON.stringify(offKept.cases) === `["${OFF_TABLE}"]`,
      'on a MEASURED epoch a closed route whose floor is a placeholder is STILL '
      + 'kept -- this is the assertion that dies when the qdFloorIsMeasured term '
      + 'is deleted, and it dies whichever box this runs on',
      JSON.stringify(offKept));
  }
  const mixed = f(rc(open, closed), ['prefill_m256_down', 'decode_m2_square']);
  ok(!mixed.reason && JSON.stringify(mixed.cases) === (DROPS
       ? '["prefill_m256_down"]' : '["prefill_m256_down","decode_m2_square"]'),
    'a mixed list is filtered, not refused -- the closed entry is dropped on a '
    + 'measured epoch and kept on a provisional one, and the rest is kept either way',
    JSON.stringify(mixed));
  const allClosed = f(rc(closed), ['decode_m2_square']);
  ok(DROPS ? /all_targets_closed\(decode_m2_square\)/.test(allClosed.reason || '')
           : !allClosed.reason && JSON.stringify(allClosed.cases) === '["decode_m2_square"]',
    'a wholly-closed target list is exit 3 where the closure was measured, and is '
    + 'kept where the floor it rests on is a placeholder', JSON.stringify(allClosed));
  ok(f(rc(open), ['prefill_m256_down']).reason === undefined &&
     f(null, ['prefill_m256_down']).reason === 'priority:receipt_absent',
    'an absent receipt is refused: the gate is opt-in, like the residency one');
  ok(f({ per_context: 'nope' }, ['prefill_m256_down']).reason === 'priority:receipt_absent',
    'a receipt whose per_context is not an array is absent, not iterated');
  ok(f(rc(open), []).reason === 'priority:no_target_cases',
    'a direction with no target cases cannot have been checked against any');
  ok(/uncovered\(decode_m2_square\)/.test(
       f(rc(open), ['prefill_m256_down', 'decode_m2_square']).reason || ''),
    'finding (62): a receipt covering only some target cases does not cover the direction');
  // The two re-derivations, which are the whole reason the receipt is worth
  // asking for. A planner can only get past these by telling the truth.
  // The borrowed floor is `prefill_m256_down`'s, whatever it is this epoch --
  // the point is that it is a DIFFERENT route's, and the gate must notice.
  //
  // On a PROVISIONAL epoch every route carries the same fail-closed floor, so
  // borrowing a sibling route's number is undetectable IN PRINCIPLE -- there is
  // nothing to notice. The lie that is still detectable, and the realistic one
  // on a box whose floors have not been measured yet, is a STALE floor: this
  // route's value from another epoch's table. Take whichever differs, and fail
  // loudly if no table anywhere disagrees, because then this check is vacuous.
  const table = g.qdNoiseFloor('decode_m2_square');
  const borrowed = [g.qdNoiseFloor('prefill_m256_down'),
    ...[...g.QD_NOISE_FLOOR_BY_MACHINE.values()].map(t => t.get('decode_m2_square'))]
    .find(v => Number.isFinite(v) && v !== table);
  ok(borrowed !== undefined,
    'some table somewhere states a different floor for decode_m2_square, so there '
    + 'is a wrong floor for the cross-check to catch');
  ok(new RegExp(`floor_mismatch\\(decode_m2_square: receipt ${borrowed}, table ${table}\\)`).test(
       f(rc({ ...closed, noise_floor: borrowed, verdict: 'open' }), ['decode_m2_square']).reason || ''),
    'a receipt that borrows another route\'s floor is caught against the table this file owns',
    f(rc({ ...closed, noise_floor: borrowed, verdict: 'open' }), ['decode_m2_square']).reason);
  ok(new RegExp(`verdict_arithmetic\\(decode_m2_square: receipt "open", ${closed.remaining_headroom}/${table} implies "closed"\\)`).test(
       f(rc({ ...closed, verdict: 'open' }), ['decode_m2_square']).reason || ''),
    'a verdict that does not follow from the receipt\'s own two numbers is refused, showing the arithmetic',
    f(rc({ ...closed, verdict: 'open' }), ['decode_m2_square']).reason);
  ok(/remaining_headroom=1\.5/.test(f(rc({ ...open, remaining_headroom: 1.5 }), ['prefill_m256_down']).reason || '') &&
     /remaining_headroom="0\.3"/.test(f(rc({ ...open, remaining_headroom: '0.3' }), ['prefill_m256_down']).reason || '') &&
     /noise_floor=0/.test(f(rc({ ...open, noise_floor: 0 }), ['prefill_m256_down']).reason || ''),
    'out-of-range, non-numeric and zero-floor rows are refused before any division');
  // Every refusal has to be distinguishable from every other (finding 60).
  const reasons = [f(null, ['prefill_m256_down']).reason, f(rc(open), []).reason,
    f(rc(open), ['prefill_m256_down', 'decode_m2_square']).reason,
    f(rc({ ...closed, noise_floor: borrowed, verdict: 'open' }), ['decode_m2_square']).reason,
    f(rc({ ...closed, verdict: 'open' }), ['decode_m2_square']).reason,
    f(rc(closed), ['decode_m2_square']).reason];
  ok(new Set(reasons.map(r => String(r).split('(')[0])).size === reasons.length,
    'the six route-priority refusals are six distinguishable tokens', JSON.stringify(reasons));

  // Finding (92): the helper reports `needs_fresh_elapsed` whenever nobody
  // supplied `--elapsed-ms`, which is what tech_lead.md's own documented
  // command does. Two things have to hold, and they pull in opposite
  // directions: the arithmetic is still checked (against the conditional
  // verdict the row carries), and the route is still KEPT.
  const unmeasured = (r) => ({ ...r, verdict: 'needs_fresh_elapsed',
    verdict_if_elapsed_confirmed: r.verdict, elapsed_is_default: true });
  const uOpen = unmeasured(open), uClosed = unmeasured(closed);
  ok(JSON.stringify(f(rc(uOpen), ['prefill_m256_down']).cases) === '["prefill_m256_down"]',
    'an unmeasured open route still passes the gate');
  const kept = f(rc(uClosed), ['decode_m2_square']);
  ok(!kept.reason && JSON.stringify(kept.cases) === '["decode_m2_square"]',
    'an unmeasured route whose CONDITIONAL verdict is closed is kept: nobody measured this kernel '
    + 'on it, and dropping it is what makes the mistake permanent (92)',
    JSON.stringify(kept));
  ok(DROPS ? /all_targets_closed\(decode_m2_square\)/.test(
               f(rc(closed), ['decode_m2_square']).reason || '')
           : !f(rc(closed), ['decode_m2_square']).reason,
    'the drop still fires on a route closed on its OWN measurement, so (92) did not '
    + 'disarm the gate -- and does not fire when the floor was never measured here (126)');
  ok(new RegExp(`verdict_arithmetic\\(decode_m2_square: receipt "open", ${closed.remaining_headroom}/${g.qdNoiseFloor('decode_m2_square')} implies "closed"\\)`).test(
       f(rc({ ...uClosed, verdict_if_elapsed_confirmed: 'open' }), ['decode_m2_square']).reason || ''),
    'the conditional verdict is re-derived too: `needs_fresh_elapsed` is not a way past the arithmetic',
    f(rc({ ...uClosed, verdict_if_elapsed_confirmed: 'open' }), ['decode_m2_square']).reason);
  ok(/verdict_arithmetic\(decode_m2_square: receipt "undefined", /.test(
       f(rc({ ...closed, verdict: 'needs_fresh_elapsed' }), ['decode_m2_square']).reason || ''),
    'a row claiming to be unmeasured without carrying the conditional verdict is refused, not waved through',
    f(rc({ ...closed, verdict: 'needs_fresh_elapsed' }), ['decode_m2_square']).reason);

  // Wiring: a gate whose result is computed and discarded is a comment (55).
  ok(/const priority = qdPriorityFilter\(d\.priority_receipt, d\.target_cases \|\| \[d\.context_id\]\);/.test(src),
    'the gate is invoked on every direction, over the same case set the engineer prompt is built with');
  ok(/\.\.\.d, target_cases: priorityCases, operator: op,/.test(src),
    'the filtered list replaces the planner\'s, so the engineer is aimed at it and not merely told');
  ok(/priority_receipt: obj\(\{/.test(src),
    'priority_receipt is in the planning schema, so a planner that returns it is not dropped');
  ok(/dropped by the route-priority gate -- \$\{priority\.reason\}/.test(src),
    'a direction dropped by this gate says which gate and why');
}

console.log('\n# the post-build policy receipt gate, executed');
{
  // Finding (69). `policy_pass` is the boolean the whole "no rocBLAS" constraint
  // reduces to, and it was consumed at five sites as a bare self-report. This
  // block executes the gate rather than grepping for it, because (57): a regex
  // proves the function exists, never that it refuses anything.
  const policyBlock = grab(/const qdPolicyReject = \(rep, label\) => \{[\s\S]*?\n\};\n/, 'qdPolicyReject');
  const reject = new Function(`${policyBlock}\nreturn qdPolicyReject;`)();
  const good = { schema: 'candidate-policy-scan', passed: true, findings: 0, advisory: 2,
    inspected: 11, files: 9, elf: 2, unreadable: 0 };
  const rep = (s) => ({ policy_pass: true, policy_postbuild: s });

  ok(reject(rep(good)) === null, 'a pass backed by a post-build receipt that saw an ELF is admitted');
  ok(reject({ policy_pass: true }) !== null,
    'a bare policy_pass with no post-build summary is refused');
  ok(reject(rep({ ...good, elf: 0 })) !== null,
    'a post-build scan that inspected no binary is refused -- DT_NEEDED is only visible there');
  ok(reject(rep({ ...good, files: 0, elf: 0 })) !== null,
    'a scan that opened no file at all is refused, not counted as passing');
  // `inspected` counts the directory entry, so an empty tree reads 1 there and 0
  // in `files`. Gating on the wrong one of those two is a gate that passes on
  // nothing at all.
  ok(reject(rep({ ...good, inspected: 1, files: 0, elf: 0 })) !== null,
    'an empty tree, which reads inspected:1, is still refused -- the gate reads files');
  ok(reject(rep({ ...good, passed: false })) !== null,
    'a receipt whose own verdict contradicts policy_pass is refused');
  ok(reject(rep({ ...good, findings: 1 })) !== null,
    'passed:true beside nonzero findings is self-contradictory and refused');
  ok(reject(rep({ ...good, unreadable: 1 })) !== null,
    'an uninspectable artifact fails closed, matching verify_engineer.md 2b');
  ok(reject(rep({ ...good, schema: 'qd-descriptor-v2' })) !== null,
    'a summary from some other tool is refused rather than read as a policy receipt');
  ok(reject(rep({ ...good, advisory: 7 })) === null,
    'advisory findings do not fail the gate -- (53): a gate that cries wolf gets ignored');

  // (60)/(61): the honest failure must keep its own name. If this gate spoke about
  // reports that never claimed a pass, a real `policy_failed` would be renamed into
  // a paperwork complaint and the caller's own refusal would never be reached.
  ok(reject({ policy_pass: false }) === null,
    'an honest policy_failed report is left alone for the caller to refuse under its own name');
  ok(reject(null) === null && reject(undefined) === null,
    'an absent report is not this gate\'s refusal to make');

  const reasons = [reject({ policy_pass: true }), reject(rep({ ...good, elf: 0 })),
    reject(rep({ ...good, files: 0 })), reject(rep({ ...good, passed: false })),
    reject(rep({ ...good, findings: 1 })), reject(rep({ ...good, unreadable: 1 })),
    reject(rep({ ...good, schema: 'qd-descriptor-v2' }))];
  ok(new Set(reasons.map(r => String(r).split('(')[0])).size === reasons.length,
    'the seven policy refusals are seven distinguishable tokens', JSON.stringify(reasons));
  ok(reasons.every(r => /^policy:/.test(String(r))),
    'every one of them is namespaced, so a log line says which gate spoke');
  ok(reject(rep(good), 'integrate') === null && /integrate/.test(String(reject({ policy_pass: true }, 'integrate'))),
    'the label names which consumer asserted the unbacked pass');

  // Wiring, all five consumers of policy_pass (63: enumerate the call sites of the
  // quantity, not of the helper).
  ok(/const seedPolicyReason = qdPolicyReject\(seed, 'canonical seed'\);/.test(src)
    && /throw new Error\(`QD bootstrap failed closed: \$\{seedPolicyReason\}`\);/.test(src),
    'the canonical seed -- the one policy claim no later stage re-checks -- throws rather than continues');
  ok(/qdPolicyReject\(ver, 'qd import verify'\)/.test(src),
    'the QD import verify loop refuses an unbacked imported elite');
  ok(/oracleDrift\(r\.ver\) \|\| qdPolicyReject\(r\.ver\)/.test(src),
    'the greedy admission path checks it beside the metric and oracle gates');
  ok(/const policyReason = r\.ver \? \(qdPolicyReject\(r\.ver\) \|\| qdTwinReject\(r\.ver\)\) : null;/.test(src),
    'the archive admission path refuses to store an elite on an unbacked pass');
  ok(/qdPolicyReject\(integrate, 'integrate'\) \|\| qdTwinReject\(integrate, 'integrate'\)/.test(src),
    'integrate discards a result whose policy claim has nothing behind it, and one '
    + 'whose hipified twin drifted -- a merge is the site where an edit is most '
    + 'likely to land in only one half of a hipify pair');
  ok(/policy_postbuild: POLICY_SUMMARY_SCHEMA/.test(src),
    'the summary is in the schemas, so an agent that returns it is not stripped');
  ok((src.match(/policy_postbuild: POLICY_SUMMARY_SCHEMA/g) || []).length >= 4,
    'in all four report schemas, not just the one that was easiest to reach');
}

console.log('\n# the hipify twin-sync gate, executed');
{
  // Finding (87), wired. The tool has been correct and uncalled for several
  // rounds. Its three exit codes are the point: 0 lockstep, 1 drift, 2 nothing
  // checked -- and 2 is the one a boolean `twin_pass` destroys.
  const block = grab(/const TWIN_LANGUAGES = [\s\S]*?const qdTwinReject = \(rep, label\) => \{[\s\S]*?\n\};\n/,
    'qdTwinReject');
  const mk = (lang) => new Function('TARGET_LANGUAGE',
    `${block}\nreturn {qdTwinReject, TWIN_APPLICABLE};`)(lang);
  const g = mk('hip');
  const t = (twin, extra) => ({ status: 'verified', correctness: 'pass',
    hip_twin_sync: twin, ...(extra || {}) });
  const good = { exit_code: 0, pairs: 1, drifted: 0 };
  const r = (rep, label) => g.qdTwinReject(rep, label);

  ok(r(t(good)) === null, 'a tree in lockstep passes');
  ok(r(t({ ...good, pairs: 3 })) === null, 'more than one pair is fine');
  ok(/twin:drifted/.test(String(r(t({ exit_code: 1, pairs: 2, drifted: 1 })))),
    'exit 1 is refused: the edit is not in the binary that was measured');
  // THE case this gate exists for. Exit 2 means the tool found nothing to
  // compare, printed `HOLE:`, and proved nothing -- and a verifier that
  // measured a build in a directory with no twins either did not build there or
  // scanned before the build.
  const hole = String(r(t({ exit_code: 2, pairs: 0, drifted: 0 })));
  ok(/twin:nothing_checked/.test(hole) && /HOLE, not its pass/.test(hole),
    'exit 2 is refused and says why, rather than reading as a pass', hole);
  // Exit 3 is the shape most easily mistaken for a pass: every line the tool
  // could read matched, and only the launch half is unread. A pair that reports
  // it has NOT been shown to be in lockstep, and a launch-only edit -- a grid,
  // a block size, a stream -- is exactly what would hide there.
  const unread = String(r(t({ exit_code: 3, pairs: 1, drifted: 0 })));
  ok(/twin:launch_unreadable/.test(unread) && /UNCHECKED/.test(unread),
    'exit 3 is refused and says which half went unread', unread);
  ok(/twin:receipt_absent/.test(String(r({ status: 'verified', correctness: 'pass' }))),
    'a verifier that never ran the tool is refused: a check nobody runs is a comment');
  ok(/twin:no_exit_code/.test(String(r(t({ pairs: 1 })))) &&
     /twin:no_exit_code/.test(String(r(t({ exit_code: '0', pairs: 1 })))),
    'a receipt with no integer exit code cannot be read as any of the three outcomes');
  ok(/twin:unknown_exit/.test(String(r(t({ exit_code: 7, pairs: 1 })))),
    'an exit code the tool cannot emit is refused rather than defaulted to pass');
  // Exit 0 has two consequences the tool itself guarantees, so violating either
  // means the receipt was written rather than run.
  ok(/twin:zero_pairs_with_exit_zero/.test(String(r(t({ exit_code: 0, pairs: 0, drifted: 0 })))),
    'exit 0 with no pairs is impossible: the tool returns 2 in that case');
  ok(/twin:exit_zero_with_drift/.test(String(r(t({ exit_code: 0, pairs: 2, drifted: 1 })))),
    'exit 0 with drift is impossible: the tool returns 1 in that case');
  ok(r(null) === null && r(undefined) === null,
    'an absent report is not this gate\'s refusal to make');
  ok(r({ status: 'failed' }) === null && r({ correctness: 'FAIL' }) === null,
    'an honest failure is refused under its own name, not renamed into paperwork (60)');
  ok(/qd import verify/.test(String(r({ status: 'verified' }, 'qd import verify'))),
    'the label names which consumer measured without checking');
  const reasons = [r({ status: 'verified' }), r(t({ pairs: 1 })), r(t({ exit_code: 1, pairs: 1, drifted: 1 })),
    r(t({ exit_code: 2, pairs: 0 })), r(t({ exit_code: 3, pairs: 1 })),
    r(t({ exit_code: 7, pairs: 1 })),
    r(t({ exit_code: 0, pairs: 0 })), r(t({ exit_code: 0, pairs: 2, drifted: 1 }))];
  ok(new Set(reasons.map(x => String(x).split('(')[0])).size === reasons.length,
    'the eight twin refusals are eight distinguishable tokens', JSON.stringify(reasons));
  ok(reasons.every(x => /^twin:/.test(String(x))),
    'every one is namespaced, so a log line says which gate spoke');

  // Scope. The default `target_language` is triton, which has no .hip twin to
  // compare -- the tool would return 2 forever and the gate would refuse every
  // candidate in the lane for a hazard that cannot occur in it.
  const tri = mk('triton');
  ok(tri.TWIN_APPLICABLE === false && tri.qdTwinReject({ status: 'verified' }) === null,
    'a triton lane is not held to a check about hipified .hip twins');
  ok(mk('HIP').TWIN_APPLICABLE === true && mk('cuda').TWIN_APPLICABLE === true,
    'hip and cuda arm it, case-insensitively -- those are the lanes ninja hipifies');
  ok(/twin:receipt_absent/.test(String(mk('hip').qdTwinReject({ status: 'verified' }))),
    'and the hip lane, which is the one this project runs, is held to it');

  // Wiring (55): a gate whose result is computed and discarded is a comment,
  // and this gate's whole finding is that the tool was never called at all.
  ok(/qdPolicyReject\(r\.ver\)\n\s*\|\| qdTwinReject\(r\.ver\);/.test(src),
    'the greedy admission path runs it beside the policy gate');
  ok(/qdPolicyReject\(r\.ver\) \|\| qdTwinReject\(r\.ver\)/.test(src),
    'the archive admission path runs it too: the archive outlives the run');
  ok(/qdTwinReject\(ver, 'qd import verify'\)/.test(src),
    'a re-verified imported snapshot is held to it as well');
  // Counted, not merely present. The sentence this asserts is plural -- the
  // receipt is declared everywhere a verdict carrying one arrives -- and a bare
  // presence match is satisfied by any single surviving declaration, so it stays
  // green while the schema that actually strips the field loses it. Three sites:
  // the two verify verdicts and the import re-verify.
  ok(/hip_twin_sync: HIP_TWIN_SCHEMA/.test(src) &&
     (src.match(/hip_twin_sync: HIP_TWIN_SCHEMA/g) || []).length === 3,
    'the receipt is in every schema, so a verifier that returns it is not stripped');
}

console.log('\n# the SOL bandwidth-ceiling provenance gate, executed');
{
  // Finding (70). `qdSolCaseValid` checks the card against itself, and every one
  // of its clauses holds by construction for a card built on a fabricated or
  // clamped ceiling -- so this gate has to be executed against cards that pass
  // that one. Grepping would prove only that it exists (57).
  const ceilBlock = grab(/const QD_SOL_CONFIDENCE = \[[\s\S]*?const qdSolCeilingReject = \(c, contextId\) => \{[\s\S]*?\n\};\n/,
    'qdSolCeilingReject');
  const reject = new Function(`${ceilBlock}\nreturn qdSolCeilingReject;`)();
  const validBlock = grab(/const qdSolCaseValid = \(c, contextId\) => \{[\s\S]*?\n\};\n/, 'qdSolCaseValid');
  const valid = new Function(`${validBlock}\nreturn qdSolCaseValid;`)();
  const MB = 1 << 20;
  const measured = { basis: 'footprint_table', confidence: 'measured_interpolated',
    extrapolated: false, footprint_bytes: 64 * MB, bracket: [32 * MB, 128 * MB] };
  // A memory-bound case: the memory floor is the binding one, so the bandwidth
  // ceiling IS the denominator.
  // Every fixture carries a witnessed compute ceiling so that the bandwidth
  // clauses below are testing the bandwidth clauses; the witness clauses get
  // their own section further down.
  const witnessed = { witnessed: true, attainment: 0.51,
    witness: 'shapeceil.py 8192^3 bf16, machine H' };
  const mem = (ceiling) => ({ name: 'decode_m8_up', measured_ms: 0.055, compute_floor_ms: 0.004,
    memory_floor_ms: 0.02, sol_ms: 0.02, sol_gap: 2.75, remaining_headroom: 1 - 1 / 2.75,
    roof: 'hbm', profile_regime: 'memory_bound', confidence: 'high', ceiling,
    compute_ceiling: witnessed });
  // ...and a compute-bound one, identical except for which floor binds.
  const comp = (ceiling) => ({ ...mem(ceiling), name: 'prefill_m2048_square',
    compute_floor_ms: 0.02, memory_floor_ms: 0.004 });

  ok(valid(mem(measured), 'decode_m8_up'), 'the fixture passes the self-consistency gate first');
  ok(reject(mem(measured), 'decode_m8_up') === null,
    'a memory-bound case on a measured interpolated ceiling is admitted');
  ok(reject(comp({ ...measured, confidence: 'low', extrapolated: true }), 'prefill_m2048_square') === null,
    'a clamped ceiling on a COMPUTE-bound case is admitted -- it is not the denominator (53)');

  // The clause with teeth.
  const clamped = { basis: 'footprint_table', confidence: 'low', extrapolated: true,
    footprint_bytes: 8 * MB, bracket: [32 * MB, 32 * MB] };
  ok(reject(mem(clamped), 'decode_m8_up') !== null,
    'a memory-bound case whose ceiling was clamped below the measured range is refused');
  ok(valid(mem(clamped), 'decode_m8_up'),
    'and that card is perfectly self-consistent -- which is why the second gate exists (59)');
  ok(reject(mem({ ...measured, confidence: 'unmeasured' }), 'decode_m8_up') !== null,
    'a physical reference peak is not a measured denominator');
  ok(reject(mem({ ...measured, basis: 'scalar', confidence: 'measured_scalar',
    footprint_bytes: undefined, bracket: undefined }), 'decode_m8_up') === null,
    'a calibrated scalar ceiling from the actual box is a measured denominator');

  ok(reject(mem(undefined), 'decode_m8_up') !== null,
    'a case that states a sol_ms but no ceiling is refused: the denominator is unaccountable');
  ok(reject(mem({ ...measured, basis: 'guess' }), 'decode_m8_up') !== null,
    'a basis the tool never emits is refused');
  ok(reject(mem({ ...measured, confidence: 'high' }), 'decode_m8_up') !== null,
    'a confidence word from some other vocabulary was not copied off a real card');
  ok(reject(mem({ ...measured, extrapolated: 'false' })) !== null,
    'the extrapolated flag must be a boolean, not the string that renders like one');
  ok(reject(mem({ ...measured, extrapolated: true }), 'decode_m8_up') !== null,
    'extrapolated:true beside a measured confidence is a contradiction the tool cannot produce');
  ok(reject(mem({ ...measured, footprint_bytes: 0 }), 'decode_m8_up') !== null,
    'a footprint_table ceiling with no footprint names no point on the table');
  ok(reject(mem({ ...measured, bracket: [128 * MB, 32 * MB] }), 'decode_m8_up') !== null,
    'a descending bracket is not a bracket');
  ok(reject(mem({ ...measured, footprint_bytes: 512 * MB }), 'decode_m8_up') !== null,
    'a footprint outside its own reported bracket, not marked extrapolated, is refused');

  const reasons = [reject(mem(undefined)), reject(mem({ ...measured, basis: 'guess' })),
    reject(mem({ ...measured, confidence: 'high' })), reject(mem({ ...measured, extrapolated: 'false' })),
    reject(mem({ ...measured, extrapolated: true })), reject(mem({ ...measured, footprint_bytes: 0 })),
    reject(mem({ ...measured, bracket: [1] })), reject(mem({ ...measured, footprint_bytes: 512 * MB })),
    reject(mem(clamped))];
  ok(new Set(reasons.map(r => String(r).split('(')[0])).size === reasons.length,
    'the nine ceiling refusals are nine distinguishable tokens (60)', JSON.stringify(reasons));
  ok(reasons.every(r => /^sol:/.test(String(r))),
    'every one is namespaced, so a log line says which gate spoke');
  ok(/Measure the footprint/.test(String(reject(mem(clamped)))),
    'the refusal that costs a cell says what would fix it, not just that it fired');

  // Wiring: both sites, plus the schema that lets the block survive the agent boundary.
  ok(/ceiling: SOL_CEILING_SCHEMA/.test(src) && /'confidence', 'ceiling', 'compute_ceiling'\]\);/.test(src),
    'the ceiling block is in the case schema and required, so a card without it fails at the boundary');
  ok(/const why = qdSolCeilingReject\(c, s\.context_id\);/.test(src),
    'site 1: the card admission path drops cases whose denominator is unbacked');
  ok(/const beforeCeilingReason = beforeCase\s*\n?\s*\? qdSolCeilingReject\(beforeCase, rc\.context_id\)/.test(src),
    'site 2: an inherited denominator is re-checked before it becomes this elite sol_resolved');
  ok(/sol_resolved: \(afterGap == null \|\| beforeCeilingReason\) \? null : \{/.test(src),
    'and sol_resolved is withheld rather than published on an unbacked ceiling');
  ok(/ceiling: beforeCase\.ceiling,/.test(src),
    'the ceiling travels with the number it produced (51)');

  // ---- (89) item 2: the compute half, executed --------------------------
  //
  // The bandwidth clauses above ask where a ceiling came from. That question
  // has no useful answer for a compute peak: `rocminfo` reads 1307 TFLOP/s off
  // this box, which is flawless provenance and unreachable, and a gap measured
  // against it says "4x left" about every route AND about rocBLAS. So the
  // compute clause asks a different question -- did anything reach it -- and
  // the fixtures below are all self-consistent, all well-provenanced, and
  // separated only by that.
  const unwitnessed = { witnessed: false };
  ok(reject(comp(measured), 'prefill_m2048_square') === null,
    'a compute-bound case on a witnessed peak is admitted');
  ok(reject({ ...comp(measured), compute_ceiling: unwitnessed }, 'prefill_m2048_square') !== null,
    'the same case on an unwitnessed peak is refused: the denominator is a nameplate');
  ok(valid({ ...comp(measured), compute_ceiling: unwitnessed }, 'prefill_m2048_square'),
    'and it too is perfectly self-consistent, which is why this gate is not the (59) one');
  ok(reject({ ...mem(measured), compute_ceiling: unwitnessed }, 'decode_m8_up') === null,
    'a MEMORY-bound case on an unwitnessed compute peak is admitted -- the peak is not the '
    + 'denominator there, and refusing it would be (53) and would fail-shut a whole arch (87)');
  ok(reject({ ...comp(measured), compute_ceiling: undefined }, 'prefill_m2048_square') !== null &&
     reject({ ...mem(measured), compute_ceiling: undefined }, 'decode_m8_up') !== null,
    'an absent block is refused on both, because absence is not the same claim as witnessed:false');
  ok(reject({ ...comp(measured), compute_ceiling: { witnessed: 'true', attainment: 0.5, witness: 'x' } },
    'prefill_m2048_square') !== null,
    'the witnessed flag must be a boolean, not the string that renders like one');
  ok(reject({ ...comp(measured), compute_ceiling: { witnessed: true, attainment: 0.5, witness: '  ' } },
    'prefill_m2048_square') !== null,
    'witnessed:true with nobody named is the unevidenced claim the field exists to replace');
  ok(reject({ ...comp(measured), compute_ceiling: { ...witnessed, attainment: 1.4 } },
    'prefill_m2048_square') !== null,
    'attainment above 1 means something outran the ceiling, so the ceiling is wrong');
  ok(reject({ ...comp(measured), compute_ceiling: { ...witnessed, attainment: 0 } },
    'prefill_m2048_square') !== null,
    'a zero attainment is not a witness, it is a missing one wearing a number');
  ok(reject({ ...mem(measured), compute_ceiling: { witnessed: true, attainment: 1.4, witness: 'x' } },
    'decode_m8_up') !== null,
    'a malformed witness is refused even where the compute peak does not bind: it is a card '
    + 'defect, not a headroom judgement');
  const witnessReasons = [
    reject({ ...comp(measured), compute_ceiling: undefined }),
    reject({ ...comp(measured), compute_ceiling: { witnessed: true, attainment: 0.5, witness: '' } }),
    reject({ ...comp(measured), compute_ceiling: { ...witnessed, attainment: 1.4 } }),
    reject({ ...comp(measured), compute_ceiling: unwitnessed })];
  ok(new Set(witnessReasons.map(r => String(r).split('(')[0])).size === witnessReasons.length,
    'the four witness refusals are four distinguishable tokens (60)', JSON.stringify(witnessReasons));
  ok(witnessReasons.every(r => /^sol:/.test(String(r))),
    'and all namespaced like the bandwidth ones');
  ok(/Witness the peak or leave the case out/.test(
    String(reject({ ...comp(measured), compute_ceiling: unwitnessed }))),
    'the refusal that costs a cell says what would fix it');
  ok(/compute_ceiling: SOL_COMPUTE_CEILING_SCHEMA/.test(src),
    'the witness block survives the agent boundary, so a card without it fails there too');
  // (`qd_v2.py` invoking its own `validate_sol_card` is checked in
  // test_qd_v2.py: this runner only serves the two lane files.)
}

console.log('\n# content-addressed artifact provenance, executed');
{
  // Finding (71). The artifact map is keyed by a content hash, so a second write
  // under one key is never new information about the tree -- only a different
  // story about where those bytes came from.
  const recBlock = grab(/const qdRecordArtifact = \(hash, record, where\) => \{[\s\S]*?\n\};\n/,
    'qdRecordArtifact');
  const mkRec = () => {
    const archive = { artifacts: {} };
    const lines = [];
    const fn = new Function('qdArchive', 'log', `${recBlock}\nreturn qdRecordArtifact;`)(
      archive, (m) => lines.push(m));
    return { archive, lines, fn };
  };
  const H = 'a'.repeat(64);
  const seed = { source_hash: H, snapshot: '/e/qd_archive/artifacts/a/workspace',
    parent_workspace: '/e/canonical', generation: 0 };

  let t = mkRec();
  t.fn(H, seed, 'canonical seed');
  ok(t.archive.artifacts[H] === seed, 'the first write of a hash is recorded as given');
  // The exact collision this was found on: a warm import of an archive produced
  // from the same canonical source.
  const imported = { source_hash: H, snapshot: '/elsewhere/snap', generation: 0 };
  const kept = t.fn(H, imported, 'warm import');
  ok(kept === seed && t.archive.artifacts[H].parent_workspace === '/e/canonical',
    'a warm import of the same bytes does not erase parent_workspace: CANONICAL from the seed');
  ok(t.lines.length === 1 && /keeps its first provenance/.test(t.lines[0]) &&
     /warm import/.test(t.lines[0]),
    'and the disagreement is logged, naming the writer that would have overwritten it',
    JSON.stringify(t.lines));

  t = mkRec();
  t.fn(H, seed, 'canonical seed');
  t.fn(H, { source_hash: H, snapshot: seed.snapshot, parent_workspace: '/e/canonical', generation: 0 },
    'round 3 admission');
  ok(t.lines.length === 0,
    'an identical re-record is silent -- (53): a gate that cries wolf gets ignored');
  t.fn(H, { source_hash: H, snapshot: seed.snapshot, parent_workspace: '/e/parent', generation: 3 },
    'round 3 admission');
  ok(t.lines.length === 1 && /generation 0/.test(t.lines[0]) && /generation 3/.test(t.lines[0]),
    'a no-op round candidate cannot restamp its own parent record with a later generation');

  // Isolate the parent_workspace clause: same snapshot, same generation, a
  // different parent. That is the import collision stripped to the one field
  // the other two clauses cannot see.
  t = mkRec();
  t.fn(H, seed, 'canonical seed');
  t.fn(H, { ...seed, parent_workspace: '/e/round2_ws' }, 'round 2 admission');
  ok(t.lines.length === 1 && t.archive.artifacts[H].parent_workspace === '/e/canonical',
    'a differing parent_workspace alone is a provenance disagreement and is kept out',
    JSON.stringify(t.lines));
  t.fn(H, { ...seed, parent_workspace: undefined }, 'round 2 admission');
  ok(t.lines.length === 2,
    'and so is dropping it entirely -- unrecorded provenance is not agreement with recorded provenance');

  const other = 'b'.repeat(64);
  t.fn(other, { source_hash: other, snapshot: '/x', generation: 1 }, 'round 1 admission');
  ok(t.archive.artifacts[other].generation === 1 && t.archive.artifacts[H].generation === 0,
    'distinct content addresses are still independent records');

  // All three writers go through it (63: enumerate the sites of the quantity).
  ok(!/qdArchive\.artifacts\[[^\]]*\] = \{/.test(src),
    'no site writes the artifact map directly any more');
  ok(/qdRecordArtifact\(seed\.source_hash, \{/.test(src) &&
     /qdRecordArtifact\(ver\.source_hash, \{/.test(src) &&
     /qdRecordArtifact\(sourceHash, \{/.test(src),
    'seed, warm import and round admission all record through the one helper');
  // The no-op gate is checked lexically here and executably by the mutation
  // probe in test_js_suite.py: it is one condition inside a filter closure, and
  // rebuilding that closure here would test my copy of it rather than the lane.
  ok(/r\.ver\.source_hash === r\.ver\.seed_source_hash/.test(src) &&
     /hash:noop_patch/.test(src),
    'a candidate whose post-patch hash equals its pre-patch hash is refused by name');
}

// Finding (76): the archive's READ path. (44) asked whether the archive fills;
// this asks whether anything downstream can see what it filled with, which is
// an independent failure and was a total one -- the mechanism record reached no
// agent. `qdSummary()` is the only archive window every role gets, so these are
// lexical checks against that one projection rather than against the writer.
console.log('\n# (76) the mechanism memory is readable, not just durable');
{
  const summary = grab(/const qdSummary = \(\) => \{[\s\S]*?\n\};\n/, 'qdSummary');
  // The per-cell projection, isolated: a field can appear in the writer at
  // :2404 and still be invisible, which is exactly the bug, so matching
  // anywhere in the file would pass while the read path stayed broken.
  const cellProj = summary.match(/cells: Object\.fromEntries[\s\S]*?\}\]\)\),/)[0];
  for (const f of ['strategy_capsule', 'parent_elite_id', 'parent_elite_ids', 'cost']) {
    ok(new RegExp(`\\b${f}:`).test(cellProj), `qdSummary cell projection exposes ${f}`);
  }
  ok(/\n    capsules: Object\.fromEntries/.test(summary),
    'qdSummary exposes the capsule ledger, not only the count that gates a repeat');
  ok(/\n    lineage: qdArchive\.lineage,/.test(summary),
    'qdSummary exposes lineage, which tech_lead.md already names');
  // (108). The count and the diversity must arrive together: `structural_coverage`
  // alone cannot distinguish an archive that found twelve mechanisms from one
  // that re-filed two, and it is the number sitting right next to it.
  ok(/\n    variant_spread: qdVariantSpread\(qdArchive\.cells\),/.test(summary),
    'qdSummary reports variant spread beside structural_coverage, not instead of it');
  ok(/structural_coverage:/.test(summary),
    'structural_coverage is still reported -- the fix is a second number, not a swap');
  // The third corner of (44) -- that the PROMPT names this field -- is checked
  // in test_qd_lane_parity.py, not here: this runner serves only the two lane
  // sources and cannot read `roles/`.

  // The key is the identity of a direction on a ROUTE. Keyed on the parent's
  // source hash it evaporated on every elite replacement -- i.e. exactly when
  // the same direction gets proposed again against the fresh parent.
  const key = grab(/const capsuleKey = QD_ENABLED[\s\S]*?;\n/, 'capsuleKey');
  ok(!/parent\.source_hash/.test(key),
    'the capsule key is not keyed on the parent source hash');
  ok(/parent\.route_id/.test(key) && /parent\.context_id/.test(key) && /mechanism/.test(key),
    'the capsule key is route + context + mechanism, so it survives elite replacement');

  // A capsule entry that carries only outcome flags is a tally: it can block a
  // literal repeat and cannot tell the next planner what was tried.
  const push = grab(/qdArchive\.capsules\[r\.d\.capsule_key\]\.push\(\{[\s\S]*?\}\);\n/, 'capsule push');
  // Pinned on the VALUE, not on the key. `capsule: {}` satisfies `capsule:`
  // perfectly while writing an entry that says a direction was tried and never
  // what it was -- which is the tally (76) exists to stop being. A mutant that
  // empties the field survived this check for exactly that reason.
  for (const [f, rhs] of [['capsule', 'r\\.d\\.strategy_capsule'],
                          ['expected_effect', 'r\\.d\\.expected_effect'],
                          ['observed_effect', 'effect'],
                          ['elite_id', 'eliteId']]) {
    ok(new RegExp(`\\b${f}: ${rhs}\\b`).test(push),
      `capsule entries carry ${f}, and carry it from the direction rather than as a placeholder`);
  }

  // (76)'s second failure mode: a prompt that asks for a field the input does
  // not contain. That is worse than asking for nothing -- it yields confident
  // text about an absent field instead of an error.
  const distill = grab(/'Distill insights, transition lessons[\s\S]*?:\n/, 'distill instruction');
  ok(/QD_ARCHIVE\.capsules/.test(distill) && /QD_ARCHIVE\.recent_transitions/.test(distill),
    'the update_qd_memory task names the archive fields it asks the tech lead to distill');

  // The property that actually matters, executed rather than asserted: the
  // negative memory must survive an elite replacement. Under the old key the
  // parent hash was part of the identity, so replacing the elite minted a
  // fresh key and the twice-failed direction became proposable again -- with
  // the archive still holding the two failures, unreachable behind a key
  // nothing would look up. Both the key and the gate are lifted out of
  // kernel_lane.js so this tests the lane and not a restatement of it.
  const keySrc = grab(/const capsuleKey = QD_ENABLED[\s\S]*?;\n/, 'capsuleKey');
  const gateSrc = grab(/const priorLocal = QD_ENABLED[\s\S]*?\n    \}\n/, 'priorLocal gate');
  const mech = JSON.stringify({ k_pipeline: 'lds_double' });
  const archive = { capsules: {} };
  const oldParent = { route_id: 'decode_m8_up', context_id: 'decode_m8_up', source_hash: 'a'.repeat(64) };
  const newParent = { route_id: 'decode_m8_up', context_id: 'decode_m8_up', source_hash: 'f'.repeat(64) };
  const keyOf = (parent) => new Function('QD_ENABLED', 'parent', 'mechanism',
    `${keySrc}\nreturn capsuleKey;`)(true, parent, mech);
  archive.capsules[keyOf(oldParent)] = [
    { operator: 'local_mutation', improved: false, regime_changed: false },
    { operator: 'local_mutation', improved: false, regime_changed: false },
  ];
  ok(keyOf(oldParent) === keyOf(newParent),
    'two different parents on the same route+mechanism share one capsule key');
  const counted = (parent) => (archive.capsules[keyOf(parent)] || []).filter(x =>
    x.operator === 'local_mutation' && x.improved !== true && x.regime_changed !== true).length;
  ok(counted(oldParent) === 2 && counted(newParent) === 2,
    'the two recorded failures are still found after the cell elite is replaced');
  // The gate is deliberately NOT asserted on separately. `!/parent.source_hash/`
  // over the gate body passes identically before and after the fix -- the
  // parent hash was never in the gate, it was in the key the gate looks up --
  // so that check would have been a green light that could not go red. The two
  // executed checks above are the ones that distinguish the two worlds, and the
  // mutation probe confirms they do. The gate itself IS executed -- in the
  // three-generation section immediately below, which is where it can be run
  // against an archive that has actually had its elite replaced.
  void gateSrc;
}

console.log('\n# item 2b acceptance, executed across three generations (76)/(78)');
{
  // Roadmap item 2b's acceptance criterion is not a speedup. Verbatim: "a
  // mechanism admitted in generation N is quotable, by name, from the planner's
  // input in generation N+2, and a direction that failed twice on a route stays
  // blocked across an elite replacement", and the item itself says both are
  // checkable here, without a GPU, and should be checked here BEFORE the smoke
  // run spends GPU on them.
  //
  // Finding (78) left this open for a reason worth keeping in view: the live
  // budget-2 run put both parents in a single round, so every mutation landed in
  // generation 1 and the run HAD no generation 2. The hole was labelled rather
  // than passed. A run cannot be made to have a generation it does not have --
  // but the archive can be driven through three generations here, which is what
  // the criterion is actually about.
  //
  // Everything load-bearing below is lifted out of kernel_lane.js and executed:
  // the capsule key, the repeat gate, and the qdSummary projection. The two
  // stubs (`qdVariantSpread`, `qdContextScore`) are scoring functions this
  // section does not assert on and that have their own sections above.
  const keySrc = grab(/const capsuleKey = QD_ENABLED[\s\S]*?;\n/, 'capsuleKey');
  const gateSrc = grab(/const priorLocal = QD_ENABLED[\s\S]*?\n    \}\n/, 'priorLocal gate');
  const summarySrc = grab(/const qdSummary = \(\) => \{[\s\S]*?\n\};\n/, 'qdSummary');

  const ROUTE = 'prefill_m1024_down';
  const OTHER = 'decode_m8_up';
  // The mechanism admitted in generation N. A name a planner could quote.
  const MECH_N = { k_pipeline: 'lds_double_buffer', epilogue: 'fused_bias' };
  const descKey = JSON.stringify(MECH_N);

  const qdArchive = {
    version: 'geak-qd-v2', classifier_version: 'v2', generation: 0,
    cells: {}, challengers: {}, artifacts: {}, transitions: [], lineage: {},
    visits: {}, capsules: {}, preconditions: {},
    coverage_stall: 0, qd_score_stall: 0, global_stall: 0, global_best: null,
  };
  const keyOf = (parent, mechanism) => new Function('QD_ENABLED', 'parent', 'mechanism',
    `${keySrc}\nreturn capsuleKey;`)(true, parent, mechanism);
  // The REAL gate. Returns null exactly where kernel_lane.js returns null, i.e.
  // where the direction is dropped from the round.
  const propose = (parent, mechanism, op) => new Function(
    'QD_ENABLED', 'qdArchive', 'capsuleKey', 'op', 'log', 'round', 'd', 'i',
    `${gateSrc}\nreturn 'proposed';`)(
      true, qdArchive, keyOf(parent, mechanism), op, () => {}, 3, { id: 'd0' }, 0);
  const summarize = () => new Function(
    'qdArchive', 'QD_CONTEXT_IDS', 'QD_ARCH', 'qdVariantSpread', 'qdContextScore',
    `${summarySrc}\nreturn qdSummary();`)(
      qdArchive, new Set([ROUTE, OTHER]), 'gfx942', () => 0, () => 0);

  // Each generation replaces the cell's elite with a NEW build -- new elite id,
  // new source hash -- which is the event that used to erase the negative
  // memory, because the old key had the parent hash in it.
  const advance = (gen, hash, capsule, outcome) => {
    const parent = { route_id: ROUTE, context_id: ROUTE, source_hash: hash };
    qdArchive.generation = gen;
    qdArchive.cells[ROUTE] = {
      elite_id: `e_gen${gen}`, context_id: ROUTE, route_id: ROUTE, source_hash: hash,
      descriptor: capsule, geomean: 1.0 + gen / 100, generation: gen,
      operator: gen ? 'local_mutation' : 'bootstrap', strategy_capsule: capsule,
      parent_elite_id: gen ? `e_gen${gen - 1}` : null, parent_elite_ids: [],
    };
    const k = keyOf(parent, JSON.stringify(capsule));
    (qdArchive.capsules[k] = qdArchive.capsules[k] || []).push({
      operator: 'local_mutation', improved: outcome, regime_changed: false,
      generation: gen, capsule, expected_effect: 'hide the K-loop LDS latency',
      observed_effect: outcome ? 'geomean +1%' : 'no change outside noise',
      elite_id: `e_gen${gen}`,
    });
    return parent;
  };

  // Generation N: the mechanism is admitted and it does not pay off.
  const pN = advance(0, 'a'.repeat(64), MECH_N, false);
  // Generation N+1: same direction, tried again on a REPLACED elite, still flat.
  const pN1 = advance(1, 'b'.repeat(64), MECH_N, false);
  // A different mechanism lands in between, so the ledger has more than one key
  // and "quotable by name" is a real lookup rather than the only thing present.
  advance(2, 'c'.repeat(64), { k_pipeline: 'async_copy' }, true);

  // --- acceptance clause 1: quotable BY NAME at N+2 --------------------------
  const planner = summarize();
  const asText = JSON.stringify(planner);
  ok(asText.includes('lds_double_buffer'),
    'the generation-N mechanism is still quotable by name from the planner input at N+2');
  const ledger = planner.capsules[keyOf(pN, descKey)];
  ok(!!ledger && ledger.attempts === 2 && ledger.unimproved_local === 2,
    'and it arrives with its outcome history, not merely as a name',
    ledger ? JSON.stringify(ledger.attempts) : 'absent');
  ok(!!ledger && ledger.recent.some(x => x.generation === 0 &&
      x.expected_effect === 'hide the K-loop LDS latency' &&
      x.observed_effect === 'no change outside noise'),
    'including what generation N EXPECTED and what it actually measured');
  // The load-bearing half of that, stated as the thing it distinguishes: the
  // CELL projection cannot carry this. Replacing the elite overwrote the cell's
  // strategy_capsule with `async_copy`, so if the ledger were not in the summary
  // the generation-N mechanism would be gone from the planner's view entirely --
  // durably stored, and unreadable. That is the (76) failure in one line.
  ok(JSON.stringify(planner.cells[ROUTE].strategy_capsule) ===
     JSON.stringify({ k_pipeline: 'async_copy' }),
    'the cell projection holds only the CURRENT elite mechanism, so the ledger is the path that carries N');

  // --- acceptance clause 2: blocked across an elite replacement --------------
  // Two recorded failures on this route+mechanism, and the elite has been
  // replaced twice since the first. Under the old parent-hash key this lookup
  // found an empty ledger and the direction was proposable again.
  ok(propose(pN1, descKey, 'local_mutation') === null,
    'a third local mutation of the twice-failed direction is REFUSED after the elite was replaced');
  ok(propose(pN1, descKey, 'parameter_tuning') === null,
    'and so is the same direction relabelled as parameter tuning');
  // The block is scoped, in both directions. Scope errors here are expensive
  // twice over: too wide and the search is choked off a route it never tried,
  // too narrow and (76) is back.
  ok(propose(pN1, descKey, 'deep_mutation') === 'proposed',
    'a DEEP mutation is not blocked -- the ledger blocks repetition, not exploration');
  ok(propose({ route_id: OTHER, context_id: OTHER, source_hash: 'b'.repeat(64) },
             descKey, 'local_mutation') === 'proposed',
    'the same mechanism on a DIFFERENT route is untouched -- the memory is route-scoped');
  ok(propose(pN1, JSON.stringify({ k_pipeline: 'async_copy' }), 'local_mutation') === 'proposed',
    'and a different mechanism on the same route is untouched -- one success is not two failures');

  // --- the truncation is honest ---------------------------------------------
  // `recent` is capped at 4 so a long run cannot crowd the prompt. A cap is only
  // safe if the count beside it stays true; otherwise a truncated list reads as
  // a complete one and the planner concludes a direction was tried twice when it
  // was tried nine times.
  for (let g = 3; g < 10; g++) advance(g, String(g).repeat(64), MECH_N, false);
  const late = summarize().capsules[keyOf(pN, descKey)];
  ok(late.attempts === 9 && late.unimproved_local === 9,
    'after nine attempts the count is nine', JSON.stringify(late.attempts));
  ok(late.recent.length === 4 && late.recent[3].generation === 9,
    'while the quoted list is capped at the most recent four', JSON.stringify(late.recent.length));
}

// ---------------------------------------------------------------------------
// # the roadmap profile may not publish a ceiling it did not earn -- (89)
// ---------------------------------------------------------------------------
{
  const block = grab(/const SOL_SHAPED_KEY = [\s\S]*?const profileSolStrip = \(rep, where\) => \{[\s\S]*?\n\};\n/,
    'profileSolStrip');
  const lines = [];
  const { profileSolStrip } = new Function('log',
    `${block}\nreturn {profileSolStrip};`)((m) => lines.push(String(m)));

  const rep = {
    bottleneck: 'latency', dispatch_count: 1,
    sol_gap_x: 12, peak_pct: 8.36, compute_floor_us: 71.6, memory_floor_us: 34.6,
    remaining_headroom: 0.917, roofline_basis: 'bf16 MFMA peak ~1.29 PFLOP/s',
    key_metrics: { valu_pct: 61.2, hbm_gbps: 2680, sol_gap: 12, pct_of_peak: 8.36 },
    top_kernels: [{ name: 'gemm', pct_of_total: 98.74 }],
    top_opportunities: ['shorten the C1 dependency chain'],
  };
  const out = profileSolStrip(rep, 'baseline profile');

  ok(out.sol_gap_x === undefined && out.peak_pct === undefined
    && out.compute_floor_us === undefined && out.memory_floor_us === undefined
    && out.remaining_headroom === undefined && out.roofline_basis === undefined,
    'every SOL-shaped top-level field is removed from the roadmap profile');
  ok(out.key_metrics.sol_gap === undefined && out.key_metrics.pct_of_peak === undefined,
    'key_metrics is scrubbed too -- additionalProperties:true is exactly where the '
    + 'ungrounded number hides from the schema');
  ok(out.key_metrics.valu_pct === 61.2 && out.key_metrics.hbm_gbps === 2680
    && out.bottleneck === 'latency' && out.dispatch_count === 1
    && out.top_kernels[0].pct_of_total === 98.74
    && out.top_opportunities.length === 1,
    'the measured counters survive -- the counters were real and were adopted; only '
    + 'the denominators were ungrounded');
  ok(/pct_of_total/.test(JSON.stringify(out)),
    'pct_of_total is not a SOL field and the prefix match does not eat it');
  ok(lines.length === 1 && /dropped ungrounded SOL fields/.test(lines[0])
    && /sol_gap_x/.test(lines[0]) && /key_metrics\.sol_gap/.test(lines[0]),
    'the removal is logged by name and qualified path, not silent -- a strip nobody '
    + 'is told about is indistinguishable from an agent that never emitted it');
  ok(typeof out.sol_note === 'string' && /selected_cell_sol/.test(out.sol_note),
    'the reader is pointed at the branch that can answer the question, so the gap '
    + 'reads as "ask elsewhere" rather than "no headroom analysis exists"');

  const clean = { bottleneck: 'compute', key_metrics: { valu_pct: 90 } };
  const untouched = profileSolStrip(clean, 'baseline profile');
  ok(untouched.sol_note === undefined && lines.length === 1,
    'a profile that published no ceiling is not annotated and logs nothing');
  ok(profileSolStrip(null, 'x') === null && profileSolStrip(undefined, 'x') === undefined,
    'a missing profile does not throw on the way through the strip');

  ok(/profileSolStrip\(profileSummary, 'baseline profile'\)/.test(src)
    && /profileSolStrip\(profileSummary, `reprofile r\$\{round\}`\)/.test(src),
    'both profile branches are stripped -- reprofile writes to the same variable the '
    + 'TechLead reads, so covering only the baseline would leak from round 1 on');
}

// ---------------------------------------------------------------------------
// # arch preconditions: a second namespace the capsule ledger cannot serve -- (86)
// ---------------------------------------------------------------------------
{
  const kindsSrc = grab(/const PRECONDITION_KINDS = \[[^\]]*\];\n/, 'PRECONDITION_KINDS');
  const writerSrc = grab(/const qdRecordPreconditions = \(arch, list, source\) => \{[\s\S]*?\n\};\n/,
    'qdRecordPreconditions');
  const lines = [];
  const arch = { preconditions: {}, generation: 3 };
  const { qdRecordPreconditions } = new Function('qdArchive', 'log',
    `${kindsSrc}\n${writerSrc}\nreturn {qdRecordPreconditions};`)(arch, (m) => lines.push(String(m)));

  const good = { id: 'gfx942-arch-guard', kind: 'build_guard',
    statement: 'the #if guard admits gfx90a only', evidence: 'hipcc error at custom_gemm.hip:41' };
  let r = qdRecordPreconditions('gfx942', [good], 'canonical seed bootstrap');
  ok(r.recorded === 1 && arch.preconditions.gfx942.length === 1,
    'an evidenced precondition is filed under the arch');
  ok(arch.preconditions.gfx942[0].arch === 'gfx942'
    && arch.preconditions.gfx942[0].generation === 3
    && arch.preconditions.gfx942[0].established_by === 'canonical seed bootstrap',
    'the record carries its own arch, generation and origin, so a persisted manifest '
    + 'read back later still says where the fact came from');

  r = qdRecordPreconditions('gfx942', [{ ...good, evidence: '   ' }], 'x');
  ok(r.recorded === 0 && r.refused === 1 && /no evidence/.test(lines[lines.length - 1]),
    'an unevidenced precondition is refused BY ID and logged -- it would otherwise be '
    + 'inherited by every later run and checkable by none');

  r = qdRecordPreconditions('gfx942', [{ ...good, id: 'k2', kind: 'obviously_made_up' }], 'x');
  ok(r.recorded === 0 && r.refused === 1 && /is not one of/.test(lines[lines.length - 1]),
    'the kind vocabulary is closed, like the descriptor axes, so the namespace stays readable');

  r = qdRecordPreconditions('gfx942', [{ ...good, statement: 'actually the guard is fine' }], 'round 2');
  ok(r.recorded === 0 && r.conflicts === 1 && arch.preconditions.gfx942.length === 1
    && arch.preconditions.gfx942[0].statement === 'the #if guard admits gfx90a only'
    && /CONFLICT/.test(lines[lines.length - 1]),
    'a precondition that flips inside one run is surfaced as a contradiction, not '
    + 'silently updated -- first writer wins');

  r = qdRecordPreconditions('gfx942', [good], 'round 3');
  ok(r.recorded === 0 && r.conflicts === 0 && arch.preconditions.gfx942.length === 1,
    'restating the same fact is idempotent and is not a conflict');

  r = qdRecordPreconditions('gfx90a', [good], 'other lane');
  ok(arch.preconditions.gfx90a.length === 1 && arch.preconditions.gfx942.length === 1,
    'the namespace is keyed by arch: a fact about one card does not appear under another');

  ok(qdRecordPreconditions('gfx942', undefined, 'x').recorded === 0
    && qdRecordPreconditions('gfx942', [], 'x').recorded === 0,
    'omitting the field is legal -- an arch that needed nothing established says so by '
    + 'omission rather than by a manufactured record');

  ok(!/capsule_key/.test(writerSrc) && !/qdDescriptorKey/.test(writerSrc),
    'preconditions are NOT filed under a descriptor key -- (71): one identity serving '
    + 'as two slots makes "failed twice on this route" unreadable');
  ok(/preconditions: \{ arch: QD_ARCH, records: qdArchive\.preconditions\[QD_ARCH\] \|\| \[\] \}/.test(src),
    'the planner sees the active arch\'s records in the same block as the capsule ledger, '
    + 'which is the read path (78)/(83) confirmed is live');
  ok(/qdRecordPreconditions\(QD_ARCH, seed\.preconditions, 'canonical seed bootstrap'\)/.test(src),
    'the write happens at bootstrap, not at admission -- a preflight repair produces no '
    + 'from_cell -> target_cell edge and would never reach the capsule write site');
}

console.log('\n# the authoritative baseline refuses an ambiguous frame, executed');
{
  // Findings (34c)/(122). `latency_ms`, `baseline_ms` and `execution_time_ms`
  // are three names for numbers that are NOT always the same measurement: in a
  // bench frame `baseline_ms` is the frozen oracle, in an archive elite row it
  // is the parent. A precedence list silently answers "which one wins" and
  // never asks "do they agree", which is how a geomean against the parent gets
  // reported as one against the oracle. The lane refuses instead of ranking.
  const baselineBlock = grab(
    /const QD_BASELINE_FIELDS = [\s\S]*?\n\}\)\.filter\(\(\[name, latency\]\)[^\n]*\n/,
    'QD_BASELINE_MS construction');
  // Fed as QD_BASELINE_PER_CASE, the QD-gated row set: outside qd_archive the
  // lane hands the block an empty list so a QD-only refusal cannot kill a
  // greedy run. These cases are all about what the block does WITH rows.
  const build = (rows) => new Function('QD_BASELINE_PER_CASE',
    `${baselineBlock}\nreturn QD_BASELINE_MS;`)(rows);
  const one = build([{ name: 'decode_m8', latency_ms: 0.0271 }]);
  ok(one.get('decode_m8') === 0.0271, 'a single field is read straight through');
  const agreeing = build([{ name: 'decode_m8', latency_ms: 0.0271,
    execution_time_ms: 0.02710015 }]);
  ok(agreeing.get('decode_m8') === 0.0271,
    'two fields within 0.1% are one measurement reported twice, not an ambiguity');
  let threw = null;
  try { build([{ name: 'prefill_m256_down', latency_ms: 0.1341, baseline_ms: 0.2004 }]); }
  catch (e) { threw = e.message; }
  ok(threw !== null, 'two baselines that disagree stop the bootstrap');
  ok(threw !== null && /prefill_m256_down/.test(threw)
     && /latency_ms=0\.1341/.test(threw) && /baseline_ms=0\.2004/.test(threw),
    'the refusal names the case and shows both numbers, so the operator can tell '
    + 'which frame is which', threw);
  let wide = null;
  try { build([{ name: 'decode_m8', latency_ms: 0.05, baseline_ms: 0.0505 }]); }
  catch (e) { wide = e.message; }
  ok(wide !== null,
    'the tolerance is tight enough that a real 1% parent-vs-oracle gap is not swallowed');
  const dropped = build([{ name: 'decode_m8', latency_ms: 0 },
    { name: 'prefill_m64', baseline_ms: 0.05 }]);
  ok(!dropped.has('decode_m8') && dropped.get('prefill_m64') === 0.05,
    'absence is dropped, not thrown on -- the bootstrap size check owns that failure '
    + 'and reports it per context');
  // The other half of (34c), lexical because it is a property of every row the
  // lane writes rather than of a function it calls: one number under an
  // ambiguous name contradicts nothing and still leaves the reader guessing
  // whether a speedup is against the oracle or against the parent.
  ok(/const QD_ROBUST_BASELINE_FRAME = 'oracle';/.test(src),
    "the lane's statistics are declared to be in the oracle frame");
  ok((src.match(/robust_baseline_frame: QD_ROBUST_BASELINE_FRAME/g) || []).length === 3,
    'all three elite construction sites -- seed, import, admission -- state their '
    + 'denominator; a label on some of them reads as a distinction, not an omission');
  ok((src.match(/elite_id: [^\n]*, cell: rc\.cell,/g) || []).length === 3,
    'there are exactly three elite construction sites, so the count above is not a '
    + 'magic number a fourth unlabelled site could satisfy');
}

console.log('\n# the ISA mechanism gate: language scope and what it may refuse, executed');
{
  // The layer above the twin gate. Twin-sync answers "was the edited file
  // compiled"; this answers "did the mechanism survive the compiler". Two things
  // are pinned here, and the first is a naming claim rather than a behaviour:
  //
  // The mode is the ONLY precondition: this must NOT consult TARGET_LANGUAGE. An
  // earlier version did, and it failed silently in the one way that matters --
  // TARGET_LANGUAGE defaults to 'triton' when the caller omits it (the greedy
  // pipeline omits it), so a whitelist turned the entire layer off with no message.
  // The real precondition is "did the build produce an AMDGPU code object", which
  // `isa_capture.py` measures directly and reports as its HOLE exit code. A loud
  // HOLE beats a silent off, so the protection lives in the non-refusal probes
  // below rather than in a language list.
  const block =
    grab(/const ISA_MODES = \['off', 'observe', 'gate'\][\s\S]*?const ISA_ENABLED = ISA_MODE !== 'off';\n/,
      'ISA_MODES..ISA_ENABLED') +
    grab(/const isaEvidenceReject = \(rep, label\) => \{[\s\S]*?\n\};\n/, 'isaEvidenceReject');
  const mk = (lang, mode) => new Function('A', 'TARGET_LANGUAGE',
    `${block}\nreturn {isaEvidenceReject, ISA_ENABLED, ISA_MODE, ISA_MODE_WARNING};`)(
    { isa_evidence: mode }, lang);

  ok(!/ISA_LANGUAGES/.test(src) && !/ISA_ENABLED[^\n]*TARGET_LANGUAGE/.test(src),
    'arming the ISA layer does not depend on TARGET_LANGUAGE -- a self-declared '
    + 'argument that defaults to triton is not a precondition, it is a silent off '
    + 'switch, and the real one (did a code object exist) is measured by the capture');
  ok(mk('triton', 'gate').ISA_ENABLED === true && mk('cuda', 'gate').ISA_ENABLED === true &&
     mk('hip', 'gate').ISA_ENABLED === true,
    'every language arms it identically; a lane whose build this cannot read gets a '
    + 'reported HOLE and an indeterminate verdict, not a gate that quietly did nothing');
  ok(mk('hip', 'gate').ISA_ENABLED === true && mk('HIP', 'observe').ISA_ENABLED === true,
    'both modes that do anything arm it');

  // Observation is ON by default, and that is load-bearing. The case this layer
  // catches -- a mechanism the compiler removed -- produces a round that reads as a
  // clean negative, so there is no symptom to escalate from. A conditional or
  // off-by-default check is therefore not a cheaper version of this check, it is one
  // that never fires on the case it was written for. Finding (87) is the standing
  // example of a correct tool nothing called.
  ok(mk('hip', undefined).ISA_MODE === 'observe' && mk('hip', undefined).ISA_ENABLED === true,
    'the default is observe: a lane that passes no isa_evidence arg still records '
    + 'whether each round\'s mechanism reached the machine code');
  ok(mk('hip', undefined).isaEvidenceReject({ status: 'verified', correctness: 'pass',
    isa_evidence: { exit_code: 0, mechanism_verdict: 'refuted' } }) === null,
    'and the default still refuses NOTHING -- default-on is an observation, not an '
    + 'enforcement, so turning it on by default cannot change a run\'s outcome');
  ok(mk('hip', 'off').ISA_ENABLED === false,
    '`off` remains a real escape hatch for an ablation or an unreadable box');
  ok(mk('hip', 'GATE ').ISA_MODE === 'gate',
    'the mode normalises case and whitespace');
  ok(mk('hip', 'nonsense').ISA_MODE === 'observe' &&
     /is not one of/.test(String(mk('hip', 'nonsense').ISA_MODE_WARNING)),
    'an unknown mode resolves to the default AND names itself -- resolving `gat` to '
    + '`off` in silence is the same silent-off failure the language whitelist had, and '
    + 'a typo is exactly how enforcement would get switched off unnoticed');
  ok(mk('hip', undefined).ISA_MODE_WARNING === '' && mk('hip', 'gate').ISA_MODE_WARNING === '',
    'and a legitimate mode warns about nothing, so the warning means what it says');

  const refuted = { status: 'verified', correctness: 'pass',
    isa_evidence: { exit_code: 0, mechanism_verdict: 'refuted',
      claims_refuted: ['widen_global_load'], unchanged_machine_code: false } };
  const gate = mk('hip', 'gate');
  ok(/isa:mechanism_refuted/.test(String(gate.isaEvidenceReject(refuted))),
    'in gate mode a refuted mechanism is refused, by name');
  ok(/not be recorded as one/.test(String(gate.isaEvidenceReject(refuted))),
    'and the refusal says the round must NOT be recorded as a slow kernel -- that '
    + 'distinction is the entire point: the mechanism was never tested, and a ledger '
    + 'row saying "no effect" closes the direction for the rest of a greedy run');
  ok(mk('hip', 'observe').isaEvidenceReject(refuted) === null,
    'observe mode records the same verdict and refuses nothing, so the signal can be '
    + 'measured on real kernels before anything is gated on it');

  // The asymmetry against (87), and the reason it is not an inconsistency. Exit 2
  // MUST refuse there, because the missing evidence is evidence about the
  // measurement itself and an unbacked measurement manufactures null results. Here
  // the measurement is already backed by twin-sync, policy and the oracle, so
  // missing ISA evidence is a gap on OUR side -- refusing on it would discard
  // correct fast kernels on any box where ROCm sits where the capture did not look.
  ok(gate.isaEvidenceReject({ status: 'verified', correctness: 'pass' }) === null,
    'a MISSING receipt does not refuse: that is our evidence gap, not a fault in the '
    + 'candidate -- the opposite of the twin gate, and deliberately so');
  ok(gate.isaEvidenceReject({ status: 'verified', correctness: 'pass',
    isa_evidence: { exit_code: 2, mechanism_verdict: 'indeterminate' } }) === null,
    'an archive HOLE does not refuse either, for the same reason');
  ok(gate.isaEvidenceReject({ status: 'verified', correctness: 'pass',
    isa_evidence: { exit_code: 0, mechanism_verdict: 'indeterminate' } }) === null,
    'nor does an indeterminate verdict -- reading absent evidence as a refutation '
    + 'would manufacture the very false negative this gate exists to prevent');
  ok(gate.isaEvidenceReject({ status: 'failed', correctness: 'FAIL',
    isa_evidence: { exit_code: 0, mechanism_verdict: 'refuted' } }) === null,
    'a report that already failed is refused on its own terms; stacking a mechanism '
    + 'complaint on top would rename the refusal (60)');

  // --- the escalation ladder, executed against fabricated histories -------
  // Extracted as a named function precisely so this block can run it. A rule about
  // when to spend the run's most expensive evidence, defended only by a regex on its
  // log message, is the shape `audit_pin_coverage.py` already caught once.
  const ladder = new Function(
    grab(/const isaEvidenceDepth = \(rounds, noImproveCount, enabled\) => \{[\s\S]*?\n\};\n/,
      'isaEvidenceDepth') + '\nreturn isaEvidenceDepth;')();
  const rnd = (depth, mechanisms) => ({
    evidence_depth: depth,
    results: (mechanisms || []).map(m => ({ mechanism: m })),
  });

  ok(ladder([], 0, true).depth === 'pattern' && ladder([rnd('pattern')], 0, true).depth === 'pattern',
    'a round that is still improving stays at pattern depth -- the ladder does not spend '
    + 'machine-code analysis on a search that is working');
  ok(ladder([rnd('pattern')], 1, true).depth === 'isa',
    'ONE non-improving round escalates to machine-code evidence: MAX_NO_IMPROVE defaults to 2, '
    + 'so a three-round stagnation window would never fire before the lane was already dead');
  ok(/actually realised/.test(String(ladder([rnd('pattern')], 1, true).reason)),
    'and the reason says what the evidence is for -- establishing whether the last direction '
    + 'was realised at all, which a flat benchmark cannot distinguish from a bad idea');
  ok(ladder([rnd('isa')], 1, true).depth === 'compiler' &&
     ladder([rnd('isa')], 1, true).from === 'isa',
    'a plateau that survived machine-code attribution escalates once more, to the compiler, '
    + 'and records what it escalated FROM');
  ok(ladder([rnd('pattern', ['refuted'])], 0, true).depth === 'compiler',
    'a REFUTED mechanism jumps straight to the compiler even on an improving round: the ISA '
    + 'has already answered "did it land" with no, so re-asking it re-derives what is known');
  ok(ladder([rnd('pattern', ['realized'])], 0, true).depth === 'pattern' &&
     ladder([rnd('pattern', ['indeterminate'])], 0, true).depth === 'pattern',
    'and neither a realized nor an indeterminate verdict triggers it -- only a positive '
    + 'contradiction does, so missing evidence never spends a deep round');
  ok(ladder([rnd('pattern')], 3, true).depth === 'isa',
    'the ladder climbs one step at a time; it does not skip to the compiler because the lane '
    + 'has been flat longer');
  ok(ladder([rnd('isa')], 5, false).depth === 'pattern',
    'and with the ISA layer off it never escalates at all, whatever the history looks like');

  // Wiring (55): a gate whose result is computed and discarded is a comment.
  ok(/const isaDepthState = isaEvidenceDepth\(history\.rounds, noImprove, ISA_ENABLED\);/.test(src),
    'the round loop actually consults the ladder rather than deciding depth inline');
  // Executed, not regex-matched. The first version of this asserted that the string
  // `pattern_after_failed_escalation` appeared in the lane -- and it did, in the
  // COMMENT beside the code, so the mutation that deleted the rule left the check
  // green. A guard that a comment can satisfy is not a guard.
  const reached = new Function(
    grab(/const isaReachedDepth = \(requestedDepth, attribution, enabled\) => \{[\s\S]*?\n\};\n/,
      'isaReachedDepth') + '\nreturn isaReachedDepth;')();
  const someAttribution = { status: 'attributed', diagnosis: 'x' };
  ok(reached('isa', someAttribution, true) === 'isa' &&
     reached('compiler', someAttribution, true) === 'compiler',
    'a depth whose analysis returned a diagnosis is recorded as reached');
  ok(reached('isa', { status: 'inconclusive', diagnosis: 'the ISA does not explain it' }, true) === 'isa',
    'an INCONCLUSIVE attribution still counts as reaching the depth -- a plateau the evidence '
    + 'cannot explain is a real finding, and an analyst forced to produce a mechanism instead of '
    + 'admitting that is an analyst inventing one');
  ok(reached('isa', null, true) === 'pattern_after_failed_escalation' &&
     reached('compiler', null, true) === 'pattern_after_failed_escalation',
    'but a requested depth with NO attribution behind it is neither `isa` nor `pattern`: the '
    + 'first says "we looked and it did not help", the second says "we never looked", and a '
    + 'failed escalation is a false negative one level above the one this layer catches');
  ok(reached('pattern', null, true) === 'pattern' && reached('isa', someAttribution, false) === 'pattern',
    'no escalation, and the layer off, both record plain pattern');
  ok(/evidence_depth: isaEffectiveDepth/.test(src) &&
     /const isaEffectiveDepth = isaReachedDepth\(isaDepthState\.depth, isaAttribution, ISA_ENABLED\);/.test(src),
    'and the round record actually stores that computed value rather than the requested depth');
  ok(/const isaReason = isaEvidenceReject\(r\.ver\);/.test(src) &&
     /if \(metricReason \|\| isaReason\) \{/.test(src),
    'the greedy admission path actually consults it, and does so WITHOUT joining the '
    + 'twin/policy chain above, whose exact text another check in this file pins');
  ok(/isa_evidence: ISA_EVIDENCE_SCHEMA/.test(src),
    'the receipt is in the verify schema, so an agent that returns it is not stripped');
  ok(/mechanism_claims: \{ type: 'array'/.test(src),
    'and the engineer can declare the claim the receipt is tested against');
}

console.log(failures ? `\nFAIL: ${failures} check(s) failed.` : '\nPASS: QD v2 archive routing and invariants hold.');
process.exit(failures ? 1 : 0);
