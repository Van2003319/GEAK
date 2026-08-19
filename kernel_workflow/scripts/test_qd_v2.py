#!/usr/bin/env python3
"""GPU-free integration tests for the unified geak-qd-v2 CLI."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("qd_v2.py")
VALID = {
    "compute_primitive": "native_mfma",
    "wave_schedule": "independent",
    "k_pipeline": "lds_single",
    "decomposition": "tile_grid",
    "output_path": "direct_store",
    "rasterization": "grouped_m",
    "plan_binding": "static",
}


# `prefill_m512_up` as it ships: 688 CTAs, 13824-byte single panel. Registers
# are not supplied here, so the binding bound is LDS alone: 65536/13824 = 4,
# hence 1216 slots and one round. The finding-(24) table wrote 3 CTAs/912 slots
# for this route -- that 3 is the occupancy the kernel's `kDoubleBuffer` gate
# is written for, not a bound the descriptor can derive, and `ctas_per_cu_cap`
# is deliberately NOT read by `residency_slots`. Both readings give rounds = 1
# here and both give rounds = 2 once the panel doubles, so the refusal below is
# insensitive to it; `round_slack` is not, which is logged as an open item.
M512_SHIPPED = {
    "tiles": 688, "slices": 1, "cu_count": 304, "cta_m": 128, "cta_n": 64,
    "waves_m": 2, "waves_n": 2, "stage_k": 32, "lds_bytes": 13824,
    "ctas_per_cu_cap": 3, "m": 512, "n": 11008, "k": 4096,
}


class QDV2CliTest(unittest.TestCase):
    def run_cli(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, str(SCRIPT), *args], text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)

    def test_hash_tree_emits_plain_source_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "kernel.hip").write_text("// candidate\n", encoding="utf-8")
            payload = json.loads(self.run_cli("hash-tree", tmp).stdout)
            self.assertEqual(64, len(payload["source_hash"]))

    def test_validate_descriptor_builds_context_cell(self):
        payload = json.loads(self.run_cli(
            "validate-descriptor", json.dumps(VALID), "--context", "decode_m8",
            "--known-context", "decode_m8", "--arch", "gfx942").stdout)
        self.assertTrue(payload["valid"])
        self.assertTrue(payload["cell"].startswith("decode_m8|native_mfma|"))

    def test_invalid_descriptor_fails_closed(self):
        invalid = {**VALID, "decomposition": "split_k"}
        run = self.run_cli("validate-descriptor", json.dumps(invalid),
                           "--arch", "gfx942", check=False)
        self.assertEqual(2, run.returncode)
        self.assertFalse(json.loads(run.stdout)["valid"])

    def test_adjacency_exposes_coupled_reduction_edges(self):
        start = {**VALID, "decomposition": "persistent_output"}
        payload = json.loads(self.run_cli("adjacency", json.dumps(start),
                                            "--arch", "gfx942").stdout)
        coupled = [n for n in payload["neighbors"] if len(n["axes"]) == 2]
        self.assertEqual(2, len(coupled))

    def test_robust_stats_uses_two_mad_bounds(self):
        payload = json.loads(self.run_cli(
            "robust-stats", json.dumps({"decode_m8": [1.0, 2.0, 3.0]})).stdout)
        row = payload["per_context"][0]
        self.assertEqual(2.0, row["median"])
        self.assertEqual(1e-9, row["lower"])
        self.assertEqual(4.0, row["upper"])

    def test_sol_card_uses_elapsed_over_sol_floor(self):
        payload = json.loads(self.run_cli(
            "sol-card", "--flops", "1e12", "--bytes", "1e9",
            "--elapsed-ms", "10", "--dtype", "bf16", "--arch", "gfx942").stdout)
        self.assertAlmostEqual(payload["sol_gap"], payload["elapsed_s"] / payload["sol_s"])
        self.assertAlmostEqual(payload["remaining_headroom"], 1.0 - 1.0 / payload["sol_gap"])

    @staticmethod
    def _quiet_loud_and_effect():
        """The current machine's quietest and loudest routes, and an effect
        size that lands between their floors.

        This pair used to be hard-coded as (prefill_m256_down, decode_m2_square)
        with a 2% effect, on machine O where their floors were 0.0097 and
        0.0416. Machine P inverted them -- prefill_m256_down is now the LOUDEST
        route at 0.0430 -- so 2% became unreadable on both and a correct table
        failed the test. The property under test is that readability is a
        property of the route, which needs any two routes far enough apart.

        Machine Q pushed it one step further: its table is PROVISIONAL and flat
        at the fail-closed default, so `min` and `max` return the SAME route and
        there is no separating effect at all. The callers below branch on that
        rather than pretend two routes differ.
        """
        import qd_robust_stats as QRS
        table = QRS.MEASURED_NOISE_FLOOR_BY_MACHINE[QRS.CURRENT_MACHINE]
        quiet = min(table, key=table.get)
        loud = max(table, key=table.get)
        return quiet, loud, (table[quiet] + table[loud]) / 2.0

    @staticmethod
    def _two_routes_and_floors():
        import qd_robust_stats as QRS
        table = QRS.MEASURED_NOISE_FLOOR_BY_MACHINE[QRS.CURRENT_MACHINE]
        ordered = sorted(table, key=table.get)
        return ordered[0], ordered[-1], table

    def test_an_effect_below_the_route_floor_is_refused_with_exit_three(self):
        # The same effect must come back readable on one route and not the
        # other in a single call -- that is the whole point of the subcommand.
        quiet, loud, effect = self._quiet_loud_and_effect()
        if quiet == loud:
            # Flat provisional epoch: no effect separates two routes, because
            # every route carries the same fail-closed floor. The exit-3 path
            # and the per-context reporting are still checked -- with an effect
            # under the common floor, which must be refused on BOTH routes.
            a, b, table = self._two_routes_and_floors()
            self.assertNotEqual(a, b)
            below = table[a] / 2.0
            run = self.run_cli("noise-floor", "--effect", repr(below),
                               "--context", a, "--context", b, check=False)
            self.assertEqual(3, run.returncode)
            out = json.loads(run.stdout)
            self.assertEqual(sorted([a, b]), sorted(out["unreadable_on"]))
            self.assertEqual({a: False, b: False},
                             {r["context"]: r["readable"] for r in out["per_context"]})
            return
        run = self.run_cli("noise-floor", "--effect", repr(effect),
                           "--context", loud, "--context", quiet, check=False)
        self.assertEqual(3, run.returncode)
        out = json.loads(run.stdout)
        self.assertEqual([loud], out["unreadable_on"])
        readable = {r["context"]: r["readable"] for r in out["per_context"]}
        self.assertEqual({loud: False, quiet: True}, readable)

    def test_a_readable_effect_exits_zero(self):
        # An effect comfortably above the quietest route's own floor is
        # readable there whatever the epoch, flat or measured.
        quiet, _loud, table = self._two_routes_and_floors()
        run = self.run_cli("noise-floor", "--effect", repr(table[quiet] * 1.5),
                           "--context", quiet, check=False)
        self.assertEqual(0, run.returncode)
        self.assertEqual([], json.loads(run.stdout)["unreadable_on"])

    def test_an_unmeasured_route_gets_the_widest_floor_not_the_mean_or_zero(self):
        out = json.loads(self.run_cli("noise-floor", "--context", "no_such_route").stdout)
        row = out["per_context"][0]
        self.assertFalse(row["measured"])
        self.assertEqual(out["default_noise_floor"], row["noise_floor"])
        # Fail-closed means the default is the MAXIMUM measured floor -- and
        # since (26) went machine-keyed, the maximum over every machine, not just
        # the current one. A mean would silently admit noise on the loud routes;
        # so, more subtly, would this epoch's own maximum. An unmeasured route
        # belongs to no epoch, so it has no claim on this epoch's spread, and on
        # machine N that spread happens to be the narrower one (0.0378 against
        # machine L's 0.072).
        import qd_robust_stats as QRS
        widest_anywhere = max(v for table in QRS.MEASURED_NOISE_FLOOR_BY_MACHINE.values()
                              for v in table.values())
        self.assertEqual(widest_anywhere, row["noise_floor"])
        widest_here = max(json.loads(self.run_cli("noise-floor").stdout)["per_context"],
                          key=lambda r: r["noise_floor"])["noise_floor"]
        self.assertGreaterEqual(row["noise_floor"], widest_here)

    def test_the_floors_say_which_machine_they_came_from(self):
        # The table is epoch-specific and the JSON is the only thing a downstream
        # reader sees. Without this field the same eleven route names carry
        # different numbers on different boxes with nothing to distinguish them.
        import qd_robust_stats as QRS
        out = json.loads(self.run_cli("noise-floor").stdout)
        self.assertEqual(QRS.CURRENT_MACHINE, out["machine"])
        self.assertEqual({r["context"]: r["noise_floor"] for r in out["per_context"]},
                         dict(QRS.MEASURED_NOISE_FLOOR_BY_MACHINE[out["machine"]]))

    def test_the_whole_table_comes_back_when_no_context_is_named(self):
        out = json.loads(self.run_cli("noise-floor").stdout)
        self.assertEqual(11, len(out["per_context"]))
        self.assertTrue(all(r["measured"] for r in out["per_context"]))
        self.assertTrue(all(r["readable"] is None for r in out["per_context"]))

    def test_the_cli_can_build_a_gfx942_card_which_is_the_arch_we_measure_on(self):
        # Every number in the ledger was taken on gfx942, and until (28) this
        # subcommand could not produce a card for it at all. The bracket is
        # asserted because it is the only field that proves the ceiling came
        # from the footprint table rather than a scalar that happens to match.
        payload = json.loads(self.run_cli(
            "sol-card", "--flops", "1e12", "--bytes", "1e8", "--elapsed-ms", "10",
            "--dtype", "bf16", "--arch", "gfx942",
            "--footprint-bytes", str(128 << 20)).stdout)
        self.assertEqual("gfx942", payload["arch"])
        self.assertEqual("footprint_table", payload["bandwidth_ceiling_basis"])
        self.assertEqual([86 << 20, 128 << 20], payload["bandwidth_ceiling_bracket"])
        self.assertEqual(2.68e12, payload["peak_bandwidth_bytes_s"])

    def test_the_cli_refuses_an_arch_with_no_measured_card(self):
        run = self.run_cli("sol-card", "--flops", "1e12", "--bytes", "1e8",
                           "--elapsed-ms", "10", "--arch", "gfx1100", check=False)
        self.assertEqual(2, run.returncode)
        self.assertEqual("", run.stdout)

    def test_route_priority_ranks_by_slack_and_flags_the_closed_route(self):
        out = json.loads(self.run_cli("route-priority").stdout)
        self.assertEqual(11, len(out["per_context"]))
        self.assertEqual("prefill_m1024_down", out["richest"])
        # Nothing closes here, and that is finding (92): no `--elapsed-ms` was
        # supplied, so every row is scored against the SHIPPED kernel's latency
        # and no row is entitled to a verdict about the candidate. The two
        # routes that already sit on the roofline (sol_gap 1.03 and 1.01) show
        # up as conditional -- measure them first, do not skip them.
        self.assertEqual([], out["closed"])
        self.assertEqual(11, len(out["needs_fresh_elapsed"]))
        # WHICH rows read conditionally closed is machine-keyed: those two on
        # O, none at all on P, where both floors narrowed enough for their
        # remaining headroom to clear them (decode_m2_square 0.0416 -> 0.0224,
        # decode_m16_square 0.0588 -> 0.0066). What must hold on every machine
        # is that the conditional list is exactly the rows whose own
        # conditional verdict says `closed` -- an independently computed second
        # list would be free to drift from the rows it summarises.
        self.assertEqual(
            sorted(r["context"] for r in out["per_context"]
                   if r["verdict_if_elapsed_confirmed"] == "closed"),
            sorted(out["conditionally_closed"]))
        # Provenance must ride along: the ranking is built on a snapshot of the
        # ship point, and a reader who cannot tell which run it came from
        # cannot tell whether it is stale.
        self.assertIn("1670", out["elapsed_provenance"])
        # Finding (35): every route now has counters, so nothing is ranked on
        # the compulsory lower bound. The field stays in the payload -- a route
        # added to the shape table before it is profiled must still announce
        # itself rather than be compared against measured rows in silence.
        self.assertEqual([], out["compulsory_traffic_routes"])

    def test_measured_traffic_can_be_supplied_per_route(self):
        out = json.loads(self.run_cli(
            "route-priority", "--context", "prefill_m512_up",
            "--traffic-bytes", json.dumps({"prefill_m512_up": 250e6})).stdout)
        row = out["per_context"][0]
        self.assertEqual("measured", row["traffic_basis"])
        self.assertEqual([], out["compulsory_traffic_routes"])
        self.assertGreater(row["traffic_amplification"], 2.0)

    def test_enough_measured_traffic_closes_a_route_that_looked_open(self):
        # The correct consequence, not a bug: if the bus is already moving
        # nearly as much as the elapsed time allows, the route's apparent slack
        # was an artifact of the compulsory assumption and there is nothing to
        # win. `prefill_m512_up` reads `open` on 400 MB and `closed` on 564.
        #
        # That band is narrow, and the narrowness is finding (35) itself: under
        # traffic-indexing the ceiling rises with the traffic, so extra bytes
        # buy back part of the time they cost. Traffic alone can only just
        # close a route -- one that arrives at sol_gap 1.0 -- and past that
        # point the claim is not "closed" but "impossible" (next test).
        #
        # The elapsed is passed explicitly even though it equals the snapshot:
        # after (92) a defaulted elapsed yields no verdict at all, so a test of
        # the TRAFFIC mechanism has to supply one or it measures the default
        # instead. Same number, honest provenance.
        elapsed = json.dumps({"prefill_m512_up": 0.16476})
        run = self.run_cli("route-priority", "--context", "prefill_m512_up",
                           "--elapsed-ms", elapsed,
                           "--traffic-bytes", json.dumps({"prefill_m512_up": 400e6}),
                           check=False)
        self.assertEqual(0, run.returncode)
        self.assertEqual([], json.loads(run.stdout)["closed"])
        run = self.run_cli("route-priority", "--context", "prefill_m512_up",
                           "--elapsed-ms", elapsed,
                           "--traffic-bytes", json.dumps({"prefill_m512_up": 564e6}),
                           check=False)
        self.assertEqual(3, run.returncode)
        self.assertEqual(["prefill_m512_up"], json.loads(run.stdout)["closed"])

    def test_traffic_that_would_beat_the_hardware_is_refused_not_ranked(self):
        # Above the closing band the model says the kernel moved 700 MB faster
        # than the bus can move 700 MB. Reporting that as `closed` would hide a
        # broken input in the one bucket nobody re-reads.
        run = self.run_cli("route-priority", "--context", "prefill_m512_up",
                           "--traffic-bytes", json.dumps({"prefill_m512_up": 700e6}),
                           check=False)
        self.assertEqual(2, run.returncode)
        self.assertEqual("", run.stdout)
        self.assertIn("faster than", run.stderr)

    def test_traffic_below_the_compulsory_minimum_fails_closed(self):
        run = self.run_cli("route-priority", "--context", "prefill_m512_up",
                           "--traffic-bytes", json.dumps({"prefill_m512_up": 1.0}),
                           check=False)
        self.assertEqual(2, run.returncode)
        self.assertEqual("", run.stdout)

    @staticmethod
    def _closing_elapsed(context):
        """An elapsed for `context` that leaves it provably closed anywhere.

        These tests used to pass the shipped latency (0.02456 for
        decode_m2_square), which closed the route only because machine O's
        floor there was 0.0416. On P it is 0.0224 and the same input reads
        `marginal`, so the tests failed on a correct table. An elapsed a hair
        above the SOL floor leaves headroom ~1e-4, under every floor in every
        table, so the closure is a fact about the arithmetic rather than about
        the box the suite happens to be running on.
        """
        import qd_route_priority as priority
        return priority.route_priority(context)["sol_ms"] * 1.0001

    def test_a_direction_aimed_only_at_closed_routes_exits_three(self):
        elapsed = self._closing_elapsed("decode_m2_square")
        run = self.run_cli("route-priority", "--context", "decode_m2_square",
                           "--elapsed-ms", json.dumps({"decode_m2_square": elapsed}),
                           check=False)
        self.assertEqual(3, run.returncode)
        self.assertEqual(["decode_m2_square"], json.loads(run.stdout)["closed"])

    def test_a_direction_aimed_at_an_unmeasured_route_is_not_refused(self):
        # Finding (92). The same route, the same shape, the same table entry --
        # but nobody measured this kernel on it, so the CLI may not refuse the
        # direction. Refusing here is what made the mistake self-sealing: the
        # route is never proposed, so it never gets the measurement that would
        # settle whether it is really closed.
        run = self.run_cli("route-priority", "--context", "decode_m2_square",
                           check=False)
        self.assertEqual(0, run.returncode)
        out = json.loads(run.stdout)
        self.assertEqual([], out["closed"])
        self.assertEqual(["decode_m2_square"], out["needs_fresh_elapsed"])
        # Whether the CONDITIONAL reading is a closure is machine-keyed (it was
        # on L/N/O, it is not on P), but the (92) invariant above is not: a
        # defaulted elapsed may never produce a refusal whichever way the
        # conditional reading falls. The list must agree with the row's own
        # conditional verdict and with nothing else.
        row = out["per_context"][0]
        self.assertEqual(["decode_m2_square"] if
                         row["verdict_if_elapsed_confirmed"] == "closed" else [],
                         out["conditionally_closed"])

    def test_a_mixed_list_is_an_answer_not_a_refusal(self):
        # The planner is told to drop the closed entries and keep the rest, so
        # exit 0 here is load-bearing: exit 3 would throw away a live target.
        run = self.run_cli("route-priority", "--context", "decode_m2_square",
                           "--context", "prefill_m256_down",
                           "--elapsed-ms", json.dumps(
                               {"decode_m2_square": self._closing_elapsed("decode_m2_square"),
                                "prefill_m256_down": 0.11816}),
                           check=False)
        self.assertEqual(0, run.returncode)
        self.assertEqual(["decode_m2_square"], json.loads(run.stdout)["closed"])

    def test_fresh_latencies_override_the_recorded_ship_point(self):
        out = json.loads(self.run_cli(
            "route-priority", "--context", "prefill_m256_down",
            "--elapsed-ms", json.dumps({"prefill_m256_down": 0.5})).stdout)
        row = out["per_context"][0]
        self.assertEqual(0.5, row["elapsed_ms"])
        self.assertEqual("caller-supplied", row["elapsed_provenance"])

    def test_an_unknown_route_fails_closed_rather_than_being_ranked(self):
        run = self.run_cli("route-priority", "--context", "prefill_m4096", check=False)
        self.assertEqual(2, run.returncode)
        self.assertEqual("", run.stdout)

    def verdict(self, current, candidate):
        run = self.run_cli("mutation-verdict", "--current", json.dumps(current),
                           "--candidate", json.dumps(candidate), check=False)
        return run, (json.loads(run.stdout) if run.stdout else None)

    def test_the_v107_double_buffer_is_refused_on_residency_and_nothing_else(self):
        # Runs 1660-1663: -43.4%, the largest single-arm loss in the ledger.
        # The CLI is the only way the search can reach that measurement, so
        # the exit code matters as much as the payload -- a caller that only
        # checks `returncode` must still be stopped.
        run, out = self.verdict(M512_SHIPPED, {**M512_SHIPPED, "lds_bytes": 27648})
        self.assertEqual(3, run.returncode)
        self.assertFalse(out["allow"])
        self.assertEqual(1, len(out["refusals"]))
        self.assertIn("residency_mutation_verdict", out["refusals"][0])
        self.assertEqual(1, out["current"]["rounds"])
        self.assertEqual(2, out["candidate"]["rounds"])

    def test_the_grown_m512_tile_is_refused_on_the_tile_axis(self):
        # Finding (21b): -8.8%, and it never crossed a round boundary, so this
        # must come out of a different rule than the one above.
        run, out = self.verdict(M512_SHIPPED, {**M512_SHIPPED, "tiles": 344,
                                               "cta_n": 128, "lds_bytes": 18432})
        self.assertEqual(3, run.returncode)
        self.assertEqual(1, out["candidate"]["rounds"])
        self.assertTrue(any("tile_mutation_verdict" in r for r in out["refusals"]))

    def test_an_unrefuted_mutation_is_allowed_with_exit_zero(self):
        # Guard against a gate that refuses everything, which is the failure
        # mode that gets a gate switched off.
        run, out = self.verdict(M512_SHIPPED, {**M512_SHIPPED, "waves_m": 4,
                                               "waves_n": 2})
        self.assertEqual(0, run.returncode)
        self.assertTrue(out["allow"])
        self.assertEqual([], out["refusals"])

    def test_a_no_op_proposal_is_refused_rather_than_costing_a_build(self):
        run, out = self.verdict(M512_SHIPPED, dict(M512_SHIPPED))
        self.assertEqual(3, run.returncode)
        self.assertIn("no change", out["refusals"][0])

    def test_a_misspelled_field_fails_closed_instead_of_allowing(self):
        # `lds_byte` would leave residency undetermined, every rule would
        # abstain, and the mutation would read as allowed. Exit 2, not 0.
        run, out = self.verdict(M512_SHIPPED, {**M512_SHIPPED, "lds_byte": 27648})
        self.assertEqual(2, run.returncode)
        self.assertIsNone(out)
        self.assertIn("lds_byte", run.stderr)

    def test_a_non_integer_field_is_rejected_rather_than_coerced(self):
        run, _ = self.verdict(M512_SHIPPED, {**M512_SHIPPED, "lds_bytes": 27648.0})
        self.assertEqual(2, run.returncode)

    def test_unknown_command_is_rejected(self):
        run = self.run_cli("rank-parents", check=False)
        self.assertNotEqual(0, run.returncode)
        self.assertEqual("", run.stdout)


class GateIsReachableFromTheLaneTest(unittest.TestCase):
    """Finding (25) was that the gate existed and nothing called it.

    A green unit test on `mutation_verdict` says nothing about whether any
    agent will ever run it, which is exactly how the refusals sat unused for
    two stages. These tests assert the wiring itself: the lane hands the helper
    to the planning phase, and the planning prompt names the subcommand. They
    are the cheapest thing that would have caught the original gap.
    """

    WORKFLOW = Path(__file__).resolve().parents[1]

    def test_the_planner_prompt_actually_names_the_subcommand(self):
        prompt = (self.WORKFLOW / "roles" / "tech_lead.md").read_text(encoding="utf-8")
        self.assertIn("mutation-verdict", prompt,
                      "tech_lead can only run the gate if its prompt names it")
        self.assertIn("$QD_EVIDENCE_HELPER", prompt)
        # A gate the planner may skip on a hunch is not a gate.
        self.assertIn("exit 3", prompt)

    def test_the_lane_hands_the_helper_to_the_planning_phase(self):
        lane = (self.WORKFLOW / "kernel_lane.js").read_text(encoding="utf-8")
        # `qdCommonInputs` is spread into `planningInputs`, which is what both
        # `select_qd_parents` and `plan_qd_mutations` receive. If either link
        # is renamed, the prompt's `$QD_EVIDENCE_HELPER` silently expands to
        # nothing and the gate is unreachable again -- with no error anywhere.
        self.assertIn("QD_EVIDENCE_HELPER", lane)
        self.assertIn("...qdCommonInputs", lane)
        self.assertIn("planningInputs", lane)

    def test_the_verifier_prompt_carries_the_four_measurement_rules(self):
        # Audited in (31): none of these appeared in ANY role prompt, so the
        # only place block rotation and `--case` isolation were written down
        # was the ledger -- which no agent reads. They are the rules that have
        # cost this project the most measurement time.
        prompt = (self.WORKFLOW / "roles" / "verify_engineer.md").read_text(encoding="utf-8")
        for rule in ("A B B A", "candidate_ms", "--case", "opposite directions"):
            with self.subTest(rule=rule):
                self.assertIn(rule, prompt)
        # And the floor is offered where a delta is read, not only where one is
        # admitted -- the early-vs-late distinction from (30).
        self.assertIn("noise-floor", prompt)

    def test_the_planner_prompt_names_the_noise_floor_gate_too(self):
        # The floor was enforced at admission for a full stage while nothing at
        # planning time could see it -- so the search could still spend a build
        # on an idea that was unmeasurable on its own target route, and only
        # find out after paying for it.
        prompt = (self.WORKFLOW / "roles" / "tech_lead.md").read_text(encoding="utf-8")
        self.assertIn("noise-floor", prompt)
        self.assertIn("--effect", prompt)
        self.assertIn("unreadable_on", prompt)

    def test_the_profiler_prompt_passes_the_footprint_the_gfx942_ceiling_needs(self):
        # Same class of gap as (25): the flag can exist and be reachable and
        # still never be passed, in which case every gfx942 card silently
        # resolves its ceiling off traffic instead of working set.
        prompt = (self.WORKFLOW / "roles" / "profile_engineer.md").read_text(encoding="utf-8")
        self.assertIn("--footprint-bytes", prompt)
        self.assertIn("bandwidth_ceiling_confidence", prompt)
        self.assertIn("bandwidth_ceiling_extrapolated", prompt)

    def test_the_planner_prompt_names_the_route_priority_gate(self):
        # Finding (33) is a redirection of the search's *aim*, so unlike the
        # other gates it is worthless unless it runs BEFORE targets are chosen.
        # A gate nothing calls is finding (25); a gate called too late is (30).
        prompt = (self.WORKFLOW / "roles" / "tech_lead.md").read_text(encoding="utf-8")
        self.assertIn("route-priority", prompt)
        self.assertIn("slack_to_floor", prompt)
        # And the reason, not just the command -- the planner has to know that
        # a bad speedup is not evidence of available headroom.
        self.assertIn("moving opponent", prompt)

    def test_a_refusal_exits_three_so_a_shell_caller_cannot_miss_it(self):
        # The prompt tells the planner to read the exit code, not the JSON.
        doubled = dict(M512_SHIPPED, lds_bytes=M512_SHIPPED["lds_bytes"] * 2)
        run = subprocess.run(
            [sys.executable, str(SCRIPT), "mutation-verdict",
             "--current", json.dumps(M512_SHIPPED),
             "--candidate", json.dumps(doubled)],
            capture_output=True, text=True)
        self.assertEqual(3, run.returncode)
        self.assertFalse(json.loads(run.stdout)["allow"])


class ArchIsNeverAssumedTest(unittest.TestCase):
    """The arch must be stated, on every subcommand whose answer depends on it.

    These tests exist because the default used to be `gfx90a` on a fleet that
    has only ever been gfx942, and nothing in the output said so.
    """

    def run_cli(self, *args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, str(SCRIPT), *args], text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)

    def test_every_arch_sensitive_subcommand_refuses_to_assume_one(self):
        invocations = {
            "validate-descriptor": ("validate-descriptor", json.dumps(VALID)),
            "adjacency": ("adjacency", json.dumps(VALID)),
            "sol-card": ("sol-card", "--flops", "1e12", "--bytes", "1e9",
                         "--elapsed-ms", "10"),
        }
        for name, argv in invocations.items():
            with self.subTest(subcommand=name):
                run = self.run_cli(*argv)
                self.assertEqual(2, run.returncode)
                self.assertEqual("", run.stdout, "a refusal must not also emit a receipt")
                self.assertIn("--arch", run.stderr)

    def test_the_two_arches_disagree_about_the_xcd_remap_mechanism(self):
        # The mechanism this whole descriptor axis exists to name is legal on
        # one arch and illegal on the other, so an unstated arch silently
        # deletes it from the search space.
        remapped = {**VALID, "rasterization": "xcd_remapped_grouped"}
        ok = self.run_cli("validate-descriptor", json.dumps(remapped), "--arch", "gfx942")
        self.assertEqual(0, ok.returncode)
        self.assertTrue(json.loads(ok.stdout)["valid"])

        no = self.run_cli("validate-descriptor", json.dumps(remapped), "--arch", "gfx90a")
        self.assertEqual(2, no.returncode)
        self.assertEqual("rule:xcd_remap_requires_multi_die",
                         json.loads(no.stdout)["reason"])

    def test_the_two_arches_disagree_about_remaining_headroom(self):
        # Same measurement, both arches. If these ever converge, the reason
        # this flag is required has gone away and this test should be the
        # thing that says so.
        cards = {}
        for arch in ("gfx90a", "gfx942"):
            run = self.run_cli("sol-card", "--flops", "1e12", "--bytes", "1e9",
                               "--elapsed-ms", "10", "--dtype", "bf16", "--arch", arch,
                               check=True)
            cards[arch] = json.loads(run.stdout)
        self.assertGreater(cards["gfx942"]["remaining_headroom"],
                           cards["gfx90a"]["remaining_headroom"] + 0.2)

    def test_every_emitted_card_has_passed_the_modules_own_validator(self):
        # (70)/(55). `validate_sol_card` was thorough, unit-tested, and called by
        # nothing outside its own tests -- so the one v2 requirement it enforces
        # (the ceiling provenance fields) was enforced nowhere on the path that
        # actually hands a card to an agent. Checked by running the CLI and
        # re-validating its output, not by grepping for the call.
        import qd_sol_card as sol
        for arch, extra in (("gfx90a", []),
                            ("gfx942", ["--footprint-bytes", str(64 << 20)])):
            run = self.run_cli("sol-card", "--flops", "1e12", "--bytes", "1e9",
                               "--elapsed-ms", "10", "--dtype", "bf16", "--arch", arch,
                               *extra, check=True)
            card = json.loads(run.stdout)
            self.assertEqual([], sol.validate_sol_card(card), f"{arch} card")
            for key in ("bandwidth_ceiling_basis", "bandwidth_ceiling_confidence",
                        "bandwidth_ceiling_extrapolated", "footprint_bytes",
                        "bandwidth_ceiling_bracket"):
                self.assertIn(key, card, f"{arch} card is missing {key}, which the lane now "
                                         "requires as its `ceiling` block")

    def test_the_helper_refuses_rather_than_emitting_a_card_it_cannot_validate(self):
        # The failure mode this guards is a future build path that adds a field
        # the validator rejects: the run must stop, not print the card anyway.
        source = (Path(__file__).resolve().parent / "qd_v2.py").read_text(encoding="utf-8")
        i = source.index("problems = sol.validate_sol_card(card)")
        self.assertLess(i, source.index("_emit(card)", i),
                        "the validator must run before the card is printed")

    def test_an_unsupported_arch_is_a_json_reason_not_a_usage_error(self):
        # `validate-descriptor` reports every other refusal as machine-readable
        # JSON, so this one must not be the exception: an agent parsing stdout
        # would see an empty string and no reason at all.
        run = self.run_cli("validate-descriptor", json.dumps(VALID), "--arch", "gfx1100")
        self.assertEqual(2, run.returncode)
        self.assertEqual("rule:unsupported_arch_or_dtype",
                         json.loads(run.stdout)["reason"])


ROLES = sorted((Path(__file__).resolve().parents[1] / "roles").glob("*.md"))
LANE_TEXT = (Path(__file__).resolve().parents[1] / "kernel_lane.js").read_text(encoding="utf-8")
PROMPT_TEXT = "\n".join(p.read_text(encoding="utf-8") for p in ROLES)


class SubcommandReachabilityTest(unittest.TestCase):
    """A gate nothing invokes is not a gate. Inventory every subcommand.

    `validate-descriptor` shipped for weeks reachable by nobody: no role
    prompt and no line of `kernel_lane.js` ever named it, while finding (44)
    -- an archive that stayed empty because descriptors were rejected over two
    axis names -- was exactly the question it answers. Nothing reported that,
    because "no caller" produces no output at all.
    """

    # Subcommands with no caller today, each with the reason it is still here.
    # Adding a name to this map is a deliberate, reviewable act; forgetting to
    # wire one is not.
    # Empty on purpose. Every subcommand the parser exposes is now invoked by a
    # role prompt or by lane code. Adding an entry here is allowed, but it is a
    # written admission that a gate exists which nothing can reach -- and a gate
    # nothing invokes produces no output, so nothing else will ever report it
    # (finding 55).
    KNOWN_UNREFERENCED: dict[str, str] = {}

    def all_subcommands(self) -> set[str]:
        import qd_v2
        actions = qd_v2._parser()._subparsers._group_actions  # noqa: SLF001
        return set(actions[0].choices)

    @staticmethod
    def invocations(name: str, text: str) -> list[int]:
        """Offsets where `name` is used AS A SUBCOMMAND, not as an English word.

        `evidence` is also a word these prompts use constantly, so a substring
        test reports the one genuinely unreachable subcommand as wired.
        """
        pattern = re.compile(r'(?:QD_EVIDENCE_HELPER"|qd_v2\.py)\s+' + re.escape(name) + r'\b')
        return [m.start() for m in pattern.finditer(text)]

    def test_every_subcommand_is_either_invoked_or_listed_as_unreferenced(self):
        for name in sorted(self.all_subcommands()):
            with self.subTest(subcommand=name):
                referenced = bool(self.invocations(name, PROMPT_TEXT)
                                  or self.invocations(name, LANE_TEXT))
                if name in self.KNOWN_UNREFERENCED:
                    continue
                self.assertTrue(
                    referenced,
                    f"{name!r} is invoked by no role prompt and no lane code. Either wire "
                    f"it or add it to KNOWN_UNREFERENCED with the reason.")

    def test_the_unreferenced_list_does_not_go_stale(self):
        # The opposite failure: a subcommand gets wired and the note claiming
        # it is unreachable stays, which is a false record of what the search
        # can do.
        for name, reason in self.KNOWN_UNREFERENCED.items():
            with self.subTest(subcommand=name):
                self.assertIn(name, self.all_subcommands(), f"{name} is not a subcommand")
                self.assertEqual(
                    [], self.invocations(name, PROMPT_TEXT) + self.invocations(name, LANE_TEXT),
                    f"{name} is now invoked; drop it from KNOWN_UNREFERENCED ({reason})")

    def test_the_planner_self_checks_its_descriptors(self):
        tech_lead = (Path(__file__).resolve().parents[1] / "roles" / "tech_lead.md").read_text(
            encoding="utf-8")
        self.assertIn("validate-descriptor", tech_lead)
        # The reason, not just the command: a planner that does not know the
        # `reason` field exists will read exit 2 as "try something else"
        # rather than "you misspelled an axis".
        self.assertIn("ineligible_reason", tech_lead)

    def test_no_prompt_invokes_an_arch_sensitive_subcommand_without_arch(self):
        # These calls now fail closed at argparse, so a prompt that omits the
        # flag costs the agent a round instead of producing a wrong ceiling.
        for name in ("sol-card", "adjacency", "validate-descriptor"):
            offsets = self.invocations(name, PROMPT_TEXT)
            self.assertTrue(offsets, f"{name} lost its prompt invocation")
            for offset in offsets:
                with self.subTest(subcommand=name, offset=offset):
                    # The flag routinely sits on a continuation line, so the
                    # window is the invocation and the shell block after it.
                    self.assertIn("--arch", PROMPT_TEXT[offset:offset + 400])


class InlineJsonArgumentTest(unittest.TestCase):
    """A descriptor longer than a filename must not be read as a filename.

    `Path(value).is_file()` raises OSError(ENAMETOOLONG) past ~255 bytes --
    pathlib swallows only ENOENT/ENOTDIR/EBADF/ELOOP -- and the handler
    reported that as "invalid JSON or JSON file". Short descriptors parsed and
    long ones did not, which reads as a quoting mistake by the caller.
    """

    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, str(SCRIPT), *args], text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_a_descriptor_longer_than_the_filename_limit_still_parses(self):
        padded = {**VALID, "_note": "x" * 400}
        blob = json.dumps(padded)
        self.assertGreater(len(blob), 255, "fixture must exceed NAME_MAX to be a regression")
        run = self.run_cli("validate-descriptor", blob, "--arch", "gfx942")
        self.assertNotIn("File name too long", run.stderr)
        # An unknown key is not an axis violation; the seven axes are all there.
        self.assertTrue(json.loads(run.stdout)["valid"])

    def test_the_file_form_still_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "descriptor.json")
            path.write_text(json.dumps(VALID), encoding="utf-8")
            run = self.run_cli("validate-descriptor", str(path), "--arch", "gfx942")
            self.assertEqual(0, run.returncode, run.stderr)
            self.assertTrue(json.loads(run.stdout)["valid"])

    def test_a_missing_file_is_reported_as_a_file_not_as_bad_json(self):
        run = self.run_cli("validate-descriptor", "/nonexistent/descriptor.json",
                           "--arch", "gfx942")
        self.assertEqual(2, run.returncode)
        self.assertIn("invalid JSON file", run.stderr)

    def test_malformed_inline_json_is_reported_as_inline(self):
        run = self.run_cli("validate-descriptor", '{"compute_primitive":', "--arch", "gfx942")
        self.assertEqual(2, run.returncode)
        self.assertIn("invalid inline JSON", run.stderr)


class EvidenceCliTest(unittest.TestCase):
    """`evidence` must separate the two reasons a claim comes back null.

    A bare null reads as "unproven" for both, but only one is actionable. A
    claim with no rule (an absence like rasterization:linear) is expected to be
    null forever; a claim WITH a rule that matched nothing means the source does
    not contain what the descriptor says it contains.
    """

    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, str(SCRIPT), *args], text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

    def evidence(self, source: str, *claims: str) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "kernel.hip").write_text(source, encoding="utf-8")
            args = ["evidence", tmp]
            for claim in claims:
                args += ["--claim", claim]
            run = self.run_cli(*args)
            self.assertEqual(0, run.returncode, run.stderr)
            return json.loads(run.stdout)

    def test_grounded_claim_is_quoted_and_neither_flagged(self):
        payload = self.evidence("acc = tl.dot(a, b, acc)\n", "compute_primitive:native_mfma")
        self.assertIn("tl.dot", payload["evidence"]["compute_primitive:native_mfma"])
        self.assertEqual([], payload["unsubstantiated"])
        self.assertEqual({}, payload["ungroundable"])

    def test_absence_claim_is_ungroundable_not_unsubstantiated(self):
        payload = self.evidence("int row0 = blockIdx.y * CTA_M;\n", "rasterization:linear")
        self.assertIsNone(payload["evidence"]["rasterization:linear"])
        self.assertIn("rasterization:linear", payload["ungroundable"])
        self.assertNotIn("rasterization:linear", payload["unsubstantiated"])

    def test_claim_with_a_rule_that_matches_nothing_is_unsubstantiated(self):
        # The mislabel signal: the descriptor says the kernel remaps across
        # XCDs, and there is no remap arithmetic anywhere in the source.
        payload = self.evidence("int row0 = blockIdx.y * CTA_M;\n",
                                "rasterization:xcd_remapped_grouped")
        self.assertEqual(["rasterization:xcd_remapped_grouped"], payload["unsubstantiated"])
        self.assertEqual({}, payload["ungroundable"])

    def test_a_mislabel_does_not_change_the_exit_code(self):
        # Deliberate: the verifier decides, because only the verifier can see
        # the disassembly that would overrule a text match (finding 53).
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "kernel.hip").write_text("int x = 1;\n", encoding="utf-8")
            run = self.run_cli("evidence", tmp,
                               "--claim", "rasterization:xcd_remapped_grouped")
            self.assertEqual(0, run.returncode)

    def test_the_two_lists_never_overlap_across_a_whole_descriptor(self):
        claims = [f"{axis}:{value}" for axis, value in VALID.items()]
        payload = self.evidence("acc = tl.dot(a, b, acc)\n", *claims)
        self.assertEqual(set(),
                         set(payload["ungroundable"]) & set(payload["unsubstantiated"]))
        for claim, quote in payload["evidence"].items():
            if quote is not None:
                with self.subTest(claim=claim):
                    self.assertNotIn(claim, payload["unsubstantiated"])
                    self.assertNotIn(claim, payload["ungroundable"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
