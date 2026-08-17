#!/usr/bin/env python3
"""The descriptor rules exist TWICE. These tests check the two copies agree.

`qd_descriptor_v2.py` and `kernel_lane.js` each carry a full implementation of
the geak-qd-v2 axis vocabulary, the legality rules, and the cell id. That is not
an accident -- the orchestrator must be able to re-check a planner's proposal
deterministically without shelling out per candidate, and the Python side is
what the agents call. But two copies of a rule drift, and when this one drifts
the symptom is silent: a refusal simply stops applying on one side, every test
of the rule stays green, and the archive fills with cells the other side thinks
are illegal.

`test_qd_archive.js` was supposed to cover this. **`node` is absent on machine
L**, so it has never run here and must keep being reported as uncovered. This
module is the part of that coverage recoverable without a JS runtime: the
vocabulary and axis order are array literals and can be parsed and compared
exactly, which is the strongest available check and also the thing most likely
to be edited on one side only.

The four conditional rules cannot be compared this way without reimplementing
the JS, which would just be a third copy to drift. They are checked lexically
for the presence of each guard inside `qdDescriptorValid`'s body -- weaker, and
labelled as such: it catches a deleted rule, not a subtly rewritten one.
"""
from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

import qd_descriptor_v2 as QD

LANE = Path(__file__).resolve().parents[1] / "kernel_lane.js"
SOURCE = LANE.read_text(encoding="utf-8")


def js_string_array(name: str) -> list[str]:
    """Parse `const <name> = ['a', 'b'];` out of the lane."""
    match = re.search(r"const\s+%s\s*=\s*(\[[^\]]*\])\s*;" % re.escape(name), SOURCE)
    if match is None:
        raise AssertionError(f"{name} is no longer a const array literal in kernel_lane.js; "
                             "this parser needs updating before the parity claim means anything")
    # Single-quoted JS string arrays are valid Python literals.
    return list(ast.literal_eval(match.group(1)))


def js_axis_order() -> dict[str, list[str]]:
    match = re.search(r"const\s+QD_AXIS_ORDER\s*=\s*\{(.*?)\n\};", SOURCE, re.S)
    if match is None:
        raise AssertionError("QD_AXIS_ORDER is no longer a flat object literal in kernel_lane.js")
    out: dict[str, list[str]] = {}
    for axis, values in re.findall(r"(\w+)\s*:\s*(\[[^\]]*\])", match.group(1)):
        out[axis] = list(ast.literal_eval(values))
    return out


def qd_descriptor_valid_body() -> str:
    """The lane's single copy of the legality rules.

    The rules live in `qdDescriptorReject`, which returns a reason token instead
    of a boolean; `qdDescriptorValid` is a one-line predicate over it. Read the
    reject function: it is where a rule can be deleted. Pointing this at the
    predicate would scan two lines and pass no matter what was removed.
    """
    match = re.search(r"const qdDescriptorReject = \(d\) => \{(.*?)\n\};", SOURCE, re.S)
    if match is None:
        raise AssertionError("qdDescriptorReject is no longer an arrow function in kernel_lane.js; "
                             "this lexical rule check needs updating before it means anything")
    predicate = re.search(r"const qdDescriptorValid = \(d\) => qdDescriptorReject\(d\) === null;", SOURCE)
    if predicate is None:
        raise AssertionError("qdDescriptorValid no longer delegates to qdDescriptorReject; the rules may "
                             "have been forked into a second copy, which is exactly what this module exists "
                             "to prevent")
    return match.group(1)


class AxisVocabularyParityTest(unittest.TestCase):
    def test_the_axis_order_is_identical_including_order(self):
        # Order is not cosmetic: adjacency steps to the immediate neighbour in
        # this sequence, so a reordered axis changes which mutations the search
        # can even reach, on one side only.
        js = js_axis_order()
        py = {axis: list(values) for axis, values in QD.AXIS_ORDER.items()}
        self.assertEqual(py, js)

    def test_every_axis_is_present_on_both_sides(self):
        self.assertEqual(sorted(QD.AXES), sorted(js_axis_order()))

    def test_the_supported_arches_match(self):
        self.assertEqual(list(QD.SUPPORTED_ARCHES), js_string_array("QD_SUPPORTED_ARCHES"))

    def test_the_multi_die_arches_match(self):
        # The one arch-conditional mechanism in the vocabulary. If these drift,
        # `xcd_remapped_grouped` is legal on one side and refused on the other
        # for the arch every measurement in this project was taken on.
        self.assertEqual(sorted(QD.MULTI_DIE_ARCHES), sorted(js_string_array("QD_MULTI_DIE_ARCHES")))

    def test_the_supported_dtypes_match(self):
        match = re.search(r"!\[([^\]]*)\]\.includes\(QD_DTYPE\)", SOURCE)
        self.assertIsNotNone(match, "the lane no longer gates QD_DTYPE against a literal list")
        js = [v.strip().strip("'\"") for v in match.group(1).split(",")]
        self.assertEqual(list(QD.SUPPORTED_DTYPES), js)


class LegalityRuleParityTest(unittest.TestCase):
    """LEXICAL, not behavioural -- see the module docstring.

    Each rule is identified by a token that only that rule uses. A deleted rule
    fails here; a rule whose condition is inverted does not. That limit is the
    reason `test_qd_archive.js` still matters and still cannot run.
    """

    def setUp(self):
        self.body = qd_descriptor_valid_body()

    def test_the_reduction_fixup_coupling_is_present(self):
        # Finding (61) split the single `reduction !== fixup` conjunct into the
        # two directions, so that the token tells the planner which side to fix.
        self.assertIn("reduction && !fixup", self.body)
        self.assertIn("fixup && !reduction", self.body)
        for value in ("split_k", "stream_k", "atomic_fixup", "workspace_fixup"):
            self.assertIn(value, self.body)

    def test_the_pingpong_family_cannot_be_valu(self):
        self.assertIn("symmetric_interleave", self.body)
        self.assertIn("symmetric_pingpong", self.body)
        self.assertIn("'valu'", self.body)
        # And the Python side agrees, which is the half this file can check
        # behaviourally rather than lexically.
        tuple_ = {"compute_primitive": "valu", "wave_schedule": "symmetric_pingpong",
                  "k_pipeline": "lds_single", "decomposition": "tile_grid",
                  "output_path": "direct_store", "rasterization": "grouped_m",
                  "plan_binding": "static"}
        self.assertFalse(QD.descriptor_valid(tuple_))
        self.assertTrue(QD.descriptor_valid({**tuple_, "compute_primitive": "native_mfma"}))

    def test_the_xcd_remap_is_gated_on_multi_die(self):
        self.assertIn("xcd_remapped_grouped", self.body)
        self.assertIn("QD_MULTI_DIE_ARCHES", self.body)
        remapped = {"compute_primitive": "native_mfma", "wave_schedule": "independent",
                    "k_pipeline": "lds_single", "decomposition": "tile_grid",
                    "output_path": "direct_store", "rasterization": "xcd_remapped_grouped",
                    "plan_binding": "static"}
        self.assertTrue(QD.descriptor_valid(remapped, arch="gfx942"))
        self.assertFalse(QD.descriptor_valid(remapped, arch="gfx90a"))

    def test_runtime_tuning_requires_something_to_tune(self):
        self.assertIn("runtime_tuned", self.body)
        static = {"compute_primitive": "native_mfma", "wave_schedule": "independent",
                  "k_pipeline": "lds_single", "decomposition": "tile_grid",
                  "output_path": "direct_store", "rasterization": "grouped_m",
                  "plan_binding": "runtime_tuned"}
        self.assertFalse(QD.descriptor_valid(static))
        self.assertTrue(QD.descriptor_valid({**static, "decomposition": "split_k",
                                              "output_path": "atomic_fixup"}))


class RefusalIsNamedTest(unittest.TestCase):
    """Every way a route can fail to produce a cell must say which way it was.

    This is finding (44)'s actual cost centre. The vocabulary bug -- three axes
    mandatory in the validator and unrequestable in the agent-facing schema --
    took weeks to find not because it was subtle but because its only symptom
    was an archive that never filled: `qdRouteCells` dropped every illegal
    route with a bare `continue`, and the bootstrap's error said "no
    classifiable route-context cells", which is equally true of a vocabulary
    bug, a mis-classifying verifier, and a genuinely unclassifiable kernel.

    So these assertions are not about message quality. A refusal that cannot
    name itself is indistinguishable from a search that ran and found nothing,
    and the two need opposite responses.
    """

    def test_no_legality_rule_refuses_anonymously(self):
        body = qd_descriptor_valid_body()
        # Every `return` out of the reject function is either a named token or
        # the single `return null` that means "legal". A bare `return false`
        # would be the old silent behaviour wearing the new signature.
        returns = re.findall(r"return\s+([^;]+);", body)
        self.assertTrue(returns, "qdDescriptorReject has no returns; the parser is wrong")
        for expr in returns:
            expr = expr.strip()
            if expr == "null":
                continue
            self.assertRegex(expr, r"^[`']?(axis|rule|descriptor):",
                             f"qdDescriptorReject returns {expr!r}, which names no axis or rule")

    def test_every_conditional_rule_has_its_own_token(self):
        body = qd_descriptor_valid_body()
        tokens = re.findall(r"return '(rule:\w+)'", body)
        for expected in ("rule:reduction_without_fixup", "rule:fixup_without_reduction",
                         "rule:pingpong_requires_matrix_core",
                         "rule:unsupported_arch_or_dtype", "rule:xcd_remap_requires_multi_die",
                         "rule:runtime_tuned_requires_reduction"):
            self.assertIn(expected, tokens,
                          f"{expected} is gone; either the rule was deleted or two rules now share "
                          "a token, and a shared token cannot tell you which one refused")
        self.assertEqual(len(tokens), len(set(tokens)), "two legality rules return the same token")

    def test_route_cells_can_report_why_a_route_produced_nothing(self):
        match = re.search(r"const qdRouteCells = \((.*?)\) => \{(.*?)\n\};", SOURCE, re.S)
        if match is None:
            raise AssertionError("qdRouteCells is no longer an arrow function in kernel_lane.js")
        params, body = match.group(1), match.group(2)
        self.assertIn("rejects", params,
                      "qdRouteCells no longer accepts a rejects sink, so its drops are silent again")
        # Each `continue` is a dropped route. Every one of them must carry a
        # note() on the same line or the one just above it -- the two shapes the
        # function uses. A bare `continue` is the exact construct finding (44)
        # hid behind for weeks.
        lines = body.splitlines()
        for i, line in enumerate(lines):
            if "continue;" not in line:
                continue
            window = line + (lines[i - 1] if i else "")
            self.assertIn("note(", window,
                          f"qdRouteCells drops a route without naming why: {line.strip()!r}")


class SummaryProjectionIsReadableTest(unittest.TestCase):
    """Item 2b bullet 4, as an invariant instead of an instance.

    The bullet reads: "Either name `recent_transitions` in the role prompts that
    are supposed to use it, or stop exposing it." That rule is not about
    `recent_transitions`. `qdSummary()` is the only archive window any agent
    gets, and a field in it that no prompt names is the *other* half of (44):
    not a value the archive failed to record, but a value it hands over to a
    reader who was never told the field exists. Such a field steers nothing and
    still costs prompt budget in every round.

    Closing the bullet for the one field it named would be (57) again -- the
    invariant applied where it came up rather than everywhere it holds. Checked
    across every top-level key instead, and it immediately found three more:
    `lineage` was exposed with a lane comment claiming tech_lead.md named it
    (grep: zero hits in any role file), and `qd_score` / `context_coverage` were
    reported to nobody while `stalls`, which counts how long each of the two has
    failed to move, was documented.

    Lives here rather than in the JS suite because the JS runner is sandboxed to
    the two lane sources and cannot open `roles/`.
    """

    # A key may be exempt, but only with a reason, and the test below fails if
    # an exemption stops matching a real key -- so a field that later becomes
    # agent-facing cannot keep its pass under a name someone recognises.
    EXEMPT = {
        "version": "the archive schema version; consumed by the loader's compatibility "
                   "check before any agent sees the summary, and a planner has no "
                   "decision that depends on it",
        "classifier_version": "same: the route classifier's version, compared by the "
                              "orchestrator to decide whether imported cells must be "
                              "re-classified. Named in director.md as provenance, not "
                              "as an input to a plan",
    }

    def keys(self) -> list[str]:
        match = re.search(r"const qdSummary = \(\) => \{.*?\n  return \{(.*?)\n  \};\n\};",
                          SOURCE, re.S)
        self.assertIsNotNone(match, "qdSummary's projection literal moved; this test reads it")
        # Scanned per character, not per line. Two of these keys hold an object
        # written inline (`stalls: { coverage: ..., global: ... }`), so a
        # line-granular depth counter returns to zero on the same line it opened
        # and reports the nested keys as top-level ones -- which would demand a
        # role prompt name `global`, a token that matches prose everywhere and
        # therefore passes without meaning anything.
        # Comments first: this projection is more comment than code, and prose
        # ending in a colon ("into a mechanism memory: the planner can now...")
        # is indistinguishable from a key to the scanner below.
        body = re.sub(r"//[^\n]*", "", match.group(1))
        depth, out = 0, []
        for mo in re.finditer(r"[{}\[\]()]|(?<![\w.'\"])([a-z_]+):(?!:)", body):
            if mo.group(1) is not None:
                if depth == 0:
                    out.append(mo.group(1))
            elif mo.group(0) in "{[(":
                depth += 1
            else:
                depth -= 1
        return sorted(set(out))

    def role_text(self) -> str:
        return "\n".join(p.read_text(encoding="utf-8")
                         for p in sorted((LANE.parent / "roles").glob("*.md")))

    def test_the_projection_is_read_at_all(self):
        """(55). If the key parser silently returned nothing, every assertion
        below would pass and read as coverage of a projection nobody checked."""
        keys = self.keys()
        self.assertGreaterEqual(len(keys), 12, keys)
        for expected in ("cells", "capsules", "lineage", "recent_transitions"):
            self.assertIn(expected, keys)

    def test_every_field_the_planner_receives_is_named_in_some_role_prompt(self):
        roles = self.role_text()
        for key in self.keys():
            if key in self.EXEMPT:
                continue
            with self.subTest(field=key):
                self.assertRegex(
                    roles, rf"\b{re.escape(key)}\b",
                    f"qdSummary hands every agent `{key}` and no role prompt names it. "
                    "Either tell a reader how to read it or stop paying prompt budget "
                    "to send it (item 2b, bullet 4).")

    # The nested projections are the same exposure one level down, and they are
    # where the volume is: `cells[*]` alone carries 22 fields into every prompt.
    # Checked separately from the top level only because the exemptions differ
    # in kind -- an unnamed top-level key is a whole view nobody was told about,
    # an unnamed nested key is usually a machine-side identifier riding along
    # inside a view the prompt does name.
    NESTED_EXEMPT = {
        "artifact": "the content-addressed workspace path; the orchestrator resolves it "
                    "and an agent that pasted it anywhere would be reaching around the "
                    "harness rather than reading the archive",
        "source_hash": "identity for the orchestrator's re-resolution, named in the "
                       "prompts as parent_source_hash, which is the form a planner "
                       "actually writes",
        "snapshot": "the raw per-case latency vector the robust interval was computed "
                    "from; kept so a later run can recompute rather than believe, and "
                    "read by qd_robust_stats.py, not by an agent",
        "visits": "the QD visit counter, an input to parent selection inside the lane; "
                  "a planner that read it would be second-guessing a selection the "
                  "orchestrator has already made and re-checks",
    }

    def nested_keys(self) -> dict[str, list[str]]:
        """Field names inside `cells[*]`, `challengers[*]` and `capsules[*]`."""
        out = {}
        for view in ("cells", "challengers", "capsules"):
            # Anchored on the inner object literal rather than on the view, so
            # the scan starts at the field level and the wrapper's own parens
            # (`fromEntries(`, `entries(`, `map(`) never enter the depth count.
            match = re.search(rf"\n    {view}: Object\.fromEntries\(Object\.entries\("
                              rf"[^\n]*?\.map\(\(\[k, \w\]\) => \[k, \{{(.*?)\n    \}}\]\)\),",
                              SOURCE, re.S)
            self.assertIsNotNone(match, f"the {view} projection moved; this test reads it")
            body = re.sub(r"//[^\n]*", "", match.group(1))
            depth, keys = 0, []
            for mo in re.finditer(r"[{}\[\]()]|(?<![\w.'\"])([a-z_]+):(?!:)", body):
                if mo.group(1) is not None:
                    if depth == 0:
                        keys.append(mo.group(1))
                elif mo.group(0) in "{[(":
                    depth += 1
                else:
                    depth -= 1
            out[view] = sorted(set(keys))
        return out

    def test_the_nested_projections_are_read_at_all(self):
        nested = self.nested_keys()
        self.assertGreaterEqual(len(nested["cells"]), 15, nested["cells"])
        for expected in ("elite_id", "strategy_capsule", "min_case", "parent_elite_ids"):
            self.assertIn(expected, nested["cells"])
        self.assertIn("remeasurement_required", nested["challengers"])

    def test_every_nested_field_is_named_in_some_role_prompt(self):
        roles = self.role_text()
        for view, keys in sorted(self.nested_keys().items()):
            for key in keys:
                if key in self.NESTED_EXEMPT or key in self.EXEMPT:
                    continue
                with self.subTest(view=view, field=key):
                    self.assertRegex(
                        roles, rf"\b{re.escape(key)}\b",
                        f"qdSummary hands every agent `{view}[*].{key}` and no role prompt "
                        "names it (item 2b, bullet 4, one level down).")

    def test_every_nested_exemption_still_matches_a_real_field(self):
        seen = {k for keys in self.nested_keys().values() for k in keys}
        for key, reason in sorted(self.NESTED_EXEMPT.items()):
            with self.subTest(field=key):
                self.assertIn(key, seen,
                              "this exemption matches no nested field; delete it")
                self.assertGreater(len(reason), 40,
                                   "an exemption without a stated reason is a silencer")

    def test_no_prompt_names_a_field_the_summary_does_not_carry(self):
        """(76)'s second failure mode, and the one it calls WORSE than the first:
        a prompt that asks for a field the input does not contain. An absent
        field does not error -- the agent writes confident text about it. The
        two tests above walk lane -> prompt; this one walks prompt -> lane, and
        without it the pair can be satisfied by a prompt describing an archive
        that does not exist.

        Only the dotted `QD_ARCHIVE.x` / `QD_ARCHIVE.cells[*].y` form is checked.
        A bare field name in prose is not a claim about the input's shape, and
        matching those would flag every English word that happens to be a key.
        """
        top = set(self.keys())
        nested = self.nested_keys()
        refs = set()
        for path in sorted((LANE.parent / "roles").glob("*.md")):
            for mo in re.finditer(r"QD_ARCHIVE\.([A-Za-z_]+)(?:\[\*\])?(?:\.([A-Za-z_]+))?",
                                  path.read_text(encoding="utf-8")):
                refs.add((path.name, mo.group(1), mo.group(2)))
        self.assertGreaterEqual(len(refs), 5, "no prompt references any archive field; "
                                              "this check would pass vacuously (55)")
        for role, head, tail in sorted(refs):
            with self.subTest(role=role, field=head if not tail else f"{head}[*].{tail}"):
                self.assertIn(head, top,
                              f"{role} asks the agent to read QD_ARCHIVE.{head}, and "
                              "qdSummary carries no such field")
                if tail is not None:
                    self.assertIn(tail, nested.get(head, []),
                                  f"{role} asks for QD_ARCHIVE.{head}[*].{tail}; the "
                                  f"{head} projection does not carry it")

    def test_the_challenger_flag_is_explained_as_a_comparability_warning(self):
        """`remeasurement_required` is a hardcoded `true`, so naming it is not
        enough: a prompt that reports it as a field tells a planner nothing. What
        it means is that a challenger's interval and an incumbent's were measured
        in different sessions and their difference is not evidence -- which is
        the same cross-session rule the epoch discipline enforces everywhere else."""
        prompt = (LANE.parent / "roles" / "tech_lead.md").read_text(encoding="utf-8")
        block = re.search(r"`challengers\[\*\]\.remeasurement_required`.*?(?=\n-|\n\n)",
                          prompt, re.S)
        self.assertIsNotNone(block, "tech_lead.md no longer explains the challenger flag")
        self.assertRegex(block.group(0), r"(not comparable|different session)")
        self.assertRegex(block.group(0), r"(not (a )?evidence|never a result|not .*verified)")

    def test_every_exemption_still_matches_a_real_field(self):
        keys = set(self.keys())
        for key, reason in sorted(self.EXEMPT.items()):
            with self.subTest(field=key):
                self.assertIn(key, keys,
                              "this exemption matches no field in qdSummary; delete it")
                self.assertGreater(len(reason), 40,
                                   "an exemption without a stated reason is a silencer")

    def test_lineage_is_documented_as_outliving_elite_replacement(self):
        """The specific thing that makes `lineage` worth its prompt budget. Naming
        the field is not enough: `cells[*].parent_elite_id` already gives the
        current occupant's parent, so a prompt that describes `lineage` as
        "where elites came from" tells a planner nothing it did not have."""
        prompt = (LANE.parent / "roles" / "tech_lead.md").read_text(encoding="utf-8")
        block = re.search(r"`QD_ARCHIVE\.lineage`.*?(?=\n\n)", prompt, re.S)
        self.assertIsNotNone(block, "tech_lead.md no longer explains lineage")
        self.assertIn("parent_elite_id", block.group(0),
                      "the contrast is the point: without it the planner cannot tell "
                      "which of the two views to reach for")
        self.assertRegex(block.group(0), r"(replaced|outlive)",
                         "what lineage adds is ancestry that survives elite replacement")


class ParentGateParityTest(unittest.TestCase):
    """(44) for the parent-provenance gate: lane, JS suite, and TechLead prompt.

    This gate decides whether a planned direction is aimed at a parent that
    still exists, in a cell that still holds it, with a SOL card measured on
    that exact build. Until `audit_pin_coverage.py` was pointed at it, the only
    thing watching it was a regex asserting its log message existed -- and the
    message named four possible causes without committing to any of them. The
    other two gates in the same map had already been extracted into named,
    executable functions; this one had not, which is (57) at its most ordinary:
    the invariant was applied where it came up rather than everywhere it holds.

    The three corners drift independently. A reason token added to the lane is
    invisible from the prompt, and a rule stated in the prompt that the lane
    does not enforce reads to the planner exactly like one it does.
    """

    def body(self):
        match = re.search(r"const qdParentReject = \(d, selections, solCards, cells\) => \{"
                          r"(.*?)\n\};\n", SOURCE, re.S)
        self.assertIsNotNone(match, "qdParentReject is gone or its signature changed; this whole "
                                    "class is now watching nothing")
        return match.group(1)

    def test_no_refusal_out_of_the_gate_is_anonymous(self):
        for expr in re.findall(r"return \{ reason: (.*?) \};", self.body(), re.S):
            with self.subTest(ret=expr[:60]):
                # Backtick or plain quote: one refusal has nothing to interpolate and
                # says so with a plain string. What matters is the token, not the quote.
                self.assertRegex(expr.strip(), r"^['`]parent:",
                                 "a refusal that does not name itself leaves the four causes "
                                 "lumped together, which is the state this gate was extracted from")

    def test_the_reason_tokens_are_distinct(self):
        tokens = re.findall(r"reason: ['`](parent:\w+)", self.body())
        for expected in ("parent:no_selection_for_direction", "parent:selection_carries_no_parent",
                         "parent:no_sol_card", "parent:source_hash_moved",
                         "parent:transition_target_is_the_parent_itself",
                         "parent:transition_illegal", "parent:crossover_sources_not_distinct"):
            self.assertIn(expected, tokens,
                          f"{expected} is gone; either the check was deleted or two checks now "
                          "share a token, and a shared token cannot say which one refused")
        self.assertEqual(len(tokens), len(set(tokens)),
                         "two refusals return the same token")

    def test_the_gate_is_actually_wired_into_the_direction_map(self):
        # (55). A gate whose result is computed and discarded is a comment. The
        # residency gate above it earned that check; so does this one.
        self.assertIn("qdParentReject(d, qdSelections, qdSolCards, qdArchive.cells)", SOURCE)
        self.assertRegex(SOURCE, r"if \(gate\.reason\) \{\s*\n\s*log\(")
        self.assertRegex(SOURCE, r"no canonical fallback -- \$\{gate\.reason\}")

    def test_every_refusal_is_executed_by_the_js_suite_not_grepped(self):
        """The corner `audit_pin_coverage.py` found empty.

        Matching each token in the suite source is weaker than proving the
        suite exercises it -- but the suite's own mutant corpus is what proves
        that, and this only has to catch a token added to the lane with no
        check written for it at all.
        """
        suite = (Path(__file__).with_name("test_qd_archive.js")).read_text(encoding="utf-8")
        self.assertIn("# the parent-provenance gate, executed", suite,
                      "the executed block is gone; the gate is back to being grepped")
        for token in sorted(set(re.findall(r"reason: ['`](parent:\w+)", self.body()))):
            with self.subTest(token=token):
                self.assertIn(token, suite,
                              f"the lane can refuse a direction with {token} and no check in the "
                              "JS suite has ever seen that happen")

    def test_the_planner_is_told_how_to_read_variant_spread(self):
        """(108)'s third corner (44). The lane now reports `variant_spread`
        beside `structural_coverage`; the JS suite checks the projection, but a
        field the planner receives and was never told how to read steers
        nothing, and that unreachability -- not the missing value -- was the
        expensive half of (44). Checked here because the JS runner is sandboxed
        to the two lane sources and cannot open `roles/`.
        """
        prompt = (LANE.parent / "roles" / "tech_lead.md").read_text(encoding="utf-8")
        for field in ("variant_spread", "distinct_variants", "top_variant_share",
                      "cells_filled", "cells_per_variant"):
            with self.subTest(field=field):
                self.assertIn(field, prompt,
                              f"the lane emits {field} and the planner is never told it exists")
        self.assertIn("structural_coverage", prompt,
                      "the contrast is the point: the prompt has to name the number "
                      "variant_spread is meant to be read against")
        self.assertRegex(
            prompt, r"(not a defect|nothing fails on it)",
            "the prompt must say concentration is not a gate -- it is legitimately 1.0 "
            "right after seed import, and a planner reading it as a fault would treat "
            "a correct archive as broken")

    def test_the_tech_lead_prompt_states_that_the_orchestrator_re_checks(self):
        prompt = (LANE.parent / "roles" / "tech_lead.md").read_text(encoding="utf-8")
        self.assertRegex(prompt, r"re-resolves its\s*\n?\s*parent, its cell and its SOL card "
                                 r"against the live archive")
        self.assertIn("no\ncanonical fallback", prompt,
                      "the prompt must say the direction is dropped rather than retargeted at the "
                      "canonical seed, or a planner reads a refusal as a retry")
        for phrase in ("no SOL card exists", "no longer matches the live", "distinct source artifacts"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, prompt)


class TransitionIsResolvedTest(unittest.TestCase):
    """The transition record must not ship hardcoded placeholders.

    `sol_gap_after`, `profile_regime_after` and the observed-effect field were
    all literal `null`/`'unknown'` in the source for the whole life of the
    archive. That is a quieter version of finding (48): the archive accumulated
    one row per mutation with no field saying what the mutation did, so nothing
    downstream could learn a mechanism from it, and nothing failed.

    `profile_regime_after` is *legitimately* unresolved -- a regime is a
    profiling verdict, not a function of a latency -- so it is excluded here
    rather than asserted on. The other two are derivable from data the lane
    already holds at that point, and this test says so out loud.
    """

    def setUp(self):
        match = re.search(r"const transition = \{(.*?)\n\s*\};", SOURCE, re.S)
        if match is None:
            raise AssertionError("the transition record is no longer an object literal in kernel_lane.js")
        self.body = match.group(1)

    def test_the_sol_gap_after_is_computed_not_placeheld(self):
        self.assertNotRegex(self.body, r"sol_gap_after:\s*null",
                            "sol_gap_after is hardcoded null again; the candidate's own verified "
                            "latency and the parent card's sol_ms are both in scope there")
        self.assertIn("sol_gap_after: afterGap", self.body)
        # And the denominator it divides by has to be the card's, not a re-derivation.
        self.assertRegex(SOURCE, r"afterMs\s*/\s*solMs")

    def test_the_observed_effect_is_derived_from_the_robust_interval(self):
        self.assertNotRegex(self.body, r"observed_effect:\s*'unknown'",
                            "observed_effect is hardcoded again")
        self.assertIn("observed_effect: effect", self.body)
        # The direction must use the same non-overlap rule as admission, or a
        # transition could read 'improved' for a candidate the cell refused.
        effect = re.search(r"const effect = (.*?);", SOURCE, re.S)
        self.assertIsNotNone(effect)
        self.assertIn("robust.lower > incumbent.robust.upper", effect.group(1))
        self.assertIn("robust.upper < incumbent.robust.lower", effect.group(1))
        self.assertIn("no_readable_change", effect.group(1))

    def test_the_resolved_gap_reaches_the_planner(self):
        # Storing it is half the job. Finding (44)'s expensive half was a value
        # the archive held and never showed the agent that needed it.
        self.assertIn("sol_resolved: e.sol_resolved || null", SOURCE,
                      "sol_resolved is stored on the elite but not exposed in qdSummary, so no "
                      "planner can ever read it")
        self.assertIn("sol_resolved: (afterGap == null || beforeCeilingReason) ? null : {", SOURCE)
        # It must not claim a regime it did not measure.
        resolved = re.search(
            r"sol_resolved: \(afterGap == null \|\| beforeCeilingReason\) \? null : \{(.*?)\n\s*\},",
            SOURCE, re.S)
        self.assertIsNotNone(resolved)
        self.assertIn("profile_regime: 'unknown'", resolved.group(1))
        self.assertIn("derived: true", resolved.group(1))
        prompts = "\n".join(p.read_text(encoding="utf-8")
                            for p in sorted((LANE.parent / "roles").glob("*.md")))
        self.assertIn("sol_resolved", prompts,
                      "no role prompt explains sol_resolved, so it is exposed and unusable")

    def test_the_declared_effect_is_recorded_next_to_the_observed_one(self):
        # Without the declaration the observation cannot be scored: "improved"
        # is not evidence that the *stated mechanism* is what improved it.
        self.assertIn("expected_effect: r.d.expected_effect", self.body)


class ResidencyGateReachabilityTest(unittest.TestCase):
    """The residency receipt must be demanded, offered, and explained together.

    This is the finding-(44) triangle applied to a new field. (44) happened
    because a validator demanded seven axes, the agent-facing schema offered
    five, and the prompt explained none of the gap -- each side individually
    defensible, the combination silently fatal. The rounds-law receipt has the
    same three sides, so it gets the same three-sided test: the orchestrator
    rejects without it, the schema asks for it, and a role prompt tells the
    planner to send it and what happens if it does not.

    Any one of the three going missing makes the gate either unreachable or
    unenforced, and both failure modes are quiet.
    """

    def test_the_orchestrator_rejects_a_direction_without_a_receipt(self):
        self.assertIn("const qdResidencyReject = (receipt) => {", SOURCE)
        self.assertIn("'residency:receipt_absent'", SOURCE)
        self.assertRegex(SOURCE, r"qdResidencyReject\(d\.residency_receipt\)",
                         "the gate function exists but nothing calls it on a direction")
        # And the call must drop the direction, not merely log it.
        call = re.search(r"const residencyReason = .*?\n(.*?)\n\s*\}", SOURCE, re.S)
        self.assertIsNotNone(call)
        self.assertIn("return null;", call.group(1))

    def test_the_schema_offers_the_field_the_orchestrator_demands(self):
        schema = re.search(r"residency_receipt: obj\(\{(.*?)\n\s*\}, \[(.*?)\]\)", SOURCE, re.S)
        self.assertIsNotNone(schema, "residency_receipt is not in the agent-facing direction schema, so "
                                     "the planner cannot send what the orchestrator requires -- this is "
                                     "exactly finding (44)")
        for field in ("allow", "current", "candidate"):
            self.assertIn(field, schema.group(1))
            self.assertIn(f"'{field}'", schema.group(2), f"{field} is offered but not required")
        # The arithmetic re-check needs all three numbers on both sides.
        for field in ("ctas", "residency_slots", "rounds"):
            self.assertIn(field, schema.group(1))

    def test_a_role_prompt_tells_the_planner_to_send_it(self):
        prompts = [p for p in sorted((LANE.parent / "roles").glob("*.md"))
                   if "residency_receipt" in p.read_text(encoding="utf-8")]
        self.assertTrue(prompts, "no role prompt mentions residency_receipt, so the field is mandatory "
                                 "and undocumented -- the planner will omit it and every direction will "
                                 "be dropped for a reason no prompt explains")
        text = "\n".join(p.read_text(encoding="utf-8") for p in prompts)
        # The consequence has to be stated, or an omission reads as optional.
        self.assertRegex(text, r"dropped by the orchestrator")

    def test_the_rounds_law_is_the_rule_being_enforced(self):
        gate = re.search(r"const qdResidencyReject = \(receipt\) => \{(.*?)\n\};", SOURCE, re.S)
        self.assertIsNotNone(gate)
        body = gate.group(1)
        self.assertIn("Math.ceil(ctas / slots)", body)
        self.assertIn("rounds_raised", body)
        # Every refusal names itself, same rule as qdDescriptorReject.
        for expr in re.findall(r"return\s+([^;]+);", body):
            expr = expr.strip()
            if expr == "null":
                continue
            self.assertRegex(expr, r"^[`']?residency:",
                             f"qdResidencyReject returns {expr!r}, which names no cause")


class CellIdParityTest(unittest.TestCase):
    def test_the_cell_id_joins_the_same_fields_in_the_same_order(self):
        # A reordered join produces a valid-looking cell id that indexes a
        # different cell, so warm-started archives would silently mis-key.
        match = re.search(r"const qdCellId = \(contextId, d\) => \{(.*?)\n\};", SOURCE, re.S)
        self.assertIsNotNone(match)
        fields = re.findall(r"d\.(\w+)", match.group(1))
        self.assertEqual(list(QD.AXES), fields)
        # And the Python side builds it from the same sequence.
        cell = QD.cell_id("decode_m8", {
            "compute_primitive": "native_mfma", "wave_schedule": "independent",
            "k_pipeline": "lds_single", "decomposition": "tile_grid",
            "output_path": "direct_store", "rasterization": "grouped_m",
            "plan_binding": "static"}, known_contexts=["decode_m8"])
        self.assertEqual("decode_m8|native_mfma|independent|lds_single|tile_grid|"
                          "direct_store|grouped_m|static", cell)


def js_object_literal_enums(const_name: str) -> dict[str, list[str]]:
    """Parse the axis->values map out of a `const <name> = { ... };` block.

    Handles both shapes the lane uses for agent-facing surfaces: a JSON-schema
    property (`axis: { type: 'string', enum: [...] }`) and a plain array
    (`axis: [...]`). Keys whose value is neither are skipped, so a descriptive
    string field like `context_id` does not read as an axis.
    """
    # Two declaration shapes in the lane: a top-level `const NAME = obj({...})`
    # and a nested property `NAME: {...}` inside the planning-inputs literal.
    match = re.search(r"(?:const\s+%s\s*=\s*(?:obj\()?|\b%s\s*:\s*)\{(.*?)\n\s*\}"
                      % (re.escape(const_name), re.escape(const_name)), SOURCE, re.S)
    if match is None:
        raise AssertionError(f"{const_name} is no longer an object literal in kernel_lane.js; "
                             "this parser needs updating before the parity claim means anything")
    body = match.group(1)
    out: dict[str, list[str]] = {}
    for axis, values in re.findall(r"(\w+)\s*:\s*\{[^{}]*?enum:\s*(\[[^\]]*\])[^{}]*?\}", body):
        out[axis] = list(ast.literal_eval(values))
    for axis, values in re.findall(r"(\w+)\s*:\s*(\[[^\]]*\])", body):
        # `enum` is the inner key of the schema shape already handled above, and
        # `items` is the evidence array's element type -- neither is an axis.
        if axis in ("enum", "items"):
            continue
        out.setdefault(axis, list(ast.literal_eval(values)))
    return out


def js_schema_required(const_name: str) -> list[str]:
    """Parse the trailing `}, ['a', 'b'])` required-list of an `obj(...)` call."""
    match = re.search(r"const\s+%s\s*=\s*obj\(\{.*?\n\}\s*,\s*(\[[^\]]*\])\s*\)\s*;"
                      % re.escape(const_name), SOURCE, re.S)
    if match is None:
        raise AssertionError(f"{const_name} is no longer an obj(...) call with a required list")
    return list(ast.literal_eval(match.group(1)))


def descriptor_examples(text: str) -> list[str]:
    """Every `"descriptor": { ... }` literal in a role prompt, braces balanced.

    A regex cannot do this: the examples are pretty-printed across lines and
    the value is an object, so `\\{[^}]*\\}` stops at the first inner brace.
    """
    out: list[str] = []
    for match in re.finditer(r'"descriptor"\s*:\s*\{', text):
        depth, i = 0, match.end() - 1
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    out.append(text[match.start():i + 1])
                    break
            i += 1
        else:  # pragma: no cover - an unbalanced prompt is a prompt bug
            raise AssertionError("unterminated descriptor example")
    return out


class AgentFacingVocabularyParityTest(unittest.TestCase):
    """Finding (44). The parity above guards the two INTERNAL copies of the
    vocabulary and never looked at the two the agent actually sees.

    `qdDescriptorValid` requires **every** key of `QD_VOCAB` to be present and
    in range -- `Object.entries(QD_VOCAB).every(([k, values]) => values.has(d[k]))`
    -- so any axis missing from the schema the agent fills makes the returned
    descriptor invalid, `qdRouteCells` drops the route with a bare `continue`,
    and the archive silently never fills. The failure has no error message and
    no failing test: it looks exactly like a search that found nothing.

    Two surfaces, both agent-facing, both previously unchecked:
      - `QD_DESCRIPTOR_SCHEMA` -- the JSON schema the response is validated
        against, so an axis absent here cannot even be returned;
      - `QD_DESCRIPTOR_AXES` -- the vocabulary handed to the planner, so an
        axis absent here is never known to exist.
    A value present in the validator and absent from both is simultaneously
    **mandatory and unrequestable**.
    """

    def test_the_response_schema_offers_every_axis_the_validator_demands(self):
        self.assertEqual({axis: list(v) for axis, v in QD.AXIS_ORDER.items()},
                         js_object_literal_enums("QD_DESCRIPTOR_SCHEMA"),
                         "an axis the validator requires but the schema omits makes every "
                         "obedient agent's descriptor invalid, silently")

    def test_the_response_schema_requires_every_axis(self):
        required = js_schema_required("QD_DESCRIPTOR_SCHEMA")
        self.assertEqual(sorted(QD.AXES), sorted(a for a in required if a in set(QD.AXES)))
        for axis in QD.AXES:
            self.assertIn(axis, required, f"{axis} is optional in the schema but mandatory "
                                          "in qdDescriptorValid")

    def test_the_planner_is_told_every_axis_and_every_value(self):
        told = js_object_literal_enums("QD_DESCRIPTOR_AXES")
        told.pop("context_id", None)
        self.assertEqual({axis: list(v) for axis, v in QD.AXIS_ORDER.items()}, told,
                         "the planner cannot propose a value it is never shown")

    def test_the_role_prompt_lists_every_axis_and_every_value(self):
        # Third copy, and the one with no schema behind it: engineer.md states
        # the vocabulary in prose. Finding (32) -- a rule an agent cannot read
        # is not a rule.
        prompt = (LANE.parent / "roles" / "engineer.md").read_text(encoding="utf-8")
        for axis, values in QD.AXIS_ORDER.items():
            expected = f"`{axis}`: `" + "|".join(values) + "`"
            self.assertIn(expected, prompt,
                          f"engineer.md does not state {axis} as {'|'.join(values)}")

    def test_every_worked_descriptor_example_in_every_role_prompt_is_complete(self):
        # Copies four, five and six. The prose list above is only in
        # engineer.md; director.md and verify_engineer.md instead carry a
        # worked JSON example, and an agent copying a five-field example
        # reproduces exactly the finding-(44) invalid descriptor. An example is
        # the most-copied surface in a prompt, so it is guarded as a rule.
        found = 0
        for prompt_path in sorted((LANE.parent / "roles").glob("*.md")):
            text = prompt_path.read_text(encoding="utf-8")
            for example in descriptor_examples(text):
                found += 1
                for axis, values in QD.AXIS_ORDER.items():
                    self.assertIn(f'"{axis}"', example,
                                  f"{prompt_path.name} shows a descriptor example missing "
                                  f"the {axis} axis; qdDescriptorValid requires it, so an "
                                  "agent copying this example is silently dropped")
                    quoted = re.search(r'"%s"\s*:\s*"([^"]*)"' % re.escape(axis), example)
                    self.assertIsNotNone(quoted, f"{prompt_path.name}: {axis} has no literal value")
                    # Two legitimate shapes: a concrete value, or -- as in
                    # engineer.md -- a `a|b|c` placeholder enumerating the whole
                    # axis. The placeholder must still enumerate it exactly, so
                    # a value added to the vocabulary and not to the template is
                    # caught here too.
                    shown = quoted.group(1)
                    self.assertIn(shown, list(values) + ["|".join(values)],
                                  f"{prompt_path.name} shows {axis}={shown}, which is "
                                  "neither a value of the closed vocabulary nor its "
                                  "full placeholder enumeration")
        self.assertGreaterEqual(found, 3, "the descriptor-example parser found fewer examples "
                                          "than the three known copies; it needs updating "
                                          "before this test means anything")


class NodeCoverageIsStillMissingTest(unittest.TestCase):
    def test_the_js_suite_exists_and_is_recorded_as_unrun(self):
        # Pin the honesty claim itself: this file must not be mistaken for
        # having replaced `test_qd_archive.js`. It covers the vocabulary, not
        # the archive admission logic, and it parses the JS rather than running
        # it.
        js_suite = LANE.with_name("scripts") / "test_qd_archive.js"
        self.assertTrue(js_suite.is_file())
        self.assertIn("LEXICAL", LegalityRuleParityTest.__doc__)


class ArchIsNeverDefaultedTest(unittest.TestCase):
    """Neither copy may invent an arch. LEXICAL on the JS side (no `node` here).

    The lane defaulted `qd_arch` to 'gfx90a' on a gfx942-only fleet, which set
    the SOL ceiling to an explicitly-unmeasured reference card and made
    `xcd_remapped_grouped` illegal for the whole run, with no line of output
    saying an arch had been assumed. Defaulting to 'gfx942' instead is the same
    bug aimed at the next box, so what is pinned here is the ABSENCE of a
    fallback, not the presence of a particular one.
    """

    def test_the_lane_does_not_fall_back_to_any_arch(self):
        match = re.search(r"const\s+QD_ARCH\s*=\s*([^;]+);", SOURCE)
        self.assertIsNotNone(match, "QD_ARCH is no longer a simple const in kernel_lane.js")
        expression = match.group(1)
        for arch in QD.SUPPORTED_ARCHES:
            self.assertNotIn(arch, expression,
                             f"QD_ARCH falls back to {arch!r}; an assumed arch must not be "
                             "reachable, whichever arch it is")

    def test_the_lane_refuses_to_start_a_qd_run_without_one(self):
        self.assertRegex(
            SOURCE, r"if\s*\(\s*QD_ENABLED\s*&&\s*!QD_ARCH\s*\)",
            "the lane no longer refuses an empty qd_arch, so a QD run can start "
            "with the arch unstated")

    def test_the_python_clis_require_it_too(self):
        # Same rule, the surface the agents actually reach. A default here
        # would put the assumption back without touching the lane.
        for name in ("qd_v2.py", "qd_sol_card.py", "qd_descriptor_v2.py"):
            with self.subTest(script=name):
                text = (Path(__file__).with_name(name)).read_text(encoding="utf-8")
                self.assertNotRegex(
                    text, r'add_argument\(\s*"--arch",\s*default=',
                    f"{name} gives --arch a default again")
                self.assertIn('"--arch", required=True', text)


class NoiseFloorParityTest(unittest.TestCase):
    """Finding (58). The measurement noise floor exists twice, for the same
    reason the descriptor rules do -- and for most of its life it existed once.

    Finding (26) established that a route's own run-to-run spread is the floor
    below which a difference is unreadable, and put the table in
    `qd_robust_stats.py`. That is the module that *analyses* measurements. The
    module that *admits* them is `kernel_lane.js`, whose `qdCaseRobust` used a
    bare `median +- 2*MAD` for the entire period. Executed on 20k pairs of arms
    identical in truth, the unfloored rule replaced the archive elite 9.34% of
    the time on `decode_m2_square` and 8.90% on `prefill_m256_down`; floored,
    0.00%. The fix is only durable if the two tables cannot drift, hence this.
    """

    # `text` defaults to the real lane. It exists so the mutation probe at the
    # bottom of this class can point the same parser at a deliberately broken
    # copy: a parity check nobody has ever seen go red is a claim, not a guard.
    def lane_machine(self, text: str | None = None) -> str:
        match = re.search(r"const\s+QD_CURRENT_MACHINE\s*=\s*'([^']+)'\s*;",
                          SOURCE if text is None else text)
        if match is None:
            raise AssertionError(
                "QD_CURRENT_MACHINE is no longer a single-quoted literal in "
                "kernel_lane.js; this parser must be updated before the parity "
                "claim means anything")
        return match.group(1)

    def lane_noise_floor_by_machine(self, text: str | None = None
                                    ) -> dict[str, dict[str, float]]:
        block = re.search(
            r"const\s+QD_NOISE_FLOOR_BY_MACHINE\s*=\s*new Map\(\s*\[(.*?)\n\]\s*\)\s*;",
            SOURCE if text is None else text, re.S)
        if block is None:
            raise AssertionError(
                "QD_NOISE_FLOOR_BY_MACHINE is no longer a `new Map([['<machine>', "
                "new Map([[k, v], ...])], ...])` literal in kernel_lane.js; this "
                "parser must be updated before the parity claim means anything")
        tables = re.findall(
            r"\['([^']+)',\s*new Map\(\s*(\[.*?\n  \])\s*\)\s*\]", block.group(1), re.S)
        if not tables:
            raise AssertionError(
                "QD_NOISE_FLOOR_BY_MACHINE parsed to zero machines -- an empty parse "
                "would make every comparison below vacuously true")
        return {machine: {k: float(v) for k, v in ast.literal_eval(body)}
                for machine, body in tables}

    def lane_noise_floor(self) -> dict[str, float]:
        return self.lane_noise_floor_by_machine()[self.lane_machine()]

    def test_the_two_tables_are_identical(self):
        import qd_robust_stats as QRS
        self.assertEqual(
            self.lane_noise_floor(), dict(QRS.MEASURED_NOISE_FLOOR),
            "the lane's noise floor has drifted from qd_robust_stats.py, which is "
            "the authoritative table; admission and analysis would disagree about "
            "what counts as a readable difference")

    def test_every_machines_table_matches_not_just_the_current_one(self):
        # Keying the floors by machine created a way for the two files to agree
        # today and diverge the moment CURRENT_MACHINE is bumped: only the
        # selected table is exercised, so a typo in a dormant epoch's numbers is
        # invisible until the epoch becomes current, at which point it silently
        # changes admission. Compare the whole structure.
        import qd_robust_stats as QRS
        self.assertEqual(
            self.lane_noise_floor_by_machine(),
            {m: dict(t) for m, t in QRS.MEASURED_NOISE_FLOOR_BY_MACHINE.items()},
            "the lane and qd_robust_stats.py disagree about some machine's floors")

    def test_both_sides_are_pinned_to_the_same_machine(self):
        # Nothing infers the epoch; it is one deliberate line in each file, which
        # is exactly the kind of pair that drifts. A lane reading machine L's
        # floors while the analysis reads N's would admit on one rule and report
        # on another -- and L is 3.3x too narrow on `prefill_m2048_square`.
        import qd_robust_stats as QRS
        self.assertEqual(self.lane_machine(), QRS.CURRENT_MACHINE)

    def test_both_sides_agree_which_epochs_are_provisional(self):
        # Finding (126). A provisional epoch's table is structurally complete
        # and entirely placeholder, so "is this floor a measurement" cannot be
        # answered by looking at the table -- both files carry an explicit set,
        # and a lane that thought Q was measured would drop routes as closed on
        # a number nobody took while the analysis reported them as unmeasured.
        import qd_robust_stats as QRS
        match = re.search(
            r"const\s+QD_PROVISIONAL_MACHINES\s*=\s*new Set\(\s*(\[[^\]]*\])\s*\)\s*;",
            SOURCE)
        self.assertIsNotNone(
            match, "QD_PROVISIONAL_MACHINES is gone from kernel_lane.js or is no "
                   "longer a `new Set([...])` literal; the lane can no longer tell a "
                   "placeholder floor from a measured one")
        self.assertEqual(set(ast.literal_eval(match.group(1))), set(QRS.PROVISIONAL_MACHINES))
        # Every name in it must be a real epoch, or the set is guarding nothing.
        for machine in QRS.PROVISIONAL_MACHINES:
            with self.subTest(machine=machine):
                self.assertIn(machine, self.lane_noise_floor_by_machine())

    def test_the_epoch_pin_check_actually_goes_red_on_a_mismatched_lane(self):
        """The mutation probe for the check above.

        This mutant used to live in `test_js_suite.py::MUTANTS`, asserting that
        the *JS suite* caught a lane repinned to another epoch. It only ever
        appeared to: what caught it was `test_qd_archive.js` hard-coding machine
        N's floor values, a third copy of a table that already exists twice --
        so the container moving to machine O turned a correct floor update into
        three red JS assertions naming the wrong cause. The JS suite cannot
        legitimately decide this one; it sees the lane and nothing else, and
        every machine in the table is a valid lane on its own terms. Which
        epoch is correct is a cross-FILE fact, so the probe belongs next to the
        cross-file check, here.
        """
        import qd_robust_stats as QRS
        by_machine = self.lane_noise_floor_by_machine()
        others = [m for m in by_machine if m != QRS.CURRENT_MACHINE]
        self.assertTrue(others, "only one epoch is recorded, so nothing can drift yet")
        for other in others:
            with self.subTest(repinned_to=other):
                mutated = SOURCE.replace(
                    f"const QD_CURRENT_MACHINE = '{QRS.CURRENT_MACHINE}';",
                    f"const QD_CURRENT_MACHINE = '{other}';", 1)
                self.assertNotEqual(
                    mutated, SOURCE,
                    "the mutant no longer applies -- QD_CURRENT_MACHINE was "
                    "reformatted and this probe is watching nothing")
                self.assertNotEqual(
                    self.lane_machine(mutated), QRS.CURRENT_MACHINE,
                    "a lane repinned to another epoch reads as still agreeing "
                    "with qd_robust_stats.py, so the parity check is vacuous")
                # And the repin must actually change what admission uses --
                # otherwise the epochs are interchangeable and the check, while
                # honest, is guarding nothing.
                self.assertNotEqual(
                    by_machine[other], dict(QRS.MEASURED_NOISE_FLOOR),
                    f"machine {other}'s floors are identical to "
                    f"{QRS.CURRENT_MACHINE}'s; one of the two tables was "
                    f"copied rather than measured")

    def test_an_unmeasured_context_gets_the_widest_floor_on_both_sides(self):
        import qd_robust_stats as QRS
        by_machine = self.lane_noise_floor_by_machine()
        self.assertRegex(
            SOURCE,
            r"const\s+QD_DEFAULT_NOISE_FLOOR\s*=\s*Math\.max\(\s*"
            r"\.\.\.\[\.\.\.QD_NOISE_FLOOR_BY_MACHINE\.values\(\)\]"
            r"\.flatMap\(\(t\) => \[\.\.\.t\.values\(\)\]\)\)",
            "the lane's default floor is no longer the widest floor measured on "
            "ANY machine; a narrower default fails OPEN, admitting noise as an "
            "elite. Note the widest floor on the CURRENT machine is also too "
            "narrow -- an unmeasured route belongs to no epoch and cannot borrow "
            "this one's spread")
        widest = max(v for table in by_machine.values() for v in table.values())
        self.assertEqual(widest, QRS.DEFAULT_NOISE_FLOOR)
        # And the two candidate rules -- "widest anywhere" and "widest here" --
        # must actually differ somewhere, otherwise this test proves nothing.
        #
        # Finding (124) again: this used to read `assertGreater(widest,
        # max(current table))`, which is a fact about WHICH epoch is current,
        # not about the rule. It holds on N/O/P and is false by construction on
        # a provisional epoch (every route already carries the default) and
        # false on L (the epoch that supplies the max). Ask the structural
        # question instead: some epoch must be strictly narrower than the
        # default, and where the CURRENT one is, say so on the live table too.
        narrower = {m: max(t.values()) for m, t in by_machine.items()
                    if max(t.values()) < widest}
        self.assertTrue(
            narrower,
            "every epoch's widest floor equals the default, so 'widest on any "
            "machine' and 'widest on this machine' are the same rule and the "
            "fallback is untested")
        if QRS.CURRENT_MACHINE in narrower:
            self.assertGreater(widest, max(self.lane_noise_floor().values()))

    def test_the_admission_radius_is_floored_not_bare_mad(self):
        # The whole point of the finding: this is the expression that decides
        # whether an elite is replaced.
        self.assertRegex(
            SOURCE,
            r"const radius = Math\.max\(2 \* mad, Math\.abs\(median\) \* qdNoiseFloor\(contextId\)\)",
            "qdCaseRobust is back to a bare 2*MAD radius; at n=3 that is a "
            "0.91-sigma interval, not a 95% one")
        self.assertNotRegex(
            SOURCE, r"lower:\s*Math\.max\(1e-9,\s*median - 2 \* mad\)",
            "the unfloored lower bound has returned")

    def test_every_harness_context_has_a_measured_floor(self):
        # A context missing from the table is not an error -- it falls back to
        # the widest floor -- but it should be a deliberate, visible gap.
        import qd_robust_stats as QRS
        lane = self.lane_noise_floor()
        self.assertEqual(len(lane), 11,
                         "the 11-case harness suite should have 11 measured floors; "
                         f"found {len(lane)}: {sorted(lane)}")
        for context, floor in lane.items():
            with self.subTest(context=context):
                self.assertGreater(floor, 0.0)
                self.assertLessEqual(floor, QRS.DEFAULT_NOISE_FLOOR)


class CellOpeningIsReadableTest(unittest.TestCase):
    """Finding (99), the planner-facing half.

    The lane writes `opened_empty_cell` and `test_qd_archive.js` checks it is
    computed correctly. Neither can see whether anyone downstream is told how to
    read it. A flag that is written and unread is not a repair -- the planner
    goes on reading `improved`, which is `false` for every cell a mechanism
    OPENS, and files the archive's best mechanism as a dud (76).
    """

    def lead_text(self) -> str:
        return (LANE.parent / "roles" / "tech_lead.md").read_text(encoding="utf-8")

    def test_the_lane_writes_the_flag(self):
        self.assertRegex(SOURCE, r"opened_empty_cell: replacement && !incumbent")

    def test_improved_was_not_widened_instead(self):
        # "Beat the incumbent" and "reached somewhere nothing had reached" are
        # different events. Folding the second into the first would make the
        # count go up and the distinction disappear.
        self.assertRegex(
            SOURCE,
            r"improved: replacement && !!incumbent && robust\.score > incumbent\.robust\.score",
            "`improved` has been widened rather than joined by a second flag; a "
            "planner can no longer tell an opening from a win")

    def test_the_planner_is_told_the_difference(self):
        text = self.lead_text()
        self.assertIn("opened_empty_cell", text,
                      "the tech lead prompt never mentions the flag, so the field is "
                      "written and unread")
        # The naive reading is exactly backwards, so the prompt has to say so
        # rather than merely list the field among the others.
        self.assertRegex(text, r"strongest evidence, not its weakest")
        self.assertRegex(text, r"1\.72x-2\.29x",
                         "the prompt states the rule without the case that produced "
                         "it; a rule with no example is one a planner talks itself out of")


class TwinSyncIsActuallyRunTest(unittest.TestCase):
    """Finding (87), the wiring half.

    `hip_twin_sync.py` was written, tested, correct, and called by nobody. This
    class is about the seam a JS suite cannot see: the tool is a script on disk,
    the receipt is produced by an agent following a prompt, and the gate is in
    the lane. If the prompt never asks for the block, the gate refuses every
    verification; if the prompt asks for a boolean instead of the exit code, the
    one outcome that matters -- exit 2, "nothing was checked" -- silently
    becomes a pass.
    """

    def verify_text(self) -> str:
        return (LANE.parent / "roles" / "verify_engineer.md").read_text(encoding="utf-8")

    def test_the_tool_still_returns_the_four_codes_the_gate_reads(self):
        tool = (Path(__file__).resolve().parent
                / "hip_twin_sync.py").read_text(encoding="utf-8")
        self.assertIn("0 = all pairs in lockstep", tool)
        self.assertIn("2 = nothing to check", tool)
        # 3 joined the set when the launch half stopped being an accepted hole.
        # Two codes now mean "not a pass but not a defect either", and the gate
        # has a distinct message for each; a tool that stopped emitting one
        # would leave that message unreachable.
        self.assertIn("3 = pairs compared, but a launch could not be normalized", tool)
        self.assertRegex(tool, r"return 3 if holes else 0",
                         "the gate distinguishes drift from a hole by exit code; if the "
                         "tool stops emitting them the gate is reading a constant")
        self.assertRegex(tool, r"if bad:\s*\n\s*return 1",
                         "drift must still outrank an unreadable launch: a pair that is "
                         "provably stale is a finding, not an unknown")

    def test_the_gate_has_a_branch_for_every_code_the_tool_can_return(self):
        """(44)'s third corner for the exit-code vocabulary itself.

        The tool and the gate are in different languages and different files, so
        adding a code to one and not the other is a two-line change that looks
        complete from either side. It fails shut today -- an unhandled code hits
        `unknown_exit` -- but "fails shut with the wrong message" is how a real
        hole gets read as a tooling glitch and waved through.
        """
        tool = (Path(__file__).resolve().parent
                / "hip_twin_sync.py").read_text(encoding="utf-8")
        # The first entry shares its line with "Exit codes:", the rest are
        # aligned under it -- so match the code and its `=`, not the indent.
        codes = set(re.findall(r"(?:^Exit codes:\s+|^\s{4,})(\d) = ", tool, re.M))
        self.assertEqual({"0", "1", "2", "3"}, codes,
                         "the documented exit codes changed; the gate below is pinned "
                         "against this list")
        lane = LANE.read_text(encoding="utf-8")
        gate = lane[lane.index("const qdTwinReject"):]
        gate = gate[:gate.index("\nconst qdMedian")]
        for code in sorted(codes - {"0"}):
            with self.subTest(exit_code=code):
                self.assertIn(f"s.exit_code === {code}", gate,
                              f"exit {code} has no branch of its own, so it is reported "
                              "as an unknown code rather than as what it means")

    def test_the_prompt_asks_for_the_receipt(self):
        text = self.verify_text()
        self.assertIn("hip_twin_sync.py", text,
                      "the verifier is never told to run the tool, so the gate refuses "
                      "every verification and the round dies on paperwork")
        self.assertIn('"hip_twin_sync"', text,
                      "the return-JSON template omits the block the orchestrator requires")

    def test_the_prompt_states_that_exit_two_is_not_a_pass(self):
        text = self.verify_text()
        self.assertRegex(text, r"no pair was found, so nothing was checked")
        self.assertRegex(text, r"This is not a\s*\n?\s*pass")

    def test_the_prompt_states_that_exit_three_is_not_a_pass_either(self):
        """The corner most likely to be skipped when a code is added.

        The gate refuses exit 3 whatever the prompt says, so leaving the prompt
        at three codes costs nothing visible -- until a verifier hits it, reads
        "every line matched", reports it as clean, and loses the candidate to a
        refusal it was never told about. That reads as a tooling glitch, which
        is how a real hole gets waved through (53) from the other direction.
        """
        text = self.verify_text()
        self.assertIn("four meanings, not two", text,
                      "the prompt still enumerates three codes")
        self.assertRegex(text, r"the launch half went\s*\n?\s*UNCHECKED")
        self.assertRegex(text, r"orchestrator refuses exit 3")
        self.assertIn("exit_code", text,
                      "asking for a boolean instead of the exit code erases the "
                      "distinction between 'in lockstep' and 'nothing was checked'")

    def test_the_gate_is_armed_only_where_the_hazard_exists(self):
        # `target_language` defaults to triton, which has no X.hip/X_hip.hip
        # pair. Arming the gate there would refuse every candidate in the lane
        # for a hazard that cannot occur in it -- fail-shut, not fail-closed.
        self.assertRegex(SOURCE, r"const TWIN_LANGUAGES = \['hip', 'cuda'\];")
        self.assertRegex(
            SOURCE, r"if \(!TWIN_APPLICABLE\) return null;",
            "the gate is armed for every language, so a triton lane dies on a "
            "check about hipified .hip twins")

    def test_the_lane_gates_on_it(self):
        self.assertRegex(SOURCE, r"const qdTwinReject = \(rep, label\) => \{")
        self.assertRegex(SOURCE, r"if \(s\.exit_code === 2\) \{",
                         "the gate does not treat exit 2 separately, so the tool's own "
                         "HOLE/ok distinction is lost at the consumer")
        self.assertEqual(4, len(re.findall(r"qdTwinReject\(", SOURCE)),
                         "the gate is defined but not called at all four paths that "
                         "accept a freshly built artefact (greedy admission, archive "
                         "admission, imported verify, integrate)")


class IntegrateReceiptsAreAskedForTest(unittest.TestCase):
    """Findings (69) and (87) on the merge path -- the (44) triangle again.

    The lane has re-derived the integrator's policy verdict from a
    `policy_postbuild` block for several rounds, and `INTEGRATE_SCHEMA` has
    declared the field for just as long. `integrator.md` never asked for it.
    So in greedy mode -- the only mode that reaches integration at all --
    every merge that claimed `policy_pass: true` was discarded on
    `policy:postbuild_summary_absent`, and the log line said so in a place
    nobody was reading. A gate the prompt cannot satisfy is not a strict
    gate; it is an unreachable branch that reads as one.
    """

    @property
    def INTEGRATOR(self):
        return (LANE.parent / "roles" / "integrator.md").read_text(encoding="utf-8")

    def test_the_prompt_asks_for_the_postbuild_summary(self):
        self.assertIn('"policy_postbuild"', self.INTEGRATOR,
                      "the lane re-derives the merge's policy verdict from this "
                      "block and refuses the merge without it")

    def test_the_prompt_asks_for_the_twin_receipt(self):
        self.assertIn('"hip_twin_sync"', self.INTEGRATOR)
        self.assertIn("hip_twin_sync.py", self.INTEGRATOR,
                      "the prompt names the field but never says how to fill it")

    def test_the_prompt_says_exit_two_is_not_a_pass(self):
        self.assertRegex(self.INTEGRATOR, r"[Ee]xit 2[^.]*not[^.]*pass")

    def test_the_example_receipt_is_one_the_gate_would_accept(self):
        # `exit_code: 0` with `pairs: 0` is the shape `qdTwinReject` refuses as a
        # fabrication, so showing it in the template would teach the refusal.
        self.assertNotRegex(self.INTEGRATOR, r'"exit_code": 0, "pairs": 0')

    def test_the_schema_carries_both(self):
        block = re.search(r"const INTEGRATE_SCHEMA = obj\(\{.*?\}, \[", SOURCE, re.S).group(0)
        self.assertIn("policy_postbuild:", block)
        self.assertIn("hip_twin_sync:", block)


class ArchPreconditionsAreAskedForAndReadTest(unittest.TestCase):
    """Finding (86), the two prompt sides of the new namespace.

    A namespace nobody writes to and nobody reads is worse than no namespace:
    it looks like coverage. So the seed prompt has to ask for the records, with
    the evidence requirement stated, and the planner prompt has to say what they
    are for -- specifically that they are constraints, not directions, since a
    planner that spends a budget slot re-establishing a precondition has been
    made worse by the feature.
    """

    @property
    def SEED(self):
        return (LANE.parent / "roles" / "verify_engineer.md").read_text(encoding="utf-8")

    @property
    def LEAD(self):
        return (LANE.parent / "roles" / "tech_lead.md").read_text(encoding="utf-8")

    def test_the_seed_prompt_asks_for_the_field(self):
        self.assertIn('"preconditions"', self.SEED)

    def test_the_seed_prompt_lists_the_closed_kind_vocabulary(self):
        text = self.SEED
        for kind in ("build_guard", "toolchain", "runtime_env",
                     "attainable_ceiling", "harness"):
            self.assertIn(kind, text)

    def test_the_kind_list_in_the_prompt_is_the_lane_list(self):
        kinds = re.search(r"const PRECONDITION_KINDS = \[([^\]]*)\]", SOURCE).group(1)
        kinds = re.findall(r"'([a-z_]+)'", kinds)
        self.assertEqual(["build_guard", "toolchain", "runtime_env",
                          "attainable_ceiling", "harness"], kinds,
                         "the lane refuses any kind not on its list; if the two drift, "
                         "the prompt teaches a value the lane throws away")

    def test_the_seed_prompt_makes_evidence_mandatory_and_says_why(self):
        self.assertRegex(self.SEED, r"`evidence` is mandatory")
        self.assertRegex(self.SEED, r"inherited by every later run")

    def test_the_seed_prompt_forbids_manufacturing_a_record(self):
        self.assertRegex(self.SEED, r"[Oo]mit the field entirely if this arch needed nothing")

    def test_the_planner_prompt_says_a_precondition_is_not_a_direction(self):
        self.assertRegex(self.LEAD, r"not a menu of directions|[Dd]o not propose one as a direction")

    def test_the_planner_prompt_does_not_read_an_empty_list_as_a_clean_arch(self):
        # (54)/(60): the orchestrator refuses unevidenced records, so absence here
        # is the absence of a claim, not a negative result.
        self.assertRegex(self.LEAD, r"not proof of a\s+clean arch")

    def test_the_field_the_planner_is_told_to_read_is_the_field_the_lane_emits(self):
        self.assertIn("QD_ARCHIVE.preconditions", self.LEAD)
        self.assertRegex(SOURCE, r"preconditions: \{ arch: QD_ARCH, records:")


class RoadmapProfileMayNotPublishASolTest(unittest.TestCase):
    """Finding (89), prompt side.

    The lane deletes SOL-shaped keys from the baseline/reprofile report, but a
    strip alone leaves the agent producing them every round and wondering why
    they vanish -- and, worse, still writing them into `profiling_summary.md`,
    which the orchestrator cannot reach. The prompt has to say the branch has
    no ceiling and why, or the fix only covers the half that travels as JSON.
    """

    @property
    def PROFILE(self):
        return (LANE.parent / "roles" / "profile_engineer.md").read_text(encoding="utf-8")

    def test_the_prompt_forbids_the_field_names_the_lane_strips(self):
        text = self.PROFILE
        for name in ("sol_", "peak_pct", "compute_floor", "memory_floor",
                     "remaining_headroom", "roofline"):
            self.assertIn(name, text,
                          f"the lane deletes {name}* but the prompt never names it, so the "
                          "agent cannot know not to emit it")

    def test_the_prompt_covers_the_markdown_the_lane_cannot_reach(self):
        self.assertRegex(self.PROFILE, r"profiling_summary\.md[^.]*(either|too)",
                         "the strip only sees the return JSON; the summary file is the "
                         "one place the removal is on the agent's honour")

    def test_the_prompt_still_allows_the_classification(self):
        # (60): a refusal must not be renamed into a paperwork problem. The branch
        # is still for classifying the bottleneck -- it just may not attach a ratio.
        self.assertRegex(self.PROFILE, r"compute- or memory-bound")

    def test_the_prompt_names_the_branch_that_can_answer(self):
        self.assertIn("PHASE=selected_cell_sol", self.PROFILE)

    def test_the_device_example_labels_its_bandwidth_as_nameplate(self):
        self.assertRegex(self.PROFILE, r"nameplate ~5\.3 TB/s",
                         "an unlabelled vendor peak in `device` is where the 2x roofline "
                         "error came from")

    def test_the_lane_strips_both_branches(self):
        self.assertEqual(2, len(re.findall(r"profileSolStrip\(profileSummary", SOURCE)),
                         "baseline and reprofile both write profileSummary, which is what "
                         "the TechLead reads")


class ComputeCeilingWitnessIsAskedForTest(unittest.TestCase):
    """Finding (89) item 2, the prompt corner of the (44) triangle.

    The lane refuses a compute-bound case whose peak nothing has been observed
    to reach, and the schema makes the block mandatory. Neither is any use if
    `profile_engineer.md` never tells the agent to transcribe the three fields
    the helper emits -- that is (100) exactly: a gate the merge path enforces
    with no prompt behind it, which turns every card into a refusal that reads
    like an empty cell (44).
    """

    @property
    def PROFILE(self):
        return (LANE.parent / "roles" / "profile_engineer.md").read_text(encoding="utf-8")

    def test_the_prompt_asks_for_every_field_the_schema_requires(self):
        text = self.PROFILE
        for name in ("compute_ceiling", "witnessed", "attainment", "witness"):
            self.assertIn(name, text,
                          f"the case schema requires {name} and the prompt never names it")

    def test_the_prompt_says_the_transcription_is_verbatim(self):
        # Same rule as the bandwidth block: a field the agent reasons about is
        # a field the agent can talk itself into.
        self.assertRegex(self.PROFILE, r"transcribed, never reasoned about")

    def test_the_prompt_states_which_cases_are_dropped(self):
        self.assertRegex(self.PROFILE, r"compute\s+floor is the binding one while `witnessed` is false",
                         "an agent that does not know which cases die cannot avoid killing them")

    def test_the_prompt_forbids_the_two_ways_out_that_are_not_fixes(self):
        text = self.PROFILE
        self.assertRegex(text, r"relabelling the roof")
        self.assertRegex(text, r"inventing a witness")

    def test_the_prompt_says_provenance_is_not_attainability(self):
        # The whole point. Without this sentence the agent reads `witnessed`
        # as "did the number come from somewhere", which every number does.
        self.assertRegex(self.PROFILE, r"[Pp]rovenance is not attainability")

    def test_the_prompt_says_memory_bound_cases_are_unaffected(self):
        # (53)/(87). An agent that thinks the gate is global will stop
        # profiling an arch that has no witness at all.
        self.assertRegex(self.PROFILE, r"memory floor binds are unaffected")

    def test_the_lane_and_the_helper_agree_on_the_three_field_names(self):
        card = (LANE.parent / "scripts" / "qd_sol_card.py").read_text(encoding="utf-8")
        for emitted, transcribed in (("compute_ceiling_witnessed", "witnessed"),
                                     ("compute_ceiling_attainment", "attainment"),
                                     ("compute_ceiling_witness", "witness")):
            self.assertIn(f'"{emitted}"', card)
            self.assertIn(f"{transcribed}: ", SOURCE)

    def test_the_helper_schema_string_was_bumped_with_the_fields(self):
        # (92): a v2 consumer reading a v3 card sees an attainability claim it
        # does not know is there. The version string is what makes it fail.
        card = (LANE.parent / "scripts" / "qd_sol_card.py").read_text(encoding="utf-8")
        self.assertIn('SCHEMA = "geak.qd-sol-card/v3"', card)
        self.assertIn("SCHEMA_V2", card)
        self.assertIn("ACCEPTED_SCHEMAS = (SCHEMA, SCHEMA_V2, SCHEMA_V1)", card)


class UnmeasuredRouteIsNotClosedTest(unittest.TestCase):
    """Finding (92), across all three sides of the (44) triangle.

    `qd_route_priority.py` now answers `needs_fresh_elapsed` when the caller
    supplied no latency, because the fallback is a different kernel's. Its own
    tests cover the module and `test_qd_archive.js` covers the lane gate. What
    neither can see is the seam: the planner produces the receipt by running a
    documented command, and that command in `tech_lead.md` passes no
    `--elapsed-ms`. So the third verdict is the ordinary case, not the exotic
    one, and a prompt that does not mention it leaves the planner to guess --
    and the guess that costs the most is the natural one, "it isn't `open`, so
    drop it".
    """

    def lead_text(self) -> str:
        return (LANE.parent / "roles" / "tech_lead.md").read_text(encoding="utf-8")

    def test_the_module_and_the_lane_agree_on_the_verdict_name(self):
        # Behavioural, not textual, and deliberately so. This used to grep
        # qd_route_priority.py for `verdict = "needs_fresh_elapsed"`, which is
        # the same claim spelled in a way that also fails when the file is
        # merely reformatted -- and `mutate_python.py` rewrites whole modules
        # through `ast.unparse`, which renders every string in single quotes.
        # The result was that this one assertion "killed" all 96 of that
        # module's mutants and the sweep reported 100% coverage for a file with
        # twenty unpinned constants in it. A parity test must object to the
        # meaning changing, never to the quotes changing.
        import qd_route_priority as RP
        self.assertEqual("needs_fresh_elapsed",
                         RP.route_priority("decode_m2_square")["verdict"],
                         "the helper no longer emits the verdict the lane below "
                         "is checked against")
        self.assertIn("row.verdict === 'needs_fresh_elapsed'", SOURCE,
                      "the lane does not recognise the verdict the helper emits, so "
                      "every receipt built from the documented command fails the "
                      "arithmetic check and every direction is dropped")

    def test_the_lane_keeps_an_unmeasured_route(self):
        # Finding (126) added a third disjunct for the same reason: a closure
        # derived from a PLACEHOLDER floor is also a claim about a measurement
        # nobody took. Both escape hatches must be present, and the drop must
        # still be reachable -- an unconditional `kept.push` would satisfy a
        # test that only looked for the words.
        self.assertRegex(
            SOURCE,
            r"if \(unmeasured \|\| derived !== 'closed' \|\| "
            r"!qdFloorIsMeasured\(context\)\) kept\.push\(context\);",
            "the gate drops a route nobody measured, which is the self-sealing "
            "half of (92): never proposed, therefore never measured")

    def test_the_conditional_verdict_is_still_re_derived(self):
        # The escape hatch has to stay checkable. If `needs_fresh_elapsed` were
        # exempt from the arithmetic, it would be the one verdict a planner can
        # assert freely -- and it is also the verdict that keeps a route alive.
        self.assertRegex(
            SOURCE,
            r"const claimed = unmeasured \? row\.verdict_if_elapsed_confirmed : row\.verdict;")
        self.assertRegex(SOURCE, r"if \(claimed !== derived\) \{")

    def test_the_planner_is_told_not_to_drop_it(self):
        text = self.lead_text()
        self.assertIn("needs_fresh_elapsed", text,
                      "the prompt documents a command that returns this verdict for "
                      "every row and never names it")
        self.assertIn("verdict_if_elapsed_confirmed", text)
        self.assertRegex(text, r"[Dd]o not drop a\s*\n?`needs_fresh_elapsed` route")
        self.assertIn("--elapsed-ms", text,
                      "the prompt says the default is stale without saying how to "
                      "supply a fresh number, which leaves the planner stuck")


class SeedIntervalIsMeasurableTest(unittest.TestCase):
    """Finding (95), the reachability half.

    The behaviour lives in `test_qd_archive.js`, which runs the real
    `qdSeedRobust` against real numbers. What that suite cannot see is whether
    anything upstream ever *produces* the repeats it reads. A lane that accepts
    `samples_ms` from a benchmark engineer who was never told to send it is the
    (44) triangle again: the floored fallback would fire on every run forever,
    correctly and invisibly, and the measured branch would be dead code that
    passes its own tests.

    So the two sides are pinned together: the lane reads the field, and a role
    prompt asks for it and says what is lost by omitting it.
    """

    def test_the_lane_reads_per_case_baseline_repeats(self):
        self.assertRegex(
            SOURCE, r"const QD_BASELINE_SAMPLES = new Map\(",
            "the lane no longer collects per-case baseline repeats, so the seed "
            "interval can only ever be the floored fallback")
        self.assertIn("samples_ms", SOURCE)

    def test_the_seed_interval_is_not_a_literal(self):
        self.assertNotRegex(
            SOURCE, r"robust:\s*\{\s*score:\s*1,\s*median:\s*1,\s*mad:\s*0,"
                    r"\s*lower:\s*1,\s*upper:\s*1\s*\}",
            "the hardcoded zero-width seed interval is back; every comparison "
            "against the seed is one-sided again (95)")
        self.assertRegex(SOURCE, r"robust: qdSeedRobust\(rc\.context_id\)")

    def test_the_fallback_is_floored_at_the_same_rule_as_a_candidate(self):
        body = re.search(r"const qdSeedRobust = \(contextId\) => \{(.*?)\n\};",
                         SOURCE, re.S)
        self.assertIsNotNone(body, "qdSeedRobust is gone or reshaped")
        self.assertIn("Math.max(2 * mad, Math.abs(median) * qdNoiseFloor(contextId))",
                      body.group(1),
                      "the seed's radius rule has drifted from the candidate's; the "
                      "two sides of every admission comparison would be computed "
                      "differently")

    def test_a_role_prompt_asks_for_the_repeats(self):
        prompts = [p for p in sorted((LANE.parent / "roles").glob("*.md"))
                   if "samples_ms" in p.read_text(encoding="utf-8")]
        self.assertTrue(prompts,
                        "no role prompt mentions samples_ms, so nothing upstream will "
                        "ever send it and the measured seed interval is dead code")
        text = "\n".join(p.read_text(encoding="utf-8") for p in prompts)
        self.assertRegex(text, r"at least three complete primed\s+repeats",
                         "the prompt asks for repeats without saying how many")
        # An omission must read as a cost, not as a default.
        self.assertRegex(text, r"floored rather than measured")
        # And fabrication must be refused explicitly: a plausible invented spread
        # is the one failure mode the lane cannot detect.
        self.assertRegex(text, r"[Dd]o not synthesize the repeats")


class FnvParity(unittest.TestCase):
    """The persistence checksum exists TWICE, for the same reason the descriptor
    rules do, and with a worse failure mode.

    `qdFnv1a32` in kernel_lane.js computes the digest of the payload it emits;
    `fnv1a32` in qd_persist_manifest.py recomputes it over the file an agent
    wrote from that payload. If the two disagree, every persistence attempt in
    every run refuses with a checksum mismatch and no cell is ever written --
    which reads exactly like a search that found nothing (44). Nothing else in
    the suite would catch it: each side is internally consistent and separately
    tested against its own vectors.
    """

    @classmethod
    def setUpClass(cls):
        try:
            from py_mini_racer import MiniRacer
        except ImportError:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest("py_mini_racer is not installed; JS side unchecked")
        lines = SOURCE.split("\n")
        try:
            start = next(i for i, l in enumerate(lines) if l.startswith("const qdFnv1a32 ="))
            end = next(i for i, l in enumerate(lines) if l.startswith("const qdAsciiJson =")) + 2
        except StopIteration:  # pragma: no cover - only if the lane is refactored
            raise AssertionError("kernel_lane.js no longer defines qdFnv1a32/qdAsciiJson")
        cls.ctx = MiniRacer()
        cls.ctx.eval("\n".join(lines[start:end]))

    def js_fnv(self, text: str) -> str:
        import json as _json
        return self.ctx.eval("qdFnv1a32(" + _json.dumps(text) + ")")

    def test_the_two_implementations_agree_on_ascii(self):
        import qd_persist_manifest as P
        for text in ("", "a", "foobar", "decode_m16_square|native_mfma|independent",
                     '{"cells":{"k":{"elite_id":"r1_s0"}},"generation":1}', "x" * 5000):
            with self.subTest(text=text[:32]):
                self.assertEqual(self.js_fnv(text), P.fnv1a32(text))

    def test_reference_vectors(self):
        # Pinned so "they agree with each other" cannot be satisfied by two
        # identically wrong functions.
        self.assertEqual(self.js_fnv(""), "811c9dc5")
        self.assertEqual(self.js_fnv("a"), "e40c292c")
        self.assertEqual(self.js_fnv("foobar"), "bf9cf968")

    def test_non_ascii_is_refused_rather_than_silently_divergent(self):
        # The JS hashes UTF-16 code units and Python hashes UTF-8 bytes, so
        # above U+007F they simply differ. The lane throws instead.
        with self.assertRaises(Exception):
            self.ctx.eval('qdFnv1a32("caf\\u00e9")')

    def test_ascii_json_stays_ascii_and_round_trips(self):
        import json as _json
        import qd_persist_manifest as P
        obj = {"cells": {"decode_m16_square|a|b": {"elite_id": "r1_s0", "note": "caf\u00e9 \u2014 na\u00efve"}},
               "n": 1.5, "t": True, "z": None}
        self.ctx.eval("var OBJ = " + _json.dumps(obj) + ";")
        text = self.ctx.eval("qdAsciiJson(OBJ)")
        self.assertTrue(text.isascii())
        self.assertEqual(_json.loads(text), obj)
        self.assertEqual(self.ctx.eval("qdFnv1a32(qdAsciiJson(OBJ))"), P.fnv1a32(text))



if __name__ == "__main__":
    unittest.main(verbosity=2)


class PlannedVsDispatchedIsRecordedTest(unittest.TestCase):
    """Six gates stand between `plan.directions` and the engineers that run.

    They are: `qdParentReject`, `qdResidencyReject`, `qdPriorityFilter`, the
    repeated-local-mutation capsule block, the dedicated-round collapse
    (`directions = [deepDir]`), and the affordability loop. The first four each
    log the direction they drop and why. The last two dropped silently, which
    made the round summary -- the only line either an operator or a later audit
    ever reads -- unable to distinguish a round that planned four directions and
    ran one from a round whose planner returned one.

    Nothing downstream can reconstruct it. The lane has no filesystem access, so
    `round_N/` contains a directory per DISPATCHED engineer and nothing about
    the ones that were shed; the archive records admissions, not intentions.
    `log()` into the workflow journal is the only channel, which is why these
    are lexical assertions on the lane rather than assertions on an artifact.

    (55): a defence wired into a path nobody executes is a comment. The
    inverse holds too -- a log line no test names is a line the next edit
    deletes without noticing.
    """

    def setUp(self):
        self.src = LANE.read_text()

    def test_the_dedicated_round_collapse_names_what_it_shed(self):
        block = re.search(r"const deepDir = directions\.find.*?\n  \}", self.src, re.S)
        self.assertIsNotNone(block, "the dedicated-round collapse is no longer where the test looks")
        body = block.group(0)
        self.assertIn("log(", body,
                      "directions = [deepDir] discards every co-planned direction without a "
                      "record; on disk that is indistinguishable from a planner that returned one")
        self.assertIn("shed", body)

    def test_an_unaffordable_direction_says_so(self):
        block = re.search(r"for \(const d of directions\) \{.*?\n  \}", self.src, re.S)
        self.assertIsNotNone(block, "the affordability loop moved")
        body = block.group(0)
        self.assertIn("unaffordable", body,
                      "a direction skipped for budget leaves no trace, so a round that ran "
                      "nothing because it was broke reads like a round that planned nothing")
        self.assertRegex(body, r"log\([^)]*d\.id", "the log line does not name which direction")

    def test_neither_shed_reason_calls_itself_a_rejection(self):
        """A budget skip and a stale-parent refusal are opposite facts. One says
        come back with more budget, the other says this direction is dead. The
        capsule ledger is keyed on repeated FAILURE, so filing an unevaluated
        direction as a rejection would eventually block a route nobody tested."""
        for token in ("shedding", "unaffordable"):
            with self.subTest(token=token):
                call = re.search(r"log\((?:[^()]|\([^()]*\))*?" + token
                                 + r"(?:[^()]|\([^()]*\))*?\);", self.src, re.S)
                self.assertIsNotNone(call, f"no log call mentions {token!r}")
                self.assertIn("Not a rejection", call.group(0),
                              f"the {token} message does not distinguish an unevaluated "
                              f"direction from a refused one")

    def test_the_round_summary_reports_both_counts(self):
        summary = re.search(r"log\(`Round \$\{round\}: planned.*?\);", self.src, re.S)
        self.assertIsNotNone(summary,
                             "the round summary no longer reports the planned count, only the "
                             "surviving one, which overstates the planner and hides the gates")
        body = summary.group(0)
        self.assertIn("plan.directions.length", body)
        self.assertIn("directions.length", body)


class RecoveredVerifyIsDistinguishedTest(unittest.TestCase):
    """A missing engineer return and a complete failing one are not one event.

    `recovered = !eng || eng.status === 'failed'` covers both, and both go to
    verify -- correctly, and deliberately: the design refuses to let an
    engineer's self-report suppress an independent measurement, and an
    engineer's own number has been wrong before. But they cost the same full
    verify cycle for different reasons. MISSING means the patch on disk was
    never examined. FAILED-but-complete means the engineer measured it, wrote
    the numbers down, and lost. Round 2's `r2_s0` was the second kind: a
    complete result, geomean 0.9007, `prefill_m256_down` at 0.2719 -- a worst
    case `qdAdmissionCheck` will refuse against QD_CELL_GUARDRAIL 0.8. Whether
    that path deserves a cheaper pre-check is a question about how often it
    happens, and the round log is the only place it can be counted.
    """

    def setUp(self):
        self.src = LANE.read_text()
        block = re.search(r"const recovered = !eng \|\| eng\.status === 'failed';"
                          r".*?\n      \}\n", self.src, re.S)
        self.assertIsNotNone(block, "the recovery branch is no longer where the test looks")
        self.body = block.group(0)

    def test_the_two_recovery_causes_print_differently(self):
        self.assertIn("MISSING", self.body)
        self.assertIn("FAILED but complete", self.body)

    def test_the_failed_branch_reports_the_numbers_the_engineer_gave(self):
        self.assertIn("primSpeedup(eng)", self.body)
        self.assertIn("QD_CELL_GUARDRAIL", self.body,
                      "the log reports a worst case without the bar it will be judged against")

    def test_the_worst_case_is_not_taken_from_qdMinCase(self):
        """qdMinCase reads per-SAMPLE arrays through qdCaseRobust and returns 0
        when any context lacks them. An engineer result carries scalar per-case
        speedups and no samples, so it would print a confident 0.0000 for a
        candidate whose reported worst case is 0.2719."""
        # Comment lines stripped first: the block contains the string
        # "NOT qdMinCase(eng)" in prose, and a check that matched that would
        # pass on a version that also still called it.
        code = "\n".join(l for l in self.body.splitlines()
                         if not l.lstrip().startswith("//"))
        self.assertNotIn("qdMinCase(eng)", code)
        self.assertIn("eng.per_case", code)

    def test_qdMinCase_really_does_return_zero_on_an_engineer_shaped_result(self):
        """The claim above, executed rather than asserted -- (55). A comment
        justifying a workaround by describing what the other function would have
        done is worth exactly as much as the check that it does it.

        The baseline map is stubbed GENEROUSLY -- every context present with a
        real latency -- so the zero cannot be blamed on a missing baseline. What
        makes `qdCaseRobust` return null here is that an engineer result has no
        `case_measurement_samples` at all, which is the actual point.
        """
        try:
            from py_mini_racer import MiniRacer
        except ImportError:  # pragma: no cover
            self.skipTest("no JS engine available (py_mini_racer not installed)")
        src = self.src
        chunks = []
        for name in ("qdMedian", "qdCaseRobust", "qdMinCase"):
            m = re.search(r"const %s = \(.*?\n\};" % name, src, re.S)
            self.assertIsNotNone(m, f"{name} moved; the probe cannot run")
            chunks.append(m.group(0))
        ctx = MiniRacer()
        ctx.eval("const QD_CONTEXT_IDS = new Set(['prefill_m256_down', 'decode_m2_square']);")
        ctx.eval("const QD_BASELINE_MS = new Map(["
                 "['prefill_m256_down', 0.134120], ['decode_m2_square', 0.027441]]);")
        ctx.eval("const qdNoiseFloor = () => 0.02;")
        ctx.eval("\n".join(chunks))
        # Sanity: with samples, it does NOT return 0. Without this the assertion
        # below would also pass on a qdMinCase that returned 0 unconditionally.
        withSamples = {"case_measurement_samples": [
            {"name": "prefill_m256_down", "samples": [0.130, 0.131, 0.129]},
            {"name": "decode_m2_square", "samples": [0.026, 0.0261, 0.0259]}]}
        self.assertGreater(ctx.call("qdMinCase", withSamples), 0.9,
                           "the probe's stubs are wrong; qdMinCase cannot read even a "
                           "well-formed verify result")
        eng = {"per_case": [{"name": "prefill_m256_down", "speedup": 0.2719},
                            {"name": "decode_m2_square", "speedup": 1.049}]}
        self.assertEqual(0, ctx.call("qdMinCase", eng),
                         "qdMinCase no longer returns 0 on a sample-free engineer result; the "
                         "comment justifying the manual per_case scan is now wrong")


class BaselineFrameAmbiguityTest(unittest.TestCase):
    """Findings (34c)/(122). Three field names, three possible frames.

    `latency_ms`, `baseline_ms` and `execution_time_ms` are not synonyms. In a
    bench frame `baseline_ms` is the FROZEN ORACLE's latency; in a QD archive
    elite row it is the PARENT's own latency. On a seed round the two coincide,
    which is exactly why the collision survived so long: the round that would
    expose it is the first round that is not a seed. A precedence list answers
    "which number wins" and never answers "do these two agree", so a suite
    geomean computed against the parent can be reported as one against the
    oracle with nothing in the run complaining.

    The lane now refuses a disagreement instead of ranking it. These tests
    execute the extracted block rather than reading it, because the interesting
    behaviour is a throw and a lexical check cannot tell a throw that fires from
    one that is unreachable.
    """

    @staticmethod
    def _ctx(rows):
        try:
            from py_mini_racer import MiniRacer
        except ImportError:  # pragma: no cover
            raise unittest.SkipTest("no JS engine available (py_mini_racer not installed)")
        import json
        match = re.search(
            r"const QD_BASELINE_FIELDS = .*?\n\}\)\.filter\(\(\[name, latency\]\).*?\);",
            SOURCE, re.S)
        if match is None:  # pragma: no cover
            raise AssertionError(
                "the QD_BASELINE_MS construction moved; this probe is checking nothing")
        ctx = MiniRacer()
        ctx.eval("const BASELINE_PER_CASE = %s;" % json.dumps(rows))
        ctx.eval(match.group(0))
        return ctx

    def test_one_field_alone_is_read(self):
        ctx = self._ctx([{"name": "decode_m2_square", "latency_ms": 0.0271}])
        self.assertEqual(0.0271, ctx.eval("QD_BASELINE_MS.get('decode_m2_square')"))

    def test_two_fields_that_agree_are_one_measurement_reported_twice(self):
        """The ordinary case, and the reason this is a tolerance and not an
        equality: a harness that emits `latency_ms` and `execution_time_ms` for
        the same timing is not ambiguous, it is redundant."""
        ctx = self._ctx([{"name": "decode_m2_square",
                          "latency_ms": 0.027100, "execution_time_ms": 0.0271002}])
        self.assertEqual(0.0271, ctx.eval("QD_BASELINE_MS.get('decode_m2_square')"))

    def test_two_fields_that_disagree_stop_the_run(self):
        with self.assertRaises(Exception) as caught:
            self._ctx([{"name": "prefill_m256_down",
                        "latency_ms": 0.1341, "baseline_ms": 0.2004}])
        message = str(caught.exception)
        self.assertIn("prefill_m256_down", message,
                      "the refusal must name the case, or the operator cannot find it")
        for field in ("latency_ms=0.1341", "baseline_ms=0.2004"):
            self.assertIn(field, message,
                          "the refusal must show both numbers; 'they disagree' without the "
                          "values leaves the reader unable to tell which frame is which")

    def test_the_tolerance_is_relative_and_tight(self):
        """0.1% -- wide enough for a JSON round-trip or a re-derived quotient,
        far too narrow for two different measurements. A parent that is 1%
        faster than the oracle is a REAL 1%, and it must not be swallowed."""
        ctx = self._ctx([{"name": "decode_m8_up",
                          "latency_ms": 0.0500, "baseline_ms": 0.050004}])
        self.assertEqual(0.05, ctx.eval("QD_BASELINE_MS.get('decode_m8_up')"))
        with self.assertRaises(Exception):
            self._ctx([{"name": "decode_m8_up",
                        "latency_ms": 0.0500, "baseline_ms": 0.0505}])

    def test_a_case_with_no_usable_field_is_dropped_not_thrown_on(self):
        """Absence is a different failure from contradiction, and it already has
        an owner: the bootstrap size check below the map throws when a context
        has no baseline. Throwing here too would turn a clear 'this context is
        missing' into an opaque per-row error."""
        ctx = self._ctx([{"name": "decode_m2_square", "latency_ms": 0},
                         {"name": "decode_m8_up", "baseline_ms": 0.0500}])
        self.assertFalse(ctx.eval("QD_BASELINE_MS.has('decode_m2_square')"),
                         "a non-positive latency is not a baseline")
        self.assertEqual(0.05, ctx.eval("QD_BASELINE_MS.get('decode_m8_up')"))

    def test_the_precedence_list_is_gone(self):
        """The regression this guards is a revert, not a rewrite: `.find(...)`
        over the three names reads as a harmless tidy-up and restores the exact
        silence the refusal exists to break."""
        self.assertNotIn(
            "[c.latency_ms, c.baseline_ms, c.execution_time_ms].find(", SOURCE,
            "the precedence chain is back; a disagreement is being ranked again")


class EliteNamesItsDenominatorTest(unittest.TestCase):
    """The other half of (34c): an unambiguous number under an ambiguous name.

    `qdAuthoritativeBaseline` refuses a CONTRADICTION, which needs two numbers.
    A row with one number named `baseline_ms` contradicts nothing and still does
    not say whether that denominator is the frozen oracle or the parent -- and
    the two differ by exactly the accumulated gain of the lineage, which is the
    whole quantity anybody reads an archive for.

    Every elite therefore states the frame its own statistics were computed in.
    All three construction sites must carry it: an elite that omits it is not
    neutral, it is a row whose denominator the reader must guess, and the
    guessable-by-default answer is the wrong one on generation >= 1.
    """

    def test_the_frame_constant_exists_and_says_oracle(self):
        self.assertIn("const QD_ROBUST_BASELINE_FRAME = 'oracle';", SOURCE)

    def test_every_elite_construction_site_carries_the_frame(self):
        """Counted, not merely present. There are three sites -- seed bootstrap,
        import, and admission -- and a label on two of them is worse than none,
        because its absence then reads as a distinction rather than an omission.
        """
        self.assertEqual(
            3, SOURCE.count("robust_baseline_frame: QD_ROBUST_BASELINE_FRAME"),
            "an elite construction site was added or lost its frame label; a row whose "
            "denominator is unstated is a row whose speedup cannot be compared across "
            "generations")
        # And three is the number of sites there ARE -- otherwise the count
        # above is a magic number that a fourth site could satisfy by leaving
        # itself unlabelled while a labelled site was duplicated.
        # `elite_id:` alone also matches schema fragments and read paths, which
        # would make the count meaningless. An archive ENTRY is the thing that
        # opens a cell, so it is the `cell: rc.cell` that identifies one.
        self.assertEqual(3, len(re.findall(r"elite_id: [^\n]*, cell: rc\.cell,", SOURCE)),
                         "the number of elite construction sites changed; recount before "
                         "trusting the label coverage above")

    def test_the_frame_is_not_read_off_the_verifier(self):
        """The label describes the lane's own arithmetic. If it were derived
        from `ver.per_case`, it would inherit exactly the ambiguity it exists to
        remove."""
        self.assertNotIn("robust_baseline_frame: ver.", SOURCE)
        self.assertNotIn("robust_baseline_frame: r.ver.", SOURCE)


class IntervalProvenanceNamesTheBoxTest(unittest.TestCase):
    """A floor is a reading of a box (26)/(58), so an interval is too.

    `measured: 8 primed baseline repeats` states how many and not WHERE. Two
    epochs' seed intervals are not comparable, and an interval sitting in an
    archive with no epoch on it has to be dated by hand from the run id -- which
    is exactly the manual step that produced the cross-epoch pooling this
    project has already had to un-pool twice.
    """

    def js_hostnames(self) -> dict[str, str | None]:
        match = re.search(r"const QD_MACHINE_HOSTNAME = new Map\((\[[\s\S]*?\])\);", SOURCE)
        self.assertIsNotNone(match, "the lane no longer mirrors the hostname table")
        literal = match.group(1).replace("null", "None")
        return {k: v for k, v in ast.literal_eval(literal)}

    def test_the_hostname_table_matches_python_exactly(self):
        """Both directions. A JS table missing an epoch would stamp
        `host unrecorded` on a box whose name we know; a JS table with an extra
        one is a name nobody registered."""
        import qd_robust_stats as QRS
        self.assertEqual(dict(QRS.MACHINE_HOSTNAME), self.js_hostnames())

    def test_an_unrecorded_host_is_stated_not_backfilled(self):
        """L and M pre-date the convention. `null` is the honest answer and it
        has to survive the mirror -- a guessed hostname is worse than none,
        because it is indistinguishable from a recorded one."""
        hosts = self.js_hostnames()
        self.assertIsNone(hosts["L"])
        self.assertIsNone(hosts["M"])
        self.assertIn("host unrecorded", SOURCE,
                      "the stamp has no branch for an epoch whose host was never recorded")

    def test_the_stamp_reaches_every_provenance_string(self):
        """Three of them -- measured, floored, and the suite roll-up's two
        outcomes. One unstamped string is the one that ends up quoted."""
        self.assertEqual(4, SOURCE.count("${QD_EPOCH_STAMP}"),
                         "an interval_provenance string is being written without an epoch on it")

    def test_the_stamp_says_when_the_table_is_provisional(self):
        """Finding (126). An interval measured against a placeholder floor is
        not wrong, but it is not the same evidence as one measured against a
        real table, and the difference has to be legible at the point of quoting
        rather than by cross-referencing the epoch elsewhere."""
        self.assertIn("floor table PROVISIONAL", SOURCE)
        self.assertIn("QD_PROVISIONAL_MACHINES.has(QD_CURRENT_MACHINE)", SOURCE)

    def test_the_stamp_is_executable_and_names_this_box(self):
        try:
            from py_mini_racer import MiniRacer
        except ImportError:  # pragma: no cover
            self.skipTest("no JS engine available (py_mini_racer not installed)")
        import qd_robust_stats as QRS
        match = re.search(
            r"const QD_MACHINE_HOSTNAME = new Map\([\s\S]*?\n\}\)\(\);\n", SOURCE)
        self.assertIsNotNone(match, "the stamp moved; this probe cannot run")
        ctx = MiniRacer()
        ctx.eval(f"const QD_CURRENT_MACHINE = {QRS.CURRENT_MACHINE!r};")
        ctx.eval("const QD_PROVISIONAL_MACHINES = new Set(%s);"
                 % list(sorted(QRS.PROVISIONAL_MACHINES)))
        ctx.eval(match.group(0))
        stamp = ctx.eval("QD_EPOCH_STAMP")
        self.assertIn(QRS.CURRENT_MACHINE, stamp)
        host = QRS.MACHINE_HOSTNAME[QRS.CURRENT_MACHINE]
        self.assertIn(host if host else "host unrecorded", stamp)
        self.assertEqual(QRS.CURRENT_MACHINE in QRS.PROVISIONAL_MACHINES,
                         "PROVISIONAL" in stamp,
                         "the provisional marker and the provisional set disagree")


class SamplesJsonContractIsStatedTest(unittest.TestCase):
    """The three rules `samples.json` has already been broken by, written down.

    A contract that lives only in a checker is a contract the producer never
    reads; the producer here is an agent whose entire specification is its role
    prompt. Each rule is paired with the tool that enforces it, so a rule cannot
    be quietly stated in prose while nothing checks it, and cannot be enforced
    by a tool nobody was told about.
    """

    def role(self) -> str:
        return (LANE.parent / "roles" / "verify_engineer.md").read_text(encoding="utf-8")

    def test_the_shape_is_per_context_arrays(self):
        role = self.role()
        self.assertIn("samples.json", role)
        self.assertIn("One key per exact harness case ID", role)
        self.assertIn("ARRAY of raw per-repeat latencies", role)

    def test_the_unit_is_declared_and_the_refusal_is_named(self):
        """Naming the exit code matters. "be careful with units" is advice; "a
        mislabelled file exits 5" is a consequence, and only the second changes
        what an agent does when it is in a hurry."""
        role = self.role()
        self.assertIn('"samples_unit": "ms"', role)
        self.assertIn("audit_floor_sensitivity.py", role)
        self.assertIn("exits 5", role)

    def test_aggregate_keys_are_forbidden_and_the_reason_given(self):
        role = self.role()
        self.assertIn("__suite_geomean__", role)
        self.assertIn("No aggregate keys", role)

    def test_the_auditor_really_does_what_the_prompt_promises(self):
        """(55)/(127). The prompt now makes three checkable claims about another
        file. Prose about a mechanism is not the mechanism, and a role file is
        the one place in this tree where a false claim is invisible to every
        other test."""
        import audit_floor_sensitivity as AFS
        self.assertIn("ms", AFS.UNIT_TO_MS)
        self.assertIsNotNone(
            AFS.unit_disagrees([0.000027, 0.000027], "ms", 0.027),
            "the prompt promises seconds-under-an-ms-label are caught")
        self.assertIsNone(AFS.declared_unit({}, "samples"),
                          "the prompt tells the engineer to DECLARE the unit, which is only "
                          "meaningful if silence is not read as a declaration")
        source = Path(AFS.__file__).read_text(encoding="utf-8")
        self.assertIn("return 5", source, "the promised exit code is not in the auditor")
