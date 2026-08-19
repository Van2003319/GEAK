#!/usr/bin/env python3
"""GPU-free tests for qd_source_hash.py."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("qd_source_hash.py")
SPEC = importlib.util.spec_from_file_location("qd_source_hash", SCRIPT)
assert SPEC and SPEC.loader
QSH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QSH
SPEC.loader.exec_module(QSH)


def _load_sibling(name: str):
    """Load another script in this directory by path, as SCRIPT is loaded above."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TreeHashTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write(self, relative: str, data: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data, encoding="utf-8")
        return path

    def test_identical_trees_hash_identically(self):
        self.write("a.py", "print(1)\n")
        self.write("sub/b.py", "print(2)\n")
        first = QSH.tree_hash(self.root)
        second = QSH.tree_hash(self.root)
        self.assertEqual(first, second)
        self.assertEqual(64, len(first))

    def test_content_change_changes_hash(self):
        self.write("a.py", "print(1)\n")
        before = QSH.tree_hash(self.root)
        self.write("a.py", "print(2)\n")
        after = QSH.tree_hash(self.root)
        self.assertNotEqual(before, after)

    def test_rename_changes_hash(self):
        self.write("a.py", "print(1)\n")
        before = QSH.tree_hash(self.root)
        (self.root / "a.py").rename(self.root / "renamed.py")
        after = QSH.tree_hash(self.root)
        self.assertNotEqual(before, after)

    def test_generated_artifacts_never_affect_the_hash(self):
        self.write("a.py", "print(1)\n")
        baseline = QSH.tree_hash(self.root)
        self.write("__pycache__/a.cpython-312.pyc", "garbage")
        self.write(".git/HEAD", "ref: refs/heads/main\n")
        self.write("build/out.o", "binary-ish")
        self.write(".torch_ext/ext.so", "binary-ish")
        self.write(".rocprofv3/counters.csv", "generated")
        self.write("logs/benchmark.log", "timing noise")
        self.write("reports/profile.json", "generated")
        self.write("top-level.o", "binary-ish")
        self.write("extension.so", "binary-ish")
        self.write("run.log", "timing noise")
        self.write("generated.hipify.cpp", "generated")
        self.write("worker_result.json", "generated")
        self.assertEqual(baseline, QSH.tree_hash(self.root))

    def test_extra_excluded_dirs_are_respected(self):
        self.write("a.py", "print(1)\n")
        baseline = QSH.tree_hash(self.root)
        self.write("scratch/notes.txt", "not part of the candidate")
        self.assertNotEqual(baseline, QSH.tree_hash(self.root))
        self.assertEqual(baseline, QSH.tree_hash(self.root, extra_excluded_dirs=["scratch"]))

    def test_nested_excluded_dir_is_excluded_too(self):
        self.write("a.py", "print(1)\n")
        self.write("sub/b.py", "print(2)\n")
        baseline = QSH.tree_hash(self.root)
        self.write("sub/__pycache__/x.pyc", "garbage")
        self.assertEqual(baseline, QSH.tree_hash(self.root))

    def test_symlink_is_hashed_by_target_not_followed(self):
        target = self.write("outside.py", "print('outside')\n")
        candidate = self.root / "candidate"
        candidate.mkdir()
        (candidate / "link.py").symlink_to(target)
        first = QSH.tree_hash(candidate)
        # Changing the symlink's target content must not change the tree hash:
        # the link is hashed by its target path text, never dereferenced.
        target.write_text("print('changed')\n", encoding="utf-8")
        second = QSH.tree_hash(candidate)
        self.assertEqual(first, second)

    def test_traversal_order_does_not_matter(self):
        # Two trees built by writing files in a different order must still
        # hash identically -- the walk sorts entries deterministically.
        self.write("z.py", "1")
        self.write("a.py", "2")
        first = QSH.tree_hash(self.root)
        with tempfile.TemporaryDirectory() as other:
            other_root = Path(other)
            (other_root / "a.py").write_text("2", encoding="utf-8")
            (other_root / "z.py").write_text("1", encoding="utf-8")
            second = QSH.tree_hash(other_root)
        self.assertEqual(first, second)

    def test_single_file_root_is_supported(self):
        f = self.write("only.py", "print(1)\n")
        self.assertEqual(64, len(QSH.tree_hash(f)))

    def test_cli_is_deterministic(self):
        self.write("a.py", "print(1)\n")
        run1 = subprocess.run([sys.executable, str(SCRIPT), str(self.root)],
                               text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        run2 = subprocess.run([sys.executable, str(SCRIPT), str(self.root)],
                               text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        self.assertEqual(run1.stdout, run2.stdout)
        payload = json.loads(run1.stdout)
        self.assertEqual("geak.qd-source-hash/v1", payload["schema"])
        self.assertEqual(64, len(payload["source_hash"]))


class DescriptorEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write(self, relative: str, data: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data, encoding="utf-8")
        return path

    def test_grounded_claim_returns_source_quote(self):
        self.write("kernel.py", "acc = tl.dot(a, b, acc)\n")
        evidence = QSH.extract_descriptor_evidence(self.root, ["compute_primitive:native_mfma"])
        found = evidence["compute_primitive:native_mfma"]
        self.assertIsNotNone(found)
        self.assertTrue(found.startswith("source:kernel.py:"))
        self.assertIn("tl.dot", found)

    def test_ungrounded_claim_returns_none_never_fabricated(self):
        self.write("kernel.py", "y = x + 1\n")
        evidence = QSH.extract_descriptor_evidence(self.root, ["compute_primitive:native_mfma"])
        self.assertIsNone(evidence["compute_primitive:native_mfma"])

    def test_claim_with_no_grounding_rule_is_none(self):
        self.write("kernel.py", "tl.dot(a, b)\n")
        evidence = QSH.extract_descriptor_evidence(self.root, ["not_a_real_claim"])
        self.assertIsNone(evidence["not_a_real_claim"])

    def test_metadata_true_grounds_a_claim_without_source_support(self):
        self.write("kernel.py", "y = x + 1\n")
        evidence = QSH.extract_descriptor_evidence(
            self.root, ["decomposition:stream_k"], metadata={"decomposition:stream_k": True})
        self.assertEqual("metadata:decomposition:stream_k=true", evidence["decomposition:stream_k"])

    def test_metadata_false_does_not_ground_and_falls_back_to_source(self):
        self.write("kernel.py", "stream_k = partials_per_cta(m, n, k)\n")
        evidence = QSH.extract_descriptor_evidence(
            self.root, ["decomposition:stream_k"], metadata={"decomposition:stream_k": False})
        self.assertTrue(evidence["decomposition:stream_k"].startswith("source:"))

    def test_metadata_empty_string_does_not_ground(self):
        self.write("kernel.py", "y = 1\n")
        evidence = QSH.extract_descriptor_evidence(
            self.root, ["decomposition:stream_k"], metadata={"decomposition:stream_k": "  "})
        self.assertIsNone(evidence["decomposition:stream_k"])

    def test_multiple_claims_are_independent(self):
        self.write("kernel.py", "tl.dot(a, b)\nping_pong = 1 - ping_pong\n")
        evidence = QSH.extract_descriptor_evidence(
            self.root, ["compute_primitive:native_mfma", "wave_schedule:symmetric_pingpong",
                        "wave_schedule:asymmetric_producer_consumer"])
        self.assertIsNotNone(evidence["compute_primitive:native_mfma"])
        self.assertIsNotNone(evidence["wave_schedule:symmetric_pingpong"])
        self.assertIsNone(evidence["wave_schedule:asymmetric_producer_consumer"])

    def test_binary_files_are_skipped_not_crashed_on(self):
        (self.root / "blob.bin").write_bytes(b"\xff\xfe\x00tl.dot\x00")
        evidence = QSH.extract_descriptor_evidence(self.root, ["compute_primitive:native_mfma"])
        self.assertIsNone(evidence["compute_primitive:native_mfma"])

    def test_excluded_dirs_are_not_scanned_for_evidence(self):
        self.write("__pycache__/cache.py", "tl.dot(a, b)\n")
        evidence = QSH.extract_descriptor_evidence(self.root, ["compute_primitive:native_mfma"])
        self.assertIsNone(evidence["compute_primitive:native_mfma"])


# The prose below is copied from the shipped v59 kernel's rasterization comment.
# It is the strongest available negative fixture precisely because it is real: it
# names XCDs six times, quotes `kGroupM = 8` verbatim, and spells out the remap
# arithmetic -- while arguing that the v58 grouping it describes LOST. Any
# rasterization rule that fires on it grounds a mechanism claim on prose that
# rejects the mechanism. A first draft of both rules did exactly that.
V59_PROSE = """
// v59: XCD-aware grouped rasterization.
// gfx942 dispatches consecutive block ids round-robin over 8 XCDs, each with
// its own L2, so blockIdx adjacency is not L2 adjacency: XCD x receives
// exactly the blocks with pid % 8 == x. v58 grouped kGroupM = 8 rows tall and
// 8 is the XCD count, so `in_group % group_m` collapsed to `pid % 8`.
// So un-shuffle the round-robin first -- p = (pid % 8) * chunk + pid / 8 --
// and only then group. v58 did not lose because grouping is wrong.
"""

V59_CODE = """
constexpr int kGroupM = 8;
constexpr int kXcds = 8;
const int chunk = nblocks / kXcds;
const int p = pid < kXcds * chunk ? (pid % kXcds) * chunk + pid / kXcds : pid;
const int group_m = min(kGroupM, tiles_m - first_m);
"""

LINEAR_CODE = """
// plain linear order; we deliberately do not remap across XCD boundaries.
int row0 = blockIdx.y * CTA_M;
int col0 = blockIdx.x * CTA_N;
"""


class RasterizationEvidenceTest(unittest.TestCase):
    """The rasterization rules must key on arithmetic, not on vocabulary.

    Every other rule in EVIDENCE_PATTERNS is keyword-shaped and would ground on
    a comment. These two are held to the stricter standard because the mechanism
    they name -- the 8-XCD remap -- is the one finding (55) re-legalized, so it
    is the one most likely to be claimed speculatively by a planner that has read
    the ledger and not yet written the code.
    """

    CLAIMS = ["rasterization:xcd_remapped_grouped", "rasterization:grouped_m"]

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def evidence_for(self, source: str) -> dict[str, str | None]:
        (self.root / "kernel.hip").write_text(source, encoding="utf-8")
        return QSH.extract_descriptor_evidence(self.root, self.CLAIMS)

    def test_real_remap_arithmetic_grounds_both_claims(self):
        found = self.evidence_for(V59_CODE)
        for claim in self.CLAIMS:
            with self.subTest(claim=claim):
                self.assertIsNotNone(found[claim], f"{claim} should ground on real v59 code")

    def test_prose_describing_the_mechanism_grounds_nothing(self):
        found = self.evidence_for(V59_PROSE)
        for claim in self.CLAIMS:
            with self.subTest(claim=claim):
                self.assertIsNone(
                    found[claim],
                    f"{claim} grounded on a comment. A comment is prose ABOUT a "
                    f"mechanism, not the mechanism (finding 53).")

    def test_linear_kernel_grounds_neither_even_though_it_says_xcd(self):
        found = self.evidence_for(LINEAR_CODE)
        for claim in self.CLAIMS:
            with self.subTest(claim=claim):
                self.assertIsNone(found[claim])

    def test_grouped_m_alone_does_not_ground_the_remap(self):
        # The weaker claim must not imply the stronger one. A plain grouped
        # swizzle is v58, which measured worse than linear.
        found = self.evidence_for("constexpr int kGroupM = 8;\nconst int group_m = kGroupM;\n")
        self.assertIsNotNone(found["rasterization:grouped_m"])
        self.assertIsNone(found["rasterization:xcd_remapped_grouped"])

    def test_triton_group_size_m_parameter_grounds_grouped_m(self):
        found = self.evidence_for("def kernel(a, b, GROUP_SIZE_M: tl.constexpr):\n    pass\n")
        self.assertIsNotNone(found["rasterization:grouped_m"])


class EvidenceIsCodeNotProseTest(unittest.TestCase):
    """Where a match is allowed to come from.

    Every case here was found by running the helper against a real candidate
    workspace for the first time, not by reasoning about it. Until finding (55)
    wired `evidence` to a caller it had never been pointed at a real tree, so
    every one of these had been latent since the module was written -- which is
    the finding restated: an unreachable gate is also an unexercised one.
    """

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write(self, relative: str, data: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data, encoding="utf-8")
        return path

    def claim(self, name: str, **kwargs) -> str | None:
        return QSH.extract_descriptor_evidence(self.root, [name], **kwargs)[name]

    def test_a_comment_denying_the_mechanism_does_not_ground_it(self):
        # The real one: v59 says "there is nothing to interleave or ping-pong
        # against when the SIMD holds one wave" and the old code read that as
        # evidence the kernel ping-pongs.
        self.write("kernel.hip", "// nothing to ping-pong against with one wave\nint x = 1;\n")
        self.assertIsNone(self.claim("wave_schedule:symmetric_pingpong"))

    def test_block_comments_are_blanked_too(self):
        self.write("kernel.hip", "/* we considered\n   a stream-k split\n   and rejected it */\n")
        self.assertIsNone(self.claim("decomposition:stream_k"))

    def test_code_on_the_same_line_as_a_comment_still_grounds(self):
        self.write("kernel.hip", "int ping_pong = 0;  // toggles the LDS buffer\n")
        self.assertIsNotNone(self.claim("wave_schedule:symmetric_pingpong"))

    def test_hash_is_not_a_comment_in_c_so_preprocessor_lines_still_ground(self):
        # Blanking `#` in C would delete #include/#if, which real rules match.
        self.write("kernel.hip", "#include <rocwmma/rocwmma.hpp>\n")
        self.assertIsNotNone(self.claim("compute_primitive:rocwmma"))

    def test_python_hash_comments_are_blanked(self):
        self.write("kernel.py", "# a prefetch would help here someday\n")
        self.assertIsNone(self.claim("k_pipeline:lds_reg_prefetch"))

    def test_the_quote_shows_original_text_not_blanked_spaces(self):
        self.write("kernel.hip",
                   "// leading note\nc = __builtin_amdgcn_mfma_f32_16x16x16bf16(a, b, c);"
                   "  // trailing note\n")
        found = self.claim("compute_primitive:native_mfma")
        self.assertIn("amdgcn_mfma", found)
        self.assertIn("note", found, "the quote should carry real surrounding context")

    def test_documentation_never_grounds_a_mechanism_claim(self):
        # A README saying the project uses MFMA is not evidence that this
        # kernel issues one. This fired on the first real workspace tried.
        self.write("README.md", "This project uses MFMA intrinsics throughout.\n")
        self.assertIsNone(self.claim("compute_primitive:native_mfma"))

    def test_documentation_still_counts_toward_the_tree_hash(self):
        # Excluded from evidence, not from identity: editing docs is a source change.
        self.write("README.md", "a\n")
        before = QSH.tree_hash(self.root)
        self.write("README.md", "b\n")
        self.assertNotEqual(before, QSH.tree_hash(self.root))

    def test_abandoned_variants_ground_claims_unless_scoped_out(self):
        # The other real one: an experimental file the build never compiles
        # happily substantiates a claim about the shipped kernel.
        self.write("src/custom_gemm.hip", "int x = 1;\n")
        self.write("research/experimental/old_try.hip", "auto v = __builtin_amdgcn_mfma_f32(a, b);\n")
        self.assertIsNotNone(self.claim("compute_primitive:native_mfma"),
                             "unscoped, the abandoned variant grounds it")
        self.assertIsNone(self.claim("compute_primitive:native_mfma", scope=["src/"]),
                          "scoped to what builds, the claim is correctly unsubstantiated")

    def test_a_saved_snapshot_still_gets_its_comments_blanked(self):
        # src/custom_gemm.hip.v100_m512waves8 is a real file. Its final suffix
        # matches no comment style, so the naive version blanked nothing in
        # exactly the files most likely to be dead variants.
        self.write("src/custom_gemm.hip.v100_m512waves8",
                   "// nothing to ping-pong against with one wave\nint x = 1;\n")
        self.assertIsNone(self.claim("wave_schedule:symmetric_pingpong"))

    def test_a_data_file_never_grounds_a_claim(self):
        # research/index.json lists approaches "considered_or_rejected"; its
        # rejected list grounded native_mfma. JSON has no comments to blank, so
        # exclusion is the only defence.
        self.write("research/index.json",
                   '{"rejected": ["guessed raw MFMA lane mappings"]}\n')
        self.assertIsNone(self.claim("compute_primitive:native_mfma"))

    def test_scoping_to_a_file_excludes_its_snapshots(self):
        # Prefix matching would pull the snapshot back in through the very flag
        # the caller used to exclude it.
        self.write("src/custom_gemm.hip", "int x = 1;\n")
        self.write("src/custom_gemm.hip.v100", "auto v = __builtin_amdgcn_mfma_f32(a, b);\n")
        self.assertIsNone(self.claim("compute_primitive:native_mfma",
                                     scope=["src/custom_gemm.hip"]))
        self.assertIsNotNone(self.claim("compute_primitive:native_mfma", scope=["src"]))

    def test_scope_matches_a_directory_with_or_without_a_trailing_slash(self):
        self.write("src/a.hip", "acc = tl.dot(a, b);\n")
        for spec in ("src", "src/"):
            with self.subTest(scope=spec):
                self.assertIsNotNone(
                    self.claim("compute_primitive:native_mfma", scope=[spec]))

    def test_scope_does_not_match_a_sibling_sharing_a_name_prefix(self):
        self.write("src_old/a.hip", "acc = tl.dot(a, b);\n")
        self.assertIsNone(self.claim("compute_primitive:native_mfma", scope=["src"]))

    def test_scope_accepts_several_prefixes(self):
        self.write("src/a.hip", "int x = 1;\n")
        self.write("kernels/b.hip", "acc = tl.dot(a, b);\n")
        self.assertIsNone(self.claim("compute_primitive:native_mfma", scope=["src/"]))
        self.assertIsNotNone(
            self.claim("compute_primitive:native_mfma", scope=["src/", "kernels/"]))


class UngroundableInventoryTest(unittest.TestCase):
    """UNGROUNDABLE_CLAIMS is documentation with a test behind it.

    Its only job is to let a reader tell "no rule yet" from "no rule possible".
    That distinction is worthless if an entry can quietly contradict the rule
    table or name an axis value that does not exist.
    """

    def test_no_claim_is_both_groundable_and_ungroundable(self):
        overlap = set(QSH.EVIDENCE_PATTERNS) & set(QSH.UNGROUNDABLE_CLAIMS)
        self.assertEqual(set(), overlap,
                         f"these claims claim both a rule and no rule: {sorted(overlap)}")

    def test_every_entry_names_a_real_axis_value(self):
        descriptor = _load_sibling("qd_descriptor_v2")
        legal = {f"{axis}:{value}"
                 for axis, values in descriptor.QD_VOCAB.items() for value in values}
        for claim in sorted(set(QSH.UNGROUNDABLE_CLAIMS) | set(QSH.EVIDENCE_PATTERNS)):
            with self.subTest(claim=claim):
                self.assertIn(claim, legal,
                              f"{claim} is not an axis:value in the v2 vocabulary")

    def test_every_vocabulary_tuple_is_accounted_for(self):
        # The point of the inventory: a new axis value cannot slip in with
        # neither a rule nor a written reason for having none.
        descriptor = _load_sibling("qd_descriptor_v2")
        legal = {f"{axis}:{value}"
                 for axis, values in descriptor.QD_VOCAB.items() for value in values}
        missing = sorted(legal - set(QSH.EVIDENCE_PATTERNS) - set(QSH.UNGROUNDABLE_CLAIMS))
        self.assertEqual([], missing,
                         f"add a grounding rule or an UNGROUNDABLE_CLAIMS reason for: {missing}")

    def test_reasons_are_written_not_placeholders(self):
        for claim, reason in sorted(QSH.UNGROUNDABLE_CLAIMS.items()):
            with self.subTest(claim=claim):
                self.assertGreater(len(reason.split()), 3,
                                   f"{claim} has no real reason recorded")


class UnderscoreIdentifierEvidenceTest(unittest.TestCase):
    """Finding (94): the extractor could not read the identifiers code uses.

    `_` is a word character, so `\\b` does not break inside `prefetch_stage` or
    `splitk_reduce_kernel`. Keyword rules therefore missed every axis whose
    implementation is named in snake_case -- while matching the same words in a
    comment perfectly well. The extractor was blind to code and open to prose,
    which is exactly backwards from what (53) established it should be.

    Both directions are pinned in every case below. A suite that only asserts
    the new hits is satisfied by deleting the word boundaries entirely.
    """

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def claim(self, name: str, source: str) -> str | None:
        (self.root / "kernel.hip").write_text(source, encoding="utf-8")
        return QSH.extract_descriptor_evidence(self.root, [name])[name]

    # --- the false negatives the fix exists to close ------------------------

    def test_a_snake_case_kernel_name_grounds_its_axis(self):
        self.assertIsNotNone(self.claim(
            "decomposition:split_k",
            "__global__ void splitk_reduce_kernel(float* c, int n) {}\n"))

    def test_a_snake_case_lambda_name_grounds_its_axis(self):
        self.assertIsNotNone(self.claim(
            "k_pipeline:lds_reg_prefetch",
            "auto prefetch_stage = [&](int kbase) { load(kbase); };\n"))

    def test_a_snake_case_helper_call_grounds_its_axis(self):
        self.assertIsNotNone(self.claim(
            "output_path:lds_staged_store",
            "store_panel<CTA_M, STAGE_K>(as[0], tid, a_reg);\n"))

    # --- the four rules the substitution would destroy ----------------------
    #
    # The retry is an OR against a spaced copy, never a replacement of the
    # original search. These four match constructs that only exist WITH their
    # underscores, so they are the reason it cannot be a substitution.

    def test_double_underscore_keywords_still_ground(self):
        self.assertIsNotNone(self.claim(
            "k_pipeline:lds_single", "__shared__ __align__(16) short tile[512];\n"))

    def test_a_builtin_with_underscores_still_grounds(self):
        self.assertIsNotNone(self.claim(
            "compute_primitive:native_mfma",
            "acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);\n"))

    def test_an_underscored_constant_still_grounds(self):
        self.assertIsNotNone(self.claim(
            "rasterization:grouped_m", "constexpr int kGroupM = 8;\n"))

    def test_a_modulo_expression_still_grounds(self):
        self.assertIsNotNone(self.claim(
            "rasterization:xcd_remapped_grouped", "int xcd = pid % kXcds;\n"))

    # --- the false positive the narrowed mfma rule closes -------------------

    def test_a_guard_macro_name_is_not_evidence_of_an_mfma_issue(self):
        """`#define GEAK_HAS_BF16_MFMA 1` says a build supports MFMA. It does
        not say this kernel issues one, and under the spaced-copy retry the old
        `\\bmfma\\b` alternative grounded on it."""
        self.assertIsNone(self.claim(
            "compute_primitive:native_mfma",
            "#define GEAK_HAS_BF16_MFMA 1\n#if !defined(QD_MFMA_ARCH)\n#endif\n"))

    def test_the_real_mnemonic_still_grounds_after_narrowing(self):
        """`v_mfma[a-z0-9_]*`, not `\\bv_mfma\\b`: the mnemonic is
        `v_mfma_f32_16x16x16bf16_1k`, and a trailing boundary would demand a
        non-word character exactly where `_f32` begins -- (94) reintroduced
        inside its own fix."""
        self.assertIsNotNone(self.claim(
            "compute_primitive:native_mfma",
            'asm volatile("v_mfma_f32_16x16x16bf16_1k %0, %1, %2, %0" : "+v"(acc));\n'))

    # --- the (53) fixes must survive the widening ---------------------------

    def test_prose_about_a_mechanism_still_does_not_ground_it(self):
        self.assertIsNone(self.claim(
            "k_pipeline:lds_reg_prefetch",
            "// a prefetch_stage helper would hide the latency; not implemented\n"))

    # --- the labelled hole (54) ---------------------------------------------

    def test_camel_case_remains_a_stated_absence(self):
        """No length-preserving substitution splits a camel hump, so camelCase
        identifiers stay invisible to keyword rules. Asserted as a MISS so the
        hole is stated rather than discovered. If this ever starts failing the
        hole was closed -- update finding (94) and delete this deliberately."""
        self.assertIsNone(self.claim(
            "wave_schedule:symmetric_pingpong", "constexpr int kPingPongStages = 2;\n"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
