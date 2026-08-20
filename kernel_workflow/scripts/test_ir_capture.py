#!/usr/bin/env python3
"""Tests for `ir_capture.py`, with no compiler and no GPU in the room.

Every external effect is injected: the compiler is a `runner` that returns
fabricated traces, and the build file is written by the test. That is not a
convenience -- it is the only way to exercise the paths that matter here, which
are all refusals. A capture that picks the wrong translation unit, or attributes
a trajectory it never tied to the measured binary, produces a real and
well-formed archive of the wrong program, and no downstream reader can tell. So
the assertions below are mostly about what this module declines to do.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ir_capture  # noqa: E402


NINJA = """ninja_required_version = 1.3
cxx = c++
nvcc = /opt/rocm/bin/hipcc

cflags = -DTORCH_EXTENSION_NAME=ext -O3
post_cflags = -D__HIP_PLATFORM_AMD__=1 -fPIC
cuda_cflags = -DWITH_HIP -O3 -D__HIP_PLATFORM_AMD__=1 -std=c++17 --offload-arch=gfx942 -fno-gpu-rdc -O3
cuda_post_cflags =
ldflags = -shared -lamdhip64

rule compile
  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags

rule cuda_compile
  command = $nvcc  $cuda_cflags -c $in -o $out $cuda_post_cflags

build custom_gemm_hip.cuda.o: cuda_compile /t/src/custom_gemm_hip.hip
build dense_bf16_gemm_hip.cuda.o: cuda_compile /t/src/dense_bf16_gemm_hip.hip
build gemm_bindings.o: compile /t/src/gemm_bindings.cpp

build ext.so: link custom_gemm_hip.cuda.o dense_bf16_gemm_hip.cuda.o gemm_bindings.o
"""

TRACE = """*** IR Dump After SROAPass on _Z1kPfPKfi ***
define void @_Z1kPfPKfi() {
  %1 = load float, ptr %p, align 4
  ret void
}
*** IR Dump After AMDGPU DAG->DAG Pattern Instruction Selection (amdgpu-isel) on _Z1kPfPKfi ***
# Machine code for function _Z1kPfPKfi: IsSSA, TracksLiveness

bb.0:
  %0:vgpr_32 = GLOBAL_LOAD_DWORDX4 %1:vreg_64, 0, 0
  S_ENDPGM 0
*** IR Dump After SI insert wait instructions (si-insert-waitcnts) on _Z1kPfPKfi ***
# Machine code for function _Z1kPfPKfi: NoPHIs, TracksLiveness

bb.0:
  %0:vgpr_32 = GLOBAL_LOAD_DWORDX4 %1:vreg_64, 0, 0
  S_WAITCNT 0
  S_ENDPGM 0
"""


def fake_runner(trace: str = TRACE, fail: tuple[str, ...] = ()):
    """A compiler that writes whatever the test says and records every command."""
    calls: list[list[str]] = []

    def run(cmd, cwd=None, env=None):
        calls.append(list(cmd))
        joined = " ".join(cmd)
        for needle in fail:
            if needle in joined:
                return 1, "", f"fabricated failure for {needle}"
        if "--version" in cmd:
            return 0, "AMD clang version 22.0.0git\nTarget: x86_64\n", ""
        # `-o PATH` outputs are created so the caller's existence checks are real.
        if "-o" in cmd:
            out = Path(cmd[cmd.index("-o") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("fabricated\n", encoding="utf-8")
        if "-print-changed=quiet" in joined:
            return 0, "", trace
        return 0, "", ""

    run.calls = calls  # type: ignore[attr-defined]
    return run


class ParseNinjaTest(unittest.TestCase):
    def test_reads_the_device_compiler_and_flags_verbatim(self):
        parsed = ir_capture.parse_ninja(NINJA)
        self.assertEqual(parsed["compiler"], "/opt/rocm/bin/hipcc")
        self.assertIn("--offload-arch=gfx942", parsed["cuda_cflags"])
        self.assertIn("-fno-gpu-rdc", parsed["cuda_cflags"])

    def test_host_edges_are_not_device_edges(self):
        """A `compile` edge carries no device code. Replaying one would produce an
        empty trajectory, which reads exactly like a kernel with nothing in it."""
        sources = [Path(e["source"]).name for e in ir_capture.parse_ninja(NINJA)["edges"]]
        self.assertEqual(sources, ["custom_gemm_hip.hip", "dense_bf16_gemm_hip.hip"])
        self.assertNotIn("gemm_bindings.cpp", sources)


class SelectEdgeTest(unittest.TestCase):
    def setUp(self):
        self.edges = ir_capture.parse_ninja(NINJA)["edges"]

    def test_the_edited_file_resolves_to_the_hipified_twin(self):
        """An engineer edits `X.hip`; ninja compiles torch's `X_hip.hip`. Both
        spellings must reach the same edge, or the caller has to know about a
        rename it never performed."""
        edge, holes = ir_capture.select_edge(self.edges, "dense_bf16_gemm.hip")
        self.assertEqual(holes, [])
        self.assertTrue(edge["source"].endswith("dense_bf16_gemm_hip.hip"))

    def test_the_twin_spelling_also_resolves(self):
        edge, _ = ir_capture.select_edge(self.edges, "dense_bf16_gemm_hip.hip")
        self.assertTrue(edge["source"].endswith("dense_bf16_gemm_hip.hip"))

    def test_several_device_units_with_no_selector_is_a_refusal(self):
        edge, holes = ir_capture.select_edge(self.edges, None)
        self.assertIsNone(edge)
        self.assertTrue(holes[0].startswith("ninja:ambiguous_edge"))

    def test_a_single_device_unit_needs_no_selector(self):
        edge, holes = ir_capture.select_edge(self.edges[:1], None)
        self.assertEqual(holes, [])
        self.assertTrue(edge["source"].endswith("custom_gemm_hip.hip"))

    def test_an_unknown_source_names_what_the_build_does_compile(self):
        edge, holes = ir_capture.select_edge(self.edges, "not_here.hip")
        self.assertIsNone(edge)
        self.assertIn("custom_gemm_hip.hip", holes[0])


class FindNinjaTest(unittest.TestCase):
    """The candidate/oracle split, which is the ambiguity that actually occurs.

    `dense_bf16_gemm_fused` builds two extensions: the candidate and the
    immutable rocBLAS oracle. Tracing the oracle yields a real trajectory of a
    program nobody is optimizing, so the tie has to break on a fact.
    """

    def _tree(self, root: Path):
        (root / ".torch_ext" / "cand").mkdir(parents=True)
        (root / ".torch_ext" / "cand" / "build.ninja").write_text(NINJA, encoding="utf-8")
        (root / ".torch_ext" / "oracle").mkdir(parents=True)
        (root / ".torch_ext" / "oracle" / "build.ninja").write_text(
            NINJA.replace("/t/src/dense_bf16_gemm_hip.hip", "/t/src/rocblas_baseline_hip.hip")
                 .replace("/t/src/custom_gemm_hip.hip", "/t/src/rocblas_bindings_hip.hip"),
            encoding="utf-8")
        (root / "src").mkdir()

    def test_the_source_file_picks_the_build_that_compiles_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._tree(root)
            found, holes = ir_capture.find_ninja(root / "src", "dense_bf16_gemm.hip")
            self.assertEqual(holes, [])
            self.assertEqual(found.parent.name, "cand")

    def test_no_selector_refuses_and_says_why(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._tree(root)
            found, holes = ir_capture.find_ninja(root / "src", None)
            self.assertIsNone(found)
            self.assertIn("oracle", holes[0])

    def test_an_unbuilt_tree_is_a_named_hole(self):
        with tempfile.TemporaryDirectory() as tmp:
            found, holes = ir_capture.find_ninja(Path(tmp) / "src", "x.hip")
            self.assertIsNone(found)
            self.assertTrue(holes[0].startswith("ninja:absent"))


class SplitStagesTest(unittest.TestCase):
    def test_order_identity_and_scope(self):
        stages, holes = ir_capture.split_stages(TRACE)
        self.assertEqual(holes, [])
        self.assertEqual([s["index"] for s in stages], [0, 1, 2])
        self.assertEqual([s["pass_id"] for s in stages],
                         ["SROAPass", "amdgpu-isel", "si-insert-waitcnts"])
        self.assertEqual(stages[0]["scope"], "_Z1kPfPKfi")

    def test_a_legacy_pass_is_identified_by_its_flag_not_its_prose(self):
        """The human name has been reworded between LLVM releases while the flag
        stayed put, and the flag is what a reader greps for."""
        stages, _ = ir_capture.split_stages(TRACE)
        self.assertEqual(stages[1]["pass_id"], "amdgpu-isel")
        self.assertIn("DAG->DAG", stages[1]["pass_name"])

    def test_a_pass_argument_keeps_its_capitals(self):
        """`post-RA-hazard-rec` is the real spelling of a real pass. A
        lowercase-only argument class fell through to the prose name for exactly
        those passes, so one pass got two identities depending on its case."""
        stages, _ = ir_capture.split_stages(
            "*** IR Dump After Post RA hazard recognizer (post-RA-hazard-rec) on f ***\nb\n")
        self.assertEqual(stages[0]["pass_id"], "post-RA-hazard-rec")

    def test_a_pass_name_containing_on_is_not_split_at_the_wrong_word(self):
        text = "*** IR Dump After Insert waits on memory (si-waits) on _Z1k ***\nbody\n"
        stages, _ = ir_capture.split_stages(text)
        self.assertEqual(stages[0]["scope"], "_Z1k")
        self.assertEqual(stages[0]["pass_id"], "si-waits")

    def test_text_before_the_first_banner_is_reported_not_dropped(self):
        """If the flags were rejected there are no banners at all, so a leading
        diagnostic is the evidence that the capture is empty for a reason."""
        stages, holes = ir_capture.split_stages(
            "clang: error: unknown argument '-print-changed'\n" + TRACE)
        self.assertEqual(len(stages), 3)
        self.assertTrue(any(h.startswith("trace:preamble") for h in holes))
        self.assertIn("unknown argument", holes[0])

    def test_an_unfiltered_trace_is_truncated_and_says_so(self):
        text = "".join(f"*** IR Dump After P{i} on f ***\nbody\n"
                       for i in range(ir_capture.MAX_STAGES + 5))
        stages, holes = ir_capture.split_stages(text)
        self.assertEqual(len(stages), ir_capture.MAX_STAGES)
        self.assertTrue(any(h.startswith("trace:truncated") for h in holes))
        self.assertIn(str(ir_capture.MAX_STAGES), holes[0])


class StageFileTest(unittest.TestCase):
    def test_machine_ir_and_llvm_ir_get_different_extensions(self):
        """`ir_signals.py` counts different vocabularies in the two, and counting
        one with the other's reader would make instruction selection look like a
        pass that deleted every load."""
        self.assertEqual(ir_capture.stage_extension("define void @f() {"), ".ll")
        self.assertEqual(
            ir_capture.stage_extension("# Machine code for function f: IsSSA"), ".mir")

    def test_files_are_written_in_trajectory_order_with_a_safe_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            stages, _ = ir_capture.split_stages(TRACE)
            written = ir_capture.write_stages(Path(tmp), stages)
            names = [w["file"] for w in written]
            self.assertEqual(names, ["stages/000-SROAPass.ll",
                                     "stages/001-amdgpu-isel.mir",
                                     "stages/002-si-insert-waitcnts.mir"])
            self.assertNotIn("body", written[0])
            for entry in written:
                self.assertTrue((Path(tmp) / entry["file"]).is_file())


class TwinOriginTest(unittest.TestCase):
    def test_the_edited_file_is_recorded_beside_the_compiled_one(self):
        self.assertEqual(ir_capture.twin_origin("/s/dense_hip.hip"), "/s/dense.hip")

    def test_a_file_that_is_not_a_twin_has_no_origin(self):
        self.assertIsNone(ir_capture.twin_origin("/s/plain.hip"))


class CaptureTest(unittest.TestCase):
    def _workspace(self, tmp: Path) -> Path:
        src = tmp / "src"
        src.mkdir(parents=True)
        (src / "dense_bf16_gemm.hip").write_text("// candidate\n", encoding="utf-8")
        ext = tmp / ".torch_ext" / "cand"
        ext.mkdir(parents=True)
        (ext / "build.ninja").write_text(NINJA, encoding="utf-8")
        return src

    def test_a_capture_with_no_isa_archive_is_partial_not_ok(self):
        """The whole liability of this module is that it recompiles. An archive
        that was never tied to the measured binary must not exit 0, or a caller
        reads `exit_code == 0` as licence to attribute the plateau to it."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = self._workspace(root)
            manifest = ir_capture.capture(
                root / "arch", src, backend="hip", source_file="dense_bf16_gemm.hip",
                runner=fake_runner())
            self.assertEqual(manifest["exit_code"], ir_capture.EXIT_PARTIAL)
            self.assertFalse(manifest["provenance"]["ir_binary_equals_measured"])
            self.assertTrue(any(h.startswith("provenance:unproven")
                                for h in manifest["holes"]))

    def test_the_replayed_command_is_the_build_s_own_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = self._workspace(root)
            runner = fake_runner()
            manifest = ir_capture.capture(
                root / "arch", src, backend="hip", source_file="dense_bf16_gemm.hip",
                kernel="_Z1kPfPKfi", runner=runner)
            trace_cmd = manifest["commands"]["trace"]
            self.assertIn("--offload-arch=gfx942", trace_cmd)
            self.assertIn("-fno-gpu-rdc", trace_cmd)
            self.assertIn("--cuda-device-only", trace_cmd)
            self.assertIn("-filter-print-funcs=_Z1kPfPKfi", trace_cmd)
            self.assertEqual(manifest["compiled_source"],
                             "/t/src/dense_bf16_gemm_hip.hip")
            self.assertEqual(manifest["edited_source"], "/t/src/dense_bf16_gemm.hip")

    def test_the_provenance_build_carries_no_evidence_flags(self):
        """Comparing the measured binary against an object this module perturbed
        would prove the perturbation, not the identity."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = self._workspace(root)
            (root / "isa").mkdir()
            (root / "isa" / "manifest.json").write_text("{}", encoding="utf-8")
            manifest = ir_capture.capture(
                root / "arch", src, backend="hip", source_file="dense_bf16_gemm.hip",
                isa_archive=root / "isa", runner=fake_runner())
            probe = manifest["commands"].get("provenance_build", "")
            self.assertIn("-c", probe)
            self.assertNotIn("print-changed", probe)
            self.assertNotIn("filter-print-funcs", probe)

    def test_output_paths_are_absolute_because_the_replay_runs_elsewhere(self):
        """Every replayed command runs with `cwd` in the scratch tree, so a
        relative `--out` makes the compiler's own `-o` fail with a path error
        that reads like a permissions problem. Caught on the real task, where it
        cost the unoptimized reference the first stage is diffed against."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = self._workspace(root)
            import os
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                manifest = ir_capture.capture(
                    Path("relative_arch"), src, backend="hip",
                    source_file="dense_bf16_gemm.hip", runner=fake_runner())
            finally:
                os.chdir(cwd)
            for name, command in manifest["commands"].items():
                if " -o " in command:
                    target = command.split(" -o ", 1)[1].split()[0]
                    self.assertTrue(target.startswith("/"),
                                    f"{name} writes to a relative path {target!r}")
            self.assertNotIn("frontend:failed", " ".join(manifest["holes"]))

    def test_stages_and_source_hash_reach_the_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = self._workspace(root)
            manifest = ir_capture.capture(
                root / "arch", src, backend="hip", source_file="dense_bf16_gemm.hip",
                runner=fake_runner())
            self.assertEqual(manifest["stage_count"], 3)
            self.assertTrue(manifest["source_hash"])
            on_disk = json.loads((root / "arch" / "manifest.json").read_text())
            self.assertEqual(on_disk["stage_count"], 3)

    def test_a_failed_trace_is_a_hole_and_writes_no_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = self._workspace(root)
            manifest = ir_capture.capture(
                root / "arch", src, backend="hip", source_file="dense_bf16_gemm.hip",
                runner=fake_runner(fail=("print-changed",)))
            self.assertEqual(manifest["exit_code"], ir_capture.EXIT_HOLE)
            self.assertEqual(manifest.get("stage_count", 0), 0)

    def test_an_empty_trace_is_a_hole_not_an_optimal_kernel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = self._workspace(root)
            manifest = ir_capture.capture(
                root / "arch", src, backend="hip", source_file="dense_bf16_gemm.hip",
                runner=fake_runner(trace=""))
            self.assertEqual(manifest["exit_code"], ir_capture.EXIT_HOLE)
            self.assertTrue(any("capture:nothing" in h for h in manifest["holes"]))

    def test_an_existing_archive_is_immutable_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = self._workspace(root)
            (root / "arch").mkdir()
            (root / "arch" / "occupied").write_text("x", encoding="utf-8")
            manifest = ir_capture.capture(root / "arch", src, backend="hip",
                                          source_file="dense_bf16_gemm.hip",
                                          runner=fake_runner())
            self.assertEqual(manifest["exit_code"], ir_capture.EXIT_ERROR)
            self.assertIn("archive:not_empty", manifest["holes"][0])

    def test_an_unsupported_backend_is_reported_not_degraded(self):
        """"No adapter" and "no trajectory" are different facts, and an empty
        archive would state the second while meaning the first."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = self._workspace(root)
            manifest = ir_capture.capture(root / "arch", src, backend="cuda",
                                          runner=fake_runner())
            self.assertEqual(manifest["exit_code"], ir_capture.EXIT_ERROR)
            self.assertTrue(any("backend:unsupported" in h for h in manifest["holes"]))


class TritonBackendTest(unittest.TestCase):
    def test_no_command_is_a_hole_rather_than_a_guess(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = ir_capture.capture(Path(tmp) / "arch", Path(tmp), backend="triton",
                                          runner=fake_runner())
            self.assertEqual(manifest["exit_code"], ir_capture.EXIT_HOLE)
            self.assertTrue(any("triton:no_command" in h for h in manifest["holes"]))

    def test_the_cache_is_isolated_and_recompilation_forced(self):
        """A cache hit emits no pass banners, and an empty trajectory then reads
        as a kernel whose passes changed nothing."""
        seen: dict = {}

        def run(cmd, cwd=None, env=None):
            if env and "TRITON_DUMP_DIR" in env:
                seen.update(env)
            return 0, "", ""

        with tempfile.TemporaryDirectory() as tmp:
            ir_capture.capture(Path(tmp) / "arch", Path(tmp), backend="triton",
                               triton_command="python bench.py", runner=run)
        self.assertEqual(seen.get("TRITON_ALWAYS_COMPILE"), "1")
        self.assertEqual(seen.get("MLIR_ENABLE_DUMP"), "1")
        self.assertTrue(seen.get("TRITON_CACHE_DIR", "").endswith("triton-cache"))

    def test_several_dumped_kernels_with_no_selector_is_a_hole(self):
        mlir = ("// -----// IR Dump Before Foo (foo) ('builtin.module' operation) //----- //\n"
                "module {\n}\n")

        def run(cmd, cwd=None, env=None):
            if not env or "TRITON_DUMP_DIR" not in env:
                return 0, "", ""  # toolchain probes carry no environment
            dump = Path(env["TRITON_DUMP_DIR"])
            for name in ("kernel_a", "kernel_b"):
                (dump / name).mkdir(parents=True, exist_ok=True)
            return 0, "", mlir

        with tempfile.TemporaryDirectory() as tmp:
            manifest = ir_capture.capture(Path(tmp) / "arch", Path(tmp), backend="triton",
                                          triton_command="python bench.py", runner=run)
        self.assertTrue(any("triton:ambiguous_kernel" in h for h in manifest["holes"]))

    def test_the_mlir_banner_is_parsed_into_stages(self):
        stages, holes = ir_capture.split_mlir_stages(
            "// -----// IR Dump Before TritonAMDGPUAccelerateMatmul "
            "(tritonamdgpu-accelerate-matmul) ('builtin.module' operation) //----- //\n"
            "module {\n}\n")
        self.assertEqual(holes, [])
        self.assertEqual(stages[0]["pass_id"], "tritonamdgpu-accelerate-matmul")
        self.assertEqual(stages[0]["when"], "Before")


class ProvenanceTest(unittest.TestCase):
    """`compare_to_measured` against a stubbed `isa_signals`.

    Stubbed rather than fixtured because the point under test is the SCOPING
    rule, not the disassembler: which kernels are compared, and which of the
    four possible outcomes counts as proof.
    """

    def setUp(self):
        self._saved = sys.modules.get("isa_signals")

        class Stub:
            data: dict = {}

            @staticmethod
            def build_signals(archive):
                return Stub.data[str(archive)]

            @staticmethod
            def _identical(a, b):
                return a == b

        self.stub = Stub
        sys.modules["isa_signals"] = Stub  # type: ignore[assignment]

    def tearDown(self):
        if self._saved is None:
            sys.modules.pop("isa_signals", None)
        else:
            sys.modules["isa_signals"] = self._saved

    def _set(self, probe, measured):
        self.stub.data = {"/probe": {"kernels": probe}, "/measured": {"kernels": measured}}

    def test_kernels_only_in_the_measured_binary_are_not_drift(self):
        """The measured artifact is a `.so` linked from every translation unit;
        the probe compiles one. Treating the difference as drift would make
        provenance unprovable by construction."""
        self._set([{"name": "hot", "op": 1}],
                  [{"name": "hot", "op": 1}, {"name": "other_tu", "op": 9}])
        result = ir_capture.compare_to_measured(Path("/probe"), Path("/measured"), "hot")
        self.assertTrue(result["ir_binary_equals_measured"])
        self.assertEqual(result["measured_only"], ["other_tu"])

    def test_a_drifted_shared_kernel_refuses(self):
        self._set([{"name": "hot", "op": 1}], [{"name": "hot", "op": 2}])
        result = ir_capture.compare_to_measured(Path("/probe"), Path("/measured"), "hot")
        self.assertFalse(result["ir_binary_equals_measured"])
        self.assertEqual(result["drifted_kernels"], ["hot"])

    def test_no_overlap_is_not_agreement(self):
        """Nothing compared must never read as nothing changed."""
        self._set([{"name": "a", "op": 1}], [{"name": "b", "op": 1}])
        result = ir_capture.compare_to_measured(Path("/probe"), Path("/measured"), None)
        self.assertFalse(result["ir_binary_equals_measured"])
        self.assertIn("nothing was compared", result["reason"])

    def test_the_selected_kernel_must_itself_be_present_in_both(self):
        self._set([{"name": "hot", "op": 1}, {"name": "cold", "op": 1}],
                  [{"name": "cold", "op": 1}])
        result = ir_capture.compare_to_measured(Path("/probe"), Path("/measured"), "hot")
        self.assertFalse(result["ir_binary_equals_measured"])
        self.assertIn("'hot'", result["reason"])

    def test_a_missing_identity_predicate_fails_loudly(self):
        """If `isa_signals` stops exposing the rule this check is defined
        against, provenance must break visibly rather than compare nothing."""
        del self.stub._identical
        result = ir_capture.compare_to_measured(Path("/probe"), Path("/measured"), None)
        self.assertFalse(result["ir_binary_equals_measured"])
        self.assertIn("identity", result["reason"])


if __name__ == "__main__":
    unittest.main()
