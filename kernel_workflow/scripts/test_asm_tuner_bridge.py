"""Static contract tests for the canonical asm_tuner lane bridge."""
import json
from pathlib import Path

import lane_args


ROOT = Path(__file__).resolve().parents[2]
LANE = (ROOT / "kernel_workflow/kernel_lane.js").read_text()
VERIFY = (ROOT / "kernel_workflow/roles/verify_engineer.md").read_text()


def test_prepare_is_a_hard_barrier_before_engineer():
    prepare = LANE.index("const prepareTail")
    engineer = LANE.index("const eng = await agentT", prepare)
    assert prepare < engineer
    window = LANE[prepare:engineer]
    assert "gpu_lock.sh" in window and "gpu_fence_run.sh" in window
    assert " prepare " in window
    assert "prepared_before_engineer" in LANE


def test_verifier_rebuilds_loads_and_smokes_materialization():
    for command in ("materialize", "verify-materialization", "--rebuild", "--in-place", "smoke"):
        assert command in VERIFY
    for identity in ("task digest", "CASES", "input builder", "correctness function",
                     "protocol", "oracle", "artifact", "generated-source"):
        assert identity.lower() in VERIFY.lower()


def test_bf16_and_fp8_48h_lanes_require_the_bridge():
    for name, task in (
        ("asmopt_bf16_20260827.json", "dense_bf16_gemm_fused"),
        ("fp8_rowwise_20260827.json", "dense_fp8_gemm_rowwise"),
    ):
        path = ROOT / "kernel_workflow/lanes" / name
        assert lane_args.check(path) == []
        args, required = lane_args.load(path)
        assert args["asm_tuner_mode"] == required["asm_tuner_mode"] == "required"
        assert args["asm_tuner_manifest"] == required["asm_tuner_manifest"]
        assert task in args["asm_tuner_manifest"]
        assert args["kernel_path"].startswith("/home/yxh/GEAK_tunerint/")


def test_fence_exposes_a_machine_checkable_boundary():
    fence = (ROOT / "kernel_workflow/scripts/gpu_fence_run.sh").read_text()
    assert "export GEAK_GPU_FENCE_HELD=1" in fence


def test_canonical_tuner_is_materialized_before_baseline_and_refreshed_after_commit():
    bootstrap = LANE.index("let canonicalTunerPreparation")
    benchmark = LANE.index("PHASE: Benchmark")
    assert bootstrap < benchmark
    bootstrap_window = LANE[bootstrap:benchmark]
    assert "asm_tuner: publish canonical suite contract" in bootstrap_window
    for published in (
        "asm_tuner_registry.json", "asm_tuner_runtime_table.json",
        "asm_tuner_selection.json",
    ):
        assert published in bootstrap_window
    commit_refresh = LANE.index("config_search:refresh canonical")
    reprofile = LANE.index("profile_engineer', 'reprofile", commit_refresh)
    assert commit_refresh < reprofile
