import json

from asm_tuner.integration import _search_round_cache


def test_protocol_v2_round_cache_is_resumable_and_artifact_bound(tmp_path):
    path = tmp_path / "round.json"
    search_id = "a" * 64
    protocol_hash = "b" * 64
    score_metric = {"schema": "geak.score-metric/v1", "version": 1,
                    "id": "sustained_throughput", "inner": 16,
                    "protocol_hash": protocol_hash, "hash": "c" * 64}
    candidates = ["cfg-a", "cfg-b", "cfg-c"]
    fingerprints = {
        name: {"artifact_sha256": str(index + 1) * 64, "isa_hash": str(index + 4) * 64}
        for index, name in enumerate(candidates)
    }
    value = {
        "schema": "geak.asm-canonical-search-round/v2",
        "search_id": search_id,
        "route": "large_m",
        "round": 0,
        "candidate_ids": candidates,
        "fingerprints": fingerprints,
        "protocol_hash": protocol_hash,
        "score_metric": score_metric,
        "correctness": {name: {"passed": True} for name in candidates},
        "receipt": {
            "protocol_hash": protocol_hash,
            "metric_id": "sustained_throughput",
            "metric": {"id": "sustained_throughput", "inner": 16},
            "medians_ms": {name: float(index + 1) for index, name in enumerate(candidates)},
            "invariants": {"paired": True, "slots_shared": True},
            "gpu_lock": {"mode": "external"},
        },
    }
    path.write_text(json.dumps(value))
    loaded = _search_round_cache(
        path, search_id=search_id, route="large_m", round_index=0,
        candidate_ids=candidates, fingerprints=fingerprints,
        protocol_hash=protocol_hash, score_metric=score_metric,
    )
    assert loaded == value

    changed = dict(fingerprints)
    changed["cfg-a"] = dict(changed["cfg-a"], artifact_sha256="f" * 64)
    assert _search_round_cache(
        path, search_id=search_id, route="large_m", round_index=0,
        candidate_ids=candidates, fingerprints=changed,
        protocol_hash=protocol_hash, score_metric=score_metric,
    ) is None
