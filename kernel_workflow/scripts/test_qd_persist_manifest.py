#!/usr/bin/env python3
"""Tests for qd_persist_manifest.

The load-bearing one is `test_admitted_cells_reach_the_manifest`: finding (96)
was an archive writer that preserved the previous generation's `cells` verbatim
while recording twelve acceptances in the transition log. Every other test here
guards a way of refusing to persist; that one guards the way of quietly not
persisting, which is the failure that actually happened and the one that leaves
no error behind.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import qd_persist_manifest as P  # noqa: E402
from qd_source_hash import tree_hash  # noqa: E402

SEED_SRC = "int gemm(void) { return 0; }\n"
CHILD_SRC = "int gemm(void) { return 1; }\n"


def _baseline(tmp_path: Path) -> Path:
    base = tmp_path / "baseline"
    (base / "src").mkdir(parents=True)
    (base / "src" / "custom_gemm.hip").write_text(SEED_SRC)
    (base / "config.yaml").write_text("arch: gfx942\n")
    return base


def _patch(tmp_path: Path, name: str, old: str, new: str) -> Path:
    """A real unified diff, applied with the real `git apply -p1`."""
    a = tmp_path / "_a" / "src"
    b = tmp_path / "_b" / "src"
    for d in (a, b):
        d.mkdir(parents=True, exist_ok=True)
    (a / "custom_gemm.hip").write_text(old)
    (b / "custom_gemm.hip").write_text(new)
    proc = subprocess.run(
        ["git", "diff", "--no-index", "--src-prefix=a/", "--dst-prefix=b/",
         "_a/src/custom_gemm.hip", "_b/src/custom_gemm.hip"],
        cwd=str(tmp_path), capture_output=True, text=True)
    # `git diff --no-index` exits 1 when the files differ, which is the case we want.
    assert proc.returncode in (0, 1), proc.stderr
    body = proc.stdout.replace("a/_a/src/", "a/src/").replace("b/_b/src/", "b/src/")
    out = tmp_path / name
    out.write_text(body)
    return out


def _child_hash(tmp_path: Path, base: Path, patch: Path) -> str:
    """What the child tree really hashes to, derived the same way the module does."""
    scratch = tmp_path / "_probe"
    if scratch.exists():
        import shutil
        shutil.rmtree(scratch)
    import shutil
    shutil.copytree(str(base), str(scratch))
    subprocess.run(["git", "apply", "-p1", str(patch)], cwd=str(scratch), check=True)
    return tree_hash(str(scratch))


def _payload(tmp_path: Path, base: Path, admissions, cells):
    return {
        "archive_dir": str(tmp_path / "qd_archive"),
        "immutable_baseline": str(base),
        "manifest": {"version": 2, "classifier_version": "geak-qd-v2", "generation": 1,
                     "cells": cells, "challengers": {}, "capsules": {},
                     "lineage": {}, "recent_transitions": [], "global_best": None},
        "admissions": admissions,
    }


def _cell(elite_id, source_hash, cell_key):
    return {cell_key: {"elite_id": elite_id, "source_hash": source_hash,
                       "artifact": source_hash, "context_id": "decode_m16_square",
                       "geomean": 1.15, "robust": {"score": 1.15, "lower": 1.07, "upper": 1.23}}}


def _adm(elite_id, source_hash, cell_key, patch):
    return {"elite_id": elite_id, "cell": cell_key, "source_hash": source_hash,
            "patch": str(patch), "generation": 1, "operator": "directed_transition",
            "parent_elite_id": "seed_decode_m16_square", "parent_workspace": "/nowhere"}


# --- the regression that (96) is -------------------------------------------

def test_admitted_cells_reach_the_manifest(tmp_path):
    base = _baseline(tmp_path)
    patch = _patch(tmp_path, "child.diff", SEED_SRC, CHILD_SRC)
    h = _child_hash(tmp_path, base, patch)
    key = "decode_m16_square|native_mfma|lds_reg_prefetch"
    pay = _payload(tmp_path, base, [_adm("r1_s1_x", h, key, patch)], _cell("r1_s1_x", h, key))

    r = P.persist(pay)

    assert r["persisted_elite_ids"] == ["r1_s1_x"]
    assert r["artifacts"][h] == "materialized"
    assert r["failures"] == []
    written = json.loads((tmp_path / "qd_archive" / "manifest.json").read_text())
    assert key in written["cells"], "the admitted cell must be in the manifest on disk"
    assert written["cells"][key]["elite_id"] == "r1_s1_x"
    assert written["generation"] == 1
    assert written["written_by"] == "qd_persist_manifest.py"


def test_previous_manifest_on_disk_does_not_win(tmp_path):
    """(96) verbatim: an older manifest is already there and says nothing changed."""
    base = _baseline(tmp_path)
    ad = tmp_path / "qd_archive"
    ad.mkdir()
    (ad / "manifest.json").write_text(json.dumps(
        {"generation": 0, "cells": {"seed_cell": {"elite_id": "seed", "source_hash": "0" * 64}}}))
    patch = _patch(tmp_path, "child.diff", SEED_SRC, CHILD_SRC)
    h = _child_hash(tmp_path, base, patch)
    key = "decode_m16_square|new"
    P.persist(_payload(tmp_path, base, [_adm("r1_x", h, key, patch)], _cell("r1_x", h, key)))

    written = json.loads((ad / "manifest.json").read_text())
    assert list(written["cells"]) == [key]
    assert "seed_cell" not in written["cells"]


# --- refusals ---------------------------------------------------------------

def test_hash_mismatch_refuses_the_artifact_and_drops_the_cell(tmp_path):
    base = _baseline(tmp_path)
    patch = _patch(tmp_path, "child.diff", SEED_SRC, CHILD_SRC)
    wrong = "b" * 64
    key = "decode_m16_square|new"
    r = P.persist(_payload(tmp_path, base, [_adm("r1_x", wrong, key, patch)],
                           _cell("r1_x", wrong, key)))

    assert r["persisted_elite_ids"] == []
    assert "claimed" in r["artifacts"][wrong]
    assert not (tmp_path / "qd_archive" / "artifacts" / wrong).exists()
    written = json.loads((tmp_path / "qd_archive" / "manifest.json").read_text())
    assert written["cells"] == {}, "a cell with no artifact must not be written"


def test_unapplicable_patch_is_refused_before_hashing(tmp_path):
    base = _baseline(tmp_path)
    patch = _patch(tmp_path, "child.diff", "something else entirely\n", CHILD_SRC)
    key = "k"
    r = P.persist(_payload(tmp_path, base, [_adm("r1_x", "c" * 64, key, patch)],
                           _cell("r1_x", "c" * 64, key)))
    assert r["persisted_elite_ids"] == []
    assert "git apply check failed" in r["artifacts"]["c" * 64]


def test_missing_patch_is_refused(tmp_path):
    base = _baseline(tmp_path)
    key = "k"
    r = P.persist(_payload(tmp_path, base, [_adm("r1_x", "d" * 64, key, tmp_path / "nope.diff")],
                           _cell("r1_x", "d" * 64, key)))
    assert r["persisted_elite_ids"] == []
    assert "patch not found" in r["artifacts"]["d" * 64]


def test_drifted_existing_artifact_is_refused_not_reused(tmp_path):
    """(71): the address is the identity. A tree that no longer hashes to its own
    name is a worse failure than a missing one, because it reads as durable."""
    base = _baseline(tmp_path)
    patch = _patch(tmp_path, "child.diff", SEED_SRC, CHILD_SRC)
    h = _child_hash(tmp_path, base, patch)
    key = "k"
    P.persist(_payload(tmp_path, base, [_adm("r1_x", h, key, patch)], _cell("r1_x", h, key)))
    ws = tmp_path / "qd_archive" / "artifacts" / h / "workspace"
    (ws / "src" / "custom_gemm.hip").write_text("int gemm(void) { return 99; }\n")

    r = P.persist(_payload(tmp_path, base, [_adm("r1_x", h, key, patch)], _cell("r1_x", h, key)))
    assert r["persisted_elite_ids"] == []
    assert "but is stored under" in r["artifacts"][h]


# --- partial failure stays partial -----------------------------------------

def test_one_bad_admission_does_not_take_the_good_one_with_it(tmp_path):
    base = _baseline(tmp_path)
    good_patch = _patch(tmp_path, "good.diff", SEED_SRC, CHILD_SRC)
    good = _child_hash(tmp_path, base, good_patch)
    bad = "e" * 64
    cells = {}
    cells.update(_cell("good_elite", good, "cell_good"))
    cells.update(_cell("bad_elite", bad, "cell_bad"))
    r = P.persist(_payload(tmp_path, base, [
        _adm("good_elite", good, "cell_good", good_patch),
        _adm("bad_elite", bad, "cell_bad", good_patch),
    ], cells))

    assert r["persisted_elite_ids"] == ["good_elite"]
    written = json.loads((tmp_path / "qd_archive" / "manifest.json").read_text())
    assert list(written["cells"]) == ["cell_good"]
    assert any(f.get("elite_ids") == ["bad_elite"] for f in r["failures"])


def test_global_best_is_cleared_when_its_artifact_is_refused(tmp_path):
    base = _baseline(tmp_path)
    patch = _patch(tmp_path, "child.diff", SEED_SRC, CHILD_SRC)
    bad = "f" * 64
    pay = _payload(tmp_path, base, [_adm("r1_x", bad, "k", patch)], _cell("r1_x", bad, "k"))
    pay["manifest"]["global_best"] = {"elite_id": "r1_x", "source_hash": bad}
    r = P.persist(pay)
    written = json.loads((tmp_path / "qd_archive" / "manifest.json").read_text())
    assert written["global_best"] is None
    assert not (tmp_path / "qd_archive" / "global_best.json").exists()
    assert any("global_best cleared" in f.get("reason", "") for f in r["failures"])


# --- mechanics --------------------------------------------------------------

def test_second_run_reuses_and_stays_persisted(tmp_path):
    base = _baseline(tmp_path)
    patch = _patch(tmp_path, "child.diff", SEED_SRC, CHILD_SRC)
    h = _child_hash(tmp_path, base, patch)
    pay = _payload(tmp_path, base, [_adm("r1_x", h, "k", patch)], _cell("r1_x", h, "k"))
    first = P.persist(pay)
    second = P.persist(pay)
    assert first["artifacts"][h] == "materialized"
    assert second["artifacts"][h] == "reused"
    assert second["persisted_elite_ids"] == ["r1_x"]


def test_one_materialization_per_content_address(tmp_path):
    """Eleven elites can share one source hash; the tree is built once."""
    base = _baseline(tmp_path)
    patch = _patch(tmp_path, "child.diff", SEED_SRC, CHILD_SRC)
    h = _child_hash(tmp_path, base, patch)
    cells = {}
    adms = []
    for i in range(4):
        cells.update(_cell(f"e{i}", h, f"cell{i}"))
        adms.append(_adm(f"e{i}", h, f"cell{i}", patch))
    r = P.persist(_payload(tmp_path, base, adms, cells))
    assert sorted(r["persisted_elite_ids"]) == ["e0", "e1", "e2", "e3"]
    assert list(r["artifacts"]) == [h]


def test_no_temporary_files_survive(tmp_path):
    base = _baseline(tmp_path)
    patch = _patch(tmp_path, "child.diff", SEED_SRC, CHILD_SRC)
    h = _child_hash(tmp_path, base, patch)
    P.persist(_payload(tmp_path, base, [_adm("r1_x", h, "k", patch)], _cell("r1_x", h, "k")))
    leftovers = [p for p in (tmp_path / "qd_archive").rglob("*")
                 if p.name.endswith(".tmp") or p.name.startswith("mat_")]
    assert leftovers == []


def test_artifact_records_its_own_reapplied_hash(tmp_path):
    base = _baseline(tmp_path)
    patch = _patch(tmp_path, "child.diff", SEED_SRC, CHILD_SRC)
    h = _child_hash(tmp_path, base, patch)
    P.persist(_payload(tmp_path, base, [_adm("r1_x", h, "k", patch)], _cell("r1_x", h, "k")))
    art = json.loads((tmp_path / "qd_archive" / "artifacts" / h / "ARTIFACT.json").read_text())
    assert art["schema"] == P.ARTIFACT_SCHEMA
    assert art["baseline_patch_verified"]["reapplied_source_hash"] == h
    assert art["materialization"] == "baseline_plus_patch"
    assert (tmp_path / "qd_archive" / "artifacts" / h / "baseline.patch").is_file()


def test_cli_round_trips_a_payload_file(tmp_path):
    base = _baseline(tmp_path)
    patch = _patch(tmp_path, "child.diff", SEED_SRC, CHILD_SRC)
    h = _child_hash(tmp_path, base, patch)
    pay = _payload(tmp_path, base, [_adm("r1_x", h, "k", patch)], _cell("r1_x", h, "k"))
    pf = tmp_path / "payload.json"
    pf.write_text(json.dumps(pay))
    proc = subprocess.run([sys.executable, str(Path(P.__file__)), "--payload", str(pf)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["persisted_elite_ids"] == ["r1_x"]


def test_caller_payload_is_not_mutated(tmp_path):
    base = _baseline(tmp_path)
    patch = _patch(tmp_path, "child.diff", SEED_SRC, CHILD_SRC)
    bad = "a" * 64
    pay = _payload(tmp_path, base, [_adm("r1_x", bad, "k", patch)], _cell("r1_x", bad, "k"))
    before = json.dumps(pay, sort_keys=True)
    P.persist(pay)
    assert json.dumps(pay, sort_keys=True) == before


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# --- merge mode and the verification block ---------------------------------

def _merge_payload(tmp_path, base, admissions, cell_updates, fields=None):
    pay = {"archive_dir": str(tmp_path / "qd_archive"),
           "immutable_baseline": str(base),
           "manifest": dict(fields or {"generation": 1}),
           "cell_updates": cell_updates,
           "admissions": admissions}
    return pay


def test_merge_mode_carries_unchanged_cells_forward(tmp_path):
    """The (96) sentence -- 'cells are carried forward verbatim' -- is true here,
    and true because code does it, for the cells that really did not change."""
    base = _baseline(tmp_path)
    ad = tmp_path / "qd_archive"
    ad.mkdir()
    seed_cells = {f"seed_{i}": {"elite_id": f"seed_{i}", "source_hash": "0" * 64, "cost": 0}
                  for i in range(11)}
    (ad / "manifest.json").write_text(json.dumps({"generation": 0, "cells": seed_cells}))
    patch = _patch(tmp_path, "child.diff", SEED_SRC, CHILD_SRC)
    h = _child_hash(tmp_path, base, patch)
    key = "decode_m16_square|lds_reg_prefetch"

    r = P.persist(_merge_payload(tmp_path, base, [_adm("r1_x", h, key, patch)],
                                 _cell("r1_x", h, key)))

    written = json.loads((ad / "manifest.json").read_text())
    assert len(written["cells"]) == 12, "11 seed cells carried forward + 1 admitted"
    assert written["cells"][key]["elite_id"] == "r1_x"
    assert written["cells"]["seed_0"]["cost"] == 0
    assert written["generation"] == 1
    assert r["persisted_elite_ids"] == ["r1_x"]


def test_merge_mode_replacement_overwrites_the_incumbent(tmp_path):
    base = _baseline(tmp_path)
    ad = tmp_path / "qd_archive"
    ad.mkdir()
    key = "decode_m64_square|seed_descriptor"
    (ad / "manifest.json").write_text(json.dumps(
        {"generation": 0, "cells": {key: {"elite_id": "seed_m64", "source_hash": "0" * 64}}}))
    patch = _patch(tmp_path, "child.diff", SEED_SRC, CHILD_SRC)
    h = _child_hash(tmp_path, base, patch)

    P.persist(_merge_payload(tmp_path, base, [_adm("r1_m64", h, key, patch)],
                             _cell("r1_m64", h, key)))

    written = json.loads((ad / "manifest.json").read_text())
    assert len(written["cells"]) == 1
    assert written["cells"][key]["elite_id"] == "r1_m64"


def test_merge_mode_refused_artifact_leaves_the_incumbent_in_place(tmp_path):
    base = _baseline(tmp_path)
    ad = tmp_path / "qd_archive"
    ad.mkdir()
    key = "decode_m64_square|seed_descriptor"
    (ad / "manifest.json").write_text(json.dumps(
        {"generation": 0, "cells": {key: {"elite_id": "seed_m64", "source_hash": "0" * 64}}}))
    patch = _patch(tmp_path, "child.diff", SEED_SRC, CHILD_SRC)
    bad = "9" * 64

    r = P.persist(_merge_payload(tmp_path, base, [_adm("r1_m64", bad, key, patch)],
                                 _cell("r1_m64", bad, key)))

    written = json.loads((ad / "manifest.json").read_text())
    assert written["cells"][key]["elite_id"] == "seed_m64", (
        "a refused challenger must not evict the incumbent it never replaced")
    assert r["persisted_elite_ids"] == []


def test_verification_describes_what_is_actually_on_disk(tmp_path):
    base = _baseline(tmp_path)
    patch = _patch(tmp_path, "child.diff", SEED_SRC, CHILD_SRC)
    h = _child_hash(tmp_path, base, patch)
    pay = _payload(tmp_path, base, [_adm("r1_x", h, "k", patch)], _cell("r1_x", h, "k"))
    pay["manifest"]["challengers"] = {"a": {}, "b": {}}
    pay["manifest"]["recent_transitions"] = [{}, {}, {}]
    v = P.persist(pay)["verification"]

    assert v["readable"] is True
    assert v["cells"] == 1 and v["generation"] == 1
    assert v["challengers"] == 2 and v["recent_transitions"] == 3
    assert v["bytes"] == (tmp_path / "qd_archive" / "manifest.json").stat().st_size

    import hashlib
    assert v["cell_keys_sha256"] == hashlib.sha256(b"k").hexdigest()
    assert v["elite_ids_sha256"] == hashlib.sha256(b"r1_x").hexdigest()


def test_verification_counts_only_cells_that_survived_their_artifact(tmp_path):
    """The count the lane compares against must exclude what was refused, or the
    assertion would fire on a correct partial persistence."""
    base = _baseline(tmp_path)
    good_patch = _patch(tmp_path, "good.diff", SEED_SRC, CHILD_SRC)
    good = _child_hash(tmp_path, base, good_patch)
    bad = "7" * 64
    cells = {}
    cells.update(_cell("good_elite", good, "cell_good"))
    cells.update(_cell("bad_elite", bad, "cell_bad"))
    r = P.persist(_payload(tmp_path, base, [
        _adm("good_elite", good, "cell_good", good_patch),
        _adm("bad_elite", bad, "cell_bad", good_patch)], cells))
    assert r["verification"]["cells"] == 1
    assert len(r["persisted_elite_ids"]) == 1


# --- the transport: payload checksum ---------------------------------------
#
# The one step of this pipeline that still has a model in it is an agent
# copying the payload into a heredoc. These pin the check that makes a mangled
# copy loud.

def test_fnv1a32_matches_the_reference_vectors():
    """Pinned against the canonical FNV-1a/32 vectors, because the other side of
    this comparison is a reimplementation in kernel_lane.js and 'they agree with
    each other' is satisfied by two identically wrong functions."""
    assert P.fnv1a32("") == "811c9dc5"
    assert P.fnv1a32("a") == "e40c292c"
    assert P.fnv1a32("foobar") == "bf9cf968"


def test_checksum_mismatch_writes_nothing(tmp_path):
    base = _baseline(tmp_path)
    patch = _patch(tmp_path, "c.diff", SEED_SRC, CHILD_SRC)
    h = _child_hash(tmp_path, base, patch)
    pay = tmp_path / "payload.json"
    pay.write_text(json.dumps(_payload(tmp_path, base, [_adm("r1_x", h, "k", patch)],
                                       _cell("r1_x", h, "k"))))
    text = pay.read_text().rstrip()
    wrong = f"{P.fnv1a32(text + ' ')}:{len(text.encode())}"
    proc = subprocess.run([sys.executable, str(Path(P.__file__)), "--payload", str(pay),
                           "--expect-checksum", wrong], capture_output=True, text=True)
    assert proc.returncode == 3, proc.stderr
    receipt = json.loads(proc.stdout)
    assert receipt["persisted_elite_ids"] == []
    assert "checksum" in receipt["failures"][0]["reason"]
    assert not (tmp_path / "qd_archive" / "manifest.json").exists()


def test_matching_checksum_persists(tmp_path):
    base = _baseline(tmp_path)
    patch = _patch(tmp_path, "c.diff", SEED_SRC, CHILD_SRC)
    h = _child_hash(tmp_path, base, patch)
    pay = tmp_path / "payload.json"
    pay.write_text(json.dumps(_payload(tmp_path, base, [_adm("r1_x", h, "k", patch)],
                                       _cell("r1_x", h, "k"))) + "\n")
    text = pay.read_text().rstrip()
    proc = subprocess.run([sys.executable, str(Path(P.__file__)), "--payload", str(pay),
                           "--expect-checksum", f"{P.fnv1a32(text)}:{len(text.encode())}"],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["persisted_elite_ids"] == ["r1_x"]


def test_verification_exposes_digests_the_lane_can_recompute(tmp_path):
    """sha256 is unavailable in the workflow JS sandbox, so a receipt carrying
    only sha256 is a receipt that must be believed."""
    base = _baseline(tmp_path)
    patch = _patch(tmp_path, "c.diff", SEED_SRC, CHILD_SRC)
    h = _child_hash(tmp_path, base, patch)
    v = P.persist(_payload(tmp_path, base, [_adm("r1_x", h, "k", patch)],
                           _cell("r1_x", h, "k")))["verification"]
    assert v["cell_keys_fnv1a32"] == P.fnv1a32("k")
    assert v["elite_ids_fnv1a32"] == P.fnv1a32("r1_x")


# --- patch-less admissions (bootstrap seed, warm import) --------------------

def test_source_tree_admission_is_copied_and_hash_checked(tmp_path):
    base = _baseline(tmp_path)
    seed_hash = tree_hash(str(base))
    adm = {"elite_id": "seed_x", "cell": "k", "source_hash": seed_hash, "patch": None,
           "source_tree": str(base), "generation": 0, "operator": "bootstrap"}
    r = P.persist(_payload(tmp_path, base, [adm], _cell("seed_x", seed_hash, "k")))
    assert r["persisted_elite_ids"] == ["seed_x"]
    art = tmp_path / "qd_archive" / "artifacts" / seed_hash
    meta = json.loads((art / "ARTIFACT.json").read_text())
    assert meta["materialization"] == "source_snapshot"
    assert tree_hash(str(art / "workspace")) == seed_hash
    # A patch-less admission still owes the archive a baseline.patch: canonical
    # promotion reads `artifacts/<hash>/baseline.patch` and does not know or
    # care how the tree got there. This seed IS the baseline, so the correct
    # patch is the empty one.
    assert (art / "baseline.patch").is_file()
    assert (art / "baseline.patch").read_text().strip() == ""
    assert meta["baseline_patch"] == "baseline.patch"
    assert "identical to the immutable baseline" in meta["baseline_patch_status"]


def test_a_derived_baseline_patch_is_verified_by_reapplication(tmp_path):
    """An import arrives as a tree, not a diff, and can be promoted to canonical.
    The patch derived for it is reapplied and hashed here, rather than at the
    moment something tries to promote it."""
    base = _baseline(tmp_path)
    imported = tmp_path / "imported"
    import shutil as _sh
    _sh.copytree(str(base), str(imported))
    (imported / "src" / "custom_gemm.hip").write_text(CHILD_SRC)
    h = tree_hash(str(imported))
    adm = {"elite_id": "import_x", "cell": "k", "source_hash": h, "patch": None,
           "source_tree": str(imported), "generation": 0, "operator": "historical_import"}
    r = P.persist(_payload(tmp_path, base, [adm], _cell("import_x", h, "k")))
    assert r["persisted_elite_ids"] == ["import_x"], r["failures"]
    art = tmp_path / "qd_archive" / "artifacts" / h
    body = (art / "baseline.patch").read_text()
    assert "custom_gemm.hip" in body
    assert "verified by reapplication" in json.loads(
        (art / "ARTIFACT.json").read_text())["baseline_patch_status"]
    # And it really does apply to the frozen baseline.
    replay = tmp_path / "replay"
    _sh.copytree(str(base), str(replay))
    subprocess.run(["git", "apply", "-p1", str(art / "baseline.patch")],
                   cwd=str(replay), check=True)
    assert tree_hash(str(replay)) == h


def test_source_tree_admission_refuses_a_wrong_hash(tmp_path):
    base = _baseline(tmp_path)
    adm = {"elite_id": "seed_x", "cell": "k", "source_hash": "3" * 64, "patch": None,
           "source_tree": str(base), "generation": 0, "operator": "bootstrap"}
    r = P.persist(_payload(tmp_path, base, [adm], _cell("seed_x", "3" * 64, "k")))
    assert r["persisted_elite_ids"] == []
    assert "copied source tree" in r["failures"][0]["reason"]


def test_admission_with_neither_patch_nor_source_tree_is_refused(tmp_path):
    base = _baseline(tmp_path)
    adm = {"elite_id": "x", "cell": "k", "source_hash": "4" * 64, "patch": None,
           "source_tree": None, "generation": 1, "operator": "directed_transition"}
    r = P.persist(_payload(tmp_path, base, [adm], _cell("x", "4" * 64, "k")))
    assert r["persisted_elite_ids"] == []
    assert "neither" in r["failures"][0]["reason"]


# --- the archive-side policy gate ------------------------------------------

FORBIDDEN = 'extern "C" void go(void) { rocblas_gemm_ex(); }\n'


def test_a_candidate_that_introduces_a_forbidden_library_is_refused(tmp_path):
    base = _baseline(tmp_path)
    patch = _patch(tmp_path, "bad.diff", SEED_SRC, FORBIDDEN)
    h = _child_hash(tmp_path, base, patch)
    r = P.persist(_payload(tmp_path, base, [_adm("r1_bad", h, "k", patch)],
                           _cell("r1_bad", h, "k")))
    assert r["persisted_elite_ids"] == []
    assert "policy scan" in r["failures"][0]["reason"]
    assert not (tmp_path / "qd_archive" / "artifacts" / h).exists()


def test_the_frozen_oracle_does_not_trip_the_policy_gate(tmp_path):
    """The baseline ships the rocBLAS oracle. Exempting it by name would be a
    per-task list in a prompt; exempting it by byte-identity to the frozen tree
    needs nothing maintained and errs toward blocking."""
    base = _baseline(tmp_path)
    (base / "src" / "rocblas_baseline.cpp").write_text(FORBIDDEN)
    patch = _patch(tmp_path, "ok.diff", SEED_SRC, CHILD_SRC)
    h = _child_hash(tmp_path, base, patch)
    r = P.persist(_payload(tmp_path, base, [_adm("r1_ok", h, "k", patch)],
                           _cell("r1_ok", h, "k")))
    assert r["persisted_elite_ids"] == ["r1_ok"], r["failures"]
    art = tmp_path / "qd_archive" / "artifacts" / h
    receipt = json.loads((art / "policy_receipt.json").read_text())
    assert receipt["passed_after_baseline_exemption"] is True
    assert receipt["exempt_because_identical_to_frozen_baseline"]


def test_editing_the_oracle_file_stops_it_being_exempt(tmp_path):
    """Byte-identity, not path: a candidate that edits the oracle owns it."""
    base = _baseline(tmp_path)
    (base / "src" / "rocblas_baseline.cpp").write_text(FORBIDDEN)
    a = tmp_path / "_c" / "src"
    b = tmp_path / "_d" / "src"
    for d in (a, b):
        d.mkdir(parents=True, exist_ok=True)
    (a / "rocblas_baseline.cpp").write_text(FORBIDDEN)
    (b / "rocblas_baseline.cpp").write_text(FORBIDDEN + "// candidate edit\n")
    proc = subprocess.run(
        ["git", "diff", "--no-index", "_c/src/rocblas_baseline.cpp", "_d/src/rocblas_baseline.cpp"],
        cwd=str(tmp_path), capture_output=True, text=True)
    patch = tmp_path / "edit.diff"
    patch.write_text(proc.stdout.replace("a/_c/src/", "a/src/").replace("b/_d/src/", "b/src/"))
    h = _child_hash(tmp_path, base, patch)
    r = P.persist(_payload(tmp_path, base, [_adm("r1_edit", h, "k", patch)],
                           _cell("r1_edit", h, "k")))
    assert r["persisted_elite_ids"] == []
    assert "policy scan" in r["failures"][0]["reason"]


# --- what the content address actually commits to ---------------------------

def _import_tree(base: Path, dest: Path) -> Path:
    """A verified workspace: the baseline plus a candidate edit."""
    import shutil as _sh
    _sh.copytree(str(base), str(dest))
    (dest / "src" / "custom_gemm.hip").write_text(CHILD_SRC)
    return dest


def _tree_admission(tree: Path, source_hash: str, elite_id: str = "import_x"):
    return {"elite_id": elite_id, "cell": "k", "source_hash": source_hash, "patch": None,
            "source_tree": str(tree), "generation": 0, "operator": "historical_import"}


def test_build_products_are_not_copied_into_the_artifact(tmp_path):
    """A verified workspace carries `.torch_ext/`, `build/` and `__pycache__/`,
    and `tree_hash` excludes every one of them. Copying them anyway gives two
    trees with different compiled objects the same content address, which is
    the one thing the address exists to prevent (71)."""
    base = _baseline(tmp_path)
    tree = _import_tree(base, tmp_path / "verified")
    ext = tree / ".torch_ext" / "geak_oracle"
    ext.mkdir(parents=True)
    (ext / "build.ninja").write_text("-lrocblas -lhipblaslt\n")
    (ext / "oracle.so").write_bytes(b"\x7fELF\x00rocblas_gemm_ex\x00")
    (tree / "build").mkdir()
    (tree / "build" / "kernel.o").write_bytes(b"\x00\x01\x02")
    (tree / "src" / "__pycache__").mkdir()
    (tree / "src" / "__pycache__" / "x.cpython-310.pyc").write_bytes(b"\x00")
    (tree / "run.log").write_text("rocblas chatter\n")

    h = tree_hash(str(tree))
    r = P.persist(_payload(tmp_path, base, [_tree_admission(tree, h)],
                           _cell("import_x", h, "k")))
    # Not merely "the hash still matches" -- the policy scan sees exactly what
    # was copied, and a scan that reports 26 rocblas findings on a candidate
    # that did nothing wrong is a gate everyone learns to wave through (53).
    assert r["persisted_elite_ids"] == ["import_x"], r["failures"]
    ws = tmp_path / "qd_archive" / "artifacts" / h / "workspace"
    assert not (ws / ".torch_ext").exists()
    assert not (ws / "build").exists()
    assert not (ws / "src" / "__pycache__").exists()
    assert not (ws / "run.log").exists()
    assert (ws / "src" / "custom_gemm.hip").read_text() == CHILD_SRC
    assert tree_hash(str(ws)) == h


def test_a_git_directory_in_the_source_tree_does_not_reach_the_artifact(tmp_path):
    """`.git` is excluded from the address too, and a stored `.git/` makes the
    derived patch unappliable: `git apply` refuses paths under it outright."""
    base = _baseline(tmp_path)
    tree = _import_tree(base, tmp_path / "verified")
    subprocess.run(["git", "init", "-q", str(tree)], check=True)
    h = tree_hash(str(tree))
    r = P.persist(_payload(tmp_path, base, [_tree_admission(tree, h)],
                           _cell("import_x", h, "k")))
    assert r["persisted_elite_ids"] == ["import_x"], r["failures"]
    art = tmp_path / "qd_archive" / "artifacts" / h
    assert not (art / "workspace" / ".git").exists()
    assert "verified by reapplication" in json.loads(
        (art / "ARTIFACT.json").read_text())["baseline_patch_status"]


def _git_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q", str(root)], check=True)


def test_derivation_inside_an_enclosing_repository_still_applies(tmp_path):
    """Finding (98). `git apply` does not treat a patch as paths relative to the
    working directory -- it resolves them against the repository root and
    *silently ignores* paths outside the directory it ran from. The real archive
    lives under `exp/` inside this checkout, so every derivation done in place
    applied zero hunks, exited 0, and reproduced the untouched baseline. The
    same code passes under `/tmp`, which is the worst possible shape for a bug.

    So this test puts the archive inside a repository on purpose. Without the
    ceiling directory it fails at the persister's own hash check, not here.
    """
    _git_repo(tmp_path)
    base = _baseline(tmp_path)
    tree = _import_tree(base, tmp_path / "verified")
    h = tree_hash(str(tree))
    r = P.persist(_payload(tmp_path, base, [_tree_admission(tree, h)],
                           _cell("import_x", h, "k")))
    assert r["persisted_elite_ids"] == ["import_x"], r["failures"]
    art = tmp_path / "qd_archive" / "artifacts" / h
    body = (art / "baseline.patch").read_text()
    assert "custom_gemm.hip" in body
    # And it applies outside the repository as well as inside it.
    import shutil as _sh
    replay = tmp_path / "replay"
    _sh.copytree(str(base), str(replay))
    subprocess.run(["git", "apply", "-p1", str(art / "baseline.patch")],
                   cwd=str(replay), check=True,
                   env={**os.environ, "GIT_CEILING_DIRECTORIES": str(replay.parent)})
    assert tree_hash(str(replay)) == h


def test_created_and_deleted_files_get_relative_patch_headers(tmp_path):
    """A created file names the DESTINATION on both sides of its header and a
    deleted file names the SOURCE on both, so rewriting only `a<ref>/` and
    `b<tree>/` leaves half of them absolute -- and an absolute path in a patch
    header is one `git apply` refuses as outside the repository."""
    _git_repo(tmp_path)
    base = _baseline(tmp_path)
    tree = _import_tree(base, tmp_path / "verified")
    (tree / "src" / "tiling.hip").write_text("// new file\n")
    (tree / "config.yaml").unlink()
    h = tree_hash(str(tree))
    r = P.persist(_payload(tmp_path, base, [_tree_admission(tree, h)],
                           _cell("import_x", h, "k")))
    assert r["persisted_elite_ids"] == ["import_x"], r["failures"]
    body = (tmp_path / "qd_archive" / "artifacts" / h / "baseline.patch").read_text()
    assert str(tmp_path) not in body, "absolute paths survived the header rewrite"
    assert "src/tiling.hip" in body and "config.yaml" in body


def test_a_non_utf8_file_does_not_refuse_the_artifact(tmp_path):
    """The derived patch is a byte stream and has to be handled as one.

    Not with an `.npz`: git base85-encodes true binary deltas, so those survive
    a UTF-8 decode by accident. The case that actually raised was a file git
    treats as TEXT -- no NUL bytes, so it goes into the patch as raw bytes --
    whose contents are not valid UTF-8. A latin-1 comment is enough, and
    refusing an artifact over the encoding of a comment would be absurd.
    """
    _git_repo(tmp_path)
    base = _baseline(tmp_path)
    tree = _import_tree(base, tmp_path / "verified")
    (tree / "fixture.npz").write_bytes(bytes(range(256)) * 8)
    (tree / "src" / "notes.hip").write_bytes(b"// caf\xe9 tiling, 32\xbd waves\n")
    h = tree_hash(str(tree))
    r = P.persist(_payload(tmp_path, base, [_tree_admission(tree, h)],
                           _cell("import_x", h, "k")))
    assert r["persisted_elite_ids"] == ["import_x"], r["failures"]
    art = tmp_path / "qd_archive" / "artifacts" / h
    assert (art / "workspace" / "fixture.npz").read_bytes()[:4] == b"\x00\x01\x02\x03"
    assert (art / "workspace" / "src" / "notes.hip").read_bytes() == \
        b"// caf\xe9 tiling, 32\xbd waves\n", "the bytes were re-encoded in transit"
    assert "verified by reapplication" in json.loads(
        (art / "ARTIFACT.json").read_text())["baseline_patch_status"]


# --- the round that admitted nothing still has something to say -------------
#
# Finding (124b). `kernel_lane.js` used to guard its round persist site on
# `qdAdmissions.length`, so a round that refuted everything wrote nothing at
# all -- and the capsule ledger, the transition edges, the stall counters and
# the generation number all ride in the SAME payload's top-level fields. Run 16
# spent two generations and ~2.5 GPU-hours refuting two mechanisms and left a
# manifest reading `generation 0`, `capsules {}`, `stalls 0/0/0`.
#
# The lane now persists unconditionally, which sends this module a payload it
# had never been given: merge shape, `cell_updates: {}`, `admissions: []`. The
# claim that it "already handles that path" was worth exactly nothing until it
# was run, so these two tests run it.

def _ledger_payload(tmp_path, base, generation, capsules, transitions, stalls):
    """The lane's MERGE shape with nothing admitted: no `cells`, no admissions."""
    return {
        "archive_dir": str(tmp_path / "qd_archive"),
        "immutable_baseline": str(base),
        "manifest": {"version": 2, "classifier_version": "geak-qd-v2",
                     "generation": generation, "challengers": {}, "capsules": capsules,
                     "lineage": {}, "recent_transitions": transitions,
                     "stalls": stalls, "global_best": None},
        "cell_updates": {},
        "admissions": [],
    }


def test_a_round_that_admitted_nothing_still_persists_its_ledger(tmp_path):
    base = _baseline(tmp_path)
    patch = _patch(tmp_path, "child.diff", SEED_SRC, CHILD_SRC)
    h = _child_hash(tmp_path, base, patch)
    key = "decode_m16_square|native_mfma|lds_reg_prefetch"
    # Round 1 admits, the ordinary way, so there is a cell on disk to preserve.
    P.persist(_payload(tmp_path, base, [_adm("r1_s1_x", h, key, patch)],
                       _cell("r1_s1_x", h, key)))

    capsules = {f"{key}|direct_store": {"mechanism": "direct_store", "attempts": 2,
                                        "observed_effect": "suite 0.8267, worst route 0.2336"}}
    transitions = [{"from_cell": key, "to_cell": key, "generation": 2, "admitted": False}]
    stalls = {"coverage": 2, "qd_score": 2, "global": 2}
    r = P.persist(_ledger_payload(tmp_path, base, 2, capsules, transitions, stalls))

    assert r["persisted_elite_ids"] == []
    assert r["artifacts"] == {}, "nothing was admitted, so nothing may be materialized"
    assert r["failures"] == [], r["failures"]

    written = json.loads((tmp_path / "qd_archive" / "manifest.json").read_text())
    # The refutation reached disk...
    assert written["generation"] == 2
    assert written["capsules"] == capsules
    assert written["recent_transitions"] == transitions
    assert written["stalls"] == stalls
    # ...and the incumbent it failed to beat is untouched. A merge with an empty
    # `cell_updates` must leave the map alone; clearing it would make an
    # unsuccessful round destructive, which is far worse than the silence it
    # replaces.
    assert list(written["cells"]) == [key]
    assert written["cells"][key]["elite_id"] == "r1_s1_x"


def test_the_ledger_only_receipt_is_still_checkable_against_the_lane(tmp_path):
    """`qdVerifyPersisted` must be doing real work on this path, not passing
    vacuously because there is nothing to compare."""
    base = _baseline(tmp_path)
    patch = _patch(tmp_path, "child.diff", SEED_SRC, CHILD_SRC)
    h = _child_hash(tmp_path, base, patch)
    key = "decode_m16_square|native_mfma|lds_reg_prefetch"
    P.persist(_payload(tmp_path, base, [_adm("r1_s1_x", h, key, patch)],
                       _cell("r1_s1_x", h, key)))

    r = P.persist(_ledger_payload(tmp_path, base, 2, {}, [], {"coverage": 1}))
    v = r["verification"]
    assert v["readable"] is True
    assert v["generation"] == 2
    # The four fields kernel_lane.js compares, recomputed here exactly as
    # `qdCellDigests()` does it: sorted keys and sorted elite ids, newline
    # joined, FNV-1a/32.
    assert v["cells"] == 1
    assert v["cell_keys_fnv1a32"] == P.fnv1a32(key)
    assert v["elite_ids_fnv1a32"] == P.fnv1a32("r1_s1_x")
    # And it is a comparison that can fail: a manifest with a different cell
    # set produces different digests on the same code path.
    other = _payload(tmp_path, base, [], {})
    other["archive_dir"] = str(tmp_path / "other_archive")
    v2 = P.persist(other)["verification"]
    assert v2["cell_keys_fnv1a32"] != v["cell_keys_fnv1a32"]
