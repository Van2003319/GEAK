#!/usr/bin/env python3
"""Deterministic persistence for the QD v2 archive: artifacts, then manifest.

This module exists because of finding (96). The archive's durable write used to
be a `tech_lead` agent task ("Materialize content-addressed artifacts, cell
references, challengers, and an atomic v2 manifest"), and on round 1 of
`qd_v2_bf16_smoke_20260816b_tw054` that agent wrote a manifest whose docstring
said "no cell changed this generation" while its own transition table carried
twelve `accepted: true` rows. Eleven new cells, one replacement and two
artifacts were dropped; the ten rejections were transcribed perfectly. The
generation was recorded as a total stall, and the capsule ledger was written
with `improved: False` on the mechanism that had just produced 1.72x-2.29x on
eight routes -- which is the field the next planner reads to decide whether a
mechanism is worth repeating (76).

Nothing in that job requires judgement. `qdSummary()` in kernel_lane.js is
already a complete, deterministic projection of the in-memory archive; writing
it to disk is a serialization, and a serialization with a model in it is a
serialization that can decide to summarize. So this is code.

Contract
--------
stdin:  one JSON object
          archive_dir         where the archive lives
          immutable_baseline  the frozen tree every patch applies to
          manifest            top-level fields to set: generation, qd_score,
                              stalls, challengers, capsules, lineage,
                              recent_transitions, global_best. If it also
                              carries `cells`, that map REPLACES the stored one
                              (full-projection mode, used by tests and by a
                              rebuild).
          cell_updates        {cell_key: elite_entry} merged onto the stored
                              cells (merge mode, used by the lane). Unchanged
                              cells are carried forward by this code rather
                              than by anyone's memory.
          admissions          [{elite_id, cell, source_hash, patch,
                                parent_workspace, generation, operator,
                                parent_elite_id, policy_receipt?}]
                              A patch-less admission (the bootstrap seed, a
                              re-verified warm import) carries `source_tree`
                              instead and is copied rather than reapplied.
          policy_immutable    paths inside the tree that are the frozen
                              oracle/baseline and are therefore reported by
                              `candidate_policy_scan.py` rather than scanned as
                              candidate code

`--expect-checksum <fnv1a32>:<bytes>` is how the producer of the payload proves
the file on disk is the payload it emitted. The transport between them is an
agent copying text into a heredoc, which is the one step of this pipeline that
still has a model in it; the checksum is what makes a mangled or summarized
transcription a loud failure instead of a plausible one. FNV-1a/32 over the
ASCII payload text with trailing whitespace stripped -- weak against an
adversary, sufficient against a paraphrase, and computable in the caller's
sandboxed JS, which has no crypto.
stdout: one JSON object
          persisted_elite_ids  the ids that are durable, and only those
          artifacts            {hash: "materialized" | "reused" | reason}
          failures             [{elite_id|source_hash, reason}]
          verification         counts and digests re-read from the file after
                               writing, so the caller can assert rather than
                               believe
          manifest             path written

The receipt is deliberately shaped like the agent's was, so
`qdPersistenceReceipt` and the rollback loop downstream of it are unchanged: an
elite absent from `persisted_elite_ids` is rolled back out of the in-memory
archive exactly as before. A partial failure must stay partial.

Materialization is by re-derivation, not by copying the engineer's directory:
the baseline is copied, the candidate's patch is applied to it, and the
resulting tree is hashed. If that hash is not the claimed `source_hash` the
artifact is refused. That is what makes the address an identity rather than a
slot (71) -- it proves the stored patch reproduces the bytes that were
measured, instead of trusting a path that has since been rebuilt in place.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qd_source_hash import (DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_FILE_NAMES,
                            DEFAULT_EXCLUDED_FILE_SUFFIXES, tree_hash)  # noqa: E402
from candidate_policy_scan import scan as policy_scan  # noqa: E402

ARTIFACT_SCHEMA = "geak.qd-artifact/v2"
HASH_SCHEMA = "geak.qd-source-hash/v1"
WORKSPACE = "workspace"


def fnv1a32(text: str) -> str:
    """FNV-1a/32 over the ASCII bytes of `text`.

    Must stay byte-for-byte identical to `qdFnv1a32` in kernel_lane.js. Both
    sides mask to 32 bits after every multiply; Python's unbounded ints make it
    easy to forget the mask and produce a number JS can never reproduce.
    """
    h = 0x811C9DC5
    for byte in text.encode("utf-8", "surrogatepass"):
        h = ((h ^ byte) * 0x01000193) & 0xFFFFFFFF
    return f"{h:08x}"


def _utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _write_json(path: Path, obj: Any) -> None:
    """Write with fsync. Every file this module leaves behind is a claim that
    something is durable, and a claim that is only in the page cache is not."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _ignore_unhashed(directory: str, names: List[str]) -> set:
    """`shutil.copytree` filter for everything `tree_hash` does not cover.

    The artifact tree has to be exactly what its address commits to. Copying a
    verified workspace wholesale brings `.torch_ext/`, `build/`, `__pycache__/`
    and friends along, and those are excluded from `tree_hash` by construction
    -- so two trees with different compiled objects would share one content
    address, which is the opposite of what the address is for (71).

    It also makes the archive-side policy scan readable. A candidate workspace
    carries the compiled frozen oracle under `.torch_ext/`, whose `build.ninja`
    and `.so` mention rocblas because the oracle links rocblas on purpose.
    Scanning them produces 26 blocking findings on a candidate that did nothing
    wrong, and a gate that reports a violation every single time is a gate
    everyone learns to wave through (53). These files are not part of the
    candidate's identity, so they are not part of the artifact, so there is
    nothing to scan and nothing to wave through.
    """
    del directory  # the exclusion is by name at any depth, as in qd_source_hash
    return {n for n in names
            if n in DEFAULT_EXCLUDED_DIRS
            or n in DEFAULT_EXCLUDED_FILE_NAMES
            or n.endswith(DEFAULT_EXCLUDED_FILE_SUFFIXES)
            or (n.endswith(".egg-info") and "*.egg-info" in DEFAULT_EXCLUDED_DIRS)}


def _copy_hashed_tree(src: Path, dest: Path) -> None:
    """Copy only the entries `tree_hash` covers, so the tree matches its name."""
    shutil.copytree(str(src), str(dest), symlinks=True, ignore=_ignore_unhashed)


def _standalone_git_env(tree: Path) -> Dict[str, str]:
    """Environment that stops `git` discovering a repository above `tree`.

    Finding (98). `git apply` run inside a checkout does not treat the patch as
    a set of paths relative to the current directory -- it resolves them against
    the repository root and, per its own documentation, *silently ignores*
    patched paths outside the directory it was run from. The archive lives under
    `exp/` inside this repository, so every derivation done in place applied
    exactly zero hunks, returned 0, and produced a copy of the untouched
    baseline. The same derivation run under `/tmp` worked perfectly, which is
    the worst possible shape for a bug: it passes wherever anyone tests it.

    Only the hash check afterwards caught it, and that is the argument for
    keeping such checks even where the step "obviously" cannot fail.
    """
    env = dict(os.environ)
    env["GIT_CEILING_DIRECTORIES"] = str(tree.parent)
    env.pop("GIT_DIR", None)
    env.pop("GIT_WORK_TREE", None)
    return env


def _apply_patch(patch: Path, tree: Path) -> Tuple[bool, str]:
    """Apply with `git apply -p1`, --check first so a partial application never
    reaches the hash step. A patch that half-applies produces a tree that hashes
    to nothing anyone claimed, which would be caught -- but it would be caught
    as a hash mismatch, which reads as the wrong diagnosis."""
    env = _standalone_git_env(tree)
    for args in (["--check"], []):
        proc = subprocess.run(
            ["git", "apply", "-p1", *args, str(patch)],
            cwd=str(tree), capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            stage = "check" if args else "apply"
            return False, f"git apply {stage} failed: {proc.stderr.strip()[:400]}"
    return True, ""


def _derive_baseline_patch(work: Path, tree: Path, baseline: Path,
                           source_hash: str) -> Tuple[bool, str]:
    """Write `work/baseline.patch` for a tree that arrived without one.

    Returns (ok, status). Not writing the file is a real outcome, not an error:
    the artifact is already proven by its own hash, and only canonical promotion
    reads the patch. Saying so in the receipt is what keeps a later promotion
    failure legible.
    """
    if not baseline.is_dir():
        return False, "no immutable baseline to diff against"
    probe = Path(tempfile.mkdtemp(dir=str(work.parent), prefix="probe_"))
    try:
        # Diff against a FILTERED view of the baseline, not the directory as it
        # sits. The artifact tree only contains what `tree_hash` covers, so a
        # raw diff reports every unhashed baseline file -- `results/policy_*.json`,
        # `.torch_ext/`, stray `.log`s -- as a deletion the candidate never made.
        # Those hunks then fail to apply against the equally-filtered replay copy,
        # and the artifact is refused for a difference that is not a difference.
        # Both sides of the diff must be the same view of the tree, and the view
        # that matters is the one the content address commits to.
        ref = probe / "baseline"
        _copy_hashed_tree(baseline, ref)
        # The candidate side is filtered too, and for the same reason rather
        # than for symmetry's sake: an artifact materialized before this filter
        # existed still has its `.git/` in it, and `git apply` refuses a patch
        # that writes there ("invalid path '.git/COMMIT_EDITMSG'"). Filtering
        # both sides makes the derived patch a function of the content address
        # and nothing else, whatever the input directory happens to carry.
        cand = probe / "candidate"
        _copy_hashed_tree(tree, cand)
        tree = cand
        # Bytes, not text. `--binary` emits literal binary deltas for any file
        # git cannot treat as text, and decoding those as UTF-8 raises -- which
        # would refuse an artifact for containing, say, a `.npz` fixture. The
        # patch is a byte stream; handling it as one is the only correct option.
        proc = subprocess.run(
            ["git", "diff", "--no-index", "--binary", "--no-color",
             "--src-prefix=a/", "--dst-prefix=b/", str(ref), str(tree)],
            capture_output=True)
        # 0 = identical (the seed's usual case: an empty patch is correct and
        # applies cleanly), 1 = differ. Anything else is a real git failure.
        if proc.returncode not in (0, 1):
            why = proc.stderr.decode("utf-8", "replace").strip()[:200]
            return False, f"git diff --no-index failed: {why}"
        # A created file names the DESTINATION on both sides of its header and a
        # deleted file names the source on both, so rewriting only `a<ref>/` and
        # `b<tree>/` leaves half of those headers absolute. Four substitutions,
        # not two.
        body = proc.stdout
        for prefix, root in ((b"a", ref), (b"b", ref), (b"a", tree), (b"b", tree)):
            body = body.replace(prefix + str(root).encode() + b"/", prefix + b"/")
        dest = work / "baseline.patch"
        dest.write_bytes(body)
        if not body.strip():
            return True, "empty: the tree is byte-identical to the immutable baseline"

        replay = probe / WORKSPACE
        _copy_hashed_tree(baseline, replay)
        ok, why = _apply_patch(dest, replay)
        if not ok:
            dest.unlink()
            return False, f"derived patch does not apply: {why}"
        actual = tree_hash(str(replay))
        if actual != source_hash:
            dest.unlink()
            return False, (f"derived patch reproduces {actual[:16]}, not "
                           f"{source_hash[:16]}")
    finally:
        shutil.rmtree(str(probe), ignore_errors=True)
    return True, "derived from the immutable baseline and verified by reapplication"


def _frozen_in_baseline(path: Any, tree: Path, baseline: Path) -> bool:
    """Is this file byte-identical to the frozen baseline's copy of itself?

    The archive-side policy scan has to exempt the oracle, and every caller
    elsewhere in the workflow does that by passing a hand-maintained list of
    oracle paths through `--immutable`. A list is the wrong instrument here: it
    is per-task, it lives in a prompt, and the failure mode of getting it wrong
    is exempting a candidate file -- a gate that quietly stops gating.

    The frozen baseline already answers the question. A file that is byte-for-
    byte what the immutable baseline ships is frozen, whatever it is named; a
    file that differs, or that the baseline does not have at all, is
    candidate-owned and its findings block. Nothing has to be maintained, and
    being wrong errs toward blocking.
    """
    if not path or not baseline.is_dir():
        return False
    try:
        rel = Path(path).resolve().relative_to(tree.resolve())
    except (ValueError, OSError):
        return False
    twin = baseline / rel
    try:
        if not twin.is_file() or twin.is_symlink():
            return False
        candidate = tree / rel
        if candidate.stat().st_size != twin.stat().st_size:
            return False
        return candidate.read_bytes() == twin.read_bytes()
    except OSError:
        return False


def materialize(archive_dir: Path, baseline: Path, source_hash: str,
                record: Dict[str, Any],
                policy_immutable: List[str] = ()) -> Tuple[bool, str]:
    """Bring `artifacts/<source_hash>/` into existence, or confirm it already is.

    Returns (ok, status). `status` is "reused", "materialized", or the reason it
    could not be done.
    """
    dest = archive_dir / "artifacts" / source_hash
    ws = dest / WORKSPACE
    if ws.is_dir():
        # (71). The address is the identity, so an existing directory is only
        # the artifact if it still hashes to its own name. A tree that drifted
        # under its address is a worse failure than a missing one, because
        # every elite pointing at it reads as durable.
        actual = tree_hash(str(ws))
        if actual == source_hash:
            return True, "reused"
        return False, (f"existing artifact tree hashes {actual[:16]} but is stored "
                       f"under {source_hash[:16]}")

    patch = Path(record["patch"]) if record.get("patch") else None
    # The bootstrap seed has no patch: it is not a change to the baseline, it is
    # the tree itself, so it is materialized by copying `source_tree` rather
    # than by reapplying a diff. The hash check afterwards is identical, and it
    # is the check -- not the route taken to the bytes -- that makes the address
    # an identity (71).
    source_tree = Path(record["source_tree"]) if record.get("source_tree") else None
    if patch is None and source_tree is None:
        return False, "admission carries neither `patch` nor `source_tree`"
    if patch is not None and not patch.is_file():
        return False, f"patch not found: {record.get('patch')!r}"
    if patch is not None and not baseline.is_dir():
        return False, f"immutable baseline not found: {baseline}"
    if patch is None and not source_tree.is_dir():
        return False, f"source tree not found: {record.get('source_tree')!r}"

    staging = archive_dir / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(dir=str(staging), prefix=f"mat_{source_hash[:12]}_"))
    try:
        tree = work / WORKSPACE
        _copy_hashed_tree(baseline if patch is not None else source_tree, tree)
        if patch is not None:
            ok, why = _apply_patch(patch, tree)
            if not ok:
                return False, why
        actual = tree_hash(str(tree))
        if actual != source_hash:
            # The patch is not a faithful record of what was measured. Refusing
            # is the whole point: persisting it would put a wrong tree behind a
            # right-looking address, and every later round would read it as the
            # elite it is not.
            how = "reapplied patch" if patch is not None else "copied source tree"
            return False, (f"{how} hashes {actual[:16]}, claimed "
                           f"{source_hash[:16]}")

        # The tree that is about to become durable is scanned here, not the
        # engineer's directory that was scanned at verify time. Those are the
        # same bytes if everything is right, and this is the step that finds out
        # when it is not -- the archive is what later rounds seed from, so a
        # forbidden dependency reaching it outlives the round that introduced
        # it. Blocking `findings` only; `advisory` is comment text (see the v2
        # note in candidate_policy_scan.py) and refusing on it would make a red
        # receipt normal, which is (53).
        receipt = policy_scan([str(tree)], list(policy_immutable))
        blocking = [f for f in (receipt.get("findings") or [])
                    if not _frozen_in_baseline(f.get("path"), tree, baseline)]
        receipt["exempt_because_identical_to_frozen_baseline"] = [
            f for f in (receipt.get("findings") or []) if f not in blocking]
        receipt["passed_after_baseline_exemption"] = not blocking
        if blocking:
            reasons = "; ".join(f"{f.get('rule')} in {f.get('path')}" for f in blocking[:4])
            return False, (f"candidate policy scan failed on the materialized tree "
                           f"({len(blocking)} blocking finding(s)): {reasons}")

        patch_status = "copied from the admission"
        if patch is not None:
            shutil.copyfile(str(patch), str(work / "baseline.patch"))
        else:
            # A patch-less artifact still owes the archive a baseline.patch:
            # canonical promotion reads `artifacts/<hash>/baseline.patch` and
            # does not care how the tree got there. Derived here, and then
            # verified the same way an offspring patch is -- reapplied to a
            # fresh copy of the baseline and hashed -- because a patch nobody
            # has applied is a claim, and this one would not be tested until
            # the moment something tried to promote the elite.
            ok, patch_status = _derive_baseline_patch(work, tree, baseline, source_hash)
            if not ok:
                _write_json(work / "baseline_patch_unavailable.json",
                            {"reason": patch_status, "source_hash": source_hash})

        _write_json(work / "hash_receipt.json",
                    {"schema": HASH_SCHEMA, "source_hash": source_hash})
        _write_json(work / "policy_receipt.json", receipt)
        if record.get("policy_receipt") and Path(record["policy_receipt"]).is_file():
            # Kept beside ours, not over it: the upstream receipt describes the
            # engineer's workspace and this one describes the artifact.
            shutil.copyfile(record["policy_receipt"], str(work / "upstream_policy_receipt.json"))
        _write_json(work / "ARTIFACT.json", {
            "schema": ARTIFACT_SCHEMA,
            "source_hash": source_hash,
            "workspace": f"<artifact>/{WORKSPACE}",
            "baseline_patch": ("baseline.patch" if (work / "baseline.patch").is_file()
                               else None),
            "baseline_patch_status": patch_status,
            "hash_receipt": "hash_receipt.json",
            "policy_receipt": ("policy_receipt.json"
                               if (work / "policy_receipt.json").is_file() else None),
            "materialization": ("baseline_plus_patch" if patch is not None
                                else "source_snapshot"),
            "baseline_patch_verified": {
                "apply": ("git apply -p1 (--check passed, then applied)"
                          if patch is not None else f"copied from {source_tree}"),
                "matches_source_hash": True,
                "reapplied_source_hash": actual,
            },
            "immutable_baseline": str(baseline) if patch is not None else None,
            "parent_workspace": record.get("parent_workspace"),
            "parent_elite_id": record.get("parent_elite_id"),
            "operator": record.get("operator"),
            "generation": record.get("generation"),
            "created_utc": _utc(),
            "written_by": "qd_persist_manifest.py",
        })
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.rename(str(work), str(dest))
        _fsync_dir(dest.parent)
        work = None  # renamed away; nothing to clean up
        return True, "materialized"
    finally:
        if work is not None and work.exists():
            shutil.rmtree(str(work), ignore_errors=True)


def persist(payload: Dict[str, Any]) -> Dict[str, Any]:
    archive_dir = Path(payload["archive_dir"])
    baseline = Path(payload.get("immutable_baseline") or "")
    fields = json.loads(json.dumps(payload.get("manifest") or {}))  # never mutate the caller's
    updates = json.loads(json.dumps(payload.get("cell_updates") or {}))
    admissions: List[Dict[str, Any]] = list(payload.get("admissions") or [])

    archive_dir.mkdir(parents=True, exist_ok=True)

    # Two shapes, one write. `manifest.cells` is the whole projection and
    # replaces what is on disk; `cell_updates` is this generation's admitted
    # entries merged onto it. The merge shape is the one the lane uses, because
    # the payload then scales with the round rather than with the archive --
    # and because "carry the unchanged cells forward" becomes something the
    # code does rather than something an agent is asked to remember (96).
    on_disk: Dict[str, Any] = {}
    manifest_path = archive_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            on_disk = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            on_disk = {}
    # Artifacts first, always. Nothing may enter the manifest whose bytes are
    # not already on disk under their own address.
    # One materialization per content address, however many elites share it.
    by_hash: Dict[str, Dict[str, Any]] = {}
    for a in admissions:
        by_hash.setdefault(a["source_hash"], a)

    artifacts: Dict[str, str] = {}
    failures: List[Dict[str, Any]] = []
    bad_hashes = set()
    for source_hash, record in sorted(by_hash.items()):
        ok, status = materialize(archive_dir, baseline, source_hash, record,
                                 payload.get("policy_immutable") or [])
        artifacts[source_hash] = status
        if not ok:
            bad_hashes.add(source_hash)
            failures.append({"source_hash": source_hash, "reason": status})

    manifest = dict(on_disk)
    manifest.update(fields)
    dropped = []
    if "cells" in fields:
        # Full projection. A cell whose artifact did not materialize must not
        # appear: writing it would produce a dangling elite -- durable-looking,
        # unreadable, and indistinguishable from a real one until something
        # tries to seed from it.
        cells = dict(manifest.get("cells") or {})
        for cell_key, entry in list(cells.items()):
            if entry.get("source_hash") in bad_hashes:
                dropped.append(entry.get("elite_id"))
                del cells[cell_key]
        manifest["cells"] = cells
    else:
        # Merge. A refused update is simply not applied, which leaves the
        # incumbent where it was. Deleting the key instead would evict an elite
        # that nothing ever beat -- the challenger's failure taking the
        # incumbent with it.
        cells = dict(manifest.get("cells") or {})
        for cell_key, entry in updates.items():
            if entry.get("source_hash") in bad_hashes:
                dropped.append(entry.get("elite_id"))
                continue
            cells[cell_key] = entry
        manifest["cells"] = cells
    if dropped:
        failures.append({"reason": "cells dropped from manifest (artifact refused)",
                         "elite_ids": sorted(x for x in dropped if x)})

    gb = manifest.get("global_best")
    if gb and gb.get("source_hash") in bad_hashes:
        manifest["global_best"] = None
        failures.append({"reason": "global_best cleared (its artifact was refused)",
                         "elite_id": gb.get("elite_id")})

    persisted = [a["elite_id"] for a in admissions
                 if a["source_hash"] not in bad_hashes
                 and (manifest.get("cells") or {}).get(a.get("cell"), {}).get("elite_id")
                 == a["elite_id"]]

    manifest["updated_utc"] = _utc()
    manifest["written_by"] = "qd_persist_manifest.py"
    _write_json(manifest_path, manifest)
    if manifest.get("global_best"):
        _write_json(archive_dir / "global_best.json", manifest["global_best"])
    _fsync_dir(archive_dir)

    return {"schema": "geak.qd-persist/v1",
            "persisted_elite_ids": persisted,
            "artifacts": artifacts,
            "failures": failures,
            "verification": verify_written(manifest_path),
            "manifest": str(manifest_path)}


def verify_written(manifest_path: Path) -> Dict[str, Any]:
    """Re-read what was just written and describe it in checkable numbers.

    The caller of this module is an agent running a command, and an agent can
    report a success it did not have. So the receipt does not say "it worked";
    it says how many cells, which cell keys (as one digest), and which
    generation are actually on disk right now. `kernel_lane.js` holds the same
    archive in memory and can compare, which turns "the persister said fine"
    into an assertion. Without this the only failure mode (96) closed is the
    honest one.
    """
    try:
        written = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        return {"readable": False, "error": str(exc)[:200]}
    cells = written.get("cells") or {}
    keys = "\n".join(sorted(cells))
    elite_ids = "\n".join(sorted(str(e.get("elite_id")) for e in cells.values()))
    return {
        "readable": True,
        "generation": written.get("generation"),
        "cells": len(cells),
        "cell_keys_sha256": hashlib.sha256(keys.encode()).hexdigest(),
        "elite_ids_sha256": hashlib.sha256(elite_ids.encode()).hexdigest(),
        # The sha256 digests above are for a human or a python caller comparing
        # two archives. These two are the ones kernel_lane.js can actually
        # recompute: the workflow JS sandbox has no crypto, so a digest it
        # cannot reproduce is a digest it must take on faith -- which is the
        # thing (96) is about.
        "cell_keys_fnv1a32": fnv1a32(keys),
        "elite_ids_fnv1a32": fnv1a32(elite_ids),
        "challengers": len(written.get("challengers") or {}),
        "capsules": len(written.get("capsules") or {}),
        "lineage": len(written.get("lineage") or {}),
        "recent_transitions": len(written.get("recent_transitions") or []),
        "bytes": manifest_path.stat().st_size,
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--payload", help="read the payload from this file instead of stdin")
    ap.add_argument("--expect-checksum", dest="expect",
                    help="<fnv1a32>:<bytes> of the payload text, as computed by its producer")
    args = ap.parse_args(argv)
    raw = Path(args.payload).read_text() if args.payload else sys.stdin.read()

    if args.expect:
        text = raw.rstrip()
        got = f"{fnv1a32(text)}:{len(text.encode())}"
        if got != args.expect.strip():
            # Nothing is written. A payload that does not match what its
            # producer emitted is not a smaller update, it is an unknown one,
            # and persisting it would put an unverified archive behind a
            # verified-looking receipt.
            json.dump({"schema": "geak.qd-persist/v1", "persisted_elite_ids": [],
                       "artifacts": {}, "manifest": None,
                       "failures": [{"reason": "payload checksum mismatch: the text on disk is "
                                               "not the text the lane emitted",
                                     "expected": args.expect.strip(), "got": got}],
                       "verification": {"readable": False,
                                        "error": "refused before writing: payload checksum mismatch"}},
                      sys.stdout, indent=1, sort_keys=True)
            sys.stdout.write("\n")
            return 3

    receipt = persist(json.loads(raw))
    json.dump(receipt, sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")
    # Exit 0 even on partial failure: the receipt, not the exit code, is what
    # the caller reads, and a nonzero exit here would be read as "nothing was
    # persisted" when some of it was. Exit 3 above is the exception, and it is
    # the one case where nothing was written at all.
    return 0


if __name__ == "__main__":
    sys.exit(main())
