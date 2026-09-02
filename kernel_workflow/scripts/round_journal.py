#!/usr/bin/env python3
"""Atomic per-candidate journal for interrupted kernel-lane rounds.

Only explicitly registered engineer completions are recoverable.  Discovery never scans random
round directories for patches: it verifies the immutable artifact and worker-receipt hashes recorded
at completion, then returns unresolved entries for independent verification in the next wave.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import workflow_state  # noqa: E402

SCHEMA = "geak.round-journal/v1"
ENTRY_VERSION = 1
PENDING_STATUSES = frozenset({"engineer_complete", "verified"})
FINAL_STATUSES = frozenset({"integrated", "rejected", "invalidated"})
TRANSITION_STATUSES = PENDING_STATUSES | FINAL_STATUSES


class JournalError(RuntimeError):
    """A journal operation is unsafe or malformed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise JournalError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _regular(path: Path) -> bool:
    try:
        return path.is_file() and not path.is_symlink()
    except OSError:
        return False


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with tmp.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            pass
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JournalError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise JournalError(f"JSON root is not an object: {path}")
    return value


def _entry_key(eval_dir: str, candidate_id: str) -> str:
    if not candidate_id.strip():
        raise JournalError("candidate_id must be non-empty")
    return hashlib.sha256((os.path.abspath(eval_dir) + "\0" + candidate_id).encode()).hexdigest()


def _entry_path(journal_dir: Path, eval_dir: str, candidate_id: str) -> Path:
    return journal_dir / "entries" / f"{_entry_key(eval_dir, candidate_id)}.json"


@contextlib.contextmanager
def _locked(journal_dir: Path) -> Iterator[None]:
    journal_dir.mkdir(parents=True, exist_ok=True)
    lock_path = journal_dir / ".lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _artifact_record(path: Path) -> dict[str, Any]:
    if not _regular(path) or path.stat().st_size <= 0:
        return {"path": str(path), "present": False, "sha256": None, "bytes": 0}
    return {"path": str(path), "present": True, "sha256": _sha256(path),
            "bytes": path.stat().st_size}


def _receipt_record(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not _regular(path):
        raise JournalError(f"worker receipt missing or not a regular file: {path}")
    payload = _read_json(path)
    return ({"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}, payload)


def _summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    keep = (
        "engineer_id", "specialty", "status", "strategy", "structure_fingerprint", "config",
        "focus_files", "patch_file", "mechanism_claims", "target_routes", "continued_from",
    )
    return {key: payload[key] for key in keep if key in payload}


def register(
    journal_dir: os.PathLike[str] | str,
    *,
    eval_dir: str,
    candidate_id: str,
    round_number: int,
    direction: str,
    phase: str,
    artifact: os.PathLike[str] | str,
    worker_receipt: os.PathLike[str] | str,
    canonical: os.PathLike[str] | str,
    deadline_epoch: int = 0,
) -> dict[str, Any]:
    root = Path(journal_dir)
    if round_number < 1:
        raise JournalError("round_number must be at least 1")
    if not direction.strip():
        raise JournalError("direction must be non-empty")
    if phase not in workflow_state.SEARCH_PHASES:
        raise JournalError(f"unsupported search phase: {phase}")
    artifact_info = _artifact_record(Path(artifact))
    try:
        receipt_info, receipt_payload = _receipt_record(Path(worker_receipt))
    except JournalError as exc:
        return {"registered": False, "schema": SCHEMA, "candidate_id": candidate_id,
                "reason": str(exc), "artifact": artifact_info}
    now = time.time_ns()
    entry = {
        "schema": SCHEMA,
        "schema_version": ENTRY_VERSION,
        "entry_id": _entry_key(eval_dir, candidate_id),
        "candidate_id": candidate_id,
        "direction": direction,
        "round": round_number,
        "phase": phase,
        "eval_dir": os.path.abspath(eval_dir),
        "created_unix_ns": now,
        "updated_unix_ns": now,
        "deadline_epoch": deadline_epoch,
        "status": "engineer_complete",
        "artifact": artifact_info,
        "worker_receipt": receipt_info,
        "worker_summary": _summary(receipt_payload),
        "parent_canonical_hashes": workflow_state.canonical_hashes(canonical),
        "verification": {"status": "pending", "receipt": None},
        "integration": {"status": "pending", "receipt": None},
        "history": [{"status": "engineer_complete", "at_unix_ns": now,
                     "reason": "engineer completion atomically registered"}],
    }
    path = _entry_path(root, eval_dir, candidate_id)
    with _locked(root):
        if path.exists():
            old = _read_json(path)
            same_artifact = (old.get("artifact") or {}).get("sha256") == artifact_info.get("sha256")
            same_receipt = (old.get("worker_receipt") or {}).get("sha256") == receipt_info["sha256"]
            if same_artifact and same_receipt:
                return {"registered": True, "idempotent": True, "schema": SCHEMA,
                        "entry": old, "entry_file": str(path)}
            return {"registered": False, "schema": SCHEMA, "candidate_id": candidate_id,
                    "reason": "immutable_registration_conflict", "entry_file": str(path),
                    "expected_artifact_sha256": (old.get("artifact") or {}).get("sha256"),
                    "actual_artifact_sha256": artifact_info.get("sha256")}
        _atomic_json(path, entry)
    return {"registered": True, "idempotent": False, "schema": SCHEMA,
            "entry": entry, "entry_file": str(path)}


def _verify_registered_bytes(entry: Mapping[str, Any]) -> str:
    receipt = entry.get("worker_receipt") or {}
    receipt_path = Path(str(receipt.get("path") or ""))
    if not _regular(receipt_path):
        return "worker_receipt_missing"
    actual_receipt = _sha256(receipt_path)
    if actual_receipt != receipt.get("sha256"):
        return (f"worker_receipt_hash_mismatch(expected={receipt.get('sha256')},"
                f"actual={actual_receipt})")
    artifact = entry.get("artifact") or {}
    if not artifact.get("present"):
        return ""
    path = Path(str(artifact.get("path") or ""))
    if not _regular(path):
        return "artifact_missing"
    actual = _sha256(path)
    if actual != artifact.get("sha256"):
        return f"artifact_hash_mismatch(expected={artifact.get('sha256')},actual={actual})"
    return ""


def _verify_verification_receipt(entry: Mapping[str, Any]) -> str:
    if entry.get("status") != "verified":
        return ""
    receipt = (entry.get("verification") or {}).get("receipt") or {}
    path = Path(str(receipt.get("path") or ""))
    if not _regular(path):
        return "verification_receipt_missing"
    actual = _sha256(path)
    if actual != receipt.get("sha256"):
        return (f"verification_receipt_hash_mismatch(expected={receipt.get('sha256')},"
                f"actual={actual})")
    return ""


def _transition_entry(
    path: Path,
    entry: dict[str, Any],
    *,
    status: str,
    reason: str,
    verification_receipt: str = "",
    integration_receipt: str = "",
) -> dict[str, Any]:
    if status not in TRANSITION_STATUSES:
        raise JournalError(f"unsupported status: {status}")
    integrity = _verify_registered_bytes(entry)
    if status == "integrated":
        integrity = integrity or _verify_verification_receipt(entry)
    if integrity and status != "invalidated":
        status = "invalidated"
        reason = integrity
    verification_path = Path(verification_receipt) if verification_receipt else None
    if verification_path is not None and not _regular(verification_path):
        status = "invalidated"
        reason = "verification_receipt_missing"
    integration_path = Path(integration_receipt) if integration_receipt else None
    if status == "integrated" and (integration_path is None or not _regular(integration_path)):
        # The source may have moved, but without the commit receipt the journal cannot claim it did.
        # Preserve it for independent revalidation rather than manufacturing an integrated record.
        status = "verified"
        reason = "integration_receipt_missing; preserve for next-wave revalidation"
    elif status == "integrated":
        try:
            commit = _read_json(integration_path)
        except JournalError as exc:
            status = "verified"
            reason = f"integration_receipt_unreadable({exc}); preserve for next-wave revalidation"
        else:
            before = commit.get("head_sha_before")
            after = commit.get("head_sha_after")
            if (commit.get("committed") is not True or not isinstance(before, str)
                    or not isinstance(after, str) or not before or not after or before == after):
                status = "verified"
                reason = "integration_receipt_does_not_prove_head_moved; preserve for revalidation"
    now = time.time_ns()
    entry["status"] = status
    entry["updated_unix_ns"] = now
    if verification_path is not None and _regular(verification_path):
        entry["verification"] = {
            "status": "passed" if status in {"verified", "integrated"} else "failed",
            "receipt": {"path": str(verification_path), "sha256": _sha256(verification_path)},
        }
    if status == "integrated":
        receipt = {"path": str(integration_path), "sha256": _sha256(integration_path)}
        entry["integration"] = {"status": "integrated", "receipt": receipt}
    elif status == "verified":
        entry["integration"] = {"status": "pending", "receipt": None, "reason": reason}
    elif status in {"rejected", "invalidated"}:
        entry["integration"] = {"status": status, "receipt": None, "reason": reason}
    entry.setdefault("history", []).append({"status": status, "at_unix_ns": now,
                                             "reason": reason})
    _atomic_json(path, entry)
    return entry


def transition(
    journal_dir: os.PathLike[str] | str,
    *,
    eval_dir: str,
    candidate_id: str,
    status: str,
    reason: str,
    verification_receipt: str = "",
    integration_receipt: str = "",
) -> dict[str, Any]:
    root = Path(journal_dir)
    path = _entry_path(root, eval_dir, candidate_id)
    with _locked(root):
        if not path.exists():
            raise JournalError(f"journal entry not found: {candidate_id}")
        entry = _transition_entry(path, _read_json(path), status=status, reason=reason,
                                  verification_receipt=verification_receipt,
                                  integration_receipt=integration_receipt)
    return {"ok": True, "schema": SCHEMA, "entry": entry, "entry_file": str(path)}


def _entry_files(root: Path) -> list[Path]:
    entries = root / "entries"
    if not entries.is_dir():
        return []
    return sorted((path for path in entries.iterdir()
                   if path.suffix == ".json" and _regular(path)),
                  key=lambda path: os.fsencode(path.name))


def discover(
    journal_dir: os.PathLike[str] | str,
    *,
    canonical: os.PathLike[str] | str,
    current_eval_dir: str,
) -> dict[str, Any]:
    root = Path(journal_dir)
    current_hashes = workflow_state.canonical_hashes(canonical)
    recoverable: list[dict[str, Any]] = []
    invalidated: list[dict[str, Any]] = []
    with _locked(root):
        for path in _entry_files(root):
            try:
                entry = _read_json(path)
            except JournalError as exc:
                invalidated.append({"entry_file": str(path), "reason": f"journal_unreadable({exc})"})
                continue
            if entry.get("schema") != SCHEMA or entry.get("schema_version") != ENTRY_VERSION:
                invalidated.append({"entry_file": str(path), "reason": "journal_schema_mismatch"})
                continue
            if entry.get("status") not in PENDING_STATUSES:
                continue
            if (entry.get("integration") or {}).get("status") not in {None, "pending"}:
                continue
            integrity = _verify_registered_bytes(entry) or _verify_verification_receipt(entry)
            if integrity:
                entry = _transition_entry(path, entry, status="invalidated", reason=integrity)
                invalidated.append({"entry_id": entry.get("entry_id"), "reason": integrity})
                continue
            if not (entry.get("artifact") or {}).get("present"):
                entry = _transition_entry(path, entry, status="rejected",
                                          reason="engineer completed without a candidate artifact")
                invalidated.append({"entry_id": entry.get("entry_id"), "reason": "no_patch"})
                continue
            worker_path = Path(str((entry.get("worker_receipt") or {}).get("path") or ""))
            worker = _read_json(worker_path)
            parent = entry.get("parent_canonical_hashes") or {}
            parent_mismatches = [kind for kind in ("source", "asm", "binary")
                                 if (parent.get(kind) or {}).get("sha256")
                                 != current_hashes[kind]["sha256"]]
            recovered = dict(entry)
            recovered["worker_result"] = worker
            recovered["parent_hash_mismatches"] = parent_mismatches
            recovered["recovery_reason"] = (
                "deadline_or_process_interrupted_after_engineer_completion; independently reverify"
            )
            recoverable.append(recovered)
    recoverable.sort(key=lambda entry: (int(entry.get("round") or 0),
                                        int(entry.get("created_unix_ns") or 0),
                                        str(entry.get("candidate_id") or "")))
    return {
        "ok": True,
        "schema": SCHEMA,
        "recoverable": recoverable,
        "invalidated": invalidated,
        "canonical_hashes": current_hashes,
        "policy": "registered-only; unregistered filesystem artifacts are never harvested",
    }


def settle_round(
    journal_dir: os.PathLike[str] | str,
    *,
    eval_dir: str,
    round_number: int,
    winner_direction: str,
    committed: bool,
    integration_receipt: str = "",
    candidate_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    root = Path(journal_dir)
    changed: list[dict[str, Any]] = []
    candidate_filter = set(candidate_ids or ())
    with _locked(root):
        for path in _entry_files(root):
            entry = _read_json(path)
            if (os.path.abspath(str(entry.get("eval_dir") or "")) != os.path.abspath(eval_dir)
                    or entry.get("round") != round_number
                    or entry.get("status") in FINAL_STATUSES
                    or (candidate_filter and entry.get("candidate_id") not in candidate_filter)):
                continue
            if winner_direction and entry.get("direction") == winner_direction:
                if entry.get("status") != "verified":
                    # A commit receipt cannot promote an engineer-only artifact past independent
                    # verification. Leave it pending so a restarted wave can perform that gate.
                    continue
                if committed:
                    status, reason = "integrated", "round winner committed into canonical"
                else:
                    status, reason = "verified", "winner not committed; preserve for next-wave revalidation"
            elif entry.get("status") == "engineer_complete" and (entry.get("artifact") or {}).get("present"):
                # Verification never reached a durable receipt (API failure/process cut). It is neither
                # a loser nor a winner yet, so leave it discoverable for the next wave.
                continue
            else:
                status, reason = "rejected", "round settled: not the committed winner"
            changed.append(_transition_entry(path, entry, status=status, reason=reason,
                                             integration_receipt=integration_receipt))
    return {"ok": True, "schema": SCHEMA, "settled": len(changed), "entries": changed}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    reg = sub.add_parser("register")
    reg.add_argument("--journal-dir", required=True)
    reg.add_argument("--eval-dir", required=True)
    reg.add_argument("--candidate-id", required=True)
    reg.add_argument("--round", required=True, type=int)
    reg.add_argument("--direction", required=True)
    reg.add_argument("--phase", required=True)
    reg.add_argument("--artifact", required=True)
    reg.add_argument("--worker-receipt", required=True)
    reg.add_argument("--canonical", required=True)
    reg.add_argument("--deadline-epoch", type=int, default=0)

    trans = sub.add_parser("transition")
    trans.add_argument("--journal-dir", required=True)
    trans.add_argument("--eval-dir", required=True)
    trans.add_argument("--candidate-id", required=True)
    trans.add_argument("--status", required=True, choices=sorted(TRANSITION_STATUSES))
    trans.add_argument("--reason", default="")
    trans.add_argument("--verification-receipt", default="")
    trans.add_argument("--integration-receipt", default="")

    disc = sub.add_parser("discover")
    disc.add_argument("--journal-dir", required=True)
    disc.add_argument("--canonical", required=True)
    disc.add_argument("--current-eval-dir", required=True)

    settle = sub.add_parser("settle")
    settle.add_argument("--journal-dir", required=True)
    settle.add_argument("--eval-dir", required=True)
    settle.add_argument("--round", required=True, type=int)
    settle.add_argument("--winner-direction", default="")
    settle.add_argument("--committed", action="store_true")
    settle.add_argument("--integration-receipt", default="")
    settle.add_argument("--candidate-id", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "register":
            result = register(args.journal_dir, eval_dir=args.eval_dir,
                              candidate_id=args.candidate_id, round_number=args.round,
                              direction=args.direction, phase=args.phase, artifact=args.artifact,
                              worker_receipt=args.worker_receipt, canonical=args.canonical,
                              deadline_epoch=args.deadline_epoch)
        elif args.command == "transition":
            result = transition(args.journal_dir, eval_dir=args.eval_dir,
                                candidate_id=args.candidate_id, status=args.status,
                                reason=args.reason,
                                verification_receipt=args.verification_receipt,
                                integration_receipt=args.integration_receipt)
        elif args.command == "discover":
            result = discover(args.journal_dir, canonical=args.canonical,
                              current_eval_dir=args.current_eval_dir)
        else:
            result = settle_round(args.journal_dir, eval_dir=args.eval_dir,
                                  round_number=args.round,
                                  winner_direction=args.winner_direction,
                                  committed=args.committed,
                                  integration_receipt=args.integration_receipt,
                                  candidate_ids=args.candidate_id)
        sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
        return 0 if result.get("ok", result.get("registered", False)) else 2
    except JournalError as exc:
        sys.stdout.write(json.dumps({"ok": False, "schema": SCHEMA, "error": str(exc)},
                                    sort_keys=True, separators=(",", ":")) + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
