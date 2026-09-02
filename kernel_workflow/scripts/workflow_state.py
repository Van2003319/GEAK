#!/usr/bin/env python3
"""Versioned, atomic persistence for a kernel-lane continuation.

The workflow runtime cannot read files itself, so agents invoke this small deterministic
capability.  It owns the on-disk schema and hashes the exact canonical source, assembly,
and durable binary artifacts before publishing STATE.json.  A resume is valid only when
both the schema and all three identities match.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import os
import shutil
import stat
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "geak.workflow-state/v2"
SCHEMA_VERSION = 2

EXCLUDED_DIRS = frozenset({
    ".git", "build", "__pycache__", ".torch_ext", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".rocprofv3", "node_modules", "dist", "logs", "reports",
    ".ipynb_checkpoints",
})
EXCLUDED_NAMES = frozenset({
    "worker_result.json", "verification_result.json", "commit_result.json",
    "policy_prebuild.json", "policy_postbuild.json",
})
SOURCE_SUFFIXES = frozenset({
    ".py", ".pyi", ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx",
    ".cu", ".cuh", ".hip", ".inl", ".inc", ".ll", ".mlir", ".td", ".cl",
    ".cmake", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".sh",
})
SOURCE_NAMES = frozenset({"CMakeLists.txt", "Makefile", "GNUmakefile", "meson.build"})
ASSEMBLY_SUFFIXES = frozenset({".s", ".asm"})
# Deliberately excludes generated .o/.so files.  The continuation tar drops build products so a
# resumed wave rebuilds on its own machine.  Durable code objects checked into the canonical tree
# remain identity-bearing and are copied into best/.
BINARY_SUFFIXES = frozenset({".co", ".hsaco", ".cubin", ".fatbin", ".bin"})
REQUIRED_STATE_FIELDS = (
    "cumulative", "cumulative_frame", "wave_local_cumulative", "insights", "ledger",
    "asm_ledger", "keypoint_citations", "best_per_case", "rounds", "bottleneck_now",
    "search_phase", "phase_stall", "cycle", "phase_log", "deep_attempts", "last_round",
    "process_log",
)
SEARCH_PHASES = frozenset({"phase1_source", "phase2_asm", "deep_explore"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class StateError(RuntimeError):
    """The state cannot be persisted or trusted."""


def _excluded_dir(name: str) -> bool:
    return name in EXCLUDED_DIRS or name.endswith(".egg-info")


def _kind_for(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in ASSEMBLY_SUFFIXES:
        return "asm"
    if suffix in BINARY_SUFFIXES:
        return "binary"
    if suffix in SOURCE_SUFFIXES or path.name in SOURCE_NAMES:
        return "source"
    return None


def _entries(root: Path) -> Iterable[tuple[str, str, bytes]]:
    root = root.resolve()
    if not root.is_dir():
        raise StateError(f"canonical root is not a directory: {root}")

    def walk(directory: Path) -> Iterable[tuple[str, str, bytes]]:
        try:
            children = sorted(directory.iterdir(), key=lambda item: os.fsencode(item.name))
        except OSError as exc:
            raise StateError(f"cannot list {directory}: {exc}") from exc
        for child in children:
            try:
                mode = child.lstat().st_mode
            except OSError as exc:
                raise StateError(f"cannot stat {child}: {exc}") from exc
            if stat.S_ISDIR(mode):
                if not _excluded_dir(child.name):
                    yield from walk(child)
                continue
            if child.name in EXCLUDED_NAMES:
                continue
            kind = _kind_for(child)
            if kind is None:
                continue
            rel = child.relative_to(root).as_posix()
            try:
                if stat.S_ISLNK(mode):
                    target = os.readlink(child).encode("utf-8", "surrogateescape")
                    # The link text alone is not a source identity: an absolute source symlink can
                    # keep the same path while its target bytes change between waves. Preserve the
                    # symlink in best/, but bind the hash to both the target name and dereferenced
                    # regular-file contents. Dangling/non-file links remain deterministic markers.
                    if child.is_file():
                        target_payload = child.read_bytes()
                        payload = target + b"\0target-file\0" + target_payload
                    else:
                        payload = target + b"\0target-unreadable-or-nonfile"
                    entry_kind = "symlink"
                elif stat.S_ISREG(mode):
                    payload = child.read_bytes()
                    entry_kind = "file"
                else:
                    continue
            except OSError as exc:
                raise StateError(f"cannot read {child}: {exc}") from exc
            yield kind, rel, entry_kind.encode() + b"\0" + payload

    yield from walk(root)


def canonical_hashes(root: os.PathLike[str] | str) -> dict[str, dict[str, Any]]:
    """Return independent deterministic identities for source, asm, and durable binaries."""
    grouped: dict[str, list[tuple[str, bytes]]] = {"source": [], "asm": [], "binary": []}
    for kind, rel, payload in _entries(Path(root)):
        grouped[kind].append((rel, payload))
    result: dict[str, dict[str, Any]] = {}
    for kind in ("source", "asm", "binary"):
        digest = hashlib.sha256()
        files: list[str] = []
        for rel, payload in sorted(grouped[kind], key=lambda item: os.fsencode(item[0])):
            files.append(rel)
            digest.update(rel.encode("utf-8", "surrogateescape"))
            digest.update(b"\0")
            digest.update(hashlib.sha256(payload).digest())
            digest.update(b"\1")
        result[kind] = {
            "algorithm": "sha256",
            "sha256": digest.hexdigest(),
            "files": len(files),
            "present": bool(files),
        }
    return result


def _validate_payload(payload: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_STATE_FIELDS if field not in payload]
    if missing:
        raise StateError("state payload missing required field(s): " + ", ".join(missing))
    phase = payload["search_phase"]
    if phase not in SEARCH_PHASES:
        raise StateError(f"search_phase must be one of {sorted(SEARCH_PHASES)}")
    for field in ("phase_stall", "cycle", "last_round"):
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise StateError(f"{field} must be a non-negative integer")
    if payload["cycle"] < 1:
        raise StateError("cycle must be at least 1")
    for field in ("cumulative", "wave_local_cumulative"):
        value = payload[field]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise StateError(f"{field} must be a finite positive number")
    if not isinstance(payload["cumulative_frame"], str) or not payload["cumulative_frame"].strip():
        raise StateError("cumulative_frame must be a non-empty string")
    if not isinstance(payload["bottleneck_now"], str):
        raise StateError("bottleneck_now must be a string")
    for field in ("insights", "ledger", "asm_ledger", "keypoint_citations", "best_per_case",
                  "rounds", "phase_log", "deep_attempts", "process_log"):
        if not isinstance(payload[field], list):
            raise StateError(f"{field} must be an array")
    if not payload["phase_log"]:
        raise StateError("phase_log must contain the active phase entry")
    active = payload["phase_log"][-1]
    if (not isinstance(active, dict) or active.get("phase") != phase
            or active.get("cycle") != payload["cycle"]):
        raise StateError("phase_log tail must match search_phase and cycle")


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
            fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _copy_ignore(_directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if _excluded_dir(name):
            ignored.add(name)
        elif name in EXCLUDED_NAMES or name.endswith((".o", ".obj", ".so", ".a", ".pyc", ".log")):
            ignored.add(name)
    return ignored


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _atomic_symlink(path: Path, target: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        os.symlink(target, tmp)
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _install_alias(state_root: Path, name: str, target: str) -> None:
    """Install a stable alias without deleting a pre-generation legacy snapshot."""
    path = state_root / name
    if path.is_symlink() and os.readlink(path) == target:
        return
    if path.exists() or path.is_symlink():
        backup = state_root / f"legacy.{name.replace('/', '_')}.{time.time_ns()}.{os.getpid()}"
        os.replace(path, backup)
    _atomic_symlink(path, target)


def _generation_state(
    payload: Mapping[str, Any], hashes: Mapping[str, object], generation: str,
) -> dict[str, Any]:
    state = dict(payload)
    state.update({
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "generation": generation,
        "canonical_hashes": hashes,
        "canonical_source_hash": hashes["source"]["sha256"],
        "canonical_asm_hash": hashes["asm"]["sha256"],
        "canonical_binary_hash": hashes["binary"]["sha256"],
        "updated_unix_ns": time.time_ns(),
    })
    return state


def persist_state(
    state_dir: os.PathLike[str] | str,
    canonical: os.PathLike[str] | str,
    payload: Mapping[str, Any],
    *,
    refresh_best: bool = False,
) -> dict[str, Any]:
    """Publish one immutable best+state generation, then atomically switch ``current``.

    Every barrier snapshots the canonical tree, including the first barrier and a round that did
    not bank.  Consequently ``STATE.json`` can never describe bytes from a different ``best``:
    the two live in one generation and the only commit point is one symlink replacement.
    ``refresh_best`` remains in the CLI for compatibility and is reported as the caller's intent;
    it no longer weakens an unbanked checkpoint into state-without-best.
    """
    _validate_payload(payload)
    state_root = Path(state_dir)
    canonical_root = Path(canonical)
    state_root.mkdir(parents=True, exist_ok=True)
    generations = state_root / "generations"
    generations.mkdir(exist_ok=True)
    generation = f"g{time.time_ns()}-{os.getpid()}-{uuid.uuid4().hex}"
    tmp = generations / f".{generation}.tmp"
    final = generations / generation
    hashes = canonical_hashes(canonical_root)
    try:
        tmp.mkdir()
        best = tmp / "best"
        shutil.copytree(canonical_root, best, symlinks=True, ignore=_copy_ignore)
        copied = canonical_hashes(best)
        if copied != hashes:
            raise StateError(
                "generation best copy hash mismatch; refusing to switch current: "
                + json.dumps({"canonical": hashes, "best": copied}, sort_keys=True)
            )
        state = _generation_state(payload, hashes, generation)
        _atomic_json(tmp / "STATE.json", state)
        _fsync_dir(tmp)
        os.replace(tmp, final)
        _fsync_dir(generations)
        _atomic_symlink(state_root / "current", f"generations/{generation}")
        # Stable compatibility paths resolve through the single atomic pointer.  They are aliases,
        # never independently replaced payloads.
        _install_alias(state_root, "best", "current/best")
        _install_alias(state_root, "STATE.json", "current/STATE.json")
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return {
        "ok": True,
        "schema": SCHEMA,
        "state_file": str(state_root / "STATE.json"),
        "canonical": str(state_root / "best"),
        "generation": generation,
        "current": str(state_root / "current"),
        "best_refreshed": True,
        "refresh_best_requested": bool(refresh_best),
        "canonical_hashes": hashes,
    }


def inspect_state(state_dir: os.PathLike[str] | str) -> dict[str, Any]:
    """Return exactly one continuation state: ``first``, ``resume``, or ``invalid``."""
    root = Path(state_dir)
    current = root / "current"
    if current.exists() or current.is_symlink():
        try:
            generation = current.resolve(strict=True)
        except OSError as exc:
            return {"ok": False, "continuation_state": "invalid", "status": "invalidated",
                    "invalidate_reason": f"current pointer is unreadable: {exc}",
                    "state_dir": str(root.resolve())}
        state_file, canonical = generation / "STATE.json", generation / "best"
        source = "generation"
    else:
        state_file, canonical = root / "STATE.json", root / "best"
        have_state = state_file.exists() or state_file.is_symlink()
        have_best = canonical.exists() or canonical.is_symlink()
        if not have_state and not have_best:
            return {"ok": True, "continuation_state": "first", "status": "first",
                    "invalidate_reason": "", "state_dir": str(root.resolve())}
        if have_state != have_best:
            return {"ok": False, "continuation_state": "invalid", "status": "invalidated",
                    "invalidate_reason": "partial continuation: STATE.json and best must both exist",
                    "state_dir": str(root.resolve())}
        source = "legacy"
    if not state_file.is_file() or not canonical.is_dir():
        return {"ok": False, "continuation_state": "invalid", "status": "invalidated",
                "invalidate_reason": "continuation generation lacks STATE.json or best",
                "state_dir": str(root.resolve()), "source": source}
    try:
        state = load_state(state_file)
        verdict = validate_state(state, canonical)
    except StateError as exc:
        verdict = {"valid": False, "status": "invalidated",
                   "invalidate_reason": str(exc), "mismatches": []}
    result = {
        **verdict,
        "ok": bool(verdict.get("valid")),
        "continuation_state": "resume" if verdict.get("valid") else "invalid",
        "state_dir": str(root.resolve()),
        "state_file": str(state_file.resolve()),
        "canonical": str(canonical.resolve()),
        "source": source,
    }
    return result


def load_state(path: os.PathLike[str] | str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StateError(f"cannot read state {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StateError("STATE.json root must be an object")
    return value


def validate_state(state: Mapping[str, Any], canonical: os.PathLike[str] | str) -> dict[str, Any]:
    reasons: list[str] = []
    if state.get("schema") != SCHEMA or state.get("schema_version") != SCHEMA_VERSION:
        reasons.append(
            f"schema_mismatch(expected={SCHEMA}/{SCHEMA_VERSION}, got="
            f"{state.get('schema')!r}/{state.get('schema_version')!r})"
        )
    try:
        _validate_payload(state)
    except StateError as exc:
        reasons.append(f"state_shape_invalid({exc})")
    current = canonical_hashes(canonical)
    expected = state.get("canonical_hashes")
    mismatches: list[dict[str, Any]] = []
    if not isinstance(expected, dict):
        reasons.append("canonical_hashes_missing")
    else:
        for kind in ("source", "asm", "binary"):
            wanted = expected.get(kind)
            got = current[kind]
            malformed = not isinstance(wanted, dict)
            if not malformed:
                files = wanted.get("files")
                present = wanted.get("present")
                malformed = (
                    wanted.get("algorithm") != "sha256"
                    or not isinstance(wanted.get("sha256"), str)
                    or SHA256_RE.fullmatch(wanted["sha256"]) is None
                    or isinstance(files, bool) or not isinstance(files, int) or files < 0
                    or not isinstance(present, bool) or present != (files > 0)
                )
            alias = state.get(f"canonical_{kind}_hash")
            if malformed:
                reasons.append(f"canonical_hash_record_invalid({kind})")
            elif alias != wanted["sha256"]:
                reasons.append(f"canonical_hash_alias_mismatch({kind})")
            if (malformed or wanted.get("sha256") != got["sha256"]
                    or wanted.get("files") != got["files"]
                    or wanted.get("present") != got["present"]):
                mismatches.append({
                    "artifact": kind,
                    "expected": wanted.get("sha256") if isinstance(wanted, dict) else None,
                    "actual": got["sha256"],
                    "expected_files": wanted.get("files") if isinstance(wanted, dict) else None,
                    "actual_files": got["files"],
                })
        if mismatches:
            reasons.append("canonical_hash_mismatch(" + ",".join(m["artifact"] for m in mismatches) + ")")
    valid = not reasons
    return {
        "valid": valid,
        "status": "valid" if valid else "invalidated",
        "invalidate_reason": "; ".join(reasons),
        "mismatches": mismatches,
        "canonical_hashes": current,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    hash_cmd = sub.add_parser("hash", help="hash a canonical tree")
    hash_cmd.add_argument("--canonical", required=True)

    inspect = sub.add_parser("inspect", help="classify continuation as first/resume/invalid")
    inspect.add_argument("--state-dir", required=True)
    inspect.add_argument("--receipt", default="")

    persist = sub.add_parser("persist", help="atomically publish one best+state generation")
    persist.add_argument("--state-dir", required=True)
    persist.add_argument("--canonical", required=True)
    persist.add_argument("--payload-file", required=True)
    persist.add_argument("--refresh-best", action="store_true")

    validate = sub.add_parser("validate", help="validate STATE.json against best/")
    validate.add_argument("--state-file", required=True)
    validate.add_argument("--canonical", required=True)
    validate.add_argument("--receipt", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = (Path(args.receipt)
               if args.command in ("validate", "inspect") and args.receipt else None)
    try:
        if args.command == "hash":
            result: dict[str, Any] = {"ok": True, "schema": SCHEMA,
                                      "canonical_hashes": canonical_hashes(args.canonical)}
            exit_code = 0
        elif args.command == "persist":
            payload = load_state(args.payload_file)
            result = persist_state(args.state_dir, args.canonical, payload,
                                   refresh_best=args.refresh_best)
            exit_code = 0
        elif args.command == "inspect":
            result = inspect_state(args.state_dir)
            exit_code = 0 if result["continuation_state"] in ("first", "resume") else 2
        else:
            state = load_state(args.state_file)
            result = validate_state(state, args.canonical)
            result["state_file"] = os.path.abspath(args.state_file)
            exit_code = 0 if result["valid"] else 2
    except StateError as exc:
        result = {
            "ok": False,
            "valid": False,
            "status": "invalidated",
            "invalidate_reason": str(exc),
            "mismatches": [],
        }
        exit_code = 2
    if receipt is not None:
        try:
            _atomic_json(receipt, result)
        except OSError as exc:
            result = {
                "ok": False,
                "valid": False,
                "status": "invalidated",
                "invalidate_reason": f"cannot write validation receipt {receipt}: {exc}",
                "mismatches": result.get("mismatches", []),
            }
            exit_code = 2
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
