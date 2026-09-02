from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Sequence

from .schema import Route


RUNTIME_TABLE_VERSION = "geak.runtime_table/v1"


class RuntimeTableError(ValueError):
    pass


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp.%d" % os.getpid())
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(str(temp), str(path))


@contextlib.contextmanager
def _lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class RuntimeTableStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @staticmethod
    def empty() -> Dict[str, Any]:
        return {
            "schema": RUNTIME_TABLE_VERSION,
            "revision": 0,
            "updated_at": None,
            "routes": {},
            "history": [],
        }

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return self.empty()
        value = json.loads(self.path.read_text())
        if value.get("schema") != RUNTIME_TABLE_VERSION:
            raise RuntimeTableError("unsupported runtime table schema %r" % value.get("schema"))
        return value

    def update_route(
        self,
        route: Route,
        entries: Sequence[Mapping[str, Any]],
        run_id: str,
        receipts_root: Path,
    ) -> Dict[str, Any]:
        if not entries:
            raise RuntimeTableError("cannot publish an empty route selection")
        with _lock(self.path.with_suffix(self.path.suffix + ".lock")):
            value = self.load()
            revision = int(value.get("revision", 0)) + 1
            previous = value.get("routes", {}).get(route.name)
            now = dt.datetime.now(dt.timezone.utc).isoformat()
            ranked = []
            for rank, entry in enumerate(entries):
                row = dict(entry)
                row["rank"] = rank
                ranked.append(row)
            value.setdefault("history", []).append({
                "revision": revision,
                "route": route.name,
                "previous_config_ids": [x["config_id"] for x in previous.get("entries", [])] if previous else [],
                "new_config_ids": [x["config_id"] for x in ranked],
                "run_id": run_id,
                "updated_at": now,
            })
            value.setdefault("routes", {})[route.name] = {
                "shape": route.to_dict(),
                "entries": ranked,
                "run_id": run_id,
                "receipts_root": str(receipts_root),
                "updated_at": now,
                "revision": revision,
            }
            value["revision"] = revision
            value["updated_at"] = now
            _atomic_write(self.path, value)
            snapshot = self.path.with_name(self.path.name + ".v%d.json" % revision)
            _atomic_write(snapshot, value)
            return value


class RuntimeSelector:
    def __init__(self, table: Mapping[str, Any]) -> None:
        if table.get("schema") != RUNTIME_TABLE_VERSION:
            raise RuntimeTableError("unsupported runtime table")
        self.table = table

    @classmethod
    def from_path(cls, path: Path) -> "RuntimeSelector":
        return cls(RuntimeTableStore(path).load())

    def select(self, route: str, rank: int = 0) -> Mapping[str, Any]:
        try:
            entries = self.table["routes"][route]["entries"]
        except KeyError as exc:
            raise RuntimeTableError("route %r is absent from runtime table" % route) from exc
        if rank < 0 or rank >= len(entries):
            raise RuntimeTableError("route %r has no rank %d" % (route, rank))
        return entries[rank]
