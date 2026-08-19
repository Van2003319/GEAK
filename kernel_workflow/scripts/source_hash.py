#!/usr/bin/env python3
"""Deterministic source-tree hashing.

`isa_capture.py` records the `source_hash` of the tree a code object came out
of, so an archived disassembly can be tied back to the exact source that
produced it. That identity has to be computed the same way every time, from
the filesystem, with no agent in the loop:

  `tree_hash` walks a candidate source tree with the SAME deterministic
  exclusions kernel_lane.js's fresh-workspace tar-copy uses (.git, build
  artifacts, __pycache__, .torch_ext, ...), never follows symlinks, and
  returns a single sha256 over a canonical listing of (relative path, entry
  kind, content hash) -- identical trees hash identically regardless of
  directory walk order or the OS's filename byte order.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

SCHEMA = "geak.source-hash/v1"


class SourceHashError(OSError):
    """The requested source tree could not be read completely and safely."""


# Mirrors kernel_lane.js's comment on its fresh-workspace tar-copy: "EXCLUDES
# build artifacts (.git/build/__pycache__/.torch_ext/...)". Directory names are
# matched exactly (not as globs) at any depth, so a nested `build/` is excluded
# just as reliably as a top-level one.
DEFAULT_EXCLUDED_DIRS: frozenset[str] = frozenset({
    ".git", "build", "__pycache__", ".torch_ext", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".rocprofv3", "node_modules", "dist", "logs", "reports",
    "*.egg-info", ".ipynb_checkpoints",
})
DEFAULT_EXCLUDED_FILE_SUFFIXES: tuple[str, ...] = (
    ".o", ".obj", ".so", ".a", ".pyc", ".log",
)
DEFAULT_EXCLUDED_FILE_NAMES: frozenset[str] = frozenset({
    "worker_result.json", "policy_prebuild.json", "policy_postbuild.json",
})


def _excluded(name: str, excluded_dirs: frozenset[str]) -> bool:
    if name in excluded_dirs:
        return True
    return name.endswith(".egg-info") and "*.egg-info" in excluded_dirs


def _iter_entries(root: Path, excluded_dirs: frozenset[str]) -> Iterable[tuple[str, str, bytes | None]]:
    """Yield (relative_posix_path, kind, payload) for every non-excluded entry.

    kind is "file", "dir", or "symlink"; payload is the file's raw bytes for
    "file", the (undereferenced) link target encoded as utf-8 for "symlink",
    and None for "dir". Symlinks are reported but never followed, so a link
    that escapes the tree can neither be hashed as its target's content nor
    silently skipped -- it shows up as its own distinct, auditable entry.
    """
    root = root.resolve()
    if not root.exists():
        raise SourceHashError(f"source root does not exist: {root}")
    if not (root.is_file() or root.is_dir()):
        raise SourceHashError(f"source root is not a regular file or directory: {root}")

    def walk(directory: Path) -> Iterable[tuple[str, str, bytes | None]]:
        try:
            children = sorted(directory.iterdir(), key=lambda p: os.fsencode(p.name))
        except OSError as exc:
            raise SourceHashError(f"cannot list {directory}: {exc}") from exc
        for child in children:
            rel = child.relative_to(root).as_posix()
            try:
                st = child.lstat()
            except OSError as exc:
                raise SourceHashError(f"cannot stat {child}: {exc}") from exc
            import stat as stat_mod
            if stat_mod.S_ISLNK(st.st_mode):
                try:
                    target = os.readlink(child)
                except OSError as exc:
                    raise SourceHashError(f"cannot read symlink {child}: {exc}") from exc
                yield rel, "symlink", target.encode("utf-8", "surrogateescape")
                continue
            if stat_mod.S_ISDIR(st.st_mode):
                if _excluded(child.name, excluded_dirs):
                    continue
                yield rel, "dir", None
                yield from walk(child)
                continue
            if stat_mod.S_ISREG(st.st_mode):
                if (child.name in DEFAULT_EXCLUDED_FILE_NAMES
                        or child.name.endswith(DEFAULT_EXCLUDED_FILE_SUFFIXES)
                        or child.name.endswith((".hipify.cpp", ".hipified.cpp", ".hipify.cu"))):
                    continue
                try:
                    yield rel, "file", child.read_bytes()
                except OSError as exc:
                    raise SourceHashError(f"cannot read {child}: {exc}") from exc

    if root.is_file():
        try:
            yield root.name, "file", root.read_bytes()
        except OSError as exc:
            raise SourceHashError(f"cannot read {root}: {exc}") from exc
        return
    yield from walk(root)


def tree_hash(root: os.PathLike[str] | str, *, extra_excluded_dirs: Sequence[str] = ()) -> str:
    """A deterministic sha256 over `root`'s content, ignoring build/VCS noise.

    Two trees with byte-identical tracked content hash identically no matter
    the traversal order, the host's locale, or which directories the OS
    happens to report first. Excluded-directory contents never influence the
    result at all -- adding, removing, or editing files under __pycache__/
    (etc.) cannot change the hash, matching kernel_lane.js's own
    fresh-workspace-excludes-build-artifacts contract.
    """
    excluded = DEFAULT_EXCLUDED_DIRS | frozenset(extra_excluded_dirs)
    digest = hashlib.sha256()
    for rel, kind, payload in sorted(_iter_entries(Path(root), excluded), key=lambda e: (e[0], e[1])):
        content_hash = hashlib.sha256(payload).hexdigest() if payload is not None else ""
        digest.update("\x00".join([rel, kind, content_hash]).encode("utf-8", "surrogateescape"))
        digest.update(b"\x01")
    return digest.hexdigest()


def _parser():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", help="source tree (or single file) to hash")
    p.add_argument("--exclude", action="append", default=[], help="extra directory name to exclude (repeatable)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    import json
    import sys
    args = _parser().parse_args(argv)
    payload = {"schema": SCHEMA, "root": os.path.abspath(args.root),
               "source_hash": tree_hash(args.root, extra_excluded_dirs=args.exclude)}
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
