#!/usr/bin/env python3
"""Deterministic, GPU-free candidate dependency policy scanner.

The scanner never executes a candidate.  It inspects candidate source, build/link
metadata, snapshots, and artifacts and emits one canonical JSON receipt.  Frozen
oracle/baseline inputs may be named explicitly with ``--immutable``; they are
reported but are not scanned as candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

SCHEMA = "geak.candidate-policy/v2"

# v2 splits comment-only text matches out of `findings` into `advisory`.
#
# WHY. v1 reported the shipped kernel as `passed: false` on every run, on two
# matches that were the words "rocBLAS" and "Tensile" inside explanatory
# comments -- prose *about* not using them. The ledger above carries a standing
# note telling the reader to ignore that result. A gate whose refusal is
# permanently noise is worse than no gate: it teaches whoever runs it that a
# red receipt is normal, and the next red receipt is the real one.
#
# `findings` stays blocking-only, so every role prompt's "any finding fails
# closed" keeps its exact meaning with no prompt edit. Nothing is hidden: a
# comment match still appears, under `advisory`, with the same evidence.
#
# Fail closed on doubt. A match counts as commented only if the file's comment
# syntax is known AND the scan of it completed cleanly; an unterminated comment
# or string literal makes the whole file uncommented, so every match in it
# blocks. Being wrong in that direction costs a false alarm; being wrong in the
# other direction lets a real dependency through, which is what the gate exists
# to stop.

# Line-comment and block-comment syntax by extension. Deliberately a short
# list: an extension not named here gets no masking at all, which is the safe
# default. Notably `.txt`, `.json`, linker response files and build logs are
# absent -- a forbidden library name in one of those is evidence regardless of
# what character precedes it.
_C_FAMILY = frozenset({".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hip", ".cu", ".cuh", ".cl"})
_HASH_FAMILY = frozenset({".py", ".sh", ".bash", ".cmake"})

# Boundaries are deliberately explicit.  In particular, CK is never searched as
# two free-standing letters: ordinary prose such as "check" must not trip it.
def _lib_pattern(name: str, guard: str = "") -> re.Pattern[str]:
    """Match a forbidden library by name, by soname, or by its `-l` link flag.

    The `-l` form was missing until 2026-08-15 and is the one that actually
    links the library: `(?<![A-Za-z0-9_])` fails on the `l` of `-lrocblas`, so
    every linker flag for every library here scanned clean. The existing
    coverage did not catch it because its fixture also contained a bare
    `rocblas_gemm_ex()`, which matched and made the file fail for the other
    reason. A rule tested only alongside a second matching rule is untested.
    """
    return re.compile(rf"(?i)(?<![A-Za-z0-9_])(?:-l\s*)?(?:lib)?{name}{guard}(?:[A-Za-z0-9_]*|\.so(?:\.\d+)*)")


TEXT_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("rocblas", _lib_pattern("rocblas")),
    ("hipblaslt", _lib_pattern("hipblaslt")),
    ("hipblas", _lib_pattern("hipblas", r"(?!lt)")),
    ("tensile", _lib_pattern("tensile")),
    ("miopen", _lib_pattern("miopen")),
    ("composable_kernel", re.compile(
        r"(?ix)(?:\bcomposable[ _-]+kernel\b|(?:^|[/'\"<])ck/(?:include|library|tensor_operation|utility)/|\bnamespace\s+ck\b|\bck::(?:tensor_operation|utility|profiler|library)\b|\blibck(?:\.so(?:\.\d+)*)?\b)")),
    ("torch_matmul", re.compile(r"\btorch\s*\.\s*(?:matmul|mm|bmm)\s*\(")),
    ("torch_linear", re.compile(r"\b(?:torch\s*\.\s*nn\s*\.\s*functional|[Ff])\s*\.\s*linear\s*\(")),
    # `dlmopen`, `dlsym` and `dlvsym` were added 2026-08-15. `dlmopen` was
    # already in SYMBOL_RULES but not here, so the same call was caught in a
    # built ELF and missed in the source that produced it -- the two tables are
    # meant to be two views of one policy, and a rule in only one of them is a
    # hole shaped like whichever surface the scan happens to reach first.
    # `dlsym`/`dlvsym` matter on their own: `dlsym(RTLD_DEFAULT, name)` resolves
    # a symbol out of the already-loaded image with no `dlopen` anywhere, and
    # with the name assembled at runtime nothing else here would see it.
    ("dynamic_loader", re.compile(
        r"(?i)(?:\bctypes\b|\bdl(?:m?open|v?sym)\s*\(|\b(?:ctypes\s*\.\s*)?(?:cdll|pydll|windll)\s*\.|\b(?:load_library|LoadLibrary[AW]?)\s*\()")),
)

# Dynamic symbol tables may omit source-like punctuation, so use exact API
# prefixes here. HIP runtime/compiler/device symbols and MFMA/rocWMMA are absent
# by design and therefore allowed.
SYMBOL_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("rocblas", re.compile(r"(?i)^rocblas(?:_|$)")),
    ("hipblaslt", re.compile(r"(?i)^hipblaslt(?:_|$)")),
    ("hipblas", re.compile(r"(?i)^hipblas(?:_|$)")),
    ("tensile", re.compile(r"(?i)^tensile(?:_|$)")),
    ("miopen", re.compile(r"(?i)^miopen(?:_|$)")),
    ("composable_kernel", re.compile(r"(?i)^(?:ck::|_?ZN?2ck)")),
    # Kept in step with the text rule above; verified against the real built
    # candidate ELF, which imports none of these, so tightening costs nothing.
    ("dynamic_loader", re.compile(r"^(?:dlopen|dlmopen|dlsym|dlvsym)$")),
)

ELF_MAGIC = b"\x7fELF"


@dataclass(frozen=True)
class Finding:
    path: str
    rule: str
    evidence: str
    inspection: str

    def object(self) -> dict[str, str]:
        return {"path": self.path, "rule": self.rule,
                "inspection": self.inspection, "evidence": self.evidence}


def _display(path: Path) -> str:
    """Stable path representation without resolving symlinks."""
    return os.path.normpath(os.path.abspath(os.fspath(path)))


def _clip(value: str, limit: int = 240) -> str:
    value = " ".join(value.replace("\x00", " ").split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _comment_spans(text: str, suffix: str) -> list[tuple[int, int]] | None:
    """Half-open comment ranges in `text`, or None when they cannot be trusted.

    None means "treat the whole file as code". It is returned for an unknown
    extension and for any file whose literals do not close, because a scanner
    that guesses wrong about where a string ends can mask real code -- and
    masked code is exactly the failure this gate must not have.
    """
    line = "//" if suffix in _C_FAMILY else "#" if suffix in _HASH_FAMILY else None
    if line is None:
        return None
    block = suffix in _C_FAMILY
    spans: list[tuple[int, int]] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if text.startswith(line, i):
            end = text.find("\n", i)
            end = n if end < 0 else end
            spans.append((i, end))
            i = end
            continue
        if block and text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end < 0:
                return None  # unterminated block comment: do not trust anything after it
            spans.append((i, end + 2))
            i = end + 2
            continue
        if ch in "\"'":
            # Triple quotes (Python) are strings, not comments: a docstring is
            # code as far as this gate is concerned. Treating one as a comment
            # would demote a real `ctypes.CDLL(...)` line sitting inside it.
            triple = text[i:i + 3] if text[i:i + 3] in ('"""', "'''") else None
            if triple:
                end = text.find(triple, i + 3)
                if end < 0:
                    return None
                i = end + 3
                continue
            j = i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == ch:
                    break
                if text[j] == "\n":
                    return None  # unterminated single-line literal: bail out
                j += 1
            else:
                return None
            if j >= n:
                return None
            i = j + 1
            continue
        i += 1
    return spans


def _in_spans(start: int, end: int, spans: Sequence[tuple[int, int]]) -> bool:
    return any(lo <= start and end <= hi for lo, hi in spans)


def _matches(text: str, path: str, inspection: str,
             spans: Sequence[tuple[int, int]] | None = None) -> tuple[list[Finding], list[Finding]]:
    """Return (blocking, advisory) findings, one of each at most per rule.

    A rule with any match outside a comment is blocking and reports that match;
    a rule matching only inside comments is advisory and reports its first.
    """
    blocking: list[Finding] = []
    advisory: list[Finding] = []
    for rule, regex in TEXT_RULES:
        first = None
        for match in regex.finditer(text):
            if first is None:
                first = match
            if spans is None or not _in_spans(match.start(), match.end(), spans):
                start = max(0, match.start() - 60)
                end = min(len(text), match.end() + 100)
                blocking.append(Finding(path, rule, _clip(text[start:end]), inspection))
                break
        else:
            if first is not None:
                start = max(0, first.start() - 60)
                end = min(len(text), first.end() + 100)
                advisory.append(Finding(path, rule, _clip(text[start:end]), inspection + "-comment"))
    return blocking, advisory


def _run_tool(argv: Sequence[str], path: str, inspection: str) -> tuple[str, list[Finding]]:
    try:
        proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, encoding="utf-8", errors="replace", check=False)
    except OSError as exc:
        return "", [Finding(path, "inspection_error", _clip(f"{argv[0]}: {exc}"), inspection)]
    if proc.returncode:
        detail = proc.stderr or proc.stdout or f"exit status {proc.returncode}"
        return "", [Finding(path, "inspection_error", _clip(f"{argv[0]}: {detail}"), inspection)]
    return proc.stdout, []


def _inspect_elf(path: Path, shown: str) -> list[Finding]:
    findings: list[Finding] = []
    dynamic, errors = _run_tool(("readelf", "-W", "-d", os.fspath(path)), shown, "elf-dynamic")
    findings.extend(errors)
    # No `spans` on any binary surface: with spans=None every match blocks,
    # which is correct -- there are no comments in a symbol table or a strings
    # dump, so the advisory list is empty by construction.
    findings.extend(_matches(dynamic, shown, "elf-dynamic")[0])

    symbols, errors = _run_tool(("readelf", "-W", "-s", os.fspath(path)), shown, "elf-symbols")
    findings.extend(errors)
    if not errors:
        for line in symbols.splitlines():
            # readelf's final column is the symbol name (possibly with @VERSION).
            fields = line.split()
            if not fields or not re.match(r"^\d+:$", fields[0]):
                continue
            symbol = fields[-1].split("@", 1)[0]
            for rule, regex in SYMBOL_RULES:
                if regex.search(symbol):
                    findings.append(Finding(shown, rule, _clip(symbol), "elf-symbols"))

    strings, errors = _run_tool(("strings", "-a", os.fspath(path)), shown, "binary-strings")
    findings.extend(errors)
    findings.extend(_matches(strings, shown, "binary-strings")[0])
    return findings


def _inspect_regular(path: Path, shown: str) -> tuple[list[Finding], list[Finding], dict[str, object]]:
    findings: list[Finding] = []
    advisory: list[Finding] = []
    try:
        data = path.read_bytes()
    except OSError as exc:
        return ([Finding(shown, "inspection_error", _clip(str(exc)), "read")], [],
                {"path": shown, "type": "unreadable"})

    record: dict[str, object] = {
        "path": shown, "type": "file", "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    if data.startswith(ELF_MAGIC):
        record["format"] = "elf"
        findings.extend(_inspect_elf(path, shown))
    else:
        try:
            text = data.decode("utf-8")
            record["format"] = "text"
            spans = _comment_spans(text, path.suffix.lower())
            # Recorded so a reader can tell "no comment matches" from "comments
            # were never identified in this file", which look identical in the
            # findings list and mean different things.
            record["comment_syntax"] = "none" if spans is None else path.suffix.lower()
            blocking, commented = _matches(text, shown, "text", spans)
            findings.extend(blocking)
            advisory.extend(commented)
        except UnicodeDecodeError:
            record["format"] = "binary"
            strings, errors = _run_tool(("strings", "-a", os.fspath(path)), shown, "binary-strings")
            findings.extend(errors)
            findings.extend(_matches(strings, shown, "binary-strings")[0])
    return findings, advisory, record


def _under(path: Path, roots: Sequence[Path]) -> bool:
    absolute = Path(os.path.abspath(os.fspath(path)))
    for root in roots:
        try:
            absolute.relative_to(root)
            return True
        except ValueError:
            pass
    return False


def scan(roots: Iterable[os.PathLike[str] | str], immutable: Iterable[os.PathLike[str] | str] = ()) -> dict[str, object]:
    """Scan roots and return a deterministic, JSON-serializable receipt."""
    roots_abs = sorted({Path(os.path.abspath(os.fspath(p))) for p in roots}, key=os.fspath)
    immutable_abs = sorted({Path(os.path.abspath(os.fspath(p))) for p in immutable}, key=os.fspath)
    findings: list[Finding] = []
    advisory: list[Finding] = []
    inspected: list[dict[str, object]] = []
    skipped: list[str] = []

    def visit(path: Path) -> None:
        shown = _display(path)
        if _under(path, immutable_abs):
            skipped.append(shown)
            return
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            findings.append(Finding(shown, "inspection_error", _clip(str(exc)), "lstat"))
            return
        if stat.S_ISLNK(mode):
            try:
                target = os.readlink(path)
            except OSError as exc:
                target = f"<unreadable: {exc}>"
            inspected.append({"path": shown, "type": "symlink", "target": target})
            findings.append(Finding(shown, "symlink", _clip(target), "filesystem"))
            # Never follow it: following either permits an escape or scans a mutable
            # target under a misleading candidate path.
            return
        if stat.S_ISDIR(mode):
            inspected.append({"path": shown, "type": "directory"})
            try:
                children = sorted(path.iterdir(), key=lambda p: os.fsencode(p.name))
            except OSError as exc:
                findings.append(Finding(shown, "inspection_error", _clip(str(exc)), "directory"))
                return
            for child in children:
                visit(child)
            return
        if stat.S_ISREG(mode):
            new_findings, new_advisory, record = _inspect_regular(path, shown)
            inspected.append(record)
            findings.extend(new_findings)
            advisory.extend(new_advisory)
            return
        inspected.append({"path": shown, "type": "special"})
        findings.append(Finding(shown, "special_file", "non-regular candidate filesystem entry", "filesystem"))

    for root in roots_abs:
        visit(root)

    # Deduplicate because the same evidence can be visible in both DT_NEEDED and
    # strings, while retaining each distinct inspection surface.
    unique = sorted(set(findings), key=lambda f: (f.path, f.rule, f.inspection, f.evidence))
    # A rule that blocks somewhere is not also reported as advisory for the same
    # file: the blocking finding is the one that matters and a duplicate under a
    # softer heading would invite arguing with it.
    blocked = {(f.path, f.rule) for f in unique}
    unique_advisory = sorted({f for f in advisory if (f.path, f.rule) not in blocked},
                             key=lambda f: (f.path, f.rule, f.inspection, f.evidence))
    inspected.sort(key=lambda item: (str(item["path"]), str(item["type"])))
    skipped = sorted(set(skipped))
    # Finding (69). The orchestrator has no filesystem, so it can never read this
    # file; the verifying agent copies a summary of it into its JSON report and
    # the orchestrator checks *that*. These counts exist so the agent copies a
    # derived object rather than hand-counting three lists -- a hand-counted
    # number is both easy to get wrong honestly and trivial to shade dishonestly,
    # and the orchestrator's whole check is that the numbers agree with each
    # other. `elf` is broken out because a post-build scan that inspected no
    # binary is a pre-build scan wearing a post-build name, and DT_NEEDED --
    # the actual way a forbidden library gets linked in -- is only visible in
    # the binary.
    summary = {
        "schema": SCHEMA,
        "passed": not unique,
        "findings": len(unique),
        "advisory": len(unique_advisory),
        "inspected": len(inspected),
        # `inspected` counts directory entries too, so it is 1 for an empty
        # directory and cannot answer "was any file actually opened?". `files`
        # can, and it is the one the orchestrator gates on. Both are kept: the
        # difference between them is itself informative when a scan was pointed
        # at a tree that turned out to be empty.
        "files": sum(1 for item in inspected if item.get("type") != "directory"),
        "elf": sum(1 for item in inspected if item.get("format") == "elf"),
        "unreadable": sum(1 for item in inspected if item.get("type") == "unreadable"),
    }
    return {
        "schema": SCHEMA,
        "summary": summary,
        "policy": "candidate-must-not-use-external-math-or-dynamic-loading",
        "roots": [_display(path) for path in roots_abs],
        "immutable": [_display(path) for path in immutable_abs],
        "allowed": ["HIP runtime/compiler/device APIs", "MFMA intrinsics", "rocWMMA header-only"],
        "passed": not unique,
        "findings": [finding.object() for finding in unique],
        # Comment-only matches. Never affects `passed`, so every role prompt's
        # "any finding fails closed" still means exactly what it says.
        "advisory": [finding.object() for finding in unique_advisory],
        "inspected": inspected,
        "skipped_immutable": skipped,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="candidate source/build/artifact/snapshot paths")
    parser.add_argument("--candidate", "--root", dest="extra_paths", action="append", default=[],
                        help="candidate path (repeatable; equivalent to a positional path)")
    parser.add_argument("--immutable", "--oracle", "--baseline", action="append", default=[],
                        help="frozen oracle/baseline path to exempt (repeatable)")
    parser.add_argument("--output", "-o", help="also write the canonical JSON receipt to this file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    roots = [*args.paths, *args.extra_paths]
    if not roots:
        _parser().error("at least one candidate path is required")
    receipt = scan(roots, args.immutable)
    payload = json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    if args.output:
        try:
            Path(args.output).write_text(payload, encoding="utf-8")
        except OSError as exc:
            print(f"candidate_policy_scan: cannot write receipt: {exc}", file=sys.stderr)
            return 2
    sys.stdout.write(payload)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
