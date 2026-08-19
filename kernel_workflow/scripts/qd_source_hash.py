#!/usr/bin/env python3
"""Deterministic source-tree hashing and conservative descriptor-evidence extraction.

kernel_lane.js's QD v2 schemas carry a `source_hash` on every engineer/verify
result and compare `ver.seed_source_hash !== proposed.source_hash` to detect
drift between what was proposed and what was actually measured, and every
descriptor is expected to travel with `descriptor_evidence`: short strings
that justify the classification. Neither is computed inside kernel_lane.js --
both are supplied by agents today. This module gives that a deterministic,
GPU-free, filesystem-honest implementation:

  * `tree_hash` walks a candidate source tree with the SAME deterministic
    exclusions kernel_lane.js's fresh-workspace tar-copy uses (.git, build
    artifacts, __pycache__, .torch_ext, ...), never follows symlinks, and
    returns a single sha256 over a canonical listing of (relative path,
    entry kind, content hash) -- identical trees hash identically regardless
    of directory walk order or the OS's filename byte order.

  * `extract_descriptor_evidence` never asserts a claim it cannot ground: it
    only returns a quote for a claim when that claim's keyword pattern
    literally matches somewhere in the source text (grounded in `source:`)
    or an explicit, literal field in caller-supplied metadata (grounded in
    `metadata:`). A claim with no match returns None -- it is the caller's
    job to not report descriptor axes it has no evidence for, and this
    function's job to never manufacture evidence to save it the trouble.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

SCHEMA = "geak.qd-source-hash/v1"


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


# --- conservative descriptor-evidence extraction ---------------------------

# Curated, deliberately narrow keyword grounding per geak-qd-v2 axis value.
# Each pattern must be an unambiguous textual signal of that specific
# mechanism in Triton/HIP source -- broad enough to catch real usage, never so
# broad it fires on ordinary prose (the same discipline candidate_policy_scan.py
# uses for its TEXT_RULES).
EVIDENCE_PATTERNS: Mapping[str, re.Pattern[str]] = {
    "compute_primitive:valu": re.compile(r"\btl\.(?:sum|add|maximum|minimum)\s*\(|(?<![A-Za-z0-9_])fma\s*\("),
    "compute_primitive:rocwmma": re.compile(r"(?i)\brocwmma\b|\bwmma::"),
    # `v_mfma[a-z0-9_]*`, not `\bmfma\b` and not `\bv_mfma\b`. The bare name
    # grounds on `#define GEAK_HAS_BF16_MFMA 1` and `#if !defined(QD_MFMA_ARCH)`
    # -- arch-guard macro names, which are evidence of nothing (94). And the
    # real mnemonic is `v_mfma_f32_16x16x16bf16_1k`, so a trailing `\b` would
    # demand a non-word character exactly where `_f32` begins, reintroducing the
    # very bug this rule is being narrowed to fix.
    "compute_primitive:native_mfma": re.compile(r"(?i)\bv_mfma[a-z0-9_]*|\btl\.dot\s*\(|__builtin_amdgcn_mfma"),
    "wave_schedule:symmetric_interleave": re.compile(r"(?i)\binterleav\w*\b.{0,40}\bwave\b|\bwave\b.{0,40}\binterleav\w*\b"),
    "wave_schedule:symmetric_pingpong": re.compile(r"(?i)\bping[-_ ]?pong\b"),
    "wave_schedule:asymmetric_producer_consumer": re.compile(r"(?i)\bproducer\b.{0,40}\bconsumer\b|\bconsumer\b.{0,40}\bproducer\b"),
    "k_pipeline:lds_single": re.compile(r"(?i)\b(?:tl\.constexpr\s+)?num_stages\s*=\s*1\b|__shared__"),
    "k_pipeline:lds_reg_prefetch": re.compile(r"(?i)\bprefetch\b"),
    "k_pipeline:lds_pingpong": re.compile(r"(?i)\bnum_stages\s*=\s*2\b"),
    "k_pipeline:lds_multistage": re.compile(r"(?i)\bnum_stages\s*=\s*[3-9]\b|\bmultistage\b|\bmulti-stage\b"),
    "decomposition:persistent_output": re.compile(r"(?i)\bpersistent\b"),
    "decomposition:split_k": re.compile(r"(?i)\bsplit[-_ ]?k\b"),
    "decomposition:stream_k": re.compile(r"(?i)\bstream[-_ ]?k\b"),
    "output_path:atomic_fixup": re.compile(r"(?i)\btl\.atomic_add\s*\(|\batomicAdd\s*\("),
    "output_path:workspace_fixup": re.compile(r"(?i)\bworkspace\b.{0,40}\breduc\w*\b|\breduc\w*\b.{0,40}\bworkspace\b"),
    "output_path:lds_staged_store": re.compile(r"(?i)\bstage\w*\b.{0,40}\bstore\b|\bstore\b.{0,40}\bstage\w*\b"),
    # The two rasterization rules below are grounded in *arithmetic*, not in the
    # words "XCD" or "group". That is deliberate and stricter than the rules
    # above: a comment explaining a mechanism is prose ABOUT the mechanism, not
    # the mechanism (finding 53). The shipped v59 kernel is the worked example --
    # its 25-line comment block names XCDs repeatedly and matches neither
    # pattern; only `pid % kXcds` and `kGroupM = 8` do.
    #
    # `xcd_remapped_grouped` needs the un-shuffle: a block/program id divided or
    # taken modulo an XCD-count identifier, or that count scaled by a stride.
    # Merely mentioning XCDs is not evidence of remapping by them.
    "rasterization:xcd_remapped_grouped": re.compile(
        r"(?i)\b(?:p|pid|program_?id|block_?idx(?:\.[xyz])?)\s*[%/]\s*[A-Za-z_]*xcds?\b"
        r"|\b[A-Za-z_]*xcds?\b\s*\*\s*[A-Za-z_]"),
    # `grouped_m` matches a group-height *declaration*: a Triton GROUP_SIZE_M
    # parameter, or a C group-height bound to a value behind a type/qualifier.
    # The declaration context is load-bearing, not decoration. The v59 comment
    # block quotes `kGroupM = 8` verbatim while arguing that v58's grouping LOST,
    # so a bare `kGroupM\s*=` grounds the claim on prose that rejects it.
    #
    # NOTE this rule also fires on an xcd_remapped_grouped kernel, which is
    # grouped too: the map answers "is the construct present", never "is this the
    # only construct present". Evidence cannot catch a claim that is understated.
    "rasterization:grouped_m": re.compile(
        r"(?i)\bgroup_?size_?m\b\s*[:=]"
        r"|\b(?:constexpr|const|int|define|template|tl\.constexpr)\b[^\n]{0,40}"
        r"\bk?group_?m\b\s*="),
}

# Axis values deliberately left ungrounded, with the reason, so the gap is a
# stated absence rather than an oversight. `extract_descriptor_evidence` returns
# None for anything not in EVIDENCE_PATTERNS either way; this table exists so a
# reader can tell "no rule yet" from "no rule possible".
UNGROUNDABLE_CLAIMS: Mapping[str, str] = {
    # Absences. You cannot grep for the code someone did not write.
    "decomposition:tile_grid": "the absence of persistent/split-k/stream-k, not a construct",
    "k_pipeline:direct_global": "the absence of an LDS stage",
    "output_path:direct_store": "the absence of a fixup path",
    "plan_binding:static": "the absence of runtime tuning",
    "rasterization:linear": "the absence of any remap or grouping",
    "wave_schedule:independent": "the absence of a pairing schedule",
    # Depth/degree claims: the construct is present in the weaker sibling too,
    # so a pattern would ground the claim without confirming its degree.
    "k_pipeline:lds_deep_single": "indistinguishable from lds_single by text alone",
    # Groundable in principle, but every narrow pattern I could write keys on one
    # kernel's private spelling (autotune_enabled/tuned_slices/GEAK_NO_AUTOTUNE),
    # and the loose form fires on a comment saying autotune was DISABLED.
    # Overfitting to one implementation is not grounding.
    "plan_binding:runtime_tuned": "no implementation-independent construct found yet",
}


# Length-preserving on purpose: matches are located in the substituted copy but
# quoted out of the original, so the two must stay in index correspondence.
_UNDERSCORE = re.compile(r"_")


def _grounded_quote(search_text: str, pattern: re.Pattern[str], *,
                    quote_text: str | None = None, radius: int = 60) -> str | None:
    """Find `pattern` in `search_text`; quote the same span out of `quote_text`.

    The two texts are the comment-blanked and original forms of one file and are
    the same length by construction (see `_blank_comments`), so the offsets are
    interchangeable. Splitting them lets a match be *required* to sit in code
    while the quote shown still carries its real surrounding context.

    A miss is retried against a copy with every `_` replaced by a space (94).
    Identifiers spell word boundaries with underscores -- `prefetch_stage`,
    `splitk_reduce_kernel`, `store_panel` -- and `\b` does not break there,
    because `_` is a word character. So keyword rules could not see the axes
    those identifiers implement, while the same words in a *comment* matched
    fine: the extractor was systematically blind to code and open to prose.

    It has to be an OR, not a substitution. `__shared__`, `num_stages = 1`,
    `__builtin_amdgcn_mfma` and `group_?size_?m` are all destroyed by replacing
    underscores, so four rules go dark if only the spaced copy is searched. The
    substitution is length-preserving precisely so the quote offsets still line
    up with the original text.

    The labelled hole (54): camelCase is not split by this and cannot be, since
    no length-preserving substitution splits a camel hump. `kPingPongStages`
    does not ground `symmetric_pingpong`, and that absence is asserted in the
    test suite so it stays a stated absence rather than a surprise.
    """
    m = pattern.search(search_text) or pattern.search(_UNDERSCORE.sub(" ", search_text))
    if not m:
        return None
    text = search_text if quote_text is None else quote_text
    start, end = max(0, m.start() - radius), min(len(text), m.end() + radius)
    quote = " ".join(text[start:end].replace("\x00", " ").split())
    return quote


# Text files that are prose by construction. A README explaining that the
# project uses MFMA is not evidence that THIS kernel issues one -- it is the
# file-granularity version of the comment trap: documentation ABOUT a mechanism,
# not the mechanism. Excluded from evidence scanning only; `tree_hash` still
# hashes them, because editing the docs does change the source tree.
# Also excludes structured data/report files. Those have no comment syntax to
# blank, and their content is prose: a real workspace's research/index.json is a
# log of approaches "considered_or_rejected", and its list of rejected ideas
# grounded compute_primitive:native_mfma on the first run against a real tree.
PROSE_SUFFIXES = frozenset({".md", ".markdown", ".rst", ".txt", ".ipynb",
                            ".json", ".jsonl", ".csv", ".tsv", ".xml", ".html"})

# Comments are blanked before matching. Most rules in EVIDENCE_PATTERNS are
# keyword-shaped -- "prefetch", "persistent", "ping-pong" -- and a keyword rule
# run over comments grounds a claim on prose ABOUT the mechanism (finding 53).
# This is not hypothetical: the shipped v59 kernel grounds
# wave_schedule:symmetric_pingpong on a comment whose actual sentence is
# "there is nothing to interleave or ping-pong against when the SIMD holds one
# wave", i.e. a comment explaining why the kernel does NOT do the thing.
# Blanking once here fixes every rule at the source instead of rewriting
# eighteen patterns into code-shaped ones and inviting false negatives (which
# would surface as spurious `unsubstantiated` mislabel alarms -- the loud-and-
# always-wrong gate that finding 53 also warns against).
_C_LIKE_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp",
                              ".hip", ".cu", ".cuh", ".cl"})
_HASH_COMMENT_SUFFIXES = frozenset({".py", ".pyi", ".sh", ".bash", ".cmake",
                                    ".mk", ".yaml", ".yml", ".toml"})
# `#` is NOT a comment in C: blanking it there would delete `#include` and
# `#if defined(...)`, which several rules legitimately match.
_C_COMMENT_RE = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)
_HASH_COMMENT_RE = re.compile(r"#[^\n]*")


def _language_suffix(rel: str) -> str:
    """The recognized source suffix of `rel`, searching right to left.

    Not simply `.suffix`, because saved variants carry trailing suffixes:
    `src/custom_gemm.hip.v100_m512waves8` is a real file in a real workspace and
    its final suffix is `.v100_m512waves8`, which matches no comment style. The
    naive version therefore blanked nothing in exactly the files most likely to
    be dead snapshots, and one of them grounded a ping-pong claim on a comment
    saying the kernel has nothing to ping-pong against.
    """
    for suffix in reversed(PurePosixPath(rel).suffixes):
        lowered = suffix.lower()
        if lowered in _C_LIKE_SUFFIXES or lowered in _HASH_COMMENT_SUFFIXES or lowered in PROSE_SUFFIXES:
            return lowered
    return PurePosixPath(rel).suffix.lower()


def _in_scope(rel: str, prefixes: tuple[str, ...]) -> bool:
    """True when `rel` is one of `prefixes` exactly, or sits under one as a dir.

    Plain `startswith` would make `--scope src/custom_gemm.hip` also select
    `src/custom_gemm.hip.v100_m512waves8` -- the saved variant the caller was
    trying to exclude. Naming a file has to mean that file.
    """
    for prefix in prefixes:
        trimmed = prefix.rstrip("/")
        if rel == trimmed or rel.startswith(trimmed + "/"):
            return True
    return False


def _blank_comments(text: str, suffix: str) -> str:
    """Replace comment bodies with spaces, preserving length and line breaks.

    Length preservation matters: matches are found in this blanked text but
    quoted out of the original, so the reader still sees real surrounding code
    rather than a field of spaces.

    Deliberately naive about string literals -- a "http://host" in a string gets
    blanked. That trade is one-directional and cheap: the cost is a missed match
    on a URL nobody grounds a mechanism claim on, against a demonstrated false
    grounding on comments in the very first real workspace this was run against.
    """
    suffix = suffix.lower()
    if suffix in _C_LIKE_SUFFIXES:
        pattern = _C_COMMENT_RE
    elif suffix in _HASH_COMMENT_SUFFIXES:
        pattern = _HASH_COMMENT_RE
    else:
        return text
    return pattern.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)


def extract_descriptor_evidence(root: os.PathLike[str] | str, claims: Sequence[str],
                                 *, metadata: Mapping[str, object] | None = None,
                                 scope: Sequence[str] | None = None
                                 ) -> dict[str, str | None]:
    """For each `claims` key (an "axis:value" string, e.g. "k_pipeline:lds_multistage"),
    return a grounded evidence quote or None.

    Grounding sources, checked in order, first match wins:
      1. `metadata[claim] is True` (or a non-empty string) -- an explicit,
         literal, caller-declared fact, tagged "metadata:<claim>=<value>" so
         it is never confused with a source-text quote.
      2. a source-tree text match against EVIDENCE_PATTERNS[claim], tagged
         "source:<relative_path>: <quote>".
    A claim absent from EVIDENCE_PATTERNS and absent from metadata is
    reported as None: this function refuses to guess evidence for a claim it
    has no grounding rule for, rather than silently accepting it on faith.

    `scope`, when given, restricts the search to files whose tree-relative path
    starts with one of the supplied prefixes. Use it to point at the sources
    that actually build. Without it the whole tree is searched, and a real
    workspace routinely carries abandoned variants under research/ or
    experimental/ whose text will happily ground a claim about a mechanism the
    shipped kernel does not contain. This function cannot know which files the
    build consumes; the caller can, so the caller has to say. Files with a
    PROSE_SUFFIXES extension are skipped either way.
    """
    root_path = Path(root)
    texts: dict[str, tuple[str, str]] | None = None

    def load_texts() -> dict[str, tuple[str, str]]:
        nonlocal texts
        if texts is not None:
            return texts
        texts = {}
        prefixes = tuple(scope) if scope else None
        for rel, kind, payload in _iter_entries(root_path, DEFAULT_EXCLUDED_DIRS):
            if kind != "file" or payload is None:
                continue
            language = _language_suffix(rel)
            if language in PROSE_SUFFIXES:
                continue
            if prefixes is not None and not _in_scope(rel, prefixes):
                continue
            try:
                original = payload.decode("utf-8")
            except UnicodeDecodeError:
                continue
            # Search the blanked text, quote from the original.
            texts[rel] = (_blank_comments(original, language), original)
        return texts

    out: dict[str, str | None] = {}
    for claim in claims:
        meta_value = None if metadata is None else metadata.get(claim)
        if meta_value is True or (isinstance(meta_value, str) and meta_value.strip()):
            shown = meta_value if isinstance(meta_value, str) else "true"
            out[claim] = f"metadata:{claim}={shown}"
            continue
        pattern = EVIDENCE_PATTERNS.get(claim)
        if pattern is None:
            out[claim] = None
            continue
        found: str | None = None
        for rel, (search_text, original) in sorted(load_texts().items()):
            quote = _grounded_quote(search_text, pattern, quote_text=original)
            if quote is not None:
                found = f"source:{rel}: {quote}"
                break
        out[claim] = found
    return out


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
