#!/usr/bin/env python3
"""Fail-closed preflight and 48-hour supervisor for fresh GEAK lanes.

The preflight path is deliberately read-only: it opens existing lock files, reads
manifests and git metadata, probes the GPU/API, and prints a report.  State and
logs are created only by ``supervise`` after every precondition passed.  A dry
run executes the same preflight and never initializes state.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import pwd
import re
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

TEMPLATE_SCHEMA = "geak.lane-template/v1"
ALLOCATION_SCHEMA = "geak.gpu-allocation/v1"
CONFIG_REGISTRY_SCHEMA = "geak.config-search/v2"
CONFIG_REGISTRY_VERSION = 2
RUNTIME_TABLE_SCHEMA = "geak.runtime_table/v3"
LAUNCH_CONTRACT_SCHEMA = "geak.asm-launch-contract/v1"
ARTIFACT_SET_SCHEMA = "geak.asm-artifact-set/v1"
BASE_COMMIT = "a9876692ed1039b85b2fdafb8c1751548fd13da1"
EXPECTED_PROTOCOL_HASH = "6c479d8f24864215652dc11c625ab8817dd868aeba273d4ae0fd61881d7523ec"
EXPECTED_SCORE_METRIC_ID = "sustained_throughput"
EXPECTED_SCORE_METRIC = {
    "schema": "geak.score-metric/v1",
    "version": 1,
    "id": EXPECTED_SCORE_METRIC_ID,
    "unit": "ms/call",
    "inner": 16,
    "protocol_hash": EXPECTED_PROTOCOL_HASH,
    "hash": "c104dd5f35ce8a7d4591d0952265a40ee72cf21e2addcb4bfd7762737dcc12b8",
}
HOURS_DEFAULT = 48.0
STOP_NEW_ROUND_MARGIN_S = 30 * 60
SLURM_MARGIN_S = 15 * 60
RESTART_BACKOFFS = (60, 300, 900)
MIN_WAVE_S = 30 * 60
MIN_FREE_BYTES = 50 * 1024**3
MIN_FREE_INODES = 100_000
PROTOCOL_ENV_VARS = (
    "GEAK_MEASUREMENT_PROTOCOL", "GEAK_MEASUREMENT_SEED", "GEAK_WARMUP_CYCLES",
    "GEAK_ROTATION_TARGET_MB", "GEAK_ASSUMED_CACHE_MB", "GEAK_MAX_WEIGHT_SLOTS",
    "GEAK_OUTPUT_RING_SIZE", "HARNESS_TIMING_PRIME_MB", "GEAK_SYNTHETIC_EVICT_MB",
    "HARNESS_CACHE_FLUSH_MB", "GEAK_LATENCY_CYCLES", "GEAK_THROUGHPUT_INNER",
    "GEAK_THROUGHPUT_CYCLES", "GEAK_SCORE_METRIC",
)

EXIT_CONFIG = 20
EXIT_MANIFEST = 21
EXIT_TUNER = 22
EXIT_IDENTITY = 23
EXIT_GPU = 24
EXIT_DEADLINE = 25
EXIT_DUPLICATE = 26
EXIT_API = 27
EXIT_RUNTIME = 70
EXIT_RESTARTS = 75

LANES = {
    "bf16": {
        "template": "kernel_workflow/lanes/bf16_fresh_48h.json.in",
        "task": "dense_bf16_gemm_fused",
    },
    "fp8": {
        "template": "kernel_workflow/lanes/fp8_fresh_48h.json.in",
        "task": "dense_fp8_gemm_rowwise",
    },
}

PERMANENT_LOG_RE = re.compile(
    r"manifest is not frozen|oracle route manifest is not trusted|protocol hash differs|"
    r"protocol_hash.*mismatch|unknown argument|is not launchable|path.*outside.*root|"
    r"registry schema mismatch|unsupported runtime table|route inventory.*mismatch|"
    r"score.?metric.*(?:mismatch|differ|drift)|metric.*drift|"
    r"config payload missing|artifact hash mismatch",
    re.IGNORECASE,
)
TRANSPORT_LOG_RE = re.compile(
    r"ECONNRESET|Connection dropped|Connection error|socket hang up|ETIMEDOUT|"
    r"EAI_AGAIN|502 Bad Gateway|503 Service|Unable to connect to API|fetch failed",
    re.IGNORECASE,
)
KNOWN_ARGS_RE = re.compile(r"const KNOWN_ARGS = new Set\(\[(.*?)\]\);", re.S)
TOKEN_RE = re.compile(r"@[A-Z][A-Z0-9_]*@")


class LaunchError(RuntimeError):
    def __init__(self, message: str, *, code: int = EXIT_CONFIG, category: str = "config") -> None:
        super().__init__(message)
        self.code = code
        self.category = category


@dataclass(frozen=True)
class Deadline:
    hard_end_epoch: int
    stop_new_round_epoch: int
    slurm_end_epoch: int
    requested_seconds: int


@dataclass(frozen=True)
class ApiConfig:
    settings_path: Path
    messages_url: str
    model: str
    headers: Mapping[str, str]
    tunnel_host: str
    tunnel_port: int


@dataclass(frozen=True)
class LaunchSpec:
    lane: str
    task: str
    root: Path
    state_dir: Path
    expected_commit: str
    template_path: Path
    allocation_path: Path
    tuner_registry: Path
    tuner_table: Path
    gpu_ids: tuple[int, ...]
    deadline: Deadline
    raw_lane: Mapping[str, Any]
    args: Mapping[str, Any]
    meta: Mapping[str, Any]
    config_sha256: str


@dataclass(frozen=True)
class WaveResult:
    returncode: int
    duration_s: int
    permanent: bool = False
    hit_hard_end: bool = False
    reason: str = ""


def utc(epoch: int | float) -> str:
    return dt.datetime.fromtimestamp(epoch, tz=dt.timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, *, code: int = EXIT_CONFIG, category: str = "config") -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchError(f"cannot read JSON {path}: {exc}", code=code, category=category) from exc
    if not isinstance(value, dict):
        raise LaunchError(f"JSON root must be an object: {path}", code=code, category=category)
    return value


def require_regular(path: Path, *, code: int, category: str) -> None:
    try:
        st = path.lstat()
    except OSError as exc:
        raise LaunchError(f"required file is missing: {path}", code=code, category=category) from exc
    if not stat.S_ISREG(st.st_mode) or path.is_symlink():
        raise LaunchError(f"required path must be a non-symlink regular file: {path}",
                          code=code, category=category)


def within(path: Path, root: Path, *, allow_missing: bool = False) -> bool:
    root = root.resolve()
    if path.exists() or not allow_missing:
        try:
            path.resolve().relative_to(root)
            return True
        except (OSError, ValueError):
            return False
    ancestor = path
    suffix: list[str] = []
    while not ancestor.exists() and ancestor != ancestor.parent:
        suffix.append(ancestor.name)
        ancestor = ancestor.parent
    try:
        resolved = ancestor.resolve()
        for item in reversed(suffix):
            resolved /= item
        resolved.relative_to(root)
        return True
    except (OSError, ValueError):
        return False


def git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["GIT_OPTIONAL_LOCKS"] = "0"
    proc = subprocess.run(["git", "-C", str(root), *args], text=True, capture_output=True, env=env)
    if check and proc.returncode:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        raise LaunchError(f"git {' '.join(args)} failed: {detail[-1] if detail else 'unknown error'}")
    return proc


def check_git(root: Path, expected_commit: str, base_commit: str = BASE_COMMIT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise LaunchError("--expected-commit must be the full 40-character lowercase commit hash")
    top = Path(git(root, "rev-parse", "--show-toplevel").stdout.strip()).resolve()
    if top != root.resolve():
        raise LaunchError(f"expected root {root.resolve()} but git reports {top}")
    head = git(root, "rev-parse", "HEAD").stdout.strip()
    if head != expected_commit:
        raise LaunchError(f"HEAD mismatch: expected {expected_commit}, found {head}")
    if git(root, "merge-base", "--is-ancestor", base_commit, "HEAD", check=False).returncode:
        raise LaunchError(f"HEAD {head} is not based on required commit {base_commit}")
    status = git(root, "status", "--porcelain=v1", "--untracked-files=all").stdout
    if status:
        first = status.splitlines()[0]
        raise LaunchError(f"worktree is dirty; first entry: {first}")
    return head


def require_tracked(
    root: Path, path: Path, *, code: int = EXIT_CONFIG, category: str = "config",
) -> None:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise LaunchError(
            f"tracked launch input is outside expected root: {path}",
            code=code, category=category,
        ) from exc
    proc = git(root, "ls-files", "--error-unmatch", "--", rel.as_posix(), check=False)
    if proc.returncode:
        raise LaunchError(
            f"launch input is not tracked by expected commit: {path}",
            code=code, category=category,
        )


def _substitute(value: Any, tokens: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        if value in tokens:
            return tokens[value]
        out = value
        for token, replacement in tokens.items():
            out = out.replace(token, str(replacement))
        leftover = TOKEN_RE.search(out)
        if leftover:
            raise LaunchError(f"unresolved lane-template token {leftover.group(0)}")
        return out
    if isinstance(value, list):
        return [_substitute(item, tokens) for item in value]
    if isinstance(value, dict):
        return {key: _substitute(item, tokens) for key, item in value.items()}
    return value


def load_allocation(root: Path, allocation_path: Path) -> dict[str, tuple[int, ...]]:
    require_regular(allocation_path, code=EXIT_CONFIG, category="config")
    require_tracked(root, allocation_path)
    doc = read_json(allocation_path)
    if doc.get("schema") != ALLOCATION_SCHEMA:
        raise LaunchError(f"GPU allocation schema must be {ALLOCATION_SCHEMA}")
    out: dict[str, tuple[int, ...]] = {}
    for lane in LANES:
        raw = doc.get("lanes", {}).get(lane)
        if not isinstance(raw, list) or not raw or any(type(item) is not int or item < 0 for item in raw):
            raise LaunchError(f"GPU allocation for {lane} must be a non-empty list of integer ids")
        if len(set(raw)) != len(raw):
            raise LaunchError(f"GPU allocation for {lane} contains duplicates")
        out[lane] = tuple(raw)
    overlap = set(out["bf16"]) & set(out["fp8"])
    if overlap:
        raise LaunchError(f"BF16/FP8 GPU allocations overlap: {sorted(overlap)}")
    return out


def known_args(source: Path) -> set[str]:
    text = source.read_text(encoding="utf-8")
    match = KNOWN_ARGS_RE.search(text)
    if not match:
        raise LaunchError(f"{source} has no KNOWN_ARGS contract")
    return set(re.findall(r"'([a-z_][a-z0-9_]*)'", match.group(1)))


def validate_lane_args(spec: LaunchSpec) -> None:
    args = spec.args
    required = spec.raw_lane.get("_require") or {}
    if not isinstance(required, dict):
        raise LaunchError("lane template _require must be an object")
    accepted = known_args(spec.root / "kernel_workflow/kernel_lane.js") & known_args(
        spec.root / "kernel_workflow/kernel_workflow.js")
    unknown = sorted(set(args) - accepted)
    if unknown:
        raise LaunchError(f"lane has unknown workflow argument(s): {', '.join(unknown)}")
    for key in ("kernel_path", "workflow_dir", "state_dir", "sol_shapes"):
        if not args.get(key):
            raise LaunchError(f"lane is missing required path {key}")
        path = Path(str(args[key]))
        if not path.is_absolute() or not within(path, spec.root, allow_missing=(key == "state_dir")):
            raise LaunchError(f"{key} is outside expected root {spec.root}: {path}")
    for key, expected in required.items():
        if key not in args:
            raise LaunchError(f"lane protocol requires {key}={expected!r}, but the key is absent")
        if args[key] != expected:
            raise LaunchError(f"lane protocol requires {key}={expected!r}, found {args[key]!r}")
    expected_paths = {
        "kernel_path": spec.root / "examples/tasks" / spec.task,
        "workflow_dir": spec.root / "kernel_workflow",
        "state_dir": spec.state_dir,
        "sol_shapes": spec.root / "examples/tasks" / spec.task / "sol_shapes.json",
    }
    for key, expected in expected_paths.items():
        if Path(str(args[key])) != expected:
            raise LaunchError(f"{key} mismatch: expected {expected}, found {args[key]}")
    if args.get("warm_start") != "off":
        raise LaunchError("fresh 48h lanes must explicitly set warm_start='off'")
    if int(args.get("deadline_epoch", 0)) != spec.deadline.stop_new_round_epoch:
        raise LaunchError("lane deadline_epoch must equal hard_end-30m")
    if str(args.get("gpu_ids")) != ",".join(map(str, spec.gpu_ids)):
        raise LaunchError("lane gpu_ids differs from the shared non-overlapping allocation")


def build_spec(*, lane: str, root: Path, state_dir: Path, expected_commit: str,
               tuner_registry: Path, tuner_table: Path, deadline: Deadline,
               allocation_path: Path | None = None) -> LaunchSpec:
    if lane not in LANES:
        raise LaunchError(f"unknown lane {lane!r}")
    if not root.is_absolute() or not state_dir.is_absolute() or not tuner_registry.is_absolute() or not tuner_table.is_absolute():
        raise LaunchError("root, state, tuner registry, and tuner table paths must all be absolute")
    root = root.resolve()
    state_dir = Path(os.path.abspath(state_dir))
    allocation_path = (allocation_path or root / "kernel_workflow/launch/gpu_allocation.json").resolve()
    if not within(state_dir, root, allow_missing=True):
        raise LaunchError(f"state directory is outside expected root: {state_dir}")
    try:
        state_relative = state_dir.relative_to(root / "exp")
    except ValueError as exc:
        raise LaunchError(f"fresh state must live below {root / 'exp'}: {state_dir}") from exc
    if not state_relative.parts:
        raise LaunchError("the exp directory itself cannot be used as one lane's state")
    allocation = load_allocation(root, allocation_path)
    gpu_ids = allocation[lane]
    template_path = root / LANES[lane]["template"]
    require_regular(template_path, code=EXIT_CONFIG, category="config")
    require_tracked(root, template_path)
    raw_template = read_json(template_path)
    meta_template = raw_template.get("_launch")
    if not isinstance(meta_template, dict) or meta_template.get("schema") != TEMPLATE_SCHEMA:
        raise LaunchError(f"lane template must carry _launch.schema={TEMPLATE_SCHEMA}")
    if meta_template.get("lane") != lane or meta_template.get("task") != LANES[lane]["task"]:
        raise LaunchError("lane template identity does not match its filename/selector")
    if meta_template.get("base_commit") != BASE_COMMIT:
        raise LaunchError("lane template base commit differs from the launcher contract")
    if meta_template.get("expected_protocol_hash") != EXPECTED_PROTOCOL_HASH:
        raise LaunchError("lane template protocol hash differs from the launcher contract")
    if meta_template.get("expected_score_metric") != EXPECTED_SCORE_METRIC:
        raise LaunchError("lane template score metric differs from the launcher contract")
    if meta_template.get("stop_new_round_margin_seconds") != STOP_NEW_ROUND_MARGIN_S:
        raise LaunchError("lane template drain margin must be exactly 30 minutes")
    gpu_csv = ",".join(map(str, gpu_ids))
    gpu_log = state_dir / ".launch/gpu_use.jsonl"
    gpu_env = f"GEAK_GPU_ALLOWED={shlex.quote(gpu_csv)} GEAK_GPU_USE_LOG={shlex.quote(str(gpu_log))}"
    tokens: dict[str, Any] = {
        "@ROOT@": str(root),
        "@STATE_DIR@": str(state_dir),
        "@GPU_IDS@": gpu_csv,
        "@GPU_LOCK_ENV@": gpu_env,
        "@STOP_NEW_ROUND_EPOCH@": deadline.stop_new_round_epoch,
        "@HARD_END_EPOCH@": deadline.hard_end_epoch,
        "@TUNER_REGISTRY@": str(tuner_registry),
        "@TUNER_TABLE@": str(tuner_table),
    }
    raw_lane = _substitute(raw_template, tokens)
    if not isinstance(raw_lane.get("_require"), dict):
        raise LaunchError("lane template _require must be an object")
    lane_known = known_args(root / "kernel_workflow/kernel_lane.js")
    dispatcher_known = known_args(root / "kernel_workflow/kernel_workflow.js")
    tuner_arg_names = {"tuner_registry", "tuner_table"}
    if tuner_arg_names <= (lane_known & dispatcher_known):
        raw_lane["tuner_registry"] = str(tuner_registry.resolve())
        raw_lane["tuner_table"] = str(tuner_table.resolve())
        raw_lane.setdefault("_require", {}).update({
            "tuner_registry": str(tuner_registry.resolve()),
            "tuner_table": str(tuner_table.resolve()),
        })
    args = {key: value for key, value in raw_lane.items() if not key.startswith("_")}
    meta = raw_lane["_launch"]
    if Path(str(meta.get("tuner_registry") or "")).resolve() != tuner_registry.resolve():
        raise LaunchError("rendered template does not bind the requested tuner registry")
    if Path(str(meta.get("tuner_table") or "")).resolve() != tuner_table.resolve():
        raise LaunchError("rendered template does not bind the requested tuner runtime table")
    spec = LaunchSpec(
        lane=lane, task=LANES[lane]["task"], root=root, state_dir=state_dir,
        expected_commit=expected_commit, template_path=template_path,
        allocation_path=allocation_path, tuner_registry=tuner_registry.resolve(),
        tuner_table=tuner_table.resolve(), gpu_ids=gpu_ids, deadline=deadline,
        raw_lane=raw_lane, args=args, meta=meta, config_sha256=sha256_json(raw_lane),
    )
    validate_lane_args(spec)
    return spec


def check_identity_and_credentials(settings_path: Path | None = None) -> ApiConfig:
    uid = os.geteuid()
    account = pwd.getpwuid(uid)
    home_env = os.environ.get("HOME", "")
    if account.pw_name != "yxh" or account.pw_dir != "/home/yxh" or home_env != "/home/yxh":
        raise LaunchError(
            f"launcher must run as yxh with HOME=/home/yxh; got user={account.pw_name} "
            f"passwd_home={account.pw_dir} HOME={home_env!r}", code=EXIT_IDENTITY, category="identity")
    settings_path = settings_path or Path(home_env) / ".claude/settings.json"
    require_regular(settings_path, code=EXIT_IDENTITY, category="identity")
    st = settings_path.stat()
    mode = stat.S_IMODE(st.st_mode)
    if st.st_uid != uid:
        raise LaunchError(f"credential file is not owned by yxh: {settings_path}",
                          code=EXIT_IDENTITY, category="identity")
    if mode & 0o077:
        raise LaunchError(f"credential permissions are {mode:04o}, require 0600 or stricter: {settings_path}",
                          code=EXIT_IDENTITY, category="identity")
    settings = read_json(settings_path, code=EXIT_IDENTITY, category="identity")
    env = settings.get("env")
    if not isinstance(env, dict):
        raise LaunchError("credential settings have no env object", code=EXIT_IDENTITY,
                          category="identity")
    base = str(env.get("ANTHROPIC_BASE_URL") or "").rstrip("/")
    model = str(env.get("ANTHROPIC_MODEL") or env.get("ANTHROPIC_DEFAULT_SONNET_MODEL") or "")
    api_key = str(env.get("ANTHROPIC_API_KEY") or "")
    auth_token = str(env.get("ANTHROPIC_AUTH_TOKEN") or "")
    if not base or not model or not (api_key or auth_token):
        raise LaunchError("credential settings are missing base URL, model, or API authentication",
                          code=EXIT_IDENTITY, category="identity")
    parsed = urllib.parse.urlsplit(base)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise LaunchError("ANTHROPIC_BASE_URL is not an http(s) URL", code=EXIT_IDENTITY,
                          category="identity")
    headers = {"content-type": "application/json", "anthropic-version": "2023-06-01"}
    if api_key:
        headers["x-api-key"] = api_key
    if auth_token:
        headers["authorization"] = f"Bearer {auth_token}"
    custom = env.get("ANTHROPIC_CUSTOM_HEADERS") or ""
    if not isinstance(custom, str):
        raise LaunchError("ANTHROPIC_CUSTOM_HEADERS must be newline-delimited text",
                          code=EXIT_IDENTITY, category="identity")
    for line in custom.splitlines():
        if not line.strip():
            continue
        if ":" not in line:
            raise LaunchError("ANTHROPIC_CUSTOM_HEADERS contains a malformed line",
                              code=EXIT_IDENTITY, category="identity")
        name, value = line.split(":", 1)
        headers[name.strip()] = value.strip()
    messages_url = base + "/v1/messages"
    return ApiConfig(settings_path, messages_url, model, headers, parsed.hostname,
                     parsed.port or (443 if parsed.scheme == "https" else 80))


def _pong_text(payload: Any) -> str:
    if isinstance(payload, dict):
        content = payload.get("content")
        if isinstance(content, list):
            texts = [str(item.get("text", "")) for item in content if isinstance(item, dict)]
            if texts:
                return "\n".join(texts).strip()
        choices = payload.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}
            if isinstance(message, dict):
                return str(message.get("content") or "").strip()
    return ""


def authenticated_pong(api: ApiConfig, timeout_s: int = 120) -> None:
    body = canonical_bytes({
        "model": api.model,
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "Reply with exactly: PONG"}],
    })
    request = urllib.request.Request(api.messages_url, data=body, headers=dict(api.headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise LaunchError(f"authenticated API PONG failed with HTTP {exc.code}",
                          code=EXIT_API, category="api") from exc
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise LaunchError(f"authenticated API PONG failed: {type(exc).__name__}",
                          code=EXIT_API, category="api") from exc
    if _pong_text(payload) != "PONG":
        raise LaunchError("authenticated API returned a non-PONG response",
                          code=EXIT_API, category="api")


def probe_tunnel(api: ApiConfig, timeout_s: int = 5) -> None:
    try:
        with socket.create_connection((api.tunnel_host, api.tunnel_port), timeout=timeout_s):
            pass
    except OSError as exc:
        raise LaunchError(f"API tunnel endpoint {api.tunnel_host}:{api.tunnel_port} is unreachable",
                          code=EXIT_API, category="api") from exc


def expected_routes(sol_path: Path) -> dict[str, tuple[int, int, int]]:
    require_regular(sol_path, code=EXIT_MANIFEST, category="manifest")
    doc = read_json(sol_path, code=EXIT_MANIFEST, category="manifest")
    rows = doc.get("routes")
    if not isinstance(rows, list) or len(rows) != 11:
        raise LaunchError(f"sol_shapes must contain exactly 11 routes: {sol_path}",
                          code=EXIT_MANIFEST, category="manifest")
    out: dict[str, tuple[int, int, int]] = {}
    for row in rows:
        try:
            name = str(row["route"])
            shape = (int(row["m"]), int(row["n"]), int(row["k"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise LaunchError(f"malformed sol_shapes route in {sol_path}", code=EXIT_MANIFEST,
                              category="manifest") from exc
        if name in out:
            raise LaunchError(f"duplicate sol_shapes route {name}", code=EXIT_MANIFEST,
                              category="manifest")
        out[name] = shape
    return out


def check_oracle_manifest(spec: LaunchSpec) -> dict[str, tuple[int, int, int]]:
    sol = Path(str(spec.args["sol_shapes"]))
    require_tracked(spec.root, sol)
    routes = expected_routes(sol)
    manifest_path = Path(str(spec.args["kernel_path"])) / "oracle_routes.v2.json"
    require_regular(manifest_path, code=EXIT_MANIFEST, category="manifest")
    require_tracked(spec.root, manifest_path)
    doc = read_json(manifest_path, code=EXIT_MANIFEST, category="manifest")
    if doc.get("schema") != "geak.oracle-route-manifest/v2":
        raise LaunchError("oracle manifest schema is not v2", code=EXIT_MANIFEST,
                          category="manifest")
    if doc.get("task") != spec.task or doc.get("frozen") is not True:
        raise LaunchError(f"oracle manifest for {spec.task} is not frozen",
                          code=EXIT_MANIFEST, category="manifest")
    if doc.get("selection_state") != "baked":
        raise LaunchError("oracle manifest selection_state must be 'baked'",
                          code=EXIT_MANIFEST, category="manifest")
    if (doc.get("selection_metric") != EXPECTED_SCORE_METRIC_ID
            or doc.get("score_metric") != EXPECTED_SCORE_METRIC):
        raise LaunchError(
            "oracle score metric is not the fresh-lane sustained_throughput contract; "
            "re-run oracle-bakeoff", code=EXIT_MANIFEST, category="manifest")
    protocol = doc.get("protocol")
    if not isinstance(protocol, dict):
        raise LaunchError("oracle manifest does not embed its measurement protocol",
                          code=EXIT_MANIFEST, category="manifest")
    protocol_body = dict(protocol)
    embedded_hash = protocol_body.pop("hash", None)
    computed_hash = hashlib.sha256(canonical_bytes(protocol_body)).hexdigest()
    hashes = {doc.get("protocol_hash"), embedded_hash, computed_hash}
    if hashes != {EXPECTED_PROTOCOL_HASH}:
        raise LaunchError(
            f"oracle protocol hash mismatch: expected {EXPECTED_PROTOCOL_HASH}, got "
            f"manifest={doc.get('protocol_hash')} embedded={embedded_hash} computed={computed_hash}",
            code=EXIT_MANIFEST, category="manifest")
    manifest_routes = doc.get("routes")
    if not isinstance(manifest_routes, dict) or set(manifest_routes) != set(routes):
        raise LaunchError("oracle route inventory does not exactly match the 11 sol_shapes routes",
                          code=EXIT_MANIFEST, category="manifest")
    for name, shape in routes.items():
        row = manifest_routes[name]
        if not isinstance(row, dict):
            raise LaunchError(f"oracle route {name} must be an object", code=EXIT_MANIFEST,
                              category="manifest")
        try:
            got = (int(row.get("M", -1)), int(row.get("N", -1)), int(row.get("K", -1)))
        except (TypeError, ValueError) as exc:
            raise LaunchError(f"oracle route {name} has non-integer dimensions",
                              code=EXIT_MANIFEST, category="manifest") from exc
        if got != shape or not row.get("backend"):
            raise LaunchError(f"oracle route {name} shape/backend mismatch: expected {shape}, got {got}",
                              code=EXIT_MANIFEST, category="manifest")
    return routes


def _relative_contract_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise LaunchError(f"{label} path is missing", code=EXIT_TUNER, category="tuner")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise LaunchError(
            f"{label} path must be root-relative without traversal: {value}",
            code=EXIT_TUNER, category="tuner",
        )
    return path


def _root_bound_file(root: Path, base: Path, value: Any, label: str) -> Path:
    raw = Path(str(value or ""))
    candidate = raw if raw.is_absolute() else base / raw
    require_regular(candidate, code=EXIT_TUNER, category="tuner")
    path = candidate.resolve()
    if not within(path, root):
        raise LaunchError(
            f"{label} is outside expected root: {path}", code=EXIT_TUNER, category="tuner",
        )
    return path


def _literal_string_assignment(path: Path, name: str) -> str:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise LaunchError(
            f"cannot parse task runtime loader: {exc}", code=EXIT_TUNER, category="tuner",
        ) from exc
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == name for target in targets):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (TypeError, ValueError) as exc:
            raise LaunchError(
                f"task runtime loader {name} must be a literal string",
                code=EXIT_TUNER, category="tuner",
            ) from exc
        if isinstance(value, str) and value:
            return value
    raise LaunchError(
        f"task runtime loader does not declare {name}", code=EXIT_TUNER, category="tuner",
    )


def _literal_assignment(path: Path, name: str) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise LaunchError(f"cannot parse task binding {path}: {exc}",
                          code=EXIT_TUNER, category="tuner") from exc
    for node in tree.body:
        targets: Sequence[ast.expr] = ()
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = (node.target,)
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            try:
                return ast.literal_eval(node.value)
            except (TypeError, ValueError) as exc:
                raise LaunchError(f"task binding {name} must be a literal",
                                  code=EXIT_TUNER, category="tuner") from exc
    raise LaunchError(f"task binding {path}:{name} is missing",
                      code=EXIT_TUNER, category="tuner")


def _function_sha256(path: Path, name: str) -> str:
    try:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise LaunchError(f"cannot parse task function {path}:{name}: {exc}",
                          code=EXIT_TUNER, category="tuner") from exc
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            segment = ast.get_source_segment(text, node)
            if segment is None:
                lines = text.splitlines(keepends=True)
                segment = "".join(lines[node.lineno - 1:node.end_lineno])
            return hashlib.sha256(segment.encode("utf-8")).hexdigest()
    raise LaunchError(f"task function {path}:{name} is missing",
                      code=EXIT_TUNER, category="tuner")


def _artifact_set_summary(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    normalized = [
        {"kind": str(row["kind"]), "path": str(row["path"]), "sha256": str(row["sha256"])}
        for row in rows
    ]
    kinds: dict[str, int] = {}
    for row in normalized:
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
    return {
        "schema": ARTIFACT_SET_SCHEMA,
        "count": len(normalized),
        "kinds": dict(sorted(kinds.items())),
        "sha256": sha256_json(normalized),
    }


def check_tuner(spec: LaunchSpec, routes: Mapping[str, tuple[int, int, int]]) -> None:
    if (spec.args.get("tuner_registry") != str(spec.tuner_registry)
            or spec.args.get("tuner_table") != str(spec.tuner_table)):
        raise LaunchError(
            "workflow integration does not accept the exact tuner_registry/tuner_table launch args",
            code=EXIT_TUNER, category="tuner")
    for path in (spec.tuner_registry, spec.tuner_table):
        if not within(path, spec.root):
            raise LaunchError(f"tuner launch input is outside expected root: {path}",
                              code=EXIT_TUNER, category="tuner")
        require_regular(path, code=EXIT_TUNER, category="tuner")
        require_tracked(spec.root, path, code=EXIT_TUNER, category="tuner")

    registry = read_json(spec.tuner_registry, code=EXIT_TUNER, category="tuner")
    table = read_json(spec.tuner_table, code=EXIT_TUNER, category="tuner")
    if (registry.get("schema") != CONFIG_REGISTRY_SCHEMA
            or registry.get("schema_version") != CONFIG_REGISTRY_VERSION):
        raise LaunchError(
            f"tuner registry schema must be {CONFIG_REGISTRY_SCHEMA} "
            f"version {CONFIG_REGISTRY_VERSION}", code=EXIT_TUNER, category="tuner",
        )
    contract = registry.get("launch_contract")
    if not isinstance(contract, dict) or contract.get("schema") != LAUNCH_CONTRACT_SCHEMA:
        raise LaunchError("tuner registry has no versioned launch_contract",
                          code=EXIT_TUNER, category="tuner")
    protocol = contract.get("protocol") or {}
    if (contract.get("task") != spec.task
            or protocol.get("schema") != "geak.measurement-protocol/v2"
            or protocol.get("version") != 2
            or protocol.get("hash") != EXPECTED_PROTOCOL_HASH
            or contract.get("protocol_hash") != EXPECTED_PROTOCOL_HASH
            or contract.get("score_metric") != EXPECTED_SCORE_METRIC
            or contract.get("score_metric_hash") != EXPECTED_SCORE_METRIC["hash"]
            or contract.get("score_metric_id") != EXPECTED_SCORE_METRIC_ID
            or contract.get("score_metric_inner") != EXPECTED_SCORE_METRIC["inner"]):
        raise LaunchError("tuner registry task/protocol/score-metric binding mismatch",
                          code=EXIT_TUNER, category="tuner")

    manifest_binding = contract.get("task_manifest")
    if not isinstance(manifest_binding, dict):
        raise LaunchError("tuner registry does not bind the task manifest",
                          code=EXIT_TUNER, category="tuner")
    manifest_rel = _relative_contract_path(manifest_binding.get("path"), "task manifest")
    expected_manifest_rel = Path("asm_tuner/manifests/tasks") / f"{spec.task}.json"
    if manifest_rel != expected_manifest_rel:
        raise LaunchError("tuner registry binds the wrong task manifest",
                          code=EXIT_TUNER, category="tuner")
    manifest_path = (spec.root / manifest_rel).resolve()
    require_regular(manifest_path, code=EXIT_TUNER, category="tuner")
    require_tracked(spec.root, manifest_path, code=EXIT_TUNER, category="tuner")
    if sha256_file(manifest_path) != manifest_binding.get("sha256"):
        raise LaunchError("tuner task-manifest digest is stale", code=EXIT_TUNER,
                          category="tuner")
    manifest = read_json(manifest_path, code=EXIT_TUNER, category="tuner")
    task_root_rel = _relative_contract_path(manifest.get("task_root"), "task root")
    task_root = (spec.root / task_root_rel).resolve()
    if (manifest.get("task") != spec.task
            or (manifest.get("protocol") or {}).get("hash") != EXPECTED_PROTOCOL_HASH
            or manifest.get("score_metric") != EXPECTED_SCORE_METRIC
            or not within(task_root, spec.root)
            or (spec.args.get("kernel_path")
                and Path(str(spec.args["kernel_path"])).resolve() != task_root)):
        raise LaunchError("tuner task manifest/root differs from the launched task",
                          code=EXIT_TUNER, category="tuner")
    materialization = manifest.get("materialization") or {}
    if not isinstance(materialization, dict):
        raise LaunchError("task manifest has no materialization contract",
                          code=EXIT_TUNER, category="tuner")
    registry_rel = _relative_contract_path(materialization.get("registry"), "task registry")
    table_rel = _relative_contract_path(materialization.get("runtime_table"), "task runtime table")
    selection_rel = _relative_contract_path(materialization.get("selection_file"), "task selection")
    loader_rel = _relative_contract_path(materialization.get("dispatch_module"), "task runtime loader")
    expected_registry = (task_root / registry_rel).resolve()
    expected_table = (task_root / table_rel).resolve()
    selection_path = (task_root / selection_rel).resolve()
    if spec.tuner_registry != expected_registry or spec.tuner_table != expected_table:
        raise LaunchError(
            "passed tuner registry/table differ from task-local materialization paths",
            code=EXIT_TUNER, category="tuner",
        )
    if (contract.get("registry") != registry_rel.as_posix()
            or contract.get("runtime_table") != table_rel.as_posix()
            or contract.get("runtime_table_path_base") != "task_root"):
        raise LaunchError("launch_contract does not bind task-local registry/table paths",
                          code=EXIT_TUNER, category="tuner")
    if contract.get("runtime_table_sha256") != sha256_file(spec.tuner_table):
        raise LaunchError("tuner registry runtime-table digest is stale",
                          code=EXIT_TUNER, category="tuner")

    selection_binding = contract.get("selection")
    if (not isinstance(selection_binding, dict)
            or selection_binding.get("path") != selection_rel.as_posix()
            or selection_binding.get("path_base") != "task_root"):
        raise LaunchError("launch_contract does not bind the task selection",
                          code=EXIT_TUNER, category="tuner")
    require_regular(selection_path, code=EXIT_TUNER, category="tuner")
    require_tracked(spec.root, selection_path, code=EXIT_TUNER, category="tuner")
    if sha256_file(selection_path) != selection_binding.get("sha256"):
        raise LaunchError("tuner selection digest is stale", code=EXIT_TUNER,
                          category="tuner")
    selection = read_json(selection_path, code=EXIT_TUNER, category="tuner")

    consumer = contract.get("workflow_consumer")
    if not isinstance(consumer, dict):
        raise LaunchError("tuner registry does not bind the workflow consumer",
                          code=EXIT_TUNER, category="tuner")
    consumer_rel = _relative_contract_path(consumer.get("path"), "workflow consumer")
    expected_consumer = Path("kernel_workflow/kernel_lane.js")
    if consumer_rel != expected_consumer:
        raise LaunchError("tuner registry binds the wrong workflow consumer",
                          code=EXIT_TUNER, category="tuner")
    consumer_path = (spec.root / consumer_rel).resolve()
    require_regular(consumer_path, code=EXIT_TUNER, category="tuner")
    require_tracked(spec.root, consumer_path, code=EXIT_TUNER, category="tuner")
    if sha256_file(consumer_path) != consumer.get("sha256"):
        raise LaunchError("tuner workflow-consumer digest is stale",
                          code=EXIT_TUNER, category="tuner")
    consumer_text = consumer_path.read_text(encoding="utf-8", errors="replace")
    executable_text = re.sub(r"^\s*//.*$", "", consumer_text, flags=re.MULTILINE)
    consumer_markers = ("asm_tuner", "runtime_table", "A.tuner_registry", "A.tuner_table")
    missing_markers = [marker for marker in consumer_markers if marker not in executable_text]
    if missing_markers:
        raise LaunchError(
            "bound workflow consumer is missing tuner integration marker(s): "
            + ", ".join(missing_markers), code=EXIT_TUNER, category="tuner")

    loader_binding = contract.get("task_runtime_loader")
    if not isinstance(loader_binding, dict):
        raise LaunchError("launch_contract does not bind the task runtime loader",
                          code=EXIT_TUNER, category="tuner")
    loader_repo_rel = _relative_contract_path(loader_binding.get("path"), "task runtime loader")
    expected_loader_rel = task_root_rel / loader_rel
    loader_path = (spec.root / loader_repo_rel).resolve()
    if (loader_repo_rel != expected_loader_rel
            or loader_binding.get("runtime_table") != table_rel.as_posix()):
        raise LaunchError("task runtime loader path/table binding mismatch",
                          code=EXIT_TUNER, category="tuner")
    require_regular(loader_path, code=EXIT_TUNER, category="tuner")
    require_tracked(spec.root, loader_path, code=EXIT_TUNER, category="tuner")
    if (sha256_file(loader_path) != loader_binding.get("sha256")
            or _literal_string_assignment(loader_path, "_TABLE_RELATIVE_PATH")
            != table_rel.as_posix()):
        raise LaunchError(
            "task-local runtime loader does not consume the passed runtime table",
            code=EXIT_TUNER, category="tuner",
        )

    if table.get("schema") != RUNTIME_TABLE_SCHEMA:
        raise LaunchError("tuner runtime table schema mismatch",
                          code=EXIT_TUNER, category="tuner")
    table_payload = dict(table)
    table_digest = table_payload.pop("table_digest", None)
    if table_digest != sha256_json(table_payload):
        raise LaunchError("tuner runtime-table digest is stale",
                          code=EXIT_TUNER, category="tuner")
    table_task = table.get("task") or {}
    table_protocol = table.get("protocol") or {}
    table_oracle = table.get("oracle") or {}
    task_digest = str(contract.get("task_digest") or "")
    source_digest = str(contract.get("source_digest") or "")
    oracle_digest = str(contract.get("oracle_manifest_sha256") or "")
    semantic_hashes = table_task.get("semantic_hashes") or {}
    binding_hashes = table_task.get("binding_hashes") or {}
    computed_source_digest = sha256_json({
        "semantic_hashes": semantic_hashes,
        "binding_hashes": binding_hashes,
    })
    if (len(task_digest) != 64 or len(source_digest) != 64 or len(oracle_digest) != 64
            or table_task.get("name") != spec.task
            or table_task.get("digest") != task_digest
            or table_task.get("source_digest") != source_digest
            or computed_source_digest != source_digest
            or table_protocol.get("schema") != "geak.measurement-protocol/v2"
            or table_protocol.get("hash") != EXPECTED_PROTOCOL_HASH
            or table.get("score_metric") != EXPECTED_SCORE_METRIC
            or table_oracle.get("manifest_sha256") != oracle_digest
            or table_oracle.get("compatible") is not True):
        raise LaunchError("tuner task/source/protocol/score/oracle digest binding mismatch",
                          code=EXIT_TUNER, category="tuner")
    if not isinstance(semantic_hashes, dict) or not semantic_hashes or not isinstance(binding_hashes, dict):
        raise LaunchError("tuner runtime table has no canonical source binding",
                          code=EXIT_TUNER, category="tuner")
    for relative, expected_hash in semantic_hashes.items():
        source_rel = _relative_contract_path(relative, "semantic task source")
        source_path = (task_root / source_rel).resolve()
        require_regular(source_path, code=EXIT_TUNER, category="tuner")
        require_tracked(spec.root, source_path, code=EXIT_TUNER, category="tuner")
        if sha256_file(source_path) != expected_hash:
            raise LaunchError(f"tuner semantic source digest is stale: {relative}",
                              code=EXIT_TUNER, category="tuner")
    oracle_rel = _relative_contract_path(manifest.get("oracle_manifest"), "oracle manifest")
    oracle_path = (task_root / oracle_rel).resolve()
    require_regular(oracle_path, code=EXIT_TUNER, category="tuner")
    require_tracked(spec.root, oracle_path, code=EXIT_TUNER, category="tuner")
    if sha256_file(oracle_path) != oracle_digest:
        raise LaunchError("tuner oracle-manifest digest is stale",
                          code=EXIT_TUNER, category="tuner")
    bindings = manifest.get("bindings") or {}
    cases_binding = bindings.get("cases") or {}
    cases_path = _root_bound_file(
        spec.root, task_root, cases_binding.get("path"), "task CASES binding",
    )
    require_tracked(spec.root, cases_path, code=EXIT_TUNER, category="tuner")
    cases_value = _literal_assignment(cases_path, str(cases_binding.get("symbol") or "CASES"))
    try:
        normalized_cases = [
            {key: case[key] for key in ("id", "regime", "M", "N", "K", "seed")}
            for case in cases_value
        ]
    except (KeyError, TypeError) as exc:
        raise LaunchError("task CASES binding is malformed",
                          code=EXIT_TUNER, category="tuner") from exc
    computed_bindings = {"cases": sha256_json(normalized_cases)}
    for binding_name in ("input_builder", "correctness"):
        binding = bindings.get(binding_name) or {}
        binding_path = _root_bound_file(
            spec.root, task_root, binding.get("path"), f"task {binding_name} binding",
        )
        require_tracked(spec.root, binding_path, code=EXIT_TUNER, category="tuner")
        computed_bindings[binding_name] = _function_sha256(
            binding_path, str(binding.get("symbol") or ""),
        )
    if computed_bindings != binding_hashes:
        raise LaunchError("tuner task binding hashes are stale",
                          code=EXIT_TUNER, category="tuner")
    route_manifest_rel = _relative_contract_path(
        manifest.get("route_manifest"), "asm route manifest",
    )
    search_manifest_rel = _relative_contract_path(
        manifest.get("search_space_manifest"), "asm search-space manifest",
    )
    route_manifest_path = (spec.root / route_manifest_rel).resolve()
    search_manifest_path = (spec.root / search_manifest_rel).resolve()
    for dependency in (route_manifest_path, search_manifest_path):
        require_regular(dependency, code=EXIT_TUNER, category="tuner")
        require_tracked(spec.root, dependency, code=EXIT_TUNER, category="tuner")
    computed_task_digest = sha256_json({
        "schema": manifest.get("schema"),
        "task": manifest.get("task"),
        "adapter": manifest.get("adapter"),
        "cases": normalized_cases,
        "semantic_hashes": semantic_hashes,
        "binding_hashes": computed_bindings,
        "route_manifest_sha256": sha256_file(route_manifest_path),
        "search_space_manifest_sha256": sha256_file(search_manifest_path),
        "protocol_hash": EXPECTED_PROTOCOL_HASH,
        "score_metric": EXPECTED_SCORE_METRIC,
        "oracle_manifest_sha256": oracle_digest,
        "materialization": {
            key: materialization[key] for key in (
                "registry", "runtime_table", "selection_file", "dispatch_module",
            )
        },
    })
    if computed_task_digest != task_digest:
        raise LaunchError("tuner canonical task digest is stale",
                          code=EXIT_TUNER, category="tuner")
    try:
        published_revision = int(table.get("revision", 0))
    except (TypeError, ValueError) as exc:
        raise LaunchError("tuner runtime-table revision is malformed",
                          code=EXIT_TUNER, category="tuner") from exc
    suite = table.get("suite") or {}
    table_routes = table.get("routes")
    if (published_revision < 1 or suite.get("atomic") is not True
            or suite.get("complete") is not True
            or suite.get("route_ids") != sorted(routes)
            or not isinstance(table_routes, dict) or set(table_routes) != set(routes)):
        raise LaunchError("tuner runtime-table is not a complete atomic route suite",
                          code=EXIT_TUNER, category="tuner")

    configs = registry.get("configs")
    registry_id = contract.get("registry_id")
    if not isinstance(configs, list) or not isinstance(registry_id, str) or not registry_id:
        raise LaunchError("tuner registry contains no launch-selected config",
                          code=EXIT_TUNER, category="tuner")
    selected_rows = [
        row for row in configs
        if isinstance(row, dict) and row.get("id") == registry_id
    ]
    if len(selected_rows) != 1:
        raise LaunchError("tuner registry launch-selected config is missing or duplicated",
                          code=EXIT_TUNER, category="tuner")
    selected_row = selected_rows[0]
    selected_config = selected_row.get("config") or {}
    build_receipt = selected_row.get("build_receipt") or {}
    if (selected_row.get("precompiled") is not True or selected_row.get("status") != "ready"
            or not isinstance(selected_config, dict) or not isinstance(build_receipt, dict)
            or selected_config.get("kind") != "asm_tuner_release"
            or selected_config.get("task") != spec.task
            or selected_config.get("task_digest") != task_digest
            or selected_config.get("protocol_hash") != EXPECTED_PROTOCOL_HASH
            or selected_config.get("score_metric") != EXPECTED_SCORE_METRIC
            or selected_config.get("score_metric_hash") != EXPECTED_SCORE_METRIC["hash"]
            or selected_config.get("score_metric_id") != EXPECTED_SCORE_METRIC_ID
            or selected_config.get("score_metric_inner") != EXPECTED_SCORE_METRIC["inner"]
            or selected_config.get("oracle_manifest_sha256") != oracle_digest
            or selected_config.get("release_id") != contract.get("release_id")
            or build_receipt.get("task") != spec.task
            or build_receipt.get("task_digest") != task_digest
            or build_receipt.get("source_digest") != source_digest
            or build_receipt.get("protocol_hash") != EXPECTED_PROTOCOL_HASH
            or build_receipt.get("score_metric") != EXPECTED_SCORE_METRIC
            or build_receipt.get("oracle_manifest_sha256") != oracle_digest):
        raise LaunchError("tuner selected registry row has stale task/source/protocol bindings",
                          code=EXIT_TUNER, category="tuner")
    release_manifest = _root_bound_file(
        spec.root, spec.tuner_registry.parent, selected_row.get("artifact"),
        "tuner release manifest",
    )
    if sha256_file(release_manifest) != selected_row.get("artifact_sha256"):
        raise LaunchError("tuner release-manifest hash mismatch",
                          code=EXIT_TUNER, category="tuner")
    release_id = str(contract.get("release_id") or "")
    if len(release_id) != 64:
        raise LaunchError("tuner release id is malformed", code=EXIT_TUNER, category="tuner")
    expected_release_root = task_root / ".geak_tuner" / "releases" / release_id
    receipts_root_rel = _relative_contract_path(table.get("receipts_root"), "table receipts root")
    if ((task_root / receipts_root_rel).resolve() != expected_release_root.resolve()
            or release_manifest != (expected_release_root / "release.json").resolve()):
        raise LaunchError("tuner release is not in the task-local content-addressed store",
                          code=EXIT_TUNER, category="tuner")

    registry_artifacts: dict[tuple[str, str, str], dict[str, str]] = {}
    raw_artifact_set = selected_row.get("artifact_set")
    if not isinstance(raw_artifact_set, list):
        raise LaunchError("tuner selected registry row artifact set is not an array",
                          code=EXIT_TUNER, category="tuner")
    for raw in raw_artifact_set:
        if not isinstance(raw, dict) or raw.get("kind") not in {
                "artifact", "source", "compile_manifest"}:
            raise LaunchError("tuner registry artifact-set row is malformed",
                              code=EXIT_TUNER, category="tuner")
        relative = _relative_contract_path(raw.get("path"), "registry artifact")
        expected_hash = str(raw.get("sha256") or "")
        artifact = _root_bound_file(
            spec.root, spec.tuner_registry.parent, relative, "registry artifact",
        )
        if len(expected_hash) != 64 or sha256_file(artifact) != expected_hash:
            raise LaunchError("tuner registry artifact-set hash mismatch",
                              code=EXIT_TUNER, category="tuner")
        if not within(artifact, task_root):
            raise LaunchError("tuner registry artifact is outside the task root",
                              code=EXIT_TUNER, category="tuner")
        normalized = artifact.relative_to(task_root).as_posix()
        row = {"kind": str(raw["kind"]), "path": normalized, "sha256": expected_hash}
        registry_artifacts[(row["kind"], row["path"], expected_hash)] = row

    table_artifacts: dict[tuple[str, str, str], dict[str, str]] = {}
    table_config_ids: set[str] = set()
    selection_outcomes = selection.get("route_outcomes")
    if not isinstance(selection_outcomes, dict) or set(selection_outcomes) != set(routes):
        raise LaunchError("tuner selection does not cover the complete route set",
                          code=EXIT_TUNER, category="tuner")
    for name, shape in routes.items():
        row = table_routes[name]
        if not isinstance(row, dict):
            raise LaunchError(f"tuner runtime-table route {name} must be an object",
                              code=EXIT_TUNER, category="tuner")
        shape_doc = row.get("shape") or {}
        try:
            got = (int(shape_doc.get("m", -1)), int(shape_doc.get("n", -1)),
                   int(shape_doc.get("k", -1)))
        except (AttributeError, TypeError, ValueError) as exc:
            raise LaunchError(f"tuner runtime-table route {name} has invalid dimensions",
                              code=EXIT_TUNER, category="tuner") from exc
        entries = row.get("entries")
        selection_doc = row.get("selection") or {}
        route_fallback = row.get("fallback") or {}
        outcome_doc = selection_outcomes.get(name) or {}
        if got != shape or not isinstance(entries, list):
            raise LaunchError(f"tuner runtime-table route {name} is incomplete",
                              code=EXIT_TUNER, category="tuner")
        if entries:
            if (selection_doc.get("kind") != "candidate"
                    or selection_doc.get("status") != "winner"
                    or selection_doc.get("config_ids")
                    != [entry.get("config_id") for entry in entries]):
                raise LaunchError(
                    f"tuner runtime-table route {name} candidate selection differs",
                    code=EXIT_TUNER, category="tuner",
                )
        elif (selection_doc.get("kind") != "canonical"
              or selection_doc.get("status") not in ("unmeasured", "no_winner")
              or route_fallback.get("kind") != "canonical"):
            raise LaunchError(
                f"tuner runtime-table route {name} lacks canonical fallback",
                code=EXIT_TUNER, category="tuner",
            )
        expected_kind = "candidate" if outcome_doc.get("selection") == "candidate" else "canonical"
        if (selection_doc.get("kind") != expected_kind
                or selection_doc.get("status") != outcome_doc.get("status")
                or selection_doc.get("reason") != outcome_doc.get("reason")
                or selection_doc.get("candidate_gates")
                != (outcome_doc.get("candidate_gates") or [])
                or selection_doc.get("config_ids")
                != (outcome_doc.get("config_ids") or [])):
            raise LaunchError(
                f"tuner table/selection route outcome differs for {name}",
                code=EXIT_TUNER, category="tuner",
            )
        for rank, entry in enumerate(entries):
            if (not isinstance(entry, dict) or entry.get("rank") != rank
                    or entry.get("task_digest") != task_digest
                    or entry.get("protocol_hash") != EXPECTED_PROTOCOL_HASH
                    or entry.get("score_metric_hash") != EXPECTED_SCORE_METRIC["hash"]
                    or entry.get("metric_id") != EXPECTED_SCORE_METRIC_ID
                    or entry.get("metric_inner") != EXPECTED_SCORE_METRIC["inner"]):
                raise LaunchError(
                    f"tuner runtime-table route {name} has a stale ranked entry",
                    code=EXIT_TUNER, category="tuner",
                )
            table_config_ids.add(str(entry.get("config_id") or ""))
            for kind, path_key, hash_key in (
                ("artifact", "artifact", "artifact_sha256"),
                ("source", "source", "source_sha256"),
                ("compile_manifest", "compile_manifest", "compile_manifest_sha256"),
            ):
                relative = _relative_contract_path(entry.get(path_key), f"table {kind}")
                expected_hash = str(entry.get(hash_key) or "")
                artifact = _root_bound_file(
                    spec.root, spec.tuner_table.parent, relative, f"table {kind}",
                )
                if len(expected_hash) != 64 or sha256_file(artifact) != expected_hash:
                    raise LaunchError(f"tuner runtime-table {kind} hash mismatch",
                                      code=EXIT_TUNER, category="tuner")
                if not within(artifact, task_root):
                    raise LaunchError(f"tuner runtime-table {kind} is outside the task root",
                                      code=EXIT_TUNER, category="tuner")
                normalized = artifact.relative_to(task_root).as_posix()
                artifact_row = {
                    "kind": kind, "path": normalized, "sha256": expected_hash,
                }
                table_artifacts[(kind, normalized, expected_hash)] = artifact_row
    if "" in table_config_ids or table_config_ids != set(selected_config.get("config_ids") or []):
        raise LaunchError("tuner registry/runtime-table config sets differ",
                          code=EXIT_TUNER, category="tuner")
    registry_rows = [registry_artifacts[key] for key in sorted(registry_artifacts)]
    table_rows = [table_artifacts[key] for key in sorted(table_artifacts)]
    if registry_rows != table_rows or contract.get("artifact_set") != _artifact_set_summary(table_rows):
        raise LaunchError("tuner registry/runtime-table artifact-set summary differs",
                          code=EXIT_TUNER, category="tuner")

    if (table.get("release_id") != contract.get("release_id")
            or selection.get("task") != spec.task
            or selection.get("task_digest") != task_digest
            or selection.get("source_digest") != source_digest
            or selection.get("release_id") != contract.get("release_id")
            or selection.get("registry_id") != registry_id
            or selection.get("protocol_hash") != EXPECTED_PROTOCOL_HASH
            or selection.get("score_metric") != EXPECTED_SCORE_METRIC
            or selection.get("oracle_manifest_sha256") != oracle_digest
            or selection.get("runtime_table") != table_rel.as_posix()
            or selection.get("runtime_table_sha256") != sha256_file(spec.tuner_table)
            or selection.get("table_digest") != table_digest
            or set(selection.get("config_ids") or []) != table_config_ids):
        raise LaunchError("tuner selection/registry/runtime-table binding mismatch",
                          code=EXIT_TUNER, category="tuner")


def gpu_sample(gpu: int, sysfs_root: Path = Path("/sys/class/drm")) -> dict[str, Any]:
    render = sysfs_root / f"renderD{128 + 8 * gpu}" / "device"
    try:
        device = render.resolve(strict=True)
        busy = int((device / "gpu_busy_percent").read_text().strip())
        vram_mb = int((device / "mem_info_vram_used").read_text().strip()) // 1048576
    except (OSError, ValueError) as exc:
        raise LaunchError(f"cannot read physical GPU {gpu} health from {render}",
                          code=EXIT_GPU, category="gpu") from exc
    return {"gpu": gpu, "device": str(device), "busy_pct": busy, "vram_mb": vram_mb}


@contextlib.contextmanager
def gpu_preflight_locks(gpu_ids: Sequence[int], *, lock_dir: Path = Path("/tmp/team_gpu_locks"),
                        sysfs_root: Path = Path("/sys/class/drm")) -> Iterator[list[dict[str, Any]]]:
    handles: list[Any] = []
    samples: list[dict[str, Any]] = []
    try:
        for gpu in gpu_ids:
            path = lock_dir / f"gpu_{gpu}.lock"
            require_regular(path, code=EXIT_GPU, category="gpu")
            if not os.access(path, os.W_OK):
                raise LaunchError(f"GPU lock is not writable by yxh: {path}", code=EXIT_GPU,
                                  category="gpu")
            handle = path.open("rb")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                handle.close()
                raise LaunchError(f"GPU {gpu} is already locked", code=EXIT_GPU,
                                  category="gpu") from exc
            handles.append(handle)
        for gpu in gpu_ids:
            sample = gpu_sample(gpu, sysfs_root)
            if sample["busy_pct"] > 5 or sample["vram_mb"] > 1024:
                raise LaunchError(
                    f"GPU {gpu} is not idle: busy={sample['busy_pct']}% vram={sample['vram_mb']}MiB",
                    code=EXIT_GPU, category="gpu")
            samples.append(sample)
        yield samples
    finally:
        for handle in reversed(handles):
            with contextlib.suppress(OSError):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def nearest_existing(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def check_storage(state_dir: Path, *, min_free_bytes: int = MIN_FREE_BYTES,
                  min_free_inodes: int = MIN_FREE_INODES) -> dict[str, int]:
    base = nearest_existing(state_dir.parent)
    if not os.access(base, os.W_OK | os.X_OK):
        raise LaunchError(f"yxh cannot create fresh state below {base}", code=EXIT_IDENTITY,
                          category="identity")
    fs = os.statvfs(base)
    free_bytes = fs.f_bavail * fs.f_frsize
    free_inodes = fs.f_favail
    if free_bytes < min_free_bytes or free_inodes < min_free_inodes:
        raise LaunchError(
            f"insufficient storage at {base}: free={free_bytes} bytes, inodes={free_inodes}; "
            f"require {min_free_bytes} bytes and {min_free_inodes} inodes",
            code=EXIT_DEADLINE, category="storage")
    return {"free_bytes": free_bytes, "free_inodes": free_inodes}


def parse_slurm_end(text: str) -> int:
    text = text.strip()
    if text.isdigit():
        return int(text)
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LaunchError(f"cannot parse SLURM end time {text!r}", code=EXIT_DEADLINE,
                          category="deadline") from exc
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return int(parsed.timestamp())


def resolve_slurm_end(explicit: int | None, job_id: str | None) -> int:
    if explicit:
        return int(explicit)
    for name in ("SLURM_JOB_END_TIME", "SLURM_END_TIME"):
        if os.environ.get(name):
            return parse_slurm_end(os.environ[name])
    job_id = job_id or os.environ.get("SLURM_JOB_ID")
    if job_id and shutil.which("scontrol"):
        proc = subprocess.run(["scontrol", "show", "job", "-o", str(job_id)],
                              text=True, capture_output=True)
        if proc.returncode == 0:
            match = re.search(r"\bEndTime=(\S+)", proc.stdout)
            if match:
                return parse_slurm_end(match.group(1))
    raise LaunchError("SLURM end time is required (--slurm-end-epoch or SLURM_JOB_END_TIME)",
                      code=EXIT_DEADLINE, category="deadline")


def new_deadline(now: int, *, hours: float, hard_end_epoch: int | None,
                 slurm_end_epoch: int) -> Deadline:
    requested = int(hours * 3600)
    if requested <= STOP_NEW_ROUND_MARGIN_S:
        raise LaunchError("requested duration is shorter than the 30-minute drain window",
                          code=EXIT_DEADLINE, category="deadline")
    hard_end = int(hard_end_epoch or (now + requested))
    actual_duration = hard_end - now
    if actual_duration < requested - 5 or actual_duration > requested + 60:
        raise LaunchError("hard end must provide exactly the requested wall-clock duration",
                          code=EXIT_DEADLINE, category="deadline")
    if hard_end > slurm_end_epoch - SLURM_MARGIN_S:
        raise LaunchError(
            f"SLURM allocation ends too early: hard_end={utc(hard_end)}, "
            f"slurm_end={utc(slurm_end_epoch)}, require 15m machine margin",
            code=EXIT_DEADLINE, category="deadline")
    return Deadline(hard_end, hard_end - STOP_NEW_ROUND_MARGIN_S,
                    slurm_end_epoch, requested)


def lock_path(root: Path, lane: str) -> Path:
    return root / "kernel_workflow/launch/locks" / f"{lane}.lock"


@contextlib.contextmanager
def lane_lock(root: Path, lane: str) -> Iterator[Any]:
    path = lock_path(root, lane)
    require_regular(path, code=EXIT_DUPLICATE, category="duplicate")
    require_tracked(root, path)
    handle = path.open("rb")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise LaunchError(f"another {lane} supervisor already holds {path}",
                          code=EXIT_DUPLICATE, category="duplicate") from exc
    try:
        yield handle
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def check_state(spec: LaunchSpec, *, resume: bool) -> None:
    if not resume:
        if spec.state_dir.exists():
            raise LaunchError(f"fresh-state path already exists: {spec.state_dir}")
        return
    if not spec.state_dir.is_dir():
        raise LaunchError(f"resume requested but state directory is absent: {spec.state_dir}")
    launch = spec.state_dir / ".launch"
    require_regular(launch / "deadline.json", code=EXIT_CONFIG, category="config")
    require_regular(launch / "lane.json", code=EXIT_CONFIG, category="config")
    persisted = read_json(launch / "deadline.json")
    expected = {
        "lane": spec.lane,
        "task": spec.task,
        "root": str(spec.root),
        "expected_commit": spec.expected_commit,
        "config_sha256": spec.config_sha256,
        "hard_end_epoch": spec.deadline.hard_end_epoch,
        "stop_new_round_epoch": spec.deadline.stop_new_round_epoch,
    }
    mismatch = {key: (persisted.get(key), value) for key, value in expected.items()
                if persisted.get(key) != value}
    if mismatch:
        raise LaunchError(f"persisted launch contract mismatch: {mismatch}")
    if sha256_json(read_json(launch / "lane.json")) != spec.config_sha256:
        raise LaunchError("persisted rendered lane does not match the expected config")
    wave_state = read_json(launch / "wave_state.json")
    active_pgid = wave_state.get("active_pgid")
    if isinstance(active_pgid, int) and active_pgid > 1 and process_group_exists(active_pgid):
        raise LaunchError(
            f"persisted wave {wave_state.get('active_wave')} still has live process group "
            f"{active_pgid}; refusing a duplicate supervisor",
            code=EXIT_DUPLICATE, category="duplicate")


def render_workflow(spec: LaunchSpec) -> str:
    body = ",\n".join(
        f"    {key}: {json.dumps(value, sort_keys=True)}" for key, value in spec.args.items())
    return (
        "Workflow({\n"
        f"  scriptPath: {json.dumps(str(spec.root / 'kernel_workflow/kernel_workflow.js'), sort_keys=True)},\n"
        "  args: {\n" + body + "\n  }\n})"
    )


def perform_preflight(spec: LaunchSpec, *, resume: bool, settings_path: Path | None = None,
                      min_free_bytes: int = MIN_FREE_BYTES,
                      min_free_inodes: int = MIN_FREE_INODES,
                      api_probe: Callable[[ApiConfig], None] = authenticated_pong,
                      tunnel_probe: Callable[[ApiConfig], None] = probe_tunnel,
                      gpu_lock_dir: Path = Path("/tmp/team_gpu_locks"),
                      sysfs_root: Path = Path("/sys/class/drm")) -> tuple[dict[str, Any], ApiConfig]:
    if min_free_bytes < MIN_FREE_BYTES or min_free_inodes < MIN_FREE_INODES:
        raise LaunchError("storage safety thresholds may be increased but not lowered",
                          code=EXIT_DEADLINE, category="storage")
    api = check_identity_and_credentials(settings_path)
    ambient_protocol = sorted(name for name in PROTOCOL_ENV_VARS if name in os.environ)
    if ambient_protocol:
        raise LaunchError(
            "ambient measurement-protocol overrides would invalidate the pinned hash: "
            + ", ".join(ambient_protocol), code=EXIT_MANIFEST, category="manifest")
    check_git(spec.root, spec.expected_commit)
    for path in (spec.template_path, spec.allocation_path,
                 Path(str(spec.args["sol_shapes"])), spec.tuner_registry, spec.tuner_table):
        if not within(path, spec.root):
            raise LaunchError(f"launch input is outside expected root: {path}")
    validate_lane_args(spec)
    check_state(spec, resume=resume)
    routes = check_oracle_manifest(spec)
    check_tuner(spec, routes)
    storage = check_storage(spec.state_dir, min_free_bytes=min_free_bytes,
                            min_free_inodes=min_free_inodes)
    now = int(time.time())
    if not resume and spec.deadline.hard_end_epoch <= now:
        raise LaunchError("proposed hard end is already expired", code=EXIT_DEADLINE,
                          category="deadline")
    if spec.deadline.hard_end_epoch > spec.deadline.slurm_end_epoch - SLURM_MARGIN_S:
        raise LaunchError("persisted hard end exceeds the safe SLURM window", code=EXIT_DEADLINE,
                          category="deadline")
    if not shutil.which("claude"):
        raise LaunchError("claude CLI is not available in the yxh PATH", code=EXIT_IDENTITY,
                          category="identity")
    tunnel_probe(api)
    api_probe(api)
    with gpu_preflight_locks(spec.gpu_ids, lock_dir=gpu_lock_dir,
                             sysfs_root=sysfs_root) as gpu_rows:
        report = {
            "status": "ok",
            "read_only": True,
            "lane": spec.lane,
            "task": spec.task,
            "root": str(spec.root),
            "commit": spec.expected_commit,
            "state": str(spec.state_dir),
            "fresh": not resume,
            "warm_start": spec.args["warm_start"],
            "gpu_ids": list(spec.gpu_ids),
            "hard_end_utc": utc(spec.deadline.hard_end_epoch),
            "stop_new_round_utc": utc(spec.deadline.stop_new_round_epoch),
            "slurm_end_utc": utc(spec.deadline.slurm_end_epoch),
            "routes": len(routes),
            "protocol_hash": EXPECTED_PROTOCOL_HASH,
            "score_metric": dict(EXPECTED_SCORE_METRIC),
            "tuner_registry": str(spec.tuner_registry),
            "tuner_table": str(spec.tuner_table),
            "storage": storage,
            "gpus": gpu_rows,
            "api": "authenticated PONG",
        }
    return report, api


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


class ControlStore:
    def __init__(self, spec: LaunchSpec) -> None:
        self.spec = spec
        self.directory = spec.state_dir / ".launch"
        self.wave_state = self.directory / "wave_state.json"
        self.supervisor_log = self.directory / "supervisor.log"
        self.health_log = self.directory / "health.jsonl"
        self.gpu_log = self.directory / "gpu_health.jsonl"
        self._io_lock = threading.Lock()

    @classmethod
    def create(cls, spec: LaunchSpec) -> "ControlStore":
        spec.state_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        spec.state_dir.mkdir(mode=0o700)
        self = cls(spec)
        self.directory.mkdir(mode=0o700)
        deadline_doc = {
            "schema": "geak.48h-deadline/v1",
            "lane": spec.lane,
            "task": spec.task,
            "root": str(spec.root),
            "expected_commit": spec.expected_commit,
            "config_sha256": spec.config_sha256,
            "hard_end_epoch": spec.deadline.hard_end_epoch,
            "stop_new_round_epoch": spec.deadline.stop_new_round_epoch,
            "slurm_end_epoch": spec.deadline.slurm_end_epoch,
            "requested_seconds": spec.deadline.requested_seconds,
            "created_at_utc": utc(time.time()),
        }
        _atomic_json(self.directory / "deadline.json", deadline_doc)
        _atomic_json(self.directory / "lane.json", dict(spec.raw_lane))
        _atomic_json(self.wave_state, {
            "next_wave": 1, "restarts_used": 0, "active_wave": None, "active_pgid": None,
        })
        self.say("initialized fresh state and persistent shared deadline")
        return self

    @classmethod
    def resume(cls, spec: LaunchSpec) -> "ControlStore":
        self = cls(spec)
        require_regular(self.wave_state, code=EXIT_CONFIG, category="config")
        return self

    def say(self, message: str) -> None:
        line = f"[{utc(time.time())}] {message}\n"
        with self._io_lock:
            with self.supervisor_log.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
            print(line, end="", flush=True)

    def append_jsonl(self, path: Path, value: Mapping[str, Any]) -> None:
        line = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
        with self._io_lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()

    def state(self) -> dict[str, Any]:
        value = read_json(self.wave_state)
        value["next_wave"] = int(value.get("next_wave", 1))
        value["restarts_used"] = int(value.get("restarts_used", 0))
        return value

    def claim_wave(self) -> int:
        value = self.state()
        wave = value["next_wave"]
        value["next_wave"] = wave + 1
        _atomic_json(self.wave_state, value)
        return wave

    def set_restarts(self, count: int) -> None:
        value = self.state()
        value["restarts_used"] = count
        _atomic_json(self.wave_state, value)

    def recover_interrupted(self, *, now: Callable[[], float] = time.time,
                            sleep: Callable[[float], None] = time.sleep) -> int:
        value = self.state()
        active_wave = value.get("active_wave")
        if active_wave is None:
            return 0
        used = value["restarts_used"]
        if used >= len(RESTART_BACKOFFS):
            self.say(f"interrupted wave {active_wave} found, but restart budget is exhausted")
            return EXIT_RESTARTS
        delay = RESTART_BACKOFFS[used]
        if int(now()) + delay >= self.spec.deadline.stop_new_round_epoch:
            self.say(f"interrupted wave {active_wave} cannot restart before stop-new-round cutoff")
            return EXIT_RESTARTS
        value.update({"active_wave": None, "active_pgid": None,
                      "restarts_used": used + 1})
        _atomic_json(self.wave_state, value)
        self.say(f"recovering interrupted wave {active_wave} as restart {used + 1}/3 after {delay}s")
        sleep(delay)
        return 0

    def mark_active(self, wave: int, pgid: int) -> None:
        value = self.state()
        value.update({"active_wave": wave, "active_pgid": pgid})
        _atomic_json(self.wave_state, value)

    def clear_active(self, wave: int) -> None:
        value = self.state()
        if value.get("active_wave") == wave:
            value.update({"active_wave": None, "active_pgid": None})
            _atomic_json(self.wave_state, value)


def wave_prompt(spec: LaunchSpec, control: ControlStore, wave: int) -> str:
    rules = spec.raw_lane.get("_wave_rules") or []
    rules_text = "\n".join(f"- {item}" for item in rules)
    invocation = render_workflow(spec)
    return f"""Run exactly one complete {spec.lane} lane wave #{wave}.

The rendered lane was generated from the committed root and is also stored at
{control.directory / 'lane.json'}. Do not retype, remove, or add any argument.
Invoke exactly this Workflow and wait until it finishes; do not background it:

{invocation}

Permanent task rules:
{rules_text}

The state is {spec.state_dir}. It is a continuation only within this supervisor's
persistent deadline. Write any wave summary below {control.directory}; never use
a state from another worktree and never edit the immutable oracle/manifest.
"""


def process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def terminate_group(proc: subprocess.Popen[Any], *, grace_s: int = 15) -> bool:
    """Drain the whole wave group, even when the direct claude child already exited.

    Returns whether a group still existed when draining began. A clean child exit
    with a surviving group is an orphaned background workflow and must not be
    treated as a successful wave.
    """
    existed = process_group_exists(proc.pid)
    if not existed:
        return False
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGTERM)
    end = time.monotonic() + grace_s
    while process_group_exists(proc.pid) and time.monotonic() < end:
        proc.poll()  # reap the direct child so a zombie does not keep the group alive
        time.sleep(0.2)
    if process_group_exists(proc.pid):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
    return True


class Watchdog(threading.Thread):
    def __init__(self, *, spec: LaunchSpec, control: ControlStore,
                 proc: subprocess.Popen[Any], api: ApiConfig, interval_s: int = 30,
                 api_interval_s: int = 600) -> None:
        super().__init__(name=f"{spec.lane}-watchdog", daemon=True)
        self.spec = spec
        self.control = control
        self.proc = proc
        self.api = api
        self.interval_s = interval_s
        self.api_interval_s = api_interval_s
        self.stop_event = threading.Event()
        self.reason = ""

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        tunnel_failures = 0
        api_failures = 0
        last_api = time.time()
        while not self.stop_event.is_set() and self.proc.poll() is None:
            now = time.time()
            tunnel_ok = True
            try:
                probe_tunnel(self.api)
                tunnel_failures = 0
            except LaunchError:
                tunnel_ok = False
                tunnel_failures += 1
            api_ok: bool | None = None
            if now - last_api >= self.api_interval_s:
                last_api = now
                try:
                    authenticated_pong(self.api)
                    api_ok = True
                    api_failures = 0
                except LaunchError:
                    api_ok = False
                    api_failures += 1
            if self.stop_event.is_set() or self.proc.poll() is not None:
                return
            gpu_rows: list[dict[str, Any]] = []
            for gpu in self.spec.gpu_ids:
                try:
                    row = gpu_sample(gpu)
                except LaunchError as exc:
                    row = {"gpu": gpu, "error": str(exc)}
                row["timestamp_utc"] = utc(now)
                gpu_rows.append(row)
                self.control.append_jsonl(self.control.gpu_log, row)
            base = nearest_existing(self.spec.state_dir)
            fs = os.statvfs(base)
            self.control.append_jsonl(self.control.health_log, {
                "timestamp_utc": utc(now),
                "pid": self.proc.pid,
                "hard_end_remaining_s": max(0, self.spec.deadline.hard_end_epoch - int(now)),
                "tunnel_ok": tunnel_ok,
                "tunnel_consecutive_failures": tunnel_failures,
                "api_pong": api_ok,
                "api_consecutive_failures": api_failures,
                "free_bytes": fs.f_bavail * fs.f_frsize,
                "free_inodes": fs.f_favail,
            })
            if tunnel_failures >= 3 or api_failures >= 2:
                self.reason = "tunnel_watchdog" if tunnel_failures >= 3 else "api_watchdog"
                self.control.say(f"watchdog stopping wave: {self.reason}")
                terminate_group(self.proc)
                return
            self.stop_event.wait(self.interval_s)


def tail_text(path: Path, limit: int = 64 * 1024) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def run_real_wave(spec: LaunchSpec, control: ControlStore, wave: int, api: ApiConfig,
                  *, watchdog_interval_s: int = 30, api_interval_s: int = 600) -> WaveResult:
    try:
        probe_tunnel(api)
        authenticated_pong(api)
    except LaunchError as exc:
        control.say(f"wave {wave} API pre-probe failed: {exc}")
        return WaveResult(EXIT_API, 0, reason="api_preprobe")
    log_path = control.directory / f"wave_{wave:04d}.log"
    env = dict(os.environ)
    env.update({
        "CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS": "0",
        "GEAK_GPU_ALLOWED": ",".join(map(str, spec.gpu_ids)),
        "GEAK_GPU_USE_LOG": str(control.directory / "gpu_use.jsonl"),
    })
    command = ["claude", "--dangerously-skip-permissions", "--effort", "max", "-p",
               wave_prompt(spec, control, wave)]
    started = int(time.time())
    control.say(f"wave {wave} starting; hard end {utc(spec.deadline.hard_end_epoch)}")
    with log_path.open("xb") as output:
        proc = subprocess.Popen(command, cwd=spec.root, env=env, stdout=output,
                                stderr=subprocess.STDOUT, start_new_session=True)
        control.mark_active(wave, proc.pid)
        watchdog = Watchdog(spec=spec, control=control, proc=proc, api=api,
                            interval_s=watchdog_interval_s, api_interval_s=api_interval_s)
        watchdog.start()
        hit_hard_end = False
        try:
            while proc.poll() is None:
                if int(time.time()) >= spec.deadline.hard_end_epoch:
                    hit_hard_end = True
                    control.say(f"wave {wave} reached external hard end; terminating process group")
                    terminate_group(proc)
                    break
                time.sleep(2)
        finally:
            watchdog.stop()
            watchdog.join(timeout=5)
            parent_was_done = proc.poll() is not None
            orphaned_group = parent_was_done and process_group_exists(proc.pid)
            terminate_group(proc)
            control.clear_active(wave)
        rc = int(proc.returncode if proc.returncode is not None else EXIT_RUNTIME)
    duration = max(0, int(time.time()) - started)
    tail = tail_text(log_path)
    permanent = bool(PERMANENT_LOG_RE.search(tail))
    if orphaned_group and not hit_hard_end:
        rc = rc or 91
    reason = watchdog.reason or ("hard_end" if hit_hard_end else
                                  "orphaned_process_group" if orphaned_group else
                                  "permanent_log" if permanent else
                                  "transport" if TRANSPORT_LOG_RE.search(tail) else "wave_exit")
    control.say(f"wave {wave} exited rc={rc} duration={duration}s reason={reason}")
    return WaveResult(rc, duration, permanent=permanent, hit_hard_end=hit_hard_end,
                      reason=reason)


def supervise_engine(spec: LaunchSpec, control: ControlStore, api: ApiConfig | None,
                     *, launch_wave: Callable[[int], WaveResult],
                     now: Callable[[], float] = time.time,
                     sleep: Callable[[float], None] = time.sleep,
                     min_wave_s: int = MIN_WAVE_S) -> int:
    state = control.state()
    restarts = state["restarts_used"]
    while True:
        current = int(now())
        if current >= spec.deadline.hard_end_epoch:
            control.say("persistent hard end already reached; launching no wave")
            return 0
        if current >= spec.deadline.stop_new_round_epoch:
            control.say("inside 30-minute drain window with no in-flight wave; launching no wave")
            return 0
        wave = control.claim_wave()
        result = launch_wave(wave)
        if result.hit_hard_end:
            control.say("48h hard end completed normally")
            return 0
        if result.permanent:
            control.say(f"wave {wave} has a permanent manifest/config failure; not restarting")
            return result.returncode or EXIT_RUNTIME
        if result.returncode == 0 and result.duration_s >= min_wave_s:
            control.say(f"wave {wave} completed; continuing under the same persistent deadline")
            continue
        if result.returncode == 0:
            result = WaveResult(90, result.duration_s, reason="implausibly_short_success")
            control.say(f"wave {wave} reported success in only {result.duration_s}s; treating as failure")
        if restarts >= len(RESTART_BACKOFFS):
            control.say("restart budget exhausted after three restarts")
            return result.returncode or EXIT_RESTARTS
        delay = RESTART_BACKOFFS[restarts]
        if int(now()) + delay >= spec.deadline.stop_new_round_epoch:
            control.say("failed wave cannot be restarted before the stop-new-round cutoff")
            return result.returncode or EXIT_RESTARTS
        restarts += 1
        control.set_restarts(restarts)
        control.say(f"scheduling restart {restarts}/3 after {delay}s")
        sleep(delay)


def persisted_deadline(state_dir: Path) -> Deadline:
    doc = read_json(state_dir / ".launch/deadline.json")
    try:
        return Deadline(int(doc["hard_end_epoch"]), int(doc["stop_new_round_epoch"]),
                        int(doc["slurm_end_epoch"]), int(doc["requested_seconds"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise LaunchError("persisted deadline is malformed") from exc


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lane", required=True, choices=sorted(LANES))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--tuner-registry", required=True, type=Path)
    parser.add_argument("--tuner-table", required=True, type=Path)
    parser.add_argument("--allocation", type=Path)
    parser.add_argument("--hours", type=float, default=HOURS_DEFAULT)
    parser.add_argument("--hard-end-epoch", type=int)
    parser.add_argument("--slurm-end-epoch", type=int)
    parser.add_argument("--slurm-job-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--credential-file", type=Path)
    parser.add_argument("--min-free-gib", type=float, default=50.0)
    parser.add_argument("--min-free-inodes", type=int, default=MIN_FREE_INODES)


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = out.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight", help="run all checks without creating state")
    add_common(preflight)
    supervise = commands.add_parser("supervise", help="run waves under one persistent 48h deadline")
    add_common(supervise)
    supervise.add_argument("--dry-run", action="store_true",
                           help="run the read-only preflight and launch nothing")
    supervise.add_argument("--watchdog-interval", type=int, default=30)
    supervise.add_argument("--api-watchdog-interval", type=int, default=600)
    render = commands.add_parser("render", help="print the resolved lane without writing it")
    add_common(render)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ns = parser().parse_args(argv)
    try:
        root = ns.root.resolve()
        state = Path(os.path.abspath(ns.state_dir))
        if ns.resume:
            deadline = persisted_deadline(state)
            slurm_end = resolve_slurm_end(ns.slurm_end_epoch, ns.slurm_job_id) \
                if (ns.slurm_end_epoch or ns.slurm_job_id or os.environ.get("SLURM_JOB_END_TIME")
                    or os.environ.get("SLURM_END_TIME") or os.environ.get("SLURM_JOB_ID")) \
                else deadline.slurm_end_epoch
            if slurm_end != deadline.slurm_end_epoch:
                raise LaunchError("resume SLURM end differs from the persisted launch contract",
                                  code=EXIT_DEADLINE, category="deadline")
        else:
            slurm_end = resolve_slurm_end(ns.slurm_end_epoch, ns.slurm_job_id)
            deadline = new_deadline(int(time.time()), hours=ns.hours,
                                    hard_end_epoch=ns.hard_end_epoch,
                                    slurm_end_epoch=slurm_end)
        spec = build_spec(lane=ns.lane, root=root, state_dir=state,
                          expected_commit=ns.expected_commit,
                          tuner_registry=ns.tuner_registry,
                          tuner_table=ns.tuner_table, deadline=deadline,
                          allocation_path=ns.allocation)
        if ns.command == "render":
            print(json.dumps(spec.raw_lane, indent=2, sort_keys=True))
            print(render_workflow(spec))
            return 0
        with lane_lock(spec.root, spec.lane):
            if ns.command == "supervise" and ns.resume \
                    and int(time.time()) >= spec.deadline.hard_end_epoch:
                check_git(spec.root, spec.expected_commit)
                check_state(spec, resume=True)
                print(json.dumps({
                    "status": "complete", "reason": "persistent hard end already reached",
                    "lane": spec.lane, "hard_end_utc": utc(spec.deadline.hard_end_epoch),
                    "state": str(spec.state_dir), "wrote": False,
                }, indent=2, sort_keys=True))
                return 0
            report, api = perform_preflight(
                spec, resume=ns.resume, settings_path=ns.credential_file,
                min_free_bytes=int(ns.min_free_gib * 1024**3),
                min_free_inodes=ns.min_free_inodes)
            print(json.dumps(report, indent=2, sort_keys=True))
            if ns.command == "preflight" or ns.dry_run:
                return 0
            os.umask(0o077)
            control = ControlStore.resume(spec) if ns.resume else ControlStore.create(spec)
            if ns.resume:
                recovery_rc = control.recover_interrupted()
                if recovery_rc:
                    return recovery_rc
            return supervise_engine(
                spec, control, api,
                launch_wave=lambda wave: run_real_wave(
                    spec, control, wave, api,
                    watchdog_interval_s=ns.watchdog_interval,
                    api_interval_s=ns.api_watchdog_interval))
    except LaunchError as exc:
        print(f"REFUSED [{exc.category}]: {exc}", file=sys.stderr)
        return exc.code
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except OSError as exc:
        print(f"FAILED [runtime]: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_RUNTIME


if __name__ == "__main__":
    raise SystemExit(main())
