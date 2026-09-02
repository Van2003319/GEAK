from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from .adapters.base import TaskAdapter
from .metadata import KernelMetadata, MetadataError, parse_assembly
from .schema import KernelConfig


COMPILE_MANIFEST_VERSION = "geak.compile_manifest/v1"


class CompileError(RuntimeError):
    pass


@dataclass(frozen=True)
class CompileResult:
    cache_key: str
    directory: Path
    manifest: Mapping[str, Any]
    cache_hit: bool

    def artifact(self, kind: str) -> Path:
        name = self.manifest["artifacts"].get(kind)
        if not name:
            raise CompileError("manifest does not contain %s artifact" % kind)
        return self.directory / name


@dataclass(frozen=True)
class CompilerFingerprint:
    executable: str
    version: str

    def to_dict(self) -> Dict[str, str]:
        return {"executable": self.executable, "version": self.version}


def shard_configs(configs: Sequence[KernelConfig], shard_size: int = 12, on_demand: bool = False) -> List[Tuple[KernelConfig, ...]]:
    if not configs:
        return []
    if on_demand:
        shard_size = 1
    elif shard_size < 8 or shard_size > 16:
        raise ValueError("shard_size must be in [8,16], or use on_demand")
    ordered = sorted(configs, key=lambda config: config.config_id)
    return [tuple(ordered[i:i + shard_size]) for i in range(0, len(ordered), shard_size)]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp.%d" % os.getpid())
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(str(temp), str(path))


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class ShardCompiler:
    def __init__(self, cache_root: Path, hipcc: Optional[str] = None, optimization: str = "-O3") -> None:
        self.cache_root = Path(cache_root)
        self.hipcc = hipcc or os.environ.get("HIPCC") or shutil.which("hipcc") or "/opt/rocm/bin/hipcc"
        self.optimization = optimization
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._fingerprint: Optional[CompilerFingerprint] = None

    @property
    def fingerprint(self) -> CompilerFingerprint:
        if self._fingerprint is None:
            proc = subprocess.run([self.hipcc, "--version"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if proc.returncode:
                raise CompileError("cannot execute hipcc: %s" % proc.stdout)
            self._fingerprint = CompilerFingerprint(str(Path(self.hipcc).resolve()), proc.stdout.strip())
        return self._fingerprint

    def _cache_key(self, adapter: TaskAdapter, configs: Sequence[KernelConfig], source: str, shared: bool) -> str:
        payload = {
            "manifest": COMPILE_MANIFEST_VERSION,
            "adapter": adapter.name,
            "source_version": adapter.source_version,
            "configs": [config.to_dict(include_provenance=False) for config in configs],
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "compiler": self.fingerprint.to_dict(),
            "optimization": self.optimization,
            "shared": shared,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    @staticmethod
    def _run(command: Sequence[str], log_path: Path) -> None:
        proc = subprocess.run(list(command), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log_path.write_text(proc.stdout)
        if proc.returncode:
            raise CompileError("command failed (%d), see %s: %s" % (proc.returncode, log_path, " ".join(command)))

    def _register_isa(self, cache_key: str, metadata: Sequence[KernelMetadata]) -> Dict[str, Optional[str]]:
        index_path = self.cache_root / "isa_index.json"
        duplicates: Dict[str, Optional[str]] = {}
        with _locked(self.cache_root / "locks" / "isa-index.lock"):
            if index_path.exists():
                index = json.loads(index_path.read_text())
            else:
                index = {"schema": "geak.isa_index/v1", "by_hash": {}}
            by_hash = index.setdefault("by_hash", {})
            for item in metadata:
                existing = by_hash.get(item.isa_hash)
                if existing and existing["config_id"] != item.config_id:
                    duplicates[item.config_id] = existing["config_id"]
                else:
                    duplicates[item.config_id] = None
                    by_hash[item.isa_hash] = {
                        "config_id": item.config_id,
                        "kernel": item.kernel,
                        "cache_key": cache_key,
                    }
            _atomic_json(index_path, index)
        return duplicates

    def compile_shard(self, adapter: TaskAdapter, configs: Sequence[KernelConfig], shared: bool = True) -> CompileResult:
        if not configs:
            raise ValueError("cannot compile empty shard")
        adapter.validate_axis_variation(configs)
        source = adapter.render_source(configs)
        cache_key = self._cache_key(adapter, configs, source, shared)
        target = self.cache_root / "shards" / cache_key
        manifest_path = target / "manifest.json"
        with _locked(self.cache_root / "locks" / (cache_key + ".lock")):
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                required = [target / name for name in manifest["artifacts"].values() if name]
                if all(path.exists() for path in required):
                    return CompileResult(cache_key, target, manifest, True)

            staging_root = self.cache_root / "staging"
            staging_root.mkdir(parents=True, exist_ok=True)
            stage = Path(tempfile.mkdtemp(prefix=cache_key[:16] + ".", dir=str(staging_root)))
            source_path = stage / "shard.hip"
            assembly_path = stage / "shard.s"
            hsaco_path = stage / "shard.hsaco"
            shared_path = stage / "shard.so"
            source_path.write_text(source)
            base = [self.hipcc, "-std=c++17", self.optimization, "--offload-arch=" + configs[0].arch.value]
            commands: List[List[str]] = []
            try:
                command = base + ["--genco", "-S", str(source_path), "-o", str(assembly_path)]
                commands.append(command)
                self._run(command, stage / "assembly.log")
                command = base + ["--genco", str(source_path), "-o", str(hsaco_path)]
                commands.append(command)
                self._run(command, stage / "hsaco.log")
                if shared:
                    command = base + ["-shared", "-fPIC", str(source_path), "-o", str(shared_path)]
                    commands.append(command)
                    self._run(command, stage / "shared.log")

                expected = {adapter.kernel_name(config): config for config in configs}
                parsed = parse_assembly(assembly_path.read_text(errors="replace"), expected)
                for kernel, config in expected.items():
                    actual = parsed[kernel].lds_bytes
                    if actual != config.lds.total_bytes:
                        raise MetadataError(
                            "%s static LDS=%d but amdhsa metadata reports %d" %
                            (config.config_id, config.lds.total_bytes, actual)
                        )
                ordered_metadata = [parsed[adapter.kernel_name(config)] for config in configs]
                duplicates = self._register_isa(cache_key, ordered_metadata)
                kernels = []
                for item in ordered_metadata:
                    row = item.to_dict()
                    row["duplicate_of"] = duplicates[item.config_id]
                    kernels.append(row)
                manifest = {
                    "schema": COMPILE_MANIFEST_VERSION,
                    "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "cache_key": cache_key,
                    "adapter": adapter.name,
                    "source_version": adapter.source_version,
                    "arch": configs[0].arch.value,
                    "compiler": self.fingerprint.to_dict(),
                    "optimization": self.optimization,
                    "axis_status": adapter.axis_status(),
                    "configs": [config.to_dict(include_derived=True) for config in configs],
                    "kernels": kernels,
                    "commands": commands,
                    "artifacts": {
                        "source": source_path.name,
                        "assembly": assembly_path.name,
                        "hsaco": hsaco_path.name,
                        "shared": shared_path.name if shared else None,
                    },
                }
                _atomic_json(stage / "manifest.json", manifest)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    shutil.rmtree(str(target))
                os.replace(str(stage), str(target))
                return CompileResult(cache_key, target, manifest, False)
            except Exception:
                failures = self.cache_root / "failures"
                failures.mkdir(parents=True, exist_ok=True)
                failed = failures / (cache_key[:16] + "." + dt.datetime.now().strftime("%Y%m%dT%H%M%S"))
                if stage.exists():
                    os.replace(str(stage), str(failed))
                raise

    def compile(
        self,
        adapter: TaskAdapter,
        configs: Sequence[KernelConfig],
        shard_size: int = 12,
        on_demand: bool = False,
        shared: bool = True,
    ) -> List[CompileResult]:
        # Validate the whole candidate set before splitting, so an inactive PGR
        # variation cannot hide by placing each value in a separate shard.
        adapter.validate_axis_variation(configs)
        return [
            self.compile_shard(adapter, shard, shared=shared)
            for shard in shard_configs(configs, shard_size=shard_size, on_demand=on_demand)
        ]
