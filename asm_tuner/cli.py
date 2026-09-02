from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import yaml

from .adapters import get_adapter
from .compiler import ShardCompiler
from .pruning import route_static_prune, shape_sort_components, shape_sort_key, static_prune
from .runtime_table import RuntimeSelector
from .schema import DType, KernelConfig, Route, validate_config
from .schema_survival import default_seed, get_fixture


def _document(path: Path) -> Any:
    text = Path(path).read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        return yaml.safe_load(text)
    return json.loads(text)


def _configs(path: Path) -> List[KernelConfig]:
    value = _document(path)
    if isinstance(value, Mapping) and "configs" in value:
        value = value["configs"]
    elif isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, list):
        raise ValueError("config document must be a config, a list, or {configs:[...]}")
    return [KernelConfig.from_dict(row) for row in value]


def _routes(path: Path) -> List[Route]:
    value = _document(path)
    if isinstance(value, Mapping) and "routes" in value:
        value = value["routes"]
    elif isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, list):
        raise ValueError("route document must be a route, a list, or {routes:[...]}")
    return [Route.from_dict(row) for row in value]


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _cache_default() -> Path:
    return Path(os.environ.get("GEAK_TUNER_CACHE", str(Path.home() / ".cache" / "geak" / "asm_tuner")))


def _cmd_fixture(args: argparse.Namespace) -> int:
    _print(get_fixture(args.name).to_dict(include_derived=True))
    return 0


def _cmd_seed(args: argparse.Namespace) -> int:
    dtype = DType.BF16 if args.adapter == "bf16" else DType.FP8_E4M3_FNUZ
    _print(default_seed(dtype).to_dict(include_derived=True))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    rows = [validate_config(config).to_dict() for config in _configs(args.configs)]
    _print({"reports": rows})
    return 0 if all(row["valid"] for row in rows) else 2


def _cmd_prune(args: argparse.Namespace) -> int:
    configs = _configs(args.configs)
    result = static_prune(configs)
    ranks: Dict[str, Any] = {}
    route_rejected: Dict[str, Any] = {}
    for route in _routes(args.routes):
        route_result = route_static_prune(result.accepted, route)
        rows = sorted(route_result.accepted, key=lambda config: shape_sort_key(config, route, args.cu_count))
        ranks[route.name] = [
            {
                "config_id": config.config_id,
                "sort_key": list(shape_sort_key(config, route, args.cu_count)),
                "soft_scores": shape_sort_components(config, route, args.cu_count),
                "grid": list(config.grid_for(route)),
            }
            for config in rows
        ]
        route_rejected[route.name] = [row.to_dict() for row in route_result.rejected]
    _print({
        "accepted": [config.config_id for config in result.accepted],
        "rejected": [row.to_dict() for row in result.rejected],
        "shape_ranks": ranks,
        "route_rejected": route_rejected,
        "grid_policy": "grid_and_reuse_are_soft_sort_only_never_hard_prune",
    })
    return 0


def _compile(args: argparse.Namespace, shared: bool):
    adapter = get_adapter(args.adapter)
    configs = _configs(args.configs)
    compiler = ShardCompiler(args.cache, hipcc=args.hipcc, optimization=args.optimization)
    return compiler.compile(
        adapter,
        configs,
        shard_size=args.shard_size,
        on_demand=args.on_demand,
        shared=shared,
    )


def _cmd_compile(args: argparse.Namespace) -> int:
    results = _compile(args, shared=args.shared)
    _print({
        "manifests": [str(result.directory / "manifest.json") for result in results],
        "cache_keys": [result.cache_key for result in results],
        "cache_hits": [result.cache_hit for result in results],
    })
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    raise RuntimeError(
        "the legacy standalone runner used synthetic inputs and cannot publish runtime_table/v2; "
        "use kernel_workflow/scripts/config_search.py search-asm with a canonical task manifest"
    )


def _cmd_select(args: argparse.Namespace) -> int:
    _print(dict(RuntimeSelector.from_path(args.table).select(args.route, args.rank)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="geak-asm-tuner", description="Phase-independent AMD GEMM config compile and selection tool")
    sub = parser.add_subparsers(dest="command", required=True)

    fixture = sub.add_parser("fixture", help="emit a vendor schema-survival record (not executable)")
    fixture.add_argument("name", choices=("bf16_solution205554", "fp8_solution262788"))
    fixture.set_defaults(func=_cmd_fixture)

    seed = sub.add_parser("seed", help="emit a non-vendor correctness seed")
    seed.add_argument("--adapter", required=True, choices=("bf16", "fp8", "fp8_rowwise"))
    seed.set_defaults(func=_cmd_seed)

    validate = sub.add_parser("validate", help="run exact static config validation")
    validate.add_argument("--configs", type=Path, required=True)
    validate.set_defaults(func=_cmd_validate)

    prune = sub.add_parser("prune", help="statically prune and shape-rank configs")
    prune.add_argument("--configs", type=Path, required=True)
    prune.add_argument("--routes", type=Path, required=True)
    prune.add_argument("--cu-count", type=int, required=True)
    prune.set_defaults(func=_cmd_prune)

    def add_compile_flags(command: argparse.ArgumentParser) -> None:
        command.add_argument("--adapter", required=True, choices=("bf16", "fp8", "fp8_rowwise"))
        command.add_argument("--configs", type=Path, required=True)
        command.add_argument("--cache", type=Path, default=_cache_default())
        command.add_argument("--shard-size", type=int, default=12)
        command.add_argument("--on-demand", action="store_true")
        command.add_argument("--hipcc")
        command.add_argument("--optimization", default="-O3")

    compile_cmd = sub.add_parser("compile", help="compile content-addressed config shards")
    add_compile_flags(compile_cmd)
    compile_cmd.add_argument("--shared", action="store_true", help="also build executable task adapter .so")
    compile_cmd.set_defaults(func=_cmd_compile)

    run = sub.add_parser("run", help="compile and legacy synthetic runner (fails closed; use search-asm)")
    add_compile_flags(run)
    run.add_argument("--routes", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--table", type=Path, required=True)
    run.add_argument("--budgets", default="1,3,7")
    run.add_argument("--eta", type=int, default=2)
    run.add_argument("--top-k", type=int, default=1)
    run.add_argument("--warmups", type=int, default=2)
    run.add_argument("--device", type=int, default=0)
    run.set_defaults(func=_cmd_run)

    select = sub.add_parser("select", help="select a route from a versioned runtime table")
    select.add_argument("--table", type=Path, required=True)
    select.add_argument("--route", required=True)
    select.add_argument("--rank", type=int, default=0)
    select.set_defaults(func=_cmd_select)
    return parser


def main(argv: Sequence[str] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(json.dumps({"status": "error", "type": type(exc).__name__, "message": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
