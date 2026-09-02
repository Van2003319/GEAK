from __future__ import annotations

import copy
import re
from typing import Any, Dict, Iterable, Mapping

from .schema import DType, KernelConfig


# Minimal records copied from ROCm 7.2 hipBLASLt gfx942 msgpack databases on
# 2026-09-02. Raw records are schema-survival inputs, never default winners and
# never read from /opt/rocm or an experiment directory at runtime. Executable
# counterparts are discovered from the manifest domains, with inactive vendor
# axes neutralized rather than hand-writing an exact vendor point.
_RAW: Dict[str, Dict[str, Any]] = {
    "bf16_solution205554": {
        "solution_id": 205554,
        "database": "TensileLibrary_BB_BB_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx942.dat",
        "kernel_name": "Cijk_Alik_Bljk_BBS_BH_UserArgs_MT224x128x64_MI16x16x1_SN_LDSB1_GRVWA8_GRVWB8_GSU1_MIWT7_4_PGR2_PLR1_SVW4_SK0_WG32_8_1_WGM32_WGMXCC1",
        "size_mapping": {
            "macroTile": [224, 128, 1], "depthU": 64, "matrixInstruction": [16, 16, 16, 1],
            "waveNum": 4, "grvwA": 8, "grvwB": 8, "gwvwD": 4, "globalSplitU": 1,
            "streamK": 0, "workGroupMapping": 32, "workGroupMappingXCC": 1,
            "staggerU": 8, "PrefetchGlobalRead": 2,
        },
        "logical_axis_note": "Tensile raw free-index order is N,M; normalized schema order is M,N.",
    },
    "fp8_solution262788": {
        "solution_id": 262788,
        "database": "TensileLibrary_F8NF8N_BF8N_HA_Bias_SABV_SAV_UA_Type_F8NB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx942.dat",
        "kernel_name": "Cijk_Alik_Bljk_F8NBS_BH_Bias_HA_S_SABV_SAV_UserArgs_MT128x128x128_MI16x16x1_SN_LDSB1_GRVWA16_GRVWB16_GSUn1_MIWT4_4_PGR2_PLR1_SVW4_SK0_WG32_8_1_WGM8_WGMXCC8",
        "size_mapping": {
            "macroTile": [128, 128, 1], "depthU": 128, "matrixInstruction": [16, 16, 32, 1],
            "waveNum": 4, "grvwA": 16, "grvwB": 16, "gwvwD": 4, "globalSplitU": -1,
            "streamK": 0, "workGroupMapping": 8, "workGroupMappingXCC": 8,
            "staggerU": 16, "PrefetchGlobalRead": 2,
        },
        "logical_axis_note": "Square raw and logical macro tiles have identical dimensions.",
    },
}


def raw_fixture(name: str) -> Dict[str, Any]:
    if name not in _RAW:
        raise KeyError("unknown fixture %r" % name)
    return copy.deepcopy(_RAW[name])


def _token_int(kernel_name: str, token: str, default: int) -> int:
    match = re.search(r"(?:^|_)%s(n?\d+)(?:_|$)" % re.escape(token), kernel_name)
    if not match:
        return default
    text = match.group(1)
    return -int(text[1:]) if text.startswith("n") else int(text)


def normalize_vendor_record(record: Mapping[str, Any], dtype: DType) -> KernelConfig:
    sm = record["size_mapping"]
    name = str(record["kernel_name"])
    raw_mt = sm["macroTile"]
    raw_miwt_match = re.search(r"_MIWT(\d+)_(\d+)(?:_|$)", name)
    if not raw_miwt_match:
        raise ValueError("vendor kernel name lacks MIWT")
    raw_miwt = (int(raw_miwt_match.group(1)), int(raw_miwt_match.group(2)))
    mfma = sm["matrixInstruction"]

    # Alik/Bijk uses free-index-0 as logical N and free-index-1 as logical M.
    bm, bn = int(raw_mt[1]), int(raw_mt[0])
    logical_miwt = (raw_miwt[1], raw_miwt[0])
    wave_grid = {
        "m": bm // (logical_miwt[0] * int(mfma[0])),
        "n": bn // (logical_miwt[1] * int(mfma[1])),
    }
    gsu = int(sm.get("globalSplitU", 1))
    if gsu < 1:  # Runtime-selectable vendor value; instantiate neutral GSU=1.
        gsu = 1
    nw = int(sm["waveNum"])
    scratch = nw * int(mfma[0]) * int(mfma[1]) * 4
    return KernelConfig.from_dict({
        "arch": "gfx942",
        "dtype": dtype.value,
        "layout": "a_mk_b_nk",
        "mfma": mfma,
        "bm": bm,
        "bn": bn,
        "bk": int(sm["depthU"]),
        "wave_grid": wave_grid,
        "pgr": _token_int(name, "PGR", int(sm.get("PrefetchGlobalRead", 1))),
        "plr": _token_int(name, "PLR", 0),
        "a_lds": {"buffers": 1, "pad": 0},
        "b_lds": {"buffers": 1, "pad": 0},
        "vector_widths": {"a": int(sm["grvwA"]), "b": int(sm["grvwB"]), "store": int(sm["gwvwD"])},
        "gsu": gsu,
        "stream_k": int(sm.get("streamK", 0)),
        "wgm": int(sm.get("workGroupMapping", 1)),
        "xcc": int(sm.get("workGroupMappingXCC", 1)),
        "stagger": int(sm.get("staggerU", 0)),
        "store": {"mode": "vector"},
        "lds_scratch_bytes": scratch,
        "provenance": {
            "kind": "vendor_schema_survival",
            "solution_id": int(record["solution_id"]),
            "database_basename": record["database"],
            "raw_macro_tile": list(raw_mt[:2]),
            "raw_miwt": list(raw_miwt),
            "logical_axis_note": record["logical_axis_note"],
        },
    })


def get_fixture(name: str) -> KernelConfig:
    record = raw_fixture(name)
    dtype = DType.BF16 if name.startswith("bf16_") else DType.FP8_E4M3_FNUZ
    return normalize_vendor_record(record, dtype)


def default_seed(dtype: DType) -> KernelConfig:
    """A conservative correctness seed; deliberately not a schema-survival record."""

    if dtype is DType.BF16:
        mfma, bk, width = [16, 16, 16, 1], 32, 8
    else:
        mfma, bk, width = [16, 16, 32, 1], 64, 16
    return KernelConfig.from_dict({
        "arch": "gfx942", "dtype": dtype.value, "layout": "a_mk_b_nk", "mfma": mfma,
        "bm": 64, "bn": 64, "bk": bk, "wave_grid": [2, 2], "pgr": 1, "plr": 0,
        "a_lds": {"buffers": 1, "pad": 0}, "b_lds": {"buffers": 1, "pad": 0},
        "vector_widths": {"a": width, "b": width, "store": 4},
        "gsu": 1, "stream_k": 0, "wgm": 1, "xcc": 1, "stagger": 0,
        "store": "vector", "lds_scratch_bytes": 4096,
        "provenance": {"kind": "mvp_correctness_seed", "vendor_default": False},
    })


def generated_survival_candidate(
    configs: Iterable[KernelConfig], name: str,
) -> KernelConfig:
    """Find the executable domain-generated counterpart of a raw vendor record."""

    target = get_fixture(name)
    matches = []
    for config in configs:
        if (
            config.dtype is target.dtype and
            (config.bm, config.bn, config.bk) == (target.bm, target.bn, target.bk) and
            config.wave_grid == target.wave_grid and
            config.miwt == target.miwt and
            config.pgr == 2 and config.plr == 0 and config.xcc == 1 and
            config.gsu == 1 and config.stream_k == 0 and
            config.a_lds.buffers == 1 and config.b_lds.buffers == 1 and
            config.a_lds.pad_elements == 0 and config.b_lds.pad_elements == 0 and
            config.vector_widths == target.vector_widths and
            config.wgm == 1 and config.stagger == 0 and
            config.store == target.store
        ):
            matches.append(config)
    if not matches:
        raise LookupError("generated domains do not contain executable survival point %s" % name)
    return sorted(matches, key=lambda config: config.config_id)[0]
