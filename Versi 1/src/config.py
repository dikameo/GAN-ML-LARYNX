"""Pemuatan konfigurasi bertingkat (_base) dan jalur keluaran.

Jalur dapat diganti lewat variabel lingkungan (berguna di Colab/Kaggle/server tanpa Docker):
  LARYNX_DATASET, LARYNX_OUTPUTS, LARYNX_HF_CACHE
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

from src.utils import load_config

_ENV = {"LARYNX_DATASET": "dataset", "LARYNX_OUTPUTS": "outputs", "LARYNX_HF_CACHE": "hf_cache"}


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load(path: Path) -> dict[str, Any]:
    cfg = load_config(path)
    base = cfg.pop("_base", None)
    if base:
        cfg = deep_merge(_load(path.parent / base), cfg)
    return cfg


def load_merged(path: str | Path) -> dict[str, Any]:
    cfg = _load(Path(path))
    for env, key in _ENV.items():
        if os.environ.get(env):
            cfg["paths"][key] = os.environ[env]
    if os.environ.get("LARYNX_OUTPUTS") and not os.environ.get("LARYNX_HF_CACHE"):
        cfg["paths"]["hf_cache"] = str(Path(os.environ["LARYNX_OUTPUTS"]) / "hf_cache")
    return cfg


def P(cfg: dict, *parts: str) -> Path:
    """Jalur di bawah direktori keluaran."""
    return Path(cfg["paths"]["outputs"]).joinpath(*parts)


def repo_root() -> Path:
    """Akar repositori (direktori yang memuat configs/base.yaml), dicari dari direktori kerja ke atas."""
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "configs" / "base.yaml").exists():
            return p
    raise FileNotFoundError("configs/base.yaml tidak ditemukan dari direktori kerja ke atas")
