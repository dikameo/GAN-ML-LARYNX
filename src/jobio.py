"""Keluaran job yang seragam untuk S1 dan S2: direktori job, daftar pasien (L3), prediksi per citra dan pasien."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import P
from src.utils import atomic_write_bytes, atomic_write_json


def job_dir(cfg: dict, scenario: str, rep: int, fold: int) -> Path:
    return P(cfg, "jobs", scenario, f"rep{rep}_fold{fold}")


def resolution_map(cfg: dict) -> dict[str, str]:
    """patient_uid -> resolusi (dari audit.csv, CP1); dipakai pemeriksaan L5 untuk S7."""
    a = pd.read_csv(P(cfg, "cp1", "audit.csv"))
    return dict(zip(a.patient_uid, a.resolution))


def write_run_info(d: Path, cfg: dict, cfg_hash: str, seed: int, extra: dict) -> None:
    import torch

    atomic_write_json(d / "run_info.json", {
        "scenario": cfg.get("scenario"), "config_hash": cfg_hash, "seed": seed,
        "git_commit": os.environ.get("GIT_COMMIT", "tidak-diketahui"),
        "image_digest": os.environ.get("IMAGE_DIGEST", "tidak-diketahui"),
        "torch": torch.__version__, **extra})


def write_preds(d: Path, name: str, meta: pd.DataFrame, idx: np.ndarray, probs: np.ndarray, thr: float) -> pd.DataFrame:
    """preds_<name>_images.csv (per citra) dan preds_<name>_patients.csv (rata-rata per pasien, FR-09)."""
    df = pd.DataFrame({"idx": idx, "patient_uid": meta.patient_uid.to_numpy()[idx],
                       "label": meta.label.to_numpy()[idx], "prob_malignant": probs})
    atomic_write_bytes(d / f"preds_{name}_images.csv", df.to_csv(index=False).encode("utf-8"))
    pat = df.groupby("patient_uid").agg(label=("label", "first"), prob_malignant=("prob_malignant", "mean"),
                                        n_images=("idx", "size")).reset_index()
    pat["pred"] = (pat.prob_malignant >= thr).astype(int)
    atomic_write_bytes(d / f"preds_{name}_patients.csv", pat.to_csv(index=False).encode("utf-8"))
    return pat


def balanced_logloss(y: np.ndarray, p1: np.ndarray, eps: float = 1e-7) -> float:
    """Rata-rata loss per kelas (kelas ganas jarang, jadi tidak boleh tenggelam)."""
    p1 = np.clip(p1, eps, 1 - eps)
    losses = [(-np.log(1 - p1[y == 0])).mean() if (y == 0).any() else None,
              (-np.log(p1[y == 1])).mean() if (y == 1).any() else None]
    return float(np.mean([v for v in losses if v is not None]))
