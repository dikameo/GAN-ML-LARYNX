"""FR-04: split per pasien, 5 lipatan x 2 ulangan, 15% pasien latih sebagai validasi. Plus split S7 (PRD §6).

Pemisahan dilakukan pada TABEL PASIEN (satu baris per pasien), bukan per citra, sehingga stratifikasi
menyeimbangkan jumlah pasien jinak/ganas per lipatan (keputusan 2026-11-08).
S7: pasien resolusi mayoritas -> latih/validasi (n_repeats pembagian acak); gabungan resolusi lain -> uji tunggal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from src.config import P, load_merged
from src.leakage import check_l1, check_l5
from src.utils import atomic_write_bytes, atomic_write_json, config_hash, job_seed


def _patient_table(patients: pd.DataFrame) -> pd.DataFrame:
    return (patients[["patient_uid", "label", "n_images"]].drop_duplicates("patient_uid")
            .sort_values("patient_uid").reset_index(drop=True))


def make_splits(patients: pd.DataFrame, cfg: dict) -> dict:
    sp, seed = cfg["splits"], cfg["seed"]
    pt = _patient_table(patients)
    uids, y = pt.patient_uid.to_numpy(), pt.label.to_numpy()
    repeats = []
    for rep in range(sp["n_repeats"]):
        skf = StratifiedGroupKFold(n_splits=sp["n_splits"], shuffle=True, random_state=job_seed(seed, "split", rep))
        folds = []
        for k, (tr_all, te) in enumerate(skf.split(np.zeros(len(uids)), y, groups=uids)):
            tr, va = train_test_split(tr_all, test_size=sp["val_fraction"], stratify=y[tr_all],
                                      random_state=job_seed(seed, "val", rep, k))
            folds.append({"fold": k, "train": sorted(uids[tr]), "val": sorted(uids[va]), "test": sorted(uids[te])})
        repeats.append({"rep": rep, "folds": folds})
    splits = {"kind": "main", "all_patients": sorted(uids), "repeats": repeats,
              "meta": {"seed": seed, "n_splits": sp["n_splits"], "n_repeats": sp["n_repeats"],
                       "val_fraction": sp["val_fraction"],
                       "patient_table_hash": config_hash(pt.astype(str).to_dict(orient="list"))}}
    splits["meta"]["hash"] = config_hash(splits)
    check_l1(splits)
    return splits


def make_s7_splits(patients: pd.DataFrame, cfg: dict) -> dict:
    """patients: audit.csv (butuh kolom resolution dan n_resolutions). Latih = resolusi mayoritas; uji = sisanya."""
    sp, seed, s7 = cfg["splits"], cfg["seed"], cfg["s7"]
    if (patients.n_resolutions > 1).any():
        raise ValueError("Ada pasien dengan >1 resolusi; S7 tidak dapat menetapkan kelompok (lihat audit)")
    major = patients.resolution.value_counts().index[0]
    tr_pool = patients[patients.resolution == major]
    test = patients[patients.resolution != major]
    pt = _patient_table(tr_pool)
    repeats = []
    for rep in range(s7["n_repeats"]):
        tr, va = train_test_split(np.arange(len(pt)), test_size=sp["val_fraction"], stratify=pt.label.to_numpy(),
                                  random_state=job_seed(seed, "s7val", rep))
        repeats.append({"rep": rep, "folds": [{"fold": 0, "train": sorted(pt.patient_uid.to_numpy()[tr]),
                                               "val": sorted(pt.patient_uid.to_numpy()[va]),
                                               "test": sorted(test.patient_uid)}]})
    splits = {"kind": "s7", "all_patients": sorted(patients.patient_uid), "major_resolution": major,
              "test_resolutions": sorted(test.resolution.unique().tolist()), "repeats": repeats,
              "meta": {"seed": seed, "n_repeats": s7["n_repeats"], "val_fraction": sp["val_fraction"],
                       "patient_table_hash": config_hash(_patient_table(patients).astype(str).to_dict(orient="list"))}}
    splits["meta"]["hash"] = config_hash(splits)
    check_l5(splits, dict(zip(patients.patient_uid, patients.resolution)))
    return splits


def fold_summary(splits: dict, patients: pd.DataFrame) -> pd.DataFrame:
    lab = dict(zip(patients.patient_uid, patients.label))
    nimg = dict(zip(patients.patient_uid, patients.n_images))
    rows = []
    for r in splits["repeats"]:
        for f in r["folds"]:
            row = {"rep": r["rep"], "fold": f["fold"]}
            for part in ("train", "val", "test"):
                ids = f[part]
                row[f"{part}_pasien"] = len(ids)
                row[f"{part}_ganas"] = int(sum(lab[i] for i in ids))
                row[f"{part}_citra"] = int(sum(nimg[i] for i in ids))
            rows.append(row)
    return pd.DataFrame(rows)


def _name(splits_name: str) -> tuple[str, str]:
    return ("splits.json", "fold_summary.csv") if splits_name == "main" else (f"splits_{splits_name}.json", f"fold_summary_{splits_name}.csv")


def save_splits(cfg: dict, splits: dict, patients: pd.DataFrame, splits_name: str = "main") -> Path:
    d, (fj, fc) = P(cfg, "splits"), _name(splits_name)
    atomic_write_json(d / fj, splits)
    atomic_write_bytes(d / fc, fold_summary(splits, patients).to_csv(index=False).encode("utf-8"))
    return d / fj


def load_splits(cfg: dict, splits_name: str | None = None) -> dict:
    fj, _ = _name(splits_name or cfg.get("splits_name", "main"))
    return json.loads((P(cfg, "splits", fj)).read_text(encoding="utf-8"))


def get_fold(splits: dict, rep: int, fold: int) -> dict:
    return splits["repeats"][rep]["folds"][fold]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--s7", action="store_true", help="buat juga split S7 (butuh resolusi seragam per pasien)")
    args = ap.parse_args()
    cfg = load_merged(args.config)
    pt = pd.read_csv(P(cfg, "cp1", "audit.csv"))
    s = make_splits(pt, cfg)
    save_splits(cfg, s, pt, "main")
    print(fold_summary(s, pt).to_string(index=False))
    if args.s7:
        s7 = make_s7_splits(pt, cfg)
        save_splits(cfg, s7, pt, "s7")
        print("\nS7: latih", s7["major_resolution"], "| uji", s7["test_resolutions"])
        print(fold_summary(s7, pt).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
