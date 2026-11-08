"""S2: embedding Swin beku -> StandardScaler -> SMOTE -> kepala linear (regresi logistik).

Semua fit (scaler, SMOTE, pemilihan C) terjadi di dalam `fit_fold` dan HANYA melihat baris latih/validasi:
larik dibungkus GuardedArray yang menolak akses ke baris uji (L2). C dipilih dari val loss seimbang pasien
validasi di dalam lipatan latih (L4); ambang tetap 0,5.

Jalankan: python -m src.train_s2 --config configs/s2.yaml --rep 0 --fold 0
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import load_merged
from src.data import Cache, find_cache
from src.embed import embedding_path
from src.jobio import balanced_logloss, job_dir, write_preds, write_run_info
from src.leakage import GuardedArray, check_l1, check_l3_job
from src.splits import get_fold, load_splits
from src.utils import atomic_write_json, config_hash, is_done, job_seed, mark_done, set_seed


def fit_fold(X, y, idx_train: np.ndarray, idx_val: np.ndarray, s2: dict, seed: int) -> dict:
    """Fase fit. X dan y WAJIB GuardedArray yang melarang indeks uji. Mengembalikan scaler dan model."""
    Xtr, ytr = X[idx_train], y[idx_train]
    scaler = StandardScaler().fit(Xtr)  # L2: hanya data latih
    Xs = scaler.transform(Xtr)
    k = max(1, min(s2["smote_k"], int(np.bincount(ytr).min()) - 1))
    Xres, yres = SMOTE(sampling_strategy=s2["smote_strategy"], k_neighbors=k, random_state=seed).fit_resample(Xs, ytr)
    Xva, yva = scaler.transform(X[idx_val]), y[idx_val]
    best = None
    for C in s2["c_grid"]:
        clf = LogisticRegression(C=C, max_iter=s2["max_iter"]).fit(Xres, yres)
        vl = balanced_logloss(yva, clf.predict_proba(Xva)[:, 1])
        if best is None or vl < best["val_loss"]:
            best = {"C": C, "val_loss": vl, "clf": clf}
    return {"scaler": scaler, "clf": best["clf"], "C": best["C"], "val_loss": best["val_loss"],
            "n_train": int(len(ytr)), "n_after_smote": int(len(yres))}


def predict(fit: dict, X, idx: np.ndarray) -> np.ndarray:
    return fit["clf"].predict_proba(fit["scaler"].transform(X[idx]))[:, 1]


def run_job(cfg: dict, rep: int, fold: int, cache_dir=None) -> str:
    scenario = cfg["scenario"]
    splits = load_splits(cfg)
    check_l1(splits)
    cache = Cache(cache_dir or find_cache(cfg))
    ed = embedding_path(cfg, cache)
    d = job_dir(cfg, scenario, rep, fold)
    h = config_hash({"cfg": cfg, "rep": rep, "fold": fold, "cache": cache.hash, "splits": splits["meta"]["hash"]})
    if is_done(d, h):
        print(f"[dilewati] {scenario} rep{rep} fold{fold} sudah DONE")
        return str(d)
    if not (ed / "DONE").exists():
        raise FileNotFoundError("Embedding belum selesai (jalankan python -m src.embed)")
    d.mkdir(parents=True, exist_ok=True)
    sf = get_fold(splits, rep, fold)
    atomic_write_json(d / "patients.json", {"train": sf["train"], "val": sf["val"], "test": sf["test"]})
    check_l3_job(json.loads((d / "patients.json").read_text(encoding="utf-8")), sf)

    seed = job_seed(cfg["seed"], scenario, rep, fold)
    set_seed(seed)
    emb = np.load(ed / "emb.npy", mmap_mode="r")
    idx_tr, idx_va, idx_te = (cache.indices_of(sf[k]) for k in ("train", "val", "test"))
    Xg = GuardedArray(emb, forbidden=idx_te)
    yg = GuardedArray(cache.labels, forbidden=idx_te)
    fit = fit_fold(Xg, yg, idx_tr, idx_va, cfg["s2"], seed)  # fase fit: uji tidak terlihat
    assert not (Xg.accessed & set(map(int, idx_te))), "L2 dilanggar"

    thr = cfg["eval"]["threshold"]
    write_preds(d, "val", cache.meta, idx_va, predict(fit, emb, idx_va), thr)
    write_preds(d, "test", cache.meta, idx_te, predict(fit, emb, idx_te), thr)  # uji baru disentuh setelah model beku
    write_run_info(d, cfg, h, seed, {"rep": rep, "fold": fold, "cache_hash": cache.hash,
                                     "splits_hash": splits["meta"]["hash"], "device": "cpu",
                                     "C_terpilih": fit["C"], "val_loss_terpilih": fit["val_loss"],
                                     "n_train_images": fit["n_train"], "n_setelah_smote": fit["n_after_smote"],
                                     "sumber_pemilihan_C": "pasien validasi dalam lipatan latih"})
    mark_done(d, h)
    return str(d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--fold", type=int, required=True)
    a = ap.parse_args()
    run_job(load_merged(a.config), a.rep, a.fold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
