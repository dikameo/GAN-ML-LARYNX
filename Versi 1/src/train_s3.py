"""S3/S4: replikasi ringkas Wang dkk. pada fitur radiomik (PRD §6, FR-08).

S3: MinMax -> LASSO -> SMOTE -> SVM linear atau LDA.
S4: S3 dengan CTGAN pada ruang fitur (sampel sintetis = 50% data latih), lalu SMOTE.
    Urutan CTGAN-sebelum-SMOTE adalah asumsi dari ringkasan PRD; periksa notebook Wang dkk. sebelum klaim replikasi.

Semua fit (scaler, LASSO, CTGAN, SMOTE, klasifikator) HANYA melihat baris latih lewat GuardedArray (L2).
Tidak ada pemilihan hiperparameter pada data uji; alpha LASSO dan C SVM ditetapkan di konfigurasi (D3).

Jalankan: python -m src.train_s3 --config configs/s3_svm.yaml --rep 0 --fold 0
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from src.config import load_merged
from src.data import Cache, find_cache
from src.jobio import job_dir, write_preds, write_run_info
from src.leakage import GuardedArray, check_l1, check_l3_job
from src.radiomics_feats import feature_dir
from src.splits import get_fold, load_splits
from src.utils import atomic_write_json, config_hash, is_done, job_seed, mark_done, set_seed


def _ctgan_augment(Xsel: np.ndarray, y: np.ndarray, c: dict, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from ctgan import CTGAN

    cols = [f"f{i}" for i in range(Xsel.shape[1])]
    df = pd.DataFrame(Xsel, columns=cols)
    df["y"] = y.astype(int)
    cuda = c["cuda"] if c["cuda"] != "auto" else None
    kw = {"epochs": c["epochs"], "batch_size": c["batch_size"], "verbose": False}
    if cuda is not None:
        kw["cuda"] = bool(cuda)
    gan = CTGAN(**kw)
    gan.fit(df, discrete_columns=["y"])
    syn = gan.sample(int(round(c["synth_fraction"] * len(df))))
    return syn[cols].to_numpy(), syn["y"].to_numpy().astype(int)


def fit_fold(X, y, idx_train: np.ndarray, idx_val: np.ndarray, cfg: dict, seed: int) -> dict:
    """Fase fit. X dan y WAJIB GuardedArray yang melarang indeks uji."""
    s3 = cfg["s3"]
    Xtr, ytr = X[idx_train], y[idx_train]
    ok = ~np.isnan(Xtr).any(axis=1)  # citra tanpa fitur (mask kosong) tidak ikut melatih; jumlahnya dicatat
    Xtr, ytr = Xtr[ok], ytr[ok]
    scaler = MinMaxScaler().fit(Xtr)
    Xs = scaler.transform(Xtr)
    lasso = Lasso(alpha=s3["lasso_alpha"], max_iter=s3["lasso_max_iter"], random_state=seed).fit(Xs, ytr)
    sel = np.flatnonzero(np.abs(lasso.coef_) > 0)
    if sel.size == 0:
        raise RuntimeError(f"LASSO (alpha={s3['lasso_alpha']}) memilih 0 fitur; tidak diganti diam-diam. Periksa alpha/fitur.")
    Xsel = Xs[:, sel]
    n_ctgan = 0
    if s3["use_ctgan"]:
        Xg, yg = _ctgan_augment(Xsel, ytr, cfg["s4_ctgan"], seed)
        Xsel, ytr = np.vstack([Xsel, Xg]), np.concatenate([ytr, yg])
        n_ctgan = len(yg)
    k = max(1, min(s3["smote_k"], int(np.bincount(ytr).min()) - 1))
    Xres, yres = SMOTE(sampling_strategy="auto", k_neighbors=k, random_state=seed).fit_resample(Xsel, ytr)
    if s3["model"] == "svm":
        clf = SVC(kernel="linear", C=s3["svm_C"], probability=True, random_state=seed)
    elif s3["model"] == "lda":
        clf = LinearDiscriminantAnalysis()
    else:
        raise ValueError(f"s3.model tidak dikenal: {s3['model']}")
    clf.fit(Xres, yres)
    return {"scaler": scaler, "sel": sel, "clf": clf, "n_selected": int(sel.size), "n_train_dropped_nan": int((~ok).sum()),
            "n_ctgan": n_ctgan, "n_after_smote": int(len(yres))}


def predict(fit: dict, X, idx: np.ndarray) -> np.ndarray:
    Z = np.asarray(X[idx], dtype=np.float64)
    p = np.full(len(idx), 0.5)  # citra tanpa fitur: probabilitas netral 0,5, dihitung dan dicatat
    ok = ~np.isnan(Z).any(axis=1)
    if ok.any():
        p[ok] = fit["clf"].predict_proba(fit["scaler"].transform(Z[ok])[:, fit["sel"]])[:, 1]
    return p


def run_job(cfg: dict, rep: int, fold: int, cache_dir=None) -> str:
    scenario = cfg["scenario"]
    splits = load_splits(cfg)
    check_l1(splits)
    cache = Cache(cache_dir or find_cache(cfg))
    fd = feature_dir(cfg, cache)
    d = job_dir(cfg, scenario, rep, fold)
    h = config_hash({"cfg": cfg, "rep": rep, "fold": fold, "cache": cache.hash, "splits": splits["meta"]["hash"]})
    if is_done(d, h):
        print(f"[dilewati] {scenario} rep{rep} fold{fold} sudah DONE")
        return str(d)
    if not (fd / "DONE").exists():
        raise FileNotFoundError("Fitur radiomik belum selesai (jalankan python -m src.radiomics_feats)")
    d.mkdir(parents=True, exist_ok=True)
    sf = get_fold(splits, rep, fold)
    atomic_write_json(d / "patients.json", {"train": sf["train"], "val": sf["val"], "test": sf["test"]})
    check_l3_job(json.loads((d / "patients.json").read_text(encoding="utf-8")), sf)

    seed = job_seed(cfg["seed"], scenario, rep, fold)
    set_seed(seed)
    feats = np.load(fd / "feat.npy", mmap_mode="r")
    idx_tr, idx_va, idx_te = (cache.indices_of(sf[k]) for k in ("train", "val", "test"))
    Xg = GuardedArray(feats, forbidden=idx_te)
    yg = GuardedArray(cache.labels, forbidden=idx_te)
    fit = fit_fold(Xg, yg, idx_tr, idx_va, cfg, seed)
    assert not (Xg.accessed & set(map(int, idx_te))), "L2 dilanggar"

    thr = cfg["eval"]["threshold"]
    write_preds(d, "val", cache.meta, idx_va, predict(fit, feats, idx_va), thr)
    write_preds(d, "test", cache.meta, idx_te, predict(fit, feats, idx_te), thr)
    write_run_info(d, cfg, h, seed, {"rep": rep, "fold": fold, "cache_hash": cache.hash, "device": "cpu",
                                     "splits_hash": splits["meta"]["hash"], "lasso_alpha": cfg["s3"]["lasso_alpha"],
                                     "n_fitur_dipilih": fit["n_selected"], "n_latih_nan_dibuang": fit["n_train_dropped_nan"],
                                     "n_ctgan": fit["n_ctgan"], "n_setelah_smote": fit["n_after_smote"],
                                     "mask_radiomics": cfg["radiomics"]["mask"]})
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
