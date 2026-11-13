"""Uji S3/S4 pada fitur TIRUAN (bukan PyRadiomics asli). Membuktikan kode berjalan dan L2 ditegakkan."""
import copy

import numpy as np
import pytest

from src.leakage import GuardedArray, LeakageError
from src.train_s3 import fit_fold, predict


def _cfg(model, alpha=0.001, ctgan=False):
    return {"s3": {"model": model, "lasso_alpha": alpha, "lasso_max_iter": 5000, "svm_C": 1.0, "smote_k": 3, "use_ctgan": ctgan},
            "s4_ctgan": {"epochs": 1, "batch_size": 10, "synth_fraction": 0.5, "cuda": False}}


def _data(n=120, d=12, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([0] * 80 + [1] * 40)
    X = rng.normal(size=(n, d))
    X[:, 0] += 1.5 * y  # satu fitur informatif
    return X, y


@pytest.mark.parametrize("model", ["svm", "lda"])
def test_fit_predict_and_l2(model):
    X, y = _data()
    idx_te = np.arange(100, 120)
    Xg, yg = GuardedArray(X, idx_te), GuardedArray(y, idx_te)
    tr = np.concatenate([np.arange(0, 60), np.arange(80, 95)])
    fit = fit_fold(Xg, yg, tr, np.array([60, 61, 95, 96]), _cfg(model), seed=1)
    assert fit["n_selected"] >= 1 and not (Xg.accessed & set(idx_te.tolist()))
    p = predict(fit, X, idx_te)
    assert p.shape == (20,) and ((p >= 0) & (p <= 1)).all()
    with pytest.raises(LeakageError):
        fit_fold(Xg, yg, np.append(tr, 105), np.array([60, 61]), _cfg(model), seed=1)


def test_lasso_selecting_nothing_fails_loudly():
    X, y = _data()
    with pytest.raises(RuntimeError, match="0 fitur"):
        fit_fold(GuardedArray(X, []), GuardedArray(y, []), np.arange(0, 90), np.array([95, 96]), _cfg("lda", alpha=1e6), seed=1)


def test_nan_features_are_handled_and_counted():
    X, y = _data()
    X[3, 2] = np.nan
    X[110, 4] = np.nan
    idx_te = np.arange(100, 120)
    fit = fit_fold(GuardedArray(X, idx_te), GuardedArray(y, idx_te), np.arange(0, 95), np.array([96, 97]), _cfg("svm"), seed=1)
    assert fit["n_train_dropped_nan"] == 1
    assert predict(fit, X, idx_te)[10] == 0.5  # citra uji tanpa fitur lengkap: probabilitas netral, tercatat


def test_ctgan_augmentation_runs():
    pytest.importorskip("ctgan")
    X, y = _data()
    fit = fit_fold(GuardedArray(X, []), GuardedArray(y, []), np.arange(0, 100), np.array([100, 101]), _cfg("lda", ctgan=True), seed=1)
    assert fit["n_ctgan"] > 0 and fit["n_after_smote"] >= fit["n_ctgan"]
