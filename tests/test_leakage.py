"""Uji kebocoran L1-L3 (PRD §10). Data tiruan."""
import copy

import numpy as np
import pytest

from src.leakage import GuardedArray, LeakageError, check_l1, check_l3_job


def _splits():
    pts = [f"P{i}" for i in range(10)]
    folds = []
    for k in range(5):
        te = pts[2 * k:2 * k + 2]
        rest = [p for p in pts if p not in te]
        folds.append({"fold": k, "train": rest[:-1], "val": rest[-1:], "test": te})
    return {"all_patients": pts, "repeats": [{"rep": 0, "folds": folds}]}


def test_l1_passes_on_valid_splits():
    assert check_l1(_splits())


def test_l1_detects_patient_in_train_and_test():
    s = _splits()
    s["repeats"][0]["folds"][0]["train"].append(s["repeats"][0]["folds"][0]["test"][0])
    with pytest.raises(LeakageError, match="L1"):
        check_l1(s)


def test_l1_detects_patient_in_val_and_test():
    s = _splits()
    f = s["repeats"][0]["folds"][1]
    f["val"] = f["val"] + [f["test"][0]]
    with pytest.raises(LeakageError, match="L1"):
        check_l1(s)


def test_l1_detects_patient_never_tested():
    s = _splits()
    f = s["repeats"][0]["folds"][4]
    f["train"] = f["train"] + f["test"][:1]
    f["test"] = f["test"][1:]
    with pytest.raises(LeakageError, match="L1"):
        check_l1(s)


def test_l3_detects_test_patient_in_job():
    f = _splits()["repeats"][0]["folds"][0]
    job = {"train": f["train"] + f["test"][:1], "val": f["val"], "test": f["test"]}
    with pytest.raises(LeakageError, match="L3"):
        check_l3_job(job, f)


def test_l3_detects_list_differing_from_split_file():
    f = _splits()["repeats"][0]["folds"][0]
    with pytest.raises(LeakageError, match="L3"):
        check_l3_job({"train": f["train"][:-1], "val": f["val"], "test": f["test"]}, f)


def test_l3_synthetic_images_must_come_from_train_patients():
    f = _splits()["repeats"][0]["folds"][0]
    job = {"train": f["train"], "val": f["val"], "test": f["test"]}
    assert check_l3_job(job, f, synthetic_source_patients=f["train"][:3])
    with pytest.raises(LeakageError, match="sintetis"):
        check_l3_job(job, f, synthetic_source_patients=f["test"])


def test_guarded_array_blocks_forbidden_rows_and_whole_array():
    g = GuardedArray(np.arange(20).reshape(10, 2), forbidden=[8, 9])
    assert g[np.array([0, 1, 2])].shape == (3, 2)
    with pytest.raises(LeakageError, match="L2"):
        g[np.array([1, 9])]
    with pytest.raises(LeakageError, match="L2"):
        g[5:10]
    with pytest.raises(LeakageError, match="L2"):
        np.asarray(g)


def test_fit_fold_never_touches_test_rows():
    """L2 pada S2: scaler, SMOTE, dan pemilihan C hanya melihat baris latih/validasi."""
    from src.train_s2 import fit_fold

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 8))
    y = np.array([0] * 40 + [1] * 20)
    idx_te = np.arange(50, 60)
    Xg, yg = GuardedArray(X, idx_te), GuardedArray(y, idx_te)
    tr = np.array(list(range(0, 35)) + list(range(40, 47)))
    va = np.array(list(range(35, 40)) + list(range(47, 50)))
    fit = fit_fold(Xg, yg, tr, va, {"smote_k": 3, "smote_strategy": "auto", "c_grid": [0.1, 1.0], "max_iter": 200}, seed=1)
    assert not (Xg.accessed & set(idx_te.tolist())) and fit["n_after_smote"] >= fit["n_train"]
    # akses ke indeks uji pada fase fit harus gagal
    with pytest.raises(LeakageError):
        fit_fold(Xg, yg, np.append(tr, 55), va, {"smote_k": 3, "smote_strategy": "auto", "c_grid": [1.0], "max_iter": 50}, seed=1)
