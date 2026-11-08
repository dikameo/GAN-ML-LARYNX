"""Uji evaluasi (FR-10) pada nilai yang bisa dihitung tangan. Data tiruan."""
import numpy as np

from src.evaluate import _boot_indices, _strata, bootstrap_ci, paired_bootstrap_diff, patient_metrics


def test_metrics_hand_computed():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.2, 0.9, 0.4])  # pred: 0,1,0,1,0 -> tp1 fp1 fn1 tn2
    m = patient_metrics(y, p, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 2)
    assert abs(m["recall"] - 0.5) < 1e-9 and abs(m["precision"] - 0.5) < 1e-9 and abs(m["f1"] - 0.5) < 1e-9
    assert abs(m["accuracy"] - 0.6) < 1e-9 and abs(m["auroc"] - 5 / 6) < 1e-9


def test_threshold_is_inclusive_at_half():
    m = patient_metrics(np.array([0, 1]), np.array([0.2, 0.5]), 0.5)
    assert m["tp"] == 1


def test_bootstrap_is_stratified():
    y = np.array([0] * 7 + [1] * 3)
    rng = np.random.default_rng(1)
    for _ in range(50):
        i = _boot_indices(_strata(y), rng)
        assert (y[i] == 1).sum() == 3 and (y[i] == 0).sum() == 7


def test_bootstrap_ci_perfect_separation_and_reproducible():
    y = np.array([0] * 10 + [1] * 5)
    p = np.where(y == 1, 0.9, 0.1)
    a = bootstrap_ci(y, p, 0.5, 200, 0.95, 7)
    b = bootstrap_ci(y, p, 0.5, 200, 0.95, 7)
    assert a == b  # seed sama -> hasil sama
    assert a["auroc"]["estimate"] == 1.0 and a["auroc"]["ci_low"] == 1.0


def test_ci_contains_estimate_on_noisy_scores():
    rng = np.random.default_rng(3)
    y = np.array([0] * 40 + [1] * 20)
    p = np.clip(rng.normal(0.35 + 0.3 * y, 0.2), 0, 1)
    r = bootstrap_ci(y, p, 0.5, 500, 0.95, 1)
    for m in ("auroc", "f1", "recall"):
        assert r[m]["ci_low"] <= r[m]["estimate"] <= r[m]["ci_high"]


def test_paired_diff_zero_for_identical_and_nonzero_for_different():
    rng = np.random.default_rng(0)
    y = np.array([0] * 30 + [1] * 15)
    p = np.clip(rng.normal(0.3 + 0.4 * y, 0.2), 0, 1)
    same = paired_bootstrap_diff(y, p, p, 0.5, 100, 0.95, 1)
    assert all(v["diff"] == 0 and v["mencakup_nol"] for v in same.values())
    worse = paired_bootstrap_diff(y, p, 1 - p, 0.5, 200, 0.95, 1)
    assert worse["auroc"]["diff"] > 0.3 and not worse["auroc"]["mencakup_nol"]
