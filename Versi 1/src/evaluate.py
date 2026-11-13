"""FR-09, FR-10, FR-13: evaluasi tingkat pasien dengan bootstrap persentil bertingkat, selisih berpasangan, tabel hasil.

- Probabilitas tiap pasien = rata-rata probabilitas citra (FR-09); antar ulangan dirata-ratakan lagi (D4).
- Bootstrap: pasien diambil ulang DENGAN pengembalian di dalam tiap kelas (bertingkat), 2.000 ulangan.
- Selisih berpasangan: sampel bootstrap yang sama dipakai untuk dua skenario (pasien yang sama).
- Perbandingan UTAMA (PRD §3): S6 lawan S1 dan S2 pada AUROC dan F1. Semua perbandingan lain berlabel 'eksploratif'.
- Interval hanya menangkap variasi pemilihan pasien uji, bukan variasi pelatihan model.
- Pemilihan rasio r (G2) memakai val_loss_balanced dari riwayat pelatihan (pasien validasi dalam lipatan latih), bukan uji (L4).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from src.config import P, load_merged
from src.jobio import job_dir
from src.splits import load_splits
from src.utils import atomic_write_bytes, atomic_write_json

METRICS = ["auroc", "f1", "recall", "precision", "accuracy"]
PRIMARY = {("S6", "S1"), ("S6", "S2")}


# ===================================================================== metrik dan bootstrap
def patient_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "recall": rec, "precision": prec,
            "accuracy": (tp + tn) / len(y), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _strata(y: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(y == c) for c in np.unique(y)]


def _boot_indices(strata: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    return np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])


def bootstrap_ci(y: np.ndarray, p: np.ndarray, thr: float, reps: int, level: float, seed: int) -> dict:
    est = patient_metrics(y, p, thr)
    rng = np.random.default_rng(seed)
    strata = _strata(y)
    draws = {m: [] for m in METRICS}
    for _ in range(reps):
        i = _boot_indices(strata, rng)
        r = patient_metrics(y[i], p[i], thr)
        for m in METRICS:
            draws[m].append(r[m])
    a = (1 - level) / 2 * 100
    out = {}
    for m in METRICS:
        lo, hi = np.nanpercentile(draws[m], [a, 100 - a])
        out[m] = {"estimate": est[m], "ci_low": float(lo), "ci_high": float(hi)}
    out["confusion"] = {k: est[k] for k in ("tp", "fp", "fn", "tn")}
    return out


def paired_bootstrap_diff(y, pa, pb, thr: float, reps: int, level: float, seed: int) -> dict:
    """Selisih A - B pada pasien yang sama. Bermakna bila interval tidak mencakup nol (PRD §3)."""
    ea, eb = patient_metrics(y, pa, thr), patient_metrics(y, pb, thr)
    rng = np.random.default_rng(seed)
    strata = _strata(y)
    d = {m: [] for m in METRICS}
    for _ in range(reps):
        i = _boot_indices(strata, rng)
        ra, rb = patient_metrics(y[i], pa[i], thr), patient_metrics(y[i], pb[i], thr)
        for m in METRICS:
            d[m].append(ra[m] - rb[m])
    a = (1 - level) / 2 * 100
    out = {}
    for m in METRICS:
        lo, hi = np.nanpercentile(d[m], [a, 100 - a])
        out[m] = {"diff": ea[m] - eb[m], "ci_low": float(lo), "ci_high": float(hi), "mencakup_nol": bool(lo <= 0 <= hi)}
    return out


def unpaired_bootstrap_diff(y1, p1, y2, p2, thr: float, reps: int, level: float, seed: int) -> dict:
    """Selisih kelompok 1 - kelompok 2 pada pasien BERBEDA (S7: dalam-distribusi vs pergeseran resolusi)."""
    e1, e2 = patient_metrics(y1, p1, thr), patient_metrics(y2, p2, thr)
    rng = np.random.default_rng(seed)
    s1, s2 = _strata(y1), _strata(y2)
    d = {m: [] for m in METRICS}
    for _ in range(reps):
        i, j = _boot_indices(s1, rng), _boot_indices(s2, rng)
        r1, r2 = patient_metrics(y1[i], p1[i], thr), patient_metrics(y2[j], p2[j], thr)
        for m in METRICS:
            d[m].append(r1[m] - r2[m])
    a = (1 - level) / 2 * 100
    out = {}
    for m in METRICS:
        lo, hi = np.nanpercentile(d[m], [a, 100 - a])
        out[m] = {"diff": e1[m] - e2[m], "ci_low": float(lo), "ci_high": float(hi), "mencakup_nol": bool(lo <= 0 <= hi)}
    return out


# ===================================================================== agregasi job
def collect_predictions(cfg: dict, scenario: str, jobs_root=None, allow_partial: bool = False) -> tuple[pd.DataFrame, dict]:
    """Baca preds_test_patients.csv dari job berstatus DONE. Tanpa allow_partial, semua job wajib selesai."""
    splits = load_splits(cfg)
    rows, expected, found = [], 0, 0
    for r in splits["repeats"]:
        for f in r["folds"]:
            expected += 1
            d = job_dir(cfg, scenario, r["rep"], f["fold"]) if jobs_root is None else Path(jobs_root) / scenario / f"rep{r['rep']}_fold{f['fold']}"
            if (d / "DONE").exists() and (d / "preds_test_patients.csv").exists():
                df = pd.read_csv(d / "preds_test_patients.csv")
                df["rep"], df["fold"] = r["rep"], f["fold"]
                rows.append(df)
                found += 1
    if found < expected and not allow_partial:
        raise RuntimeError(f"{scenario}: hanya {found}/{expected} job DONE; agregasi hanya membaca job selesai (PRD §8)")
    if not rows:
        raise RuntimeError(f"{scenario}: tidak ada job DONE")
    return pd.concat(rows, ignore_index=True), {"jobs_found": found, "jobs_expected": expected}


def aggregate_patients(preds: pd.DataFrame, average_repeats: bool = True) -> pd.DataFrame:
    g = preds.groupby("patient_uid")
    return g.agg(label=("label", "first"), prob=("prob_malignant", "mean"), n_ulangan=("rep", "nunique")).reset_index()


def fold_table(preds: pd.DataFrame, thr: float) -> pd.DataFrame:
    rows = []
    for (rep, fold), g in preds.groupby(["rep", "fold"]):
        m = patient_metrics(g.label.to_numpy(), g.prob_malignant.to_numpy(), thr)
        rows.append({"rep": rep, "fold": fold, "n_pasien": len(g), "n_ganas": int(g.label.sum()), **m})
    return pd.DataFrame(rows)


def evaluate_preds(cfg: dict, scenario: str, preds: pd.DataFrame, cov: dict, extra_note: str = "") -> dict:
    ev = cfg["eval"]
    pat = aggregate_patients(preds, ev["average_repeats"])
    y, p = pat.label.to_numpy(), pat.prob.to_numpy()
    res = bootstrap_ci(y, p, ev["threshold"], ev["bootstrap_reps"], ev["ci_level"], ev["bootstrap_seed"])
    note = "PARSIAL: tidak semua job selesai" if cov["jobs_found"] < cov["jobs_expected"] else \
        "lengkap; interval = bootstrap persentil bertingkat atas pasien"
    rows = [{"scenario": scenario, "metric": m, "estimate": res[m]["estimate"], "ci_low": res[m]["ci_low"],
             "ci_high": res[m]["ci_high"], "n_pasien": len(pat), "n_ganas": int(y.sum()),
             "job_selesai": f"{cov['jobs_found']}/{cov['jobs_expected']}", "catatan": (note + " " + extra_note).strip()}
            for m in METRICS]
    out_dir = P(cfg, "results")
    atomic_write_bytes(out_dir / f"results_{scenario}.csv", pd.DataFrame(rows).to_csv(index=False).encode("utf-8"))
    atomic_write_bytes(out_dir / f"fold_metrics_{scenario}.csv", fold_table(preds, ev["threshold"]).to_csv(index=False).encode("utf-8"))
    atomic_write_bytes(out_dir / f"patient_probs_{scenario}.csv", pat.to_csv(index=False).encode("utf-8"))
    return {"scenario": scenario, "coverage": cov, "table": pd.DataFrame(rows), "confusion": res["confusion"], "patients": pat}


def evaluate_scenario(cfg: dict, scenario: str, allow_partial: bool = False) -> dict:
    preds, cov = collect_predictions(cfg, scenario, allow_partial=allow_partial)
    return evaluate_preds(cfg, scenario, preds, cov)


def load_result(cfg: dict, scenario: str) -> dict:
    return {"scenario": scenario, "patients": pd.read_csv(P(cfg, "results", f"patient_probs_{scenario}.csv"))}


def compare_scenarios(cfg: dict, a: dict, b: dict, label: str | None = None) -> pd.DataFrame:
    """Selisih berpasangan A - B. Label otomatis: 'utama' hanya untuk S6 vs S1/S2; selain itu 'eksploratif'."""
    ev = cfg["eval"]
    label = label or ("utama" if (a["scenario"], b["scenario"]) in PRIMARY else "eksploratif")
    m = a["patients"].merge(b["patients"], on="patient_uid", suffixes=("_a", "_b"))
    assert (m.label_a == m.label_b).all(), "label pasien tidak konsisten antar skenario"
    d = paired_bootstrap_diff(m.label_a.to_numpy(), m.prob_a.to_numpy(), m.prob_b.to_numpy(), ev["threshold"],
                              ev["bootstrap_reps"], ev["ci_level"], ev["bootstrap_seed"])
    rows = [{"A": a["scenario"], "B": b["scenario"], "metric": k, "selisih_A_minus_B": v["diff"], "ci_low": v["ci_low"],
             "ci_high": v["ci_high"], "mencakup_nol": v["mencakup_nol"], "n_pasien": len(m), "status": label,
             "utama_untuk_metrik": k in ("auroc", "f1") if label == "utama" else False} for k, v in d.items()]
    df = pd.DataFrame(rows)
    atomic_write_bytes(P(cfg, "results", f"compare_{a['scenario']}_vs_{b['scenario']}.csv"), df.to_csv(index=False).encode("utf-8"))
    return df


# ===================================================================== rasio sintetis (G2, L4)
def _min_val_loss(cfg: dict, scenario: str, rep: int, fold: int) -> float | None:
    h = job_dir(cfg, scenario, rep, fold) / "history.csv"
    if not (job_dir(cfg, scenario, rep, fold) / "DONE").exists() or not h.exists():
        return None
    return float(pd.read_csv(h).val_loss_balanced.min())


def assemble_selected(cfg: dict, base: str, ratios: list[float], allow_partial: bool = False) -> tuple[pd.DataFrame, dict]:
    """Untuk tiap (rep, lipatan) pilih rasio dengan val_loss_balanced terendah (pasien VALIDASI), ambil prediksi UJI-nya."""
    splits = load_splits(cfg)
    rows, sel, expected, found = [], [], 0, 0
    for r in splits["repeats"]:
        for f in r["folds"]:
            expected += 1
            vals = {ra: _min_val_loss(cfg, f"{base}_r{int(round(ra * 100))}", r["rep"], f["fold"]) for ra in ratios}
            ok = {k: v for k, v in vals.items() if v is not None}
            if len(ok) < len(ratios) and not allow_partial:
                raise RuntimeError(f"{base} rep{r['rep']} fold{f['fold']}: hanya {len(ok)}/{len(ratios)} rasio selesai")
            if not ok:
                continue
            best = min(ok, key=lambda k: (ok[k], k))
            d = job_dir(cfg, f"{base}_r{int(round(best * 100))}", r["rep"], f["fold"])
            df = pd.read_csv(d / "preds_test_patients.csv")
            df["rep"], df["fold"], df["ratio"] = r["rep"], f["fold"], best
            rows.append(df)
            sel.append({"rep": r["rep"], "fold": f["fold"], "ratio_terpilih": best,
                        **{f"val_loss_r{int(round(k * 100))}": v for k, v in ok.items()},
                        "sumber_pemilihan": "val_loss_balanced pasien validasi dalam lipatan latih"})
            found += 1
    if not rows:
        raise RuntimeError(f"{base}: tidak ada job rasio yang DONE")
    atomic_write_bytes(P(cfg, "results", f"ratio_selection_{base}.csv"), pd.DataFrame(sel).to_csv(index=False).encode("utf-8"))
    return pd.concat(rows, ignore_index=True), {"jobs_found": found, "jobs_expected": expected}


def evaluate_selected(cfg: dict, base: str, allow_partial: bool = False) -> dict:
    ratios = cfg["gan"]["ratios"]
    preds, cov = assemble_selected(cfg, base, ratios, allow_partial)
    return evaluate_preds(cfg, base, preds, cov, extra_note="rasio dipilih per lipatan dari validasi (G2, L4)")


def evaluate_all_ratios(cfg: dict, base: str, allow_partial: bool = False) -> pd.DataFrame:
    """Semua rasio dilaporkan (G2: r=25%, 50%, 100% semuanya)."""
    out = []
    for ra in cfg["gan"]["ratios"]:
        r = evaluate_scenario(cfg, f"{base}_r{int(round(ra * 100))}", allow_partial)
        out.append(r["table"])
    return pd.concat(out, ignore_index=True)


def select_best_config(cfg: dict, bases: tuple[str, ...] = ("S5", "S6")) -> dict:
    """Konfigurasi terbaik untuk S7: rasio dengan rata-rata val_loss_balanced terendah di seluruh lipatan (validasi saja)."""
    splits = load_splits(cfg, "main")
    out = {}
    for base in bases:
        means = {}
        for ra in cfg["gan"]["ratios"]:
            v = [_min_val_loss(cfg, f"{base}_r{int(round(ra * 100))}", r["rep"], f["fold"]) for r in splits["repeats"] for f in r["folds"]]
            if any(x is None for x in v):
                raise RuntimeError(f"{base}_r{int(round(ra * 100))}: ada job belum DONE; konfigurasi terbaik butuh semua lipatan")
            means[ra] = float(np.mean(v))
        best = min(means, key=lambda k: (means[k], k))
        out[base] = {"ratio": best, "mean_val_loss_balanced": {str(k): v for k, v in means.items()},
                     "source": "val_loss_balanced pasien validasi dalam lipatan latih (L4); uji tidak dipakai"}
    sel = P(cfg, "cp4", "lambda_selection.json")
    if sel.exists() and "S6" in out:
        out["S6"]["lambda_roi"] = json.loads(sel.read_text(encoding="utf-8"))["chosen"]
    atomic_write_json(P(cfg, "results", "best_config.json"), out)
    return out


# ===================================================================== S7
def s7_gap(cfg: dict, main_scenario: str, s7_scenario: str) -> pd.DataFrame:
    """Selisih kinerja dalam-distribusi (pasien resolusi mayoritas pada CV utama) vs pergeseran (uji gabungan S7).
    Pasien berbeda dan kondisi pelatihan tidak persis sama (CV utama juga melatih pada resolusi lain) -> EKSPLORATIF."""
    ev = cfg["eval"]
    audit = pd.read_csv(P(cfg, "cp1", "audit.csv"))
    major = load_splits(cfg, "s7")["major_resolution"]
    a = pd.read_csv(P(cfg, "results", f"patient_probs_{main_scenario}.csv")).merge(audit[["patient_uid", "resolution"]], on="patient_uid")
    ind = a[a.resolution == major]
    shifted = pd.read_csv(P(cfg, "results", f"patient_probs_{s7_scenario}.csv"))
    d = unpaired_bootstrap_diff(ind.label.to_numpy(), ind.prob.to_numpy(), shifted.label.to_numpy(), shifted.prob.to_numpy(),
                                ev["threshold"], ev["bootstrap_reps"], ev["ci_level"], ev["bootstrap_seed"])
    e_in = patient_metrics(ind.label.to_numpy(), ind.prob.to_numpy(), ev["threshold"])
    e_sh = patient_metrics(shifted.label.to_numpy(), shifted.prob.to_numpy(), ev["threshold"])
    rows = [{"metode": main_scenario, "metric": m, "dalam_distribusi": e_in[m], "pergeseran_S7": e_sh[m],
             "selisih": v["diff"], "ci_low": v["ci_low"], "ci_high": v["ci_high"], "mencakup_nol": v["mencakup_nol"],
             "n_dalam": len(ind), "n_pergeseran": len(shifted), "status": "eksploratif"} for m, v in d.items()]
    df = pd.DataFrame(rows)
    atomic_write_bytes(P(cfg, "results", f"s7_gap_{main_scenario}.csv"), df.to_csv(index=False).encode("utf-8"))
    return df


# ===================================================================== tabel (FR-13)
ORDER = ["S1", "S2", "S3_svm", "S3_lda", "S4_svm", "S4_lda", "S5", "S6", "S7_S1", "S7_S5", "S7_S6",
         "S3_svm_a01", "S3_lda_a01", "S4_svm_a01", "S4_lda_a01"]


def make_tables(cfg: dict) -> dict:
    res = []
    for sc in ORDER + [f"{b}_r{int(round(r * 100))}" for b in ("S5", "S6") for r in cfg["gan"]["ratios"]]:
        p = P(cfg, "results", f"results_{sc}.csv")
        if p.exists():
            res.append(pd.read_csv(p))
    if not res:
        raise RuntimeError("belum ada results_*.csv")
    allr = pd.concat(res, ignore_index=True)
    allr["sel"] = allr.apply(lambda r: f"{r.estimate:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]", axis=1)
    wide = allr.pivot(index="scenario", columns="metric", values="sel")[METRICS]
    meta = allr.groupby("scenario").agg(n_pasien=("n_pasien", "first"), job_selesai=("job_selesai", "first"),
                                        catatan=("catatan", "first"))
    t33 = wide.join(meta)
    t33["status"] = ["eksploratif (sensitivitas LASSO)" if s.endswith("_a01") else
                     ("rincian rasio" if "_r" in s else "") for s in t33.index]
    t33 = t33.reindex([s for s in ORDER if s in t33.index] + [s for s in t33.index if s not in ORDER])
    atomic_write_bytes(P(cfg, "results", "tabel_3_3.csv"), t33.reset_index().to_csv(index=False).encode("utf-8"))
    atomic_write_bytes(P(cfg, "results", "tabel_3_3.md"), t33.reset_index().to_markdown(index=False).encode("utf-8")
                       if _has_tabulate() else t33.reset_index().to_csv(index=False).encode("utf-8"))

    lit = yaml.safe_load(open("configs/literature.yaml", encoding="utf-8"))["rows"]
    t34 = pd.DataFrame(lit)
    ours = []
    for sc in ("S1", "S2", "S6"):
        r = allr[allr.scenario == sc]
        if not r.empty:
            f1 = r[r.metric == "f1"].iloc[0]
            ours.append({"sumber": f"Penelitian ini ({sc})", "data": "CE-NBI 210 pasien, CV 5x2 per pasien",
                         "metrik": "F1 pasien [CI 95%]", "nilai_validasi_silang": f"{f1.estimate:.3f} [{f1.ci_low:.3f}, {f1.ci_high:.3f}]",
                         "nilai_uji": None, "catatan": "interval = bootstrap pasien; bukan median fold seperti Wang dkk."})
    t34 = pd.concat([t34, pd.DataFrame(ours)], ignore_index=True)
    atomic_write_bytes(P(cfg, "results", "tabel_3_4.csv"), t34.to_csv(index=False).encode("utf-8"))
    return {"tabel_3_3": t33, "tabel_3_4": t34}


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def run_comparisons(cfg: dict) -> pd.DataFrame:
    """Perbandingan utama (S6 vs S1, S6 vs S2) + eksploratif lain yang hasilnya tersedia."""
    pairs = [("S6", "S1"), ("S6", "S2"), ("S6", "S5"), ("S5", "S1"), ("S1", "S2"), ("S6", "S3_svm"), ("S6", "S3_lda"),
             ("S6", "S4_svm"), ("S6", "S4_lda")]
    out = []
    for a, b in pairs:
        if (P(cfg, "results", f"patient_probs_{a}.csv").exists() and P(cfg, "results", f"patient_probs_{b}.csv").exists()):
            out.append(compare_scenarios(cfg, load_result(cfg, a), load_result(cfg, b)))
    if not out:
        raise RuntimeError("belum ada pasangan skenario yang hasilnya lengkap")
    df = pd.concat(out, ignore_index=True)
    atomic_write_bytes(P(cfg, "results", "comparisons_all.csv"), df.to_csv(index=False).encode("utf-8"))
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--selected", default=None, help="S5 atau S6: evaluasi dengan rasio terpilih per lipatan")
    ap.add_argument("--select-best", action="store_true", help="tulis best_config.json (butuh S5 dan S6 selesai)")
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--compare", action="store_true", help="semua perbandingan berpasangan (S6 vs S1/S2 = utama; lain = eksploratif)")
    ap.add_argument("--s7-gap", default=None, metavar="METODE", help="S1, S5, atau S6: selisih dalam-distribusi vs pergeseran S7")
    ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args()
    cfg = load_merged(a.config)
    if a.s7_gap:
        print(s7_gap(cfg, a.s7_gap, f"S7_{a.s7_gap}").to_string(index=False))
    if a.compare:
        print(run_comparisons(cfg).to_string(index=False))
    if a.select_best:
        print(json.dumps(select_best_config(cfg), indent=2))
    if a.selected:
        evaluate_all_ratios(cfg, a.selected, a.allow_partial)
        print(evaluate_selected(cfg, a.selected, a.allow_partial)["table"].to_string(index=False))
    if a.scenario:
        print(evaluate_scenario(cfg, a.scenario, a.allow_partial)["table"].to_string(index=False))
    if a.tables:
        make_tables(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
