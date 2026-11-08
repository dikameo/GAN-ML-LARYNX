"""Uji ujung-ke-ujung pada dataset TIRUAN (45 pasien). Membuktikan kode berjalan dari audit sampai evaluasi;
BUKAN bukti kinerja apa pun pada data asli."""
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest
import yaml

from src.audit import run_audit
from src.config import P
from src.data import Cache, find_cache
from src.leakage import LeakageError, check_l1, check_l3_dir, check_l3_job
from src.preprocess import build_cache, cache_hash
from src.splits import get_fold, load_splits, make_s7_splits, make_splits, save_splits


@pytest.fixture(scope="module")
def stage(synth):
    cfg, _ = synth
    summary = run_audit(cfg)
    pt = pd.read_csv(P(cfg, "cp1", "audit.csv"))
    cache_dir = build_cache(cfg, workers=1)
    splits = make_splits(pt, cfg)
    save_splits(cfg, splits, pt)
    s7 = make_s7_splits(pt, cfg)
    save_splits(cfg, s7, pt, "s7")
    return {"cfg": cfg, "summary": summary, "pt": pt, "cache_dir": cache_dir, "splits": splits, "s7": s7}


def test_audit_counts_and_reference_differences(stage):
    s = stage["summary"]
    assert s["n_patients"] == 45 and s["n_images"] == 135
    ref = pd.DataFrame(s["perbandingan_referensi"])
    assert (ref.selisih_reference_tabel_3_2 != 0).all()  # data tiruan memang berbeda; selisih dilaporkan, tidak disembunyikan
    assert s["xlsx"]["status"] == "tidak_ada" and s["id_bertabrakan_antar_kelas"] == []
    assert "per_resolusi" in s["s7"] and s["s7"]["pasien_multi_resolusi"] == 0


def test_cache_shapes_dtypes_and_skip(stage):
    cfg, d = stage["cfg"], stage["cache_dir"]
    c = Cache(d)
    assert c.images.shape == (135, 256, 256, 3) and c.images.dtype == np.uint8
    assert c.roi.shape == (135, 256, 256) and set(np.unique(c.roi[:5])) <= {0, 1}
    assert c.hash == cache_hash(cfg, pd.read_csv(P(cfg, "cp1", "audit_images.csv")))
    t0 = time.time()
    assert build_cache(cfg, workers=1) == d and time.time() - t0 < 5  # idempoten: dilewati
    assert (c.images[0].sum(axis=2) == 0).any()  # di luar area pandang = hitam


def test_cache_hash_ignores_roi_lock(stage):
    cfg = json.loads(json.dumps(stage["cfg"]))
    img = pd.read_csv(P(cfg, "cp1", "audit_images.csv"))
    h0 = cache_hash(cfg, img)
    cfg["preprocess"]["roi"]["locked"] = True
    assert cache_hash(cfg, img) == h0
    cfg["preprocess"]["roi"]["threshold"] = 0.2
    assert cache_hash(cfg, img) != h0


def test_splits_properties(stage):
    s = stage["splits"]
    check_l1(s)
    assert len(s["repeats"]) == 2 and all(len(r["folds"]) == 5 for r in s["repeats"])
    pt = stage["pt"].set_index("patient_uid")
    for r in s["repeats"]:
        for f in r["folds"]:
            assert pt.loc[f["test"]].label.sum() >= 1  # tiap uji punya pasien ganas
            assert abs(len(f["val"]) / (len(f["val"]) + len(f["train"])) - 0.15) < 0.06
    again = make_splits(stage["pt"], stage["cfg"])
    assert again["meta"]["hash"] == s["meta"]["hash"]  # deterministik


def test_s1_job_runs_resumes_and_skips(stage):
    from src.jobio import job_dir
    from src.train import run_job

    cfg = stage["cfg"]
    d = job_dir(cfg, "S1", 0, 0)
    run_job(cfg, 0, 0)
    for n in ("DONE", "last.pt", "best.pt", "patients.json", "run_info.json", "history.csv",
              "preds_test_images.csv", "preds_test_patients.csv", "preds_val_patients.csv"):
        assert (d / n).exists(), n
    sf = get_fold(stage["splits"], 0, 0)
    pp = pd.read_csv(d / "preds_test_patients.csv")
    assert set(pp.patient_uid) == set(sf["test"]) and pp.prob_malignant.between(0, 1).all()
    assert (pp.pred == (pp.prob_malignant >= 0.5).astype(int)).all()
    assert not [f for f in d.iterdir() if f.suffix == ".tmp"]  # penulisan atomik tidak menyisakan sementara
    check_l3_job(json.loads((d / "patients.json").read_text()), sf)

    m0 = (d / "preds_test_patients.csv").stat().st_mtime_ns
    run_job(cfg, 0, 0)  # sudah DONE -> dilewati
    assert (d / "preds_test_patients.csv").stat().st_mtime_ns == m0

    hist0 = pd.read_csv(d / "history.csv")
    (d / "DONE").unlink()  # simulasi terputus sebelum DONE: dilanjutkan dari last.pt, tidak melatih ulang dari nol
    run_job(cfg, 0, 0)
    assert (d / "DONE").exists() and len(pd.read_csv(d / "history.csv")) == len(hist0)


def test_job_with_changed_config_is_not_skipped(stage):
    from src.jobio import job_dir
    from src.train import run_job

    cfg = json.loads(json.dumps(stage["cfg"]))
    run_job(cfg, 0, 0)
    cfg["train"]["lr"] = 5e-5
    d = job_dir(cfg, "S1", 0, 0)
    old = (d / "DONE").read_text()
    run_job(cfg, 0, 0)
    assert (d / "DONE").read_text() != old  # konfigurasi berubah -> dijalankan ulang (RULES §3)


def test_cli_train_runs_as_script(stage, tmp_path):
    cfgp = tmp_path / "cfg.yaml"
    cfgp.write_text(yaml.safe_dump(stage["cfg"]))
    r = subprocess.run([sys.executable, "-m", "src.train", "--config", str(cfgp), "--rep", "0", "--fold", "1"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    assert (P(stage["cfg"], "jobs", "S1", "rep0_fold1") / "DONE").exists()


def test_s2_pipeline_and_leak_guard(stage, cfg_s2):
    from src.embed import embedding_path, run_embed
    from src.train_s2 import run_job

    run_embed(cfg_s2)
    emb = np.load(embedding_path(cfg_s2, Cache(find_cache(cfg_s2))) / "emb.npy")
    assert emb.shape == (135, 128) and np.isfinite(emb).all() and emb.std() > 0
    for rep in (0, 1):
        for fold in range(5):
            run_job(cfg_s2, rep, fold)
    info = json.loads((P(cfg_s2, "jobs", "S2", "rep0_fold0") / "run_info.json").read_text())
    assert info["C_terpilih"] in cfg_s2["s2"]["c_grid"] and info["n_setelah_smote"] >= info["n_train_images"]
    assert check_l3_dir(P(cfg_s2, "jobs"), "S2", stage["splits"]) == 10


def test_l3_tampering_is_detected(stage, cfg_s2):
    d = P(cfg_s2, "jobs", "S2", "rep0_fold0")
    pj = json.loads((d / "patients.json").read_text())
    sf = get_fold(stage["splits"], 0, 0)
    bad = dict(pj, train=pj["train"] + sf["test"][:1])
    with pytest.raises(LeakageError):
        check_l3_job(bad, sf)


def test_evaluation_full_partial_and_comparison(stage, cfg_s2):
    from src.evaluate import compare_scenarios, evaluate_scenario

    cfg = stage["cfg"]
    r2 = evaluate_scenario(cfg_s2, "S2")  # 10/10 job selesai
    t2 = r2["table"]
    assert set(t2.metric) == {"auroc", "f1", "recall", "precision", "accuracy"} and (t2.job_selesai == "10/10").all()
    assert (t2.ci_low <= t2.estimate + 1e-9).all() and (t2.estimate <= t2.ci_high + 1e-9).all()
    assert len(r2["patients"]) == 45 and (r2["patients"].n_ulangan == 2).all()
    with pytest.raises(RuntimeError, match="job DONE"):
        evaluate_scenario(cfg, "S1")  # S1 baru sebagian -> tanpa allow_partial harus menolak
    r1 = evaluate_scenario(cfg, "S1", allow_partial=True)
    assert r1["table"].catatan.str.startswith("PARSIAL").all()
    cmp_ = compare_scenarios(cfg, r1, r2, label="eksploratif")
    assert (cmp_.status == "eksploratif").all() and len(cmp_) == 5


# ===================================================================== S5/S6: citra sintetis pada klasifikator
def _fake_gan_job(cfg, scenario, rep, fold, source_patients, n=6):
    from src.jobio import job_dir

    gdir = job_dir(cfg, f"{scenario}_gan", rep, fold)
    gdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(gdir / "synth.npz", images=np.random.default_rng(0).integers(0, 255, (n, 256, 256, 3), dtype=np.uint8))
    (gdir / "synth_meta.json").write_text(json.dumps({"n": n, "class": 1, "source_patients": source_patients}))
    (gdir / "DONE").write_text("x")
    return gdir


def test_classifier_with_synthetic_subset_and_l3(stage):
    from src.jobio import job_dir
    from src.train import run_job

    cfg = json.loads(json.dumps(stage["cfg"]))
    cfg["scenario"] = "S5"
    cfg["synthetic"] = {"gan_scenario": "S5", "ratio": 0.5}
    sf = get_fold(stage["splits"], 0, 0)
    _fake_gan_job(cfg, "S5", 0, 0, sf["train"])
    run_job(cfg, 0, 0)
    d = job_dir(cfg, "S5_r50", 0, 0)
    info = json.loads((d / "run_info.json").read_text())
    assert (d / "DONE").exists() and info["n_synthetic"] == 3 and info["n_generated"] == 6 and info["ratio"] == 0.5
    pp = pd.read_csv(d / "preds_test_patients.csv")
    assert set(pp.patient_uid) == set(sf["test"])  # sintetis hanya di data latih; uji berisi pasien asli saja

    sf1 = get_fold(stage["splits"], 0, 1)
    _fake_gan_job(cfg, "S5", 0, 1, sf1["train"] + sf1["test"][:1])  # sumber sintetis mengandung pasien uji
    with pytest.raises(LeakageError):
        run_job(cfg, 0, 1)


def test_classifier_requires_gan_done_and_ratio(stage):
    from src.train import resolve_ratio, run_job

    cfg = json.loads(json.dumps(stage["cfg"]))
    cfg["scenario"] = "S6"
    cfg["synthetic"] = {"gan_scenario": "S6", "ratio": 0.25}
    with pytest.raises(FileNotFoundError, match="GAN"):
        run_job(cfg, 1, 4)  # GAN belum DONE -> urutan wajib ditegakkan
    cfg["synthetic"]["ratio"] = None
    with pytest.raises(ValueError):
        resolve_ratio(cfg)


def test_ratio_selection_uses_validation_loss_not_test(stage):
    from src.evaluate import assemble_selected, select_best_config
    from src.jobio import job_dir

    cfg = json.loads(json.dumps(stage["cfg"]))
    ratios = cfg["gan"]["ratios"]
    rng = np.random.default_rng(0)
    for r in stage["splits"]["repeats"]:
        for f in r["folds"]:
            for k, ra in enumerate(ratios):
                d = job_dir(cfg, f"SX_r{int(round(ra * 100))}", r["rep"], f["fold"])
                d.mkdir(parents=True, exist_ok=True)
                # val loss terendah SELALU pada rasio 0.5; probabilitas uji sengaja paling baik pada rasio lain
                pd.DataFrame({"val_loss_balanced": [0.5 if ra == 0.5 else 0.9, 0.6 if ra == 0.5 else 1.0]}).to_csv(d / "history.csv", index=False)
                pat = pd.DataFrame({"patient_uid": f["test"], "label": [int(p in set(stage["pt"][stage["pt"].label == 1].patient_uid)) for p in f["test"]]})
                pat["prob_malignant"] = pat.label * (0.99 if ra != 0.5 else 0.6) + (1 - pat.label) * 0.1
                pat["n_images"], pat["pred"] = 3, (pat.prob_malignant >= 0.5).astype(int)
                pat.to_csv(d / "preds_test_patients.csv", index=False)
                (d / "DONE").write_text("x")
    preds, cov = assemble_selected(cfg, "SX", ratios)
    assert (preds.ratio == 0.5).all() and cov == {"jobs_found": 10, "jobs_expected": 10}
    assert (P(cfg, "results", "ratio_selection_SX.csv")).exists()
    best = select_best_config(cfg, bases=("SX",))
    assert best["SX"]["ratio"] == 0.5 and "uji tidak dipakai" in best["SX"]["source"]


# ===================================================================== S7
def test_s7_splits_enforce_l5(stage):
    from src.leakage import check_l5

    s7, pt = stage["s7"], stage["pt"]
    res = dict(zip(pt.patient_uid, pt.resolution))
    assert check_l5(s7, res) and s7["kind"] == "s7" and len(s7["repeats"]) == stage["cfg"]["s7"]["n_repeats"]
    f = s7["repeats"][0]["folds"][0]
    assert {res[p] for p in f["train"] + f["val"]} == {s7["major_resolution"]}
    assert s7["major_resolution"] not in {res[p] for p in f["test"]}
    bad = json.loads(json.dumps(s7))
    bad["repeats"][0]["folds"][0]["train"].append(bad["repeats"][0]["folds"][0]["test"][0])
    with pytest.raises(LeakageError, match="L5"):
        check_l5(bad, res)


def test_s7_classifier_job_and_gap(stage):
    from src.evaluate import evaluate_scenario, s7_gap
    from src.jobio import job_dir
    from src.train import run_job

    cfg7 = json.loads(json.dumps(stage["cfg"]))
    cfg7["scenario"], cfg7["splits_name"] = "S7_S1", "s7"
    run_job(cfg7, 0, 0)
    d = job_dir(cfg7, "S7_S1", 0, 0)
    assert (d / "DONE").exists()
    s7f = get_fold(load_splits(cfg7), 0, 0)
    assert set(pd.read_csv(d / "preds_test_patients.csv").patient_uid) == set(s7f["test"])
    r = evaluate_scenario(cfg7, "S7_S1", allow_partial=True)
    assert r["table"].catatan.str.startswith("PARSIAL").all()
    # selisih dalam-distribusi vs pergeseran memakai hasil CV utama S1 yang sudah ada dari uji sebelumnya
    cfgm = json.loads(json.dumps(stage["cfg"]))
    for rep in (0, 1):
        for fold in range(5):
            from src.train import run_job as rj
            rj(cfgm, rep, fold)
    evaluate_scenario(cfgm, "S1")
    gap = s7_gap(cfg7, "S1", "S7_S1")
    assert set(gap.status) == {"eksploratif"} and len(gap) == 5 and (gap.n_pergeseran == len(s7f["test"])).all()


def test_tables_and_primary_comparison_labels(stage, cfg_s2):
    from src.evaluate import compare_scenarios, load_result, make_tables, run_comparisons

    cfg = stage["cfg"]
    from src.evaluate import evaluate_scenario
    evaluate_scenario(cfg_s2, "S2")
    evaluate_scenario(cfg, "S1")
    assert compare_scenarios(cfg, load_result(cfg, "S1"), load_result(cfg, "S2")).status.eq("eksploratif").all()
    # S6 lawan S1 adalah perbandingan UTAMA; labelnya otomatis, tidak bisa lupa
    s6 = load_result(cfg, "S2").copy()
    s6["scenario"] = "S6"
    assert compare_scenarios(cfg, s6, load_result(cfg, "S1")).status.eq("utama").all()
    c = run_comparisons(cfg)
    assert set(c.status) <= {"utama", "eksploratif"}
    t = make_tables(cfg)
    assert {"S1", "S2"} <= set(t["tabel_3_3"].index) and "Penelitian ini (S1)" in set(t["tabel_3_4"].sumber)
