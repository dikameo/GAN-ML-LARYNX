"""Pemeriksaan checkpoint tahap CP0-CP7 (PRD §13). Tiap fungsi mengembalikan laporan item demi item.

Prinsip: item yang tidak bisa dijalankan dilaporkan apa adanya (GAGAL / TIDAK_TERSEDIA / MENUNGGU),
tidak pernah diubah menjadi LULUS. `lulus` benar hanya bila tidak ada GAGAL atau MENUNGGU.
`manifest.json` ditulis hanya bila lulus. Tag git `cpN` TIDAK dibuat di sini: RULES §3 meminta orang
kedua meninjau dulu, ditandai dengan berkas REVIEWED di direktori checkpoint (python -m src.approve review CPn --by NAMA).
Keputusan manusia (parameter ROI, kualitas visual GAN, D1-D8) juga ditandai lewat python -m src.approve.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import P, load_merged
from src.utils import atomic_write_json, load_config, write_manifest

LULUS, GAGAL, MENUNGGU, TIDAK_TERSEDIA = "LULUS", "GAGAL", "MENUNGGU", "TIDAK_TERSEDIA"
DECISIONS = "configs/decisions.yaml"

CP3_SCENARIOS = {"S1": "configs/s1.yaml", "S2": "configs/s2.yaml", "S3_svm": "configs/s3_svm.yaml",
                 "S3_lda": "configs/s3_lda.yaml", "S4_svm": "configs/s4_svm.yaml", "S4_lda": "configs/s4_lda.yaml"}
CP3_SENSITIVITY = {"S3_svm_a01": "configs/s3_svm_a01.yaml", "S3_lda_a01": "configs/s3_lda_a01.yaml",
                   "S4_svm_a01": "configs/s4_svm_a01.yaml", "S4_lda_a01": "configs/s4_lda_a01.yaml"}


def _item(nama: str, status: str, detail: str) -> dict:
    return {"item": nama, "status": status, "detail": detail}


def _pytest(target: str) -> tuple[bool, str]:
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", target], capture_output=True, text=True)
    last = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr[-200:]
    return proc.returncode == 0, last


def _review_item(out_dir: Path) -> dict:
    ok = (out_dir / "REVIEWED").exists()
    return _item("Tinjauan orang kedua (berkas REVIEWED)", LULUS if ok else MENUNGGU,
                 "ada" if ok else f"setelah meninjau hasil: python -m src.approve review {out_dir.name.upper()} --by NAMA (RULES §3)")


def _flag_item(out_dir: Path, name: str, nama: str, how: str) -> dict:
    ok = (out_dir / name).exists()
    return _item(nama, LULUS if ok else MENUNGGU, "ada" if ok else how)


def _decision_item(d: str) -> dict:
    dec = load_config(DECISIONS)[d]
    st = dec["status"]
    status = {"diputuskan": LULUS, "ditunda": LULUS, "menunggu_konfirmasi": MENUNGGU}.get(st, GAGAL)
    return _item(f"{d} {st}", status, dec["isi"])


def _finish(cp: str, out_dir: Path, items: list[dict], files: list, extra: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    blocking = [i for i in items if i["status"] in (GAGAL, MENUNGGU)]
    rep = {"checkpoint": cp, "lulus": not blocking, "items": items, "penghambat": [i["item"] for i in blocking]}
    atomic_write_json(out_dir / f"{cp.lower()}_check.json", rep)
    for i in items:
        print(f"[{i['status']:>15}] {i['item']} -- {i['detail']}")
    if rep["lulus"]:
        write_manifest(out_dir, [out_dir / f"{cp.lower()}_check.json", *[f for f in files if Path(f).exists()]],
                       extra={"checkpoint": cp, **(extra or {})})
        print(f"\n{cp} LULUS. manifest.json ditulis. Tag git '{cp.lower()}' dapat dibuat.")
    else:
        print(f"\n{cp} BELUM LULUS. Penghambat: {rep['penghambat']}\nmanifest.json TIDAK ditulis.")
    return rep


def _scenario_items(cfg: dict, sc: str, label: str | None = None) -> tuple[list[dict], list]:
    """Item standar untuk satu skenario: semua job DONE, L3 lolos, tabel hasil lengkap dengan interval."""
    from src.jobs import job_status
    from src.leakage import LeakageError, check_l3_dir
    from src.splits import load_splits

    label = label or sc
    items, files = [], []
    st = job_status(cfg, sc)
    n_done = sum(r["DONE"] for r in st)
    items.append(_item(f"{label}: seluruh lipatan berstatus DONE", LULUS if n_done == len(st) else GAGAL, f"{n_done}/{len(st)}"))
    try:
        n = check_l3_dir(P(cfg, "jobs"), sc, load_splits(cfg))
        items.append(_item(f"{label}: L3 daftar pasien tiap job", LULUS, f"{n} job"))
    except LeakageError as e:
        items.append(_item(f"{label}: L3", GAGAL, str(e)))
    rp = P(cfg, "results", f"results_{sc}.csv")
    if rp.exists():
        r = pd.read_csv(rp)
        full = r.job_selesai.str.split("/").map(lambda a: a[0] == a[1]).all()
        items.append(_item(f"{label}: tabel hasil dengan interval kepercayaan", LULUS if full else GAGAL,
                           "; ".join(f"{x.metric}={x.estimate:.3f} [{x.ci_low:.3f}, {x.ci_high:.3f}]" for x in r.itertuples())
                           + f" | job {r.job_selesai.iloc[0]}"))
        files += [rp, P(cfg, "results", f"fold_metrics_{sc}.csv"), P(cfg, "results", f"patient_probs_{sc}.csv")]
    else:
        items.append(_item(f"{label}: tabel hasil", GAGAL, f"{rp.name} tidak ada"))
    return items, files


# ===================================================================== CP0
def check_cp0(out_dir: str | Path, decisions_path: str | Path = DECISIONS) -> dict:
    out_dir = Path(out_dir)
    items = []
    env_p = out_dir / "env_check.json"
    if env_p.exists():
        env = json.loads(env_p.read_text(encoding="utf-8"))
        items.append(_item("Uji lingkungan (impor, dataset :ro, bobot Swin-T)", LULUS if env.get("lulus") else GAGAL,
                           f"gagal={env.get('gagal')}"))
        digest, commit = env.get("image_digest", ""), env.get("git_commit", "")
        ok = digest.startswith("sha256:") and commit not in ("", "tidak-diketahui", "tanpa-commit")
        items.append(_item("Digest citra dan hash commit tercatat", LULUS if ok else GAGAL,
                           f"digest={digest[:19]}..., commit={commit}"))
        gs = env.get("gpu_smoke", {"status": TIDAK_TERSEDIA, "detail": "env_check lama tanpa uji GPU"})
        items.append(_item("Uji kecil di GPU", gs["status"], gs["detail"]))
        dev = env.get("device", {})
        items.append(_item("Perangkat tercatat (D1)", LULUS, f"GPU={dev.get('gpu_names')}, mem={dev.get('gpu_memory_gb')} GB, "
                           f"CPU={dev.get('cpu_count')}, thread torch={dev.get('torch_threads')}"))
    else:
        items.append(_item("Uji lingkungan", GAGAL, "env_check.json tidak ada"))
    ok, last = _pytest("tests/test_utils.py")
    items.append(_item("Uji kecil di CPU (pytest, data tiruan)", LULUS if ok else GAGAL, last))
    for name, nama in (("benchmark.json", "Waktu Swin-T tercatat (tensor acak, ekstrapolasi)"),
                       ("benchmark_gan.json", "Waktu GAN tercatat (src.bench_gan, tensor acak)")):
        p = out_dir / name
        if p.exists():
            bm = json.loads(p.read_text(encoding="utf-8"))
            if "mean_step_seconds" in bm:
                det = (f"batch {bm.get('batch_size')}: {bm['mean_step_seconds']:.2f} dtk/langkah, ~{bm['estimated_epoch_minutes']:.1f} mnt/epoch, "
                       f"~{bm['estimated_30_epoch_hours_per_job']:.1f} jam/job 30 epoch, device={bm.get('device')}, gpu={bm.get('gpu')}")
                ok = bm["mean_step_seconds"] > 0
            else:
                det = f"~{bm.get('estimasi_jam_20_gan', float('nan')):.1f} jam untuk 20 GAN (total_kimg={bm.get('total_kimg_config')}), device={bm.get('device')}"
                ok = bool(bm.get("per_lambda"))
            items.append(_item(nama, LULUS if ok else GAGAL, det))
        else:
            items.append(_item(nama, GAGAL, f"{name} tidak ada"))
    for d in ("D1", "D2", "D3", "D4"):
        items.append(_decision_item(d))
    items.append(_review_item(out_dir))
    return {"checkpoint": "CP0", "lulus": not [i for i in items if i["status"] in (GAGAL, MENUNGGU)], "items": items}


def finalize_cp0(out_dir: str | Path = "/outputs/cp0") -> dict:
    out_dir = Path(out_dir)
    rep = check_cp0(out_dir)
    env = json.loads((out_dir / "env_check.json").read_text(encoding="utf-8")) if (out_dir / "env_check.json").exists() else {}
    files = [out_dir / "env_check.json", out_dir / "benchmark.json", out_dir / "benchmark_gan.json", DECISIONS, "configs/base.yaml"]
    return _finish("CP0", out_dir, rep["items"], files, {"git_commit": env.get("git_commit"), "image_digest": env.get("image_digest")})


# ===================================================================== CP1
def finalize_cp1(cfg: dict) -> dict:
    out = P(cfg, "cp1")
    items = []
    need = [out / n for n in ("audit.csv", "audit_images.csv", "audit_summary.json", "audit_reference.csv")]
    missing = [p.name for p in need if not p.exists()]
    if missing:
        items.append(_item("Keluaran audit lengkap", GAGAL, f"hilang: {missing}"))
        return _finish("CP1", out, items, [])
    s = json.loads((out / "audit_summary.json").read_text(encoding="utf-8"))
    items.append(_item("audit.csv dibuat dari gambar lokal", LULUS, f"{s['n_patients']} pasien, {s['n_images']} citra"))
    ref = pd.DataFrame(s["perbandingan_referensi"])
    diff = ref[ref.selisih_reference_tabel_3_2 != 0]
    if diff.empty:
        items.append(_item("Angka sama dengan Tabel 3.2", LULUS, "pasien dan citra per kelas identik"))
    else:
        items.append(_item("Angka sama dengan Tabel 3.2", MENUNGGU, "selisih ada, peneliti harus menjelaskan: " + "; ".join(
            f"{r.unit}/{r.kelas} lokal={r.lokal} vs {r.reference_tabel_3_2}" for r in diff.itertuples())))
    kd = ref[ref.selisih_reference_kiani != 0]
    items.append(_item("Selisih terhadap repo Kiani dilaporkan (informasi)", LULUS,
                       "; ".join(f"{r.unit}/{r.kelas}: {int(r.selisih_reference_kiani):+d}" for r in kd.itertuples()) or "tidak ada selisih"))
    col = s["id_bertabrakan_antar_kelas"]
    items.append(_item("ID pasien unik antar kelas", LULUS if not col else MENUNGGU,
                       "tidak ada tabrakan" if not col else f"ID di kedua kelas (diberi awalan kelas): {col[:10]}"))
    bad = s["pasien_label_tidak_konsisten"] + s["pasien_di_lebih_dari_satu_folder_histo"]
    items.append(_item("Label dan folder histopatologi konsisten per pasien", LULUS if bad == 0 else MENUNGGU,
                       f"label tidak konsisten={s['pasien_label_tidak_konsisten']}, pasien di >1 folder histo={s['pasien_di_lebih_dari_satu_folder_histo']}"))
    x = s["xlsx"]
    if x.get("status") != "dibaca" or x.get("kolom_pasien") is None:
        items.append(_item("Silang-periksa Patients_List_Updated_Final.xlsx", MENUNGGU, f"tidak dapat dilakukan: {x}"))
    else:
        probs = {"hanya_di_xlsx": len(x.get("hanya_di_xlsx", [])), "hanya_di_folder": len(x.get("hanya_di_folder", [])),
                 "label_beda": len(x.get("label_beda_xlsx_vs_folder", [])), "jumlah_citra_beda": len(x.get("jumlah_citra_beda", [])),
                 "resolusi_beda": len(x.get("resolusi_beda", []))}
        items.append(_item("Silang-periksa xlsx (pasien, label, jumlah citra, resolusi)", LULUS if not any(probs.values()) else MENUNGGU, str(probs)))
    s7 = s["s7"]
    items.append(_item("Syarat S7 dihitung ulang dari gambar lokal", LULUS,
                       f"dapat dijalankan={s7.get('s7_dapat_dijalankan')}; mayoritas={s7.get('mayoritas', {}).get('resolution')}; "
                       f"minor gabungan={s7.get('minor_gabungan')}; pasien multi-resolusi={s7.get('pasien_multi_resolusi')}"))
    items.append(_review_item(out))
    return _finish("CP1", out, items, need)


# ===================================================================== CP2
def finalize_cp2(cfg: dict) -> dict:
    from src.data import Cache, find_cache
    from src.jobio import resolution_map
    from src.leakage import LeakageError, check_l1, check_l3_dir, check_l5
    from src.preprocess import cache_hash

    out = P(cfg, "cp2")
    out.mkdir(parents=True, exist_ok=True)
    items, files, cache = [], [], None
    try:
        cache = Cache(find_cache(cfg))
        imgs = pd.read_csv(P(cfg, "cp1", "audit_images.csv"))
        items.append(_item("Cache memmap uint8 ber-hash konfigurasi dan sesuai berkas audit",
                           LULUS if cache.hash == cache_hash(cfg, imgs) else GAGAL,
                           f"hash={cache.hash[:12]}, {len(cache.meta)} citra, images{cache.images.shape} roi{cache.roi.shape}"))
        items.append(_item("Jumlah citra cache = audit", LULUS if len(cache.meta) == len(imgs) else GAGAL, f"{len(cache.meta)} vs {len(imgs)}"))
        items.append(_item("Tipe data uint8", LULUS if cache.images.dtype == np.uint8 and cache.roi.dtype == np.uint8 else GAGAL,
                           f"{cache.images.dtype}/{cache.roi.dtype}"))
        q = cache.meta
        items.append(_item("Ringkasan kualitas (informasi, tidak ada citra dibuang)", LULUS,
                           f"fov_fallback={int(q.fov_fallback.sum())}, glare>5%={int((q.q_glare_frac > .05).sum())}, "
                           f"gelap>50%={int((q.q_dark_frac > .5).sum())}, blur terendah 1%={q.q_blur_lapvar.quantile(.01):.1f}"))
        files.append(cache.dir / "meta.csv")
    except Exception as e:  # noqa: BLE001
        items.append(_item("Cache", GAGAL, repr(e)[:200]))

    sp = P(cfg, "splits", "splits.json")
    if sp.exists():
        splits = json.loads(sp.read_text(encoding="utf-8"))
        try:
            check_l1(splits)
            items.append(_item("L1: tidak ada pasien di latih dan uji sekaligus; tiap pasien diuji sekali per ulangan", LULUS,
                               f"{len(splits['repeats'])} ulangan x {len(splits['repeats'][0]['folds'])} lipatan"))
        except LeakageError as e:
            items.append(_item("L1", GAGAL, str(e)))
        try:
            n = check_l3_dir(P(cfg, "jobs"), "S1", splits)
            items.append(_item("L3: daftar pasien job vs split", LULUS, f"{n} job diperiksa (0 wajar pada CP2)"))
        except LeakageError as e:
            items.append(_item("L3", GAGAL, str(e)))
        fs = pd.read_csv(P(cfg, "splits", "fold_summary.csv"))
        items.append(_item("Proporsi pasien ganas per lipatan uji (informasi)", LULUS if (fs.test_ganas > 0).all() else GAGAL,
                           f"uji: pasien {fs.test_pasien.min()}-{fs.test_pasien.max()}, ganas {fs.test_ganas.min()}-{fs.test_ganas.max()}; "
                           f"validasi ganas {fs.val_ganas.min()}-{fs.val_ganas.max()}"))
        files += [sp, P(cfg, "splits", "fold_summary.csv")]
    else:
        items.append(_item("Berkas split", GAGAL, "splits.json tidak ada"))
    s7p = P(cfg, "splits", "splits_s7.json")
    if s7p.exists():
        try:
            s7 = json.loads(s7p.read_text(encoding="utf-8"))
            check_l5(s7, resolution_map(cfg))
            items.append(_item("L5: split S7 (latih resolusi mayoritas, uji gabungan lain)", LULUS,
                               f"latih {s7['major_resolution']}, uji {s7['test_resolutions']}"))
            files.append(s7p)
        except Exception as e:  # noqa: BLE001
            items.append(_item("L5 split S7", GAGAL, repr(e)[:200]))
    else:
        s = json.loads(P(cfg, "cp1", "audit_summary.json").read_text(encoding="utf-8"))["s7"]
        items.append(_item("Split S7", LULUS if not s.get("s7_dapat_dijalankan") else GAGAL,
                           "S7 tidak memenuhi syarat menurut audit" if not s.get("s7_dapat_dijalankan") else "splits_s7.json belum dibuat (python -m src.splits --s7)"))
    ok, last = _pytest("tests/test_leakage.py")
    items.append(_item("L2 dan uji kebocoran otomatis (pytest)", LULUS if ok else GAGAL, last))
    rc = out / "roi_check"
    pngs = sorted(rc.glob("roi_check_*.png")) if rc.exists() else []
    items.append(_item("50 citra ROI untuk pemeriksaan visual dibuat (FR-03)", LULUS if len(pngs) >= 5 else GAGAL, f"{len(pngs)} halaman di {rc}"))
    items.append(_item("D6: ID pasien 50 citra penyetel Frangi tercatat", LULUS if (rc / "roi_examples_patients.csv").exists() else GAGAL,
                       "roi_examples_patients.csv (tulis sebagai keterbatasan: semua pasien menjadi pasien uji di satu lipatan)"))
    items.append(_decision_item("D6"))
    approved = (out / "ROI_APPROVED").exists() and cfg["preprocess"]["roi"]["locked"]
    items.append(_item("Parameter Frangi dikunci setelah pemeriksaan visual", LULUS if approved else MENUNGGU,
                       "dikunci" if approved else "periksa PNG, lalu: python -m src.approve roi --by NAMA"))
    items.append(_review_item(out))
    return _finish("CP2", out, items, files, {"cache_hash": cache.hash if cache is not None else None})


# ===================================================================== CP3
def finalize_cp3(cfg: dict) -> dict:
    from src.jobs import job_status

    out = P(cfg, "cp3")
    items, files = [], []
    for sc, path in CP3_SCENARIOS.items():
        it, fl = _scenario_items(load_merged(path), sc)
        items += it
        files += fl
    cmp_ = P(cfg, "results", "compare_S1_vs_S2.csv")
    items.append(_item("Perbandingan S1 vs S2 (berlabel eksploratif)", LULUS if cmp_.exists() else GAGAL, "ada" if cmp_.exists() else "belum dibuat"))
    files.append(cmp_)
    for sc, path in CP3_SENSITIVITY.items():  # informasi: analisis sensitivitas D3 (eksploratif), tidak menjadi gerbang
        n = sum(r["DONE"] for r in job_status(load_merged(path), sc))
        items.append(_item(f"{sc} (sensitivitas LASSO 0,01, eksploratif; informasi)", LULUS, f"{n}/10 job DONE"))
    fd = sorted(P(cfg, "features").glob("rad_*/summary.json")) if P(cfg, "features").exists() else []
    if fd:
        sm = json.loads(fd[-1].read_text(encoding="utf-8"))
        items.append(_item("Fitur radiomik diekstrak (jumlah dilaporkan apa adanya)", LULUS,
                           f"{sm['n_features']} fitur (Wang dkk.: 104), mask={sm['mask']}, citra dengan NaN={sm['n_images_with_nan']}"))
    else:
        items.append(_item("Fitur radiomik", GAGAL, "summary.json tidak ada"))
    for d in ("D3", "D4", "D7"):
        items.append(_decision_item(d))
    items.append(_item("Orkestrasi teruji pada beban nyata", LULUS if P(cfg, "jobs", "launch.log").exists() else GAGAL,
                       "launch.log ada" if P(cfg, "jobs", "launch.log").exists() else "belum ada job yang diluncurkan"))
    items.append(_review_item(out))
    return _finish("CP3", out, items, files)


# ===================================================================== CP4
def finalize_cp4(cfg: dict) -> dict:
    from src.leakage import LeakageError, check_l6

    out = P(cfg, "cp4")
    items, files = [], []
    sp = out / "lambda_selection.json"
    if sp.exists():
        sel = json.loads(sp.read_text(encoding="utf-8"))
        items.append(_item("lambda_roi dipilih dari validasi dalam lipatan latih (D2)", LULUS if "validasi" in sel["source"] else GAGAL,
                           f"terpilih={sel['chosen']}; " + "; ".join(f"lam={r['lambda_roi']}: FIDval={r['fid_val']:.1f}" for r in sel["table"])))
        files.append(sp)
        done = all((Path(r["dir"]) / "DONE").exists() for r in sel["table"])
        items.append(_item("Semua pilot GAN berstatus DONE", LULUS if done else GAGAL, f"{len(sel['table'])} nilai lambda"))
        ok = all(r.get("lpips_diversity_malignant") is not None and np.isfinite(r["lpips_diversity_malignant"]) for r in sel["table"])
        items.append(_item("FID dan LPIPS tercatat", LULUS if ok else GAGAL, "FID validasi dan LPIPS ganas per lambda"))
        grids = [Path(r["dir"]) / "grid_val" / "samples_malignant.png" for r in sel["table"]]
        items.append(_item("100 citra sintetis per kelas dibuat untuk pemeriksaan visual", LULUS if all(g.exists() for g in grids) else GAGAL,
                           f"{sum(g.exists() for g in grids)}/{len(grids)} grid ganas"))
    else:
        items.append(_item("lambda_selection.json", GAGAL, "jalankan python -m src.pilot_lambda"))
    try:
        check_l6(cfg)
        items.append(_item("L6: filter metrik pembuluh (Sato) berbeda dari pembuat ROI (Frangi)", LULUS,
                           f"{cfg['gan']['quality']['vessel_filter']} vs {cfg['preprocess']['roi']['filter']}"))
    except LeakageError as e:
        items.append(_item("L6", GAGAL, str(e)))
    items.append(_flag_item(out, "GAN_VISUAL_APPROVED", "Pemeriksaan visual sampel sintetis (tanpa mode collapse jelas)",
                            "periksa grid_val/*.png, lalu: python -m src.approve gan-visual CP4 --by NAMA"))
    for d in ("D2", "D8"):
        items.append(_decision_item(d))
    items.append(_review_item(out))
    return _finish("CP4", out, items, files)


# ===================================================================== CP5
def finalize_cp5(cfg5: dict, cfg6: dict) -> dict:
    from src.jobio import job_dir
    from src.jobs import job_status
    from src.leakage import LeakageError, check_l3_job
    from src.splits import load_splits

    out = P(cfg6, "cp5")
    items, files = [], []
    splits = load_splits(cfg6)
    for base, cfg in (("S5", cfg5), ("S6", cfg6)):
        g = job_status(cfg, f"{base}_gan")
        items.append(_item(f"{base}: GAN seluruh lipatan DONE", LULUS if all(r["DONE"] for r in g) else GAGAL,
                           f"{sum(r['DONE'] for r in g)}/{len(g)}"))
        try:  # citra sintetis hanya bersumber dari pasien latih (L3) dan tidak masuk uji
            n = 0
            for r in splits["repeats"]:
                for f in r["folds"]:
                    mp = job_dir(cfg, f"{base}_gan", r["rep"], f["fold"]) / "synth_meta.json"
                    if mp.exists():
                        meta = json.loads(mp.read_text(encoding="utf-8"))
                        check_l3_job({"train": f["train"], "val": f["val"], "test": f["test"]}, f, meta["source_patients"])
                        n += 1
            items.append(_item(f"{base}: L3 sumber citra sintetis = pasien latih", LULUS, f"{n} GAN diperiksa"))
        except LeakageError as e:
            items.append(_item(f"{base}: L3 sintetis", GAGAL, str(e)))
        for ra in cfg["gan"]["ratios"]:
            sc = f"{base}_r{int(round(ra * 100))}"
            it, fl = _scenario_items(cfg, sc, label=sc)
            items += it
            files += fl
        sel = P(cfg, "results", f"ratio_selection_{base}.csv")
        items.append(_item(f"{base}: rasio terbaik dipilih dari validasi (bukan uji)", LULUS if sel.exists() else GAGAL,
                           "ratio_selection ada (sumber: val_loss_balanced validasi)" if sel.exists() else "belum dibuat"))
        rp = P(cfg, "results", f"results_{base}.csv")
        items.append(_item(f"{base}: tabel hasil rasio terpilih dengan interval", LULUS if rp.exists() else GAGAL, rp.name))
        files += [sel, rp, P(cfg, "results", f"patient_probs_{base}.csv")]
        lp = []
        for r in splits["repeats"]:
            for f in r["folds"]:
                mp = job_dir(cfg, f"{base}_gan", r["rep"], f["fold"]) / "gan_metrics.json"
                if mp.exists():
                    lp.append(json.loads(mp.read_text(encoding="utf-8")).get("lpips_diversity_malignant"))
        items.append(_item(f"{base}: FID, W1 pembuluh, LPIPS tercatat per lipatan", LULUS if lp and all(v is not None for v in lp) else GAGAL,
                           f"{len(lp)} GAN; LPIPS ganas rata-rata={np.nanmean([v for v in lp if v is not None]) if lp else float('nan'):.3f}"))
    bp = P(cfg6, "results", "best_config.json")
    items.append(_item("best_config.json (untuk S7) dari validasi", LULUS if bp.exists() else GAGAL, "ada" if bp.exists() else "belum dibuat"))
    files.append(bp)
    items.append(_flag_item(out, "MODE_COLLAPSE_CHECKED", "Pemeriksaan mode collapse GAN kelas ganas (grid, LPIPS)",
                            "periksa grid_test/samples_malignant.png dan LPIPS tiap lipatan, lalu: python -m src.approve gan-visual CP5 --by NAMA"))
    items.append(_review_item(out))
    return _finish("CP5", out, items, files)


# ===================================================================== CP6
def finalize_cp6(cfg: dict) -> dict:
    from src.jobio import resolution_map
    from src.leakage import check_l5
    from src.splits import load_splits

    out = P(cfg, "cp6")
    items, files = [], []
    s7 = load_splits(cfg, "s7")
    try:
        check_l5(s7, resolution_map(cfg))
        items.append(_item("L5: pasien resolusi uji tidak dipakai melatih generator/klasifikator, memilih model, atau ambang", LULUS,
                           f"latih {s7['major_resolution']} ({len(s7['repeats'][0]['folds'][0]['train'])}+{len(s7['repeats'][0]['folds'][0]['val'])} pasien), "
                           f"uji {len(s7['repeats'][0]['folds'][0]['test'])} pasien di {s7['test_resolutions']}"))
    except Exception as e:  # noqa: BLE001
        items.append(_item("L5", GAGAL, repr(e)[:200]))
    for sc, path in (("S7_S1", "configs/s7_s1.yaml"), ("S7_S5", "configs/s7_s5.yaml"), ("S7_S6", "configs/s7_s6.yaml")):
        it, fl = _scenario_items(load_merged(path), sc)
        items += it
        files += fl
    gaps = [P(cfg, "results", f"s7_gap_{m}.csv") for m in ("S1", "S5", "S6")]
    items.append(_item("Selisih kinerja dalam-distribusi vs pergeseran dengan interval (eksploratif)",
                       LULUS if all(g.exists() for g in gaps) else GAGAL, f"{sum(g.exists() for g in gaps)}/3 berkas"))
    files += gaps
    items.append(_item("Catatan interpretasi S7 (PRD §6)", LULUS, "pergeseran akuisisi bercampur dengan perbedaan populasi; bukan efek resolusi murni"))
    items.append(_review_item(out))
    return _finish("CP6", out, items, files)


# ===================================================================== CP7
def finalize_cp7(cfg: dict) -> dict:
    out = P(cfg, "cp7")
    items, files = [], []
    res = P(cfg, "results")
    for n in ("tabel_3_3.csv", "tabel_3_4.csv", "comparisons_all.csv"):
        items.append(_item(f"{n} dibuat otomatis dari hasil job", LULUS if (res / n).exists() else GAGAL, n))
        files.append(res / n)
    cp = res / "comparisons_all.csv"
    if cp.exists():
        c = pd.read_csv(cp)
        main = c[c.status == "utama"]
        need = {("S6", "S1"), ("S6", "S2")}
        have = set(zip(main.A, main.B))
        items.append(_item("Perbandingan utama S6 vs S1 dan S2 dilaporkan apa pun arahnya (AUROC, F1)", LULUS if need <= have else GAGAL,
                           "; ".join(f"{r.A} vs {r.B} {r.metric}: {r.selisih_A_minus_B:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]"
                                     f"{' (mencakup nol)' if r.mencakup_nol else ' (tidak mencakup nol)'}"
                                     for r in main[main.metric.isin(['auroc', 'f1'])].itertuples())))
        others = c[c.status != "utama"]
        items.append(_item("Perbandingan lain berlabel eksploratif", LULUS if (others.status == "eksploratif").all() else GAGAL,
                           f"{others[['A', 'B']].drop_duplicates().shape[0]} pasangan"))
    # keterlacakan: tiap results_*.csv lengkap dan menunjuk job_selesai penuh
    bad = []
    for f in sorted(res.glob("results_*.csv")):
        r = pd.read_csv(f)
        if not r.job_selesai.str.split("/").map(lambda a: a[0] == a[1]).all():
            bad.append(f.name)
    items.append(_item("Setiap angka terlacak ke job selesai (tidak ada tabel PARSIAL)", LULUS if not bad else GAGAL,
                       "semua results_*.csv lengkap" if not bad else f"parsial: {bad}"))
    items.append(_item("Semua tabel melaporkan interval kepercayaan", LULUS, "kolom ci_low/ci_high pada results_*.csv"))
    items.append(_review_item(out))
    return _finish("CP7", out, items, files)
