"""Membangun notebook 00-07 (satu per checkpoint). Notebook hanya memanggil fungsi di src/.
Jalankan dari akar repositori: python scripts/make_notebooks.py"""
import nbformat as nbf
from nbformat.v4 import new_code_cell as C, new_markdown_cell as M, new_notebook

SETUP = """import json, os, sys
from pathlib import Path
import pandas as pd
from IPython.display import display, Image

def _root():
    p = Path.cwd().resolve()
    for q in [p, *p.parents]:
        if (q / 'configs' / 'base.yaml').exists():
            return q
    raise FileNotFoundError('configs/base.yaml tidak ditemukan')
ROOT = _root(); os.chdir(ROOT); sys.path.insert(0, str(ROOT))

from src.config import load_merged, P
from src.launch import run_module

CFG = 'configs/base.yaml'
cfg = load_merged(CFG)
DEVICES = None      # mis. [0, 1]: satu rantai job per GPU lewat CUDA_VISIBLE_DEVICES (PRD §8)
REPS, FOLDS = None, None   # None = semua. Membatasi JOB YANG DILUNCURKAN saja, bukan memilih hasil; gerbang butuh semuanya.
"""


def save(name, cells):
    nb = new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    nbf.write(nb, f"notebooks/{name}.ipynb")
    print("ditulis", name)


# ------------------------------------------------------------------ 00
save("00_setup", [
    M("# 00 – Setup dan benchmark (CP0)\n\nNotebook hanya memanggil fungsi di `src/`. Pekerjaan berat dijalankan sebagai skrip terpisah.\n\n"
      "**Kejujuran hasil:** benchmark memakai tensor acak (cache data belum ada sampai CP2), jadi waktu epoch adalah ekstrapolasi, "
      "bukan epoch nyata. Bila tidak ada GPU, uji GPU dilaporkan `TIDAK_TERSEDIA`, bukan lulus. Benchmark dari mesin lain "
      "tidak dipakai ulang: berkas lama dipindah ke `*.bak`."),
    C(SETUP + """
import platform
OUT = P(cfg, 'cp0'); OUT.mkdir(parents=True, exist_ok=True)

def fresh(p):
    \"\"\"True bila berkas ada DAN berasal dari mesin ini; berkas dari mesin lain dipindah ke .bak.\"\"\"
    if not p.exists():
        return False
    host = json.loads(p.read_text()).get('host')
    if host == platform.node():
        return True
    p.rename(p.with_suffix('.json.bak'))
    return False
print('keluaran CP0 ->', OUT, '| host:', platform.node())"""),
    M("## 1. Uji lingkungan (impor, dataset baca-saja, bobot Swin-T, uji GPU)"),
    C("rc = run_module('src.env_check', ['--config', CFG, '--out', str(OUT / 'env_check.json')])\nprint('kode keluar:', rc)"),
    M("## 2. Benchmark Swin-T (dilewati bila `benchmark.json` dari mesin ini sudah ada)"),
    C("rc = run_module('src.benchmark', ['--config', CFG, '--out', str(OUT / 'benchmark.json')],\n"
      "                skip_if_exists=OUT / 'benchmark.json' if fresh(OUT / 'benchmark.json') else None)\nprint('kode keluar:', rc)"),
    C("bm = json.loads((OUT / 'benchmark.json').read_text())\n"
      "for k in ['device', 'gpu', 'amp', 'data', 'batch_size', 'torch_threads', 'cpu_count', 'mean_step_seconds', 'images_per_second',\n"
      "          'assumed_train_images', 'estimated_epoch_minutes', 'estimated_30_epoch_hours_per_job', 'peak_gpu_gb', 'ram_rss_gb']:\n"
      "    print(f'{k}: {bm.get(k)}')\n"
      "print('\\nBatas atas S1 (10 lipatan x 30 epoch, tanpa henti dini):', round(bm['estimated_30_epoch_hours_per_job'] * 10, 1), 'jam / jumlah GPU')"),
    M("## 3. Benchmark GAN (`src.bench_gan`): dasar menetapkan `gan.total_kimg` dan menghitung waktu total (PRD §8 Fase 0)"),
    C("rc = run_module('src.bench_gan', ['--config', 'configs/s6.yaml', '--out', str(OUT / 'benchmark_gan.json'), '--iters', '5', '--warmup', '2'],\n"
      "                skip_if_exists=OUT / 'benchmark_gan.json' if fresh(OUT / 'benchmark_gan.json') else None)\nprint('kode keluar:', rc)"),
    M("## 4. Pemeriksaan checkpoint CP0 dan `manifest.json` (sel terakhir)"),
    C("from src.checkpoint import finalize_cp0\nreport = finalize_cp0(OUT)\nassert report['checkpoint'] == 'CP0'"),
])

# ------------------------------------------------------------------ 01
save("01_audit", [
    M("# 01 – Audit data (CP1, FR-01)\n\nAngka dihitung dari **gambar lokal**, lalu dibandingkan dengan Tabel 3.2 proposal "
      "(150 jinak / 60 ganas pasien; 7.657 / 3.487 citra). Selisih dilaporkan apa adanya. Dataset hanya dibaca."),
    C(SETUP),
    M("## 1. Jalankan audit (dilewati bila `audit.csv` sudah ada)"),
    C("rc = run_module('src.audit', ['--config', CFG], skip_if_exists=P(cfg, 'cp1', 'audit.csv'))\nprint('kode keluar:', rc)"),
    M("## 2. Perbandingan dengan Tabel 3.2 dan repo Kiani"),
    C("s = json.loads(P(cfg, 'cp1', 'audit_summary.json').read_text())\n"
      "display(pd.DataFrame(s['perbandingan_referensi']))\n"
      "print('ID pasien bertabrakan antar kelas:', s['id_bertabrakan_antar_kelas'])\n"
      "print('Pasien label tidak konsisten:', s['pasien_label_tidak_konsisten'])\n"
      "print('Pasien di >1 folder histopatologi:', s['pasien_di_lebih_dari_satu_folder_histo'])"),
    M("## 3. Silang-periksa dengan `Patients_List_Updated_Final.xlsx`"),
    C("for k, v in s['xlsx'].items():\n    print(k, ':', (v if not isinstance(v, list) or len(v) <= 15 else v[:15] + ['...']))"),
    M("## 4. Resolusi dan syarat S7 (dihitung ulang dari gambar lokal)"),
    C("display(pd.DataFrame(s['s7']['per_resolusi']))\nprint('mayoritas:', s['s7'].get('mayoritas'))\n"
      "print('minor gabungan:', s['s7'].get('minor_gabungan'))\nprint('pasien dengan >1 resolusi:', s['s7']['pasien_multi_resolusi'])\n"
      "print('S7 dapat dijalankan:', s['s7'].get('s7_dapat_dijalankan'))"),
    M("## 5. Pemeriksaan checkpoint CP1 dan `manifest.json` (sel terakhir)"),
    C("from src.checkpoint import finalize_cp1\nreport = finalize_cp1(cfg)\nassert report['checkpoint'] == 'CP1'"),
])

# ------------------------------------------------------------------ 02
save("02_praproses_split", [
    M("# 02 – Praproses, ROI, dan split (CP2, FR-02 sampai FR-04)\n\nLima langkah praproses → cache memmap uint8 (256×256, ber-hash) + peta ROI Frangi. "
      "Split per pasien 5 lipatan × 2 ulangan, 15% pasien latih sebagai validasi; split S7. Uji kebocoran L1 sampai L3 dan L5.\n\n"
      "Normalisasi ImageNet tetap (diterapkan saat dibaca), tidak ada citra dibuang, parameter Frangi dikunci setelah pemeriksaan visual."),
    C(SETUP),
    M("## 1. Praproses dan cache (idempoten; dilewati bila cache ber-hash sama sudah DONE)"),
    C("rc = run_module('src.preprocess', ['--config', CFG])\nprint('kode keluar:', rc)"),
    M("## 2. Split per pasien (dan S7 bila audit menyatakan memenuhi syarat)"),
    C("s7_ok = json.loads(P(cfg, 'cp1', 'audit_summary.json').read_text())['s7'].get('s7_dapat_dijalankan')\n"
      "rc = run_module('src.splits', ['--config', CFG] + (['--s7'] if s7_ok else []))\nprint('S7 dapat dijalankan:', s7_ok, '| kode keluar:', rc)\n"
      "display(pd.read_csv(P(cfg, 'splits', 'fold_summary.csv')))"),
    M("## 3. Pemeriksaan visual ROI (FR-03)\n\nPeriksa gambar. Bila ROI wajar: `python -m src.approve roi --by NAMA` (mengunci parameter dan menandai). "
      "Bila tidak, ubah parameter di `configs/base.yaml` lalu ulangi; **jangan** memilih parameter berdasarkan hasil klasifikasi."),
    C("d = P(cfg, 'cp2', 'roi_check')\nfor f in sorted(d.glob('roi_check_*.png'))[:2]:\n    display(Image(filename=str(f)))\n"
      "print('Semua halaman ada di', d)\ndisplay(pd.read_csv(d / 'roi_examples_patients.csv').head(10))"),
    M("## 4. Pemeriksaan checkpoint CP2 dan `manifest.json` (sel terakhir)"),
    C("from src.checkpoint import finalize_cp2\nreport = finalize_cp2(cfg)\nassert report['checkpoint'] == 'CP2'"),
])

# ------------------------------------------------------------------ 03
save("03_baseline", [
    M("# 03 – Baseline dan pembanding S1 sampai S4 (CP3)\n\n"
      "- **S1:** Swin-T pralatih, weighted sampling, AdamW 1e-4, wd 0,05, batch 32, maks 30 epoch, henti dini sabar 5.\n"
      "- **S2:** embedding Swin beku (sekali), SMOTE di dalam lipatan latih, kepala linear.\n"
      "- **S3:** fitur PyRadiomics → MinMax → LASSO → SMOTE → SVM linear / LDA.\n"
      "- **S4:** S3 + CTGAN pada ruang fitur (50% data latih). Analisis sensitivitas LASSO 0,01 berlabel eksploratif (D3).\n\n"
      "Pelatihan dijalankan sebagai skrip yang diluncurkan dari sel. Tiap job punya `last.pt` (atomik) dan `DONE`; job selesai dilewati. "
      "**Jangan ganti seed atau buang lipatan karena hasil jelek.**"),
    C(SETUP + """
from src.jobs import launch_scenario, job_status
from src.evaluate import evaluate_scenario, compare_scenarios

MAIN = [('S2', 'configs/s2.yaml', 'src.train_s2'), ('S3_svm', 'configs/s3_svm.yaml', 'src.train_s3'),
        ('S3_lda', 'configs/s3_lda.yaml', 'src.train_s3'), ('S4_svm', 'configs/s4_svm.yaml', 'src.train_s3'),
        ('S4_lda', 'configs/s4_lda.yaml', 'src.train_s3'), ('S1', 'configs/s1.yaml', 'src.train')]   # S1 terakhir (paling lama)
SENS = [('S3_svm_a01', 'configs/s3_svm_a01.yaml', 'src.train_s3'), ('S3_lda_a01', 'configs/s3_lda_a01.yaml', 'src.train_s3'),
        ('S4_svm_a01', 'configs/s4_svm_a01.yaml', 'src.train_s3'), ('S4_lda_a01', 'configs/s4_lda_a01.yaml', 'src.train_s3')]
RUN_SENSITIVITY = True       # analisis sensitivitas D3 (eksploratif)
ALLOW_PARTIAL = REPS is not None or FOLDS is not None   # tabel parsial otomatis ditandai PARSIAL dan tidak memenuhi gerbang

bm = json.loads(P(cfg, 'cp0', 'benchmark.json').read_text())
print('Benchmark CP0:', round(bm['estimated_epoch_minutes'], 1), 'menit/epoch ->', round(bm['estimated_30_epoch_hours_per_job'], 1), 'jam per job S1 (batas atas)')"""),
    M("## 1. Fitur untuk S2 dan S3/S4 (dihitung sekali; dapat dilanjutkan)"),
    C("rc = run_module('src.embed', ['--config', 'configs/s2.yaml']); print('embedding, kode keluar:', rc)\n"
      "rc = run_module('src.radiomics_feats', ['--config', 'configs/s3_svm.yaml']); print('radiomics, kode keluar:', rc)"),
    M("## 2. Luncurkan job (10 rep × lipatan per skenario)"),
    C("for sc, path, module in MAIN:\n    print('=== ', sc)\n    launch_scenario(path, module, REPS, FOLDS, DEVICES if module == 'src.train' else None)\n"
      "    display(pd.DataFrame(job_status(load_merged(path), sc)).T)"),
    M("## 3. Analisis sensitivitas LASSO 0,01 (eksploratif, D3)"),
    C("if RUN_SENSITIVITY:\n    for sc, path, module in SENS:\n        print('=== ', sc)\n        launch_scenario(path, module, REPS, FOLDS, None)"),
    M("## 4. Evaluasi tingkat pasien (AUROC, F1, recall, presisi, akurasi) dengan bootstrap persentil bertingkat 2.000 ulangan"),
    C("R = {}\nfor sc, path, _ in MAIN + (SENS if RUN_SENSITIVITY else []):\n    R[sc] = evaluate_scenario(load_merged(path), sc, allow_partial=ALLOW_PARTIAL)\n"
      "display(pd.concat([r['table'].assign(skenario=sc)[['skenario', 'metric', 'estimate', 'ci_low', 'ci_high', 'job_selesai']] for sc, r in R.items()]))"),
    M("## 5. Selisih berpasangan S1 − S2 (**eksploratif**; perbandingan utama PRD adalah S6 lawan S1 dan S2)"),
    C("display(compare_scenarios(cfg, R['S1'], R['S2'], label='eksploratif'))"),
    M("## 6. Pemeriksaan checkpoint CP3 dan `manifest.json` (sel terakhir)"),
    C("from src.checkpoint import finalize_cp3\nreport = finalize_cp3(cfg)\nassert report['checkpoint'] == 'CP3'"),
])

# ------------------------------------------------------------------ 04
save("04_pilot_gan", [
    M("# 04 – Pilot λ_roi dan GAN (CP4, D2, FR-06)\n\nGAN kondisional-kelas bergaya StyleGAN2 dengan D_roi (decisions.yaml D8). "
      "λ_roi dipilih dari **FID pada pasien validasi** di dalam lipatan latih (rep 0, lipatan 0); kriteria ditetapkan di konfigurasi sebelum "
      "eksperimen. Citra uji tidak dipakai. Tetapkan `gan.total_kimg` lebih dulu dari `benchmark_gan.json` (CP0)."),
    C(SETUP),
    C("bg = json.loads(P(cfg, 'cp0', 'benchmark_gan.json').read_text())\n"
      "print('total_kimg di konfigurasi:', cfg['gan']['total_kimg'], '| estimasi jam 20 GAN:', round(bg['estimasi_jam_20_gan'], 1), '(dibagi jumlah GPU)')\n"
      "print('Bila terlalu lama/pendek, ubah gan.total_kimg di configs/base.yaml SEBELUM menjalankan sel berikut.')"),
    M("## 1. Pilot (dapat dilanjutkan; job DONE dilewati)"),
    C("rc = run_module('src.pilot_lambda', ['--config', 'configs/s6.yaml']); print('kode keluar:', rc)"),
    M("## 2. Hasil pilot dan pemeriksaan visual"),
    C("sel = json.loads(P(cfg, 'cp4', 'lambda_selection.json').read_text())\nprint('lambda_roi terpilih:', sel['chosen'], '|', sel['source'])\n"
      "display(pd.DataFrame(sel['table']))\n"
      "for r in sel['table']:\n    g = Path(r['dir']) / 'grid_val' / 'samples_malignant.png'\n    print('lambda =', r['lambda_roi'])\n    display(Image(filename=str(g)))"),
    M("Periksa grid. Bila tidak ada mode collapse yang jelas: `python -m src.approve gan-visual CP4 --by NAMA`."),
    M("## 3. Pemeriksaan checkpoint CP4 dan `manifest.json` (sel terakhir)"),
    C("from src.checkpoint import finalize_cp4\nreport = finalize_cp4(cfg)\nassert report['checkpoint'] == 'CP4'"),
])

# ------------------------------------------------------------------ 05
save("05_s5_s6", [
    M("# 05 – S5 (cGAN tanpa ROI) dan S6 (cGAN berpandu ROI) (CP5, FR-06, FR-07)\n\nSatu GAN per lipatan membangkitkan 100% jumlah citra ganas latih "
      "sekali; subset 25%/50% diambil darinya. Rantai per lipatan: GAN → sintetis → Swin-T ×3 rasio (urutan wajib). "
      "Rasio terbaik dipilih per lipatan dari **validasi**, bukan uji (G2, L4). Semua rasio dilaporkan."),
    C(SETUP + """
from src.jobs import launch_gan_chain
from src.evaluate import evaluate_all_ratios, evaluate_selected, select_best_config, compare_scenarios, load_result

cfg5, cfg6 = load_merged('configs/s5.yaml'), load_merged('configs/s6.yaml')
ALLOW_PARTIAL = REPS is not None or FOLDS is not None"""),
    M("## 1. S5 – 10 rantai (GAN λ=0 → 3 rasio)"),
    C("launch_gan_chain('configs/s5.yaml', reps=REPS, folds=FOLDS, devices=DEVICES)"),
    M("## 2. S6 – 10 rantai (GAN berpandu ROI → 3 rasio). λ_roi dari CP4."),
    C("launch_gan_chain('configs/s6.yaml', reps=REPS, folds=FOLDS, devices=DEVICES)"),
    M("## 3. Evaluasi: semua rasio, dan rasio terpilih per lipatan dari validasi"),
    C("for base, c in (('S5', cfg5), ('S6', cfg6)):\n    display(evaluate_all_ratios(c, base, ALLOW_PARTIAL))\n"
      "    r = evaluate_selected(c, base, ALLOW_PARTIAL)\n    print(base, 'rasio terpilih (validasi):')\n    display(r['table'])\n"
      "    display(pd.read_csv(P(c, 'results', f'ratio_selection_{base}.csv')).head(10))"),
    M("## 4. Konfigurasi terbaik untuk S7 (dari validasi) dan ablation ROI S6 − S5 (**eksploratif**)"),
    C("if not ALLOW_PARTIAL:\n    print(json.dumps(select_best_config(cfg6), indent=2))\n"
      "display(compare_scenarios(cfg6, load_result(cfg6, 'S6'), load_result(cfg5, 'S5')))"),
    M("## 5. Pemeriksaan checkpoint CP5 dan `manifest.json` (sel terakhir)\n\nPeriksa `grid_test/samples_malignant.png` dan LPIPS tiap lipatan untuk mode collapse (`python -m src.approve gan-visual CP5 --by NAMA`)."),
    C("from src.checkpoint import finalize_cp5\nreport = finalize_cp5(cfg5, cfg6)\nassert report['checkpoint'] == 'CP5'"),
])

# ------------------------------------------------------------------ 06
save("06_s7", [
    M("# 06 – S7: ketahanan terhadap pergeseran resolusi (CP6)\n\nLatih (generator dan klasifikator) hanya pada pasien resolusi mayoritas; uji gabungan "
      "resolusi lain (L5). Metode: S1 dan konfigurasi terbaik S5 dan S6 dari validasi. **Catatan interpretasi (PRD §6):** proporsi ganas berbeda "
      "antar resolusi, sehingga ini mengukur pergeseran akuisisi yang bercampur dengan perbedaan populasi, bukan efek resolusi murni."),
    C(SETUP + """
from src.jobs import launch_scenario, launch_gan_chain
from src.evaluate import evaluate_scenario, s7_gap

cfg7 = load_merged('configs/s7_s1.yaml')   # splits_name = s7
s7 = json.loads(P(cfg, 'splits', 'splits_s7.json').read_text())
print('latih:', s7['major_resolution'], '| uji:', s7['test_resolutions'], '| n_repeats:', len(s7['repeats']))"""),
    M("## 1. S7_S1"),
    C("launch_scenario('configs/s7_s1.yaml', 'src.train', REPS, None, DEVICES)"),
    M("## 2. S7_S5 dan S7_S6 (GAN dilatih hanya pada pasien resolusi mayoritas; rasio = konfigurasi terbaik dari validasi)"),
    C("launch_gan_chain('configs/s7_s5.yaml', ratios=[None], reps=REPS, devices=DEVICES)\n"
      "launch_gan_chain('configs/s7_s6.yaml', ratios=[None], reps=REPS, devices=DEVICES)"),
    M("## 3. Hasil dengan interval kepercayaan, dan selisih dalam-distribusi vs pergeseran (**eksploratif**)"),
    C("ALLOW_PARTIAL = REPS is not None\nfor sc, path in (('S7_S1', 'configs/s7_s1.yaml'), ('S7_S5', 'configs/s7_s5.yaml'), ('S7_S6', 'configs/s7_s6.yaml')):\n"
      "    display(evaluate_scenario(load_merged(path), sc, ALLOW_PARTIAL)['table'])\n"
      "for m in ('S1', 'S5', 'S6'):\n    display(s7_gap(cfg7, m, f'S7_{m}'))"),
    M("## 4. Pemeriksaan checkpoint CP6 dan `manifest.json` (sel terakhir)"),
    C("from src.checkpoint import finalize_cp6\nreport = finalize_cp6(cfg7)\nassert report['checkpoint'] == 'CP6'"),
])

# ------------------------------------------------------------------ 07
save("07_evaluasi_akhir", [
    M("# 07 – Evaluasi akhir (CP7, FR-10, FR-13)\n\nPerbandingan utama (ditetapkan sebelum eksperimen): **S6 lawan S1 dan S2** pada AUROC dan F1, dilaporkan apa pun arahnya. "
      "Semua perbandingan lain berlabel *eksploratif*. Setelah CP7 angka tidak diubah; koreksi berarti mengulang dari checkpoint yang terdampak."),
    C(SETUP + "\nfrom src.evaluate import run_comparisons, make_tables"),
    M("## 1. Perbandingan berpasangan (bootstrap pasien yang sama)"),
    C("c = run_comparisons(cfg)\ndisplay(c[c.status == 'utama'])\nprint('\\nEksploratif:')\ndisplay(c[c.status != 'utama'])"),
    M("## 2. Tabel 3.3 dan Tabel 3.4 (dibuat otomatis dari hasil job)\n\nTabel 3.4 memuat baris literatur dari `configs/literature.yaml`; lengkapi dari proposal. Tidak ada angka yang dikarang."),
    C("t = make_tables(cfg)\ndisplay(t['tabel_3_3'])\ndisplay(t['tabel_3_4'])"),
    M("## 3. Pemeriksaan checkpoint CP7 dan `manifest.json` (sel terakhir)"),
    C("from src.checkpoint import finalize_cp7\nreport = finalize_cp7(cfg)\nassert report['checkpoint'] == 'CP7'"),
])
