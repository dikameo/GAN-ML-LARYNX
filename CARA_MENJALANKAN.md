# Cara Menjalankan

Panduan langkah demi langkah untuk menjalankan seluruh pipeline (S1 sampai S7) di mesin ber-GPU. Gambaran umum ada di
[README.md](README.md); spesifikasi di [PRD.md](PRD.md); aturan kerja di [RULES.md](RULES.md).

> **Status kode:** kode S3 sampai S7 dan GAN belum pernah dieksekusi (hanya diperiksa sintaksnya). Langkah 3 (pytest)
> wajib dijalankan dulu agar bug ketahuan sebelum menyentuh data asli.

---

## 0. Yang dibutuhkan

| Kebutuhan | Keterangan |
| --- | --- |
| GPU NVIDIA | Satu atau lebih. Tanpa GPU, S1 dan GAN tidak layak (benchmark CPU laptop: sekitar 31 jam per epoch S1). |
| Driver NVIDIA | Plus NVIDIA Container Toolkit bila memakai Docker. |
| Internet | Untuk mengunci dependensi dan mengunduh bobot Swin-T serta bobot Inception/LPIPS (sekali). |
| Disk kosong | Sekitar 15 GB (dataset 1,4 GB, cache 3 GB, embedding, fitur, sintetis, checkpoint). Tulis ke disk lokal, bukan folder sinkronisasi. |
| Dataset | Folder `data/Larynx_CE-NBI_Dataset/` (berisi `Benign/`, `Malignant/`, `Patients_List_Updated_Final.xlsx`). Dipasang **baca-saja**. |

Dataset adalah data medis. Jangan diunggah ke layanan pihak ketiga tanpa pertimbangan (RULES §7).

## 1. Memindahkan proyek ke mesin tujuan

Salin seluruh folder proyek **kecuali** `outputs/` dari laptop (isinya hasil CPU dan tidak berlaku di mesin lain):

```bash
rsync -a --exclude outputs --exclude .git  "Training/"  mesin-tujuan:~/larynx/   # atau salin lewat disk/USB
cd ~/larynx && git init   # bila .git tidak ikut disalin
```

Pastikan struktur ini ada di mesin tujuan:

```
larynx/
├── data/Larynx_CE-NBI_Dataset/   (Benign/, Malignant/, xlsx)
├── configs/  src/  notebooks/  tests/  scripts/
├── PRD.md  RULES.md  LOG.md  README.md  CARA_MENJALANKAN.md
├── Dockerfile  requirements.in  requirements.gpu.in
└── outputs/                      (dibuat otomatis)
```

## 2. Menyiapkan lingkungan (pilih A atau B)

### A. Docker (disarankan; Linux atau Windows+WSL2)

```bash
chmod +x scripts/*.sh
./scripts/lock.sh --gpu      # menghasilkan requirements.gpu.lock (internet; hanya sekali)
./scripts/build.sh --gpu     # membangun citra larynx-pipeline-gpu
```

Cek GPU terlihat di dalam kontainer:

```bash
./scripts/run.sh --gpu python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Perintah selanjutnya di panduan ini ditulis dengan awalan `RUN=` supaya ringkas:

```bash
RUN="./scripts/run.sh --gpu"
```

Windows PowerShell: `.\scripts\lock.ps1` hanya untuk CPU. Untuk GPU, buat kunci lewat WSL (`./scripts/lock.sh --gpu`), lalu
`.\scripts\build.ps1 -Gpu` dan jalankan dengan `.\scripts\run.ps1 -Gpu <perintah>`. Argumen berawalan `--` harus dikutip,
contoh: `.\scripts\run.ps1 -Gpu python -m src.run_all '--stage' all`.

### B. Tanpa Docker (Colab, Kaggle, server)

```bash
python -m venv .venv && source .venv/bin/activate
pip install pip-tools
pip-compile --strip-extras -o requirements.gpu.lock requirements.gpu.in
pip install -r requirements.gpu.lock
export LARYNX_DATASET=/path/ke/Larynx_CE-NBI_Dataset
export LARYNX_OUTPUTS=/path/persisten/outputs     # WAJIB disk persisten; sesi cloud bisa hilang
RUN=""                                            # tanpa awalan; jalankan perintah langsung
```

Catatan Colab/Kaggle: simpan `LARYNX_OUTPUTS` di Drive atau dataset persisten. Job dapat dilanjutkan dari `last.pt`
setelah sesi putus, jadi cukup jalankan ulang perintah yang sama.

## 3. Uji kode dengan data tiruan (WAJIB, belum menyentuh data asli)

```bash
$RUN python -m pytest -x -q tests
```

- Menggunakan dataset sintetis kecil dan model Swin/GAN mini; tidak mengunduh bobot, kecuali `test_s3.py::test_ctgan_*`
  yang dilewati otomatis bila `ctgan` tidak terpasang.
- Lulus berarti **kode berjalan dan aturan kebocoran (L1 sampai L3, L5) ditegakkan**. Itu bukan bukti kinerja apa pun.
- Bila ada yang gagal, perbaiki dulu (kirim keluarannya ke saya). Jangan lanjut ke data asli.

## 4. Menjalankan semua tahap

```bash
$RUN python -m src.run_all --stage all
```

`run_all` menjalankan notebook 00 sampai 07 berurutan dan **berhenti di checkpoint pertama yang belum lulus**, mencetak
penghambatnya beserta perintah yang diperlukan. Ulangi perintah yang sama setelah menyelesaikannya; job `DONE` dilewati.

Kode keluar: `0` semua lulus, `2` berhenti di checkpoint, `1` notebook gagal.

Opsi: `--stage cp3` (satu tahap), `--from cp3 --to cp5` (rentang).

### Urutan dan apa yang akan diminta

| Tahap | Notebook | Yang terjadi | Perlu kamu lakukan |
| --- | --- | --- | --- |
| CP0 | `00_setup` | uji lingkungan, benchmark Swin-T dan GAN | `approve decide D2 D3 D4`; setelah meninjau: `approve review CP0` |
| CP1 | `01_audit` | audit dari gambar, bandingkan Tabel 3.2 | bila ada selisih angka, jelaskan di LOG.md; `approve review CP1` |
| CP2 | `02_praproses_split` | cache, peta ROI, split 5×2, split S7 | periksa PNG ROI → `approve roi`; `approve decide D6`; `approve review CP2` |
| CP3 | `03_baseline` | S1, S2, S3, S4 (+ sensitivitas) | `approve decide D7`; `approve review CP3` |
| CP4 | `04_pilot_gan` | pilot λ_roi (D2) | periksa grid GAN → `approve gan-visual CP4`; `approve decide D8`; `approve review CP4` |
| CP5 | `05_s5_s6` | S5, S6, rasio terbaik dari validasi | periksa grid dan LPIPS → `approve gan-visual CP5`; `approve review CP5` |
| CP6 | `06_s7` | S7 | `approve review CP6` |
| CP7 | `07_evaluasi_akhir` | perbandingan utama, Tabel 3.3 dan 3.4 | isi `configs/literature.yaml`; `approve review CP7` |

Perintah persetujuan (semua butuh `--by NAMA`):

```bash
$RUN python -m src.approve decide D2 D3 D4 --by "Nama"      # konfirmasi keputusan di configs/decisions.yaml
$RUN python -m src.approve roi --by "Nama"                  # kunci parameter Frangi setelah periksa PNG
$RUN python -m src.approve gan-visual CP4 --by "Nama"       # sesudah memeriksa grid sampel GAN (CP5 untuk mode collapse)
$RUN python -m src.approve review CP1 --by "Nama"           # tinjauan orang kedua (RULES §3)
```

Setelah checkpoint lulus dan ditinjau, buat tag: `git tag cp1` (dan seterusnya).

`--self-review "Nama"` pada `run_all` menandai REVIEWED otomatis. Itu **melewati** ketentuan orang kedua dan tercatat sebagai
self-review; pakai hanya dengan sadar.

## 5. Sebelum menjalankan tahap GAN (CP4)

1. Lihat estimasi waktu di `outputs/cp0/benchmark_gan.json` (`estimasi_jam_20_gan`, dibagi jumlah GPU).
2. Tetapkan `gan.total_kimg` di `configs/base.yaml`. Nilai bawaan (500) **sementara** dan belum berasal dari benchmark.
3. Bila waktu tidak cukup untuk 2 ulangan penuh, PRD mengizinkan 1 ulangan untuk skenario GAN dengan pengurangan itu ditulis
   di hasil. Ubah `splits.n_repeats` **sebelum** CP2 dan catat di `decisions.yaml` (jangan ubah setelah melihat hasil).

## 6. Beberapa GPU

Di sel setup notebook 03, 05, dan 06, ubah `DEVICES = None` menjadi `DEVICES = [0, 1]` (daftar nomor GPU). Setiap GPU
menjalankan satu rantai job (GAN lipatan k → klasifikator tiga rasio) lewat `CUDA_VISIBLE_DEVICES`. Pilihan lain (Docker):

```bash
GPUS='"device=0,1"' ./scripts/run.sh --gpu python -m src.run_all --stage cp5
```

## 7. Menjalankan per skenario

Perintah berikut berdiri sendiri (tanpa notebook) dan semuanya idempoten: job `DONE` dilewati, yang terputus dilanjutkan.
`RUN` adalah awalan dari bagian 2. Tambahkan `--devices 0 1` pada perintah `src.jobs` untuk memakai beberapa GPU.
Opsi `--reps 0` atau `--folds 0 1` hanya membatasi **job yang diluncurkan** (mis. untuk uji coba satu lipatan);
tabel hasil dari proses parsial otomatis ditandai `PARSIAL` dan tidak memenuhi gerbang checkpoint.

### 7.0 Prasyarat bersama (sekali)

```bash
CFG=configs/base.yaml
$RUN python -m src.env_check --config $CFG --out /outputs/cp0/env_check.json   # CP0
$RUN python -m src.audit --config $CFG                                          # CP1 -> outputs/cp1/audit.csv
$RUN python -m src.preprocess --config $CFG                                     # CP2 -> cache memmap + contoh ROI
$RUN python -m src.approve roi --by "Nama"                                      # setelah memeriksa PNG di outputs/cp2/roi_check/
$RUN python -m src.splits --config $CFG --s7                                    # CP2 -> split 5x2 + split S7
```

Cek status job kapan saja: `$RUN python -m src.jobs status --config configs/s1.yaml`.

### 7.1 S1 — Swin-T pralatih, weighted sampling (baseline)

```bash
$RUN python -m src.jobs scenario --config configs/s1.yaml --module src.train          # 10 job (2 ulangan x 5 lipatan)
$RUN python -m src.evaluate --config configs/s1.yaml --scenario S1
```
Satu job saja: `$RUN python -m src.train --config configs/s1.yaml --rep 0 --fold 0`.
Hasil: `outputs/results/results_S1.csv` (+ `fold_metrics_S1.csv`, `patient_probs_S1.csv`); kurva latih per job di `history.csv`.

### 7.2 S2 — embedding Swin beku + SMOTE + kepala linear

```bash
$RUN python -m src.embed --config configs/s2.yaml                                     # embedding semua citra, sekali
$RUN python -m src.jobs scenario --config configs/s2.yaml --module src.train_s2
$RUN python -m src.evaluate --config configs/s2.yaml --scenario S2
```
Hasil: `results_S2.csv`; `emb.npy` di `outputs/embeddings/`.

### 7.3 S3 — replikasi Wang dkk. (PyRadiomics → MinMax → LASSO → SMOTE → SVM linear / LDA)

```bash
$RUN python -m src.radiomics_feats --config configs/s3_svm.yaml                       # fitur sekali, dipakai S3 dan S4
$RUN python -m src.jobs scenario --config configs/s3_svm.yaml --module src.train_s3
$RUN python -m src.jobs scenario --config configs/s3_lda.yaml --module src.train_s3
$RUN python -m src.evaluate --config $CFG --scenario S3_svm
$RUN python -m src.evaluate --config $CFG --scenario S3_lda
```
Sensitivitas LASSO 0,01 (eksploratif, D3): ulangi dengan `configs/s3_svm_a01.yaml` dan `configs/s3_lda_a01.yaml`, skenario `S3_svm_a01` / `S3_lda_a01`.
Keputusan terkait: D3 dan D7 (`approve decide D3 D7`). Jumlah fitur aktual tercatat di `outputs/features/rad_*/summary.json`.

### 7.4 S4 — S3 + CTGAN pada ruang fitur (50% data latih)

```bash
$RUN python -m src.jobs scenario --config configs/s4_svm.yaml --module src.train_s3
$RUN python -m src.jobs scenario --config configs/s4_lda.yaml --module src.train_s3
$RUN python -m src.evaluate --config $CFG --scenario S4_svm
$RUN python -m src.evaluate --config $CFG --scenario S4_lda
```
Sensitivitas: `configs/s4_svm_a01.yaml`, `configs/s4_lda_a01.yaml`. Butuh fitur dari 7.3.

### 7.5 S5 — cGAN tanpa ROI (λ_roi = 0), rasio sintetis 25%, 50%, 100%

```bash
$RUN python -m src.jobs gan-chain --config configs/s5.yaml                            # per lipatan: GAN -> Swin-T x 3 rasio
$RUN python -m src.evaluate --config configs/s5.yaml --selected S5                    # semua rasio + rasio terpilih (validasi)
```
Pastikan `gan.total_kimg` sudah ditetapkan (bagian 5). Hasil: `results_S5_r25.csv`, `_r50`, `_r100`, `results_S5.csv`
(rasio terbaik per lipatan dari **validasi**), `ratio_selection_S5.csv`; citra sintetis dan metrik GAN di
`outputs/jobs/S5_gan/rep<r>_fold<k>/` (`synth.npz`, `gan_metrics.json`, `grid_test/*.png`).

### 7.6 S6 — cGAN berpandu ROI vaskular (metode usulan)

```bash
$RUN python -m src.pilot_lambda --config configs/s6.yaml                              # pilot λ_roi dari validasi (D2) -> outputs/cp4/lambda_selection.json
$RUN python -m src.approve gan-visual CP4 --by "Nama"                                 # setelah memeriksa grid_val di tiap folder pilot
$RUN python -m src.jobs gan-chain --config configs/s6.yaml
$RUN python -m src.evaluate --config configs/s6.yaml --selected S6
$RUN python -m src.evaluate --config configs/s6.yaml --select-best                    # best_config.json (butuh S5 dan S6 selesai)
```
Hasil sama bentuknya dengan S5. λ_roi terpilih di `outputs/cp4/lambda_selection.json`.

### 7.7 S7 — ketahanan terhadap pergeseran resolusi

Prasyarat: split S7 (7.0), S1 (7.1), S5 dan S6 (7.5, 7.6), dan `best_config.json`.

```bash
$RUN python -m src.jobs scenario --config configs/s7_s1.yaml --module src.train
$RUN python -m src.jobs gan-chain --config configs/s7_s5.yaml --ratios none          # rasio terbaik dari best_config.json
$RUN python -m src.jobs gan-chain --config configs/s7_s6.yaml --ratios none
$RUN python -m src.evaluate --config configs/s7_s1.yaml --scenario S7_S1
$RUN python -m src.evaluate --config configs/s7_s5.yaml --scenario S7_S5
$RUN python -m src.evaluate --config configs/s7_s6.yaml --scenario S7_S6
$RUN python -m src.evaluate --config configs/s7_s1.yaml --s7-gap S1                   # selisih dalam-distribusi vs pergeseran (eksploratif)
$RUN python -m src.evaluate --config configs/s7_s1.yaml --s7-gap S5
$RUN python -m src.evaluate --config configs/s7_s1.yaml --s7-gap S6
```
Hasil: `results_S7_*.csv`, `s7_gap_*.csv`. Ingat catatan interpretasi: pergeseran akuisisi bercampur perbedaan populasi.

### 7.8 Evaluasi akhir (setelah semua skenario)

```bash
$RUN python -m src.evaluate --config $CFG --compare        # S6 vs S1 dan S6 vs S2 = 'utama'; lainnya 'eksploratif'
$RUN python -m src.evaluate --config $CFG --tables        # Tabel 3.3 dan 3.4
```
Hasil: `comparisons_all.csv`, `tabel_3_3.csv` / `.md`, `tabel_3_4.csv`. Isi dulu `configs/literature.yaml` untuk Tabel 3.4.
Lalu jalankan pemeriksaan gerbang: `$RUN python -m src.run_all --stage cp7`.

Urutan wajib: GAN lipatan k → klasifikator S5/S6 lipatan k (`src.train` menolak berjalan bila GAN-nya belum `DONE`).

## 8. Terputus di tengah jalan

Cukup jalankan ulang perintah yang sama.
- Pelatihan dilanjutkan dari `last.pt` (ditulis atomik tiap epoch; GAN tiap `gan.snapshot_kimg`).
- Job yang sudah `DONE` dengan hash konfigurasi sama dilewati.
- Mengubah konfigurasi mengubah hash, sehingga job terkait dijalankan ulang dari awal.
- Job yang gagal dua kali berturut-turut menghentikan antrean dan dicatat di `outputs/jobs/launch.log`. Catat penyebabnya di
  LOG.md sebelum melanjutkan (RULES §4).

## 9. Hasil ada di mana

| Isi | Lokasi |
| --- | --- |
| Tabel hasil, perbandingan, Tabel 3.3/3.4 | `outputs/results/` |
| Audit data | `outputs/cp1/audit.csv`, `audit_summary.json` |
| Contoh ROI (Gambar 3.2) | `outputs/cp2/roi_check/` |
| Cache dan split | `outputs/cache/`, `outputs/splits/` |
| Prediksi per pasien tiap job | `outputs/jobs/<skenario>/rep<r>_fold<k>/preds_test_patients.csv` |
| Citra sintetis dan metrik GAN | `outputs/jobs/<S5|S6>_gan/rep<r>_fold<k>/` |
| Status dan hash tiap checkpoint | `outputs/cp<N>/cp<N>_check.json`, `manifest.json` |

## 10. Masalah yang mungkin muncul

| Gejala | Kemungkinan penyebab dan tindakan |
| --- | --- |
| `lock.sh` gagal pada `pyradiomics` | Belum terverifikasi dengan NumPy 2. Coba pasang NumPy <2 di `requirements*.in`, atau lewati S3/S4 dengan menghapusnya dari daftar `MAIN` di notebook 03 **dan catat sebagai pengurangan cakupan di PRD** (RULES §6). |
| `CUDA out of memory` | Turunkan `train.batch_size` (S1) atau `gan.batch_size`; catat perubahannya di LOG.md **sebelum** menjalankan hasil apa pun. |
| `GPU smoke: TIDAK_TERSEDIA` | Kontainer tidak melihat GPU. Pastikan `--gpus all`, driver, dan NVIDIA Container Toolkit. |
| Tahap berhenti dengan `MENUNGGU` | Bukan galat: butuh keputusan manusia (bagian 4). Jalankan perintah `approve` yang dicetak lalu ulangi. |
| `lambda_roi belum ditetapkan` | Jalankan CP4 (`src.pilot_lambda`) lebih dulu. |
| `ratio 'best' butuh best_config.json` | Selesaikan CP5 lalu `--select-best` sebelum S7. |
| `GAN ... belum DONE` | Urutan salah; jalankan `src.train_gan` untuk lipatan itu dulu. |
| `LASSO memilih 0 fitur` | Sengaja tidak diganti diam-diam. Periksa fitur/alpha; ubah di konfigurasi dan catat di `decisions.yaml`. |
| Hasil tampak jelek | Laporkan apa adanya. **Dilarang** mengganti seed, membuang lipatan, atau menjalankan ulang karena hasilnya jelek (RULES §1). |

## 11. Yang tidak boleh dilakukan

- Menyentuh data uji untuk memilih apa pun (rasio, λ_roi, ambang, arsitektur). Semua pilihan memakai pasien validasi di dalam lipatan latih.
- Menulis ke folder dataset.
- `pip install` manual di dalam kontainer. Tambah pustaka di `requirements*.in`, kunci ulang, bangun ulang citra.
- Mengubah angka di laporan setelah CP7. Koreksi berarti mengulang dari checkpoint yang terdampak.
