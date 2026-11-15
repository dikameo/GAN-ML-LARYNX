# Augmentasi GAN Berpandu ROI untuk Klasifikasi Lesi Laring pada Citra CE-NBI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-GPU-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Classifier](https://img.shields.io/badge/Classifier-Swin--T-009688?style=for-the-badge)
![GAN](https://img.shields.io/badge/Augmentasi-GAN_ROI--guided-673AB7?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-GPU-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-TODO-2E7D32?style=for-the-badge)

**Repositori Riset & Dokumentasi Teknis**
**TODO: Program Studi, Fakultas, Universitas**

[![Status](https://img.shields.io/badge/Publikasi-TODO_(DOI/draf)-00629B?style=for-the-badge)](#1-ringkasan-publikasi--tim-peneliti)

</div>

> **Status:** kerangka README. Semua tanda `TODO` menandai data yang harus diisi dari hasil eksperimen asli. Tidak ada angka hasil yang diisi dari asumsi.

---

## Daftar Isi

1. [Ringkasan Publikasi & Tim Peneliti](#1-ringkasan-publikasi--tim-peneliti)
2. [Abstrak & Kontribusi Ilmiah](#2-abstrak--kontribusi-ilmiah)
3. [Latar Belakang Klinis & Perumusan Masalah](#3-latar-belakang-klinis--perumusan-masalah)
4. [Dataset & Pipeline Preprocessing](#4-dataset--pipeline-preprocessing)
5. [Arsitektur Sistem & Metodologi](#5-arsitektur-sistem--metodologi)
6. [Hasil Eksperimen & Analisis Kuantitatif](#6-hasil-eksperimen--analisis-kuantitatif)
7. [Analisis Kualitas GAN & Batas yang Diketahui](#7-analisis-kualitas-gan--batas-yang-diketahui)
8. [Stack Teknologi & Struktur Repositori](#8-stack-teknologi--struktur-repositori)
9. [Panduan Replikasi (Langkah demi Langkah)](#9-panduan-replikasi-langkah-demi-langkah)
10. [Alur Checkpoint & Persetujuan Manusia](#10-alur-checkpoint--persetujuan-manusia)
11. [Pemecahan Masalah](#11-pemecahan-masalah)
12. [Sitasi, Referensi & Lisensi](#12-sitasi-referensi--lisensi)

---

## 1. Ringkasan Publikasi & Tim Peneliti

| Atribut | Detail |
|:--------|:-------|
| **Judul** | TODO |
| **Venue / Status** | TODO (jurnal, volume, halaman, atau "draf") |
| **DOI** | TODO |
| **Afiliasi** | TODO |
| **Dokumen spesifikasi** | [PRD.md](PRD.md) |
| **Aturan kerja** | [RULES.md](RULES.md) |
| **Keputusan terbuka & perubahan PRD** | [configs/decisions.yaml](configs/decisions.yaml) |
| **Catatan eksperimen** | [LOG.md](LOG.md) |

### Penulis & Kontribusi

| Peran | Nama | Tanggung Jawab | Kontak |
|:------|:-----|:---------------|:-------|
| TODO | **Wahyu Andika** | TODO (lihat LOG.md awal tiap fase) | TODO |
| TODO | **Ahmad Nur Mu'minin** | TODO (lihat LOG.md awal tiap fase) | TODO |
| TODO | TODO (pembimbing) | TODO | TODO |

### Akses Cepat ke Sumber Daya

| Modul | Lokasi | Deskripsi |
|:------|:-------|:----------|
| **Konfigurasi** | [`configs/`](configs/) | `base.yaml` (semua angka tetap), `s1..s7_*.yaml` (skenario), `decisions.yaml`, `literature.yaml` |
| **Kode sumber** | [`src/`](src/) | Paket Python: audit, preprocess, splits, train, GAN, evaluate |
| **Jaringan GAN** | [`src/gan/`](src/gan/) | G/D/D_roi, ADA-lite, Frangi diferensiabel, metrik FID/LPIPS/Sato |
| **Notebook** | [`notebooks/`](notebooks/) | `00_setup` sampai `07_evaluasi_akhir`, satu per checkpoint |
| **Uji** | [`tests/`](tests/) | Uji dengan data tiruan |
| **Keluaran** | `outputs/` | Cache, split, job, hasil (tidak masuk git) |

---

## 2. Abstrak & Kontribusi Ilmiah

### Abstrak

TODO. Tulis 1 paragraf latar (klasifikasi lesi laring dari citra CE-NBI, keterbatasan data), 1 paragraf metode (augmentasi GAN berpandu ROI Frangi, klasifikator Swin-T, tujuh skenario S1–S7), dan 1 paragraf temuan.

**Temuan utama** (isi dengan angka dari `outputs/` setelah CP7):

1. TODO: efek augmentasi GAN berpandu ROI terhadap kinerja hilir (S5/S6 vs S1), disertai interval kepercayaan.
2. TODO: perbandingan dengan jalur radiomics (S3/S4) dan replikasi Wang dkk.
3. TODO: kinerja di bawah pergeseran akuisisi (S7).

### Kontribusi Utama

- **Augmentasi GAN berpandu ROI:** diskriminator ROI (`D_roi`) dengan peta pembuluh dari filter Frangi diferensiabel, ADA-lite untuk data kecil.
- **Desain eksperimen tujuh skenario** yang membandingkan deep learning, embedding, radiomics, dan augmentasi sintetis dengan protokol split yang sama.
- **Pipeline idempoten dan dapat dilanjutkan:** tiap job menulis `last.pt` secara atomik, `DONE` berisi hash konfigurasi, job selesai dilewati.
- **Tata kelola checkpoint dengan persetujuan manusia** untuk keputusan yang tidak boleh diotomatisasi (ROI, tinjauan visual GAN, penyimpangan dari PRD).

---

## 3. Latar Belakang Klinis & Perumusan Masalah

TODO. Isi: (a) mengapa CE-NBI dipakai untuk lesi laring, (b) mengapa data terbatas dan tidak seimbang, (c) hipotesis bahwa GAN yang dipandu struktur pembuluh (ROI) menghasilkan sampel yang lebih berguna untuk klasifikasi daripada GAN biasa.

Pertanyaan penelitian:

| Kode | Pertanyaan | Skenario terkait |
|:----:|:-----------|:-----------------|
| RQ1 | TODO | S1, S5, S6 |
| RQ2 | TODO | S2, S3, S4 |
| RQ3 | TODO | S7 |

---

## 4. Dataset & Pipeline Preprocessing

### 4.1 Dataset

Dataset: **Larynx CE-NBI** (`data/Larynx_CE-NBI_Dataset/`, **baca-saja**). Sumber dan sitasi: TODO.

| Kelas | Jumlah Total | Train | Validasi | Uji |
|:------|-------------:|------:|---------:|----:|
| TODO | TODO | TODO | TODO | TODO |
| **Total** | TODO | TODO | TODO | TODO |

Isi tabel dari keluaran CP1 (`notebooks/01_*`, audit data; bandingkan dengan Tabel 3.2 proposal dan jelaskan selisihnya).

> **Catatan data:** ini dataset medis. Jangan unggah ke layanan pihak ketiga tanpa pertimbangan (RULES §7).

### 4.2 Pipeline Preprocessing

Alur modul: `audit` → `preprocess` → `splits` → `train` / `train_s2` / `train_s3` / `train_gan` → `evaluate`.

1. **Audit** (`src.audit`): verifikasi jumlah dan struktur data.
2. **Cache & ROI Frangi** (`src.preprocess`): hasil cache di `outputs/`. Parameter Frangi dikunci setelah pemeriksaan visual (`approve roi`).
3. **Split** (`src.splits`): TODO (rasio, stratifikasi, seed, split `s7` terpisah untuk pergeseran akuisisi).
4. **Augmentasi konvensional:** TODO.

<div align="center">
  <img src="docs/images/TODO_dataset_samples.png" alt="Contoh citra CE-NBI" width="800">
  <p><em>Gambar 1: Contoh citra CE-NBI per kelas. TODO.</em></p>
</div>

---

## 5. Arsitektur Sistem & Metodologi

<div align="center">
  <img src="docs/images/TODO_pipeline.png" alt="Diagram pipeline" width="900">
  <p><em>Gambar 2: Diagram alur end-to-end. TODO.</em></p>
</div>

### 5.1 Tujuh Skenario Eksperimen

| Skenario | Deskripsi | Modul | Konfigurasi |
|:---------|:----------|:------|:------------|
| **S1** | Baseline Swin-T | `src.train` | `configs/s1.yaml` |
| **S2** | Embedding + klasifikator | `src.embed` → `src.train_s2` | `configs/s2.yaml` |
| **S3, S4** | Radiomics (SVM/LDA); `_a01` adalah sensitivitas LASSO 0,01 (eksploratif) | `src.radiomics_feats` → `src.train_s3` | `configs/s3_*.yaml`, `s4_*.yaml` |
| **S5, S6** | Augmentasi GAN; S6 memakai pilot λ_roi | `src.pilot_lambda` (S6), `src.train_gan`, `src.train --ratio` | `configs/s5.yaml`, `s6.yaml` |
| **S7** | Uji pergeseran akuisisi (rantai S1/S5/S6, split `s7`) | `src.train` / rantai GAN | `configs/s7_s1.yaml`, `s7_s5.yaml`, `s7_s6.yaml` |

### 5.2 Klasifikator

Swin-T. TODO: ukuran input, bobot awal, optimizer, learning rate, epoch, early stopping (semua angka tetap ada di `configs/base.yaml`; salin nilainya ke sini).

### 5.3 GAN Berpandu ROI

Komponen di `src/gan/`:

| Komponen | Fungsi |
|:---------|:-------|
| `G` | Generator. TODO detail arsitektur |
| `D` | Diskriminator global |
| `D_roi` | Diskriminator pada ROI hasil Frangi |
| ADA-lite | Augmentasi adaptif diskriminator untuk data kecil |
| Frangi diferensiabel | Peta pembuluh yang dapat dilalui gradien |
| Metrik | FID, LPIPS, Sato |

Fungsi loss (isi dengan persamaan dari PRD):

$$
\mathcal{L}_{G} = \mathcal{L}_{\text{adv}} + \lambda_{\text{roi}} \, \mathcal{L}_{\text{roi}} \tag{1}
$$

TODO: verifikasi bentuk persamaan dengan PRD sebelum dipublikasikan. Nilai λ_roi dipilih lewat pilot (`src.pilot_lambda`, CP4).

### 5.4 Radiomics (S3/S4)

PyRadiomics dengan seleksi fitur LASSO, klasifikator SVM dan LDA. Mask PyRadiomics adalah area pandang, bukan lesi (D7), sehingga jumlah fitur dilaporkan apa adanya (Wang dkk.: 104). Urutan CTGAN-sebelum-SMOTE di S4 adalah asumsi dari ringkasan PRD; cocokkan dengan notebook Wang dkk. sebelum mengklaim replikasi.

### 5.5 Metrik Evaluasi

TODO: daftar metrik (akurasi, F1, AUC, dll.), metode interval kepercayaan, dan uji statistik. Interval kepercayaan hanya menangkap variasi pemilihan pasien uji, bukan variasi pelatihan model.

---

## 6. Hasil Eksperimen & Analisis Kuantitatif

> Semua tabel di bagian ini diisi dari `notebooks/07_evaluasi_akhir` (CP7). Jangan isi dari memori atau perkiraan.

### 6.1 Perbandingan Utama S1–S6

| Skenario | Akurasi | F1 (makro) | AUC | 95% CI | Catatan |
|:---------|:-------:|:----------:|:---:|:------:|:--------|
| S1 | TODO | TODO | TODO | TODO | Baseline |
| S2 | TODO | TODO | TODO | TODO | |
| S3 | TODO | TODO | TODO | TODO | |
| S4 | TODO | TODO | TODO | TODO | |
| S5 | TODO | TODO | TODO | TODO | Rasio terbaik dari validasi: TODO |
| S6 | TODO | TODO | TODO | TODO | λ_roi = TODO |

### 6.2 Perbandingan dengan Literatur (Tabel 3.4)

Angka literatur disimpan di `configs/literature.yaml`. Saat ini hanya angka Wang dkk. dari PRD yang terisi.

| Studi | Metode | Metrik | Nilai |
|:------|:-------|:-------|:-----:|
| Wang dkk. | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO |
| **Penelitian ini** | TODO | TODO | TODO |

### 6.3 Pergeseran Akuisisi (S7)

| Kondisi | S1 | S5 | S6 |
|:--------|:--:|:--:|:--:|
| In-distribusi | TODO | TODO | TODO |
| Pergeseran | TODO | TODO | TODO |

Keterbatasan S7: kelompok uji gabungan 4 resolusi, sehingga hasil mengukur pergeseran akuisisi bercampur perbedaan populasi, bukan efek resolusi murni. Selisih in-distribusi vs pergeseran tidak berpasangan dan kondisi pelatihannya tidak identik, jadi analisis ini eksploratif.

<div align="center">
  <img src="docs/images/TODO_confusion_matrix.png" alt="Confusion matrix" width="600">
  <p><em>Gambar 3: Confusion matrix model terbaik. TODO.</em></p>
</div>

---

## 7. Analisis Kualitas GAN & Batas yang Diketahui

### 7.1 Kualitas Sampel GAN

| Metrik | GAN biasa | GAN + ROI | Catatan |
|:-------|:---------:|:---------:|:--------|
| FID | TODO | TODO | Inception-ImageNet torchvision |
| LPIPS | TODO | TODO | |
| Sato | TODO | TODO | |

FID memakai Inception-ImageNet dari torchvision sehingga **tidak sebanding dengan angka literatur**. Bukti utama adalah kinerja hilir.

<div align="center">
  <img src="docs/images/TODO_gan_samples.png" alt="Grid sampel GAN" width="800">
  <p><em>Gambar 4: Grid sampel GAN (diperiksa manusia pada CP4 dan CP5). TODO.</em></p>
</div>

### 7.2 Batas yang Diketahui

- S7: kelompok uji gabungan 4 resolusi; hasil mengukur pergeseran akuisisi bercampur perbedaan populasi, bukan efek resolusi murni.
- Selisih in-distribusi vs pergeseran di S7 tidak berpasangan dan kondisi pelatihannya tidak identik: eksploratif.
- Interval kepercayaan hanya menangkap variasi pemilihan pasien uji, bukan variasi pelatihan model.
- Mask PyRadiomics adalah area pandang, bukan lesi (D7); jumlah fitur dilaporkan apa adanya (Wang dkk.: 104).
- FID memakai Inception-ImageNet torchvision; tidak sebanding dengan angka literatur.
- Urutan CTGAN-sebelum-SMOTE di S4 adalah asumsi dari ringkasan PRD.

---

## 8. Stack Teknologi & Struktur Repositori

### 8.1 Stack

| Kategori | Paket | Kegunaan |
|:---------|:------|:---------|
| Deep learning | `torch`, `torchvision` | Swin-T, GAN |
| Radiomics | `pyradiomics`, `scikit-learn` | Fitur, SVM/LDA, LASSO |
| Data sintetis tabular | TODO (CTGAN, SMOTE) | S4 |
| Metrik GAN | TODO (FID, LPIPS) | Evaluasi sampel |
| Infrastruktur | Docker + NVIDIA Container Toolkit | Lingkungan terkunci |

Versi pasti ada di `requirements.gpu.lock` dan `requirements.cpu.lock` (dibuat oleh `scripts/lock.sh`).

### 8.2 Struktur Repositori

```text
.
├── PRD.md                   # Spesifikasi
├── RULES.md                 # Aturan kerja
├── LOG.md                   # Catatan eksperimen
├── configs/
│   ├── base.yaml            # Semua angka tetap
│   ├── s1.yaml ... s7_*.yaml
│   ├── decisions.yaml       # Keputusan terbuka & perubahan PRD
│   └── literature.yaml      # Angka literatur (Tabel 3.4)
├── data/
│   └── Larynx_CE-NBI_Dataset/   # baca-saja
├── src/
│   ├── audit.py, preprocess.py, splits.py
│   ├── train.py, train_s2.py, train_s3.py, train_gan.py
│   ├── embed.py, radiomics_feats.py, pilot_lambda.py
│   ├── evaluate.py, approve.py, run_all.py
│   └── gan/                 # G, D, D_roi, ADA-lite, Frangi, FID/LPIPS/Sato
├── notebooks/               # 00_setup ... 07_evaluasi_akhir
├── tests/                   # Uji dengan data tiruan
├── scripts/                 # lock, build, run (.sh dan .ps1)
└── outputs/                 # Cache, split, job, hasil (tidak masuk git)
```

---

## 9. Panduan Replikasi (Langkah demi Langkah)

Ada dua jalur: **A. Docker** (disarankan, hasil paling reprodusibel) dan **B. Tanpa Docker** (Colab, Kaggle, server).

### 9.0 Prasyarat

| Kebutuhan | Jalur A (Docker) | Jalur B (tanpa Docker) |
|:----------|:-----------------|:-----------------------|
| OS | Linux, atau Windows + WSL2 | Linux/Colab/Kaggle |
| GPU | NVIDIA + driver + NVIDIA Container Toolkit | NVIDIA GPU dengan CUDA |
| Python | Di dalam citra | 3.11+ |
| Internet | Sekali, untuk mengunci dependensi | Sekali, untuk `pip install` |
| Disk | TODO GB (cache + outputs) | TODO GB |
| Waktu | TODO jam (estimasi jam GAN tercetak di notebook 04) | TODO jam |

Cek GPU terlihat:

```bash
nvidia-smi
```

### 9.1 Ambil Kode

```bash
git clone TODO_URL_REPO
cd TODO_NAMA_REPO
```

### 9.2 Siapkan Dataset

Letakkan dataset di `data/Larynx_CE-NBI_Dataset/` (baca-saja). Sumber dan cara akses: TODO.

```bash
ls data/Larynx_CE-NBI_Dataset | head
```

### 9.3 Jalur A: Docker

```bash
chmod +x scripts/*.sh

# 1. Kunci dependensi (butuh internet; sekali)
./scripts/lock.sh --gpu            # Windows: scripts\lock.ps1 hanya untuk CPU; pakai WSL untuk --gpu

# 2. Bangun citra
./scripts/build.sh --gpu

# 3. Uji kode dengan data tiruan DULU (menemukan bug sebelum menyentuh data asli)
./scripts/run.sh --gpu python -m pytest -x -q tests

# 4. Jalankan semua tahap; berhenti di checkpoint pertama yang butuh keputusan manusia
./scripts/run.sh --gpu python -m src.run_all --stage all
```

Windows PowerShell:

```powershell
.\scripts\build.ps1 -Gpu
.\scripts\run.ps1 -Gpu python -m src.run_all '--stage' all
```

Argumen berawalan `--` harus dikutip di PowerShell.

Beberapa GPU: ubah `DEVICES = [0, 1]` di sel setup notebook 03, 05, dan 06.

### 9.4 Jalur B: Tanpa Docker

```bash
# 1. Pasang dependensi dari lock (lock dibuat di mesin ber-internet)
pip install -r requirements.gpu.lock

# 2. Arahkan ke dataset (baca-saja) dan folder keluaran (HARUS persisten)
export LARYNX_DATASET=/path/ke/Larynx_CE-NBI_Dataset
export LARYNX_OUTPUTS=/path/persisten/outputs   # Drive/disk; sesi Colab/Kaggle bisa hilang

# 3. Uji dengan data tiruan
python -m pytest -x -q tests

# 4. Jalankan semua tahap
python -m src.run_all --stage all
```

### 9.5 Isi Keputusan Sebelum Menjalankan

1. `gan.total_kimg` di `configs/base.yaml`, setelah `bench_gan` di CP0. Estimasi jam GAN tercetak di notebook 04.
2. `configs/literature.yaml`: isi Tabel 3.4 dari proposal (hanya angka Wang dkk. dari PRD yang sudah terisi).
3. D4, D7, D8 di `configs/decisions.yaml`. D7 dan D8 adalah penyimpangan dari PRD dan perlu disetujui kedua penulis.
4. Pembagian tugas tiap penulis: tulis di `LOG.md` pada awal tiap fase.

### 9.6 Menjalankan Tahap demi Tahap

`run_all` berjalan berurutan dan berhenti di checkpoint pertama yang belum lulus, lalu mencetak penghambatnya. Selesaikan penghambat, lalu ulangi perintah yang sama (idempoten).

```bash
# Satu tahap saja (nilai --stage: TODO, cek dengan --help)
python -m src.run_all --help
python -m src.run_all --stage all
```

Jalankan satu skenario secara manual (opsional, untuk debugging). Cocokkan nama flag dengan `--help` tiap modul sebelum dipakai:

```bash
# S1: baseline
python -m src.train --help
python -m src.train --config configs/s1.yaml        # TODO: verifikasi nama flag

# S2: embedding lalu klasifikator
python -m src.embed --config configs/s2.yaml        # TODO: verifikasi nama flag
python -m src.train_s2 --config configs/s2.yaml

# S3/S4: radiomics lalu klasifikator (SVM/LDA)
python -m src.radiomics_feats --config configs/s3_svm.yaml   # TODO: nama file config
python -m src.train_s3 --config configs/s3_svm.yaml

# S5/S6: pilot lambda (S6), latih GAN, lalu klasifikator dengan rasio sintetis
python -m src.pilot_lambda --config configs/s6.yaml
python -m src.train_gan --config configs/s6.yaml
python -m src.train --config configs/s6.yaml --ratio 1.0      # TODO: nilai rasio dari validasi
```

### 9.7 Melanjutkan Job yang Terputus

Tiap job menulis `last.pt` secara atomik dan `DONE` berisi hash konfigurasi.

- Job selesai: dilewati otomatis.
- Job terputus: dilanjutkan dari `last.pt`.
- Konfigurasi berubah: hash tidak cocok, job dijalankan ulang.

Cukup jalankan ulang perintah yang sama:

```bash
python -m src.run_all --stage all
```

### 9.8 Verifikasi Hasil

```bash
# Struktur keluaran
ls outputs/

# Cek job selesai
find outputs -name DONE | sort

# Uji unit ulang bila kode diubah
python -m pytest -x -q tests
```

Bandingkan angka di `notebooks/07_evaluasi_akhir` dengan Bagian 6. Tolerance numerik antar mesin GPU: TODO (isi setelah replikasi pertama pada mesin lain).

---

## 10. Alur Checkpoint & Persetujuan Manusia

| Checkpoint | Notebook | Isi | Persetujuan yang diminta |
|:----------:|:--------:|:----|:-------------------------|
| CP0 | 00 | Lingkungan, benchmark Swin-T dan GAN | `approve decide D2 D3 D4`; REVIEWED |
| CP1 | 01 | Audit data (bandingkan Tabel 3.2) | Selisih angka dijelaskan; REVIEWED |
| CP2 | 02 | Cache, ROI Frangi, split, L1–L3, L5 | `approve roi`; `approve decide D6`; REVIEWED |
| CP3 | 03 | S1–S4 (+ sensitivitas) | `approve decide D7`; REVIEWED |
| CP4 | 04 | Pilot λ_roi (D2) | `approve gan-visual CP4`; `approve decide D8`; REVIEWED |
| CP5 | 05 | S5, S6, rasio terbaik dari validasi | `approve gan-visual CP5`; REVIEWED |
| CP6 | 06 | S7 (L5) | REVIEWED |
| CP7 | 07 | Perbandingan utama, Tabel 3.3 dan 3.4 | REVIEWED |

Perintah persetujuan:

```bash
python -m src.approve decide D2 D3 D4 --by NAMA     # konfirmasi keputusan di decisions.yaml
python -m src.approve roi --by NAMA                  # setelah memeriksa PNG ROI (mengunci parameter Frangi)
python -m src.approve gan-visual CP4 --by NAMA       # setelah memeriksa grid sampel GAN
python -m src.approve review CP1 --by NAMA           # tinjauan orang kedua (RULES §3)
git tag cp1                                          # tandai checkpoint
```

> **Peringatan:** `python -m src.run_all --self-review NAMA` menandai REVIEWED otomatis. Ini **melewati** ketentuan tinjauan orang kedua dan tercatat sebagai self-review. Pakai hanya dengan sadar.

---

## 11. Pemecahan Masalah

| Gejala | Kemungkinan Penyebab | Tindakan |
|:-------|:---------------------|:---------|
| `nvidia-smi` gagal di dalam kontainer | NVIDIA Container Toolkit belum terpasang | Pasang toolkit, restart Docker |
| Sesi Colab/Kaggle putus di tengah GAN | `LARYNX_OUTPUTS` tidak persisten | Arahkan ke Drive/disk persisten, jalankan ulang perintah yang sama |
| Job diulang dari awal | Hash konfigurasi berubah | Cek perubahan di `configs/`; kembalikan atau terima pengulangan |
| `run_all` berhenti | Checkpoint butuh persetujuan | Baca penghambat yang tercetak, jalankan perintah `approve` yang sesuai |
| PowerShell salah baca `--stage` | Argumen berawalan `--` tidak dikutip | Tulis `'--stage'` |
| Out of memory GPU | Batch terlalu besar untuk GPU | TODO: parameter di `base.yaml` |
| Angka audit beda dari Tabel 3.2 | Perbedaan versi dataset | Jelaskan selisih di LOG.md (syarat CP1) |

---

## 12. Sitasi, Referensi & Lisensi

### 12.1 Cara Mengutip

```bibtex
@article{TODO_key,
  author  = {Andika, Wahyu and Mu'minin, Ahmad Nur and TODO},
  title   = {TODO},
  journal = {TODO},
  year    = {TODO},
  doi     = {TODO}
}
```

### 12.2 Referensi Utama

1. **Wang dkk.** TODO. Judul, venue, tahun, DOI.
2. TODO: dataset CE-NBI laring.
3. TODO: Swin Transformer (Liu dkk., 2021).
4. TODO: Frangi dkk., 1998.
5. TODO: StyleGAN2-ADA (Karras dkk., 2020).
6. TODO: PyRadiomics (van Griethuysen dkk., 2017).

### 12.3 Lisensi

TODO: pilih lisensi kode dan lisensi dokumen/figur, lalu tambahkan berkas `LICENSE`.

**Disclaimer klinis:** repositori ini hanya untuk penelitian akademik. Tidak ditujukan atau disertifikasi untuk diagnosis klinis otonom.
