# Pipeline eksperimen: klasifikasi lesi laring CE-NBI dengan augmentasi GAN berpandu ROI

Spesifikasi ada di [PRD.md](PRD.md); aturan kerja di [RULES.md](RULES.md); keputusan terbuka dan perubahan PRD di
[configs/decisions.yaml](configs/decisions.yaml); catatan eksperimen di [LOG.md](LOG.md).


## Struktur

| Lokasi | Isi |
| --- | --- |
| `data/Larynx_CE-NBI_Dataset/` | dataset, **baca-saja** |
| `configs/` | `base.yaml` (semua angka tetap), `s1..s7_*.yaml` (skenario), `decisions.yaml`, `literature.yaml` |
| `src/` | paket Python. Alur: `audit` → `preprocess` → `splits` → `train` / `train_s2` / `train_s3` / `train_gan` → `evaluate` |
| `src/gan/` | jaringan G/D/D_roi, ADA-lite, Frangi diferensiabel, metrik FID/LPIPS/Sato |
| `notebooks/` | `00_setup` … `07_evaluasi_akhir` (satu per checkpoint; hanya memanggil `src/`) |
| `tests/` | uji dengan data tiruan |
| `outputs/` | cache, split, job, hasil (tidak masuk git) |

## Skenario → perintah job

| Skenario | Modul | Konfigurasi |
| --- | --- | --- |
| S1 | `src.train` | `configs/s1.yaml` |
| S2 | `src.embed` lalu `src.train_s2` | `configs/s2.yaml` |
| S3, S4 (SVM/LDA; `_a01` = sensitivitas LASSO 0,01, eksploratif) | `src.radiomics_feats` lalu `src.train_s3` | `configs/s3_*.yaml`, `s4_*.yaml` |
| S5, S6 | `src.pilot_lambda` (S6), `src.train_gan`, `src.train --ratio` | `configs/s5.yaml`, `s6.yaml` |
| S7 | `src.train` / rantai GAN, split `s7` | `configs/s7_s1.yaml`, `s7_s5.yaml`, `s7_s6.yaml` |

Tiap job: `last.pt` ditulis atomik, `DONE` berisi hash konfigurasi, job selesai dilewati, terputus dilanjutkan.

## Menjalankan di mesin GPU

### A. Docker (Linux / Windows+WSL2, driver NVIDIA + NVIDIA Container Toolkit)

```bash
chmod +x scripts/*.sh
# 1. kunci dependensi (internet; sekali) lalu bangun citra
./scripts/lock.sh --gpu            # Windows: scripts\lock.ps1 hanya untuk CPU; pakai WSL untuk --gpu
./scripts/build.sh --gpu
# 2. uji kode dengan data tiruan DULU (menemukan bug sebelum menyentuh data asli)
./scripts/run.sh --gpu python -m pytest -x -q tests
# 3. jalankan semua tahap; berhenti di checkpoint pertama yang butuh keputusan manusia
./scripts/run.sh --gpu python -m src.run_all --stage all
```

Windows PowerShell: `.\scripts\build.ps1 -Gpu`, `.\scripts\run.ps1 -Gpu python -m src.run_all '--stage' all`
(argumen berawalan `--` harus dikutip). Beberapa GPU: ubah `DEVICES = [0, 1]` di sel setup notebook 03, 05, 06.

### B. Tanpa Docker (Colab, Kaggle, server)

```bash
pip install -r requirements.gpu.lock            # setelah lock dibuat di mesin ber-internet
export LARYNX_DATASET=/path/ke/Larynx_CE-NBI_Dataset   # baca-saja
export LARYNX_OUTPUTS=/path/persisten/outputs            # harus persisten (Drive/disk); sesi bisa hilang
python -m pytest -x -q tests
python -m src.run_all --stage all
```

Catatan data: dataset medis, jangan diunggah ke layanan pihak ketiga tanpa pertimbangan (RULES §7).

## Alur checkpoint dan persetujuan manusia

`src.run_all` menjalankan notebook berurutan dan **berhenti** di checkpoint pertama yang belum lulus, lalu mencetak
penghambat. Ulangi perintah yang sama setelah menyelesaikannya (idempoten).

| Checkpoint | Notebook | Isi | Persetujuan manusia yang akan diminta |
| --- | --- | --- | --- |
| CP0 | 00 | lingkungan, benchmark Swin-T dan GAN | `approve decide D2 D3 D4`; REVIEWED |
| CP1 | 01 | audit data (bandingkan Tabel 3.2) | selisih angka dijelaskan; REVIEWED |
| CP2 | 02 | cache, ROI Frangi, split, L1–L3, L5 | `approve roi`; `approve decide D6`; REVIEWED |
| CP3 | 03 | S1–S4 (+ sensitivitas) | `approve decide D7`; REVIEWED |
| CP4 | 04 | pilot λ_roi (D2) | `approve gan-visual CP4`; `approve decide D8`; REVIEWED |
| CP5 | 05 | S5, S6, rasio terbaik dari validasi | `approve gan-visual CP5`; REVIEWED |
| CP6 | 06 | S7 (L5) | REVIEWED |
| CP7 | 07 | perbandingan utama, Tabel 3.3 dan 3.4 | REVIEWED |

```bash
python -m src.approve decide D2 D3 D4 --by NAMA     # konfirmasi keputusan di decisions.yaml
python -m src.approve roi --by NAMA                  # setelah memeriksa PNG ROI (mengunci parameter Frangi)
python -m src.approve gan-visual CP4 --by NAMA       # setelah memeriksa grid sampel GAN
python -m src.approve review CP1 --by NAMA           # tinjauan orang kedua (RULES §3), lalu tag git: git tag cp1
```

`python -m src.run_all --self-review NAMA` menandai REVIEWED otomatis; itu **melewati** ketentuan orang kedua dan tercatat
sebagai self-review. Pakai hanya dengan sadar.

## Hal yang perlu kamu putuskan/isi sebelum menjalankan

1. `gan.total_kimg` (setelah `bench_gan` di CP0). Estimasi jam 20 GAN tercetak di notebook 04.
2. `configs/literature.yaml`: isi Tabel 3.4 dari proposal (hanya angka Wang dkk. dari PRD yang terisi).
3. D4, D7, D8 di `decisions.yaml` (lihat isinya; D7 dan D8 adalah penyimpangan dari PRD yang perlu disetujui kedua penulis).
4. Pembagian tugas Wahyu Andika dan Ahmad Nur Mu'minin: tulis di LOG.md awal tiap fase.

## Batas yang sudah diketahui

- S7: kelompok uji gabungan 4 resolusi; hasil mengukur pergeseran akuisisi bercampur perbedaan populasi, bukan efek resolusi murni.
- Selisih in-distribusi vs pergeseran di S7 tidak berpasangan dan kondisi pelatihannya tidak identik: eksploratif.
- Interval kepercayaan hanya menangkap variasi pemilihan pasien uji, bukan variasi pelatihan model.
- Mask PyRadiomics adalah area pandang, bukan lesi (D7); jumlah fitur dilaporkan apa adanya (Wang dkk.: 104).
- FID memakai Inception-ImageNet torchvision; tidak sebanding dengan angka literatur. Bukti utama adalah kinerja hilir.
- Urutan CTGAN-sebelum-SMOTE di S4 adalah asumsi dari ringkasan PRD; cocokkan dengan notebook Wang dkk. sebelum klaim replikasi.
