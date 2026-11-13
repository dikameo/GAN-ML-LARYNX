# Cara menjalankan Versi 2

Panduan langkah demi langkah untuk menjalankan notebook 00 sampai 07 di Windows, tanpa Docker. Untuk apa yang berubah dari Versi 1, lihat [perbandingan.md](../perbandingan.md).

> **Status:** notebook belum pernah dijalankan dengan torch, GPU, dan dataset penuh. Bagian 5 menjelaskan uji kecil yang sebaiknya dijalankan lebih dulu. Bila ada error, catat teksnya apa adanya dan jangan mengubah seed atau membuang lipatan agar berjalan.

## 1. Prasyarat

| Kebutuhan | Keterangan |
|---|---|
| GPU | NVIDIA RTX 3070 Ti 8 GB (atau lain dengan VRAM sejenis) dengan driver NVIDIA terbaru |
| Python | **3.11** (sudah terpasang di `C:\Users\Huawei\AppData\Local\Programs\Python\Python311`) |
| Disk kosong | perkiraan kasar 30 GB di `Versi 2/outputs` (cache 2,2 GB + ROI 0,7 GB + sintetis dan checkpoint); cek `outputs/` secara berkala |
| RAM | 16 GB disarankan (data latih GAN dimuat ke RAM, sekitar 1,5 GB per GAN) |
| Internet | pada jalan pertama, untuk bobot Swin-T, Inception-v3 (FID), dan LPIPS |
| Dataset | sudah ada di `Training/data/Larynx_CE-NBI_Dataset` (jalur diatur di `configs/base.yaml`) |
| Kompiler C++ | bisa diperlukan untuk PyRadiomics (notebook 03); lihat bagian 8 |

Docker **tidak** diperlukan. Docker Desktop boleh ditutup agar tidak memakan RAM.

## 2. Menyiapkan lingkungan (sekali saja)

Buka PowerShell di folder `Versi 2`:

```powershell
cd "d:\SEMESTER 7\MACHINE LEARNING\Training\Versi 2"
& "C:\Users\Huawei\AppData\Local\Programs\Python\Python311\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install transformers numpy pandas scipy pyyaml psutil matplotlib tqdm jupyterlab ipykernel
python -m ipykernel install --user --name larynx-v2 --display-name "Larynx V2"
```

Bila `Activate.ps1` ditolak, jalankan sekali `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

Pustaka lain (scikit-learn, scikit-image, PyRadiomics, dst.) dipasang oleh tiap notebook lewat sel `INSTALL = True`, atau dengan `pip install` manual yang daftarnya ada di sel pertama tiap notebook.

Cek GPU terbaca:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Hasilnya harus `True` dan nama GPU. Jika `False`, lihat bagian 8.

## 3. Membuka notebook

```powershell
jupyter lab
```

atau buka folder `Versi 2` di VS Code lalu pilih kernel **Larynx V2** (`.venv`). Selalu jalankan notebook dari **atas ke bawah**, satu per satu, dengan **Restart Kernel & Run All** atau sel demi sel.

Aturan umum:
- Satu notebook = satu checkpoint (CP0 sampai CP7). Jangan lanjut ke notebook berikutnya sebelum sel terakhir (`finalize_cpN`) berstatus **LULUS**.
- Item berstatus `MENUNGGU` adalah keputusan manusia. Itu bukan error.
- Notebook berikutnya membaca hasil notebook sebelumnya dari `outputs/`, jadi urutan 00 → 07 wajib.
- Menutup Jupyter di tengah pelatihan aman: jalankan ulang sel peluncuran dan job yang sudah `DONE` dilewati, sedangkan yang terputus lanjut dari `last.pt`.

## 4. Urutan dan isi tiap notebook

| No. | Notebook | Yang dilakukan | Yang harus Anda lakukan |
|---|---|---|---|
| 00 | `00_setup` | Uji lingkungan, benchmark Swin-T dan GAN (menyapu `micro_batch`), estimasi jam | Ubah `configs/base.yaml`: `gan.micro_batch` dan `gan.total_kimg` sesuai hasil; catat di `LOG.md`; konfirmasi D1 sampai D4 |
| 01 | `01_audit` | Audit data dari gambar lokal, bandingkan dengan Tabel 3.2 | Tinjau tabel; jelaskan selisih (bila ada) di `LOG.md` |
| 02 | `02_praproses_split` | Cache, peta ROI Frangi, split per pasien, uji kebocoran tiruan | Periksa gambar `roi_check_*.png`, lalu kunci ROI (`approve_roi`) |
| 03 | `03_baseline` | S1 sampai S4 | Menunggu; S1 paling lama |
| 04 | `04_pilot_gan` | Pilot λ_roi (4 GAN pada 1 lipatan) | Periksa grid sampel, lalu `approve_flag(...)` |
| 05 | `05_s5_s6` | S5 dan S6: GAN + 3 rasio per lipatan | Menunggu (2 sampai 3 malam menurut `revisi.md`; ukur sendiri) |
| 06 | `06_s7` | S7 (pergeseran resolusi) | Menunggu |
| 07 | `07_evaluasi_akhir` | Perbandingan, Tabel 3.3 dan 3.4 | Tinjau; angka tidak diubah setelah CP7 |

### Notebook 00: bagian terpenting sebelum yang lain
1. Jalankan semua sel. Sel benchmark GAN butuh CUDA.
2. Baca tabel sapuan `micro_batch` (kolom `status`, `peak_gpu_gb`). `OOM` berarti ukuran itu tidak muat.
3. Baca `rekomendasi` dan `estimasi.jam_total_GAN` di `outputs/cp0/benchmark_gan.json`.
4. Edit `configs/base.yaml` dengan tangan: `gan.micro_batch` (nilai rekomendasi) dan `gan.total_kimg` (sesuai tenggat).
5. Tulis keputusan itu di `LOG.md` **sebelum** menjalankan notebook 04. Mengubah konfigurasi mengubah hash; `last.pt` lama otomatis diabaikan.

### Sel persetujuan (nama bawaan `None`)
Isi nama hanya setelah benar-benar meninjau, lalu jalankan sel itu:

| Sel | Notebook | Efek |
|---|---|---|
| `REVIEWER` → `approve_review` | semua | Menandai tinjauan orang kedua (`REVIEWED`); RULES §3 meminta orang selain pengerjanya |
| `DECIDER` → `approve_decisions` | 00 | Mengubah status D1 sampai D4 di `decisions.yaml` |
| `ROI_APPROVER` → `approve_roi` | 02 | Mengubah `preprocess.roi.locked: true` di `base.yaml` |
| `VISUAL_APPROVER` | 04 | Menandai grid pilot lolos |
| `COLLAPSE_APPROVER` | 05 | Menandai tidak ada mode collapse |

## 5. Uji kecil sebelum pelatihan sungguhan (disarankan)

Karena kode belum pernah dijalankan dengan torch dan GPU, uji dulu seluruh alur pada skala kecil di folder keluaran terpisah.

1. Set folder keluaran sebelum membuka Jupyter (variabel ini menimpa `paths.outputs`):
   ```powershell
   $env:LARYNX_OUTPUTS = "d:\SEMESTER 7\MACHINE LEARNING\Training\Versi 2\outputs_uji"
   jupyter lab
   ```
2. **Sementara** ubah di `configs/base.yaml`: `train.max_epochs: 1`, `gan.total_kimg: 2`, `gan.snapshot_kimg: 1`, `gan.fid_samples: 50`, `gan.fid_fake_samples: 200`, `gan.quality.lpips_pairs: 32`, `pilot.lambdas: [0.5]`.
3. Jalankan notebook 00 sampai 05 dengan `REPS, FOLDS = [0], [0]` (sel konfigurasi 03 dan 05). Sel `finalize_cpN` akan melaporkan `PARSIAL`/`GAGAL`; itu wajar pada uji ini.
4. Putuskan sambungan di tengah (hentikan sel) lalu jalankan lagi, untuk menguji lanjut dari `last.pt`.
5. **Kembalikan** semua nilai `base.yaml` ke semula, tutup Jupyter, hapus `outputs_uji`, dan buka Jupyter lagi **tanpa** `LARYNX_OUTPUTS`.

Uji ini hanya memastikan kode tidak error. Angka hasilnya tidak dipakai dan tidak dicatat sebagai hasil.

## 6. Menjalankan sungguhan

1. Pastikan `base.yaml` sudah dikembalikan dan keputusan bagian 4 (notebook 00) tercatat.
2. Tutup aplikasi lain yang memakai GPU. Tampilan Windows sudah memakai sekitar 0,8 sampai 1 GB VRAM.
3. Jalankan notebook 00 sampai 07 berurutan.
4. Laptop: colok daya, matikan mode tidur dan hibernasi. Pelatihan GAN berjam-jam.
5. Untuk membatasi sementara, ubah `REPS, FOLDS` di sel konfigurasi (hanya membatasi job yang diluncurkan, bukan memilih hasil). Tabel yang dihasilkan otomatis ditandai `PARSIAL` dan tidak memenuhi gerbang.

## 7. Letak hasil

Semua di `Versi 2/outputs/`:

| Lokasi | Isi |
|---|---|
| `cp0/` sampai `cp7/` | Laporan checkpoint (`cpN_check.json`) dan `manifest.json` (hanya bila lulus) |
| `cache/` | Cache citra 256×256 dan peta ROI (memmap) |
| `splits/` | `splits.json`, `fold_summary.csv`, split S7 |
| `jobs/<skenario>/rep0_foldK/` | `last.pt`, `best.pt`, `DONE`, prediksi, `run_info.json`, `patients.json` |
| `jobs/launch.log` | Catatan tiap job beserta lama dan percobaan |
| `results/` | `results_*.csv`, `tabel_3_3.csv/.md`, `tabel_3_4.csv`, `comparisons_all.csv` |
| `_workers/` | Modul sementara yang dibuat notebook (bukan sumber kode) |

## 8. Masalah yang mungkin muncul

| Gejala | Penyebab dan tindakan |
|---|---|
| `torch.cuda.is_available()` = False | Torch CPU yang terpasang, atau driver lama. Pasang ulang dengan `--index-url https://download.pytorch.org/whl/cu128` dan perbarui driver NVIDIA |
| `CUDA out of memory` | Turunkan `gan.micro_batch` (16 → 8 → 4) untuk GAN, atau `train.batch_size` untuk Swin-T. Catat di `LOG.md` **sebelum** menjalankan. Job OOM diulang sekali otomatis; gagal dua kali menghentikan antrean |
| Job gagal dua kali | Antrean berhenti dengan pesan; catat penyebabnya di `LOG.md` (RULES §4), perbaiki, lalu jalankan ulang sel yang sama |
| DataLoader macet atau error `spawn` di Windows | Set `loader.num_workers: 0` di `base.yaml` untuk uji, lalu coba naikkan lagi |
| Memori RAM penuh saat praproses | Turunkan `preprocess.workers` (default 2) |
| `pip install pyradiomics` gagal | Pasang Microsoft C++ Build Tools, atau cari roda siap pakai untuk Python 3.11. Tanpa ini S3 dan S4 tidak berjalan dan CP3 tetap `GAGAL` |
| Unduhan bobot (Swin-T, Inception, LPIPS) gagal | Periksa internet; jalankan ulang sel. Bobot Swin-T disimpan di `outputs/hf_cache` |
| `FileNotFoundError ... audit_images.csv` | Notebook 01 belum dijalankan atau `outputs/` berbeda (cek `LARYNX_OUTPUTS`) |
| Item `MENUNGGU` di `finalize_cpN` | Keputusan manusia belum dicatat; jalankan sel persetujuan yang sesuai setelah meninjau |
| `last.pt dari konfigurasi lain diabaikan` | `base.yaml` berubah sejak job dimulai (hash berbeda); job dimulai dari awal. Kembalikan nilainya bila tidak disengaja |

## 9. Aturan yang tidak boleh dilanggar

- Jangan mengganti seed, membuang lipatan, atau memilih parameter berdasarkan hasil uji (RULES §3 dan §4).
- Setiap perubahan `configs/*.yaml` ditulis di `LOG.md` sebelum dijalankan.
- Jangan menulis apa pun ke `Training/data/` (tidak dapat dipasang baca-saja di Windows).
- Jangan menjalankan dua job GPU sekaligus pada satu GPU 8 GB.
- Setelah CP7, angka tidak diubah; koreksi berarti mengulang dari checkpoint yang terdampak.
