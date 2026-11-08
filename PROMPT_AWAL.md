Kamu membantu saya membangun pipeline eksperimen untuk tugas akhir: klasifikasi lesi laring jinak/ganas pada citra CE-NBI, dengan augmentasi GAN berpandu ROI pembuluh dan Swin Transformer. Kita bekerja sampai skenario S2 selesai (checkpoint CP0 sampai CP3), tidak lebih.

## Baca dulu, sebelum menulis kode apa pun
1. `PRD.md` (spesifikasi; bagian 5, 6, 7, 8, 10, 13 paling penting)
2. `RULES.md` (aturan kerja; bagian 1 tentang kejujuran hasil tidak boleh dilanggar)
3. `docs/Referensi_GitHub_dan_Skenario.txt`
Setelah membaca, ringkas dalam 8 baris apa yang akan kamu bangun dan sebutkan hal yang menurutmu ambigu atau salah di PRD. Tunggu jawaban saya sebelum mulai.

## Struktur folder (jangan diubah tanpa bertanya)
- `data/Larynx_CE-NBI_Dataset/` : dataset, BACA-SAJA, jangan ditulis atau dipindah
- `src/` : paket Python (praproses, split, model, pelatihan, evaluasi)
- `notebooks/` : satu notebook per checkpoint (`00_setup`, `01_audit`, `02_praproses_split`, `03_baseline`)
- `configs/` : konfigurasi (seed, hiperparameter), tidak ada angka ajaib di kode
- `outputs/` : cache, checkpoint, hasil job

## Tugas (kerjakan berurutan, berhenti di tiap checkpoint)
- **CP0:** `Dockerfile`, `requirements.lock`, JupyterLab di dalam citra, uji lingkungan, skrip benchmark waktu satu epoch Swin-T.
- **CP1:** audit data dari gambar lokal: jumlah pasien dan citra per kelas, resolusi per pasien, pemeriksaan syarat S7. Bandingkan dengan Tabel 3.2 proposal (150 jinak / 60 ganas pasien, 7.657 / 3.487 citra). Laporkan selisih apa adanya.
- **CP2:** praproses lima langkah, cache memmap uint8, split per pasien (StratifiedGroupKFold, 5 lipatan x 2 ulangan), uji otomatis kebocoran L1 sampai L3.
- **CP3:** S1 (Swin-T tiny pralatih, weighted sampling, AdamW 1e-4, wd 0,05, batch 32, maks 30 epoch, henti dini sabar 5) dan S2 (embedding Swin beku, SMOTE, kepala linear). Evaluasi per pasien: AUROC, F1, recall, bootstrap persentil bertingkat 2.000 ulangan.

## Aturan keras
- Pelatihan dijalankan sebagai skrip (`python -m src.train --config ...`) yang diluncurkan dari notebook, bukan di dalam sel. Setiap job punya `last.pt` (ditulis atomik) dan berkas `DONE`; job yang sudah `DONE` dilewati.
- Semua split per pasien. Jangan pernah memakai data uji untuk memilih apa pun. Jangan ganti seed atau buang lipatan karena hasil jelek.
- Notebook hanya memanggil fungsi di `src/`. Sel terakhir tiap notebook menjalankan pemeriksaan checkpoint dan menulis `manifest.json`.
- Jangan membuat atau mengarang angka hasil. Kalau sesuatu tidak bisa dijalankan (tidak ada GPU, bobot tidak bisa diunduh), katakan itu, jangan diganti dengan versi palsu.
- Jalankan dan uji kode dengan data tiruan kecil di CPU sebelum menyentuh dataset asli. Laporkan jelas bahwa uji ini hanya membuktikan kode berjalan.
- Kalau PRD dan RULES bertentangan, PRD menang. Kalau kamu menemukan kesalahan di PRD, laporkan, jangan diam-diam diubah.

## Cara bekerja dengan saya
- Bahasa Indonesia, santai tapi sopan. Pakai "saya/kamu", jangan "lo/gue".
- Jujur dan kritis: kalau menurutmu keputusan saya keliru, jelaskan alasannya, jangan sekadar setuju.
- Berikan kode lengkap yang bisa disalin dan dijalankan, bukan potongan.
- Berhenti di setiap checkpoint, tunjukkan hasil pemeriksaannya, dan tunggu persetujuan saya sebelum lanjut.
- Satu perubahan per commit; pesan commit menyebut ID persyaratan (misal FR-05).

## Informasi yang belum pasti (tanyakan ke saya, jangan menebak)
- Perangkat keras pelatihan belum ditetapkan (GPU lokal, lab, atau Colab/Kaggle).
- Pembagian tugas antara Wahyu Andika dan Ahmad Nur Mu'minin belum ditentukan.
