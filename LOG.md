# LOG.md – Buku catatan eksperimen

Satu baris per kejadian: tanggal, siapa, apa yang dijalankan, hasil, keputusan. Ditulis saat kejadian (RULES.md bagian 4).

| Tanggal | Siapa | Kejadian | Hasil | Keputusan |
| --- | --- | --- | --- | --- |
| 2026-11-08 | Peneliti + asisten AI | Pembacaan PRD, RULES, referensi; daftar ambiguitas PRD | 16 butir dilaporkan | D1: pelatihan di CPU (laptop, tanpa GPU NVIDIA, RAM 7,4 GB, 8 inti logis). Cakupan dibatasi CP0–CP3 dengan S1 dan S2; S3, S4, D3, benchmark StyleGAN2-ADA ditunda. Normalisasi memakai mean/std ImageNet tetap, bukan statistik per lipatan (ubah PRD §5 langkah 4). Swin-T lewat Hugging Face transformers. |
| 2026-11-08 | Asisten AI | `git init`, `.gitignore`, `LOG.md` | Repositori dibuat | Pembagian tugas Wahyu dan Ahmad ditunda sampai awal Fase 1 |
| 2026-11-08 | Asisten AI | Benchmark Swin-T di laptop CPU (kontainer 2 CPU, 3,8 GiB, 1 thread torch; batch 8, 3 langkah terukur, tensor acak) | 0,068 citra/dtk; ~31 jam/epoch; ~928 jam per job S1 (30 epoch) | D1: pelatihan dipindah ke mesin lain; S1 tidak layak di laptop. benchmark.json laptop disimpan sebagai referensi |
| 2026-11-08 | Asisten AI | pytest data tiruan (31 uji) untuk CP0-CP3 versi awal | 31 lulus, 181 dtk (hanya membuktikan kode berjalan) | - |
| 2026-11-08 | Asisten AI | Menulis kode S3-S7, GAN, CP4-CP7, notebook 00-07, run_all, approve (TIDAK dijalankan atas permintaan) | hanya diperiksa sintaksnya (ast.parse) | Wajib jalankan pytest di mesin tujuan sebelum data asli; lock dependensi perlu dibangun ulang |
