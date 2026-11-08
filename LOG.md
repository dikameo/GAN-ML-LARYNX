# LOG.md – Buku catatan eksperimen

Satu baris per kejadian: tanggal, siapa, apa yang dijalankan, hasil, keputusan. Ditulis saat kejadian (RULES.md bagian 4).

| Tanggal | Siapa | Kejadian | Hasil | Keputusan |
| --- | --- | --- | --- | --- |
| 2026-11-08 | Peneliti + asisten AI | Pembacaan PRD, RULES, referensi; daftar ambiguitas PRD | 16 butir dilaporkan | D1: pelatihan di CPU (laptop, tanpa GPU NVIDIA, RAM 7,4 GB, 8 inti logis). Cakupan dibatasi CP0–CP3 dengan S1 dan S2; S3, S4, D3, benchmark StyleGAN2-ADA ditunda. Normalisasi memakai mean/std ImageNet tetap, bukan statistik per lipatan (ubah PRD §5 langkah 4). Swin-T lewat Hugging Face transformers. |
| 2026-11-08 | Asisten AI | `git init`, `.gitignore`, `LOG.md` | Repositori dibuat | Pembagian tugas Wahyu dan Ahmad ditunda sampai awal Fase 1 |
