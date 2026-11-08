# RULES.md – Aturan Kerja Proyek

Berkas ini berisi kebiasaan kerja. Aturan yang menentukan validitas hasil (L1–L6, gerbang CP0–CP8) ada di PRD.md dan tidak diubah dari sini. Bila dua aturan bertentangan, PRD menang. Satu aturan hanya ditulis di satu tempat. Setiap perubahan di berkas ini dicatat di bagian paling bawah.

## 1. Kejujuran hasil (paling penting)
- Data uji tidak dibuka, dilihat, atau dipakai untuk memutuskan apa pun sebelum konfigurasi dikunci. Keputusan (rasio, λ_roi, ambang, arsitektur) memakai validasi di dalam lipatan latih.
- Dilarang mengganti seed, membuang lipatan, atau menjalankan ulang job karena hasilnya jelek. Job gagal teknis boleh diulang dengan seed yang sama; hasil jelek tetap dilaporkan.
- Perbandingan utama ditetapkan di PRD (S6 lawan S1 dan S2). Perbandingan lain wajib diberi label "eksploratif" di tabel dan teks.
- Hasil negatif dilaporkan sama lengkap dengan hasil positif.
- Setelah CP7, angka tidak diubah. Koreksi berarti mengulang dari checkpoint yang terdampak (lihat PRD bagian 13).

## 2. Docker
- Semua perintah pipeline dijalankan di dalam kontainer, bukan di mesin host.
- Tidak ada `pip install` manual di dalam kontainer. Tambah pustaka lewat `requirements.lock`, lalu bangun ulang citra.
- Data dipasang baca-saja. Tidak ada yang menulis ke folder dataset.

## 3. Checkpoint
- Tahap berikutnya tidak dimulai sebelum pemeriksaan checkpoint tahap sebelumnya lolos.
- Job yang punya berkas `DONE` tidak dijalankan ulang kecuali konfigurasinya berubah.
- Setelah checkpoint tahap lolos: buat `manifest.json` dan tag git `cpN`.
- Satu orang menjalankan pemeriksaan, orang lain meninjau hasilnya sebelum tag dibuat.

## 4. Eksperimen dan pencatatan
- Seed ada di berkas konfigurasi, bukan di kode.
- Setiap job mencatat: konfigurasi, seed, hash data, hash commit, digest citra Docker.
- Buku catatan eksperimen (`LOG.md`), satu baris per kejadian: tanggal, siapa, apa yang dijalankan, hasil, keputusan. Ditulis saat itu juga, bukan belakangan.
- Tidak ada angka di laporan yang tidak bisa ditelusuri ke satu job.
- Job yang gagal dua kali berturut-turut dihentikan, penyebabnya dicatat, baru dilanjutkan.

## 5. Kode dan git
- Satu pustaka praproses dipakai pipeline dan website.
- Perubahan perilaku lewat berkas konfigurasi, bukan edit kode.
- Satu perubahan, satu commit; pesan commit menyebut ID persyaratan (misal FR-06).
- Tidak ada force push ke cabang utama. Kerja di cabang sendiri, gabungkan setelah ditinjau satu orang lagi.
- `.gitignore` memuat: dataset, cache, bobot model, citra sintetis, hasil besar. Yang masuk git hanya kode, konfigurasi, `manifest.json`, dan tabel hasil.
- Kode dari asisten AI atau sumber luar dijalankan dan diuji dulu sebelum dipakai. Angka dan klaim tidak ditulis dari ingatan; ambil dari keluaran job atau dari sumber yang dibuka.

## 6. Kerja berdua
- Pembagian tugas ditulis di LOG.md di awal tiap fase (siapa memegang apa). Tidak ada dua orang mengedit berkas yang sama di waktu yang sama.
- Perubahan pada PRD yang mengubah ruang lingkup, skenario, atau metrik disetujui kedua penulis dan dicatat di catatan perubahan PRD dengan tanggal dan alasan.
- Keputusan terbuka D1–D6 yang sudah diputuskan ditulis di PRD, bukan hanya di chat.

## 7. Cadangan dan data
- Checkpoint, `manifest.json`, dan hasil job disalin ke penyimpanan kedua (cloud atau disk lain) setiap kali satu checkpoint tahap lolos. Sesi cloud bisa hilang kapan saja.
- Dataset memakai lisensi CC BY 4.0, tetapi tetap data medis: tidak diunggah ke layanan pihak ketiga, dan website tidak menyimpan citra unggahan (WR-08).
- Hapus citra sintetis lama hanya setelah checkpoint yang memakainya ditandai selesai, dan hanya bila generatornya masih tersimpan.

## Catatan perubahan
- 2026-11-08: dibuat.
- 2026-11-08: ditambah bagian kejujuran hasil, pencatatan, git, kerja berdua, dan cadangan.
