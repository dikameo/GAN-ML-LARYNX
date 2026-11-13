# Parameter Konfigurasi

Tabel semua parameter di [base.yaml](base.yaml) dan penimpaannya di konfigurasi skenario. Nilai diambil langsung dari berkas YAML.
Keputusan terbuka dan perubahan PRD ada di [decisions.yaml](decisions.yaml).

**Aturan:** parameter tidak boleh diubah setelah melihat hasil uji (RULES §1). Mengubah parameter mengubah hash konfigurasi,
sehingga job terkait dijalankan ulang dari awal. Perubahan yang memengaruhi hasil dicatat di LOG.md.

**Kolom Sumber:**
- **PRD** = ditetapkan di PRD.md (bagian atau ID dicantumkan).
- **Keputusan** = diputuskan dalam sesi kerja 2026-11-08 (tercatat di LOG.md atau decisions.yaml).
- **ROI_Figure** = nilai awal dari `data/.../ROI_Figure/roi_figure.py` (skrip Frangi proposal).
- **Implementasi** = pilihan saya saat menulis kode, **bukan dari PRD dan belum disetel**. Anggap sebagai titik awal yang boleh
  disetel sebelum eksperimen utama, dengan alasan tercatat; jangan disetel dengan melihat hasil uji.

---

## 1. Umum

| Parameter | Nilai | Arti | Sumber |
| --- | --- | --- | --- |
| `seed` | 42 | Seed dasar. Seed tiap job diturunkan dari (seed, skenario, ulangan, lipatan) dan dicatat di `run_info.json`. | RULES §4 |
| `splits_name` | `main` | Berkas split yang dipakai: `main` (`splits.json`) atau `s7` (`splits_s7.json`). | Implementasi |
| `paths.dataset` | `/data/Larynx_CE-NBI_Dataset` | Lokasi dataset (baca-saja). Dapat diganti dengan `LARYNX_DATASET`. | RULES §2 |
| `paths.outputs` | `/outputs` | Cache, split, job, hasil. Dapat diganti dengan `LARYNX_OUTPUTS`. | – |
| `paths.hf_cache` | `/outputs/hf_cache` | Cache bobot Hugging Face. Dapat diganti dengan `LARYNX_HF_CACHE`. | – |

## 2. Model (Swin-T)

| Parameter | Nilai | Arti | Sumber |
| --- | --- | --- | --- |
| `model.hf_name` | `microsoft/swin-tiny-patch4-window7-224` | Bobot Swin-T tiny pralatih ImageNet. | PRD §14, FR-05 |
| `model.num_classes` | 2 | Jinak (0) dan ganas (1). | PRD §1 |
| `model.input_size` | 224 | Ukuran masukan klasifikator. | PRD §5 langkah 3 |
| `model.pretrained` | true | Muat bobot pralatih. `false` hanya untuk uji tiruan. | FR-05 |
| `model.arch_overrides` | null | Mengecilkan arsitektur untuk uji tiruan. | Implementasi |

## 3. Praproses dan cache (FR-02, FR-03)

| Parameter | Nilai | Arti | Sumber |
| --- | --- | --- | --- |
| `preprocess.cache_size` | 256 | Ukuran citra di cache; klasifikator di-resize ke 224 saat dibaca. | PRD §5 |
| `preprocess.normalize` | `imagenet` | Normalisasi mean/std ImageNet tetap, diterapkan saat dibaca (tidak disimpan di cache). Mengubah PRD §5 langkah 4. | Keputusan |
| `preprocess.imagenet_mean` / `imagenet_std` | [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225] | Statistik ImageNet. | Standar |
| `preprocess.workers` | 2 | Jumlah proses untuk membangun cache. Naikkan di mesin dengan banyak inti. | Implementasi |
| `preprocess.fov.threshold` | 15 | Ambang intensitas piksel untuk area pandang. | ROI_Figure |
| `preprocess.fov.min_area_fraction` | 0.05 | Di bawah ini deteksi area pandang dianggap gagal; seluruh citra dipakai dan ditandai `fov_fallback`. | Implementasi |
| `preprocess.quality.filter_enabled` | false | Metrik kualitas (buram, silau, gelap) dihitung dan dilaporkan; **tidak ada citra dibuang**. | Keputusan |
| `preprocess.quality.saturation_level` | 250 | Tingkat piksel jenuh untuk metrik silau. | Implementasi |
| `preprocess.quality.dark_level` | 30 | Tingkat piksel gelap untuk metrik gelap. | Implementasi |
| `preprocess.roi.locked` | false | Parameter ROI dikunci setelah pemeriksaan visual 50 citra. Diubah lewat `python -m src.approve roi`. | FR-03 |
| `preprocess.roi.filter` | `frangi` | Filter pembuat ROI. Metrik pembuluh GAN wajib memakai filter lain (L6). | PRD §10 L6 |
| `preprocess.roi.sigmas` | [1, 1.5, 2, 3] | Skala filter Frangi. | ROI_Figure |
| `preprocess.roi.alpha` / `beta` | 0.5 / 0.5 | Parameter bentuk Frangi. | ROI_Figure |
| `preprocess.roi.gamma` | null | Otomatis (skimage). | ROI_Figure |
| `preprocess.roi.threshold` | 0.10 | Ambang biner peta ROI (tetap). | ROI_Figure |
| `preprocess.roi.min_obj` | 30 | Buang komponen ROI lebih kecil dari ini (piksel). | ROI_Figure |
| `preprocess.roi.fov_erode` | 4 | Erosi area pandang pada 256 px (nilai asli 15 px pada ~1000 px, diskalakan). | Implementasi |
| `preprocess.roi.green_blur` | 1.0 | Sigma blur Gauss pada kanal hijau. | ROI_Figure |
| `preprocess.roi.norm_percentile` | 99.5 | Persentil normalisasi respons Frangi. | ROI_Figure |
| `preprocess.roi_examples.n` / `seed` | 50 / 42 | Jumlah dan seed pemilihan citra untuk pemeriksaan visual ROI; ID pasiennya dicatat (D6). | FR-03, D6 |

## 4. Audit data (FR-01)

| Parameter | Nilai | Arti | Sumber |
| --- | --- | --- | --- |
| `audit.reference_tabel_3_2.patients` | jinak 150, ganas 60 | Acuan jumlah pasien. | PRD §5 |
| `audit.reference_tabel_3_2.images` | jinak 7657, ganas 3487 | Acuan jumlah citra. | PRD §5 |
| `audit.reference_kiani.patients` | jinak 151, ganas 59 | Hanya untuk perbandingan. | PRD §5 |
| `audit.reference_kiani.images` | jinak 7715, ganas 3442 | Hanya untuk perbandingan. | PRD §5 |
| `audit.s7_min_patients_per_class` | 10 | Syarat minimal pasien per kelas per kelompok S7. | PRD §6 |

## 5. Split (FR-04)

| Parameter | Nilai | Arti | Sumber |
| --- | --- | --- | --- |
| `splits.n_splits` | 5 | Jumlah lipatan. | FR-04 |
| `splits.n_repeats` | **1 (Versi 2)**; Versi 1: 2 | Jumlah ulangan validasi silang (total 10 lipatan). Mengurangi ke 1 diizinkan proposal untuk skenario GAN, harus diputuskan **sebelum** CP2 dan ditulis di hasil. | FR-04 |
| `splits.val_fraction` | 0.15 | Porsi pasien latih yang dijadikan validasi. | FR-04 |

## 6. Komputasi dan pemuatan data

| Parameter | Nilai | Arti | Sumber |
| --- | --- | --- | --- |
| `compute.torch_threads` | null | Jumlah thread torch; null = semua CPU yang terlihat. | Implementasi |
| `compute.amp` | true | Presisi campuran fp16, hanya aktif bila ada CUDA (klasifikator; GAN selalu fp32). | Implementasi |
| `loader.num_workers` | `auto` | `auto` = 0 di CPU, min(6, CPU−1) di GPU; atau angka. | Implementasi |

## 7. Augmentasi klasik dan pelatihan Swin-T (S1, S5, S6, S7)

| Parameter | Nilai | Arti | Sumber |
| --- | --- | --- | --- |
| `augment.crop_scale` | [0.8, 1.0] | Rentang luas potongan acak. | Keputusan |
| `augment.crop_ratio` | [0.9, 1.1] | Rentang rasio aspek potongan. | Implementasi |
| `augment.rotation_deg` | 10 | Rotasi acak maksimum (derajat). | Implementasi |
| `augment.hflip` / `vflip` | true / true | Balik horizontal dan vertikal. | Implementasi |
| (tidak ada color jitter) | – | Warna NBI bersifat diagnostik. | Keputusan |
| `train.optimizer` | `adamw` | Optimizer. | FR-05 |
| `train.lr` | 1.0e-4 | Laju belajar, konstan (tanpa jadwal). | FR-05, Keputusan |
| `train.weight_decay` | 0.05 | Peluruhan bobot. | FR-05 |
| `train.batch_size` | 32 | Ukuran batch. | FR-05 |
| `train.max_epochs` | 30 | Epoch maksimal. | FR-05 |
| `train.early_stop_patience` | 5 | Henti dini bila tidak membaik 5 epoch. | FR-05 |
| `train.early_stop_metric` | `val_loss_balanced` | Rata-rata loss per kelas pada pasien validasi. | Keputusan |
| `train.min_delta` | 0.0 | Perbaikan minimal agar dihitung membaik. | Implementasi |
| `train.weighted_sampling` | true | Pengambilan sampel berbobot kebalikan frekuensi kelas. | FR-05 |
| `train.checkpoint_every_epochs` | 1 | Frekuensi penulisan `last.pt`. | FR-12 |

## 8. S2: embedding beku + SMOTE + kepala linear

| Parameter | Nilai | Arti | Sumber |
| --- | --- | --- | --- |
| `s2.embed_batch_size` | 32 | Batch saat menghitung embedding. | Implementasi |
| `s2.c_grid` | [0.01, 0.1, 1.0, 10.0] | Kandidat regularisasi regresi logistik; dipilih dari val loss pasien validasi (L4). | Implementasi |
| `s2.smote_k` | 5 | Tetangga SMOTE (dikecilkan otomatis bila kelas terlalu kecil). | Implementasi |
| `s2.smote_strategy` | `auto` | Seimbangkan kelas 1:1. | Keputusan |
| `s2.max_iter` | 1000 | Iterasi maksimal regresi logistik. | Implementasi |

## 9. S3 dan S4: replikasi Wang dkk. (fitur radiomik)

| Parameter | Nilai | Arti | Sumber |
| --- | --- | --- | --- |
| `radiomics.mask` | `fov` | Mask ekstraksi: area pandang (karena tidak ada mask lesi). Alternatif `roi`. Perlu konfirmasi D7. | D7 |
| `radiomics.channel` | `gray` | Kanal citra untuk fitur (`gray` atau `green`). | Implementasi |
| `radiomics.bin_width` | 25 | Lebar bin tekstur PyRadiomics. | Implementasi (default umum) |
| `radiomics.workers` | null | Jumlah proses; null = semua CPU. | Implementasi |
| `radiomics.chunk` | 256 | Citra per potongan progres. | Implementasi |
| `s3.model` | `svm` | `svm` (linear) atau `lda`; ditimpa di `s3_*.yaml`. | FR-08 |
| `s3.lasso_alpha` | 0.001 | Alpha LASSO. Proposal 0,001 (utama); kode Wang dkk. 0,01 (sensitivitas). | D3 |
| `s3.lasso_max_iter` | 10000 | Iterasi maksimal LASSO. | Implementasi |
| `s3.svm_C` | 1.0 | Regularisasi SVM linear (tetap). | Implementasi |
| `s3.smote_k` | 5 | Tetangga SMOTE. | Implementasi |
| `s3.use_ctgan` | false | `true` pada `s4_*.yaml` (S4). | PRD §6 |
| `s4_ctgan.epochs` | 100 | Epoch CTGAN. | Referensi Wang dkk. |
| `s4_ctgan.batch_size` | 50 | Batch CTGAN (kelipatan 10). | Referensi Wang dkk. |
| `s4_ctgan.synth_fraction` | 0.5 | Sampel sintetis = 50% data latih. | PRD §6 |
| `s4_ctgan.cuda` | `auto` | Pakai GPU bila ada. | Implementasi |

## 10. GAN kondisional (S5, S6, S7)

Implementasi ringkas bergaya StyleGAN2, bukan kode resmi (lihat D8). **Semua hiperparameter GAN di bawah, kecuali yang ditandai PRD,
adalah pilihan implementasi yang belum disetel.**

| Parameter | Nilai | Arti | Sumber |
| --- | --- | --- | --- |
| `gan.resolution` | 256 | Resolusi citra GAN. | FR-06 |
| `gan.z_dim` / `w_dim` | 256 / 256 | Dimensi laten dan gaya. | Implementasi |
| `gan.map_layers` | 4 | Lapisan jaringan pemetaan. | Implementasi |
| `gan.channels` | 4:256, 8:256, 16:256, 32:256, 64:128, 128:64, 256:32 | Kanal generator per resolusi. | Implementasi |
| `gan.d_channels` | 256:32, 128:64, 64:128, 32:256, 16:256, 8:256, 4:256 | Kanal diskriminator per resolusi. | Implementasi |
| `gan.batch_size` | 32 | Batch GAN. | Implementasi |
| `gan.lr_g` / `lr_d` | 2.0e-4 / 2.0e-4 | Laju belajar G dan D. | Implementasi |
| `gan.betas` | [0.0, 0.99] | Beta Adam. | Implementasi |
| `gan.r1_gamma` / `r1_interval` | 10.0 / 16 | Regularisasi R1 dan frekuensinya. | Implementasi |
| `gan.ema_kimg` | 10.0 | Setengah-umur EMA generator. | Implementasi |
| **`gan.total_kimg`** | **500** | **SEMENTARA.** Total citra latih (ribuan). Tetapkan setelah `python -m src.bench_gan`. | PRD §8 Fase 0 |
| `gan.snapshot_kimg` | 25 | Jarak `last.pt` dan pemantauan FID. | FR-06 |
| `gan.fid_samples` | 500 | Jumlah sampel untuk FID pemantauan. | Implementasi |
| `gan.lambda_roi` | 0.0 | Bobot loss D_roi. S5 = 0; S6 = null (dibaca dari `lambda_selection.json`). | FR-06, D2 |
| `gan.ada.enabled` | true | Augmentasi adaptif (versi ringan). | FR-06 |
| `gan.ada.target` / `kimg` / `interval` | 0.6 / 500 / 4 | Target r_t, kecepatan penyesuaian p, interval pembaruan. | Implementasi |
| `gan.vesselness.sigmas` | [1.0, 1.5, 2.0, 3.0] | Skala Frangi diferensiabel untuk D_roi. | Implementasi |
| `gan.vesselness.beta` / `c` / `blur` | 0.5 / 0.08 / 1.0 | Parameter bentuk, ambang struktur, blur awal. | Implementasi |
| `gan.vesselness.fov_thr` / `fov_erode` | −0.9 / 4 | Batas dan erosi area pandang pada citra [−1, 1]. | Implementasi |
| `gan.vesselness.norm` / `threshold` / `temperature` | 0.25 / 0.2 / 0.05 | Normalisasi dan sigmoid peta ROI lunak. | Implementasi |
| `gan.quality.vessel_filter` | `sato` | Filter metrik pembuluh; harus ≠ Frangi (L6). | PRD §10 L6 |
| `gan.quality.vessel_sigmas` / `vessel_threshold` | [1.0, 2.0, 3.0] / 0.02 | Parameter Sato untuk fraksi area pembuluh. | Implementasi |
| `gan.quality.n_grid` | 100 | Citra sintetis per kelas untuk pemeriksaan visual. | FR-06 |
| `gan.quality.lpips_pairs` / `lpips_net` | 500 / `alex` | Pasangan dan jaringan LPIPS untuk keragaman. | FR-06 |
| `gan.generate_fraction` | 1.0 | Bangkitkan 100% jumlah citra ganas asli, sekali. | FR-07 |
| `gan.ratios` | [0.25, 0.5, 1.0] | Rasio sintetis yang dilaporkan semuanya. | G2 |

## 11. Pilot λ_roi (D2) dan S7

| Parameter | Nilai | Arti | Sumber |
| --- | --- | --- | --- |
| `pilot.rep` / `fold` | 0 / 0 | Lipatan tempat pilot λ_roi (pakai pasien validasi lipatan ini). | D2 |
| `pilot.lambdas` | [0.1, 0.5, 1.0, 2.0] | Kandidat λ_roi. | Implementasi |
| `pilot.criterion` | `val_fid` | Pilih λ dengan FID validasi terendah; seri → λ lebih kecil. Ditetapkan sebelum eksperimen. | D2 |
| `pilot.total_kimg` | null | null = `gan.total_kimg`. | Implementasi |
| `s7.n_repeats` | 3 | Jumlah pembagian latih/validasi acak atas pasien resolusi mayoritas. | Keputusan (perlu persetujuan) |
| `s7.methods` | [S1, S5, S6] | Metode yang diuji di S7. | PRD §6 |

## 12. Evaluasi (FR-09, FR-10)

| Parameter | Nilai | Arti | Sumber |
| --- | --- | --- | --- |
| `eval.threshold` | 0.5 | Ambang keputusan pasien (tetap, tidak disetel). | PRD §6, L4 |
| `eval.bootstrap_reps` | 2000 | Ulangan bootstrap persentil bertingkat. | G1 |
| `eval.ci_level` | 0.95 | Tingkat interval kepercayaan. | G1 |
| `eval.bootstrap_seed` | 42 | Seed bootstrap. | RULES §4 |
| `eval.average_repeats` | true | Probabilitas pasien dirata-ratakan antar ulangan. Menunggu konfirmasi D4. | D4 |

## 13. Benchmark (hanya CP0)

| Parameter | Nilai | Arti | Sumber |
| --- | --- | --- | --- |
| `benchmark.warmup_steps` | 2 | Langkah pemanasan. | Implementasi |
| `benchmark.timed_steps` | 6 | Langkah yang diukur. | Implementasi |
| `benchmark.assumed_train_images` | 7578 | Asumsi ukuran latih untuk ekstrapolasi satu epoch (11.144 × 80% × 85%). Asumsi, bukan pengukuran. | Implementasi |

---

## 14. Penimpaan per skenario

Setiap berkas skenario mewarisi `base.yaml` (`_base`) dan hanya menimpa yang ada di tabel ini.

| Berkas | `scenario` | Penimpaan |
| --- | --- | --- |
| `s1.yaml` | S1 | – |
| `s2.yaml` | S2 | – |
| `s3_svm.yaml` / `s3_lda.yaml` | S3_svm / S3_lda | `s3.model` = svm / lda; `lasso_alpha` 0.001; `use_ctgan` false |
| `s4_svm.yaml` / `s4_lda.yaml` | S4_svm / S4_lda | seperti S3 dengan `use_ctgan` true |
| `s3_*_a01.yaml`, `s4_*_a01.yaml` | `*_a01` | `lasso_alpha` 0.01 (sensitivitas D3, **eksploratif**) |
| `s5.yaml` | S5 | `gan.lambda_roi` 0.0; `synthetic.gan_scenario` S5; `ratio` lewat `--ratio` |
| `s6.yaml` | S6 | `gan.lambda_roi` null (dari pilot D2); `synthetic.gan_scenario` S6; `ratio` lewat `--ratio` |
| `s7_s1.yaml` | S7_S1 | `splits_name` s7 |
| `s7_s5.yaml` | S7_S5 | `splits_name` s7; `lambda_roi` 0.0; `synthetic.ratio` diisi per job; rasio terpilih dari validasi DI DALAM split S7 (`best_ratio_S7_S5.json`) |
| `s7_s6.yaml` | S7_S6 | `splits_name` s7; `lambda_roi` null (dari pilot D2, lihat catatan); `synthetic.ratio` diisi per job; rasio terpilih dari validasi di dalam split S7 |

## 15. Parameter yang paling perlu diperhatikan sebelum menjalankan

1. `gan.total_kimg` – masih sementara.
2. Seluruh hiperparameter GAN bertanda *Implementasi* – belum pernah disetel; boleh disetel hanya sebelum eksperimen utama, memakai validasi atau pilot, dan dicatat.
3. `splits.n_repeats` – putuskan 2 atau 1 (untuk skenario GAN) sebelum CP2.
4. `radiomics.mask` – menunggu D7.
5. `eval.average_repeats` – menunggu D4.

---

## 16. Perbandingan parameter antar skenario (S1 sampai S7)

Tabel ini membandingkan apa yang **berbeda** antar skenario. Parameter yang tidak disebut sama untuk semua yang memakainya
(seed, evaluasi, ambang 0,5, bootstrap 2.000, probabilitas pasien = rata-rata citra).

### 16.1 Ringkasan per skenario

| Skenario | Fungsi di penelitian | Representasi masukan | Model | Penyeimbang kelas | Data sintetis |
| --- | --- | --- | --- | --- | --- |
| **S1** | Baseline | citra 224×224 | Swin-T pralatih, seluruh bobot di-*fine-tune* | weighted sampling | tidak ada |
| **S2** | Pembanding augmentasi klasik | embedding Swin beku (768 dimensi) | StandardScaler → regresi logistik (kepala linear) | SMOTE 1:1 pada embedding | tidak ada (SMOTE = interpolasi di ruang fitur) |
| **S3** | Replikasi Wang dkk. | fitur PyRadiomics (mask area pandang) | MinMax → LASSO → SVM linear atau LDA | SMOTE | tidak ada |
| **S4** | Replikasi Wang dkk. + GAN fitur | fitur PyRadiomics | seperti S3 | CTGAN (50% data latih) lalu SMOTE | sampel tabular (CTGAN) |
| **S5** | Ablation panduan ROI | citra 224×224 | Swin-T seperti S1 | weighted sampling | citra ganas dari cGAN **tanpa** D_roi (λ=0), rasio 25/50/100% |
| **S6** | **Metode usulan** | citra 224×224 | Swin-T seperti S1 | weighted sampling | citra ganas dari cGAN **dengan** D_roi (λ dari pilot), rasio 25/50/100% |
| **S7_S1** | Ketahanan resolusi (baseline) | citra 224×224 | seperti S1 | weighted sampling | tidak ada |
| **S7_S5** | Ketahanan resolusi | citra 224×224 | seperti S5 pada rasio terbaik | weighted sampling | cGAN λ=0, 1 rasio terbaik (dari validasi) |
| **S7_S6** | Ketahanan resolusi | citra 224×224 | seperti S6 pada rasio terbaik | weighted sampling | cGAN dengan D_roi, 1 rasio terbaik (dari validasi) |

Perbandingan **utama** (ditetapkan sebelum eksperimen): S6 lawan S1 dan S6 lawan S2 pada AUROC dan F1. Semua pasangan lain berlabel *eksploratif*.

### 16.2 Matriks parameter

`–` berarti tidak berlaku.

| Parameter | S1 | S2 | S3 | S4 | S5 | S6 | S7_S1 | S7_S5 | S7_S6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Konfigurasi | `s1.yaml` | `s2.yaml` | `s3_svm/lda.yaml` | `s4_svm/lda.yaml` | `s5.yaml` | `s6.yaml` | `s7_s1.yaml` | `s7_s5.yaml` | `s7_s6.yaml` |
| Modul job | `src.train` | `src.embed`, `src.train_s2` | `src.radiomics_feats`, `src.train_s3` | sama dengan S3 | `src.train_gan`, `src.train --ratio` | sama dengan S5 | `src.train` | rantai GAN | rantai GAN |
| `splits_name` | main | main | main | main | main | main | **s7** | **s7** | **s7** |
| Data latih | 5×2 lipatan, semua resolusi | idem | idem | idem | idem | idem | pasien resolusi mayoritas saja | idem | idem |
| Data uji | pasien lipatan uji | idem | idem | idem | idem | idem | gabungan resolusi lain (tunggal) | idem | idem |
| Jumlah ulangan | 2 (`splits.n_repeats`) | 2 | 2 | 2 | 2 | 2 | 3 (`s7.n_repeats`) | 3 | 3 |
| Jumlah job | 10 | 10 (+1 embedding) | 10 per model (SVM, LDA) | 10 per model | 10 GAN + 30 Swin | 10 GAN + 30 Swin | 3 | 3 GAN + 3 Swin | 3 GAN + 3 Swin |
| Sumber daya | GPU | GPU sekali (embedding), lalu CPU | CPU | CPU (GPU opsional untuk CTGAN) | GPU | GPU | GPU | GPU | GPU |
| Optimizer / lr / wd | AdamW / 1e-4 / 0,05 | – | – | – | idem S1 | idem S1 | idem S1 | idem S1 | idem S1 |
| Batch / epoch maks / sabar | 32 / 30 / 5 | – | – | – | idem S1 | idem S1 | idem S1 | idem S1 | idem S1 |
| Augmentasi klasik | crop 0,8–1,0, rotasi 10°, flip H dan V | tidak (hanya resize) | – | – | idem S1 | idem S1 | idem S1 | idem S1 | idem S1 |
| Penyeimbang kelas | weighted sampling | SMOTE (`smote_k` 5, `auto`) | SMOTE (`s3.smote_k` 5) | CTGAN lalu SMOTE | weighted sampling (menghitung sintetis) | idem S5 | weighted sampling | idem | idem |
| Pemilihan hiperparameter | henti dini pada val loss seimbang | C dari `c_grid` (val loss seimbang) | **tetap** (tidak ada pemilihan) | **tetap** | henti dini; rasio per lipatan dari val loss | idem S5 | henti dini | rasio = `best_config.json` | rasio = `best_config.json` |
| LASSO alpha | – | – | 0,001 (0,01 = `_a01`) | 0,001 (0,01 = `_a01`) | – | – | – | – | – |
| Pengklasifikasi akhir | Swin-T + kepala linear 2 kelas | regresi logistik | SVM linear (C=1, probabilitas) atau LDA | idem S3 | Swin-T | Swin-T | Swin-T | Swin-T | Swin-T |
| Mask radiomik | – | – | `fov` (D7) | `fov` (D7) | – | – | – | – | – |
| CTGAN (epoch / batch / fraksi) | – | – | – | 100 / 50 / 0,5 | – | – | – | – | – |
| `gan.lambda_roi` | – | – | – | – | **0,0** | **null → dari pilot** (kandidat 0,1; 0,5; 1,0; 2,0) | – | 0,0 | null → dari pilot |
| Diskriminator D_roi | – | – | – | – | tidak ada | ada | – | tidak ada | ada |
| `gan.total_kimg` | – | – | – | – | 500 (sementara) | 500 (sementara) | – | 500 (sementara) | 500 (sementara) |
| `gan.ratios` (sintetis dari jumlah ganas asli) | – | – | – | – | 0,25; 0,5; 1,0 (semua dilaporkan) | idem S5 | – | 0,25; 0,5; 1,0 (semua dilatih; 1 dipilih) | idem S7_S5 |
| Rasio terbaik dipilih dari | – | – | – | – | validasi tiap lipatan (L4) | idem S5 | – | validasi dalam split S7 (L5) | validasi dalam split S7 (L5) |
| Hasil utama | `results_S1.csv` | `results_S2.csv` | `results_S3_svm/lda.csv` | `results_S4_svm/lda.csv` | `results_S5.csv`, `_r25/50/100` | `results_S6.csv`, `_r25/50/100` | `results_S7_S1.csv` | `results_S7_S5.csv` | `results_S7_S6.csv` |
| Status perbandingan | pembanding utama | pembanding utama | eksploratif | eksploratif | eksploratif (ablation ROI) | **utama** (lawan S1 dan S2) | eksploratif | eksploratif | eksploratif |

Catatan:
- S3 dan S4 memakai fitur yang sama (`radiomics_feats` dijalankan sekali); beda keduanya hanya `s3.use_ctgan`.
- S5 dan S6 berbeda **hanya** pada `gan.lambda_roi` (0 lawan nilai dari pilot) dan ada tidaknya D_roi. Semua hiperparameter GAN lain dan klasifikatornya sama. Satu hal lain ikut berbeda: seed job diturunkan dari nama skenario, jadi inisialisasi dan urutan acak S5 dan S6 tidak identik. Selisihnya dapat dikaitkan dengan panduan ROI hanya sejauh variasi seed itu lebih kecil daripada selisihnya (itu salah satu alasan interval kepercayaan diperlukan).
- Pada S7, perbedaan terhadap S1, S5, dan S6 hanya pada split (`splits_name: s7`), jumlah ulangan, dan (S5/S6) rasio tunggal yang dipilih dari validasi di dalam split S7 (bukan `best_config.json` CV utama, karena CV utama melibatkan pasien uji S7; L5). Catatan: `lambda_roi` S7_S6 masih dari pilot D2 (lipatan 0 CV utama), yang belum bebas dari pasien resolusi minor.
- Varian sensitivitas LASSO 0,01 (`*_a01`) berlabel eksploratif dan tidak menjadi gerbang CP3.
