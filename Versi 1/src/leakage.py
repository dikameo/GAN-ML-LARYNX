"""Pengendalian kebocoran L1-L3 (PRD §10) sebagai uji otomatis yang menggagalkan job."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class LeakageError(AssertionError):
    pass


def check_l1(splits: dict) -> bool:
    """L1: tidak ada pasien di latih/validasi dan uji sekaligus; tiap pasien diuji tepat sekali per ulangan."""
    allp = set(splits["all_patients"])
    for r in splits["repeats"]:
        tested: list[str] = []
        for f in r["folds"]:
            tr, va, te = set(f["train"]), set(f["val"]), set(f["test"])
            if tr & te:
                raise LeakageError(f"L1: pasien latih juga di uji (rep {r['rep']} lipatan {f['fold']}): {sorted(tr & te)[:5]}")
            if va & te:
                raise LeakageError(f"L1: pasien validasi juga di uji (rep {r['rep']} lipatan {f['fold']}): {sorted(va & te)[:5]}")
            if tr & va:
                raise LeakageError(f"L1: pasien latih juga di validasi (rep {r['rep']} lipatan {f['fold']})")
            if tr | va | te != allp:
                raise LeakageError(f"L1: gabungan partisi != seluruh pasien (rep {r['rep']} lipatan {f['fold']})")
            tested += list(te)
        if sorted(tested) != sorted(allp):
            raise LeakageError(f"L1: pada rep {r['rep']} tiap pasien harus diuji tepat sekali")
    return True


def check_l5(splits: dict, resolution_of: dict[str, str]) -> bool:
    """L5 (S7): pasien resolusi uji tidak dipakai melatih generator/klasifikator atau memilih model;
    semua pasien latih/validasi berresolusi mayoritas, semua pasien uji bukan."""
    major = splits["major_resolution"]
    for r in splits["repeats"]:
        for f in r["folds"]:
            tr, va, te = set(f["train"]), set(f["val"]), set(f["test"])
            if (tr | va) & te or tr & va:
                raise LeakageError(f"L5: partisi S7 saling tumpang tindih (rep {r['rep']})")
            bad = [p for p in tr | va if resolution_of[p] != major]
            if bad:
                raise LeakageError(f"L5: pasien non-mayoritas di latih/validasi: {sorted(bad)[:5]}")
            bad = [p for p in te if resolution_of[p] == major]
            if bad:
                raise LeakageError(f"L5: pasien resolusi mayoritas di uji S7: {sorted(bad)[:5]}")
    return True


def check_splits(splits: dict, resolution_of: dict[str, str] | None = None) -> bool:
    """Pilih pemeriksaan sesuai jenis split. S7 butuh peta resolusi (dari audit.csv)."""
    if splits.get("kind") == "s7":
        if resolution_of is None:
            raise LeakageError("L5: peta resolusi pasien diperlukan untuk memeriksa split S7")
        return check_l5(splits, resolution_of)
    return check_l1(splits)


def check_l6(cfg: dict) -> bool:
    """L6: metrik struktur pembuluh memakai filter berbeda dari pembuat ROI."""
    if cfg["gan"]["quality"]["vessel_filter"].lower() == cfg["preprocess"]["roi"]["filter"].lower():
        raise LeakageError("L6: filter metrik pembuluh sama dengan filter pembuat ROI")
    return True


def check_l3_job(job_patients: dict, split_fold: dict, synthetic_source_patients: list[str] | None = None) -> bool:
    """L3: daftar pasien yang dipakai job sama dengan partisi latih/validasi; tidak ada pasien uji di dalamnya.
    Citra sintetis (bila ada) hanya boleh bersumber dari pasien latih."""
    test = set(split_fold["test"])
    used = set(job_patients["train"]) | set(job_patients["val"])
    if used & test:
        raise LeakageError(f"L3: job memakai pasien uji: {sorted(used & test)[:5]}")
    if set(job_patients["train"]) != set(split_fold["train"]) or set(job_patients["val"]) != set(split_fold["val"]):
        raise LeakageError("L3: daftar pasien job berbeda dari berkas split")
    if set(job_patients["test"]) != test:
        raise LeakageError("L3: daftar pasien uji job berbeda dari berkas split")
    if synthetic_source_patients is not None and not set(synthetic_source_patients) <= set(split_fold["train"]):
        raise LeakageError("L3: citra sintetis bersumber dari pasien non-latih")
    return True


def check_l3_dir(jobs_root: str | Path, scenario: str, splits: dict) -> int:
    """Periksa semua job selesai pada satu skenario; kembalikan jumlah job yang diperiksa."""
    n = 0
    for r in splits["repeats"]:
        for f in r["folds"]:
            pj = Path(jobs_root) / scenario / f"rep{r['rep']}_fold{f['fold']}" / "patients.json"
            if pj.exists():
                check_l3_job(json.loads(pj.read_text(encoding="utf-8")), f)
                n += 1
    return n


class GuardedArray:
    """L2: pembungkus larik yang menolak akses ke baris terlarang (indeks uji) dan penyerahan seluruh larik.
    Dipakai pada fase fit (scaler, SMOTE, pemilihan C): fit hanya boleh melihat indeks latih/validasi."""

    def __init__(self, arr, forbidden):
        self._arr = arr
        self._forbidden = np.asarray(sorted(set(map(int, forbidden))), dtype=np.int64)
        self.accessed: set[int] = set()

    def __len__(self):
        return len(self._arr)

    @property
    def shape(self):
        return self._arr.shape

    def __array__(self, *a, **k):
        raise LeakageError("L2: seluruh larik diminta pada fase fit; gunakan indeks latih/validasi saja")

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            pos = np.arange(len(self._arr))[idx]
        else:
            idx_a = np.asarray(idx)
            pos = np.flatnonzero(idx_a) if idx_a.dtype == bool else idx_a.ravel().astype(np.int64)
        if np.intersect1d(pos, self._forbidden).size:
            raise LeakageError("L2: fase fit mengakses baris data uji")
        self.accessed.update(int(p) for p in pos)
        return np.asarray(self._arr[idx])
