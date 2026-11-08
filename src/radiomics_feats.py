"""S3/S4 (FR-08): ekstraksi fitur PyRadiomics untuk SEMUA citra di cache, dihitung sekali.

Ekstraksi tidak memfit apa pun (tidak ada skala/seleksi), jadi satu berkas fitur aman dipakai semua lipatan (L2).
D7: dataset tidak punya mask lesi, sehingga mask = area pandang (radiomics.mask=fov) atau peta pembuluh (roi).
Jumlah fitur yang dihasilkan DICATAT apa adanya (Wang dkk. menyebut 104; angka di sini tergantung versi PyRadiomics).
Dapat dilanjutkan (progress.json).

Jalankan: python -m src.radiomics_feats --config configs/s3_svm.yaml
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from src.config import P, load_merged
from src.data import Cache, find_cache
from src.utils import atomic_write_json, config_hash, is_done, mark_done

_EXT = None
_RAD: dict = {}


def _extractor(rad: dict):
    from radiomics import featureextractor

    settings = {"binWidth": rad["bin_width"], "force2D": True, "label": 1, "correctMask": True}
    ex = featureextractor.RadiomicsFeatureExtractor(**settings)
    ex.disableAllFeatures()
    for cls in ("firstorder", "shape2D", "glcm", "glrlm", "glszm", "gldm", "ngtdm"):
        ex.enableFeatureClassByName(cls)
    return ex


def _init(rad: dict) -> None:
    global _EXT, _RAD
    import logging

    logging.getLogger("radiomics").setLevel(logging.ERROR)
    _RAD = rad
    _EXT = _extractor(rad)


def image_and_mask(img_rgb: np.ndarray, roi: np.ndarray, rad: dict) -> tuple[np.ndarray, np.ndarray]:
    gray = img_rgb[..., 1] if rad["channel"] == "green" else img_rgb.mean(axis=2)
    gray = gray.astype(np.float32)
    if rad["mask"] == "roi":
        mask = roi.astype(np.uint8)
    else:  # fov: piksel di luar area pandang bernilai 0 di cache
        mask = ndi.binary_fill_holes(img_rgb.max(axis=2) > 0).astype(np.uint8)
    return gray, mask


def features_of(args) -> dict:
    img, roi = args
    import SimpleITK as sitk

    gray, mask = image_and_mask(img, roi, _RAD)
    if mask.sum() < 10:  # mask terlalu kecil -> fitur tak terdefinisi; NaN dicatat, tidak diisi diam-diam
        return {}
    r = _EXT.execute(sitk.GetImageFromArray(gray), sitk.GetImageFromArray(mask))
    return {k: float(v) for k, v in r.items() if k.startswith("original_")}


def extract_all(cfg: dict, cache_dir=None) -> Path:
    cache = Cache(cache_dir or find_cache(cfg))
    rad = cfg["radiomics"]
    h = config_hash({"rad": rad, "cache": cache.hash})
    d = P(cfg, "features", f"rad_{h[:12]}")
    if is_done(d, h):
        print(f"[dilewati] fitur {d.name} sudah DONE")
        return d
    d.mkdir(parents=True, exist_ok=True)
    N = len(cache.meta)
    prog = d / "progress.json"
    done_n = json.loads(prog.read_text())["done"] if prog.exists() else 0
    names = json.loads((d / "names.json").read_text()) if (d / "names.json").exists() else None
    mm = None
    workers = rad["workers"] or None
    with Pool(workers, initializer=_init, initargs=(rad,)) as pool:
        for s in range(done_n, N, rad["chunk"]):
            idx = range(s, min(s + rad["chunk"], N))
            rows = pool.map(features_of, [(np.array(cache.images[i]), np.array(cache.roi[i])) for i in idx])
            if names is None:
                first = next((r for r in rows if r), None)
                if first is None:
                    raise RuntimeError("tidak ada fitur yang dapat diekstrak pada potongan pertama")
                names = sorted(first)
                atomic_write_json(d / "names.json", names)
            if mm is None:
                mm = np.lib.format.open_memmap(d / "feat.npy", mode="r+" if (d / "feat.npy").exists() else "w+",
                                               dtype=np.float32, shape=(N, len(names)))
            for i, r in zip(idx, rows):
                mm[i] = [r.get(n, np.nan) for n in names]
            mm.flush()
            atomic_write_json(prog, {"done": idx[-1] + 1, "total": N})
            print(f"  radiomics {idx[-1] + 1}/{N}", flush=True)
    n_nan = int(np.isnan(np.load(d / "feat.npy", mmap_mode="r")).any(axis=1).sum())
    atomic_write_json(d / "summary.json", {"n_features": len(names), "n_images": N, "n_images_with_nan": n_nan,
                                           "mask": rad["mask"], "channel": rad["channel"]})
    mark_done(d, h)
    return d


def feature_dir(cfg: dict, cache: Cache) -> Path:
    h = config_hash({"rad": cfg["radiomics"], "cache": cache.hash})
    return P(cfg, "features", f"rad_{h[:12]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    print(extract_all(load_merged(a.config)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
