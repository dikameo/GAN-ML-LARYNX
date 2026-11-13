"""FR-02, FR-03: praproses lima langkah dan cache memmap uint8.

Langkah (urutan tetap, PRD §5):
 1. metrik kualitas (buram, silau, gelap) dihitung dan disimpan; TIDAK ada citra dibuang kecuali
    quality.filter_enabled=true (keputusan 2026-11-08: tidak ada aturan ambang yang disepakati)
 2. area pandang: ambang intensitas + isi lubang + komponen terbesar, crop kotak pembatas, di luar area = hitam,
    lalu diberi bantalan hitam menjadi persegi (rasio aspek tidak terdistorsi)
 3. resize ke cache_size (256); klasifikator me-resize ke 224 saat dibaca
 4. normalisasi: mean/std ImageNet tetap, diterapkan saat dibaca (tidak disimpan di cache)
 5. peta ROI pembuluh: filter Frangi pada kanal hijau di dalam area pandang, ambang tetap
"""
from __future__ import annotations

import argparse
import os
import shutil
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import frangi, gaussian

from src.config import P, load_merged
from src.utils import atomic_write_bytes, atomic_write_json, config_hash, is_done, mark_done


# ---------------------------------------------------------------- langkah 2
def fov_mask(rgb: np.ndarray, thr: float, min_area_fraction: float) -> tuple[np.ndarray, bool]:
    gray = rgb.mean(axis=2)
    m = ndi.binary_fill_holes(gray > thr)
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    if m.mean() < min_area_fraction:
        return np.ones(gray.shape, bool), True  # deteksi gagal -> seluruh citra, ditandai
    return m, False


def crop_square(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    img = (rgb * mask[..., None])[y0:y1, x0:x1]
    msk = mask[y0:y1, x0:x1]
    h, w = msk.shape
    s = max(h, w)
    pad_img = np.zeros((s, s, 3), np.uint8)
    pad_msk = np.zeros((s, s), bool)
    oy, ox = (s - h) // 2, (s - w) // 2
    pad_img[oy:oy + h, ox:ox + w] = img
    pad_msk[oy:oy + h, ox:ox + w] = msk
    return pad_img, pad_msk


# ---------------------------------------------------------------- langkah 5
def _remove_small(m: np.ndarray, min_size: int) -> np.ndarray:
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[lab]


def roi_map(img: np.ndarray, fov: np.ndarray, r: dict) -> np.ndarray:
    fe = ndi.binary_erosion(fov, iterations=int(r["fov_erode"]))
    if fe.sum() < 100:
        fe = fov
    g = img[..., 1].astype(np.float32) / 255.0
    g = np.where(fe, g, g[fe].mean() if fe.any() else 0.0)
    g = gaussian(g, r["green_blur"])
    v = frangi(g, sigmas=tuple(r["sigmas"]), alpha=r["alpha"], beta=r["beta"], gamma=r["gamma"], black_ridges=True)
    v = v * fe
    v = np.clip(v / (np.percentile(v[fe], r["norm_percentile"]) + 1e-8), 0, 1) if fe.any() else v
    m = (v > r["threshold"]) & fe
    return _remove_small(m, int(r["min_obj"])).astype(np.uint8)


# ---------------------------------------------------------------- gabungan
def process_array(rgb: np.ndarray, pp: dict) -> dict:
    S = pp["cache_size"]
    mask, fallback = fov_mask(rgb, pp["fov"]["threshold"], pp["fov"]["min_area_fraction"])
    sq, sqm = crop_square(rgb, mask)
    img = np.asarray(Image.fromarray(sq).resize((S, S), Image.LANCZOS), np.uint8)
    fov = np.asarray(Image.fromarray(sqm.astype(np.uint8) * 255).resize((S, S), Image.BILINEAR)) > 127
    gray = img.mean(axis=2)
    q = pp["quality"]
    inside = fov if fov.any() else np.ones_like(fov)
    lap = ndi.laplace(gray)
    quality = {
        "q_blur_lapvar": float(lap[inside].var()),
        "q_brightness": float(gray[inside].mean()),
        "q_glare_frac": float(((img.max(axis=2) >= q["saturation_level"]) & inside).mean()),
        "q_dark_frac": float((gray[inside] < q["dark_level"]).mean()),
        "q_fov_frac": float(fov.mean()),
        "fov_fallback": bool(fallback),
    }
    return {"img": img, "roi": roi_map(img, fov, pp["roi"]), "quality": quality}


def process_file(path: str | Path, pp: dict) -> dict:
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"), np.uint8)
    return process_array(rgb, pp)


_PP: dict = {}
_ROOT: Path | None = None


def _init(pp: dict, root: str) -> None:
    global _PP, _ROOT
    _PP, _ROOT = pp, Path(root)


def _work(rel: str) -> dict:
    return process_file(_ROOT / rel, _PP)


# ---------------------------------------------------------------- cache
def cache_hash(cfg: dict, images: pd.DataFrame) -> str:
    pp = {k: v for k, v in cfg["preprocess"].items() if k not in ("workers", "imagenet_mean", "imagenet_std", "normalize")}
    pp["roi"] = {k: v for k, v in pp["roi"].items() if k != "locked"}  # mengunci parameter tidak boleh memaksa cache dibangun ulang
    pp["roi_examples"] = None  # pemilihan contoh visual tidak memengaruhi isi cache
    listing = list(zip(images.rel_path, images.bytes))
    return config_hash({"pp": pp, "files": listing})


def build_cache(cfg: dict, images: pd.DataFrame | None = None, workers: int | None = None) -> Path:
    """Dijalankan sekali. Bila cache dengan hash sama sudah DONE, dilewati."""
    if images is None:
        images = pd.read_csv(P(cfg, "cp1", "audit_images.csv"))
    images = images.reset_index(drop=True)
    h = cache_hash(cfg, images)
    root = P(cfg, "cache")
    final = root / h[:12]
    if final.exists() and is_done(final, h):
        atomic_write_json(root / "current.json", {"hash": h, "dir": str(final)})
        print(f"[dilewati] cache {final.name} sudah DONE")
        return final
    tmp = root / (h[:12] + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    N, S = len(images), cfg["preprocess"]["cache_size"]
    imgs = np.lib.format.open_memmap(tmp / "images.npy", mode="w+", dtype=np.uint8, shape=(N, S, S, 3))
    rois = np.lib.format.open_memmap(tmp / "roi.npy", mode="w+", dtype=np.uint8, shape=(N, S, S))
    qual = []
    workers = workers or cfg["preprocess"]["workers"]
    rels = images.rel_path.tolist()
    if workers > 1:
        pool = Pool(workers, initializer=_init, initargs=(cfg["preprocess"], cfg["paths"]["dataset"]))
        it = pool.imap(_work, rels, chunksize=8)
    else:
        _init(cfg["preprocess"], cfg["paths"]["dataset"])
        pool, it = None, map(_work, rels)
    for i, r in enumerate(it):
        imgs[i], rois[i] = r["img"], r["roi"]
        qual.append(r["quality"])
        if (i + 1) % 500 == 0:
            print(f"  praproses {i + 1}/{N}", flush=True)
    if pool is not None:
        pool.close()
        pool.join()
    imgs.flush()
    rois.flush()
    del imgs, rois
    meta = pd.concat([images.reset_index(drop=True), pd.DataFrame(qual)], axis=1)
    meta.insert(0, "idx", np.arange(N))
    (tmp / "meta.csv").write_text(meta.to_csv(index=False), encoding="utf-8")
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    mark_done(final, h)
    atomic_write_json(root / "current.json", {"hash": h, "dir": str(final)})
    return final


# ---------------------------------------------------------------- pemeriksaan visual ROI (FR-03, D6)
def save_roi_examples(cfg: dict, cache_dir: str | Path, out_dir: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cache_dir, out_dir = Path(cache_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(cache_dir / "meta.csv")
    imgs = np.load(cache_dir / "images.npy", mmap_mode="r")
    rois = np.load(cache_dir / "roi.npy", mmap_mode="r")
    ex = cfg["preprocess"]["roi_examples"]
    rng = np.random.default_rng(ex["seed"])
    pick = np.sort(rng.choice(len(meta), size=min(ex["n"], len(meta)), replace=False))
    rows = meta.iloc[pick][["idx", "rel_path", "patient_uid", "label", "class_name"]].copy()
    rows["roi_fraction"] = [float(rois[i].mean()) for i in pick]
    atomic_write_bytes(out_dir / "roi_examples_patients.csv", rows.to_csv(index=False).encode("utf-8"))  # D6
    per_page = 10
    for page in range(0, len(pick), per_page):
        chunk = pick[page:page + per_page]
        fig, axes = plt.subplots(len(chunk), 3, figsize=(7.5, 2.5 * len(chunk)), squeeze=False)
        for ax_row, i in zip(axes, chunk):
            ov = np.asarray(imgs[i]).astype(np.float32).copy()
            m = np.asarray(rois[i]).astype(bool)
            ov[m] = 0.35 * ov[m] + 0.65 * np.array([255, 255, 0], np.float32)
            ax_row[0].imshow(imgs[i])
            ax_row[1].imshow(rois[i], cmap="gray", vmin=0, vmax=1)
            ax_row[2].imshow(ov.astype(np.uint8))
            ax_row[0].set_ylabel(f"{meta.patient_uid[i]}", fontsize=7)
            for a in ax_row:
                a.set_xticks([])
                a.set_yticks([])
        axes[0][0].set_title("citra 256", fontsize=8)
        axes[0][1].set_title("ROI Frangi", fontsize=8)
        axes[0][2].set_title("overlay", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"roi_check_{page // per_page + 1}.png", dpi=110)
        plt.close(fig)
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_merged(args.config)
    d = build_cache(cfg)
    save_roi_examples(cfg, d, P(cfg, "cp2", "roi_check"))
    print("cache:", d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
