"""Pembacaan cache memmap dan Dataset PyTorch. Job membaca cache, bukan JPEG (FR-02)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from src.config import P


def find_cache(cfg: dict) -> Path:
    cur = P(cfg, "cache", "current.json")
    if not cur.exists():
        raise FileNotFoundError("Cache belum dibuat (jalankan CP2: python -m src.preprocess)")
    return Path(json.loads(cur.read_text(encoding="utf-8"))["dir"])


class Cache:
    def __init__(self, cache_dir: str | Path):
        self.dir = Path(cache_dir)
        self.hash = (self.dir / "DONE").read_text(encoding="utf-8").strip()
        self.meta = pd.read_csv(self.dir / "meta.csv")
        self.images = np.load(self.dir / "images.npy", mmap_mode="r")
        self.roi = np.load(self.dir / "roi.npy", mmap_mode="r")

    def indices_of(self, patients) -> np.ndarray:
        return self.meta.index[self.meta.patient_uid.isin(set(patients))].to_numpy()

    @property
    def labels(self) -> np.ndarray:
        return self.meta.label.to_numpy()


def build_transform(cfg: dict, train: bool) -> v2.Compose:
    size, aug, pp = cfg["model"]["input_size"], cfg["augment"], cfg["preprocess"]
    t: list = []
    if train:
        t.append(v2.RandomResizedCrop(size, scale=tuple(aug["crop_scale"]), ratio=tuple(aug["crop_ratio"]), antialias=True))
        if aug["hflip"]:
            t.append(v2.RandomHorizontalFlip())
        if aug["vflip"]:
            t.append(v2.RandomVerticalFlip())
        if aug["rotation_deg"]:
            t.append(v2.RandomRotation(aug["rotation_deg"]))
    else:
        t.append(v2.Resize((size, size), antialias=True))
    t += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(pp["imagenet_mean"], pp["imagenet_std"])]
    return v2.Compose(t)


class ImageDS(Dataset):
    """Mengembalikan (tensor ternormalisasi ImageNet, label, indeks cache).
    `synthetic` = (array uint8 N,H,W,3) opsional; dilabeli ganas (1) dan indeks cache -1 (S5/S6/S7)."""

    def __init__(self, cache: Cache, idx: np.ndarray, cfg: dict, train: bool, synthetic: np.ndarray | None = None):
        self.cache, self.idx = cache, np.asarray(idx)
        self.syn = synthetic
        self.n_real = len(self.idx)
        self.tf = build_transform(cfg, train)
        syn_labels = np.ones(0 if synthetic is None else len(synthetic), dtype=cache.labels.dtype)
        self.labels = np.concatenate([cache.labels[self.idx], syn_labels])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        if i < self.n_real:
            j = int(self.idx[i])
            arr = np.array(self.cache.images[j])
        else:
            j = -1
            arr = np.array(self.syn[i - self.n_real])
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return self.tf(x), int(self.labels[i]), j
