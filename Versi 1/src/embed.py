"""S2: embedding Swin-T beku untuk SEMUA citra di cache, dihitung sekali (PRD §6, FR-08).

Tidak ada fit pada data di sini (bobot pralatih ImageNet, normalisasi ImageNet tetap), jadi satu
berkas embedding aman dipakai semua lipatan tanpa risiko L2. Dapat dilanjutkan (progress.json).

Jalankan: python -m src.embed --config configs/s2.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.config import P, load_merged
from src.data import Cache, ImageDS, find_cache
from src.models import build_encoder, configure_threads, device_of, embed
from src.utils import atomic_write_json, config_hash, is_done, mark_done, set_seed


def embedding_path(cfg: dict, cache: Cache) -> Path:
    tag = config_hash({"model": cfg["model"], "pp": cfg["preprocess"]["imagenet_mean"], "cache": cache.hash})[:12]
    return P(cfg, "embeddings", f"emb_{tag}")


def run_embed(cfg: dict, cache_dir=None) -> Path:
    cache = Cache(cache_dir or find_cache(cfg))
    d = embedding_path(cfg, cache)
    h = config_hash({"model": cfg["model"], "cache": cache.hash, "mean": cfg["preprocess"]["imagenet_mean"]})
    if is_done(d, h):
        print(f"[dilewati] embedding {d.name} sudah DONE")
        return d
    d.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["seed"])
    configure_threads(cfg)
    device = device_of(cfg)
    enc = build_encoder(cfg).to(device)
    N = len(cache.meta)
    dim = int(enc.num_features)
    f = d / "emb.npy"
    prog = d / "progress.json"
    done_n = json.loads(prog.read_text())["done"] if prog.exists() and f.exists() else 0
    mm = np.lib.format.open_memmap(f, mode="r+" if f.exists() and done_n else "w+", dtype=np.float32, shape=(N, dim))
    bs = cfg["s2"]["embed_batch_size"]
    chunk = bs * 8
    for s in range(done_n, N, chunk):
        idx = np.arange(s, min(s + chunk, N))
        ds = ImageDS(cache, idx, cfg, train=False)
        for j, e in embed(enc, ds, bs, device, cfg):
            mm[j] = e
        mm.flush()
        atomic_write_json(prog, {"done": int(idx[-1] + 1), "total": N})
        print(f"  embedding {idx[-1] + 1}/{N}", flush=True)
    del mm
    mark_done(d, h)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    print(run_embed(load_merged(a.config)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
