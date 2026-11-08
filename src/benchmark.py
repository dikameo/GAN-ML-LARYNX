"""Benchmark CP0: waktu langkah latih Swin-T pada tensor ACAK, lalu ekstrapolasi ke satu epoch.

PENTING (kejujuran hasil): ini bukan epoch nyata. Data adalah tensor acak 224x224 karena cache
belum ada (CP2). Ekstrapolasi memakai assumed_train_images dari konfigurasi dan mengabaikan
waktu pembacaan data serta augmentasi, jadi waktu epoch nyata akan >= angka ini.

Jalankan: python -m src.benchmark --config configs/base.yaml --out /outputs/cp0/benchmark.json
"""
from __future__ import annotations

import argparse
import os
import time

import psutil
import torch

from src.utils import atomic_write_json, config_hash, load_config, set_seed


def build_model(hf_name: str, cache_dir: str, num_classes: int, pretrained: bool):
    from transformers import AutoConfig, AutoModelForImageClassification

    if pretrained:
        return AutoModelForImageClassification.from_pretrained(
            hf_name, cache_dir=cache_dir, num_labels=num_classes, ignore_mismatched_sizes=True)
    conf = AutoConfig.from_pretrained(hf_name, cache_dir=cache_dir, num_labels=num_classes)
    return AutoModelForImageClassification.from_config(conf)


def run(cfg: dict, pretrained: bool, batch_size: int | None = None, timed: int | None = None) -> dict:
    set_seed(cfg["seed"])
    tr, bm, md = cfg["train"], cfg["benchmark"], cfg["model"]
    bs = batch_size or tr["batch_size"]
    n_timed = timed or bm["timed_steps"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(md["hf_name"], cfg["paths"]["hf_cache"], md["num_classes"], pretrained).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    size = md["input_size"]
    x = torch.randn(bs, 3, size, size, device=device)
    y = torch.randint(0, md["num_classes"], (bs,), device=device)
    lossf = torch.nn.CrossEntropyLoss()

    def step():
        opt.zero_grad(set_to_none=True)
        loss = lossf(model(pixel_values=x).logits, y)
        loss.backward()
        opt.step()

    for _ in range(bm["warmup_steps"]):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        step()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    mean_step = sum(times) / len(times)
    steps_per_epoch = -(-bm["assumed_train_images"] // bs)
    return {
        "device": device,
        "pretrained_weights": pretrained,
        "data": "tensor acak (BUKAN data nyata)",
        "batch_size": bs,
        "timed_steps": n_timed,
        "step_seconds": times,
        "mean_step_seconds": mean_step,
        "images_per_second": bs / mean_step,
        "assumed_train_images": bm["assumed_train_images"],
        "estimated_epoch_seconds": mean_step * steps_per_epoch,
        "estimated_epoch_minutes": mean_step * steps_per_epoch / 60,
        "estimated_30_epoch_hours_per_job": mean_step * steps_per_epoch * tr["max_epochs"] / 3600,
        "ram_rss_gb": psutil.Process().memory_info().rss / 1e9,
        "torch_threads": torch.get_num_threads(),
        "cpu_count": os.cpu_count(),
        "config_hash": config_hash(cfg),
        "git_commit": os.environ.get("GIT_COMMIT", "tidak-diketahui"),
        "image_digest": os.environ.get("IMAGE_DIGEST", "tidak-diketahui"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--random-init", action="store_true",
                    help="tanpa bobot pralatih (tanpa jaringan); arsitektur sama sehingga waktu langkah setara")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--timed-steps", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = run(cfg, pretrained=not args.random_init, batch_size=args.batch_size, timed=args.timed_steps)
    atomic_write_json(args.out, res)
    for k, v in res.items():
        if k != "step_seconds":
            print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
