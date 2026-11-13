"""Benchmark Fase 0 untuk GAN (pengganti benchmark StyleGAN2-ADA resmi, decisions.yaml D8): detik per kimg dan memori.
Data ACAK; hanya mengukur kecepatan, bukan kualitas. Hasil dipakai untuk menetapkan gan.total_kimg dan menghitung
waktu total (jumlah waktu job / jumlah GPU).

Jalankan: python -m src.bench_gan --config configs/s6.yaml --out /outputs/cp0/benchmark_gan.json
"""
from __future__ import annotations

import argparse
import copy
import os
import platform
import time

import torch
import torch.nn.functional as F

from src.config import load_merged
from src.gan.augment import augment
from src.gan.vesselness import roi_input, roi_soft
from src.models import configure_threads, device_of
from src.train_gan import build_nets
from src.utils import atomic_write_json, set_seed


def bench(cfg: dict, lam: float, iters: int, warmup: int) -> dict:
    set_seed(cfg["seed"])
    dev, g = device_of(cfg), cfg["gan"]
    B, res, v = g["batch_size"], g["resolution"], g["vesselness"]
    G, D, Droi = build_nets(cfg, lam, dev)
    optG = torch.optim.Adam(G.parameters(), lr=g["lr_g"], betas=tuple(g["betas"]))
    optD = torch.optim.Adam(list(D.parameters()) + (list(Droi.parameters()) if Droi else []), lr=g["lr_d"], betas=tuple(g["betas"]))
    real = torch.rand(B, 3, res, res, device=dev) * 2 - 1
    y = torch.randint(2, (B,), device=dev)
    times = []
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for i in range(warmup + iters):
        t0 = time.perf_counter()
        with torch.no_grad():
            fake = G(torch.randn(B, G.z_dim, device=dev), y)
        loss = F.softplus(D(augment(fake, 0.2), y)).mean() + F.softplus(-D(augment(real, 0.2), y)).mean()
        if Droi is not None:
            with torch.no_grad():
                xr, xf = roi_input(real, roi_soft(real, v)), roi_input(fake, roi_soft(fake, v))
            loss = loss + F.softplus(Droi(xf, y)).mean() + F.softplus(-Droi(xr, y)).mean()
        optD.zero_grad(); loss.backward(); optD.step()
        fake = G(torch.randn(B, G.z_dim, device=dev), y)
        lg = F.softplus(-D(augment(fake, 0.2), y)).mean()
        if Droi is not None:
            lg = lg + lam * F.softplus(-Droi(roi_input(fake, roi_soft(fake, v)), y)).mean()
        optG.zero_grad(); lg.backward(); optG.step()
        if dev == "cuda":
            torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    mean = sum(times) / len(times)
    return {"lambda_roi": lam, "mean_iter_seconds": mean, "seconds_per_kimg": mean * 1000 / B,
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--lambda-roi", type=float, default=1.0, help="nilai > 0 untuk mengukur biaya D_roi")
    a = ap.parse_args()
    cfg = load_merged(a.config)
    configure_threads(cfg)
    rows = [bench(cfg, 0.0, a.iters, a.warmup), bench(cfg, a.lambda_roi, a.iters, a.warmup)]
    tk = cfg["gan"]["total_kimg"]
    est = {r["lambda_roi"]: r["seconds_per_kimg"] * tk / 3600 for r in rows}
    out = {"host": platform.node(), "device": device_of(cfg), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "data": "tensor acak (BUKAN data nyata)", "batch_size": cfg["gan"]["batch_size"], "total_kimg_config": tk,
           "per_lambda": rows,
           "estimasi_jam_per_gan": {str(k): v for k, v in est.items()},
           "estimasi_jam_20_gan": est[0.0] * 10 + est[a.lambda_roi] * 10,
           "catatan": "estimasi_jam_20_gan = 10 GAN tanpa ROI (S5) + 10 GAN dengan ROI (S6), tanpa evaluasi/pembangkitan; "
                      "bagi dengan jumlah GPU untuk waktu total (PRD §8)",
           "git_commit": os.environ.get("GIT_COMMIT", "tidak-diketahui")}
    atomic_write_json(a.out, out)
    for k, v in out.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
