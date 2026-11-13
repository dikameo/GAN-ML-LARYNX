"""FR-06, FR-07, FR-11: GAN kondisional kelas dengan D_roi (S6) atau tanpa (S5, lambda_roi=0).

Jalankan: python -m src.train_gan --config configs/s6.yaml --rep 0 --fold 0
- Dilatih HANYA pada citra pasien partisi latih (L3); validasi dipakai memantau FID; uji hanya untuk pelaporan akhir.
- last.pt ditulis atomik tiap snapshot_kimg; terputus -> dilanjutkan; DONE berisi hash konfigurasi.
- Setelah selesai: bangkitkan 100% jumlah citra ganas latih SEKALI (synth.npz); subset 25%/50% diambil dari sini
  oleh src.train. Metrik kualitas dihitung terhadap citra uji untuk PELAPORAN, bukan seleksi.
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.config import P, load_merged
from src.data import Cache, find_cache
from src.gan.augment import augment
from src.gan.metrics import (fid_from_feats, inception_feats, lpips_diversity, load_inception, save_grid,
                             vessel_w1)
from src.gan.networks import Discriminator, Generator
from src.gan.vesselness import roi_input, roi_soft
from src.jobio import job_dir, resolution_map, write_run_info
from src.leakage import check_l3_job, check_l6, check_splits
from src.models import device_of
from src.splits import get_fold, load_splits
from src.utils import (atomic_torch_save, atomic_write_bytes, atomic_write_json, config_hash, is_done, job_seed,
                       mark_done, set_seed)


# ------------------------------------------------------------------ util data
def to_float(u8: torch.Tensor, device: str) -> torch.Tensor:
    return u8.to(device, non_blocking=True).float() / 127.5 - 1.0


def load_u8(cache: Cache, idx: np.ndarray, res: int) -> torch.Tensor:
    """(N,3,res,res) uint8 di CPU. Diperkecil dengan area-interpolasi bila res != ukuran cache."""
    out = []
    for s in range(0, len(idx), 256):
        a = torch.from_numpy(np.array(cache.images[np.sort(idx[s:s + 256])])).permute(0, 3, 1, 2)
        if a.shape[-1] != res:
            a = F.interpolate(a.float(), size=(res, res), mode="area").round().clamp(0, 255).to(torch.uint8)
        out.append(a)
    return torch.cat(out) if out else torch.zeros(0, 3, res, res, dtype=torch.uint8)


def sorted_labels(cache: Cache, idx: np.ndarray) -> np.ndarray:
    return cache.labels[np.sort(idx)]


@torch.no_grad()
def sample_u8(G: Generator, n: int, label: int | None, labels_pool: torch.Tensor | None, bs: int, device: str,
              seed: int, out_size: int | None = None) -> np.ndarray:
    """Bangkitkan n citra (N,H,W,3) uint8. label=None -> label diambil dari labels_pool (distribusi empiris)."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    out = []
    for s in range(0, n, bs):
        b = min(bs, n - s)
        z = torch.randn(b, G.z_dim, generator=g).to(device)
        y = (torch.full((b,), label, dtype=torch.long) if label is not None
             else labels_pool[torch.randint(len(labels_pool), (b,), generator=g)]).to(device)
        x = G(z, y).clamp(-1, 1)
        if out_size and x.shape[-1] != out_size:
            x = F.interpolate(x, size=(out_size, out_size), mode="bilinear", align_corners=False)
        out.append(((x + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1, 1, 3), np.uint8)


def atomic_savez(path: Path, **arrs) -> None:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrs)
    atomic_write_bytes(path, buf.getvalue())


# ------------------------------------------------------------------ inti pelatihan
def build_nets(cfg: dict, lam: float, device: str):
    g = cfg["gan"]
    G = Generator(g["z_dim"], g["w_dim"], 2, g["channels"], g["resolution"], g["map_layers"]).to(device)
    D = Discriminator(3, 2, g["d_channels"], g["resolution"]).to(device)
    Droi = Discriminator(4, 2, g["d_channels"], g["resolution"]).to(device) if lam > 0 else None
    return G, D, Droi


def train_core(cfg: dict, real_u8: torch.Tensor, labels: torch.Tensor, val_u8: np.ndarray | None, lam: float,
               total_kimg: float, d: Path, h: str, seed: int, device: str) -> dict:
    g = cfg["gan"]
    B = g["batch_size"]
    G, D, Droi = build_nets(cfg, lam, device)
    G_ema = copy.deepcopy(G).eval().requires_grad_(False)
    optG = torch.optim.Adam(G.parameters(), lr=g["lr_g"], betas=tuple(g["betas"]))
    d_params = list(D.parameters()) + (list(Droi.parameters()) if Droi is not None else [])
    optD = torch.optim.Adam(d_params, lr=g["lr_d"], betas=tuple(g["betas"]))
    n_iters = int(total_kimg * 1000 / B)
    snap_every = max(1, int(g["snapshot_kimg"] * 1000 / B))
    ada, v = g["ada"], g["vesselness"]
    st = {"it": 0, "p": 0.0, "history": []}
    last = d / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        if ck.get("config_hash") == h:
            G.load_state_dict(ck["G"]); G_ema.load_state_dict(ck["G_ema"]); D.load_state_dict(ck["D"])
            if Droi is not None and ck.get("Droi"):
                Droi.load_state_dict(ck["Droi"])
            optG.load_state_dict(ck["optG"]); optD.load_state_dict(ck["optD"])
            st = {k: ck[k] for k in ("it", "p", "history")}
            torch.set_rng_state(ck["rng_torch"]); np.random.set_state(ck["rng_np"]); random.setstate(ck["rng_py"])
            print(f"[lanjut] GAN dari iterasi {st['it']}/{n_iters}")
        else:
            print("[peringatan] last.pt GAN dari konfigurasi lain diabaikan")

    inc = load_inception(device) if val_u8 is not None else None
    val_feats = inception_feats(inc, val_u8, device) if inc is not None else None
    ada_stat: list[float] = []
    N = len(real_u8)
    labels = labels.long()
    t0 = time.time()
    for it in range(st["it"], n_iters):
        p = st["p"] if ada["enabled"] else 0.0
        idx = torch.randint(N, (B,))
        real, y = to_float(real_u8[idx], device), labels[idx].to(device)
        yg = labels[torch.randint(N, (B,))].to(device)
        # ---- langkah D
        G.requires_grad_(False)
        D.requires_grad_(True)
        if Droi is not None:
            Droi.requires_grad_(True)
        with torch.no_grad():
            fake = G(torch.randn(B, G.z_dim, device=device), yg)
        d_real = D(augment(real, p), y)
        loss_D = F.softplus(D(augment(fake, p), yg)).mean() + F.softplus(-d_real).mean()
        if Droi is not None:
            with torch.no_grad():
                xr, xf = roi_input(real, roi_soft(real, v)), roi_input(fake, roi_soft(fake, v))
            loss_D = loss_D + F.softplus(Droi(augment(xf, p, color=False), yg)).mean() \
                + F.softplus(-Droi(augment(xr, p, color=False), y)).mean()
        optD.zero_grad(set_to_none=True)
        loss_D.backward()
        if it % g["r1_interval"] == 0:  # R1 lazy
            ra = augment(real, p).detach().requires_grad_(True)
            gr, = torch.autograd.grad(D(ra, y).sum(), ra, create_graph=True)
            r1 = gr.square().sum(dim=(1, 2, 3)).mean() * (g["r1_gamma"] / 2) * g["r1_interval"]
            if Droi is not None:
                xa = augment(xr, p, color=False).detach().requires_grad_(True)
                gr2, = torch.autograd.grad(Droi(xa, y).sum(), xa, create_graph=True)
                r1 = r1 + gr2.square().sum(dim=(1, 2, 3)).mean() * (g["r1_gamma"] / 2) * g["r1_interval"]
            r1.backward()
        optD.step()
        # ---- langkah G
        G.requires_grad_(True)
        D.requires_grad_(False)
        if Droi is not None:
            Droi.requires_grad_(False)
        fake = G(torch.randn(B, G.z_dim, device=device), yg)
        loss_G = F.softplus(-D(augment(fake, p), yg)).mean()
        if Droi is not None:
            loss_G = loss_G + lam * F.softplus(-Droi(augment(roi_input(fake, roi_soft(fake, v)), p, color=False), yg)).mean()
        optG.zero_grad(set_to_none=True)
        loss_G.backward()
        optG.step()
        # ---- EMA dan ADA
        with torch.no_grad():
            beta = 0.5 ** (B / (g["ema_kimg"] * 1000))
            for pe, pg in zip(G_ema.parameters(), G.parameters()):
                pe.copy_(pg.lerp(pe, beta))
            ada_stat.append(float(torch.sign(d_real).mean()))
            if ada["enabled"] and (it + 1) % ada["interval"] == 0:
                rt = float(np.mean(ada_stat))
                st["p"] = float(min(1.0, max(0.0, st["p"] + np.sign(rt - ada["target"]) * B * ada["interval"] / (ada["kimg"] * 1000))))
                ada_stat = []
        st["it"] = it + 1
        if st["it"] % snap_every == 0 or st["it"] == n_iters:
            rec = {"kimg": st["it"] * B / 1000, "loss_D": float(loss_D), "loss_G": float(loss_G), "ada_p": st["p"],
                   "seconds": time.time() - t0}
            if inc is not None:
                n = min(g["fid_samples"], len(val_u8))
                fk = sample_u8(G_ema, n, None, labels, 32, device, seed + st["it"])
                rec["fid_val"] = fid_from_feats(val_feats[:n] if n < len(val_feats) else val_feats, inception_feats(inc, fk, device))
            st["history"].append(rec)
            print(rec, flush=True)
            atomic_torch_save({"G": G.state_dict(), "G_ema": G_ema.state_dict(), "D": D.state_dict(),
                               "Droi": Droi.state_dict() if Droi is not None else None, "optG": optG.state_dict(),
                               "optD": optD.state_dict(), "config_hash": h, "rng_torch": torch.get_rng_state(),
                               "rng_np": np.random.get_state(), "rng_py": random.getstate(), **st}, last)
    atomic_torch_save(G_ema.state_dict(), d / "G_ema.pt")
    return {"G_ema": G_ema, "history": st["history"], "labels": labels, "inception": inc}


def quality_report(cfg: dict, G_ema: Generator, real_u8_np: np.ndarray, real_labels: np.ndarray, device: str,
                   seed: int, out_dir: Path | None, inc=None) -> dict:
    """FID, W1 pembuluh (Sato), LPIPS terhadap citra asli `real_u8_np` (N,H,W,3) — validasi (pilot) atau uji (pelaporan)."""
    check_l6(cfg)
    q, size = cfg["gan"]["quality"], real_u8_np.shape[1]
    inc = inc or load_inception(device)
    fr_all = inception_feats(inc, real_u8_np, device)
    pool = torch.as_tensor(real_labels)
    fake_all = sample_u8(G_ema, len(real_u8_np), None, pool, 32, device, seed, size)
    rep = {"fid_all": fid_from_feats(fr_all, inception_feats(inc, fake_all, device)), "n_real": int(len(real_u8_np))}
    per = {}
    for c, name in ((0, "benign"), (1, "malignant")):
        real_c = real_u8_np[real_labels == c]
        fake_c = sample_u8(G_ema, max(q["n_grid"], min(len(real_c), 500)), c, None, 32, device, seed + 10 + c, size)
        if len(real_c) > 1:
            per[f"fid_{name}"] = fid_from_feats(inception_feats(inc, real_c, device), inception_feats(inc, fake_c, device))
            per[f"vessel_w1_{name}"] = vessel_w1(real_c[:300], fake_c[:300], q)
        per[f"lpips_diversity_{name}"] = lpips_diversity(fake_c, q["lpips_pairs"], q["lpips_net"], device, seed)
        if out_dir is not None:
            save_grid(fake_c[:q["n_grid"]], out_dir / f"samples_{name}.png")
    rep.update(per)
    return rep


# ------------------------------------------------------------------ job
def resolve_lambda(cfg: dict) -> float:
    lam = cfg["gan"]["lambda_roi"]
    if lam is None:
        sp = P(cfg, "cp4", "lambda_selection.json")
        if not sp.exists():
            raise FileNotFoundError(f"lambda_roi belum ditetapkan: jalankan python -m src.pilot_lambda (D2) -> {sp}")
        lam = json.loads(sp.read_text(encoding="utf-8"))["chosen"]
    return float(lam)


def run_gan_job(cfg: dict, rep: int, fold: int, lam: float, d: Path, mode: str = "full", cache_dir=None,
                total_kimg: float | None = None) -> dict:
    """mode='full' (S5/S6): latih, bangkitkan sintetis, laporan terhadap UJI. mode='pilot': latih, laporan terhadap VALIDASI."""
    splits = load_splits(cfg)
    check_splits(splits, resolution_map(cfg) if splits.get("kind") == "s7" else None)
    check_l6(cfg)
    cache = Cache(cache_dir or find_cache(cfg))
    sf = get_fold(splits, rep, fold)
    tk = total_kimg or cfg["gan"]["total_kimg"]
    h = config_hash({"cfg": cfg, "rep": rep, "fold": fold, "lam": lam, "mode": mode, "tk": tk, "cache": cache.hash,
                     "splits": splits["meta"]["hash"]})
    if is_done(d, h):
        print(f"[dilewati] GAN {d} sudah DONE")
        return json.loads((d / "gan_metrics.json").read_text(encoding="utf-8"))
    d.mkdir(parents=True, exist_ok=True)
    atomic_write_json(d / "patients.json", {"train": sf["train"], "val": sf["val"], "test": sf["test"]})
    check_l3_job(json.loads((d / "patients.json").read_text(encoding="utf-8")), sf)

    seed = job_seed(cfg["seed"], cfg["scenario"], "gan", rep, fold, lam)
    set_seed(seed)
    device = device_of(cfg)
    torch.backends.cudnn.benchmark = True
    res = cfg["gan"]["resolution"]
    idx_tr, idx_va = np.sort(cache.indices_of(sf["train"])), np.sort(cache.indices_of(sf["val"]))
    real_u8 = load_u8(cache, idx_tr, res)
    labels = torch.from_numpy(cache.labels[idx_tr])
    val_np = np.array(cache.images[idx_va])
    if val_np.shape[1] != res:
        val_np = F.interpolate(torch.from_numpy(val_np).permute(0, 3, 1, 2).float(), size=(res, res), mode="area").round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
    write_run_info(d, cfg, h, seed, {"rep": rep, "fold": fold, "lambda_roi": lam, "mode": mode, "total_kimg": tk,
                                     "device": device, "cache_hash": cache.hash, "splits_hash": splits["meta"]["hash"],
                                     "n_train_images": int(len(idx_tr)), "n_val_images": int(len(idx_va)),
                                     "catatan": "GAN ringkas bergaya StyleGAN2, ADA-lite (decisions.yaml D8)"})
    out = train_core(cfg, real_u8, labels, val_np, lam, tk, d, h, seed, device)
    G_ema, inc = out["G_ema"], out["inception"]
    atomic_write_bytes(d / "gan_history.json", json.dumps(out["history"], indent=2).encode("utf-8"))

    if mode == "pilot":
        metrics = quality_report(cfg, G_ema, val_np, cache.labels[idx_va], device, seed, d / "grid_val", inc)
        metrics["evaluated_on"] = "pasien validasi dalam lipatan latih (BUKAN uji)"
    else:
        n_syn = int(round(cfg["gan"]["generate_fraction"] * int((cache.labels[idx_tr] == 1).sum())))
        syn = sample_u8(G_ema, n_syn, 1, None, 32, device, seed + 99, 256)
        atomic_savez(d / "synth.npz", images=syn)
        atomic_write_json(d / "synth_meta.json", {"n": int(n_syn), "class": 1, "source_patients": sf["train"],
                                                  "lambda_roi": lam, "seed": seed + 99, "never_in_val_or_test": True})
        idx_te = np.sort(cache.indices_of(sf["test"]))
        test_np = np.array(cache.images[idx_te])
        metrics = quality_report(cfg, G_ema, test_np, cache.labels[idx_te], device, seed, d / "grid_test", inc)
        metrics["evaluated_on"] = "citra uji lipatan ini, UNTUK PELAPORAN (FR-11), tidak dipakai memilih apa pun"
        metrics["n_synthetic_generated"] = int(n_syn)
    metrics["lambda_roi"] = lam
    metrics["final_fid_val_monitor"] = out["history"][-1].get("fid_val") if out["history"] else None
    atomic_write_json(d / "gan_metrics.json", metrics)
    mark_done(d, h)
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--fold", type=int, required=True)
    a = ap.parse_args()
    cfg = load_merged(a.config)
    lam = resolve_lambda(cfg)
    cfg["gan"]["lambda_roi"] = lam
    run_gan_job(cfg, a.rep, a.fold, lam, job_dir(cfg, f"{cfg['scenario']}_gan", a.rep, a.fold))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
