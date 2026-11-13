"""Klasifikator Swin-T pralatih (FR-05): S1, dan S5/S6/S7 bila `synthetic` diisi. Satu job = satu (rep, lipatan).

Jalankan: python -m src.train --config configs/s1.yaml --rep 0 --fold 0
          python -m src.train --config configs/s6.yaml --rep 0 --fold 0 --ratio 0.25
- last.pt ditulis atomik tiap epoch; terputus -> dilanjutkan dari last.pt
- DONE berisi hash konfigurasi; job DONE dengan hash sama dilewati
- data uji baru disentuh SETELAH pelatihan selesai (model terbaik dipilih dari validasi dalam lipatan latih)
- citra sintetis hanya ditambahkan ke data LATIH; validasi dan uji hanya citra asli
"""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import WeightedRandomSampler

from src.config import P, load_merged
from src.data import Cache, ImageDS, find_cache
from src.jobio import balanced_logloss, job_dir, resolution_map, write_preds, write_run_info
from src.leakage import check_l3_job, check_splits
from src.models import (autocast, build_classifier, configure_threads, device_of, make_loader, predict_probs,
                        use_amp)
from src.splits import get_fold, load_splits
from src.utils import (atomic_torch_save, atomic_write_bytes, atomic_write_json, config_hash, is_done, job_seed,
                       mark_done, set_seed)


def resolve_ratio(cfg: dict) -> float | None:
    syn = cfg.get("synthetic")
    if not syn:
        return None
    r = syn.get("ratio")
    if r == "best":
        bp = P(cfg, "results", "best_config.json")
        if not bp.exists():
            raise FileNotFoundError(f"ratio 'best' butuh {bp} (jalankan src.evaluate --select-best setelah S5/S6 selesai)")
        return float(json.loads(bp.read_text(encoding="utf-8"))[syn["best_from"]]["ratio"])
    if r is None:
        raise ValueError("synthetic.ratio belum diisi (gunakan --ratio atau 'best')")
    return float(r)


def scenario_name(cfg: dict) -> str:
    syn = cfg.get("synthetic")
    if syn and syn.get("ratio") not in (None, "best"):
        return f"{cfg['scenario']}_r{int(round(float(syn['ratio']) * 100))}"
    return cfg["scenario"]


def load_synthetic(cfg: dict, rep: int, fold: int, sf: dict) -> tuple[np.ndarray | None, dict]:
    """Muat subset citra sintetis (bersarang: awalan dari urutan acak tetap) dan periksa L3."""
    ratio = resolve_ratio(cfg)
    if ratio is None:
        return None, {}
    gdir = job_dir(cfg, f"{cfg['synthetic']['gan_scenario']}_gan", rep, fold)
    if not (gdir / "DONE").exists():
        raise FileNotFoundError(f"GAN {gdir} belum DONE; jalankan python -m src.train_gan lebih dulu (urutan wajib, PRD §8)")
    meta = json.loads((gdir / "synth_meta.json").read_text(encoding="utf-8"))
    check_l3_job({"train": sf["train"], "val": sf["val"], "test": sf["test"]}, sf, synthetic_source_patients=meta["source_patients"])
    arr = np.load(gdir / "synth.npz")["images"]
    perm = np.random.default_rng(job_seed(cfg["seed"], "synorder", rep, fold)).permutation(len(arr))
    k = int(round(ratio * len(arr)))
    return arr[perm[:k]], {"ratio": ratio, "n_synthetic": int(k), "n_generated": int(len(arr)), "gan_dir": str(gdir)}


def run_job(cfg: dict, rep: int, fold: int, cache_dir=None) -> str:
    scenario = scenario_name(cfg)
    splits = load_splits(cfg)
    check_splits(splits, resolution_map(cfg) if splits.get("kind") == "s7" else None)  # L1/L5 sebelum job apa pun
    cache = Cache(cache_dir or find_cache(cfg))
    d = job_dir(cfg, scenario, rep, fold)
    h = config_hash({"cfg": cfg, "rep": rep, "fold": fold, "cache": cache.hash, "splits": splits["meta"]["hash"]})
    if is_done(d, h):
        print(f"[dilewati] {scenario} rep{rep} fold{fold} sudah DONE")
        return str(d)
    d.mkdir(parents=True, exist_ok=True)

    sf = get_fold(splits, rep, fold)
    atomic_write_json(d / "patients.json", {"train": sf["train"], "val": sf["val"], "test": sf["test"]})
    check_l3_job(json.loads((d / "patients.json").read_text(encoding="utf-8")), sf)  # L3
    syn_arr, syn_info = load_synthetic(cfg, rep, fold, sf)

    seed = job_seed(cfg["seed"], scenario, rep, fold)
    set_seed(seed)
    threads = configure_threads(cfg)
    device = device_of(cfg)
    tr = cfg["train"]
    amp = use_amp(cfg)

    idx_tr, idx_va = cache.indices_of(sf["train"]), cache.indices_of(sf["val"])
    ds_tr = ImageDS(cache, idx_tr, cfg, True, synthetic=syn_arr)
    ds_va = ImageDS(cache, idx_va, cfg, False)
    y_tr = ds_tr.labels
    cls_count = np.bincount(y_tr, minlength=2).astype(float)
    weights = (1.0 / np.maximum(cls_count, 1))[y_tr] if tr["weighted_sampling"] else np.ones(len(y_tr))

    model = build_classifier(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    lossf = torch.nn.CrossEntropyLoss()

    st = {"epoch": 0, "best": float("inf"), "bad": 0, "history": []}
    last = d / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        if ck.get("config_hash") == h:
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            if ck.get("scaler"):
                scaler.load_state_dict(ck["scaler"])
            st = {k: ck[k] for k in ("epoch", "best", "bad", "history")}
            torch.set_rng_state(ck["rng_torch"])
            np.random.set_state(ck["rng_np"])
            random.setstate(ck["rng_py"])
            print(f"[lanjut] dari epoch {st['epoch']}")
        else:
            print("[peringatan] last.pt dari konfigurasi lain diabaikan")

    write_run_info(d, cfg, h, seed, {"rep": rep, "fold": fold, "cache_hash": cache.hash, "scenario_job": scenario,
                                     "splits_hash": splits["meta"]["hash"], "splits_kind": splits.get("kind"),
                                     "device": device, "amp": amp, "torch_threads": threads,
                                     "n_train_real_images": int(len(idx_tr)), "n_val_images": int(len(idx_va)),
                                     **syn_info})

    while st["epoch"] < tr["max_epochs"] and st["bad"] < tr["early_stop_patience"]:
        ep, t0 = st["epoch"], time.time()
        g = torch.Generator()
        g.manual_seed(seed + ep)
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(ds_tr),
                                        replacement=True, generator=g)
        model.train()
        tot, n = 0.0, 0
        for x, y, _ in make_loader(ds_tr, cfg, tr["batch_size"], sampler=sampler):
            opt.zero_grad(set_to_none=True)
            with autocast(cfg):
                logits = model(pixel_values=x.to(device, non_blocking=True)).logits
            loss = lossf(logits.float(), y.to(device, non_blocking=True))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(y)
            n += len(y)
        pv = predict_probs(model, ds_va, tr["batch_size"], device, cfg)
        val_loss = balanced_logloss(ds_va.labels, pv)
        try:
            pat_auc = float(roc_auc_score(*_patient_arrays(cache, idx_va, pv)))
        except ValueError:
            pat_auc = float("nan")
        st["history"].append({"epoch": ep + 1, "train_loss": tot / max(n, 1), "val_loss_balanced": val_loss,
                              "val_auc_patient": pat_auc, "seconds": time.time() - t0})
        print(st["history"][-1], flush=True)
        if val_loss < st["best"] - tr["min_delta"]:
            st["best"], st["bad"] = val_loss, 0
            atomic_torch_save(model.state_dict(), d / "best.pt")
        else:
            st["bad"] += 1
        st["epoch"] = ep + 1
        if st["epoch"] % tr["checkpoint_every_epochs"] == 0:
            atomic_torch_save({"model": model.state_dict(), "opt": opt.state_dict(), "config_hash": h,
                               "scaler": scaler.state_dict() if amp else None,
                               "rng_torch": torch.get_rng_state(), "rng_np": np.random.get_state(),
                               "rng_py": random.getstate(), **st}, last)

    # ---- pelatihan selesai; baru sekarang data uji disentuh
    model.load_state_dict(torch.load(d / "best.pt", map_location="cpu"))
    thr = cfg["eval"]["threshold"]
    write_preds(d, "val", cache.meta, idx_va, predict_probs(model, ds_va, tr["batch_size"], device, cfg), thr)
    idx_te = cache.indices_of(sf["test"])
    ds_te = ImageDS(cache, idx_te, cfg, False)
    write_preds(d, "test", cache.meta, idx_te, predict_probs(model, ds_te, tr["batch_size"], device, cfg), thr)
    atomic_write_bytes(d / "history.csv", pd.DataFrame(st["history"]).to_csv(index=False).encode("utf-8"))
    mark_done(d, h)
    return str(d)


def _patient_arrays(cache: Cache, idx: np.ndarray, probs: np.ndarray):
    df = pd.DataFrame({"p": cache.meta.patient_uid.to_numpy()[idx], "y": cache.labels[idx], "s": probs})
    g = df.groupby("p").agg(y=("y", "first"), s=("s", "mean"))
    return g.y.to_numpy(), g.s.to_numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--ratio", type=float, default=None, help="rasio sintetis (S5/S6): 0.25, 0.5, 1.0")
    a = ap.parse_args()
    cfg = load_merged(a.config)
    if a.ratio is not None:
        if not cfg.get("synthetic"):
            raise SystemExit("--ratio diberikan tetapi konfigurasi tidak punya blok synthetic")
        cfg["synthetic"]["ratio"] = a.ratio
    run_job(cfg, a.rep, a.fold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
