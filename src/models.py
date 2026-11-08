"""Swin-T (Hugging Face transformers): klasifikator S1/S5/S6/S7 dan encoder beku S2. Util perangkat (CPU/GPU)."""
from __future__ import annotations

import contextlib
import os

import numpy as np
import torch
from torch.utils.data import DataLoader


def configure_threads(cfg: dict) -> int:
    """Thread torch = compute.torch_threads bila diisi, selain itu semua CPU yang terlihat."""
    n = (cfg.get("compute") or {}).get("torch_threads") or os.cpu_count() or 1
    torch.set_num_threads(int(n))
    return int(n)


def device_of(cfg: dict | None = None) -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def num_workers(cfg: dict) -> int:
    v = cfg["loader"]["num_workers"]
    if v == "auto":
        return 0 if device_of() == "cpu" else max(0, min(6, (os.cpu_count() or 2) - 1))
    return int(v)


def use_amp(cfg: dict) -> bool:
    return bool((cfg.get("compute") or {}).get("amp", False)) and device_of() == "cuda"


def autocast(cfg: dict):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp(cfg) else contextlib.nullcontext()


def make_loader(ds, cfg: dict, batch_size: int, **kw) -> DataLoader:
    nw = num_workers(cfg)
    return DataLoader(ds, batch_size=batch_size, num_workers=nw, pin_memory=device_of() == "cuda",
                      persistent_workers=nw > 0, **kw)


def _swin_config(cfg: dict):
    from transformers import AutoConfig, SwinConfig

    m = cfg["model"]
    if m["pretrained"]:
        conf = AutoConfig.from_pretrained(m["hf_name"], cache_dir=cfg["paths"]["hf_cache"])
    else:  # uji tiruan: arsitektur kecil tanpa bobot, tidak memerlukan jaringan
        conf = SwinConfig(image_size=m["input_size"])
    for k, v in (m.get("arch_overrides") or {}).items():
        setattr(conf, k, v)
    conf.num_labels = m["num_classes"]
    return conf


def build_classifier(cfg: dict):
    from transformers import AutoModelForImageClassification, SwinForImageClassification

    m = cfg["model"]
    if m["pretrained"]:
        return AutoModelForImageClassification.from_pretrained(
            m["hf_name"], cache_dir=cfg["paths"]["hf_cache"], num_labels=m["num_classes"], ignore_mismatched_sizes=True)
    return SwinForImageClassification(_swin_config(cfg))


def build_encoder(cfg: dict):
    from transformers import SwinModel

    m = cfg["model"]
    if m["pretrained"]:
        return SwinModel.from_pretrained(m["hf_name"], cache_dir=cfg["paths"]["hf_cache"])
    return SwinModel(_swin_config(cfg))


@torch.no_grad()
def predict_probs(model, ds, batch_size: int, device: str, cfg: dict | None = None) -> np.ndarray:
    """Probabilitas kelas ganas (indeks 1) untuk tiap citra pada ds, urutan sama dengan ds."""
    model.eval()
    out = []
    loader = make_loader(ds, cfg, batch_size, shuffle=False) if cfg else DataLoader(ds, batch_size=batch_size)
    for x, _, _ in loader:
        with (autocast(cfg) if cfg else contextlib.nullcontext()):
            logits = model(pixel_values=x.to(device, non_blocking=True)).logits
        out.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


@torch.no_grad()
def embed(encoder, ds, batch_size: int, device: str, cfg: dict | None = None):
    encoder.eval()
    loader = make_loader(ds, cfg, batch_size, shuffle=False) if cfg else DataLoader(ds, batch_size=batch_size)
    for x, _, j in loader:
        with (autocast(cfg) if cfg else contextlib.nullcontext()):
            e = encoder(pixel_values=x.to(device, non_blocking=True)).pooler_output
        yield j.numpy(), e.float().cpu().numpy()
