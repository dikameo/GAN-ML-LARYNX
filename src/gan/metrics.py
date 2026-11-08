"""FR-11: FID, jarak Wasserstein-1 fraksi area pembuluh (filter SATO, bukan Frangi: L6), keragaman LPIPS, grid sampel.

FID di sini memakai Inception-v3 ImageNet dari torchvision (bukan TF-Inception standar); nilainya tidak sebanding
langsung dengan angka literatur. Risiko FID pada CE-NBI dicatat di PRD §11: kinerja hilir adalah bukti utama."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from scipy.stats import wasserstein_distance


def load_inception(device: str):
    from torchvision.models import Inception_V3_Weights, inception_v3

    m = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True, transform_input=False)
    m.fc = torch.nn.Identity()
    return m.eval().to(device)


@torch.no_grad()
def inception_feats(model, imgs_u8: np.ndarray, device: str, bs: int = 64) -> np.ndarray:
    """imgs_u8 (N,H,W,3) uint8 -> (N,2048)."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(imgs_u8), bs):
        x = torch.from_numpy(np.ascontiguousarray(imgs_u8[i:i + bs])).to(device).permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        out.append(model((x - mean) / std).float().cpu().numpy())
    return np.concatenate(out)


def fid_from_feats(a: np.ndarray, b: np.ndarray) -> float:
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + off) @ (s2 + off))
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def vessel_fractions(imgs_u8: np.ndarray, q: dict) -> np.ndarray:
    """Fraksi area pembuluh per citra dengan filter Sato pada kanal hijau di dalam area pandang (L6)."""
    from scipy import ndimage as ndi
    from skimage.filters import sato

    out = []
    for im in imgs_u8:
        g = im[..., 1].astype(np.float32) / 255.0
        fov = ndi.binary_erosion(im.max(axis=2) > 0, iterations=4)
        if fov.sum() < 100:
            out.append(0.0)
            continue
        g = np.where(fov, g, g[fov].mean())
        v = sato(g, sigmas=q["vessel_sigmas"], black_ridges=True)
        out.append(float(((v > q["vessel_threshold"]) & fov).sum() / fov.sum()))
    return np.asarray(out)


def vessel_w1(real_u8: np.ndarray, fake_u8: np.ndarray, q: dict) -> float:
    return float(wasserstein_distance(vessel_fractions(real_u8, q), vessel_fractions(fake_u8, q)))


@torch.no_grad()
def lpips_diversity(imgs_u8: np.ndarray, n_pairs: int, net: str, device: str, seed: int) -> float:
    """Rata-rata LPIPS antar pasangan acak citra sintetis sekelas (lebih tinggi = lebih beragam)."""
    import lpips

    if len(imgs_u8) < 2:
        return float("nan")
    fn = lpips.LPIPS(net=net, verbose=False).to(device).eval()
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(imgs_u8), n_pairs)
    j = (i + rng.integers(1, len(imgs_u8), n_pairs)) % len(imgs_u8)
    vals = []
    for s in range(0, n_pairs, 32):
        a = torch.from_numpy(imgs_u8[i[s:s + 32]]).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1
        b = torch.from_numpy(imgs_u8[j[s:s + 32]]).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1
        vals.append(fn(a, b).flatten().cpu().numpy())
    return float(np.concatenate(vals).mean())


def save_grid(imgs_u8: np.ndarray, path: str | Path, n_col: int = 10) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(imgs_u8)
    n_row = -(-n // n_col)
    fig, axes = plt.subplots(n_row, n_col, figsize=(1.6 * n_col, 1.6 * n_row), squeeze=False)
    for k, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if k < n:
            ax.imshow(imgs_u8[k])
    fig.tight_layout(pad=0.1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)
