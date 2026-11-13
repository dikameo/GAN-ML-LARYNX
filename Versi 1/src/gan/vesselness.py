"""Frangi diferensiabel (torch) untuk D_roi. Real dan palsu melewati operator yang SAMA.

Ini aproksimasi untuk pelatihan, bukan skimage.filters.frangi: peta ROI biner untuk cache/Gambar 3.2 tetap dari
src.preprocess (skimage). Pembuluh gelap pada kanal hijau -> nilai eigen besar positif.
Citra masuk dalam rentang [-1, 1], (B,3,H,W).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _kernels(sigma: float, device, dtype):
    r = int(math.ceil(3 * sigma))
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    g1 = -x / sigma ** 2 * g
    g2 = (x ** 2 / sigma ** 4 - 1 / sigma ** 2) * g
    return g, g1, g2, r


def _sep(img: torch.Tensor, kx: torch.Tensor, ky: torch.Tensor, r: int) -> torch.Tensor:
    x = F.pad(img, (r, r, r, r), mode="reflect")
    x = F.conv2d(x, kx.view(1, 1, 1, -1))
    return F.conv2d(x, ky.view(1, 1, -1, 1))


def _blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return img
    g, _, _, r = _kernels(sigma, img.device, img.dtype)
    return _sep(img, g, g, r)


def fov_mask(rgb: torch.Tensor, thr: float, erode: int) -> torch.Tensor:
    """Area pandang (B,1,H,W) dari piksel yang tidak hitam; tanpa gradien."""
    with torch.no_grad():
        m = (rgb.max(dim=1, keepdim=True).values > thr).float()
        if erode > 0:
            k = 2 * erode + 1
            m = -F.max_pool2d(-m, k, stride=1, padding=erode)
    return m


def frangi_soft(gray: torch.Tensor, v: dict) -> torch.Tensor:
    """gray (B,1,H,W) dalam [0,1] -> respons vesselness (B,1,H,W) >= 0."""
    gray = _blur(gray, v["blur"])
    best = None
    for s in v["sigmas"]:
        g, g1, g2, r = _kernels(float(s), gray.device, gray.dtype)
        hxx = _sep(gray, g2, g, r) * s ** 2
        hyy = _sep(gray, g, g2, r) * s ** 2
        hxy = _sep(gray, g1, g1, r) * s ** 2
        tmp = torch.sqrt(((hxx - hyy) / 2) ** 2 + hxy ** 2 + 1e-12)
        l1, l2 = (hxx + hyy) / 2 + tmp, (hxx + hyy) / 2 - tmp
        small = torch.where(l1.abs() < l2.abs(), l1, l2)
        large = torch.where(l1.abs() < l2.abs(), l2, l1)
        rb = small.abs() / (large.abs() + 1e-8)
        s2 = small ** 2 + large ** 2
        resp = torch.exp(-rb ** 2 / (2 * v["beta"] ** 2)) * (1 - torch.exp(-s2 / (2 * v["c"] ** 2)))
        resp = torch.where(large > 0, resp, torch.zeros_like(resp))  # pembuluh gelap
        best = resp if best is None else torch.maximum(best, resp)
    return best


def roi_soft(rgb: torch.Tensor, v: dict) -> torch.Tensor:
    """Peta ROI lunak (B,1,H,W) dalam [0,1]; dibatasi ke area pandang yang telah diperkecil."""
    fov = fov_mask(rgb, v["fov_thr"], v["fov_erode"])
    green = (rgb[:, 1:2] + 1) / 2
    mean = (green * fov).sum(dim=(1, 2, 3), keepdim=True) / (fov.sum(dim=(1, 2, 3), keepdim=True) + 1e-6)
    green = torch.where(fov > 0, green, mean.expand_as(green))  # isi luar area dengan rata-rata, seperti skimage
    resp = frangi_soft(green, v) / v["norm"]
    return torch.sigmoid((resp - v["threshold"]) / v["temperature"]) * fov


def roi_input(rgb: torch.Tensor, roi: torch.Tensor) -> torch.Tensor:
    """Masukan D_roi (B,4,H,W): citra yang dimaskir ROI (di luar ROI = hitam) + peta ROI itu sendiri, rentang [-1,1]."""
    return torch.cat([(rgb + 1) * roi - 1, roi * 2 - 1], dim=1)
