"""ADA-lite: augmentasi diferensiabel dengan probabilitas p yang disesuaikan otomatis (bukan ADA resmi).
Operasi: transpose (bersama flip menghasilkan rot90), flip, translasi (refleksi), kecerahan, kontras (hanya warna=True)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def augment(x: torch.Tensor, p: float, color: bool = True) -> torch.Tensor:
    if p <= 0:
        return x
    B, dev = x.shape[0], x.device

    def flag(shape=(B, 1, 1, 1)):
        return (torch.rand(shape, device=dev) < p).to(x.dtype)

    x = torch.where(flag().bool(), x.transpose(2, 3), x)
    fx = 1 - 2 * flag((B,))
    fy = 1 - 2 * flag((B,))
    tx = (torch.rand(B, device=dev) * 2 - 1) * 0.25 * flag((B,))
    ty = (torch.rand(B, device=dev) * 2 - 1) * 0.25 * flag((B,))
    theta = torch.zeros(B, 2, 3, device=dev, dtype=x.dtype)
    theta[:, 0, 0], theta[:, 1, 1], theta[:, 0, 2], theta[:, 1, 2] = fx, fy, tx, ty
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    if color:
        x = x + torch.randn(B, 1, 1, 1, device=dev, dtype=x.dtype) * 0.2 * flag()
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) * torch.exp2(torch.randn(B, 1, 1, 1, device=dev, dtype=x.dtype) * 0.5 * flag()) + mean
    return x
