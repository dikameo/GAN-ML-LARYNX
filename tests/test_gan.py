"""Uji GAN (S5/S6) pada model sangat kecil dengan data ACAK. Hanya membuktikan kode berjalan dan bentuk keluaran benar;
tidak ada klaim tentang kualitas citra. FID/LPIPS (butuh unduhan bobot) tidak diuji di sini."""
import copy

import numpy as np
import pytest
import torch

from src.gan.augment import augment
from src.gan.networks import Discriminator, Generator
from src.gan.vesselness import roi_input, roi_soft
from src.train_gan import build_nets, sample_u8, train_core


def tiny(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["gan"].update(resolution=32, z_dim=16, w_dim=16, map_layers=2, channels={4: 16, 8: 16, 16: 16, 32: 8},
                      d_channels={32: 8, 16: 16, 8: 16, 4: 16}, batch_size=4, total_kimg=0.02, snapshot_kimg=0.008,
                      r1_interval=2, fid_samples=8)
    return cfg


def test_network_shapes_and_roi_range(synth):
    cfg = tiny(synth[0])
    G, D, Droi = build_nets(cfg, 1.0, "cpu")
    y = torch.tensor([0, 1, 0, 1])
    x = G(torch.randn(4, 16), y)
    assert x.shape == (4, 3, 32, 32) and torch.isfinite(x).all()
    assert D(x, y).shape == (4, 1)
    roi = roi_soft(x.detach().clamp(-1, 1), cfg["gan"]["vesselness"])
    assert roi.shape == (4, 1, 32, 32) and float(roi.min()) >= 0 and float(roi.max()) <= 1
    assert Droi(roi_input(x.detach(), roi), y).shape == (4, 1)
    assert build_nets(cfg, 0.0, "cpu")[2] is None  # S5: tanpa D_roi


def test_vesselness_responds_to_dark_line():
    v = {"sigmas": [1.0, 1.5], "beta": 0.5, "c": 0.08, "blur": 0.5, "fov_thr": -0.9, "fov_erode": 2,
         "norm": 0.25, "threshold": 0.2, "temperature": 0.05}
    img = torch.zeros(1, 3, 32, 32)       # latar abu (0.0) di dalam area pandang
    img[:, :, :, 15:17] = -0.5            # garis gelap mirip pembuluh
    roi = roi_soft(img, v)
    assert float(roi[0, 0, 16, 16]) > float(roi[0, 0, 16, 6]) + 0.3


def test_augment_identity_at_zero_and_shape_preserved():
    x = torch.randn(4, 3, 32, 32)
    assert torch.equal(augment(x, 0.0), x)
    assert augment(x, 1.0).shape == x.shape and augment(torch.randn(4, 4, 32, 32), 1.0, color=False).shape == (4, 4, 32, 32)


@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_train_core_runs_resumes_and_samples(synth, tmp_path, lam):
    cfg = tiny(synth[0])
    real = torch.randint(0, 255, (16, 3, 32, 32), dtype=torch.uint8)
    labels = torch.tensor([0, 1] * 8)
    out = train_core(cfg, real, labels, None, lam, 0.02, tmp_path, "h1", 1, "cpu")
    assert (tmp_path / "G_ema.pt").exists() and (tmp_path / "last.pt").exists() and len(out["history"]) >= 2
    n_hist = len(out["history"])
    out2 = train_core(cfg, real, labels, None, lam, 0.02, tmp_path, "h1", 1, "cpu")  # sudah selesai -> tidak melatih ulang
    assert len(out2["history"]) == n_hist
    imgs = sample_u8(out["G_ema"], 5, 1, None, 4, "cpu", 0, 64)
    assert imgs.shape == (5, 64, 64, 3) and imgs.dtype == np.uint8
