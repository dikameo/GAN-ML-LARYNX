"""Generator kondisional-kelas bergaya StyleGAN2 (konvolusi termodulasi, noise, skip-RGB) dan diskriminator residual
dengan proyeksi kelas. Ringkas: tanpa style mixing, path-length regularization, dan equalized learning rate (lihat D8)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolutions(resolution: int) -> list[int]:
    r, out = 4, []
    while r <= resolution:
        out.append(r)
        r *= 2
    return out


class ModConv(nn.Module):
    def __init__(self, cin: int, cout: int, w_dim: int, k: int = 3, demod: bool = True, up: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(cout, cin, k, k) / math.sqrt(cin * k * k))
        self.affine = nn.Linear(w_dim, cin)
        nn.init.constant_(self.affine.bias, 1.0)
        self.k, self.demod, self.up, self.cout = k, demod, up, cout

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        s = self.affine(w)
        wt = self.weight[None] * s[:, None, :, None, None]
        if self.demod:
            wt = wt * torch.rsqrt(wt.square().sum(dim=(2, 3, 4), keepdim=True) + 1e-8)
        if self.up:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            H, W = H * 2, W * 2
        x = x.reshape(1, B * C, H, W)
        out = F.conv2d(x, wt.reshape(B * self.cout, C, self.k, self.k), padding=self.k // 2, groups=B)
        return out.reshape(B, self.cout, H, W)


class SynthLayer(nn.Module):
    def __init__(self, cin, cout, w_dim, up=False):
        super().__init__()
        self.conv = ModConv(cin, cout, w_dim, 3, True, up)
        self.noise_strength = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(cout))

    def forward(self, x, w):
        x = self.conv(x, w)
        x = x + torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype) * self.noise_strength
        return F.leaky_relu(x + self.bias.view(1, -1, 1, 1), 0.2)


class Generator(nn.Module):
    def __init__(self, z_dim: int, w_dim: int, n_classes: int, channels: dict, resolution: int, map_layers: int):
        super().__init__()
        self.z_dim, self.n_classes, self.resolution = z_dim, n_classes, resolution
        self.embed = nn.Linear(n_classes, w_dim)
        layers, d = [], z_dim + w_dim
        for _ in range(map_layers):
            layers += [nn.Linear(d, w_dim), nn.LeakyReLU(0.2)]
            d = w_dim
        self.mapping = nn.Sequential(*layers)
        ch = {int(k): v for k, v in channels.items()}
        self.const = nn.Parameter(torch.randn(ch[4], 4, 4))
        self.res = _resolutions(resolution)
        self.conv0, self.conv1, self.torgb = nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict()
        for r in self.res:
            if r > 4:
                self.conv0[str(r)] = SynthLayer(ch[r // 2], ch[r], w_dim, up=True)
            self.conv1[str(r)] = SynthLayer(ch[r], ch[r], w_dim)
            self.torgb[str(r)] = ModConv(ch[r], 3, w_dim, k=1, demod=False)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = z * torch.rsqrt(z.square().mean(dim=1, keepdim=True) + 1e-8)
        c = self.embed(F.one_hot(y, self.n_classes).to(z.dtype))
        c = c * torch.rsqrt(c.square().mean(dim=1, keepdim=True) + 1e-8)
        w = self.mapping(torch.cat([z, c], dim=1))
        x = self.const[None].expand(z.shape[0], -1, -1, -1)
        img = None
        for r in self.res:
            if r > 4:
                x = self.conv0[str(r)](x, w)
            x = self.conv1[str(r)](x, w)
            rgb = self.torgb[str(r)](x, w)
            img = rgb if img is None else F.interpolate(img, scale_factor=2, mode="bilinear", align_corners=False) + rgb
        return img


class DBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.c0 = nn.Conv2d(cin, cin, 3, padding=1)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1, bias=False)

    def forward(self, x):
        y = F.avg_pool2d(F.leaky_relu(self.c1(F.leaky_relu(self.c0(x), 0.2)), 0.2), 2)
        return (self.skip(F.avg_pool2d(x, 2)) + y) / math.sqrt(2)


def minibatch_std(x: torch.Tensor, group: int = 4) -> torch.Tensor:
    B, C, H, W = x.shape
    G = min(group, B)
    while B % G:
        G -= 1
    y = x.reshape(G, -1, C, H, W)
    y = y - y.mean(dim=0)
    y = (y.square().mean(dim=0) + 1e-8).sqrt().mean(dim=(1, 2, 3))
    return torch.cat([x, y.reshape(-1, 1, 1, 1).repeat(G, 1, H, W)], dim=1)


class Discriminator(nn.Module):
    def __init__(self, in_ch: int, n_classes: int, channels: dict, resolution: int):
        super().__init__()
        ch = {int(k): v for k, v in channels.items()}
        self.res = _resolutions(resolution)[::-1]  # besar -> kecil
        self.fromrgb = nn.Conv2d(in_ch, ch[self.res[0]], 1)
        self.blocks = nn.ModuleList([DBlock(ch[r], ch[r // 2]) for r in self.res[:-1]])
        c4 = ch[4]
        self.conv = nn.Conv2d(c4 + 1, c4, 3, padding=1)
        self.fc = nn.Linear(c4 * 16, c4)
        self.out = nn.Linear(c4, 1)
        self.embed = nn.Embedding(n_classes, c4)
        self.c4 = c4

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.fromrgb(x), 0.2)
        for b in self.blocks:
            h = b(h)
        h = F.leaky_relu(self.conv(minibatch_std(h)), 0.2)
        h = F.leaky_relu(self.fc(h.flatten(1)), 0.2)
        return self.out(h) + (h * self.embed(y)).sum(dim=1, keepdim=True) / math.sqrt(self.c4)
