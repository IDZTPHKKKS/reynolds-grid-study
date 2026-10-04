"""Small fully-convolutional residual CNN for next-step vorticity prediction."""
from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, 3, padding=1, padding_mode="circular")
        self.conv2 = nn.Conv2d(width, width, 3, padding=1, padding_mode="circular")
        self.act = nn.GELU()

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return x + h


class VorticityCNN(nn.Module):
    """Residual CNN that predicts omega(t + dt) = omega(t) + f(x)."""

    def __init__(self, in_channels: int, width: int = 32, n_blocks: int = 4):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, width, 3, padding=1, padding_mode="circular")
        self.blocks = nn.ModuleList([ResBlock(width) for _ in range(n_blocks)])
        self.head = nn.Conv2d(width, 1, 3, padding=1, padding_mode="circular")
        self.act = nn.GELU()

    def forward(self, x):
        omega_channel = x[:, :1]
        h = self.act(self.stem(x))
        for b in self.blocks:
            h = b(h)
        residual = self.head(h)
        return omega_channel + residual

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class SpectralConv2d(nn.Module):
    """Fourier layer (Li et al. 2021): multiply the lowest `modes` Fourier modes by
    learned complex weights."""

    def __init__(self, width: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (width * width)
        # two blocks: positive and negative ky (rfft keeps kx >= 0 only)
        self.w1 = nn.Parameter(scale * torch.randn(width, width, modes, modes, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.randn(width, width, modes, modes, dtype=torch.cfloat))

    def forward(self, x):
        b, c, ny, nx = x.shape
        # rfft/irfft on CPU-compatible complex path; MPS lacks some complex ops
        dev = x.device
        x_hat = torch.fft.rfft2(x.cpu() if dev.type == "mps" else x)
        m = self.modes
        out = torch.zeros(b, c, ny, nx // 2 + 1, dtype=torch.cfloat, device=x_hat.device)
        w1 = self.w1.to(x_hat.device); w2 = self.w2.to(x_hat.device)
        out[:, :, :m, :m] = torch.einsum("bixy,ioxy->boxy", x_hat[:, :, :m, :m], w1)
        out[:, :, -m:, :m] = torch.einsum("bixy,ioxy->boxy", x_hat[:, :, -m:, :m], w2)
        return torch.fft.irfft2(out, s=(ny, nx)).to(dev)


class FNO2d(nn.Module):
    """Small FNO with the same interface as VorticityCNN: input channels -> next-step
    vorticity as `omega_channel + residual(x)`."""

    def __init__(self, in_channels: int, width: int = 32, n_layers: int = 4, modes: int = 12):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.spectral = nn.ModuleList([SpectralConv2d(width, modes) for _ in range(n_layers)])
        self.pointwise = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.proj = nn.Sequential(nn.Conv2d(width, 64, 1), nn.GELU(), nn.Conv2d(64, 1, 1))
        self.act = nn.GELU()

    def forward(self, x):
        omega_channel = x[:, :1]
        h = self.lift(x)
        for i, (s, p) in enumerate(zip(self.spectral, self.pointwise)):
            h = s(h) + p(h)
            if i < len(self.spectral) - 1:
                h = self.act(h)
        return omega_channel + self.proj(h)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class _UBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, padding_mode="circular"), nn.GELU(),
            nn.Conv2d(c_out, c_out, 3, padding=1, padding_mode="circular"), nn.GELU())

    def forward(self, x):
        return self.net(x)


class UNet2d(nn.Module):
    """Small periodic U-Net (3 levels) with the same residual output as VorticityCNN."""

    def __init__(self, in_channels: int, width: int = 32):
        super().__init__()
        w = width
        self.enc1, self.enc2, self.enc3 = _UBlock(in_channels, w), _UBlock(w, 2 * w), _UBlock(2 * w, 4 * w)
        self.pool = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2, self.dec1 = _UBlock(6 * w, 2 * w), _UBlock(3 * w, w)
        self.head = nn.Conv2d(w, 1, 3, padding=1, padding_mode="circular")

    def forward(self, x):
        omega_channel = x[:, :1]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        return omega_channel + self.head(d1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
