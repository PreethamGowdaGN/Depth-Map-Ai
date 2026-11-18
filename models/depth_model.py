"""DepthNet — lightweight RGB-guided depth upsampling / completion CNN.

Input (all at the OUTPUT resolution H×W):
    rgb   : (B,3,H,W) guidance image in [0,1]  (zeros when RGB is disabled)
    init  : (B,1,H,W) initial depth = hole-filled LR depth, bicubic-upsampled,
            normalised to [0,1] per image
    conf  : (B,1,H,W) confidence = upsampled validity mask (1 = measured)

Output: refined depth (B,1,H,W) = init + predicted residual.

Architecture (≈1.5 M parameters)
    • Two encoders: RGB branch and depth branch, each 4 scales (1, 1/2, 1/4, 1/8)
      built from residual blocks.
    • Gated fusion at every scale: RGB features pass through a learned
      sigmoid gate that is computed from BOTH branches. The gate can close
      on texture edges that have no depth support — the network-level
      counterpart of our classical texture-copy prevention.
    • Multi-scale decoder with skip connections; a residual depth
      prediction is produced at 1/4, 1/2 and full resolution (deep
      supervision during training, coarse-to-fine refinement).
    • Predicting a *residual* over the bicubic initialisation means the
      network only has to learn corrections (edges, holes, noise), which
      trains fast and keeps smooth regions stable.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return x + self.c2(self.act(self.c1(x)))


def down(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.LeakyReLU(0.1, inplace=True), ResBlock(cout))


class GatedFusion(nn.Module):
    """fused = depth_feat + gate(rgb, depth) * proj(rgb_feat)."""

    def __init__(self, ch: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(2 * ch, ch, 3, padding=1), nn.Sigmoid())
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, f_rgb, f_d):
        g = self.gate(torch.cat([f_rgb, f_d], 1))
        return f_d + g * self.proj(f_rgb)


class DepthNet(nn.Module):
    def __init__(self, base: int = 16):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 6
        act = nn.LeakyReLU(0.1, inplace=True)
        # encoders
        self.rgb0 = nn.Sequential(nn.Conv2d(3, c1, 3, padding=1), act, ResBlock(c1))
        self.rgb1, self.rgb2, self.rgb3 = down(c1, c2), down(c2, c3), down(c3, c4)
        self.d0 = nn.Sequential(nn.Conv2d(2, c1, 3, padding=1), nn.LeakyReLU(0.1, inplace=True), ResBlock(c1))
        self.d1, self.d2, self.d3 = down(c1, c2), down(c2, c3), down(c3, c4)
        self.f0, self.f1, self.f2, self.f3 = GatedFusion(c1), GatedFusion(c2), GatedFusion(c3), GatedFusion(c4)
        # bottleneck (large receptive field for big holes)
        self.bott = nn.Sequential(ResBlock(c4), ResBlock(c4))
        # decoder
        self.up2 = nn.Sequential(nn.Conv2d(c4 + c3, c3, 3, padding=1), nn.LeakyReLU(0.1, inplace=True), ResBlock(c3))
        self.up1 = nn.Sequential(nn.Conv2d(c3 + c2, c2, 3, padding=1), nn.LeakyReLU(0.1, inplace=True), ResBlock(c2))
        self.up0 = nn.Sequential(nn.Conv2d(c2 + c1, c1, 3, padding=1), nn.LeakyReLU(0.1, inplace=True), ResBlock(c1))
        self.head2 = nn.Conv2d(c3, 1, 3, padding=1)
        self.head1 = nn.Conv2d(c2, 1, 3, padding=1)
        self.head0 = nn.Conv2d(c1, 1, 3, padding=1)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, rgb, init, conf, return_all: bool = False):
        r0 = self.rgb0(rgb); r1 = self.rgb1(r0); r2 = self.rgb2(r1); r3 = self.rgb3(r2)
        d0 = self.d0(torch.cat([init, conf], 1)); d1 = self.d1(d0); d2 = self.d2(d1); d3 = self.d3(d2)
        s0, s1, s2 = self.f0(r0, d0), self.f1(r1, d1), self.f2(r2, d2)
        x = self.bott(self.f3(r3, d3))
        x2 = self.up2(torch.cat([self._up(x, s2), s2], 1))
        x1 = self.up1(torch.cat([self._up(x2, s1), s1], 1))
        x0 = self.up0(torch.cat([self._up(x1, s0), s0], 1))
        # coarse-to-fine residuals
        res2 = self.head2(x2)
        res1 = self.head1(x1) + self._up(res2, x1)
        res0 = self.head0(x0) + self._up(res1, x0)
        out = init + res0
        if return_all:
            return out, [init + self._up(res1, init), init + self._up(res2, init)]
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
