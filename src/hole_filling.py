"""Hole filling for missing depth measurements.

Two complementary methods:

1. ``push_pull_fill`` — multi-scale normalised convolution. Builds a
   pyramid of (depth*mask, mask), averages valid samples at coarser and
   coarser levels until every pixel has support, then pulls values back
   down only into the holes. Always fills everything; smooth.

2. ``guided_fill`` — edge-aware diffusion. Each hole pixel takes a
   weighted average of valid neighbours, where the weight drops when the
   RGB colour differs (likely a different object). Repeating this grows
   the filled region inward from hole borders without leaking across
   object boundaries. Remaining pixels are finished by push-pull.
"""
from __future__ import annotations

import cv2
import numpy as np


def _shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift array by (dy, dx) with zero padding (no wrap-around)."""
    out = np.zeros_like(a)
    h, w = a.shape[:2]
    ys_src = slice(max(0, -dy), min(h, h - dy))
    ys_dst = slice(max(0, dy), min(h, h + dy))
    xs_src = slice(max(0, -dx), min(w, w - dx))
    xs_dst = slice(max(0, dx), min(w, w + dx))
    out[ys_dst, xs_dst] = a[ys_src, xs_src]
    return out


def push_pull_fill(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    if mask.all():
        return depth.copy()
    if not mask.any():
        return np.zeros_like(depth)
    levels = [(depth * mask, mask.astype(np.float32))]
    while True:
        num, den = levels[-1]
        if (den > 0).all() or min(num.shape) <= 2:
            break
        h, w = num.shape
        h2, w2 = (h + 1) // 2, (w + 1) // 2
        pn = np.zeros((h2 * 2, w2 * 2), np.float32)
        pd = np.zeros_like(pn)
        pn[:h, :w] = num
        pd[:h, :w] = den
        num2 = pn.reshape(h2, 2, w2, 2).sum((1, 3))
        den2 = pd.reshape(h2, 2, w2, 2).sum((1, 3))
        levels.append((num2, den2))
    # coarsest estimate
    num, den = levels[-1]
    est = np.where(den > 0, num / np.maximum(den, 1e-9), 0).astype(np.float32)
    if not (den > 0).all():
        est[den <= 0] = est[den > 0].mean() if (den > 0).any() else 0
    for num, den in reversed(levels[:-1]):
        h, w = num.shape
        up = cv2.resize(est, (w, h), interpolation=cv2.INTER_LINEAR)
        cur = np.where(den > 0, num / np.maximum(den, 1e-9), 0)
        # blend by support: strong support keeps own value, weak support -> coarse
        alpha = np.clip(den, 0, 1)
        est = (alpha * cur + (1 - alpha) * up).astype(np.float32)
    out = depth.copy()
    out[~mask] = est[~mask]
    return out


def guided_fill(depth: np.ndarray, mask: np.ndarray, rgb: np.ndarray | None = None,
                sigma_color: float = 0.08, radius: int = 2, iterations: int = 12) -> np.ndarray:
    """Edge-aware iterative hole filling (RGB-guided if rgb given)."""
    d = depth.astype(np.float32).copy()
    m = mask.copy()
    if m.all():
        return d
    guide = None
    if rgb is not None:
        guide = rgb if rgb.shape[:2] == d.shape else cv2.resize(rgb, d.shape[::-1], interpolation=cv2.INTER_AREA)
        guide = guide.astype(np.float32)
    offsets = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)
               if (dy, dx) != (0, 0)]
    for _ in range(iterations):
        holes = ~m
        if not holes.any():
            break
        num = np.zeros_like(d)
        den = np.zeros_like(d)
        mf = m.astype(np.float32)
        for dy, dx in offsets:
            ws = np.exp(-(dy * dy + dx * dx) / (2 * radius * radius))
            w = ws * _shift(mf, dy, dx)
            if guide is not None:
                diff = ((guide - _shift(guide, dy, dx)) ** 2).sum(-1)
                w = w * np.exp(-diff / (2 * sigma_color ** 2))
            num += w * _shift(d, dy, dx)
            den += w
        newly = holes & (den > 1e-3)
        d[newly] = num[newly] / den[newly]
        m = m | newly
    if not m.all():
        d = push_pull_fill(d, m)
    return d


def fill_holes(depth: np.ndarray, mask: np.ndarray, rgb: np.ndarray | None = None,
               method: str = "guided") -> np.ndarray:
    if method == "push_pull":
        return push_pull_fill(depth, mask)
    if method == "guided":
        return guided_fill(depth, mask, rgb)
    raise ValueError(f"Unknown hole filling method: {method}")
