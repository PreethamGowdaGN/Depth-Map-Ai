"""Boundary-aware refinement.

Problem: upsampling blurs depth discontinuities into ramps of "flying
pixels" whose depth belongs to neither the foreground nor the background.

Fix: inside a thin band around detected discontinuities, replace each
pixel by a *joint weighted median* of its neighbourhood:

* weights favour neighbours with similar RGB colour (same object),
* neighbours that are themselves inside the uncertain band get a low
  weight, so the median is decided by confident pixels on either side,
* a median (not a mean) always returns a depth that really exists on
  one side — it snaps instead of averaging, so no new in-between depths
  are invented.

Outside the band nothing is changed, so RGB texture on flat surfaces
can never be copied into the depth (texture-copy prevention).
"""
from __future__ import annotations

import cv2
import numpy as np

from .edge_detection import depth_discontinuities


def weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Row-wise weighted median of (N, K) arrays."""
    order = np.argsort(values, axis=1)
    v = np.take_along_axis(values, order, 1)
    w = np.take_along_axis(weights, order, 1)
    cw = np.cumsum(w, axis=1)
    half = 0.5 * cw[:, -1:]
    idx = (cw < half).sum(1)
    idx = np.clip(idx, 0, v.shape[1] - 1)
    return v[np.arange(v.shape[0]), idx]


def boundary_band(depth: np.ndarray, mask: np.ndarray | None = None, band: int = 2,
                  rel_threshold: float = 0.02) -> np.ndarray:
    disc = depth_discontinuities(depth, mask, rel_threshold)
    k = 2 * band + 1
    return cv2.dilate(disc.astype(np.uint8), np.ones((k, k), np.uint8)) > 0


def boundary_refine(depth: np.ndarray, rgb: np.ndarray | None = None, mask: np.ndarray | None = None,
                    strength: float = 0.5, radius: int = 3, sigma_color: float = 0.08,
                    iterations: int = 2, band_px: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Sharpen depth discontinuities. Returns (refined_depth, band_mask)."""
    d = depth.astype(np.float32).copy()
    if mask is None:
        mask = d > 0
    if strength <= 0:
        return d, np.zeros(d.shape, bool)
    H, W = d.shape
    band = band_px if band_px is not None else max(1, int(round(1 + 3 * strength)))
    zone_all = np.zeros(d.shape, bool)
    offsets = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    for _ in range(iterations):
        zone = boundary_band(d, mask, band) & mask
        zone_all |= zone
        ys, xs = np.nonzero(zone)
        if ys.size == 0:
            break
        vals = np.empty((ys.size, len(offsets)), np.float32)
        wts = np.empty_like(vals)
        sure = np.empty(vals.shape, bool)  # confident neighbour (outside band, valid)
        c0 = rgb[ys, xs] if rgb is not None else None
        for k, (dy, dx) in enumerate(offsets):
            yy = np.clip(ys + dy, 0, H - 1)
            xx = np.clip(xs + dx, 0, W - 1)
            vals[:, k] = d[yy, xx]
            w = np.exp(-(dy * dy + dx * dx) / (2.0 * radius ** 2)) * np.ones(ys.size, np.float32)
            if c0 is not None:
                diff = ((rgb[yy, xx] - c0) ** 2).sum(-1)
                w = w * np.exp(-diff / (2 * sigma_color ** 2))
            w = w * np.where(zone[yy, xx], 0.15, 1.0) * mask[yy, xx]
            wts[:, k] = w
            sure[:, k] = (~zone[yy, xx]) & mask[yy, xx]
        wts += 1e-8
        new = weighted_median(vals, wts)
        # Only touch *flying pixels*: values lying well between the two surfaces
        # (confident neighbours' min / max). Pixels already on a surface are kept,
        # so sharp, correct edges (e.g. from the CNN) are never eroded.
        none = ~sure.any(1)
        sure[none] = True  # no confident neighbour: use the whole window's extremes
        lo = np.where(sure, vals, np.inf).min(1)
        hi = np.where(sure, vals, -np.inf).max(1)
        cur = d[ys, xs]
        span = hi - lo
        mid = np.minimum(np.abs(cur - lo), np.abs(cur - hi)) / np.maximum(span, 1e-6)
        flying = np.isfinite(span) & (span > 0.03 * np.maximum(cur, 0.1)) & (mid > 0.2)
        d[ys[flying], xs[flying]] = new[flying]  # snap (never blend)
    return d, zone_all
