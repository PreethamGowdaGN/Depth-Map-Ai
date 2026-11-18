"""Edge-aware filters: guided filter, joint bilateral upsampling, denoising.

Guided filter (He, Sun & Tang, 2010/2013)
    Inside every small window the output is a *linear function of the
    guide image*: q = a·I + b. Where the guide has an edge, q can have an
    edge too; where the guide is flat, q is smoothed. Runs in O(N) with
    box filters.

Joint bilateral upsampling (Kopf et al., 2007)
    Each high-res pixel averages nearby *low-res* depth samples, weighted
    by (1) spatial distance and (2) how similar the guide colour is.
    Samples from the other side of an object boundary (different colour)
    get almost no weight, so edges stay sharp.

Edge-aware denoising
    Mask-aware median (kills speckle/outliers) followed by a bilateral
    filter on depth itself — range weights on *depth* differences, so
    depth discontinuities are not smoothed and no RGB texture is involved.
"""
from __future__ import annotations

import cv2
import numpy as np

from .hole_filling import push_pull_fill
from .interpolation import hr_coords


# --------------------------------------------------------------------------- #
# Guided filter
# --------------------------------------------------------------------------- #
def _box(x: np.ndarray, r: int) -> np.ndarray:
    return cv2.boxFilter(x, cv2.CV_32F, (2 * r + 1, 2 * r + 1), normalize=True,
                         borderType=cv2.BORDER_REFLECT)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 8,
                  eps: float = 1e-3) -> np.ndarray:
    """Guided filter with grey (H,W) or colour (H,W,3) guide."""
    p = src.astype(np.float32)
    I = guide.astype(np.float32)
    if I.ndim == 2:
        mI, mp = _box(I, radius), _box(p, radius)
        cov = _box(I * p, radius) - mI * mp
        var = _box(I * I, radius) - mI * mI
        a = cov / (var + eps)
        b = mp - a * mI
        return _box(a, radius) * I + _box(b, radius)

    mean_I = np.stack([_box(I[..., c], radius) for c in range(3)], -1)
    mean_p = _box(p, radius)
    cov_Ip = np.stack([_box(I[..., c] * p, radius) for c in range(3)], -1) - mean_I * mean_p[..., None]
    # 3x3 covariance of the guide in each window
    sigma = np.empty(I.shape[:2] + (3, 3), np.float32)
    for i in range(3):
        for j in range(i, 3):
            v = _box(I[..., i] * I[..., j], radius) - mean_I[..., i] * mean_I[..., j]
            sigma[..., i, j] = v
            sigma[..., j, i] = v
    sigma += eps * np.eye(3, dtype=np.float32)
    a = np.linalg.solve(sigma, cov_Ip[..., None])[..., 0]          # (H,W,3)
    b = mean_p - (a * mean_I).sum(-1)
    mean_a = np.stack([_box(a[..., c], radius) for c in range(3)], -1)
    mean_b = _box(b, radius)
    return (mean_a * I).sum(-1) + mean_b


def guided_upsample(lr_filled: np.ndarray, rgb_hr: np.ndarray | None, scale: int,
                    out_shape=None, edge_preservation: float = 0.5,
                    init: np.ndarray | None = None) -> np.ndarray:
    """Depth upsampling = bicubic initialisation + guided filtering.

    ``edge_preservation`` in [0,1] → smaller eps → sharper edges.
    Without RGB, the bicubic depth guides itself (self-guided filtering).
    """
    from .interpolation import bicubic

    out_shape = out_shape or (lr_filled.shape[0] * scale, lr_filled.shape[1] * scale)
    if init is None:
        init, _ = bicubic(lr_filled, np.ones(lr_filled.shape, bool), scale, out_shape)
    radius = max(2, scale + 1)
    if rgb_hr is not None:
        eps = 10 ** (-1.5 - 2.5 * edge_preservation)      # 3e-2 … 1e-4 (RGB in [0,1])
        guide = rgb_hr
    else:
        lo, hi = float(init.min()), float(init.max())
        guide = (init - lo) / max(hi - lo, 1e-6)
        eps = 10 ** (-2.0 - 2.5 * edge_preservation)
    out = guided_filter(guide, init, radius, eps)
    lo, hi = float(lr_filled.min()), float(lr_filled.max())
    return np.clip(out, lo, hi).astype(np.float32)


# --------------------------------------------------------------------------- #
# Joint bilateral upsampling
# --------------------------------------------------------------------------- #
def joint_bilateral_upsample(lr: np.ndarray, mask: np.ndarray | None, guide_hr: np.ndarray | None,
                             scale: int, out_shape=None, sigma_spatial: float = 1.0,
                             sigma_range: float = 0.1, radius: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Joint bilateral upsampling using only *valid* LR measurements.

    guide_hr: HR RGB in [0,1] (H,W,3) or a grey guide (H,W). If None, the
    bilinear depth itself (normalised) is the guide (plain bilateral
    upsampling).
    Returns (depth_hr, valid_hr).
    """
    lr = lr.astype(np.float32)
    if mask is None:
        mask = lr > 0
    h_lr, w_lr = lr.shape
    out_shape = out_shape or (h_lr * scale, w_lr * scale)
    H, W = out_shape
    off = scale // 2

    if guide_hr is None:
        from .interpolation import bilinear
        init, _ = bilinear(push_pull_fill(lr, mask), None, scale, out_shape)
        lo, hi = float(init.min()), float(init.max())
        guide_hr = (init - lo) / max(hi - lo, 1e-6)
    g = guide_hr.astype(np.float32)
    if g.ndim == 2:
        g = g[..., None]

    mx, my = hr_coords(out_shape, scale)
    bx, by = np.floor(mx).astype(np.int32), np.floor(my).astype(np.int32)
    num = np.zeros(out_shape, np.float32)
    den = np.zeros(out_shape, np.float32)
    mf = mask.astype(np.float32)
    inv2ss = 1.0 / (2 * sigma_spatial ** 2)
    inv2sr = 1.0 / (2 * sigma_range ** 2)
    for oy in range(-radius + 1, radius + 1):
        qy = np.clip(by + oy, 0, h_lr - 1)
        gy = np.clip(qy * scale + off, 0, H - 1)
        for ox in range(-radius + 1, radius + 1):
            qx = np.clip(bx + ox, 0, w_lr - 1)
            gx = np.clip(qx * scale + off, 0, W - 1)
            ds = (mx - qx) ** 2 + (my - qy) ** 2
            dg = ((g - g[gy, gx]) ** 2).sum(-1)
            wgt = np.exp(-ds * inv2ss - dg * inv2sr) * mf[qy, qx]
            num += wgt * lr[qy, qx]
            den += wgt
    valid = den > 1e-8
    out = np.where(valid, num / np.maximum(den, 1e-12), 0).astype(np.float32)
    return out, valid


# --------------------------------------------------------------------------- #
# Edge-aware denoising
# --------------------------------------------------------------------------- #
def denoise_depth(depth: np.ndarray, mask: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Remove sensor noise without blurring depth discontinuities.

    strength in [0,1]. 0 = off.
    """
    if strength <= 0 or not mask.any():
        return depth.astype(np.float32).copy()
    filled = push_pull_fill(depth, mask).astype(np.float32)
    k = 3 if strength < 0.7 else 5
    med = cv2.medianBlur(filled, k)
    ref = float(np.median(depth[mask]))
    sigma_r = max(ref * (0.01 + 0.06 * strength), 1e-4)
    bil = cv2.bilateralFilter(med, d=5, sigmaColor=sigma_r, sigmaSpace=1.5 + strength)
    out = np.where(mask, bil, 0).astype(np.float32)
    return out
