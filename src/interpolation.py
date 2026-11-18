"""Interpolation baselines: nearest neighbour, bilinear, bicubic.

All methods are *mask-aware*: missing depth (mask == False) is never
averaged in as "zero depth". We use normalised convolution:

    D_hr = interp(D * M) / interp(M)

so only valid measurements contribute. Pixels with too little valid
support stay invalid (returned mask False).

Coordinate convention: LR pixel i is located at HR pixel i*s + s//2
(point sampling, see ``preprocessing.degrade_depth``).
"""
from __future__ import annotations

import cv2
import numpy as np

_INTERP = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
}


def hr_coords(out_shape, scale: int) -> tuple[np.ndarray, np.ndarray]:
    """LR sampling coordinates for every HR pixel (float32 maps for cv2.remap)."""
    h, w = out_shape
    off = scale // 2
    ys = (np.arange(h, dtype=np.float32) - off) / scale
    xs = (np.arange(w, dtype=np.float32) - off) / scale
    map_x, map_y = np.meshgrid(xs, ys)
    return map_x, map_y


def _remap(img: np.ndarray, map_x, map_y, interp) -> np.ndarray:
    return cv2.remap(img.astype(np.float32), map_x, map_y, interp,
                     borderMode=cv2.BORDER_REPLICATE)


def upsample(lr: np.ndarray, mask: np.ndarray | None, scale: int, method: str = "bicubic",
             out_shape: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Upsample a (possibly incomplete) LR depth map.

    Returns (depth_hr float32, mask_hr bool).
    """
    if method not in _INTERP:
        raise ValueError(f"Unknown interpolation method: {method}")
    if scale < 1:
        raise ValueError("scale must be >= 1")
    lr = np.asarray(lr, np.float32)
    if mask is None:
        mask = lr > 0
    out_shape = out_shape or (lr.shape[0] * scale, lr.shape[1] * scale)
    mx, my = hr_coords(out_shape, scale)
    interp = _INTERP[method]
    m = mask.astype(np.float32)

    if mask.all():
        out = _remap(lr, mx, my, interp)
        valid = np.ones(out_shape, bool)
    elif method == "nearest":
        out = _remap(lr * m, mx, my, interp)
        valid = _remap(m, mx, my, interp) > 0.5
    else:
        num = _remap(lr * m, mx, my, interp)
        den = _remap(m, mx, my, interp)
        valid = den > 0.5
        out = np.where(valid, num / np.maximum(den, 1e-6), 0)

    vals = lr[mask]
    if vals.size:
        out = np.clip(out, vals.min(), vals.max())  # bicubic overshoot guard
    out = out.astype(np.float32)
    out[~valid] = 0
    return out, valid


def nearest(lr, mask, scale, out_shape=None):
    return upsample(lr, mask, scale, "nearest", out_shape)


def bilinear(lr, mask, scale, out_shape=None):
    return upsample(lr, mask, scale, "bilinear", out_shape)


def bicubic(lr, mask, scale, out_shape=None):
    return upsample(lr, mask, scale, "bicubic", out_shape)
