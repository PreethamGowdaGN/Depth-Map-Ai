"""Data augmentation for RGB-D training pairs.

Geometric augmentations are applied identically to RGB and depth so
they stay aligned. Photometric augmentations touch RGB only (depth
values must never be altered by colour changes).
"""
from __future__ import annotations

import numpy as np


def random_crop(rgb, depth, mask, size, rng):
    h, w = depth.shape
    ch, cw = size
    if h < ch or w < cw:
        raise ValueError(f"Crop {size} larger than image {(h, w)}")
    y = int(rng.integers(0, h - ch + 1))
    x = int(rng.integers(0, w - cw + 1))
    return rgb[y:y + ch, x:x + cw], depth[y:y + ch, x:x + cw], mask[y:y + ch, x:x + cw]


def random_flip(rgb, depth, mask, rng):
    if rng.random() < 0.5:
        rgb, depth, mask = rgb[:, ::-1], depth[:, ::-1], mask[:, ::-1]
    return np.ascontiguousarray(rgb), np.ascontiguousarray(depth), np.ascontiguousarray(mask)


def color_jitter(rgb, rng, strength=0.15):
    b = rng.uniform(-strength, strength)
    c = rng.uniform(1 - strength, 1 + strength)
    s = rng.uniform(1 - strength, 1 + strength)
    grey = rgb.mean(-1, keepdims=True)
    out = (rgb - 0.5) * c + 0.5 + b
    out = grey + (out - grey) * s
    return np.clip(out, 0, 1).astype(np.float32)


def depth_scale_jitter(depth, rng, lo=0.7, hi=1.4):
    """Multiply all depths by one factor (a scene twice as far looks the same)."""
    return (depth * rng.uniform(lo, hi)).astype(np.float32)


def rgb_dropout(rgb, rng, p=0.1):
    """Teach the model to work without RGB guidance (RGB toggle in the app)."""
    if rng.random() < p:
        return np.zeros_like(rgb), False
    return rgb, True
