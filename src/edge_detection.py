"""Edge detection on RGB and depth.

* ``sobel_magnitude`` — gradient strength.
* ``canny_rgb`` — thin RGB edges (object outlines *and* texture).
* ``depth_gradient`` — relative depth gradient |∇D| / D (scale-invariant,
  so a 10 cm jump at 1 m counts like a 1 m jump at 10 m).
* ``depth_discontinuities`` — pixels where depth jumps between objects.
* ``structure_edges`` — RGB edges that agree with a depth change. Only these
  are allowed to steer depth, which is how we avoid copying texture.
"""
from __future__ import annotations

import cv2
import numpy as np


def to_gray(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        return rgb.astype(np.float32)
    return cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2GRAY)


def sobel_magnitude(img: np.ndarray) -> np.ndarray:
    g = to_gray(img) if img.ndim == 3 else img.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def canny_rgb(rgb: np.ndarray, low: int = 50, high: int = 120) -> np.ndarray:
    g8 = np.clip(to_gray(rgb) * 255, 0, 255).astype(np.uint8)
    g8 = cv2.GaussianBlur(g8, (3, 3), 0)
    return cv2.Canny(g8, low, high) > 0


def canny_depth(depth: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is None:
        mask = depth > 0
    vals = depth[mask]
    if vals.size == 0:
        return np.zeros(depth.shape, bool)
    lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
    d8 = (np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    e = cv2.Canny(d8, 30, 80) > 0
    return e & mask


def depth_gradient(depth: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Relative depth gradient magnitude (unitless). Invalid neighbourhoods -> 0."""
    if mask is None:
        mask = depth > 0
    d = depth.astype(np.float32)
    mag = sobel_magnitude(d) / 8.0  # Sobel kernel sums to 8 over a unit step
    rel = mag / np.maximum(d, 1e-3)
    valid = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    rel[~valid] = 0
    return rel


def depth_discontinuities(depth: np.ndarray, mask: np.ndarray | None = None,
                          rel_threshold: float = 0.04) -> np.ndarray:
    """Binary map of depth discontinuities (occlusion boundaries).

    A pixel is a discontinuity if the max |ΔD| to a 4-neighbour exceeds
    ``rel_threshold`` × local depth (e.g. 4 %). Slanted planes have small
    per-pixel differences and are therefore not flagged.
    """
    if mask is None:
        mask = depth > 0
    d = depth.astype(np.float32)
    jump = np.zeros_like(d)
    both = np.zeros(d.shape, bool)
    for axis in (0, 1):
        diff = np.abs(np.diff(d, axis=axis))
        mv = np.logical_and(np.take(mask, range(0, d.shape[axis] - 1), axis=axis),
                            np.take(mask, range(1, d.shape[axis]), axis=axis))
        diff = np.where(mv, diff, 0)
        pad = [(0, 0), (0, 0)]
        pad[axis] = (0, 1)
        a = np.pad(diff, pad)
        pad[axis] = (1, 0)
        b = np.pad(diff, pad)
        jump = np.maximum(jump, np.maximum(a, b))
    rel = jump / np.maximum(d, 1e-3)
    return (rel > rel_threshold) & mask


def structure_edges(rgb: np.ndarray, depth_init: np.ndarray, mask: np.ndarray | None = None,
                    band: int = 3, rel_threshold: float = 0.03) -> np.ndarray:
    """RGB edges that coincide with a (blurry) depth discontinuity.

    The upsampled depth has blurred discontinuities; we widen them by
    ``band`` pixels and keep only RGB Canny edges inside that zone.
    Texture edges on flat surfaces are discarded.
    """
    disc = depth_discontinuities(depth_init, mask, rel_threshold * 0.5)
    k = 2 * band + 1
    zone = cv2.dilate(disc.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    return canny_rgb(rgb) & zone


def edge_report(rgb: np.ndarray | None, depth: np.ndarray, mask: np.ndarray) -> dict:
    out = {
        "Depth Sobel": sobel_magnitude(depth),
        "Depth Canny": canny_depth(depth, mask),
        "Depth gradient (relative)": depth_gradient(depth, mask),
        "Depth discontinuities": depth_discontinuities(depth, mask),
    }
    if rgb is not None:
        out["RGB Sobel"] = sobel_magnitude(rgb)
        out["RGB Canny"] = canny_rgb(rgb)
        out["Structure edges (RGB ∩ depth)"] = structure_edges(rgb, depth, mask)
    return out
