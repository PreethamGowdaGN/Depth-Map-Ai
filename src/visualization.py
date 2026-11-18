"""Colourising depth, error maps and file export helpers."""
from __future__ import annotations

import io

import cv2
import numpy as np


def colorize(depth: np.ndarray, mask: np.ndarray | None = None, lo: float | None = None,
             hi: float | None = None, cmap: int = cv2.COLORMAP_TURBO, invert: bool = True) -> np.ndarray:
    """Depth -> RGB uint8. Near = warm, far = cool (invert=True). Invalid = black."""
    if mask is None:
        mask = np.isfinite(depth) & (depth > 0)
    vals = depth[mask]
    if vals.size == 0:
        return np.zeros(depth.shape + (3,), np.uint8)
    lo = float(np.percentile(vals, 1)) if lo is None else lo
    hi = float(np.percentile(vals, 99)) if hi is None else hi
    n = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    if invert:
        n = 1 - n
    c = cv2.applyColorMap((n * 255).astype(np.uint8), cmap)
    c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
    c[~mask] = 0
    return c


def error_map(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, vmax: float | None = None) -> np.ndarray:
    err = np.abs(pred - gt)
    err[~mask] = 0
    if vmax is None:
        vmax = float(np.percentile(err[mask], 99)) if mask.any() else 1.0
    n = np.clip(err / max(vmax, 1e-6), 0, 1)
    c = cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
    c[~mask] = 40
    return c


def overlay_edges(base_rgb: np.ndarray, edges: np.ndarray, color=(0, 240, 255)) -> np.ndarray:
    img = base_rgb.copy()
    if img.dtype != np.uint8:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    img[edges] = color
    return img


def to_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    return np.clip(rgb * 255, 0, 255).astype(np.uint8) if rgb.dtype != np.uint8 else rgb


def gray_to_rgb(x: np.ndarray, normalize: bool = True) -> np.ndarray:
    x = x.astype(np.float32)
    if normalize:
        x = x / max(float(np.percentile(x, 99.5)), 1e-6)
    g = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    return np.stack([g] * 3, -1)


def png_bytes(img: np.ndarray) -> bytes:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def depth_png16_bytes(depth_m: np.ndarray, units_per_metre: float = 1000.0) -> bytes:
    """Export metric depth as 16-bit PNG (default: millimetres; 0 = invalid)."""
    d = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
    raw = np.clip(np.round(d * units_per_metre), 0, 65535).astype(np.uint16)
    ok, buf = cv2.imencode(".png", raw)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def npy_bytes(arr: np.ndarray) -> bytes:
    bio = io.BytesIO()
    np.save(bio, arr.astype(np.float32))
    return bio.getvalue()


def side_by_side(images: list[np.ndarray], labels: list[str] | None = None, pad: int = 8) -> np.ndarray:
    """Concatenate equally-sized RGB uint8 images horizontally with labels."""
    h = max(i.shape[0] for i in images)
    tiles = []
    for k, im in enumerate(images):
        if im.shape[0] != h:
            im = cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h), interpolation=cv2.INTER_NEAREST)
        t = cv2.copyMakeBorder(im, 30 if labels else 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(11, 14, 30))
        if labels:
            cv2.putText(t, labels[k], (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    return np.concatenate(tiles, 1)
