"""Evaluation metrics. All accuracy metrics use only pixels where the ground
truth is valid AND the prediction is valid (``eval_mask``). Coverage (how
many GT pixels the method actually filled) is reported separately so a
method cannot look good by leaving hard pixels empty.

Metrics
-------
RMSE, MAE        : metres (or relative units for non-metric input)
PSNR             : dB, peak = GT depth range in the valid region
SSIM             : structural similarity on depth normalised by GT range,
                   averaged over valid pixels
Edge F1          : F-score between predicted and GT depth discontinuities,
                   with a 2-pixel matching tolerance (edge preservation)
Boundary RMSE    : RMSE inside a 3-pixel band around GT discontinuities
                   (boundary accuracy — where blurring hurts most)
Texture-copy idx : mean relative depth gradient of the prediction on pixels
                   that are flat in GT but have strong RGB edges (lower = better)
δ<1.25           : fraction of pixels with max(p/g, g/p) < 1.25 (standard
                   depth-estimation accuracy)
"""
from __future__ import annotations

import math
import time
import tracemalloc
from contextlib import contextmanager

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from .edge_detection import canny_rgb, depth_discontinuities, depth_gradient


def _eval_mask(pred, gt, gt_mask=None, pred_mask=None):
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: prediction {pred.shape} vs ground truth {gt.shape}")
    m = np.isfinite(gt) & (gt > 0) if gt_mask is None else gt_mask.copy()
    m &= np.isfinite(pred)
    if pred_mask is not None:
        m &= pred_mask
    return m


def rmse(pred, gt, mask) -> float:
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((pred[mask] - gt[mask]) ** 2)))


def mae(pred, gt, mask) -> float:
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(pred[mask] - gt[mask])))


def psnr(pred, gt, mask) -> float:
    if mask.sum() == 0:
        return float("nan")
    peak = float(gt[mask].max() - gt[mask].min())
    mse = float(np.mean((pred[mask] - gt[mask]) ** 2))
    if peak <= 0:
        return float("nan")
    if mse <= 1e-20:
        return float("inf")
    return 10.0 * math.log10(peak ** 2 / mse)


def ssim(pred, gt, mask) -> float:
    if mask.sum() < 49 or min(gt.shape) < 7:
        return float("nan")
    lo, hi = float(gt[mask].min()), float(gt[mask].max())
    rng = max(hi - lo, 1e-6)
    g = np.where(mask, (gt - lo) / rng, 0).astype(np.float64)
    p = np.where(mask, (pred - lo) / rng, 0).astype(np.float64)
    _, smap = structural_similarity(g, p, data_range=1.0, full=True)
    return float(smap[mask].mean())


def delta_accuracy(pred, gt, mask, thr: float = 1.25) -> float:
    m = mask & (pred > 0)
    if m.sum() == 0:
        return float("nan")
    r = np.maximum(pred[m] / gt[m], gt[m] / pred[m])
    return float((r < thr).mean())


def edge_f1(pred, gt, mask, tol: int = 2, rel_threshold: float = 0.04) -> dict:
    ge = depth_discontinuities(gt, mask, rel_threshold)
    pe = depth_discontinuities(pred, mask, rel_threshold)
    if ge.sum() == 0:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    ge_d = cv2.dilate(ge.astype(np.uint8), k) > 0
    pe_d = cv2.dilate(pe.astype(np.uint8), k) > 0
    precision = float((pe & ge_d).sum() / max(pe.sum(), 1)) if pe.sum() else 0.0
    recall = float((ge & pe_d).sum() / ge.sum())
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def boundary_rmse(pred, gt, mask, band: int = 3, rel_threshold: float = 0.04) -> float:
    ge = depth_discontinuities(gt, mask, rel_threshold)
    if ge.sum() == 0:
        return float("nan")
    k = np.ones((2 * band + 1, 2 * band + 1), np.uint8)
    zone = (cv2.dilate(ge.astype(np.uint8), k) > 0) & mask
    return rmse(pred, gt, zone)


def texture_copy_index(pred, gt, rgb, mask) -> float:
    if rgb is None:
        return float("nan")
    flat = depth_gradient(gt, mask) < 0.01
    flat = cv2.erode(flat.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    tex = canny_rgb(rgb) & flat & mask
    if tex.sum() < 20:
        return float("nan")
    return float(depth_gradient(pred, mask)[tex].mean())


def evaluate(pred: np.ndarray, gt: np.ndarray, gt_mask: np.ndarray | None = None,
             pred_mask: np.ndarray | None = None, rgb: np.ndarray | None = None) -> dict:
    """Compute every accuracy metric. Returns a flat dict (NaN = unavailable)."""
    if gt_mask is None:
        gt_mask = np.isfinite(gt) & (gt > 0)
    m = _eval_mask(pred, gt, gt_mask, pred_mask)
    ef = edge_f1(pred, gt, m)
    return {
        "RMSE": rmse(pred, gt, m),
        "MAE": mae(pred, gt, m),
        "PSNR": psnr(pred, gt, m),
        "SSIM": ssim(pred, gt, m),
        "δ<1.25": delta_accuracy(pred, gt, m),
        "Edge F1": ef["f1"],
        "Edge precision": ef["precision"],
        "Edge recall": ef["recall"],
        "Boundary RMSE": boundary_rmse(pred, gt, m),
        "Texture-copy idx": texture_copy_index(pred, gt, rgb, m),
        "Coverage": float(m.sum() / max(gt_mask.sum(), 1)),
    }


# Direction of "better" for charts / tables
HIGHER_IS_BETTER = {"PSNR", "SSIM", "δ<1.25", "Edge F1", "Edge precision", "Edge recall", "Coverage"}


@contextmanager
def measure():
    """Measure wall-clock latency and peak Python/NumPy memory.

    Usage::
        with measure() as m:
            run()
        m["latency_s"], m["peak_mem_mb"]
    Note: tracemalloc sees NumPy buffers but not PyTorch's C++ allocator;
    for the deep model the RSS delta (psutil) is also recorded.
    """
    stats: dict = {}
    try:
        import psutil
        proc = psutil.Process()
        rss0 = proc.memory_info().rss
    except Exception:  # pragma: no cover
        proc, rss0 = None, 0
    started = not tracemalloc.is_tracing()
    if started:
        tracemalloc.start()
    tracemalloc.reset_peak()
    t0 = time.perf_counter()
    try:
        yield stats
    finally:
        stats["latency_s"] = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        stats["peak_mem_mb"] = peak / 2 ** 20
        if started:
            tracemalloc.stop()
        if proc is not None:
            stats["rss_delta_mb"] = max(proc.memory_info().rss - rss0, 0) / 2 ** 20
