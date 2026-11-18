"""Hybrid reconstruction engine + a single entry point for every method.

Hybrid pipeline (each stage is returned so the UI can show it):

  1. Initial interpolation : mask-aware bicubic (push-pull filled)        D_bic
  2. Deep prediction       : DepthNet on the RAW low-res depth + validity
                             mask + RGB (learned denoising, completion and
                             RGB-gated edge placement)                     D_cnn
                             (falls back to joint bilateral upsampling if no
                             trained checkpoint exists)
  3. Edge analysis         : depth discontinuities of D_cnn
  4. Edge-aware data-fidelity correction:
                             r = D_LR − D_cnn sampled at the measured pixels,
                             upsampled with a joint bilateral filter whose range
                             kernel is the predicted DEPTH:
                                 D = D_cnn + JBU_depth(r)
                             -> the output agrees with the real measurements,
                             low-frequency CNN bias is removed, corrections never
                             cross depth edges and RGB texture cannot be copied.
  5. (optional) RGB-JBU fusion at structure edges
  6. (optional) weighted-median flying-pixel snapping

Why this fusion (measured on VALIDATION scenes, seeds 100 000+, before any test run):
  * pre-denoising the CNN input doubled its error  -> CNN gets raw input;
  * RGB-JBU fusion and median snapping each made the trained CNN worse
    (they move correct CNN edges onto texture / camouflage edges)
    -> kept as switchable stages, OFF by default, reported in the ablation;
  * residual correction improved RMSE, boundary RMSE and Edge F1 at every
    scale and noise level tested -> ON by default.
The fusion is therefore a learned estimate constrained by measurements through
a classical edge-aware filter — not an average of unrelated outputs.

Settings mapping for the hybrid: noise reduction -> spatial σ of the residual
filter (more averaging of noisy measurements); edge preservation ->
range σ of the residual filter; RGB guidance -> DepthNet RGB input; hole filling
-> whether missing regions are output.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .edge_detection import canny_rgb, depth_discontinuities, structure_edges
from .filters import denoise_depth, guided_upsample, joint_bilateral_upsample
from .hole_filling import guided_fill, push_pull_fill
from .interpolation import upsample
from .refinement import boundary_refine


@dataclass
class Settings:
    noise_reduction: float = 0.0      # 0 = off, 1 = strong (classical pre-denoise; hurts edges at low noise)
    edge_preservation: float = 0.6    # 0 = soft, 1 = sharpest
    hole_filling: bool = True
    use_rgb: bool = True
    # hybrid-only ablation switches
    use_deep: bool = True
    use_residual: bool = True         # edge-aware data-fidelity correction
    use_edges: bool = True            # structure-edge gating for the optional JBU fusion
    use_jbu_fusion: bool = False      # optional; measured harmful with trained DepthNet
    use_refinement: bool = False      # optional; measured harmful with trained DepthNet

    def to_dict(self):
        return asdict(self)


@dataclass
class Result:
    depth: np.ndarray
    valid: np.ndarray
    method: str
    latency_s: float = 0.0
    stages: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def _lr_rgb(rgb_hr, lr_shape):
    if rgb_hr is None:
        return None
    return cv2.resize(rgb_hr, (lr_shape[1], lr_shape[0]), interpolation=cv2.INTER_AREA)


def preprocess_lr(lr, mask, rgb_hr, s: Settings):
    """Denoise + fill at low resolution. Returns (lr_proc, mask_proc, stage_dict).

    stage_dict["_denoised"] holds the denoised-but-unfilled LR (input to DepthNet,
    which does its own completion from the raw validity mask, as in training).
    """
    stages = {}
    d = lr.astype(np.float32)
    if s.noise_reduction > 0:
        d = denoise_depth(d, mask, s.noise_reduction)
        stages["LR denoised"] = d
    stages["_denoised"] = d
    if s.hole_filling and not mask.all():
        d = guided_fill(d, mask, _lr_rgb(rgb_hr, lr.shape) if s.use_rgb else None)
        m = np.ones(mask.shape, bool)
        stages["LR hole-filled"] = d
    else:
        m = mask.copy()
    return d, m, stages


def _jbu(lr_p, m_p, rgb_hr, scale, out_shape, s: Settings):
    sigma_r = 0.25 - 0.2 * s.edge_preservation   # 0.25 (soft) … 0.05 (sharp)
    return joint_bilateral_upsample(lr_p, m_p, rgb_hr if s.use_rgb else None, scale, out_shape,
                                    sigma_spatial=1.0, sigma_range=sigma_r, radius=2)


def _deep(lr_p, m_p, rgb_hr, scale, out_shape, s: Settings):
    from models.inference import get_runner
    runner = get_runner()
    out = runner.predict(lr_p, m_p, scale, rgb_hr if s.use_rgb else None, out_shape)
    return out, runner.trained


def reconstruct(method: str, lr: np.ndarray, mask: np.ndarray | None, scale: int,
                rgb_hr: np.ndarray | None = None, settings: Settings | None = None,
                out_shape=None) -> Result:
    """Run one reconstruction method end-to-end."""
    s = settings or Settings()
    if mask is None:
        mask = np.isfinite(lr) & (lr > 0)
    if not mask.any():
        raise ValueError("Depth input has no valid pixels.")
    out_shape = out_shape or (lr.shape[0] * scale, lr.shape[1] * scale)
    if rgb_hr is not None and rgb_hr.shape[:2] != tuple(out_shape):
        rgb_hr = cv2.resize(rgb_hr, (out_shape[1], out_shape[0]), interpolation=cv2.INTER_AREA)
    t0 = time.perf_counter()
    notes = []
    lr_p, m_p, stages = preprocess_lr(lr, mask, rgb_hr, s)
    # where are the outputs allowed to be valid if hole filling is OFF?
    _, support = upsample(lr, mask, scale, "bilinear", out_shape)

    if method in ("nearest", "bilinear", "bicubic"):
        d, v = upsample(lr_p, m_p, scale, method, out_shape)
    elif method == "guided":
        filled = lr_p if m_p.all() else push_pull_fill(lr_p, m_p)
        d = guided_upsample(filled, rgb_hr if s.use_rgb else None, scale, out_shape, s.edge_preservation)
        v = np.ones(out_shape, bool)
    elif method == "jbu":
        d, v = _jbu(lr_p, m_p, rgb_hr, scale, out_shape, s)
    elif method == "deep":
        # DepthNet denoises and completes by itself (trained on raw noisy input);
        # classical pre-denoising measurably hurts it, so it always gets raw LR.
        d, trained = _deep(lr, mask, rgb_hr, scale, out_shape, s)
        v = np.ones(out_shape, bool)
        if not trained:
            notes.append("DepthNet has NO trained checkpoint: output comes from random weights.")
    elif method == "hybrid":
        d, v, st, nt = _hybrid(lr, mask, lr_p, m_p, rgb_hr, scale, out_shape, s)
        st.pop("_raw", None)
        stages.update(st)
        notes += nt
    else:
        raise ValueError(f"Unknown method '{method}'")

    if not s.hole_filling:
        v = v & support
    d = np.where(v, d, 0).astype(np.float32)
    stages.pop("_denoised", None)
    return Result(d, v, method, time.perf_counter() - t0, stages, notes)


def _hybrid(lr, mask, lr_p, m_p, rgb_hr, scale, out_shape, s: Settings):
    stages, notes = {}, []
    rgb = rgb_hr if s.use_rgb else None
    filled = lr_p if m_p.all() else push_pull_fill(lr_p, m_p)
    d_bic, _ = upsample(filled, None, scale, "bicubic", out_shape)
    stages["1 · Bicubic init"] = d_bic

    # 2 · Deep prediction (raw LR + validity mask, exactly the training input)
    base = None
    if s.use_deep:
        d_cnn, trained = _deep(lr, mask, rgb_hr, scale, out_shape, s)
        if trained:
            base = d_cnn
            stages["2 · DepthNet"] = d_cnn
        else:
            notes.append("No trained DepthNet checkpoint — hybrid uses joint bilateral upsampling as its base.")
    if base is None:  # deep disabled or untrained -> classical edge-aware base
        base, jv = _jbu(lr_p, m_p, rgb_hr, scale, out_shape, s)
        base = np.where(jv, base, d_bic)
        stages["2 · Joint bilateral base"] = base

    # 3 · Edge analysis (shown to the user; drives the optional stages)
    disc = depth_discontinuities(base)
    stages["3 · Depth discontinuities"] = disc.astype(np.float32)

    # 4 · Data-fidelity correction with an edge-aware filter.
    #     r = measured LR depth − prediction at the same pixels; r is upsampled by a
    #     joint bilateral filter whose RANGE kernel uses the predicted DEPTH (not RGB),
    #     so corrections never leak across depth edges and RGB texture cannot enter.
    out = base
    if s.use_residual:
        off = scale // 2
        samp = base[off::scale, off::scale][: lr.shape[0], : lr.shape[1]]
        valid = mask & (samp > 0)
        resid = np.where(valid, lr - samp, 0).astype(np.float32)  # RAW measurements
        g = (base - base.min()) / max(float(np.ptp(base)), 1e-6)
        r_hr, rv = joint_bilateral_upsample(
            resid, valid, g, scale, out_shape,
            sigma_spatial=1.0 + 2.0 * s.noise_reduction,            # more averaging of noisy residuals
            sigma_range=max(0.005, 0.04 - 0.03 * s.edge_preservation),  # sharper = narrower depth kernel
            radius=3)
        corr = np.where(rv, r_hr, 0).astype(np.float32)
        stages["4 · Residual correction (m)"] = corr
        out = base + corr

    # 5 · Optional: RGB joint-bilateral fusion at structure edges (off by default:
    #     it HURT accuracy on validation once DepthNet was trained — see ablation).
    if s.use_jbu_fusion and rgb is not None:
        d_jbu, jv = _jbu(lr_p, m_p, rgb_hr, scale, out_shape, s)
        d_jbu = np.where(jv, d_jbu, out)
        edges = structure_edges(rgb, out, None, band=max(2, scale // 2 + 1)) if s.use_edges else canny_rgb(rgb)
        dist = cv2.distanceTransform((~edges).astype(np.uint8), cv2.DIST_L2, 3)
        w = np.exp(-(dist ** 2) / (2 * (1.0 + 1.5 * s.edge_preservation) ** 2)).astype(np.float32)
        stages["5 · Structure edges" if s.use_edges else "5 · RGB edges (ungated)"] = edges.astype(np.float32)
        stages["5 · JBU fusion weight"] = w
        out = (1 - w) * out + w * d_jbu

    # 6 · Optional: flying-pixel snapping (off by default for the same reason).
    if s.use_refinement and s.edge_preservation > 0:
        out, band = boundary_refine(out, rgb, None, strength=s.edge_preservation)
        stages["6 · Boundary band"] = band.astype(np.float32)

    lo, hi = float(lr[mask].min()), float(lr[mask].max())
    out = np.clip(out, lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    stages["Final"] = out
    return out.astype(np.float32), np.ones(out_shape, bool), stages, notes


def run_all(lr, mask, scale, rgb_hr=None, settings=None, methods=None, out_shape=None) -> dict:
    from config import METHODS
    methods = methods or list(METHODS.values())
    return {m: reconstruct(m, lr, mask, scale, rgb_hr, settings, out_shape) for m in methods}
