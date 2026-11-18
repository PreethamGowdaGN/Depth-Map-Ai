import numpy as np
import pytest

from src.edge_detection import (canny_rgb, depth_discontinuities, depth_gradient, sobel_magnitude,
                                structure_edges)
from src.filters import denoise_depth, guided_filter, guided_upsample, joint_bilateral_upsample
from src.hole_filling import guided_fill, push_pull_fill
from src.interpolation import upsample
from src.refinement import boundary_refine, weighted_median
from tests.conftest import step_depth


# ---------------- interpolation ----------------
@pytest.mark.parametrize("method", ["nearest", "bilinear", "bicubic"])
@pytest.mark.parametrize("scale", [2, 4, 8])
@pytest.mark.parametrize("size", [(8, 8), (15, 22), (30, 40)])
def test_interpolation_shapes_and_constant(method, scale, size):
    lr = np.full(size, 2.0, np.float32)
    out, v = upsample(lr, None, scale, method)
    assert out.shape == (size[0] * scale, size[1] * scale)
    assert v.all()
    assert np.allclose(out, 2.0, atol=1e-5)


def test_interpolation_ignores_missing_pixels():
    lr = np.full((10, 10), 3.0, np.float32)
    m = np.ones_like(lr, bool)
    lr[4:6, 4:6] = 0
    m[4:6, 4:6] = False
    out, v = upsample(lr, m, 4, "bilinear")
    assert np.allclose(out[v], 3.0, atol=1e-4)  # zeros never averaged in


def test_nearest_reproduces_samples():
    lr = np.arange(16, dtype=np.float32).reshape(4, 4) + 1
    out, _ = upsample(lr, None, 4, "nearest")
    assert np.allclose(out[2::4, 2::4], lr)


def test_unknown_method():
    with pytest.raises(ValueError):
        upsample(np.ones((4, 4)), None, 2, "lanczos9")


# ---------------- guided / JBU ----------------
def test_guided_filter_identity_on_constant():
    g = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
    out = guided_filter(g, np.full((32, 32), 5.0, np.float32), 4, 1e-3)
    assert np.allclose(out, 5.0, atol=1e-3)


@pytest.mark.parametrize("scale", [2, 4, 8])
def test_guided_upsample_beats_bicubic(scene, scale):
    gt = scene["depth"]
    lr = gt[scale // 2::scale, scale // 2::scale]
    out = guided_upsample(lr, scene["rgb"], scale, gt.shape, edge_preservation=0.6)
    bic, _ = upsample(lr, None, scale, "bicubic", gt.shape)
    rm = lambda x: np.sqrt(((x - gt) ** 2).mean())
    assert rm(out) < rm(bic)


@pytest.mark.parametrize("scale", [2, 4, 8])
def test_jbu_preserves_step_edge(scale):
    gt = step_depth(64, 64)
    rgb = np.zeros((64, 64, 3), np.float32)
    rgb[:, :32] = 1.0
    lr = gt[scale // 2::scale, scale // 2::scale]
    out, v = joint_bilateral_upsample(lr, None, rgb, scale, (64, 64), sigma_range=0.05)
    assert v.all()
    bil, _ = upsample(lr, None, scale, "bilinear")
    err_j = np.abs(out - gt).mean()
    assert err_j <= np.abs(bil - gt).mean() + 1e-6
    assert err_j < 0.05


def test_jbu_without_guide_runs():
    lr = np.random.default_rng(1).random((10, 12)).astype(np.float32) + 1
    out, v = joint_bilateral_upsample(lr, None, None, 4)
    assert out.shape == (40, 48) and v.all()


def test_denoise_reduces_noise_keeps_edge():
    gt = step_depth(64, 64)
    noisy = gt + np.random.default_rng(0).normal(0, 0.03, gt.shape).astype(np.float32)
    out = denoise_depth(noisy, np.ones_like(gt, bool), 0.7)
    assert np.abs(out - gt).mean() < np.abs(noisy - gt).mean()
    assert abs(out[:, 28].mean() - 1.0) < 0.05 and abs(out[:, 36].mean() - 3.0) < 0.1


# ---------------- edges ----------------
def test_discontinuity_detected_on_step_not_on_ramp():
    step = step_depth()
    assert depth_discontinuities(step)[:, 31:33].all()
    ramp = np.tile(np.linspace(1, 3, 64, dtype=np.float32), (64, 1))
    assert not depth_discontinuities(ramp).any()


def test_sobel_and_gradient():
    s = step_depth()
    assert sobel_magnitude(s)[:, 32].min() > 0
    g = depth_gradient(s)
    assert g[:, 31:33].max() > 0.1 and g[:, 5].max() == 0


def test_structure_edges_reject_texture():
    depth = step_depth(64, 64)
    rgb = np.zeros((64, 64, 3), np.float32)
    rgb[:, :32] = 0.8
    rgb[10:12, 40:60] = 1.0  # texture stripe on the flat far surface
    se = structure_edges(rgb, depth)
    ce = canny_rgb(rgb)
    assert ce[5:15, 40:60].any()       # Canny sees the stripe
    assert not se[5:15, 40:60].any()   # structure edges do not
    assert se[:, 28:36].any()          # but keep the real boundary


# ---------------- holes ----------------
def test_push_pull_fills_everything():
    d = np.full((32, 32), 2.0, np.float32)
    m = np.ones_like(d, bool)
    m[5:20, 5:20] = False
    d[~m] = 0
    out = push_pull_fill(d, m)
    assert np.allclose(out, 2.0, atol=1e-4)


def test_guided_fill_respects_colour_boundary():
    gt = step_depth(32, 32)
    rgb = np.zeros((32, 32, 3), np.float32)
    rgb[:, :16] = 1.0
    m = np.ones_like(gt, bool)
    m[8:24, 12:20] = False  # hole straddling the edge
    out = guided_fill(np.where(m, gt, 0), m, rgb)
    assert np.abs(out - gt)[8:24, 12:20].mean() < 0.2


def test_fill_with_no_holes_is_identity():
    d = np.random.default_rng(0).random((8, 8)).astype(np.float32) + 1
    assert np.array_equal(guided_fill(d, np.ones_like(d, bool)), d)


# ---------------- refinement ----------------
def test_weighted_median():
    v = np.array([[1, 2, 3, 100.0]])
    w = np.array([[1, 1, 1, 0.1]])
    assert weighted_median(v, w)[0] in (2, 3)


def test_boundary_refine_removes_flying_pixels():
    gt = step_depth(32, 32)
    blurred = gt.copy()
    blurred[:, 14:18] = np.linspace(1, 3, 4)  # flying-pixel ramp
    rgb = np.zeros((32, 32, 3), np.float32)
    rgb[:, :32 // 2] = 1.0
    out, band = boundary_refine(blurred, rgb, strength=1.0)
    assert band.any()
    assert np.abs(out - gt).mean() < np.abs(blurred - gt).mean()
    vals = np.unique(np.round(out[:, 12:20], 3))
    assert set(vals) <= {1.0, 3.0}  # snapped to real depths only
