import math

import numpy as np
import pytest

from src.metrics import evaluate, mae, measure, psnr, rmse, ssim
from tests.conftest import step_depth


def test_perfect_prediction():
    gt = step_depth()
    e = evaluate(gt.copy(), gt)
    assert e["RMSE"] == 0 and e["MAE"] == 0
    assert math.isinf(e["PSNR"])
    assert e["SSIM"] == pytest.approx(1.0, abs=1e-6)
    assert e["Edge F1"] == pytest.approx(1.0)
    assert e["Coverage"] == 1.0


def test_known_values():
    gt = np.full((10, 10), 2.0, np.float32)
    gt[0, 0] = 4.0
    pred = gt + 0.1
    m = np.ones_like(gt, bool)
    assert rmse(pred, gt, m) == pytest.approx(0.1, rel=1e-5)
    assert mae(pred, gt, m) == pytest.approx(0.1, rel=1e-5)
    assert psnr(pred, gt, m) == pytest.approx(10 * math.log10(4 / 0.01), rel=1e-4)


def test_invalid_pixels_ignored():
    gt = np.full((20, 20), 2.0, np.float32)
    gt[:5] = 0          # invalid GT
    pred = np.full_like(gt, 2.0)
    pred[:5] = 99       # garbage where GT invalid -> must not count
    e = evaluate(pred, gt)
    assert e["RMSE"] == 0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate(np.ones((4, 4)), np.ones((5, 5)))


def test_no_valid_pixels_returns_nan():
    e = evaluate(np.ones((8, 8)), np.zeros((8, 8)))
    assert math.isnan(e["RMSE"])


def test_ssim_small_image_nan():
    assert math.isnan(ssim(np.ones((4, 4)), np.ones((4, 4)), np.ones((4, 4), bool)))


def test_blurred_edges_lower_f1():
    gt = step_depth(64, 64)
    blur = gt.copy()
    blur[:, 24:40] = np.linspace(1, 3, 16)
    assert evaluate(blur, gt)["Edge F1"] < evaluate(gt, gt)["Edge F1"]
    assert evaluate(blur, gt)["Boundary RMSE"] > 0


def test_measure_records():
    with measure() as m:
        np.zeros((256, 256)).sum()
    assert m["latency_s"] >= 0 and m["peak_mem_mb"] > 0
