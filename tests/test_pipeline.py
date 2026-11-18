"""Deep model, hybrid pipeline, all methods end-to-end, and download outputs."""
import io

import cv2
import numpy as np
import pytest
import torch

from config import METHODS
from models.depth_model import DepthNet
from models.inference import DepthModelRunner, build_network_inputs
from src.hybrid import Settings, reconstruct
from src.metrics import evaluate
from src.preprocessing import degrade_depth
from src.visualization import colorize, depth_png16_bytes, npy_bytes, png_bytes, side_by_side


def test_depthnet_forward_shapes():
    net = DepthNet().eval()
    x = torch.rand(2, 3, 64, 96)
    d = torch.rand(2, 1, 64, 96)
    c = torch.ones(2, 1, 64, 96)
    with torch.no_grad():
        out, aux = net(x, d, c, return_all=True)
    assert out.shape == (2, 1, 64, 96) and len(aux) == 2


def test_runner_predict_odd_sizes(tmp_path):
    r = DepthModelRunner(checkpoint=tmp_path / "missing.pt")  # untrained, must say so
    assert r.trained is False
    lr = np.random.default_rng(0).random((13, 17)).astype(np.float32) + 1
    out = r.predict(lr, np.ones_like(lr, bool), 4, None)
    assert out.shape == (52, 68) and np.isfinite(out).all()


def test_build_inputs_normalised():
    lr = np.array([[1, 2], [3, 4]], np.float32)
    init, conf, lo, hi = build_network_inputs(lr, lr > 0, 2)
    assert init.shape == (4, 4) and conf.min() >= 0 and conf.max() <= 1
    assert lo == 1


def test_training_step_reduces_loss():
    from training.dataset import SyntheticRGBD, collate
    from training.train import total_loss
    torch.manual_seed(0)
    ds = SyntheticRGBD("val", length=4, size=(64, 64), fixed_degradation={"scale": 4, "noise_std": 0.0, "hole_fraction": 0.0, "sparsity": 1.0}, augment=False)
    b = collate([ds[i] for i in range(4)])
    net = DepthNet()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    l0, _ = total_loss(net, b, "cpu")
    for _ in range(15):
        loss, _ = total_loss(net, b, "cpu")
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < l0.item()


@pytest.mark.parametrize("method", list(METHODS.values()))
@pytest.mark.parametrize("scale", [2, 4, 8])
def test_every_method_every_scale(scene, method, scale):
    lr, m = degrade_depth(scene["depth"], scale, 0.01, 0.1, 1.0, np.random.default_rng(0))
    r = reconstruct(method, lr, m, scale, scene["rgb"], Settings(), scene["depth"].shape)
    assert r.depth.shape == scene["depth"].shape
    assert np.isfinite(r.depth).all()
    assert r.valid.mean() > 0.95
    e = evaluate(r.depth, scene["depth"], None, r.valid, scene["rgb"])
    assert e["RMSE"] < 1.5  # sane: scene depths are 0.8–9 m


def test_hybrid_beats_bicubic(scene):
    lr, m = degrade_depth(scene["depth"], 4, 0.01, 0.1, 1.0, np.random.default_rng(0))
    eb = evaluate(reconstruct("bicubic", lr, m, 4, scene["rgb"]).depth, scene["depth"])
    eh = evaluate(reconstruct("hybrid", lr, m, 4, scene["rgb"]).depth, scene["depth"])
    assert eh["RMSE"] < eb["RMSE"]
    assert eh["Boundary RMSE"] < eb["Boundary RMSE"]


def test_hybrid_exposes_stages(scene):
    lr, m = degrade_depth(scene["depth"], 4, 0.01, 0.1, 1.0, np.random.default_rng(0))
    r = reconstruct("hybrid", lr, m, 4, scene["rgb"])
    assert "Final" in r.stages and "1 · Bicubic init" in r.stages
    assert any("Residual" in k for k in r.stages)
    r2 = reconstruct("hybrid", lr, m, 4, scene["rgb"], Settings(use_jbu_fusion=True, use_refinement=True))
    assert any("fusion weight" in k for k in r2.stages) and any("Boundary" in k for k in r2.stages)


def test_settings_toggle_behaviour(scene):
    lr, m = degrade_depth(scene["depth"], 4, 0.0, 0.3, 1.0, np.random.default_rng(0))
    on = reconstruct("bicubic", lr, m, 4, scene["rgb"], Settings(hole_filling=True))
    off = reconstruct("bicubic", lr, m, 4, scene["rgb"], Settings(hole_filling=False))
    assert on.valid.mean() == 1.0 and off.valid.mean() < 1.0
    no_rgb = reconstruct("hybrid", lr, m, 4, None, Settings())
    assert no_rgb.depth.shape == scene["depth"].shape


def test_no_rgb_all_methods(scene):
    lr, m = degrade_depth(scene["depth"], 2, 0.0, 0.0, 1.0, np.random.default_rng(0))
    for meth in METHODS.values():
        r = reconstruct(meth, lr, m, 2, None, Settings(use_rgb=False))
        assert np.isfinite(r.depth).all()


def test_empty_depth_raises():
    with pytest.raises(ValueError):
        reconstruct("bicubic", np.zeros((8, 8)), np.zeros((8, 8), bool), 2)


# ---------------- outputs / downloads ----------------
def test_png16_roundtrip():
    d = np.array([[1.234, 0.0], [2.5, np.nan]], np.float32)
    raw = cv2.imdecode(np.frombuffer(depth_png16_bytes(d), np.uint8), cv2.IMREAD_UNCHANGED)
    assert raw.dtype == np.uint16 and raw.tolist() == [[1234, 0], [2500, 0]]


def test_npy_roundtrip():
    d = np.random.default_rng(0).random((5, 6)).astype(np.float32)
    assert np.array_equal(np.load(io.BytesIO(npy_bytes(d))), d)


def test_visualisation_bytes(scene):
    c = colorize(scene["depth"])
    assert c.dtype == np.uint8 and c.shape == scene["depth"].shape + (3,)
    sheet = side_by_side([c, c], ["a", "b"])
    img = cv2.imdecode(np.frombuffer(png_bytes(sheet), np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] > c.shape[1] * 2
