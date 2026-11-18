"""Inference wrapper for DepthNet (CPU or GPU).

``build_network_inputs`` is shared by training and inference so the model
always sees exactly the same kind of input.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import CHECKPOINT_PATH
from models.depth_model import DepthNet, count_parameters
from src.hole_filling import push_pull_fill
from src.interpolation import bicubic, bilinear


def build_network_inputs(lr: np.ndarray, mask: np.ndarray, scale: int, out_shape=None):
    """LR depth -> (init_norm, conf, lo, hi) at output resolution."""
    out_shape = out_shape or (lr.shape[0] * scale, lr.shape[1] * scale)
    filled = push_pull_fill(lr, mask)
    init, _ = bicubic(filled, np.ones(lr.shape, bool), scale, out_shape)
    conf, _ = bilinear(mask.astype(np.float32), np.ones(lr.shape, bool), scale, out_shape)
    vals = lr[mask]
    lo, hi = float(vals.min()), float(vals.max())
    # robust range: never smaller than 30 % of mean depth (avoids blow-up on flat crops)
    hi = max(hi, lo + 0.3 * max(float(vals.mean()), 1e-3))
    init_n = (init - lo) / (hi - lo)
    return init_n.astype(np.float32), np.clip(conf, 0, 1).astype(np.float32), lo, hi


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DepthModelRunner:
    def __init__(self, checkpoint: Path | str = CHECKPOINT_PATH, device: torch.device | None = None):
        self.device = device or get_device()
        self.model = DepthNet().to(self.device).eval()
        self.checkpoint_path = Path(checkpoint)
        self.meta: dict = {}
        self.trained = False
        if self.checkpoint_path.exists():
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model"])
            self.meta = ckpt.get("meta", {})
            self.trained = True

    @property
    def n_params(self) -> int:
        return count_parameters(self.model)

    @torch.no_grad()
    def predict(self, lr: np.ndarray, mask: np.ndarray, scale: int, rgb_hr: np.ndarray | None,
                out_shape=None) -> np.ndarray:
        init_n, conf, lo, hi = build_network_inputs(lr, mask, scale, out_shape)
        H, W = init_n.shape
        rgb = rgb_hr if rgb_hr is not None else np.zeros((H, W, 3), np.float32)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().to(self.device)
        x_rgb = t(rgb).permute(2, 0, 1)[None]
        x_init = t(init_n)[None, None]
        x_conf = t(conf)[None, None]
        ph, pw = (-H) % 8, (-W) % 8  # pad to multiple of 8 (3 down-samplings)
        if ph or pw:
            pad = (0, pw, 0, ph)
            x_rgb, x_init, x_conf = (F.pad(x, pad, mode="replicate") for x in (x_rgb, x_init, x_conf))
        out = self.model(x_rgb, x_init, x_conf)[0, 0, :H, :W].cpu().numpy()
        return np.clip(out * (hi - lo) + lo, lo, hi).astype(np.float32)


@lru_cache(maxsize=1)
def get_runner() -> DepthModelRunner:
    torch.set_num_threads(max(1, torch.get_num_threads()))
    return DepthModelRunner()


def training_history() -> dict | None:
    p = CHECKPOINT_PATH.with_suffix(".history.json")
    if p.exists():
        return json.loads(p.read_text())
    return None
