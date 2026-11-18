"""Datasets for training / evaluating DepthNet.

* ``SyntheticRGBD``  — procedural scenes (src/synthetic.py); no download.
* ``FolderRGBD``     — any real dataset converted to
                        <root>/rgb/*.png  +  <root>/depth/*.png (16-bit, mm)
                        (see training/prepare_nyu.py for NYU Depth V2).

Each item returns the network inputs exactly as ``models.inference``
builds them at test time, plus the normalised ground truth.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from models.inference import build_network_inputs
from src.preprocessing import degrade_depth
from src.synthetic import SPLITS, generate_scene
from training.augmentations import (color_jitter, depth_scale_jitter, random_crop,
                                    random_flip, rgb_dropout)


def sample_degradation(rng: np.random.Generator) -> dict:
    """Random sensor model: scale, noise, holes, sparsity."""
    return {
        "scale": int(rng.choice([2, 4, 8])),
        "noise_std": float(rng.choice([0.0, rng.uniform(0.002, 0.02)])),
        "hole_fraction": float(rng.choice([0.0, rng.uniform(0.02, 0.3)])),
        "sparsity": float(1.0 if rng.random() < 0.75 else rng.uniform(0.15, 0.7)),
    }


def make_item(rgb, gt, gt_mask, deg, rng, use_rgb=True):
    s = deg["scale"]
    lr, lr_mask = degrade_depth(gt, s, deg["noise_std"], deg["hole_fraction"], deg["sparsity"], rng, gt_mask)
    if lr_mask.sum() < 4:  # degenerate: keep at least a few samples
        lr_mask[::2, ::2] = gt_mask[s // 2::s, s // 2::s][::2, ::2]
        lr = np.where(lr_mask, gt[s // 2::s, s // 2::s], 0).astype(np.float32)
    init_n, conf, lo, hi = build_network_inputs(lr, lr_mask, s, gt.shape)
    gt_n = (gt - lo) / (hi - lo)
    if not use_rgb:
        rgb = np.zeros_like(rgb)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    return {
        "rgb": t(rgb).permute(2, 0, 1),
        "init": t(init_n)[None],
        "conf": t(conf)[None],
        "gt": t(gt_n)[None],
        "gt_mask": t(gt_mask.astype(np.float32))[None],
        "scale": s,
    }


class SyntheticRGBD(Dataset):
    def __init__(self, split: str = "train", length: int = 10_000, size=(128, 128),
                 fixed_degradation: dict | None = None, augment: bool = True):
        self.lo, self.hi = SPLITS[split]
        self.length = length
        self.size = size
        self.fixed = fixed_degradation
        self.augment = augment and split == "train"
        self.split = split

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        seed = self.lo + (i % (self.hi - self.lo))
        rng = np.random.default_rng(seed * 7 + 1 if self.split != "train" else None)
        h, w = self.size
        sc = generate_scene(seed if self.split != "train" else int(rng.integers(self.lo, self.hi)),
                            h=h + 16, w=w + 16)
        rgb, gt, m = random_crop(sc["rgb"], sc["depth"], sc["mask"], self.size, rng)
        use_rgb = True
        if self.augment:
            rgb, gt, m = random_flip(rgb, gt, m, rng)
            rgb = color_jitter(rgb, rng)
            gt = depth_scale_jitter(gt, rng)
            rgb, use_rgb = rgb_dropout(rgb, rng)
        deg = self.fixed or sample_degradation(rng)
        return make_item(rgb, gt, m, deg, rng, use_rgb)


class FolderRGBD(Dataset):
    """Real data in <root>/rgb/NAME.png and <root>/depth/NAME.png (uint16 mm)."""

    def __init__(self, root, size=(256, 256), units_per_metre=1000.0, augment=True,
                 fixed_degradation: dict | None = None):
        self.root = Path(root)
        self.names = sorted(p.stem for p in (self.root / "rgb").glob("*.png"))
        if not self.names:
            raise FileNotFoundError(f"No PNGs in {self.root/'rgb'}")
        self.size, self.upm, self.augment, self.fixed = size, units_per_metre, augment, fixed_degradation

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        rng = np.random.default_rng(None if self.augment else i)
        n = self.names[i]
        rgb = cv2.cvtColor(cv2.imread(str(self.root / "rgb" / f"{n}.png")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        d = cv2.imread(str(self.root / "depth" / f"{n}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / self.upm
        m = d > 0
        rgb, d, m = random_crop(rgb, d, m, self.size, rng)
        use_rgb = True
        if self.augment:
            rgb, d, m = random_flip(rgb, d, m, rng)
            rgb = color_jitter(rgb, rng)
            rgb, use_rgb = rgb_dropout(rgb, rng)
        deg = self.fixed or sample_degradation(rng)
        return make_item(rgb, d, m, deg, rng, use_rgb)


def collate(batch):
    keys = ["rgb", "init", "conf", "gt", "gt_mask"]
    out = {k: torch.stack([b[k] for b in batch]) for k in keys}
    out["scale"] = torch.tensor([b["scale"] for b in batch])
    return out
