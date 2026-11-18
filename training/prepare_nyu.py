"""Convert NYU Depth V2 (labelled subset, nyu_depth_v2_labeled.mat) into the
folder format used by FolderRGBD:

    <out>/{train,val,test}/rgb/00001.png
    <out>/{train,val,test}/depth/00001.png   (uint16, millimetres)

Split: a fixed random (seed 0) 745 / 50 / 654 train / val / test split of the
1449 labelled images — the same sizes as the common 795 / 654 split, with 50
training images held out for validation. Requires: pip install h5py

    python training/prepare_nyu.py --mat data/nyu_depth_v2_labeled.mat --out data/nyu
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True)
    ap.add_argument("--out", default="data/nyu")
    ap.add_argument("--crop", type=int, default=8, help="remove white border (pixels)")
    args = ap.parse_args()
    import h5py

    f = h5py.File(args.mat, "r")
    images = f["images"]   # (N, 3, W, H)
    depths = f["depths"]   # (N, W, H) metres
    n = images.shape[0]
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    splits = {"train": idx[:745], "val": idx[745:795], "test": idx[795:]}
    out = Path(args.out)
    c = args.crop
    for name, ids in splits.items():
        (out / name / "rgb").mkdir(parents=True, exist_ok=True)
        (out / name / "depth").mkdir(parents=True, exist_ok=True)
        for k, i in enumerate(ids):
            rgb = np.transpose(images[i], (2, 1, 0))[c:-c, c:-c]
            d = np.transpose(depths[i], (1, 0))[c:-c, c:-c]
            cv2.imwrite(str(out / name / "rgb" / f"{k:05d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out / name / "depth" / f"{k:05d}.png"), np.clip(d * 1000, 0, 65535).astype(np.uint16))
        print(name, len(ids))


if __name__ == "__main__":
    main()
