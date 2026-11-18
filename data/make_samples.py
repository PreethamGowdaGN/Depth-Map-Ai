"""Create the demo / test samples in data/samples/.

All scenes use seeds from the TEST range (200,000+), which the model never
saw during training (train seeds 0-99,999, val 100,000-100,999).

Also writes data/samples/upload_example/ — ready-made files in the exact
format a user would upload (RGB PNG, 16-bit low-res depth PNG in mm,
16-bit ground-truth PNG) so the upload flow can be tried immediately.

Run:  python data/make_samples.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import degrade_depth  # noqa: E402
from src.synthetic import generate_scene  # noqa: E402

OUT = ROOT / "data" / "samples"
H, W = 384, 512

SAMPLES = [
    dict(slug="01_indoor_scene", name="Indoor scene (standard)", seed=200_003,
         gen={}, degradation=dict(noise_std=0.01, hole_fraction=0.08, sparsity=1.0),
         description="Typical cluttered scene: slanted planes, curved objects, moderate noise and holes."),
    dict(slug="02_thin_structures", name="Thin structures", seed=200_011,
         gen=dict(thin_prob=0.8, n_objects=(7, 11)), degradation=dict(noise_std=0.005, hole_fraction=0.0, sparsity=1.0),
         description="Many 2–5 px poles and wires — they nearly vanish after 8× downsampling."),
    dict(slug="03_texture_heavy", name="Texture-heavy surfaces", seed=200_021,
         gen=dict(texture_prob=1.0), degradation=dict(noise_std=0.005, hole_fraction=0.0, sparsity=1.0),
         description="Stripes, checkerboards and blobs painted on flat surfaces: the texture-copy trap."),
    dict(slug="04_camouflage", name="Camouflaged objects", seed=200_031,
         gen=dict(camouflage_prob=0.9, texture_prob=0.2), degradation=dict(noise_std=0.005, hole_fraction=0.0, sparsity=1.0),
         description="Objects coloured like the background — RGB guidance cannot see these boundaries."),
    dict(slug="05_heavy_noise", name="Heavy measurement noise", seed=200_041,
         gen={}, degradation=dict(noise_std=0.04, hole_fraction=0.05, sparsity=1.0),
         description="4 % depth-proportional Gaussian noise (≈4 cm at 1 m)."),
    dict(slug="06_large_holes", name="Large missing regions", seed=200_051,
         gen={}, degradation=dict(noise_std=0.005, hole_fraction=0.45, sparsity=1.0),
         description="45 % of measurements missing in large blobs."),
    dict(slug="07_very_sparse", name="Very sparse measurements (5 %)", seed=200_061,
         gen={}, degradation=dict(noise_std=0.005, hole_fraction=0.0, sparsity=0.05),
         description="Only 5 % of low-res pixels measured (LiDAR-like)."),
    dict(slug="08_reflective", name="Reflective surface", seed=200_071,
         gen=dict(reflective=True), degradation=dict(noise_std=0.005, hole_fraction=0.0, sparsity=1.0),
         description="A specular patch where the depth sensor returns nothing."),
    dict(slug="09_misaligned", name="Misaligned RGB / depth", seed=200_081,
         gen={}, degradation=dict(noise_std=0.005, hole_fraction=0.0, sparsity=1.0), rgb_shift_px=6,
         description="RGB shifted 6 px relative to depth (calibration error)."),
]


def save_depth16(path, depth_m):
    cv2.imwrite(str(path), np.clip(np.round(depth_m * 1000), 0, 65535).astype(np.uint16))


def save_rgb(path, rgb):
    cv2.imwrite(str(path), cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for s in SAMPLES:
        d = OUT / s["slug"]
        d.mkdir(exist_ok=True)
        sc = generate_scene(s["seed"], H, W, **s["gen"])
        save_rgb(d / "rgb.png", sc["rgb"])
        save_depth16(d / "depth_gt.png", sc["depth"])
        if sc["holes"].any():
            cv2.imwrite(str(d / "sensor_holes.png"), sc["holes"].astype(np.uint8) * 255)
        meta = {k: s[k] for k in ("name", "seed", "degradation", "description")}
        meta["rgb_shift_px"] = s.get("rgb_shift_px", 0)
        meta["size"] = [H, W]
        meta["units"] = "depth_gt.png: uint16 millimetres"
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        print("wrote", d)

    # Upload example in user-facing file format
    ex = ROOT / "data" / "upload_example"
    ex.mkdir(exist_ok=True)
    sc = generate_scene(200_099, H, W)
    lr, m = degrade_depth(sc["depth"], 4, 0.01, 0.1, 1.0, np.random.default_rng(5))
    save_rgb(ex / "rgb_512x384.png", sc["rgb"])
    save_depth16(ex / "depth_lowres_x4_128x96.png", np.where(m, lr, 0))
    save_depth16(ex / "depth_groundtruth_512x384.png", sc["depth"])
    np.save(ex / "depth_lowres_x4_128x96.npy", np.where(m, lr, np.nan).astype(np.float32))
    (ex / "README.txt").write_text(
        "Upload example (unseen test scene, seed 200099)\n"
        "rgb_512x384.png                - RGB guidance image\n"
        "depth_lowres_x4_128x96.png     - low-res depth, uint16 millimetres, 0 = missing\n"
        "depth_lowres_x4_128x96.npy     - same in float32 metres, NaN = missing\n"
        "depth_groundtruth_512x384.png  - ground truth, uint16 millimetres\n"
        "In the Reconstruction Studio choose 'Upload low-res depth', scale 4x.\n")
    print("wrote", ex)


if __name__ == "__main__":
    main()
