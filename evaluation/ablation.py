"""Ablation study: remove one hybrid component at a time.

    python -m evaluation.ablation --n 12 --scale 4
"""
from __future__ import annotations

import argparse
from dataclasses import replace

from config import RESULTS_DIR
from evaluation.common import DEFAULT_DEG, fmt, mean_rows, run_case, save_json, test_scenes
from src.hybrid import Settings

FULL = Settings()
VARIANTS = {
    "Full hybrid (default)": FULL,
    "− Residual correction (= DepthNet only)": replace(FULL, use_residual=False),
    "− Deep learning (JBU base + residual)": replace(FULL, use_deep=False),
    "− RGB guidance": replace(FULL, use_rgb=False),
    "− Hole filling (holes left empty)": replace(FULL, hole_filling=False),
    "+ Noise reduction 0.5": replace(FULL, noise_reduction=0.5),
    "+ RGB-JBU fusion, structure-edge gated": replace(FULL, use_jbu_fusion=True, use_edges=True),
    "+ RGB-JBU fusion, ungated (every RGB edge)": replace(FULL, use_jbu_fusion=True, use_edges=False),
    "+ Flying-pixel snapping": replace(FULL, use_refinement=True),
}
COLS = ["RMSE", "MAE", "PSNR", "SSIM", "Edge F1", "Boundary RMSE", "Texture-copy idx", "Coverage", "Latency (s)"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--scale", type=int, default=4)
    args = ap.parse_args(argv)
    scenes = list(test_scenes(args.n, 240, 320))
    rows = []
    for name, st in VARIANTS.items():
        res = [run_case("hybrid", sc, args.scale, DEFAULT_DEG, st, seed=seed)[0] for seed, sc in scenes]
        r = mean_rows(res)
        r["variant"] = name
        rows.append(r)
        print(name, fmt(r["RMSE"]), fmt(r["Boundary RMSE"]), flush=True)
    save_json("ablation.json", {"scale": args.scale, "n_scenes": len(scenes), "degradation": DEFAULT_DEG, "rows": rows})
    md = [f"| Variant | " + " | ".join(COLS) + " |", "|---|" + "---|" * len(COLS)]
    for r in rows:
        md.append(f"| {r['variant']} | " + " | ".join(fmt(r[c], 2 if c == 'PSNR' else 4) for c in COLS) + " |")
    (RESULTS_DIR / "ablation.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
