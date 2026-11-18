"""Failure-case analysis: stress-test conditions, bicubic vs JBU vs hybrid.

    python -m evaluation.failure_analysis --n 6 --scale 4
Saves metrics + a visual sheet per case in results/failure_cases/.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from config import METHOD_LABELS, RESULTS_DIR
from evaluation.common import fmt, mean_rows, run_case, save_json, test_scenes
from src.visualization import colorize, error_map, png_bytes, side_by_side, to_uint8_rgb

CASES = {
    "Large missing regions": dict(deg=dict(noise_std=0.005, hole_fraction=0.45, sparsity=1.0)),
    "Strong depth boundaries": dict(deg=dict(noise_std=0.005, hole_fraction=0.0, sparsity=1.0), gen=dict(n_objects=(10, 14))),
    "Heavy measurement noise": dict(deg=dict(noise_std=0.04, hole_fraction=0.0, sparsity=1.0)),
    "Thin structures": dict(deg=dict(noise_std=0.005, hole_fraction=0.0, sparsity=1.0), gen=dict(thin_prob=0.85, n_objects=(7, 11))),
    "Reflective surfaces": dict(deg=dict(noise_std=0.005, hole_fraction=0.0, sparsity=1.0), gen=dict(reflective=True)),
    "Texture-heavy surfaces": dict(deg=dict(noise_std=0.005, hole_fraction=0.0, sparsity=1.0), gen=dict(texture_prob=1.0)),
    "Misaligned RGB/depth (6 px)": dict(deg=dict(noise_std=0.005, hole_fraction=0.0, sparsity=1.0), shift=6),
    "Very sparse (5 %)": dict(deg=dict(noise_std=0.005, hole_fraction=0.0, sparsity=0.05)),
}
METHODS = ["bicubic", "jbu", "deep", "hybrid"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--scale", type=int, default=4)
    args = ap.parse_args(argv)
    out_dir = RESULTS_DIR / "failure_cases"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case, cfg in CASES.items():
        scenes = list(test_scenes(args.n, 240, 320, **cfg.get("gen", {})))
        per = {m: [] for m in METHODS}
        vis = None
        for k, (seed, sc) in enumerate(scenes):
            rgb = np.roll(sc["rgb"], cfg["shift"], axis=1) if cfg.get("shift") else None
            holes = sc["holes"] if sc["holes"].any() else None
            outs = {}
            for m in METHODS:
                e, r = run_case(m, sc, args.scale, cfg["deg"], seed=seed, extra_holes=holes, rgb_override=rgb)
                per[m].append(e)
                outs[m] = r
            if k == 0:
                lo, hi = np.percentile(sc["depth"], 1), np.percentile(sc["depth"], 99)
                tiles = [to_uint8_rgb(sc["rgb"] if rgb is None else rgb)]
                tiles += [colorize(outs[m].depth, outs[m].valid, lo, hi) for m in METHODS]
                tiles += [colorize(sc["depth"], None, lo, hi), error_map(outs["hybrid"].depth, sc["depth"], sc["mask"])]
                vis = side_by_side(tiles, ["RGB"] + [METHOD_LABELS[m].split(" (")[0] for m in METHODS] + ["Ground truth", "Hybrid error"])
        slug = case.lower().replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "").replace("%", "pct")
        (out_dir / f"{slug}.png").write_bytes(png_bytes(vis))
        for m in METHODS:
            r = mean_rows(per[m])
            r.update(case=case, method=METHOD_LABELS[m].split(" (")[0])
            rows.append(r)
            print(f"{case:32s} {m:8s} RMSE {fmt(r['RMSE'])} BRMSE {fmt(r['Boundary RMSE'])} EdgeF1 {fmt(r['Edge F1'],3)}", flush=True)
    save_json("failure_analysis.json", {"scale": args.scale, "n_scenes_per_case": args.n, "rows": rows,
                                        "cases": {k: {kk: vv for kk, vv in v.items()} for k, v in CASES.items()}})
    cols = ["RMSE", "Boundary RMSE", "Edge F1", "Texture-copy idx", "SSIM"]
    md = ["| Case | Method | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for r in rows:
        md.append(f"| {r['case']} | {r['method']} | " + " | ".join(fmt(r[c]) for c in cols) + " |")
    (RESULTS_DIR / "failure_analysis.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
