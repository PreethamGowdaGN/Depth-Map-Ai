"""Comparative evaluation of all 7 methods at 2x, 4x, 8x on held-out scenes.

    python -m evaluation.benchmark --n 20

Writes results/benchmark.json and results/benchmark.md.
"""
from __future__ import annotations

import argparse

from config import METHOD_LABELS, METHODS, RESULTS_DIR
from evaluation.common import (DEFAULT_DEG, checkpoint_meta, fmt, mean_rows, run_case, save_json,
                               test_scenes)
from src.hybrid import Settings

COLS = ["RMSE", "MAE", "PSNR", "SSIM", "δ<1.25", "Edge F1", "Boundary RMSE", "Texture-copy idx", "Latency (s)", "Peak mem (MB)"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--h", type=int, default=240)
    ap.add_argument("--w", type=int, default=320)
    args = ap.parse_args(argv)
    scales = [2, 4, 8]
    settings = Settings()
    scenes = list(test_scenes(args.n, args.h, args.w))
    acc = {(m, s): [] for m in METHODS.values() for s in scales}
    for i, (seed, sc) in enumerate(scenes):
        for s in scales:
            for m in METHODS.values():
                e, _ = run_case(m, sc, s, DEFAULT_DEG, settings, seed=seed)
                acc[(m, s)].append(e)
        print(f"scene {i+1}/{len(scenes)} done", flush=True)
    rows = []
    for (m, s), lst in acc.items():
        r = mean_rows(lst)
        r.update(method=m, scale=s)
        rows.append(r)

    def avg(method, key):
        return sum(r[key] for r in rows if r["method"] == method) / len(scales)

    summary = {
        "hybrid_vs_bicubic_rmse_reduction_pct": 100 * (1 - avg("hybrid", "RMSE") / avg("bicubic", "RMSE")),
        "hybrid_vs_bicubic_boundary_rmse_reduction_pct": 100 * (1 - avg("hybrid", "Boundary RMSE") / avg("bicubic", "Boundary RMSE")),
        "best_rmse_method_per_scale": {s: min((r for r in rows if r["scale"] == s), key=lambda r: r["RMSE"])["method"] for s in scales},
    }
    out = {"n_scenes": len(scenes), "seeds": [s for s, _ in scenes], "size": [args.h, args.w], "scales": scales,
           "degradation": DEFAULT_DEG, "settings": settings.to_dict(), "checkpoint": checkpoint_meta(),
           "rows": rows, "summary": summary}
    save_json("benchmark.json", out)

    md = [f"# Benchmark — {len(scenes)} held-out synthetic scenes ({args.w}×{args.h})\n",
          f"Degradation: point-sampled downsampling, noise σ={DEFAULT_DEG['noise_std']} (×depth), "
          f"{DEFAULT_DEG['hole_fraction']*100:.0f}% blob holes. Settings: {settings.to_dict()}\n"]
    for s in scales:
        md.append(f"\n## {s}×\n\n| Method | " + " | ".join(COLS) + " |\n|---|" + "---|" * len(COLS))
        for r in [r for r in rows if r["scale"] == s]:
            md.append(f"| {METHOD_LABELS[r['method']]} | " + " | ".join(fmt(r[c], 2 if c in ('PSNR', 'Peak mem (MB)') else 4) for c in COLS) + " |")
    md.append(f"\n**Summary:** hybrid vs bicubic RMSE reduction {summary['hybrid_vs_bicubic_rmse_reduction_pct']:.1f}%, "
              f"boundary RMSE reduction {summary['hybrid_vs_bicubic_boundary_rmse_reduction_pct']:.1f}% (mean of 2×/4×/8×).")
    (RESULTS_DIR / "benchmark.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
