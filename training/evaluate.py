"""Evaluate methods on a real RGB-D folder dataset (e.g. NYU test split).

    python -m training.evaluate --root data/nyu/test --scales 2 4 8 --limit 100

Writes results/real_eval_<folder>.json. On synthetic data use
evaluation/benchmark.py instead.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import METHODS, RESULTS_DIR  # noqa: E402
from src.hybrid import Settings, reconstruct  # noqa: E402
from src.metrics import evaluate  # noqa: E402
from src.preprocessing import degrade_depth  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--scales", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--units", type=float, default=1000.0)
    args = ap.parse_args(argv)
    root = Path(args.root)
    names = sorted(p.stem for p in (root / "rgb").glob("*.png"))[: args.limit]
    acc = {}
    for n in names:
        rgb = cv2.cvtColor(cv2.imread(str(root / "rgb" / f"{n}.png")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        gt = cv2.imread(str(root / "depth" / f"{n}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / args.units
        for s in args.scales:
            h, w = (gt.shape[0] // s) * s, (gt.shape[1] // s) * s
            g, im = gt[:h, :w], rgb[:h, :w]
            lr, m = degrade_depth(g, s, 0.01, 0.0, 1.0, np.random.default_rng(0), g > 0)
            for meth in METHODS.values():
                r = reconstruct(meth, lr, m, s, im, Settings(), g.shape)
                e = evaluate(r.depth, g, g > 0, r.valid, im)
                e["Latency (s)"] = r.latency_s
                acc.setdefault((meth, s), []).append(e)
        print("done", n, flush=True)
    rows = []
    for (m, s), lst in acc.items():
        row = {k: float(np.nanmean([x[k] for x in lst])) for k in lst[0]}
        row.update(method=m, scale=s)
        rows.append(row)
    RESULTS_DIR.mkdir(exist_ok=True)
    p = RESULTS_DIR / f"real_eval_{root.name}.json"
    p.write_text(json.dumps({"root": str(root), "n": len(names), "rows": rows}, indent=1))
    print("saved", p)


if __name__ == "__main__":
    main()
