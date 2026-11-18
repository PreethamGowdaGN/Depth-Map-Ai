"""Shared helpers for the offline evaluation scripts."""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import RESULTS_DIR  # noqa: E402
from src.hybrid import Settings, reconstruct  # noqa: E402
from src.metrics import evaluate, measure  # noqa: E402
from src.preprocessing import degrade_depth  # noqa: E402
from src.synthetic import SPLITS, generate_scene  # noqa: E402

TEST_SEED0 = SPLITS["test"][0]
DEFAULT_DEG = {"noise_std": 0.01, "hole_fraction": 0.1, "sparsity": 1.0}


def test_scenes(n: int, h: int, w: int, **gen):
    for i in range(n):
        seed = TEST_SEED0 + 500 + i  # 200500+ : disjoint from demo samples (200003-200099)
        yield seed, generate_scene(seed, h, w, **gen)


def run_case(method, scene, scale, deg=None, settings=None, seed=0, extra_holes=None, rgb_override=None):
    deg = deg or DEFAULT_DEG
    lr, m = degrade_depth(scene["depth"], scale, deg["noise_std"], deg["hole_fraction"], deg["sparsity"],
                          np.random.default_rng(seed), scene["mask"])
    if extra_holes is not None:
        off = scale // 2
        m &= ~extra_holes[off::scale, off::scale]
        lr[~m] = 0
    rgb = scene["rgb"] if rgb_override is None else rgb_override
    with measure() as mm:
        r = reconstruct(method, lr, m, scale, rgb, settings or Settings(), scene["depth"].shape)
    e = evaluate(r.depth, scene["depth"], scene["mask"], r.valid, scene["rgb"])
    e["Latency (s)"] = r.latency_s
    e["Peak mem (MB)"] = mm["peak_mem_mb"]
    return e, r


def mean_rows(rows: list[dict]) -> dict:
    keys = rows[0].keys()
    return {k: float(np.nanmean([r[k] for r in rows])) for k in keys}


def save_json(name: str, obj: dict):
    RESULTS_DIR.mkdir(exist_ok=True)
    obj["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    p = RESULTS_DIR / name
    p.write_text(json.dumps(obj, indent=1, default=float))
    print("saved", p)
    return p


def checkpoint_meta():
    from models.inference import get_runner
    r = get_runner()
    return {"trained": r.trained, **r.meta}


def fmt(v, d=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{d}f}"


class Timer:
    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        print(f"  ({time.time() - self.t:.1f}s)")
