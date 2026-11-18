"""Train DepthNet.

Examples
--------
CPU, synthetic data (what produced the shipped checkpoint):
    python -m training.train --iters 3000 --batch 8

GPU / Colab with NYU Depth V2 converted by training/prepare_nyu.py:
    python -m training.train --data folder --root data/nyu/train \
        --val-root data/nyu/val --iters 40000 --batch 16 --crop 256
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CHECKPOINT_PATH  # noqa: E402
from models.depth_model import DepthNet, count_parameters  # noqa: E402
from training.dataset import FolderRGBD, SyntheticRGBD, collate  # noqa: E402


def gradient_loss(pred, gt, m):
    dxp, dxg = pred[..., :, 1:] - pred[..., :, :-1], gt[..., :, 1:] - gt[..., :, :-1]
    dyp, dyg = pred[..., 1:, :] - pred[..., :-1, :], gt[..., 1:, :] - gt[..., :-1, :]
    mx, my = m[..., :, 1:] * m[..., :, :-1], m[..., 1:, :] * m[..., :-1, :]
    return ((dxp - dxg).abs() * mx).sum() / mx.sum().clamp(min=1) + \
           ((dyp - dyg).abs() * my).sum() / my.sum().clamp(min=1)


def boundary_weight(gt, m):
    gx = F.pad((gt[..., :, 1:] - gt[..., :, :-1]).abs(), (0, 1, 0, 0))
    gy = F.pad((gt[..., 1:, :] - gt[..., :-1, :]).abs(), (0, 0, 0, 1))
    edge = ((gx + gy) > 0.03).float()
    edge = F.max_pool2d(edge, 5, stride=1, padding=2)  # 2-px band around edges
    return 1.0 + 3.0 * edge * m


def depth_loss(pred, gt, m):
    w = boundary_weight(gt, m)
    l1 = ((pred - gt).abs() * w * m).sum() / (w * m).sum().clamp(min=1)
    return l1 + 0.5 * gradient_loss(pred, gt, m)


def total_loss(model, batch, device):
    rgb, init, conf = batch["rgb"].to(device), batch["init"].to(device), batch["conf"].to(device)
    gt, m = batch["gt"].to(device), batch["gt_mask"].to(device)
    out, aux = model(rgb, init, conf, return_all=True)
    loss = depth_loss(out, gt, m)
    for a in aux:  # deep supervision on coarser residual outputs
        loss = loss + 0.25 * depth_loss(a, gt, m)
    return loss, out


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    se_m = se_b = n = 0.0
    for b in loader:
        gt, m = b["gt"].to(device), b["gt_mask"].to(device)
        out = model(b["rgb"].to(device), b["init"].to(device), b["conf"].to(device))
        init = b["init"].to(device)
        se_m += (((out - gt) ** 2) * m).sum().item()
        se_b += (((init - gt) ** 2) * m).sum().item()
        n += m.sum().item()
    model.train()
    return float(np.sqrt(se_m / n)), float(np.sqrt(se_b / n))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "folder"], default="synthetic")
    ap.add_argument("--root"); ap.add_argument("--val-root")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default=str(CHECKPOINT_PATH))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=0, help="stop early after N minutes (0 = off)")
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} threads={torch.get_num_threads()}")
    size = (args.crop, args.crop)
    fixed_val = {"scale": 4, "noise_std": 0.01, "hole_fraction": 0.1, "sparsity": 1.0}
    if args.data == "synthetic":
        train_ds = SyntheticRGBD("train", length=args.iters * args.batch, size=size)
        val_ds = SyntheticRGBD("val", length=64, size=size, fixed_degradation=fixed_val, augment=False)
        dataset_desc = "Procedural synthetic RGB-D scenes (src/synthetic.py), seeds 0-99,999 train / 100,000-100,999 val"
    else:
        train_ds = FolderRGBD(args.root, size=size)
        val_ds = FolderRGBD(args.val_root or args.root, size=size, augment=False, fixed_degradation=fixed_val)
        dataset_desc = f"Folder dataset: {args.root}"
    tl = DataLoader(train_ds, batch_size=args.batch, shuffle=args.data == "folder",
                    num_workers=args.workers, collate_fn=collate, drop_last=True)
    vl = DataLoader(val_ds, batch_size=8, collate_fn=collate)

    model = DepthNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.iters, pct_start=0.1)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and out_path.exists():
        model.load_state_dict(torch.load(out_path, map_location=device, weights_only=False)["model"])
    print(f"DepthNet parameters: {count_parameters(model):,}")

    history = {"iter": [], "train_loss": [], "val_rmse_model": [], "val_rmse_bicubic": []}
    best = float("inf")
    t0 = time.time()
    it = 0
    running = []
    while it < args.iters:
        for batch in tl:
            loss, _ = total_loss(model, batch, device)
            if not torch.isfinite(loss) or loss.item() > 50:  # guard against a bad batch
                print(f"skipping batch at iter {it}: loss={loss.item():.3g}", flush=True)
                it += 1
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            running.append(loss.item())
            it += 1
            timed_out = args.max_minutes and (time.time() - t0) / 60 > args.max_minutes
            if it % args.val_every == 0 or it == args.iters or timed_out:
                vm, vb = validate(model, vl, device)
                tr = float(np.mean(running)); running = []
                for k, v in zip(history, (it, tr, vm, vb)):
                    history[k].append(v)
                print(f"iter {it:5d}  loss {tr:.4f}  val RMSE model {vm:.4f}  bicubic {vb:.4f}  "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
                if vm < best:
                    best = vm
                    meta = {
                        "trained": True,
                        "dataset": dataset_desc,
                        "iterations": it,
                        "batch_size": args.batch,
                        "crop": args.crop,
                        "device": str(device),
                        "train_minutes": round((time.time() - t0) / 60, 1),
                        "val_rmse_normalised": vm,
                        "val_rmse_bicubic_normalised": vb,
                        "val_setting": fixed_val,
                        "parameters": count_parameters(model),
                        "saved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    }
                    torch.save({"model": model.state_dict(), "meta": meta}, out_path)
                out_path.with_suffix(".history.json").write_text(json.dumps(history, indent=1))
            if it >= args.iters or timed_out:
                break
        if timed_out:
            print("Stopping early: time budget reached.")
            break
    print(f"done. best val RMSE (normalised) {best:.4f}. checkpoint: {out_path}")


if __name__ == "__main__":
    main()
