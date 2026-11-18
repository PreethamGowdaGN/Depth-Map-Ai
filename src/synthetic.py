"""Procedural RGB-D scene generator.

Why: the official RGB-D datasets (NYU Depth V2, Middlebury, KITTI) are
large downloads. This generator gives an unlimited, perfectly labelled
supply of training and test scenes that contain exactly the hard cases
of Problem Statement 4:

* depth discontinuities between objects at different distances,
* slanted planes and curved (sphere-like) surfaces — smooth geometry
  that must NOT be cut into steps,
* thin structures (poles, wires),
* RGB texture painted on flat surfaces (stripes, checkerboards, blobs)
  — the trap that causes texture-copy artifacts,
* camouflaged objects whose colour matches the background — where RGB
  guidance gives no help,
* Lambertian shading derived from surface normals, so RGB correlates with
  geometry the way it does in real photos.

Each scene is fully determined by its integer seed, so the train / val /
test splits are disjoint by seed range (see ``SPLITS``).
"""
from __future__ import annotations

import cv2
import numpy as np

SPLITS = {"train": (0, 100_000), "val": (100_000, 101_000), "test": (200_000, 201_000)}


def _texture(rng, h, w):
    kind = rng.integers(0, 5)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if kind == 0:  # stripes
        ang = rng.uniform(0, np.pi)
        period = rng.uniform(6, 20)
        t = 0.5 + 0.5 * np.sign(np.sin((xx * np.cos(ang) + yy * np.sin(ang)) * 2 * np.pi / period))
    elif kind == 1:  # checkerboard
        p = int(rng.integers(6, 20))
        t = (((xx // p) + (yy // p)) % 2).astype(np.float32)
    elif kind == 2:  # blobs
        small = rng.random((max(h // 12, 2), max(w // 12, 2))).astype(np.float32)
        t = (cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC) > 0.5).astype(np.float32)
    elif kind == 3:  # fine noise (wood/fabric)
        t = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 1.2)
        t = (t - t.min()) / max(np.ptp(t), 1e-6)
    else:  # "text"-like horizontal dashes
        t = np.zeros((h, w), np.float32)
        for _ in range(int(rng.integers(5, 15))):
            y = int(rng.integers(0, h))
            x0 = int(rng.integers(0, w))
            cv2.line(t, (x0, y), (min(w - 1, x0 + int(rng.integers(10, 60))), y), 1.0, int(rng.integers(1, 3)))
    return t


def _color(rng, near=None):
    if near is not None:
        return np.clip(near + rng.normal(0, 0.04, 3), 0, 1)
    return rng.uniform(0.05, 0.95, 3)


def generate_scene(seed: int, h: int = 240, w: int = 320, n_objects: tuple[int, int] = (4, 9),
                   texture_prob: float = 0.5, camouflage_prob: float = 0.2,
                   thin_prob: float = 0.35, reflective: bool = False) -> dict:
    """Return dict(rgb float32 (h,w,3) in [0,1], depth float32 metres, mask, holes).

    ``holes`` marks pixels a real sensor would miss (reflective patch);
    empty unless reflective=True.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Background: tilted wall + floor
    wall_d = rng.uniform(4.0, 8.0)
    wall = wall_d + rng.uniform(-1.5, 1.5) * (xx / w - 0.5) + rng.uniform(-0.8, 0.8) * (yy / h - 0.5)
    horizon = rng.uniform(0.45, 0.7) * h
    floor = np.where(yy > horizon, 1.2 * h * 0.9 / np.maximum(yy - horizon, 1e-3) + 0.8, np.inf)
    depth = np.minimum(wall, floor).astype(np.float32)
    bg_col, floor_col = _color(rng), _color(rng)
    rgb = np.where((floor < wall)[..., None], floor_col, bg_col).astype(np.float32)
    if rng.random() < texture_prob:
        t = _texture(rng, h, w)
        rgb = rgb * (0.75 + 0.25 * t[..., None])

    n = int(rng.integers(*n_objects))
    obj_depths = np.sort(rng.uniform(0.8, wall_d * 0.9, n))[::-1]  # far to near
    for d0 in obj_depths:
        kind = rng.random()
        shape_mask = np.zeros((h, w), np.uint8)
        obj_depth = np.full((h, w), d0, np.float32)
        if kind < thin_prob:  # thin pole / wire
            th = int(rng.integers(2, 6))
            x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
            ang = rng.uniform(0, np.pi)
            L = rng.uniform(0.4, 1.2) * max(h, w)
            x1, y1 = int(x0 + L * np.cos(ang)), int(y0 + L * np.sin(ang))
            cv2.line(shape_mask, (x0, y0), (x1, y1), 1, th)
        elif kind < thin_prob + 0.3:  # slanted box face / rectangle
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
            bw, bh = rng.uniform(0.1, 0.4) * w, rng.uniform(0.1, 0.45) * h
            ang = rng.uniform(-30, 30)
            box = cv2.boxPoints(((cx, cy), (bw, bh), ang)).astype(np.int32)
            cv2.fillPoly(shape_mask, [box], 1)
            obj_depth = d0 + rng.uniform(-0.6, 0.6) * (xx - cx) / w + rng.uniform(-0.4, 0.4) * (yy - cy) / h
        elif kind < thin_prob + 0.55:  # sphere / ellipsoid
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
            r = rng.uniform(0.06, 0.2) * min(h, w) * 2
            ex = rng.uniform(0.7, 1.3)
            q = ((xx - cx) / (r * ex)) ** 2 + ((yy - cy) / r) ** 2
            shape_mask = (q < 1).astype(np.uint8)
            obj_depth = d0 - 0.25 * d0 * (r / max(h, w)) * np.sqrt(np.clip(1 - q, 0, 1)) * 2
        else:  # polygon
            k = int(rng.integers(3, 6))
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
            rad = rng.uniform(0.08, 0.25) * min(h, w) * 1.5
            angs = np.sort(rng.uniform(0, 2 * np.pi, k))
            pts = np.stack([cx + rad * np.cos(angs), cy + rad * np.sin(angs)], 1).astype(np.int32)
            cv2.fillPoly(shape_mask, [pts], 1)
            obj_depth = d0 + rng.uniform(-0.3, 0.3) * (xx - cx) / w
        sm = (shape_mask > 0) & (obj_depth < depth)
        if not sm.any():
            continue
        depth[sm] = obj_depth[sm]
        col = _color(rng, near=bg_col if rng.random() < camouflage_prob else None)
        layer = np.broadcast_to(col, (h, w, 3)).astype(np.float32)
        if rng.random() < texture_prob:
            t = _texture(rng, h, w)
            col2 = _color(rng)
            layer = layer * (1 - 0.5 * t[..., None]) + col2 * 0.5 * t[..., None]
        rgb[sm] = layer[sm]

    holes = np.zeros((h, w), bool)
    if reflective:
        cx, cy = rng.uniform(0.3, 0.7) * w, rng.uniform(0.3, 0.7) * h
        q = ((xx - cx) / (0.15 * w)) ** 2 + ((yy - cy) / (0.12 * h)) ** 2
        holes = q < 1
        rgb[holes] = np.clip(rgb[holes] * 0.3 + 0.75, 0, 1)  # specular highlight

    # Lambertian shading from surface normals
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    f = 300.0 / np.maximum(depth, 0.3)
    nx, ny, nz = -gx * f, -gy * f, np.ones_like(depth)
    norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    L = np.array([0.3, -0.4, 0.86], np.float32)
    shade = np.clip((nx * L[0] + ny * L[1] + nz * L[2]) / norm, 0, 1)
    atten = 1.0 / (1 + 0.04 * depth)
    rgb = rgb * (0.45 + 0.55 * shade[..., None]) * (0.7 + 0.3 * atten[..., None])
    rgb = cv2.GaussianBlur(rgb, (3, 3), 0.6)
    rgb = np.clip(rgb + rng.normal(0, 0.01, rgb.shape), 0, 1).astype(np.float32)
    mask = np.ones((h, w), bool)
    return {"rgb": rgb, "depth": depth.astype(np.float32), "mask": mask, "holes": holes}
