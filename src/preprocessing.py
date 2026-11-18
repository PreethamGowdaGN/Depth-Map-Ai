"""Input loading, validation and degradation.

Key idea: a depth map is *numbers* (distance per pixel), not a picture.
We therefore load depth files without any colour conversion, keep a
boolean *validity mask* (True = real measurement) and convert to metres
using an explicit unit scale.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import cv2
import numpy as np

try:  # tifffile reads float32 TIFF depth reliably; OpenCV is the fallback
    import tifffile
except Exception:  # pragma: no cover
    tifffile = None


class InputError(ValueError):
    """Raised for user-facing validation problems."""


@dataclass
class DepthData:
    depth: np.ndarray            # float32 metres (or relative units); invalid = 0
    mask: np.ndarray             # bool, True where measurement is valid
    source_dtype: str
    is_metric: bool              # False when depth came from a colourised / 8-bit image
    warnings: list[str] = field(default_factory=list)

    @property
    def shape(self):
        return self.depth.shape


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _read_bytes(file_or_bytes) -> tuple[bytes, str]:
    name = getattr(file_or_bytes, "name", "") or ""
    if hasattr(file_or_bytes, "getvalue"):
        data = file_or_bytes.getvalue()
    elif hasattr(file_or_bytes, "read"):
        data = file_or_bytes.read()
    elif isinstance(file_or_bytes, (bytes, bytearray)):
        data = bytes(file_or_bytes)
    else:  # path
        with open(file_or_bytes, "rb") as f:
            data = f.read()
        name = str(file_or_bytes)
    if not data:
        raise InputError("The uploaded file is empty.")
    return data, name.lower()


def load_rgb(file_or_bytes) -> np.ndarray:
    """Load an RGB guidance image as float32 in [0, 1], shape (H, W, 3)."""
    data, _ = _read_bytes(file_or_bytes)
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise InputError("Could not decode the RGB image. Use PNG or JPG.")
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb


def _decode_depth_array(data: bytes, name: str) -> np.ndarray:
    if name.endswith(".npy"):
        arr = np.load(io.BytesIO(data), allow_pickle=False)
    elif name.endswith(".npz"):
        z = np.load(io.BytesIO(data), allow_pickle=False)
        arr = z[z.files[0]]
    elif name.endswith((".tif", ".tiff")) and tifffile is not None:
        arr = tifffile.imread(io.BytesIO(data))
    else:
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise InputError("Could not decode the depth file. Use 16-bit PNG, TIFF or .npy.")
    return np.asarray(arr)


def load_depth(file_or_bytes, units_per_metre: float | None = None) -> DepthData:
    """Load a depth map and return metres + validity mask.

    * uint16 PNG/TIFF -> divided by ``units_per_metre`` (default 1000 = mm).
    * float TIFF / .npy -> assumed already in metres unless units given.
    * 8-bit or 3-channel images are accepted but flagged as *relative* depth,
      because they are usually visualisations, not measurements.
    """
    data, name = _read_bytes(file_or_bytes)
    arr = _decode_depth_array(data, name)
    warnings: list[str] = []
    is_metric = True
    src_dtype = str(arr.dtype)

    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = arr[..., :3]
        chans = arr.astype(np.float32)
        spread = float(np.abs(chans - chans.mean(axis=2, keepdims=True)).mean())
        colourful = spread > (2.0 if arr.dtype == np.uint8 else 0.01)
        if colourful:
            warnings.append(
                "This depth file has colour channels — it looks like a colourised "
                "visualisation or a normal photo, not raw depth. It was converted to "
                "grey levels and treated as RELATIVE depth (no metric units)."
            )
        else:
            warnings.append("3-channel grey depth image: treated as RELATIVE depth (no units).")
        if arr.dtype == np.uint8:
            chans = chans / 255.0
        arr = chans.mean(axis=2)
        is_metric = False
    if arr.ndim != 2:
        raise InputError(f"Depth must be a 2-D array, got shape {arr.shape}.")

    if np.issubdtype(arr.dtype, np.integer):
        if arr.dtype == np.uint8:
            is_metric = False
            warnings.append(
                "8-bit depth has only 256 levels and no units; treated as RELATIVE depth."
            )
            depth = arr.astype(np.float32) / 255.0
        else:
            upm = units_per_metre or 1000.0
            depth = arr.astype(np.float32) / float(upm)
    else:
        depth = arr.astype(np.float32)
        if units_per_metre:
            depth = depth / float(units_per_metre)
        if not is_metric:
            m = np.nanmax(np.where(np.isfinite(depth), depth, np.nan)) if np.isfinite(depth).any() else 1
            depth = depth / max(float(m), 1e-6)

    depth, mask = sanitize_depth(depth)
    if mask.sum() == 0:
        raise InputError("The depth map contains no valid (non-zero, finite) values.")
    frac = 1 - mask.mean()
    if frac > 0.5:
        warnings.append(f"{frac*100:.0f}% of depth pixels are missing — treated as sparse input.")
    return DepthData(depth, mask, src_dtype, is_metric, warnings)


def sanitize_depth(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replace NaN/inf/negative values by 0 and build the validity mask."""
    depth = np.asarray(depth, dtype=np.float32).copy()
    mask = np.isfinite(depth) & (depth > 0)
    depth[~mask] = 0.0
    return depth, mask


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def check_alignment(rgb_shape, depth_shape, scale: int, tol: float = 0.02) -> list[str]:
    """Check that RGB (high-res) and depth (low-res) describe the same view."""
    msgs = []
    rh, rw = rgb_shape[:2]
    dh, dw = depth_shape[:2]
    if abs(rh / rw - dh / dw) / (rh / rw) > tol:
        msgs.append(
            f"Aspect ratio mismatch: RGB {rw}×{rh} vs depth {dw}×{dh}. "
            "The images may not be aligned; the RGB will be resized to fit."
        )
    exp = (dh * scale, dw * scale)
    if (rh, rw) != exp and not msgs:
        msgs.append(
            f"RGB is {rw}×{rh} but {scale}× upsampling of {dw}×{dh} depth gives "
            f"{exp[1]}×{exp[0]}; the RGB will be resized to the output size."
        )
    return msgs


def edge_alignment_score(rgb: np.ndarray, depth: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of strong depth edges that lie near an RGB edge (0..1).

    A low score (< ~0.3) is a hint that RGB and depth are misaligned.
    """
    from .edge_detection import canny_rgb, depth_discontinuities

    d_edges = depth_discontinuities(depth, mask)
    if d_edges.sum() < 20:
        return float("nan")
    r_edges = canny_rgb(rgb)
    near = cv2.dilate(r_edges.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return float((d_edges & near).sum() / d_edges.sum())


def fit_rgb_to(rgb: np.ndarray, shape) -> np.ndarray:
    h, w = shape[:2]
    if rgb.shape[:2] == (h, w):
        return rgb
    return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA if rgb.shape[0] > h else cv2.INTER_CUBIC)


# --------------------------------------------------------------------------- #
# Degradation (used to make LR inputs from ground truth for evaluation)
# --------------------------------------------------------------------------- #
def degrade_depth(gt: np.ndarray, scale: int, noise_std: float = 0.0,
                  hole_fraction: float = 0.0, sparsity: float = 1.0,
                  rng: np.random.Generator | None = None,
                  gt_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a low-resolution, noisy, incomplete depth sensor.

    * Downsample by *point sampling* (like a real low-res sensor; no blur
      across depth edges that would invent in-between depths).
    * Add depth-proportional Gaussian noise (``noise_std`` metres at 1 m).
    * Punch random blob-shaped holes (``hole_fraction`` of pixels).
    * Keep only a random ``sparsity`` fraction of the remaining pixels.
    Returns (lr_depth, lr_mask).
    """
    rng = rng or np.random.default_rng(0)
    if gt_mask is None:
        gt_mask = gt > 0
    off = scale // 2
    lr = gt[off::scale, off::scale].astype(np.float32).copy()
    m = gt_mask[off::scale, off::scale].copy()
    h, w = lr.shape
    if noise_std > 0:
        lr = lr + rng.normal(0, noise_std, lr.shape).astype(np.float32) * np.maximum(lr, 0.1)
    if hole_fraction > 0:
        noise = rng.random((max(h // 6, 2), max(w // 6, 2))).astype(np.float32)
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
        thr = np.quantile(noise, hole_fraction)
        m &= noise > thr
    if sparsity < 1.0:
        m &= rng.random((h, w)) < sparsity
    m &= lr > 0
    lr[~m] = 0
    return lr.astype(np.float32), m


def normalize_for_display(depth: np.ndarray, mask: np.ndarray | None = None,
                          lo: float | None = None, hi: float | None = None) -> np.ndarray:
    if mask is None:
        mask = depth > 0
    vals = depth[mask]
    if vals.size == 0:
        return np.zeros_like(depth, dtype=np.float32)
    lo = float(np.percentile(vals, 1)) if lo is None else lo
    hi = float(np.percentile(vals, 99)) if hi is None else hi
    out = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    out[~mask] = 0
    return out.astype(np.float32)
