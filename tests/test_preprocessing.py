import io

import cv2
import numpy as np
import pytest

from src.preprocessing import (InputError, check_alignment, degrade_depth, load_depth, load_rgb,
                               sanitize_depth)


def _png(arr):
    ok, buf = cv2.imencode(".png", arr)
    assert ok
    return buf.tobytes()


class Named(io.BytesIO):
    def __init__(self, data, name):
        super().__init__(data)
        self.name = name


def test_load_uint16_png_preserves_values():
    raw = np.array([[1000, 2500], [0, 65535]], np.uint16)
    d = load_depth(Named(_png(raw), "d.png"), units_per_metre=1000)
    assert d.is_metric
    assert d.depth[0, 0] == pytest.approx(1.0)
    assert d.depth[0, 1] == pytest.approx(2.5)
    assert not d.mask[1, 0]  # zero = invalid
    assert d.mask.sum() == 3


def test_load_npy_with_nan_and_inf():
    arr = np.array([[1.0, np.nan], [np.inf, -2.0]], np.float32)
    bio = io.BytesIO()
    np.save(bio, arr)
    d = load_depth(Named(bio.getvalue(), "d.npy"))
    assert d.mask.tolist() == [[True, False], [False, False]]
    assert np.isfinite(d.depth).all()


def test_colour_image_flagged_as_relative():
    rgb = np.random.default_rng(0).integers(0, 255, (16, 16, 3), dtype=np.uint8)
    d = load_depth(Named(_png(rgb), "photo.png"))
    assert not d.is_metric
    assert any("colour" in w for w in d.warnings)


def test_8bit_flagged_relative():
    g = np.full((8, 8), 100, np.uint8)
    d = load_depth(Named(_png(g), "g.png"))
    assert not d.is_metric


def test_empty_upload_rejected():
    with pytest.raises(InputError):
        load_depth(Named(b"", "x.png"))


def test_all_invalid_rejected():
    with pytest.raises(InputError):
        load_depth(Named(_png(np.zeros((4, 4), np.uint16)), "z.png"))


def test_garbage_rejected():
    with pytest.raises(InputError):
        load_rgb(Named(b"not an image", "x.jpg"))


def test_load_rgb_range():
    img = np.full((5, 7, 3), 255, np.uint8)
    rgb = load_rgb(Named(_png(img), "a.png"))
    assert rgb.shape == (5, 7, 3) and rgb.max() == pytest.approx(1.0)


def test_sanitize():
    d, m = sanitize_depth(np.array([np.nan, -1, 0, 2.0]))
    assert m.tolist() == [False, False, False, True]
    assert d.tolist() == [0, 0, 0, 2.0]


@pytest.mark.parametrize("scale", [2, 4, 8])
def test_degrade_shapes(scale):
    gt = np.ones((64, 96), np.float32) * 2
    lr, m = degrade_depth(gt, scale, 0.01, 0.2, 0.5, np.random.default_rng(0))
    assert lr.shape == (64 // scale, 96 // scale)
    assert (lr[~m] == 0).all() and (lr[m] > 0).all()
    assert m.mean() < 1.0


def test_alignment_warning():
    assert check_alignment((480, 640), (120, 160), 4) == []
    assert check_alignment((480, 480), (120, 160), 4)  # aspect mismatch
