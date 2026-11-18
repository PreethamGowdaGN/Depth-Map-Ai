import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthetic import generate_scene  # noqa: E402


@pytest.fixture(scope="session")
def scene():
    return generate_scene(200_900, 96, 128)


def step_depth(h=64, w=64, near=1.0, far=3.0):
    """Left half near, right half far: one vertical discontinuity."""
    d = np.full((h, w), far, np.float32)
    d[:, : w // 2] = near
    return d


@pytest.fixture
def step():
    return step_depth()
