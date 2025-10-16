"""Tests for core functions."""

import numpy as np
import pytest

from imglabel.core import compute_similar_mask


def test_compute_similar_mask():
    """Test similar mask computation."""
    # Create a dummy HSV image
    img_hsv = np.random.rand(20, 20, 3)
    avg_hsv = np.array([0.1, 0.4, 0.5])
    mask = compute_similar_mask(img_hsv, avg_hsv)
    assert mask.shape == (20, 20)
    assert mask.dtype == bool
