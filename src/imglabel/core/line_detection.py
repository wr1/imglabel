"""Detect lines in images."""

import numpy as np
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks


def detect_lines(img_gray, sigma=1.5):
    """Detect lines in the grayscale image using Hough transform."""
    edges = canny(img_gray, sigma=sigma)
    h, theta, d = hough_line(edges)
    threshold = 10 + (3.0 - sigma) * 5  # Adjust threshold
    accum, angles, dists = hough_line_peaks(h, theta, d, num_peaks=100, threshold=threshold)
    lines = list(zip(dists, angles))
    return lines


def get_line_boundary_points(rho, theta, width, height):
    """Get the two boundary points of a line within the image."""
    a = np.cos(theta)
    b = np.sin(theta)
    points = []
    # Intersect with left x=0
    if abs(a) > 1e-6:
        y = (rho - 0 * a) / b
        if 0 <= y <= height:
            points.append((0, y))
    # Right x=width
    if abs(a) > 1e-6:
        y = (rho - width * a) / b
        if 0 <= y <= height:
            points.append((width, y))
    # Top y=0
    if abs(b) > 1e-6:
        x = (rho - 0 * b) / a
        if 0 <= x <= width:
            points.append((x, 0))
    # Bottom y=height
    if abs(b) > 1e-6:
        x = (rho - height * b) / a
        if 0 <= x <= width:
            points.append((x, height))
    return points[:2]
