"""Core functions for image processing and labeling."""

import matplotlib.patches as patches
import numpy as np
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage
from skimage.color import rgb2hsv, rgb2gray
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
from skimage.measure import find_contours
import yaml


def load_and_coarsen_image(image_path, coarsen_factor=4):
    """Load image and coarsen by factor, return RGB and HSV versions."""
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    img_coarsened = ndimage.zoom(
        img_array.astype(float), (1 / coarsen_factor, 1 / coarsen_factor, 1), order=1
    ).astype(np.uint8)
    img_hsv = rgb2hsv(img_coarsened)
    return img_coarsened, img_hsv


def compute_similar_mask(img_hsv, avg_hsv, threshold_hue=0.05, threshold_sat=0.2):
    """Compute mask of similar patches using HSV."""
    height, width, _ = img_hsv.shape
    # Sliding windows for H, S, V
    windows_h = sliding_window_view(img_hsv[:, :, 0], (5, 5))
    windows_s = sliding_window_view(img_hsv[:, :, 1], (5, 5))
    windows_v = sliding_window_view(img_hsv[:, :, 2], (5, 5))
    
    # For hue (circular): compute mean using sin/cos
    sin_h = np.sin(2 * np.pi * windows_h)
    cos_h = np.cos(2 * np.pi * windows_h)
    mean_sin = sin_h.mean(axis=(2, 3))
    mean_cos = cos_h.mean(axis=(2, 3))
    mean_h = np.arctan2(mean_sin, mean_cos) / (2 * np.pi)
    mean_h = np.where(mean_h < 0, mean_h + 1, mean_h)
    
    # For saturation: simple mean
    mean_s = windows_s.mean(axis=(2, 3))
    
    # Hue distance (circular)
    hue_dist = np.minimum(np.abs(mean_h - avg_hsv[0]), 1 - np.abs(mean_h - avg_hsv[0]))
    
    # Saturation distance
    sat_dist = np.abs(mean_s - avg_hsv[1])
    
    # Similar mask
    similar_mask = (hue_dist < threshold_hue) & (sat_dist < threshold_sat)
    
    # Expand to full image mask
    full_mask = np.zeros((height, width), dtype=bool)
    full_mask[2:height-2, 2:width-2] = similar_mask
    return full_mask


def compute_mixed_mask(img_hsv, img_rgb, selected_hsv, selected_shape, threshold_hue=0.05, threshold_sat=0.2, threshold_shape=0.5):
    """Compute mask combining HSV and shape similarity."""
    height, width = img_rgb.shape[:2]
    # HSV mask
    hsv_mask = compute_similar_mask(img_hsv, selected_hsv, threshold_hue, threshold_sat)
    # Shape mask: edge density
    img_gray = rgb2gray(img_rgb)
    edges = canny(img_gray, sigma=1.0)
    windows_edges = sliding_window_view(edges, (5, 5))
    edge_densities = windows_edges.mean(axis=(2, 3))
    shape_mask = (edge_densities > selected_shape * (1 - threshold_shape)) & (edge_densities < selected_shape * (1 + threshold_shape))
    # Expand
    full_shape_mask = np.zeros((height, width), dtype=bool)
    full_shape_mask[2:height-2, 2:width-2] = shape_mask
    # Combine
    mixed_mask = hsv_mask & full_shape_mask
    return mixed_mask


def cluster_and_get_polygons(full_mask):
    """Cluster the mask and return polygon patches."""
    struct_elem = ndimage.generate_binary_structure(2, 2)
    dilated = ndimage.binary_dilation(full_mask, structure=struct_elem, iterations=2)
    labeled, num_features = ndimage.label(dilated)
    polygons = []
    for label in range(1, num_features + 1):
        component = labeled == label
        contours = find_contours(component.astype(float), 0.5)
        for cnt in contours:
            points = cnt[:, [1, 0]]
            poly = patches.Polygon(
                points, closed=True, fill=False, edgecolor="red", linewidth=2
            )
            polygons.append(poly)
    return polygons


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


def save_criteria(avg_hsv, threshold_hue, threshold_sat, filename="criteria.yaml"):
    """Save filter criteria to YAML."""
    criteria = {
        "hsv": avg_hsv.tolist(),
        "threshold_hue": threshold_hue,
        "threshold_sat": threshold_sat
    }
    with open(filename, "w") as f:
        yaml.dump(criteria, f)


def load_criteria(filename="criteria.yaml"):
    """Load filter criteria from YAML."""
    with open(filename, "r") as f:
        criteria = yaml.safe_load(f)
    return np.array(criteria["hsv"]), criteria["threshold_hue"], criteria["threshold_sat"]
