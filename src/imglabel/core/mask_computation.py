"""Compute similarity masks."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from skimage.color import rgb2gray
from skimage.feature import canny


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
