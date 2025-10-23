"""Cluster masks and get polygons."""

import matplotlib.patches as patches
import numpy as np
from scipy import ndimage
from skimage.measure import find_contours


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


def select_boundary_component(full_mask, y, x):
    """If the component at (y,x) touches boundary, return mask of only that component, else return full_mask."""
    labeled, num_features = ndimage.label(full_mask)
    if num_features == 0:
        return full_mask
    component_label = labeled[y, x]
    if component_label == 0:
        return full_mask
    component_mask = labeled == component_label
    height, width = full_mask.shape
    touches_boundary = (
        np.any(component_mask[0, :]) or
        np.any(component_mask[-1, :]) or
        np.any(component_mask[:, 0]) or
        np.any(component_mask[:, -1])
    )
    if touches_boundary:
        return component_mask
    else:
        return full_mask
