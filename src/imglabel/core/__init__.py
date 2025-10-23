"""Core package for image processing."""

from .image_loading import load_and_coarsen_image
from .mask_computation import compute_similar_mask, compute_mixed_mask
from .clustering import cluster_and_get_polygons, select_boundary_component
from .line_detection import detect_lines, get_line_boundary_points
from .criteria import save_criteria, load_criteria
