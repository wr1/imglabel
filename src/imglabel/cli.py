"""Command-line interface for imglabel."""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys

from .core import (
    load_and_coarsen_image,
    compute_similar_mask,
    cluster_and_get_polygons,
    save_criteria,
    load_criteria,
    select_boundary_component,
)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Label paint damages in images.")
    parser.add_argument("image", help="Path to the image file.")
    parser.add_argument(
        "-c",
        "--criteria",
        help="Path to criteria YAML file to apply directly.",
    )
    args = parser.parse_args()

    try:
        img_coarsened, img_hsv = load_and_coarsen_image(args.image)
    except Exception as e:
        print(f"Error loading image: {e}", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots()
    ax.imshow(img_coarsened)
    ax.set_title("Paint damage labeling")

    if args.criteria:
        # Apply criteria directly
        try:
            avg_hsv, threshold_hue, threshold_sat = load_criteria(args.criteria)
        except Exception as e:
            print(f"Error loading criteria: {e}", file=sys.stderr)
            sys.exit(1)
        full_mask = compute_similar_mask(img_hsv, avg_hsv, threshold_hue, threshold_sat)
        polygons = cluster_and_get_polygons(full_mask)
        for poly in polygons:
            ax.add_patch(poly)
        plt.show()
    else:
        # Interactive mode
        bbox_patches = []
        selected_hsv = None
        threshold_hue = 0.05
        threshold_sat = 0.2

        def on_click(event):
            nonlocal bbox_patches, selected_hsv
            if event.xdata is None or event.ydata is None:
                return
            x, y = int(event.xdata), int(event.ydata)
            height, width, _ = img_hsv.shape
            half_size = 2
            x_start = max(0, x - half_size)
            x_end = min(width, x + half_size + 1)
            y_start = max(0, y - half_size)
            y_end = min(height, y + half_size + 1)
            patch_hsv = img_hsv[y_start:y_end, x_start:x_end]
            if patch_hsv.size == 0:
                return
            # Compute average HSV
            h_patch = patch_hsv[:, :, 0]
            s_patch = patch_hsv[:, :, 1]
            v_patch = patch_hsv[:, :, 2]
            # Hue mean (circular)
            sin_h = np.sin(2 * np.pi * h_patch)
            cos_h = np.cos(2 * np.pi * h_patch)
            mean_sin = np.mean(sin_h)
            mean_cos = np.mean(cos_h)
            mean_h = np.arctan2(mean_sin, mean_cos) / (2 * np.pi)
            if mean_h < 0:
                mean_h += 1
            mean_s = np.mean(s_patch)
            mean_v = np.mean(v_patch)
            selected_hsv = np.array([mean_h, mean_s, mean_v])
            
            for patch in bbox_patches:
                patch.remove()
            bbox_patches.clear()
            full_mask = compute_similar_mask(img_hsv, selected_hsv, threshold_hue, threshold_sat)
            full_mask = select_boundary_component(full_mask, y, x)
            polygons = cluster_and_get_polygons(full_mask)
            for poly in polygons:
                ax.add_patch(poly)
                bbox_patches.append(poly)
            plt.draw()

        def on_close(event):
            if selected_hsv is not None:
                save_criteria(selected_hsv, threshold_hue, threshold_sat)
                print("Filter criteria saved to criteria.yaml")

        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("close_event", on_close)
        plt.show()


if __name__ == "__main__":
    main()
