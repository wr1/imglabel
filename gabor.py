import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.image as mpimg  # For imread


# -----------------------------
# Gabor Filter Functions
# -----------------------------
def gabor_kernel(size, sigma, theta, lambda_, gamma, psi=0):
    """Create a 2D Gabor kernel."""
    size = int(size)
    y, x = np.mgrid[-size : size + 1, -size : size + 1]
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gb = np.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2)) * np.cos(
        2 * np.pi * x_theta / lambda_ + psi
    )
    return gb


def apply_gabor_filters(img_gray, size=15, sigma=3.0, lambda_=8.0, gamma=0.5):
    """Apply horizontal and vertical Gabor filters."""
    kernel_h = gabor_kernel(size, sigma, 0, lambda_, gamma)
    kernel_v = gabor_kernel(size, sigma, np.pi / 2, lambda_, gamma)

    response_h = ndimage.convolve(img_gray, kernel_h, mode="reflect")
    response_v = ndimage.convolve(img_gray, kernel_v, mode="reflect")

    # Combine responses (energy where both orientations are strong)
    combined = np.sqrt(response_h**2 + response_v**2)
    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
    return combined


def detect_regions_auto(response_map, threshold=0.6):
    """Threshold-based automatic detection."""
    mask = response_map > threshold
    mask = ndimage.binary_opening(mask, structure=np.ones((5, 5)))
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5)))
    return mask


def detect_regions_with_click(response_map, click_point, tolerance=0.12):
    """Flood-fill from user click using response value tolerance."""
    from scipy.ndimage import label, find_objects

    seed_val = response_map[click_point[1], click_point[0]]
    mask = np.abs(response_map - seed_val) <= tolerance

    labeled, num_features = label(mask)
    component_id = labeled[click_point[1], click_point[0]]
    if component_id == 0:
        return np.zeros_like(mask, dtype=bool)

    region_mask = labeled == component_id
    return region_mask


# -----------------------------
# Main Script
# -----------------------------
def main(image_path, click_mode=False):
    # Load image
    img = mpimg.imread(image_path)
    if img.ndim == 3:
        img_gray = np.dot(
            img[..., :3], [0.2989, 0.5870, 0.1140]
        )  # Convert to grayscale
    else:
        img_gray = img

    img_gray = img_gray.astype(np.float32)
    h, w = img_gray.shape

    # Apply Gabor filters
    response_map = apply_gabor_filters(
        img_gray, size=21, sigma=4.0, lambda_=10.0, gamma=0.5
    )

    # Auto detection
    mask_auto = detect_regions_auto(response_map, threshold=0.55)

    # Interactive click mode
    mask_click = None
    click_point = None
    if click_mode:
        plt.figure(figsize=(10, 8))
        plt.imshow(img_gray, cmap="gray")
        plt.title("Click on a region with 90-degree textile pattern")
        plt.axis("off")
        click = plt.ginput(1, timeout=0)[0]  # Wait for one click
        plt.close()

        click_point = (int(click[0]), int(click[1]))
        if 0 <= click_point[0] < w and 0 <= click_point[1] < h:
            mask_click = detect_regions_with_click(
                response_map, click_point, tolerance=0.15
            )
        else:
            print("Click outside image bounds.")
            mask_click = np.zeros_like(mask_auto)

    # -----------------------------
    # Visualization
    # -----------------------------
    fig, axs = plt.subplots(1, 4 if mask_click is not None else 3, figsize=(16, 5))

    axs[0].imshow(img_gray, cmap="gray")
    axs[0].set_title("Original Grayscale")
    axs[0].axis("off")

    axs[1].imshow(response_map, cmap="hot")
    axs[1].set_title("Gabor Response Map")
    axs[1].axis("off")

    axs[2].imshow(img_gray, cmap="gray")
    axs[2].imshow(mask_auto, cmap="Greens", alpha=0.5)
    axs[2].set_title("Auto Detected Regions")
    axs[2].axis("off")

    if mask_click is not None:
        axs[3].imshow(img_gray, cmap="gray")
        axs[3].imshow(mask_click, cmap="Blues", alpha=0.6)
        axs[3].plot(
            click_point[0], click_point[1], "r+", markersize=15, markeredgewidth=2
        )
        axs[3].set_title("Click-Based Region")
        axs[3].axis("off")

    plt.tight_layout()
    plt.show()

    # Optional: Save results
    # plt.imsave('mask_auto.png', mask_auto, cmap='gray')
    # if mask_click is not None:
    #     plt.imsave('mask_click.png', mask_click, cmap='gray')


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python gabor.py <image_path> [--click]")
        sys.exit(1)

    image_path = sys.argv[1]
    click_mode = "--click" in sys.argv

    main(image_path, click_mode=click_mode)
