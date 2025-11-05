# --------------------------------------------------------------
#  gabor_strength_overlay.py
# --------------------------------------------------------------
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle


# ------------------- Gabor kernels -------------------
def gabor_kernel(size, sigma, theta, lambda_, gamma, psi=0):
    size = int(size)
    y, x = np.mgrid[-size : size + 1, -size : size + 1]
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gb = np.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2)) * np.cos(
        2 * np.pi * x_theta / lambda_ + psi
    )
    return gb


def apply_gabor_filters(img_gray, size=21, sigma=4.0, lambda_=10.0, gamma=0.5):
    kernel_h = gabor_kernel(size, sigma, 0, lambda_, gamma)
    kernel_v = gabor_kernel(size, sigma, np.pi / 2, lambda_, gamma)

    resp_h = ndimage.convolve(img_gray, kernel_h, mode="reflect")
    resp_v = ndimage.convolve(img_gray, kernel_v, mode="reflect")

    combined = np.sqrt(resp_h**2 + resp_v**2)
    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
    return combined


# ------------------- Region growing -------------------
def region_from_click(response_map, click_point, tolerance=0.15):
    """Return a binary mask of the connected component that contains the click."""
    seed_val = response_map[click_point[1], click_point[0]]
    candidate = np.abs(response_map - seed_val) <= tolerance

    labeled, _ = ndimage.label(candidate)
    component_id = labeled[click_point[1], click_point[0]]
    if component_id == 0:
        return np.zeros_like(candidate, dtype=bool)

    return labeled == component_id


# ------------------- Auto detection (unchanged) -------------------
def detect_regions_auto(response_map, thresh=0.55):
    mask = response_map > thresh
    mask = ndimage.binary_opening(mask, structure=np.ones((5, 5)))
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5)))
    return mask


# ------------------- Main -------------------
def main(image_path, click_mode=False):
    # --- load ---
    img = mpimg.imread(image_path)
    if img.ndim == 3:
        gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        gray = img.copy()
    gray = gray.astype(np.float32)

    # --- Gabor response ---
    response = apply_gabor_filters(gray, size=21, sigma=4.0, lambda_=10.0, gamma=0.5)

    # --- auto mask ---
    mask_auto = detect_regions_auto(response, thresh=0.55)

    # --- click-based region (strength map) ---
    click_point = None
    strength_mask = None  # <-- will hold the *continuous* overlay
    if click_mode:
        plt.figure(figsize=(8, 6))
        plt.imshow(gray, cmap="gray")
        plt.title("Click inside the 90° textile band")
        plt.axis("off")
        click = plt.ginput(1, timeout=0)[0]
        plt.close()

        click_point = (int(click[0]), int(click[1]))
        if not (
            0 <= click_point[0] < gray.shape[1] and 0 <= click_point[1] < gray.shape[0]
        ):
            print("Click outside image – skipping click mode")
        else:
            binary_region = region_from_click(response, click_point, tolerance=0.15)
            # *** NEW: keep the original response values inside the region ***
            strength_mask = response.copy()
            strength_mask[~binary_region] = np.nan  # transparent outside region

    # --- visualisation ---
    n_cols = 4 if strength_mask is not None else 3
    fig, axs = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    # 1) original
    axs[0].imshow(gray, cmap="gray")
    axs[0].set_title("Original")
    axs[0].axis("off")

    # 2) raw Gabor response
    im = axs[1].imshow(response, cmap="hot", vmin=0, vmax=1)
    axs[1].set_title("Gabor Response")
    axs[1].axis("off")
    plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)

    # 3) auto-threshold overlay
    axs[2].imshow(gray, cmap="gray")
    axs[2].imshow(mask_auto, cmap="Greens", alpha=0.5)
    axs[2].set_title("Auto-Detected")
    axs[2].axis("off")

    if strength_mask is not None:
        # 4) **continuous strength overlay**
        axs[3].imshow(gray, cmap="gray")
        # viridis = blue (weak) → yellow (strong)
        im_strength = axs[3].imshow(
            strength_mask, cmap="viridis", vmin=0, vmax=1, alpha=0.7
        )
        axs[3].plot(
            click_point[0], click_point[1], "r+", markersize=15, markeredgewidth=2
        )
        axs[3].set_title("Click-Based Strength")
        axs[3].axis("off")
        plt.colorbar(
            im_strength, ax=axs[3], fraction=0.046, pad=0.04, label="Pattern strength"
        )

    plt.tight_layout()
    plt.show()


# ------------------- CLI -------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python gabor_strength_overlay.py <image> [--click]")
        sys.exit(1)
    img_path = sys.argv[1]
    click = "--click" in sys.argv
    main(img_path, click_mode=click)
