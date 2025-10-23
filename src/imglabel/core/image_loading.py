"""Load and coarsen images."""

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.color import rgb2hsv


def load_and_coarsen_image(image_path, coarsen_factor=4):
    """Load image and coarsen by factor, return RGB and HSV versions."""
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    img_coarsened = ndimage.zoom(
        img_array.astype(float), (1 / coarsen_factor, 1 / coarsen_factor, 1), order=1
    ).astype(np.uint8)
    img_hsv = rgb2hsv(img_coarsened)
    return img_coarsened, img_hsv
