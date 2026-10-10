# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import sys

# Allow helpers package to be found when this module is imported standalone
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from helpers.dataloader import load_image, get_data_path

SPHINX_IMAGE = "2560px-Great_Sphinx_of_Giza_-_20080716a.jpg"

def read_image():
    """Load and preprocess the Sphinx image: scale by 0.5 and convert to greyscale."""
    image_path = get_data_path(SPHINX_IMAGE)
    return load_image(image_path, scale_factor=2, as_gray=True)

# Skeleton Code will generate a 1D Gaussian Kernel for given sigma.
# You need to implement 2D convolution as efficiently as possible.
class GaussianFilt:
    def __init__(self, sigma):
        self.sigma = sigma

    def gauss_kernel(self):
        """Generate a normalised 1D Gaussian kernel of appropriate size for self.sigma.

        Returns:
            gauss_1d (ndarray): Shape (k_size, 1) — column vector kernel.
        """
        sigma = float(self.sigma)
        if sigma <= 0:
            raise ValueError("sigma must be positive.")

        k_size = int(6 * sigma + 1)
        offsets = np.arange(k_size, dtype=np.float64) - (k_size - 1) / 2.0
        kernel = np.exp(-(offsets ** 2) / (2.0 * sigma ** 2))
        kernel /= np.sum(kernel)
        return np.expand_dims(kernel.astype(np.float64), axis=1)

    def my_conv_method(self, image):
        """Perform separable 2D Gaussian convolution via two 1D passes.

        Args:
            image (ndarray): 2D greyscale image (H, W), any numeric dtype.

        Returns:
            conv_img (ndarray): Blurred image of same shape, dtype uint8.
        """
        image_array = np.asarray(image)
        if image_array.ndim != 2:
            raise ValueError("Image must be a 2D array.")

        kernel = self.gauss_kernel().ravel()
        pad = kernel.size // 2

        padded = np.pad(image_array.astype(np.float64), ((pad, pad), (pad, pad)), mode="edge")
        filtered_x = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded)
        filtered_xy = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="valid"), 0, filtered_x)

        return np.clip(filtered_xy, 0, 255).astype(np.uint8)
