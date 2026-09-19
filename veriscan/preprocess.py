"""Image preprocessing and enhancement pipeline for VeriScan.

Provides basic preparation and targeted retry enhancements:
- Grayscale conversion
- Deskewing
- Denoising & adaptive thresholding
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- 2x upscale & unsharp mask sharpening
"""

from __future__ import annotations
import numpy as np
import cv2
from typing import Tuple, Optional, List
from veriscan.quality import calculate_skew


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale if necessary."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def deskew(image: np.ndarray, angle: Optional[float] = None) -> np.ndarray:
    """Rotate image by negative skew angle to rectify orientation."""
    if angle is None:
        angle = calculate_skew(image)

    if abs(angle) < 0.5:
        return image.copy()

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(
        image, rot_mat, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return deskewed


def denoise_and_threshold(image: np.ndarray) -> np.ndarray:
    """Apply bilateral filtering and Otsu thresholding."""
    gray = to_grayscale(image)
    # Bilateral filter to smooth texture while preserving text edges
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    # Adaptive threshold
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def enhance_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Enhance local contrast using CLAHE (for degraded / unevenly lit scans)."""
    gray = to_grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)
    return enhanced


def upscale_and_sharpen(image: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """Upscale low-resolution image and apply unsharp mask filter."""
    h, w = image.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Unsharp mask
    gaussian = cv2.GaussianBlur(upscaled, (0, 0), 2.0)
    sharpened = cv2.addWeighted(upscaled, 1.5, gaussian, -0.5, 0)
    return sharpened


def crop_region(image: np.ndarray, bbox: List[int], padding: int = 15) -> np.ndarray:
    """
    Safely crop sub-region [x, y, w, h] from image with optional padding.
    """
    x, y, w, h = bbox
    img_h, img_w = image.shape[:2]

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)

    return image[y1:y2, x1:x2].copy()
