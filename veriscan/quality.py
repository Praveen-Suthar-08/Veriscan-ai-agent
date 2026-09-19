"""Image quality assessment module for VeriScan.

Calculates blur (Laplacian variance), skew angle, and resolution DPI.
"""

from __future__ import annotations
from typing import Tuple, List, Dict, Any
import numpy as np
import cv2
from pathlib import Path
import yaml  # type: ignore

from veriscan.schemas import QualityMetrics

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
_CONFIG: Dict[str, Any] = {}
if _CONFIG_PATH.exists():
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict):
                _CONFIG = data
    except Exception:
        _CONFIG = {}

MIN_BLUR_LAPLACIAN = _CONFIG.get("quality", {}).get("min_blur_laplacian", 100.0)
MAX_SKEW_DEGREES = _CONFIG.get("quality", {}).get("max_skew_degrees", 3.0)
MIN_DPI = _CONFIG.get("quality", {}).get("min_dpi", 150)


def calculate_blur(image: np.ndarray) -> float:
    """Calculate blur metric using variance of Laplacian."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(laplacian.var())
    return round(variance, 2)


def calculate_skew(image: np.ndarray) -> float:
    """Calculate skew angle in degrees using image moments / minAreaRect on contours."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Threshold inverted
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find non-zero points
    pts = cv2.findNonZero(thresh)
    if pts is None or len(pts) < 100:
        return 0.0

    rect = cv2.minAreaRect(pts)
    angle = rect[-1]

    # Normalize OpenCV angle convention (-90 to 0)
    if angle < -45:
        angle = -(90 + angle)
    elif angle > 45:
        angle = 90 - angle
    else:
        angle = -angle

    # Limit to reasonable document skew range
    if abs(angle) > 45:
        return 0.0

    return round(float(angle), 2)


def assess_image_quality(image: np.ndarray, estimated_dpi: int = 200) -> QualityMetrics:
    """Run comprehensive image quality assessment."""
    warnings: List[str] = []

    # 1. Blur
    blur_score = calculate_blur(image)
    is_blurry = blur_score < MIN_BLUR_LAPLACIAN
    if is_blurry:
        warnings.append(
            f"Image appears blurry (Laplacian variance {blur_score:.1f} < {MIN_BLUR_LAPLACIAN:.1f}). OCR accuracy may be degraded."
        )

    # 2. Skew
    skew_angle = calculate_skew(image)
    is_skewed = abs(skew_angle) > MAX_SKEW_DEGREES
    if is_skewed:
        warnings.append(
            f"Document skew detected ({skew_angle:.1f}°). Automatic deskewing recommended."
        )

    # 3. Resolution / DPI
    is_low_res = estimated_dpi < MIN_DPI
    if is_low_res:
        warnings.append(
            f"Low estimated resolution ({estimated_dpi} DPI < {MIN_DPI} DPI). Text may be illegible."
        )

    return QualityMetrics(
        blur_score=blur_score,
        is_blurry=is_blurry,
        skew_angle=skew_angle,
        is_skewed=is_skewed,
        resolution_dpi=estimated_dpi,
        is_low_res=is_low_res,
        warnings=warnings
    )
