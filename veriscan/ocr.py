"""OCR Engine abstraction supporting EasyOCR, Tesseract, and cached token storage."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
try:
    import numpy as np  # type: ignore
except ImportError:
    np = None  # type: ignore

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None  # type: ignore

from veriscan.schemas import Token

# Optional library imports
try:
    import pytesseract  # type: ignore
except (ImportError, Exception):
    pytesseract = None

try:
    import easyocr  # type: ignore
except (ImportError, Exception):
    easyocr = None



class OCREngine(ABC):
    """Abstract base class for OCR engines."""

    @abstractmethod
    def extract_tokens(self, image: np.ndarray) -> List[Token]:
        """Perform OCR on image (BGR numpy array) and return list of Token objects."""
        pass


class EasyOCREngine(OCREngine):
    """EasyOCR implementation with lazy reader initialization."""

    def __init__(self, languages: Optional[List[str]] = None, gpu: bool = False):
        self.languages = languages or ["en"]
        self.gpu = gpu
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            if easyocr is None:
                raise ImportError("EasyOCR is not installed. Run 'pip install easyocr'.")
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
        return self._reader

    def extract_tokens(self, image: np.ndarray) -> List[Token]:
        try:
            reader = self._get_reader()
            # EasyOCR expects RGB or grayscale
            if len(image.shape) == 3:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb = image
            results = reader.readtext(rgb)
        except Exception:
            return []

        tokens: List[Token] = []

        for i, (bbox_pts, text, conf) in enumerate(results):
            text_str = str(text).strip()
            if not text_str:
                continue

            # Convert 4 corner points [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] to [x, y, w, h]
            pts = np.array(bbox_pts, dtype=np.int32)
            x_min = int(np.min(pts[:, 0]))
            y_min = int(np.min(pts[:, 1]))
            x_max = int(np.max(pts[:, 0]))
            y_max = int(np.max(pts[:, 1]))
            w = max(1, x_max - x_min)
            h = max(1, y_max - y_min)

            tokens.append(
                Token(
                    text=text_str,
                    conf=round(float(conf), 3),
                    bbox=[x_min, y_min, w, h],
                    line_id=i
                )
            )

        return tokens


class TesseractEngine(OCREngine):
    """Tesseract OCR engine using pytesseract.image_to_data."""

    def __init__(self, psm: int = 6):
        self.psm = psm

    def extract_tokens(self, image: np.ndarray) -> List[Token]:
        if pytesseract is None:
            raise ImportError("pytesseract is not installed. Run 'pip install pytesseract'.")

        if len(image.shape) == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = image

        config = f"--psm {self.psm}"
        try:
            data = pytesseract.image_to_data(rgb, config=config, output_type=pytesseract.Output.DICT)
        except Exception:
            return []


        tokens: List[Token] = []
        n_boxes = len(data["text"])

        for i in range(n_boxes):
            text = str(data["text"][i]).strip()
            conf_str = data["conf"][i]
            try:
                conf = float(conf_str) / 100.0
            except (ValueError, TypeError):
                conf = 0.0

            if not text or conf < 0:
                continue

            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])
            line_id = int(data["line_num"][i])

            tokens.append(
                Token(
                    text=text,
                    conf=round(min(1.0, max(0.0, conf)), 3),
                    bbox=[x, y, w, h],
                    line_id=line_id
                )
            )

        return tokens


def get_ocr_engine(engine_name: str = "easyocr") -> OCREngine:
    """Factory to get the requested OCR engine with intelligent fallback."""
    engine_name = engine_name.lower().strip()
    if engine_name == "tesseract":
        return TesseractEngine()
    return EasyOCREngine()


def save_tokens_cache(tokens: List[Token], cache_path: Path) -> None:
    """Save tokens to JSON cache file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = [t.model_dump() for t in tokens]
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_tokens_cache(cache_path: Path) -> Optional[List[Token]]:
    """Load cached tokens if file exists."""
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [Token(**item) for item in data]
    except Exception:
        return None
