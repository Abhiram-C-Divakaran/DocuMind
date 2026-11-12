import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import pytesseract
from PIL import Image

from backend.app.config import settings
from backend.app.logger import logger
from ml.ocr.pdf_converter import convert_pdf_to_images
from ml.ocr.preprocessor import preprocess_image_for_ocr


@dataclass
class OCRToken:
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x0, y0, x1, y1) in original pixel coordinates
    norm_bbox: Tuple[int, int, int, int]  # (x0, y0, x1, y1) normalized to 0-1000 range
    page_num: int
    line_num: int
    block_num: int


@dataclass
class OCRPage:
    page_num: int
    width: int
    height: int
    text: str
    tokens: List[OCRToken] = field(default_factory=list)


@dataclass
class OCRResult:
    full_text: str
    pages: List[OCRPage]
    tokens: List[OCRToken]
    avg_confidence: float
    total_pages: int
    processed_pages: int
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def locate_tesseract_binary() -> Optional[str]:
    """
    Finds the Tesseract binary across environment variables, PATH,
    and common standard installation directories on Windows and Linux.
    """
    if settings.TESSERACT_CMD and Path(settings.TESSERACT_CMD).exists():
        return settings.TESSERACT_CMD

    path_binary = shutil.which("tesseract")
    if path_binary:
        return path_binary

    candidate_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/opt/homebrew/bin/tesseract",
    ]

    for cand in candidate_paths:
        if cand and Path(cand).exists():
            return cand

    return None


def get_tesseract_status() -> Tuple[bool, Optional[str], str]:
    """
    Checks if Tesseract OCR is available.
    Returns:
        (is_available, executable_path, message)
    """
    binary = locate_tesseract_binary()
    if not binary:
        msg = (
            "Tesseract executable is not installed or not in PATH. "
            "Installation instructions:\n"
            "- Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki "
            "and add 'C:\\Program Files\\Tesseract-OCR' to PATH or set TESSERACT_CMD in .env.\n"
            "- Ubuntu/Debian: Run 'sudo apt-get update && sudo apt-get install -y tesseract-ocr'.\n"
            "- macOS: Run 'brew install tesseract'."
        )
        return False, None, msg

    try:
        pytesseract.pytesseract.tesseract_cmd = binary
        version = pytesseract.get_tesseract_version()
        return True, binary, f"Tesseract v{version} detected at '{binary}'"
    except Exception as e:
        return False, binary, f"Failed to invoke Tesseract at '{binary}': {str(e)}"


def normalize_coordinates(
    x: int, y: int, w: int, h: int, img_w: int, img_h: int
) -> Tuple[int, int, int, int]:
    """
    Normalizes a pixel bounding box (x, y, w, h) to a 0–1000 integer coordinate space
    format [x0, y0, x1, y1] for multimodal transformer (LayoutLMv3) compatibility.
    """
    if img_w <= 0 or img_h <= 0:
        return (0, 0, 0, 0)

    x0 = max(0, min(1000, int((x / img_w) * 1000)))
    y0 = max(0, min(1000, int((y / img_h) * 1000)))
    x1 = max(0, min(1000, int(((x + w) / img_w) * 1000)))
    y1 = max(0, min(1000, int(((y + h) / img_h) * 1000)))

    return (x0, y0, x1, y1)


def execute_tesseract_ocr(
    file_path: Path,
    dpi: int = settings.OCR_PDF_DPI,
    max_pages: int = settings.OCR_MAX_PAGES,
) -> OCRResult:
    """
    Executes real Tesseract OCR on a document (PDF, PNG, JPEG).
    Extracts text, word-level bounding boxes, normalized coordinates, and token confidences.
    """
    is_available, binary, msg = get_tesseract_status()
    if not is_available:
        raise RuntimeError(msg)

    pytesseract.pytesseract.tesseract_cmd = binary

    # 1. Load images depending on file type
    suffix = file_path.suffix.lower()
    images: List[np.ndarray] = []
    total_pages = 1
    warnings: List[str] = []

    if suffix == ".pdf":
        images, total_pages = convert_pdf_to_images(
            file_path, dpi=dpi, max_pages=max_pages
        )
        if total_pages > max_pages:
            warnings.append(
                f"Document contains {total_pages} pages; processed first {max_pages} pages per resource limit."
            )
    elif suffix in [".png", ".jpg", ".jpeg"]:
        try:
            # Use PIL then convert to numpy RGB to avoid OpenCV Unicode path issues on Windows
            pil_img = Image.open(str(file_path))
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            images = [np.array(pil_img)]
            total_pages = 1
        except Exception as e:
            raise ValueError(f"Could not load image '{file_path.name}': {e}")
    else:
        raise ValueError(f"Unsupported file format for OCR: {suffix}")

    if not images:
        raise ValueError(f"No readable image frames extracted from '{file_path.name}'.")

    all_tokens: List[OCRToken] = []
    all_pages: List[OCRPage] = []
    full_text_parts: List[str] = []

    for page_idx, raw_img in enumerate(images, start=1):
        h, w = raw_img.shape[:2]

        # 2. Preprocess image for OCR
        processed_img = preprocess_image_for_ocr(raw_img)

        # 3. Extract text and token data
        try:
            page_text = pytesseract.image_to_string(processed_img, config="--psm 6")
        except Exception as e:
            logger.warning(f"Error on page {page_idx} full text extraction: {e}")
            page_text = ""

        try:
            data = pytesseract.image_to_data(
                processed_img,
                output_type=pytesseract.Output.DICT,
                config="--psm 6",
            )
        except Exception as e:
            logger.warning(f"Error on page {page_idx} token data extraction: {e}")
            data = {"text": [], "conf": [], "left": [], "top": [], "width": [], "height": [], "line_num": [], "block_num": []}

        page_tokens: List[OCRToken] = []
        n_boxes = len(data.get("text", []))

        for i in range(n_boxes):
            word_str = data["text"][i].strip()
            conf_val = float(data["conf"][i])

            # Filter out non-word empty noise
            if not word_str or conf_val < 0:
                continue

            orig_bbox = (
                int(data["left"][i]),
                int(data["top"][i]),
                int(data["left"][i] + data["width"][i]),
                int(data["top"][i] + data["height"][i]),
            )
            norm_bbox = normalize_coordinates(
                data["left"][i], data["top"][i], data["width"][i], data["height"][i], w, h
            )

            token = OCRToken(
                text=word_str,
                confidence=conf_val / 100.0,  # Normalize 0-100 to 0.0-1.0
                bbox=orig_bbox,
                norm_bbox=norm_bbox,
                page_num=page_idx,
                line_num=int(data["line_num"][i]),
                block_num=int(data["block_num"][i]),
            )
            page_tokens.append(token)
            all_tokens.append(token)

        full_text_parts.append(page_text.strip())
        all_pages.append(
            OCRPage(
                page_num=page_idx,
                width=w,
                height=h,
                text=page_text.strip(),
                tokens=page_tokens,
            )
        )

    # Calculate overall average OCR confidence
    if all_tokens:
        avg_conf = sum(t.confidence for t in all_tokens) / len(all_tokens)
    else:
        avg_conf = 0.0
        warnings.append("No readable text tokens detected in document.")

    combined_full_text = "\n\n--- Page Break ---\n\n".join(full_text_parts)

    return OCRResult(
        full_text=combined_full_text,
        pages=all_pages,
        tokens=all_tokens,
        avg_confidence=round(avg_conf, 3),
        total_pages=total_pages,
        processed_pages=len(images),
        warnings=warnings,
        metadata={
            "tesseract_path": binary,
            "dpi": dpi,
            "total_tokens": len(all_tokens),
        },
    )

