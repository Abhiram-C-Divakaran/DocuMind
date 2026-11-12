from pathlib import Path
from typing import List, Tuple
import numpy as np
import pymupdf


def convert_pdf_to_images(
    pdf_path: Path,
    dpi: int = 300,
    max_pages: int = 5,
) -> Tuple[List[np.ndarray], int]:
    """
    Converts pages of a PDF document to RGB NumPy image arrays at the requested DPI.

    Returns:
        Tuple of (images_list, total_pages_in_pdf)
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

    images: List[np.ndarray] = []
    doc = pymupdf.open(str(pdf_path))
    total_pages = len(doc)

    try:
        pages_to_render = min(total_pages, max_pages)
        scale = dpi / 72.0
        matrix = pymupdf.Matrix(scale, scale)

        for page_idx in range(pages_to_render):
            page = doc[page_idx]
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )
            # Make a copy so buffer remains valid after pixmap destruction
            images.append(img_array.copy())

        return images, total_pages
    finally:
        doc.close()

