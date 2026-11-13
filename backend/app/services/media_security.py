from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from backend.app.config import settings
from backend.app.core.exceptions import FileValidationError

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


def validate_media_safety(content: bytes, content_type: str, filename: str) -> None:
    """
    Validates uploaded PDF/image safety without exposing host paths:
    - Enforces bounded pixel count on PNG/JPEG headers to defeat decompression bombs.
    - Enforces PDF structural validity and maximum page count.
    """
    if content_type in ("image/png", "image/jpeg"):
        try:
            Image.MAX_IMAGE_PIXELS = settings.MAX_IMAGE_PIXELS
            with Image.open(BytesIO(content)) as img:
                width, height = img.size
                if width <= 0 or height <= 0:
                    raise FileValidationError("Image has invalid zero or negative dimensions.")
                if width * height > settings.MAX_IMAGE_PIXELS:
                    raise FileValidationError(
                        f"Image pixel count ({width * height}) exceeds maximum permitted limit ({settings.MAX_IMAGE_PIXELS})."
                    )
                img.verify()
        except Image.DecompressionBombError:
            raise FileValidationError("Image exceeds decompression safety limits (decompression bomb rejected).")
        except FileValidationError:
            raise
        except (UnidentifiedImageError, OSError, ValueError):
            raise FileValidationError(f"Uploaded file '{Path(filename).name}' is not a readable image.")

    elif content_type == "application/pdf":
        if fitz is not None:
            try:
                doc = fitz.open(stream=content, filetype="pdf")
                try:
                    page_count = doc.page_count
                finally:
                    doc.close()
                if page_count <= 0:
                    raise FileValidationError("Uploaded PDF contains no pages.")
                if page_count > settings.MAX_PDF_PAGES_HARD_LIMIT:
                    raise FileValidationError(
                        f"PDF page count ({page_count}) exceeds maximum permitted limit ({settings.MAX_PDF_PAGES_HARD_LIMIT})."
                    )
            except FileValidationError:
                raise
            except Exception:
                raise FileValidationError("Uploaded PDF is malformed or unreadable.")

