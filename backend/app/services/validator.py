import os
from typing import Tuple
from backend.app.config import settings
from backend.app.core.exceptions import FileSizeLimitExceededError, FileValidationError

# Allowed MIME types and corresponding extensions & magic byte signatures
ALLOWED_MIME_TYPES = {
    "application/pdf": [".pdf"],
    "image/png": [".png"],
    "image/jpeg": [".jpg", ".jpeg"],
}

# Signatures (magic bytes)
SIGNATURES = {
    "application/pdf": [b"%PDF-"],
    "image/png": [b"\x89PNG\r\n\x1a\n"],
    "image/jpeg": [b"\xff\xd8\xff"],
}


def detect_magic_signature(content: bytes) -> str | None:
    """Detect format from raw byte header."""
    if len(content) < 8:
        return None

    if content.startswith(b"%PDF-"):
        return "application/pdf"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"

    return None


def validate_file(
    filename: str,
    content_type: str,
    file_bytes: bytes,
    max_size_bytes: int = settings.max_file_size_bytes,
) -> Tuple[str, str]:
    """
    Validates uploaded file against:
    1. Empty file checks
    2. File size constraints
    3. Filename extension
    4. Declared MIME type
    5. Actual magic byte signature

    Returns:
        Tuple of (verified_mime_type, normalized_extension)
    """
    if not file_bytes or len(file_bytes) == 0:
        raise FileValidationError("Uploaded file is empty.")

    if len(file_bytes) > max_size_bytes:
        raise FileSizeLimitExceededError(
            f"File size ({len(file_bytes)} bytes) exceeds maximum allowed limit of {settings.MAX_FILE_SIZE_MB} MB.",
            max_size_mb=settings.MAX_FILE_SIZE_MB,
        )

    # 1. Normalize and check extension
    _, ext = os.path.splitext(filename.lower())
    if not ext:
        raise FileValidationError("Filename must have an extension (.pdf, .png, .jpg, .jpeg).")

    # Normalize .jpeg to .jpg for consistent naming
    if ext == ".jpeg":
        ext = ".jpg"

    valid_extensions = [ext for exts in ALLOWED_MIME_TYPES.values() for ext in exts]
    if ext not in valid_extensions and ext != ".jpg":
        raise FileValidationError(
            f"Unsupported file extension '{ext}'. Supported formats: PDF, PNG, JPEG."
        )

    # 2. Check declared MIME type
    normalized_mime = content_type.lower().split(";")[0].strip()
    if normalized_mime not in ALLOWED_MIME_TYPES:
        raise FileValidationError(
            f"Unsupported MIME type '{normalized_mime}'. Allowed: {', '.join(ALLOWED_MIME_TYPES.keys())}."
        )

    # 3. Check actual file signature (magic bytes)
    detected_mime = detect_magic_signature(file_bytes)
    if not detected_mime:
        raise FileValidationError(
            "File header signature could not be verified or is not a valid PDF, PNG, or JPEG."
        )

    if detected_mime != normalized_mime:
        raise FileValidationError(
            f"File signature mismatch: header identifies as '{detected_mime}' but declared MIME is '{normalized_mime}'."
        )

    # 4. Ensure extension matches detected MIME
    allowed_exts_for_detected = ALLOWED_MIME_TYPES[detected_mime]
    if ext not in allowed_exts_for_detected:
        raise FileValidationError(
            f"File extension '{ext}' does not match detected format '{detected_mime}'."
        )

    return detected_mime, ext

