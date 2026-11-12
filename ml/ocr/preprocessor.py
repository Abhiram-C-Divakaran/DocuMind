import cv2
import numpy as np


def detect_and_correct_orientation(gray: np.ndarray) -> np.ndarray:
    """
    Detects severe skew and corrects orientation using minimum-area bounding box
    of dominant text contours.
    """
    try:
        # Threshold image to find text regions
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 100:
            return gray

        # Compute minimum area rectangle
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle
        else:
            angle = -angle

        # Only correct moderate deskew angles between 0.5 and 25 degrees
        if 0.5 < abs(angle) < 25.0:
            (h, w) = gray.shape[:2]
            center = (w // 2, h // 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                gray, rot_mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )
            return rotated
    except Exception:
        pass
    return gray


def preprocess_image_for_ocr(
    image: np.ndarray,
    enhance_contrast: bool = True,
    apply_threshold: bool = True,
) -> np.ndarray:
    """
    Prepares a document image for OCR:
    1. Grayscale conversion
    2. Orientation deskew
    3. Contrast limited adaptive histogram equalization (CLAHE)
    4. Gaussian denoising
    5. Adaptive thresholding
    """
    # 1. Grayscale conversion
    if len(image.shape) == 3:
        # Handle both RGB (from PyMuPDF) and BGR (from cv2.imread)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 2. Orientation deskew
    deskewed = detect_and_correct_orientation(gray)

    # 3. Contrast enhancement using CLAHE
    if enhance_contrast:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(deskewed)
    else:
        enhanced = deskewed

    # 4. Light Gaussian blur for noise suppression
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # 5. Adaptive thresholding / Otsu binarization
    if apply_threshold:
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            25,
            11,
        )
        return binary

    return blurred
