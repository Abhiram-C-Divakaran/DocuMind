import io
import shutil
from datetime import date
from decimal import Decimal
from pathlib import Path
import numpy as np
import pytest
import pymupdf
from PIL import Image, ImageDraw, ImageFont

from backend.app.services.document_service import document_service, get_extractor
from backend.app.core.exceptions import FileValidationError
from backend.app.models.document import DocumentStatus
from ml.extraction.date import DateExtractor
from ml.extraction.financial import FinancialExtractor, clean_monetary_value
from ml.extraction.pipeline import pipeline
from ml.extraction.vendor import VendorExtractor
from ml.ocr.engine import (
    OCRPage,
    OCRResult,
    OCRToken,
    execute_tesseract_ocr,
    get_tesseract_status,
    normalize_coordinates,
)
from ml.ocr.pdf_converter import convert_pdf_to_images
from ml.ocr.preprocessor import preprocess_image_for_ocr


def create_synthetic_receipt_image(
    vendor: str = "SUPERMARKET DELUXE",
    address: str = "456 Market St, Suite 100, New York, NY 10001",
    doc_date: str = "Date: 15/10/2024",
    total: str = "TOTAL: $ 125.50",
    tax: str = "TAX: $ 10.50",
) -> bytes:
    """Creates a synthetic receipt image in memory using Pillow."""
    img = Image.new("RGB", (600, 800), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    lines = [
        (vendor, 20),
        ("TAX INVOICE", 55),
        (address, 85),
        ("Tel: +1 (555) 019-2834", 110),
        ("-" * 50, 130),
        (doc_date, 150),
        ("-" * 50, 175),
        ("ITEM 1                          $ 50.00", 200),
        ("ITEM 2                          $ 65.00", 230),
        ("-" * 50, 260),
        ("SUBTOTAL                        $ 115.00", 280),
        (f"{tax}                        $ 10.50", 305),
        (total, 335),
        ("-" * 50, 365),
        ("THANK YOU FOR SHOPPING WITH US!", 400),
    ]

    for text, y in lines:
        draw.text((30, y), text, fill=(0, 0, 0))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# 1. Vendor Extraction Unit Tests
def test_vendor_extraction_prominence():
    text = (
        "MEGA MART INC\n"
        "RECEIPT #98234\n"
        "123 Main Road, Bangalore 560001\n"
        "Date: 12/04/2024\n"
        "Total: Rs 450.00\n"
    )
    res = OCRResult(full_text=text, pages=[], tokens=[], avg_confidence=0.9, total_pages=1, processed_pages=1)
    field = VendorExtractor().extract(res)
    assert field.value == "MEGA MART INC"
    assert field.confidence > 0.6


def test_vendor_extraction_skips_common_headers():
    text = (
        "TAX INVOICE\n"
        "WELCOME\n"
        "APEX COFFEE ROASTERS\n"
        "42 Park Lane, London\n"
        "Total: £12.50\n"
    )
    res = OCRResult(full_text=text, pages=[], tokens=[], avg_confidence=0.9, total_pages=1, processed_pages=1)
    field = VendorExtractor().extract(res)
    assert field.value == "APEX COFFEE ROASTERS"


# 2. Date Formats & Ambiguity Unit Tests
def test_date_extraction_various_formats():
    extractor_dmy = DateExtractor(default_order="DMY")

    # DD/MM/YYYY
    res1 = OCRResult(full_text="DATED: 25/12/2023 Total: $50", pages=[], tokens=[], avg_confidence=0.9, total_pages=1, processed_pages=1)
    assert extractor_dmy.extract(res1).value == date(2023, 12, 25)

    # YYYY-MM-DD
    res2 = OCRResult(full_text="Invoice Date: 2024-03-18", pages=[], tokens=[], avg_confidence=0.9, total_pages=1, processed_pages=1)
    assert extractor_dmy.extract(res2).value == date(2024, 3, 18)

    # Textual Date: 15 Oct 2024
    res3 = OCRResult(full_text="Date: 15 Oct 2024", pages=[], tokens=[], avg_confidence=0.9, total_pages=1, processed_pages=1)
    assert extractor_dmy.extract(res3).value == date(2024, 10, 15)


def test_ambiguous_date_locale_preference():
    # 05/06/2024: May 6th in MDY, June 5th in DMY
    res = OCRResult(full_text="Transaction Date: 05/06/2024", pages=[], tokens=[], avg_confidence=0.9, total_pages=1, processed_pages=1)

    dmy_extractor = DateExtractor(default_order="DMY")
    f_dmy = dmy_extractor.extract(res)
    assert f_dmy.value == date(2024, 6, 5)
    assert f_dmy.metadata["is_ambiguous"] is True

    mdy_extractor = DateExtractor(default_order="MDY")
    f_mdy = mdy_extractor.extract(res)
    assert f_mdy.value == date(2024, 5, 6)
    assert f_mdy.metadata["is_ambiguous"] is True


# 3. Total & Tax Selection Unit Tests
def test_financial_extraction_totals_and_taxes():
    text = (
        "GROCERY DEPOT\n"
        "ITEM A  $ 20.00\n"
        "ITEM B  $ 30.00\n"
        "SUBTOTAL  $ 50.00\n"
        "TAX  $ 5.00\n"
        "GRAND TOTAL  $ 55.00\n"
        "CASH  $ 60.00\n"
        "CHANGE  $ 5.00\n"
    )
    res = OCRResult(full_text=text, pages=[], tokens=[], avg_confidence=0.9, total_pages=1, processed_pages=1)
    fe = FinancialExtractor()
    total_field, tax_field = fe.extract_totals_and_taxes(res)

    assert total_field.value == Decimal("55.00")
    assert tax_field.value == Decimal("5.00")


def test_clean_monetary_value():
    assert clean_monetary_value("$1,250.50") == Decimal("1250.50")
    assert clean_monetary_value("€ 99.99") == Decimal("99.99")
    assert clean_monetary_value("₹ 1500") == Decimal("1500.00")
    assert clean_monetary_value("invalid") is None


# 4. Currency Normalization Unit Tests
def test_currency_normalization():
    fe = FinancialExtractor()
    assert fe.extract_currency("TOTAL: $ 50.00").value == "USD"
    assert fe.extract_currency("AMOUNT PAID: ₹ 1,200.00").value == "INR"
    assert fe.extract_currency("NET TOTAL: € 45.00").value == "EUR"
    assert fe.extract_currency("BALANCE: £ 25.00").value == "GBP"


# 5. Missing Fields & Uncertain Values
def test_missing_fields_return_none():
    res = OCRResult(full_text="Random text without dates or prices", pages=[], tokens=[], avg_confidence=0.5, total_pages=1, processed_pages=1)
    result = pipeline.process(res)
    assert result.total_amount.value is None
    assert result.document_date.value is None
    assert result.needs_manual_review is True
    assert any("Total amount was not detected" in e for e in result.validation_errors)


# 6. Bounding Box Normalization
def test_bounding_box_normalization():
    norm = normalize_coordinates(x=50, y=100, w=100, h=50, img_w=1000, img_h=2000)
    assert norm == (50, 50, 150, 75)

    # Edge clamping
    clamped = normalize_coordinates(x=-10, y=-5, w=2000, h=3000, img_w=1000, img_h=1000)
    assert clamped[0] >= 0 and clamped[1] >= 0
    assert clamped[2] <= 1000 and clamped[3] <= 1000


# 7. PDF to Image Conversion
def test_pdf_to_images_conversion(tmp_path: Path):
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=400)
    page.insert_text((50, 50), "Hello DocuMind Test")
    pdf_path = tmp_path / "test_doc.pdf"
    doc.save(str(pdf_path))
    doc.close()

    images, total_pages = convert_pdf_to_images(pdf_path, dpi=150, max_pages=2)
    assert total_pages == 1
    assert len(images) == 1
    assert isinstance(images[0], np.ndarray)
    assert images[0].shape[0] > 0 and images[0].shape[1] > 0


# 8. Image Preprocessor
def test_image_preprocessor():
    raw_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    processed = preprocess_image_for_ocr(raw_bgr)
    assert len(processed.shape) == 2  # Grayscale/binary 2D array
    assert processed.shape == (100, 100)


# 9. Unsupported Extraction Method Rejection
@pytest.mark.asyncio
async def test_unsupported_extraction_method_rejection():
    with pytest.raises(FileValidationError) as exc:
        get_extractor("NON_EXISTENT_METHOD")
    assert "Unsupported extraction method" in str(exc.value.detail)

    with pytest.raises(FileValidationError) as exc_lm:
        get_extractor("LAYOUTLMV3")
    assert "unavailable in Phase 2A" in str(exc_lm.value.detail)


# 10. Tesseract System Dependency Status Check
def test_tesseract_dependency_status():
    is_avail, path, msg = get_tesseract_status()
    assert isinstance(is_avail, bool)
    assert isinstance(msg, str)
    if not is_avail:
        assert "Installation instructions" in msg

# 11. Synthetic Receipt Image End-to-End Extraction Test
def test_synthetic_receipt_extraction_pipeline():
    # Simulate realistic OCR text from a synthetic receipt
    ocr_text = (
        "DELUXE DINER & CAFE\n"
        "123 BAKER STREET, LONDON NW1 6XE\n"
        "TEL: 020-7946-0920\n"
        "--------------------------------\n"
        "DATE: 18/11/2024  TIME: 14:32\n"
        "--------------------------------\n"
        "COFFEE LATTE         £  3.50\n"
        "BLUEBERRY MUFFIN     £  4.00\n"
        "CHEESE SANDWICH      £  6.50\n"
        "--------------------------------\n"
        "SUBTOTAL             £ 14.00\n"
        "VAT (20%)            £  2.80\n"
        "TOTAL DUE            £ 16.80\n"
        "--------------------------------\n"
        "CARD PAYMENT         £ 16.80\n"
    )
    tokens = [
        OCRToken("DELUXE", 0.95, (10, 10, 60, 25), (16, 12, 100, 31), 1, 1, 1),
        OCRToken("DINER", 0.95, (70, 10, 120, 25), (116, 12, 200, 31), 1, 1, 1),
        OCRToken("TOTAL", 0.96, (10, 200, 50, 215), (16, 250, 83, 268), 1, 8, 1),
        OCRToken("DUE", 0.95, (60, 200, 90, 215), (100, 250, 150, 268), 1, 8, 1),
        OCRToken("£", 0.90, (150, 200, 160, 215), (250, 250, 266, 268), 1, 8, 1),
        OCRToken("16.80", 0.97, (170, 200, 210, 215), (283, 250, 350, 268), 1, 8, 1),
    ]
    ocr_result = OCRResult(
        full_text=ocr_text,
        pages=[OCRPage(1, 600, 800, ocr_text, tokens)],
        tokens=tokens,
        avg_confidence=0.94,
        total_pages=1,
        processed_pages=1,
    )

    extracted = pipeline.process(ocr_result)
    assert extracted.vendor_name.value == "DELUXE DINER & CAFE"
    assert "BAKER STREET" in extracted.vendor_address.value
    assert extracted.document_date.value == date(2024, 11, 18)
    assert extracted.total_amount.value == Decimal("16.80")
    assert extracted.currency.value == "GBP"
    assert extracted.tax_amount.value == Decimal("2.80")
    assert extracted.overall_confidence > 0.8
    assert extracted.needs_manual_review is False
    assert len(extracted.validation_errors) == 0


def test_tesseract_ocr_missing_binary_actionable_error(tmp_path: Path, monkeypatch):
    # Ensure test exercises missing binary handling reliably
    from ml.ocr import engine
    monkeypatch.setattr(engine, "locate_tesseract_binary", lambda: None)

    dummy_file = tmp_path / "dummy.png"
    dummy_file.write_bytes(b"\x89PNG\r\n\x1a\n")

    with pytest.raises(RuntimeError) as exc_info:
        execute_tesseract_ocr(dummy_file)

    msg = str(exc_info.value)
    assert "Tesseract executable is not installed" in msg
    assert "Installation instructions" in msg
