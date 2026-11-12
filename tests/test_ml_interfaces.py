from pathlib import Path
import pytest
from ml.extractors import LayoutLMv3Extractor, PlaceholderExtractor, TesseractBaselineExtractor
from ml.ocr.engine import get_tesseract_status


@pytest.mark.asyncio
async def test_placeholder_extractor_contract():
    extractor = PlaceholderExtractor()
    assert extractor.method_name == "PLACEHOLDER"
    
    result = await extractor.extract(Path("dummy_receipt.pdf"))
    assert result.is_placeholder is True
    assert result.extraction_method == "PLACEHOLDER"
    assert result.confidence is None
    assert len(result.validation_errors) > 0
    assert "Phase 1 placeholder" in result.validation_errors[0]
    assert result.metadata["phase"] == 1


@pytest.mark.asyncio
async def test_tesseract_extractor_contract():
    extractor = TesseractBaselineExtractor()
    assert extractor.method_name == "TESSERACT_BASELINE"

    status = extractor.check_system_dependencies()
    assert "tesseract_installed" in status
    assert "status_message" in status


@pytest.mark.asyncio
async def test_layoutlmv3_extractor_contract():
    extractor = LayoutLMv3Extractor()
    assert extractor.method_name == "LAYOUTLMV3"

    with pytest.raises(NotImplementedError) as exc:
        await extractor.extract(Path("dummy_invoice.pdf"))
    assert "unavailable in Phase 2A" in str(exc.value)
