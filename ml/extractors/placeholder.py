from pathlib import Path
from typing import Any
from ml.base import BaseExtractor, ExtractionOutput


class PlaceholderExtractor(BaseExtractor):
    """
    Explicit Phase 1 placeholder extractor.
    Returns honest placeholder indicators without fabricating model metrics or extracted contents.
    """

    @property
    def method_name(self) -> str:
        return "PLACEHOLDER"

    async def extract(self, file_path: Path, **kwargs: Any) -> ExtractionOutput:
        return ExtractionOutput(
            extraction_method=self.method_name,
            is_placeholder=True,
            structured_data=None,
            raw_text=None,
            confidence=None,
            validation_errors=[
                "Extraction pipeline is in Phase 1 placeholder mode. "
                "Tesseract baseline or fine-tuned LayoutLMv3 pipeline will be activated in Phase 2."
            ],
            metadata={
                "file_name": file_path.name,
                "phase": 1,
                "status": "NOT_EXECUTED",
                "notes": "No fake inference performed."
            }
        )

