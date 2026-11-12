from pathlib import Path
from typing import Any
from ml.base import BaseExtractor, ExtractionOutput


class LayoutLMv3Extractor(BaseExtractor):
    """
    Multimodal LayoutLMv3 transformer extraction interface.
    Accepts 2D bounding boxes, token sequences, and document images for token classification.
    Phase 1 establishes the typed contract and validation boundaries without downloading multi-GB weights.
    """

    @property
    def method_name(self) -> str:
        return "LAYOUTLMV3"

    async def extract(self, file_path: Path, **kwargs: Any) -> ExtractionOutput:
        return ExtractionOutput(
            extraction_method=self.method_name,
            is_placeholder=True,
            structured_data=None,
            raw_text=None,
            confidence=None,
            validation_errors=[
                "LayoutLMv3 transformer pipeline interface registered. Model fine-tuning on SROIE dataset "
                "and weights loading are scheduled for Phase 2."
            ],
            metadata={
                "file_name": file_path.name,
                "model_architecture": "microsoft/layoutlmv3-base",
                "phase": 1,
                "token_classes": [
                    "B-VENDOR", "I-VENDOR",
                    "B-DATE", "I-DATE",
                    "B-TOTAL", "I-TOTAL",
                    "B-ADDRESS", "I-ADDRESS"
                ],
            }
        )

