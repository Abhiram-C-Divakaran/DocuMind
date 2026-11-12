from pathlib import Path
from typing import Any
from ml.base import BaseExtractor, ExtractionOutput


class LayoutLMv3Extractor(BaseExtractor):
    """
    Multimodal LayoutLMv3 transformer extraction interface.
    Scheduled for fine-tuning on SROIE benchmark in Phase 2B.
    Unavailable for inference in Phase 2A.
    """

    @property
    def method_name(self) -> str:
        return "LAYOUTLMV3"

    async def extract(self, file_path: Path, **kwargs: Any) -> ExtractionOutput:
        raise NotImplementedError(
            "LayoutLMv3 multimodal extraction is unavailable in Phase 2A. "
            "Model fine-tuning and weight checkpoints are scheduled for Phase 2B. "
            "Please use extraction_method='TESSERACT_BASELINE'."
        )
