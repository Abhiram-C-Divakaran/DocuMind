from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExtractionOutput:
    extraction_method: str
    is_placeholder: bool
    structured_data: Optional[Dict[str, Any]] = None
    raw_text: Optional[str] = None
    confidence: Optional[float] = None
    field_confidences: Optional[Dict[str, float]] = None
    validation_errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "extraction_method": self.extraction_method,
            "is_placeholder": self.is_placeholder,
            "structured_data": self.structured_data,
            "raw_text": self.raw_text,
            "confidence": self.confidence,
            "field_confidences": self.field_confidences,
            "validation_errors": self.validation_errors,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class BaseExtractor(ABC):
    """Abstract interface for document information extraction pipelines."""

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Returns the identifier of the extraction method."""
        pass

    @abstractmethod
    async def extract(self, file_path: Path, **kwargs: Any) -> ExtractionOutput:
        """
        Extracts structured document entities and text from the given document file.
        """
        pass
