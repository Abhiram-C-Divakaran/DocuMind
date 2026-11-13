import enum
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Index, String, Text, func
from sqlalchemy.orm import relationship
from backend.app.database import Base


class ExtractionMethod(str, enum.Enum):
    PLACEHOLDER = "PLACEHOLDER"
    TESSERACT_BASELINE = "TESSERACT_BASELINE"
    LAYOUTLMV3 = "LAYOUTLMV3"


class ExtractionResult(Base):
    __tablename__ = "extraction_results"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    job_id = Column(String(36), ForeignKey("processing_jobs.id", ondelete="SET NULL"), nullable=True, unique=True, index=True)
    extraction_method = Column(String(50), nullable=False, default=ExtractionMethod.TESSERACT_BASELINE.value)
    structured_data = Column(JSON, nullable=True)
    confidence = Column(Float, nullable=True)
    field_confidences = Column(JSON, nullable=True)
    raw_text = Column(Text, nullable=True)
    validation_errors = Column(JSON, nullable=False, default=list)
    extraction_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

    document = relationship("Document", back_populates="extractions")

    def __repr__(self) -> str:
        return f"<ExtractionResult(id={self.id}, doc_id={self.document_id}, method={self.extraction_method})>"
