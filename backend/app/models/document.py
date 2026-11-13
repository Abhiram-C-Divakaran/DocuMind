import enum
import uuid
from datetime import datetime
from typing import List, Optional
from sqlalchemy import JSON, BigInteger, Boolean, Column, DateTime, Enum, ForeignKey, Index, String, Text, func
from sqlalchemy.orm import relationship
from backend.app.database import Base


class DocumentStatus(str, enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    # Nullable for legacy pre-Phase-4A records; legacy unowned docs remain ADMIN-accessible rather than auto-assigned.
    owner_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    original_filename = Column(String(255), nullable=False)
    stored_filename = Column(String(255), nullable=False, unique=True, index=True)
    file_path = Column(String(512), nullable=False)
    file_size_bytes = Column(BigInteger, nullable=False)
    content_type = Column(String(100), nullable=False)
    status = Column(
        String(20),
        default=DocumentStatus.PENDING.value,
        nullable=False,
        index=True
    )
    review_status = Column(
        String(30),
        default="NOT_REQUIRED",
        nullable=False,
        index=True,
    )
    needs_review = Column(Boolean, default=False, nullable=False, index=True)
    review_reasons = Column(JSON, nullable=False, default=list)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    owner = relationship("User", back_populates="documents")
    extractions = relationship(
        "ExtractionResult",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="desc(ExtractionResult.created_at)"
    )
    review = relationship(
        "DocumentReview",
        back_populates="document",
        uselist=False,
        cascade="all, delete-orphan",
    )
    processing_jobs = relationship(
        "ProcessingJob",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="desc(ProcessingJob.created_at)",
    )

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, filename={self.original_filename}, status={self.status})>"

