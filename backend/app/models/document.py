import enum
import uuid
from datetime import datetime
from typing import List, Optional
from sqlalchemy import BigInteger, Column, DateTime, Enum, ForeignKey, Index, String, Text, func
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
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    extractions = relationship(
        "ExtractionResult",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="desc(ExtractionResult.created_at)"
    )

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, filename={self.original_filename}, status={self.status})>"

