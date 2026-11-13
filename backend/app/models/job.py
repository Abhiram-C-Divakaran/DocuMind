import enum
import uuid
from sqlalchemy import JSON, Boolean, Column, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import relationship
from backend.app.database import Base


class JobStatus(str, enum.Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    RETRYING = "RETRYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    task_id = Column(String(64), nullable=True, index=True)
    idempotency_key = Column(String(128), nullable=False, unique=True, index=True)
    extraction_method = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False, default=JobStatus.QUEUED.value, index=True)
    attempts = Column(Integer, nullable=False, default=0)
    max_attempts = Column(Integer, nullable=False, default=3)
    request_review = Column(Boolean, nullable=False, default=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    heartbeat_at = Column(DateTime(timezone=True), nullable=True, index=True)
    error_code = Column(String(64), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    document = relationship("Document", back_populates="processing_jobs")

    def __repr__(self) -> str:
        return f"<ProcessingJob(id={self.id}, document_id={self.document_id}, status={self.status})>"


class OutboxEvent(Base):
    __tablename__ = "outbox_events"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    aggregate_type = Column(String(50), nullable=False, default="processing_job")
    aggregate_id = Column(String(36), nullable=False, index=True)
    event_type = Column(String(64), nullable=False, default="DOCUMENT_PROCESS_REQUESTED")
    payload = Column(JSON, nullable=False)
    published_at = Column(DateTime(timezone=True), nullable=True, index=True)
    publish_attempts = Column(Integer, nullable=False, default=0)
    last_error = Column(String(512), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

    def __repr__(self) -> str:
        return f"<OutboxEvent(id={self.id}, aggregate_id={self.aggregate_id}, published={self.published_at is not None})>"

