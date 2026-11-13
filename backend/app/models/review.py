import enum
import uuid
from datetime import datetime, timezone
from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import relationship
from backend.app.database import Base


class ReviewStatus(str, enum.Enum):
    NOT_REQUIRED = "NOT_REQUIRED"
    PENDING_REVIEW = "PENDING_REVIEW"
    IN_REVIEW = "IN_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class DocumentReview(Base):
    __tablename__ = "document_reviews"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    document_id = Column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    extraction_result_id = Column(
        String(36),
        ForeignKey("extraction_results.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    review_status = Column(
        String(30),
        default=ReviewStatus.PENDING_REVIEW.value,
        nullable=False,
        index=True,
    )
    corrected_data = Column(JSON, nullable=True)
    changed_fields = Column(JSON, nullable=False, default=list)
    reviewer_identifier = Column(String(120), nullable=False, default="local-anonymous")
    reviewer_notes = Column(Text, nullable=True)
    version = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    approved_at = Column(DateTime(timezone=True), nullable=True)

    document = relationship("Document", back_populates="review")
    extraction_result = relationship("ExtractionResult")
    field_corrections = relationship(
        "FieldCorrection",
        back_populates="review",
        cascade="all, delete-orphan",
        order_by="desc(FieldCorrection.created_at)",
    )
    audit_events = relationship(
        "ReviewAuditEvent",
        back_populates="review",
        cascade="all, delete-orphan",
        order_by="desc(ReviewAuditEvent.event_timestamp)",
    )

    def __repr__(self) -> str:
        return f"<DocumentReview(id={self.id}, doc_id={self.document_id}, status={self.review_status}, v={self.version})>"


class FieldCorrection(Base):
    __tablename__ = "field_corrections"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    review_id = Column(
        String(36),
        ForeignKey("document_reviews.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    field_name = Column(String(64), nullable=False, index=True)
    original_value = Column(Text, nullable=True)
    corrected_value = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )

    review = relationship("DocumentReview", back_populates="field_corrections")

    def __repr__(self) -> str:
        return f"<FieldCorrection(id={self.id}, review_id={self.review_id}, field={self.field_name})>"


class ReviewAuditEvent(Base):
    __tablename__ = "review_audit_events"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    review_id = Column(
        String(36),
        ForeignKey("document_reviews.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    action = Column(String(50), nullable=False, index=True)
    from_status = Column(String(30), nullable=True)
    to_status = Column(String(30), nullable=True)
    changed_fields = Column(JSON, nullable=False, default=list)
    actor_identifier = Column(String(120), nullable=False, default="local-anonymous")
    notes = Column(Text, nullable=True)
    event_timestamp = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    review = relationship("DocumentReview", back_populates="audit_events")

    def __repr__(self) -> str:
        return f"<ReviewAuditEvent(id={self.id}, review_id={self.review_id}, action={self.action})>"

