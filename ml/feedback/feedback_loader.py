import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

from backend.app.config import settings
from backend.app.models.document import Document
from backend.app.models.review import DocumentReview, ReviewStatus


@dataclass
class FeedbackRecord:
    document_id: str
    extraction_id: Optional[str]
    review_id: str
    original_filename: str
    stored_filename: str
    content_type: str
    extraction_method: Optional[str]
    heuristic_confidence: Optional[float]
    original_predictions: Dict[str, Any]
    corrected_values: Dict[str, Any]
    changed_fields: List[str]
    raw_text: Optional[str]
    extraction_metadata: Dict[str, Any]
    review_status: str
    reviewer_identifier: str
    reviewer_notes: Optional[str]
    approved_at: Optional[str]
    review_version: int
    ocr_tokens: List[Dict[str, Any]] = field(default_factory=list)


async def load_feedback_from_session(
    db: AsyncSession,
) -> Tuple[List[FeedbackRecord], Dict[str, int]]:
    """
    Loads ONLY APPROVED human-reviewed records from the active database session,
    recording exact exclusion counts for REJECTED and unapproved items.
    """
    query = (
        select(DocumentReview)
        .options(
            selectinload(DocumentReview.document),
            selectinload(DocumentReview.extraction_result),
        )
        .order_by(DocumentReview.approved_at.asc(), DocumentReview.created_at.asc())
    )
    rows = list((await db.execute(query)).scalars().all())

    approved_records: List[FeedbackRecord] = []
    excluded_counts: Dict[str, int] = {
        "rejected": 0,
        "pending_review": 0,
        "in_review": 0,
        "not_required": 0,
        "missing_corrected_data": 0,
        "missing_source_document": 0,
    }

    for rev in rows:
        if rev.review_status == ReviewStatus.REJECTED.value:
            excluded_counts["rejected"] += 1
            continue
        if rev.review_status == ReviewStatus.PENDING_REVIEW.value:
            excluded_counts["pending_review"] += 1
            continue
        if rev.review_status == ReviewStatus.IN_REVIEW.value:
            excluded_counts["in_review"] += 1
            continue
        if rev.review_status != ReviewStatus.APPROVED.value:
            excluded_counts["not_required"] += 1
            continue

        if rev.corrected_data is None:
            excluded_counts["missing_corrected_data"] += 1
            continue

        doc: Optional[Document] = rev.document
        if doc is None:
            excluded_counts["missing_source_document"] += 1
            continue

        ext = rev.extraction_result
        meta = dict(ext.extraction_metadata or {}) if ext else {}
        ocr_tokens = meta.get("ocr_tokens") if isinstance(meta.get("ocr_tokens"), list) else []

        approved_records.append(
            FeedbackRecord(
                document_id=doc.id,
                extraction_id=ext.id if ext else None,
                review_id=rev.id,
                original_filename=doc.original_filename,
                stored_filename=doc.stored_filename,
                content_type=doc.content_type,
                extraction_method=ext.extraction_method if ext else None,
                heuristic_confidence=ext.confidence if ext else None,
                original_predictions=dict(ext.structured_data or {}) if ext else {},
                corrected_values=dict(rev.corrected_data),
                changed_fields=list(rev.changed_fields or []),
                raw_text=ext.raw_text if ext else None,
                extraction_metadata=meta,
                review_status=rev.review_status,
                reviewer_identifier=rev.reviewer_identifier,
                reviewer_notes=rev.reviewer_notes,
                approved_at=rev.approved_at.isoformat() if rev.approved_at else None,
                review_version=rev.version,
                ocr_tokens=ocr_tokens,
            )
        )

    return approved_records, excluded_counts


async def _load_via_url(db_url: str) -> Tuple[List[FeedbackRecord], Dict[str, int]]:
    engine = create_async_engine(db_url, future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with session_factory() as db:
            return await load_feedback_from_session(db)
    finally:
        await engine.dispose()


def load_approved_feedback_records(
    db_url: Optional[str] = None,
) -> Tuple[List[FeedbackRecord], Dict[str, int]]:
    """Synchronous entrypoint loading approved feedback records from SQLite/PostgreSQL."""
    target_url = db_url or settings.DATABASE_URL
    return asyncio.run(_load_via_url(target_url))

