import json
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Set, Tuple
from sqlalchemy import desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.config import settings
from backend.app.core.exceptions import ConflictError, DocumentNotFoundError, FileValidationError, PermissionDeniedError
from backend.app.logger import logger
from backend.app.models.document import Document, DocumentStatus
from backend.app.models.extraction import ExtractionResult
from backend.app.models.review import (
    DocumentReview,
    FieldCorrection,
    ReviewAuditEvent,
    ReviewStatus,
)
from backend.app.models.user import User, UserRole
from backend.app.schemas.review import (
    DocumentReviewResponse,
    FieldCorrectionResponse,
    ReviewAuditEventResponse,
    ReviewCorrectionPayload,
    ReviewDecisionRequest,
    ReviewHistoryResponse,
    ReviewListResponse,
    ReviewMetricsResponse,
    ReviewQueueItem,
    ReviewUpdateRequest,
)

TRACKED_FIELDS = [
    "vendor_name",
    "vendor_address",
    "document_date",
    "total_amount",
    "currency",
    "tax_amount",
    "line_items",
]

CRITICAL_FIELDS = ["vendor_name", "document_date", "total_amount"]

ALLOWED_TRANSITIONS: Dict[str, Set[str]] = {
    ReviewStatus.NOT_REQUIRED.value: {
        ReviewStatus.NOT_REQUIRED.value,
        ReviewStatus.PENDING_REVIEW.value,
        ReviewStatus.IN_REVIEW.value,
        ReviewStatus.APPROVED.value,
        ReviewStatus.REJECTED.value,
    },
    ReviewStatus.PENDING_REVIEW.value: {
        ReviewStatus.IN_REVIEW.value,
        ReviewStatus.APPROVED.value,
        ReviewStatus.REJECTED.value,
    },
    ReviewStatus.IN_REVIEW.value: {
        ReviewStatus.IN_REVIEW.value,
        ReviewStatus.APPROVED.value,
        ReviewStatus.REJECTED.value,
    },
    ReviewStatus.APPROVED.value: {
        ReviewStatus.IN_REVIEW.value,
    },
    ReviewStatus.REJECTED.value: {
        ReviewStatus.IN_REVIEW.value,
        ReviewStatus.PENDING_REVIEW.value,
    },
}


def _normalize_comparable(field_name: str, value: Any) -> Any:
    if field_name == "line_items":
        return value if value else []
    if value is None:
        return None
    if field_name in ("total_amount", "tax_amount"):
        s = str(value).strip()
        if not s:
            return None
        try:
            return str(Decimal(s).quantize(Decimal("0.01")))
        except (InvalidOperation, ValueError):
            return s
    s = str(value).strip()
    return s if s else None


def _serialize_field_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def compute_missing_critical_fields(structured_data: Optional[Dict[str, Any]]) -> List[str]:
    if not structured_data:
        return list(CRITICAL_FIELDS)
    missing: List[str] = []
    for f in CRITICAL_FIELDS:
        val = structured_data.get(f)
        if val is None or (isinstance(val, str) and not val.strip()):
            missing.append(f)
    return missing


def compute_changed_fields(
    original_data: Optional[Dict[str, Any]],
    corrected_data: Optional[Dict[str, Any]],
) -> List[str]:
    if not corrected_data:
        return []
    orig = original_data or {}
    changed: List[str] = []
    for field_name in TRACKED_FIELDS:
        ov = _normalize_comparable(field_name, orig.get(field_name))
        cv = _normalize_comparable(field_name, corrected_data.get(field_name))
        if ov != cv:
            changed.append(field_name)
    return changed


def evaluate_review_eligibility(
    extraction: Optional[ExtractionResult],
    document_status: str,
    explicit_request: bool = False,
    confidence_threshold: Optional[float] = None,
) -> Tuple[bool, List[str], List[str]]:
    """
    Determines whether an extracted document should enter the review queue
    using heuristic confidence, missing critical fields, financial validation errors,
    ambiguous dates, conflicting signals, or explicit user request.
    """
    threshold = (
        confidence_threshold
        if confidence_threshold is not None
        else settings.REVIEW_CONFIDENCE_THRESHOLD
    )
    reasons: List[str] = []

    if document_status == DocumentStatus.FAILED.value:
        reasons.append("extraction_failed")
        return True, reasons, list(CRITICAL_FIELDS)

    if document_status == DocumentStatus.PENDING.value:
        if explicit_request:
            reasons.append("user_requested_review")
        return bool(reasons), reasons, []

    structured = extraction.structured_data if extraction else None
    missing_critical = compute_missing_critical_fields(structured)

    if extraction is None:
        reasons.append("missing_extraction_record")
        return True, reasons, missing_critical

    if extraction.confidence is None or extraction.confidence < threshold:
        reasons.append(f"low_heuristic_confidence (<{threshold:.2f})")

    for mf in missing_critical:
        reasons.append(f"missing_critical_field:{mf}")

    val_errors = extraction.validation_errors or []
    lower_errors = " | ".join(val_errors).lower()

    if any(
        kw in lower_errors
        for kw in ("inconsistent financial", "invalid total", "invalid tax", "cannot be negative")
    ):
        reasons.append("financial_validation_failed")

    meta = extraction.extraction_metadata or {}
    if "ambiguous" in lower_errors or meta.get("date_ambiguous"):
        reasons.append("ambiguous_date_parsing")

    if "conflict" in lower_errors or meta.get("conflicting_predictions"):
        reasons.append("conflicting_predictions")

    if explicit_request:
        reasons.append("user_requested_review")

    # Deduplicate preserving order
    seen: Set[str] = set()
    unique_reasons: List[str] = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            unique_reasons.append(r)

    return len(unique_reasons) > 0, unique_reasons, missing_critical


class ReviewService:
    async def initialize_review_after_extraction(
        self,
        db: AsyncSession,
        document: Document,
        extraction: Optional[ExtractionResult],
        explicit_request: bool = False,
    ) -> DocumentReview:
        """
        Evaluates eligibility right after upload/extraction and creates the initial
        DocumentReview + audit trail entry without altering original predictions.
        """
        needs_review, reasons, _ = evaluate_review_eligibility(
            extraction=extraction,
            document_status=document.status,
            explicit_request=explicit_request,
        )
        initial_review_status = (
            ReviewStatus.PENDING_REVIEW.value
            if needs_review
            else ReviewStatus.NOT_REQUIRED.value
        )

        document.needs_review = needs_review
        document.review_reasons = reasons
        document.review_status = initial_review_status

        review = DocumentReview(
            document_id=document.id,
            extraction_result_id=extraction.id if extraction else None,
            review_status=initial_review_status,
            corrected_data=None,
            changed_fields=[],
            reviewer_identifier=settings.DEFAULT_REVIEWER_ID,
            version=1,
        )
        db.add(review)
        await db.flush()

        event = ReviewAuditEvent(
            review_id=review.id,
            action="QUEUE_FLAGGED" if needs_review else "REVIEW_INITIALIZED",
            from_status=None,
            to_status=initial_review_status,
            changed_fields=[],
            actor_identifier=settings.DEFAULT_REVIEWER_ID,
            notes=(
                f"Auto-queued for review: {', '.join(reasons)}"
                if needs_review
                else "Extraction completed above confidence threshold; voluntary review remains available."
            ),
        )
        db.add(event)
        await db.commit()
        await db.refresh(review)
        return review

    async def _load_document_with_review(
        self, db: AsyncSession, document_id: str
    ) -> Tuple[Document, Optional[ExtractionResult], Optional[DocumentReview]]:
        query = (
            select(Document)
            .where(Document.id == document_id)
            .options(
                selectinload(Document.extractions),
                selectinload(Document.review).selectinload(DocumentReview.field_corrections),
                selectinload(Document.review).selectinload(DocumentReview.audit_events),
            )
        )
        result = await db.execute(query)
        document = result.scalar_one_or_none()
        if not document:
            raise DocumentNotFoundError(document_id=document_id)

        latest_extraction = document.extractions[0] if document.extractions else None
        return document, latest_extraction, document.review

    async def _ensure_review_record(
        self, db: AsyncSession, document_id: str, user: Optional[User] = None
    ) -> Tuple[Document, Optional[ExtractionResult], DocumentReview]:
        document, latest_extraction, review = await self._load_document_with_review(db, document_id)
        if user is not None and user.role != UserRole.ADMIN.value:
            if document.owner_id is None or document.owner_id != user.id:
                raise PermissionDeniedError("You do not have access to this document.")
        if review is not None:
            return document, latest_extraction, review

        # Backfill cleanly for pre-Phase-3A documents already in DB
        needs_review, reasons, _ = evaluate_review_eligibility(
            extraction=latest_extraction,
            document_status=document.status,
            explicit_request=False,
        )
        initial_status = (
            ReviewStatus.PENDING_REVIEW.value
            if needs_review
            else ReviewStatus.NOT_REQUIRED.value
        )
        document.needs_review = needs_review
        document.review_reasons = reasons
        document.review_status = initial_status

        review = DocumentReview(
            document_id=document.id,
            extraction_result_id=latest_extraction.id if latest_extraction else None,
            review_status=initial_status,
            corrected_data=None,
            changed_fields=[],
            reviewer_identifier=settings.DEFAULT_REVIEWER_ID,
            version=1,
        )
        db.add(review)
        await db.flush()
        db.add(
            ReviewAuditEvent(
                review_id=review.id,
                action="REVIEW_CREATED",
                from_status=None,
                to_status=initial_status,
                changed_fields=[],
                actor_identifier=settings.DEFAULT_REVIEWER_ID,
                notes="Initialized review record for existing document.",
            )
        )
        await db.commit()
        _, latest_extraction, review = await self._load_document_with_review(db, document_id)
        return document, latest_extraction, review  # type: ignore[return-value]

    def _build_review_response(
        self,
        document: Document,
        extraction: Optional[ExtractionResult],
        review: DocumentReview,
    ) -> DocumentReviewResponse:
        original_data = extraction.structured_data if extraction else None
        missing_critical = compute_missing_critical_fields(
            review.corrected_data if review.corrected_data is not None else original_data
        )
        return DocumentReviewResponse(
            id=review.id,
            document_id=document.id,
            extraction_result_id=review.extraction_result_id or (extraction.id if extraction else None),
            review_status=review.review_status,
            document_status=document.status,
            extraction_method=extraction.extraction_method if extraction else None,
            confidence=extraction.confidence if extraction else None,
            field_confidences=extraction.field_confidences if extraction else None,
            validation_errors=list(extraction.validation_errors or []) if extraction else [],
            review_reasons=list(document.review_reasons or []),
            missing_critical_fields=missing_critical,
            original_data=original_data,
            corrected_data=review.corrected_data,
            changed_fields=list(review.changed_fields or []),
            raw_text=extraction.raw_text if extraction else None,
            reviewer_identifier=review.reviewer_identifier,
            reviewer_notes=review.reviewer_notes,
            version=review.version,
            created_at=review.created_at,
            updated_at=review.updated_at,
            approved_at=review.approved_at,
        )

    def _validate_transition(self, current_status: str, target_status: str) -> None:
        allowed = ALLOWED_TRANSITIONS.get(current_status, set())
        if target_status not in allowed:
            raise ConflictError(
                f"Invalid review state transition from '{current_status}' to '{target_status}'.",
                error_code="INVALID_STATE_TRANSITION",
                extra={"from_status": current_status, "to_status": target_status},
            )

    def _check_expected_version(self, expected_version: Optional[int], actual_version: int) -> None:
        if expected_version is not None and expected_version != actual_version:
            raise ConflictError(
                f"Review record was modified concurrently (expected version {expected_version}, current version {actual_version}). Refresh before saving.",
                error_code="VERSION_CONFLICT",
                extra={"expected_version": expected_version, "current_version": actual_version},
            )

    async def _atomic_version_bump(
        self,
        db: AsyncSession,
        review: DocumentReview,
        **update_values: Any,
    ) -> int:
        current_version = review.version
        next_version = current_version + 1
        stmt = (
            update(DocumentReview)
            .where(DocumentReview.id == review.id, DocumentReview.version == current_version)
            .values(version=next_version, **update_values)
        )
        res = await db.execute(stmt)
        if res.rowcount != 1:
            await db.rollback()
            raise ConflictError(
                "Concurrent modification detected while updating review record.",
                error_code="VERSION_CONFLICT",
            )
        return next_version

    def _record_field_corrections(
        self,
        db: AsyncSession,
        review_id: str,
        original_data: Optional[Dict[str, Any]],
        previous_corrected: Optional[Dict[str, Any]],
        new_corrected: Dict[str, Any],
    ) -> List[str]:
        """
        Records FieldCorrection entries for every field that changed compared to
        the prior draft (or original extraction if no draft existed yet), preserving
        original_data untouched.
        """
        orig = original_data or {}
        baseline = previous_corrected if previous_corrected is not None else orig
        edited_in_this_step: List[str] = []

        for field_name in TRACKED_FIELDS:
            prev_val = _normalize_comparable(field_name, baseline.get(field_name))
            next_val = _normalize_comparable(field_name, new_corrected.get(field_name))
            if prev_val != next_val:
                edited_in_this_step.append(field_name)
                db.add(
                    FieldCorrection(
                        review_id=review_id,
                        field_name=field_name,
                        original_value=_serialize_field_value(orig.get(field_name)),
                        corrected_value=_serialize_field_value(new_corrected.get(field_name)),
                    )
                )
        return edited_in_this_step

    async def get_document_review(
        self, db: AsyncSession, document_id: str, user: Optional[User] = None
    ) -> DocumentReviewResponse:
        document, extraction, review = await self._ensure_review_record(db, document_id, user=user)
        return self._build_review_response(document, extraction, review)

    async def save_review_draft(
        self,
        db: AsyncSession,
        document_id: str,
        payload: ReviewUpdateRequest,
        user: Optional[User] = None,
    ) -> DocumentReviewResponse:
        document, extraction, review = await self._ensure_review_record(db, document_id, user=user)
        self._check_expected_version(payload.expected_version, review.version)

        prev_status = review.review_status
        target_status = ReviewStatus.IN_REVIEW.value
        if prev_status in (ReviewStatus.APPROVED.value, ReviewStatus.REJECTED.value):
            raise ConflictError(
                f"Cannot modify draft when review status is '{prev_status}'. Reopen review first.",
                error_code="INVALID_STATE_TRANSITION",
                extra={"current_status": prev_status},
            )
        self._validate_transition(prev_status, target_status)

        orig_data = extraction.structured_data if extraction else None
        new_data_dict = payload.corrected_data.to_json_dict()
        changed_from_original = compute_changed_fields(orig_data, new_data_dict)
        edited_now = self._record_field_corrections(
            db,
            review.id,
            orig_data,
            review.corrected_data,
            new_data_dict,
        )

        actor = (payload.reviewer_identifier or "").strip() or (user.email if user else settings.DEFAULT_REVIEWER_ID)
        now = datetime.now(timezone.utc)

        await self._atomic_version_bump(
            db,
            review,
            review_status=target_status,
            corrected_data=new_data_dict,
            changed_fields=changed_from_original,
            reviewer_identifier=actor,
            reviewer_notes=payload.reviewer_notes if payload.reviewer_notes is not None else review.reviewer_notes,
            updated_at=now,
        )
        document.review_status = target_status
        document.updated_at = now

        action_name = "REVIEW_STARTED" if prev_status in (ReviewStatus.NOT_REQUIRED.value, ReviewStatus.PENDING_REVIEW.value) else "DRAFT_SAVED"
        db.add(
            ReviewAuditEvent(
                review_id=review.id,
                action=action_name,
                from_status=prev_status,
                to_status=target_status,
                changed_fields=edited_now or changed_from_original,
                actor_identifier=actor,
                notes=payload.reviewer_notes,
            )
        )
        await db.commit()
        _, extraction, refreshed_review = await self._load_document_with_review(db, document_id)
        return self._build_review_response(document, extraction, refreshed_review)  # type: ignore[arg-type]

    async def approve_review(
        self,
        db: AsyncSession,
        document_id: str,
        payload: ReviewDecisionRequest,
        user: Optional[User] = None,
    ) -> DocumentReviewResponse:
        document, extraction, review = await self._ensure_review_record(db, document_id, user=user)
        self._check_expected_version(payload.expected_version, review.version)

        prev_status = review.review_status
        orig_data = extraction.structured_data if extraction else None

        # Idempotent approval check when already APPROVED
        if prev_status == ReviewStatus.APPROVED.value:
            if payload.corrected_data is not None:
                incoming = payload.corrected_data.to_json_dict()
                existing_eff = review.corrected_data if review.corrected_data is not None else (orig_data or {})
                if compute_changed_fields(existing_eff, incoming):
                    raise ConflictError(
                        "Review is already APPROVED; reopen it before changing approved fields.",
                        error_code="INVALID_STATE_TRANSITION",
                    )
            return self._build_review_response(document, extraction, review)

        self._validate_transition(prev_status, ReviewStatus.APPROVED.value)

        if payload.corrected_data is not None:
            effective_data = payload.corrected_data.to_json_dict()
        elif review.corrected_data is not None:
            # Re-validate persisted draft through strict Pydantic schema before final approval
            validated = ReviewCorrectionPayload.model_validate(review.corrected_data)
            effective_data = validated.to_json_dict()
        elif orig_data is not None:
            validated = ReviewCorrectionPayload.model_validate(orig_data)
            effective_data = validated.to_json_dict()
        else:
            raise FileValidationError(
                "Cannot approve a document that has neither extracted fields nor validated corrected data."
            )

        changed_from_original = compute_changed_fields(orig_data, effective_data)
        edited_now = self._record_field_corrections(
            db,
            review.id,
            orig_data,
            review.corrected_data,
            effective_data,
        )

        actor = (payload.reviewer_identifier or "").strip() or (user.email if user else settings.DEFAULT_REVIEWER_ID)
        now = datetime.now(timezone.utc)

        await self._atomic_version_bump(
            db,
            review,
            review_status=ReviewStatus.APPROVED.value,
            corrected_data=effective_data,
            changed_fields=changed_from_original,
            reviewer_identifier=actor,
            reviewer_notes=payload.reviewer_notes if payload.reviewer_notes is not None else review.reviewer_notes,
            approved_at=now,
            updated_at=now,
        )
        document.review_status = ReviewStatus.APPROVED.value
        document.updated_at = now

        db.add(
            ReviewAuditEvent(
                review_id=review.id,
                action="APPROVED",
                from_status=prev_status,
                to_status=ReviewStatus.APPROVED.value,
                changed_fields=edited_now or changed_from_original,
                actor_identifier=actor,
                notes=payload.reviewer_notes,
            )
        )
        await db.commit()
        _, extraction, refreshed_review = await self._load_document_with_review(db, document_id)
        return self._build_review_response(document, extraction, refreshed_review)  # type: ignore[arg-type]

    async def reject_review(
        self,
        db: AsyncSession,
        document_id: str,
        payload: ReviewDecisionRequest,
        user: Optional[User] = None,
    ) -> DocumentReviewResponse:
        document, extraction, review = await self._ensure_review_record(db, document_id, user=user)
        self._check_expected_version(payload.expected_version, review.version)

        prev_status = review.review_status
        if prev_status == ReviewStatus.REJECTED.value:
            return self._build_review_response(document, extraction, review)

        self._validate_transition(prev_status, ReviewStatus.REJECTED.value)

        orig_data = extraction.structured_data if extraction else None
        new_data_dict = (
            payload.corrected_data.to_json_dict()
            if payload.corrected_data is not None
            else review.corrected_data
        )
        changed_from_original = compute_changed_fields(orig_data, new_data_dict)
        edited_now: List[str] = []
        if new_data_dict is not None:
            edited_now = self._record_field_corrections(
                db,
                review.id,
                orig_data,
                review.corrected_data,
                new_data_dict,
            )

        actor = (payload.reviewer_identifier or "").strip() or (user.email if user else settings.DEFAULT_REVIEWER_ID)
        now = datetime.now(timezone.utc)

        await self._atomic_version_bump(
            db,
            review,
            review_status=ReviewStatus.REJECTED.value,
            corrected_data=new_data_dict,
            changed_fields=changed_from_original,
            reviewer_identifier=actor,
            reviewer_notes=payload.reviewer_notes if payload.reviewer_notes is not None else review.reviewer_notes,
            approved_at=None,
            updated_at=now,
        )
        document.review_status = ReviewStatus.REJECTED.value
        document.updated_at = now

        db.add(
            ReviewAuditEvent(
                review_id=review.id,
                action="REJECTED",
                from_status=prev_status,
                to_status=ReviewStatus.REJECTED.value,
                changed_fields=edited_now or changed_from_original,
                actor_identifier=actor,
                notes=payload.reviewer_notes,
            )
        )
        await db.commit()
        _, extraction, refreshed_review = await self._load_document_with_review(db, document_id)
        return self._build_review_response(document, extraction, refreshed_review)  # type: ignore[arg-type]

    async def reopen_review(
        self,
        db: AsyncSession,
        document_id: str,
        payload: ReviewDecisionRequest,
        user: Optional[User] = None,
    ) -> DocumentReviewResponse:
        document, extraction, review = await self._ensure_review_record(db, document_id, user=user)
        self._check_expected_version(payload.expected_version, review.version)

        prev_status = review.review_status
        self._validate_transition(prev_status, ReviewStatus.IN_REVIEW.value)

        actor = (payload.reviewer_identifier or "").strip() or (user.email if user else settings.DEFAULT_REVIEWER_ID)
        now = datetime.now(timezone.utc)

        await self._atomic_version_bump(
            db,
            review,
            review_status=ReviewStatus.IN_REVIEW.value,
            approved_at=None,
            reviewer_identifier=actor,
            reviewer_notes=payload.reviewer_notes if payload.reviewer_notes is not None else review.reviewer_notes,
            updated_at=now,
        )
        document.review_status = ReviewStatus.IN_REVIEW.value
        document.updated_at = now

        db.add(
            ReviewAuditEvent(
                review_id=review.id,
                action="REOPENED_FOR_REVIEW",
                from_status=prev_status,
                to_status=ReviewStatus.IN_REVIEW.value,
                changed_fields=[],
                actor_identifier=actor,
                notes=payload.reviewer_notes or "Review reopened for further editing.",
            )
        )
        await db.commit()
        _, extraction, refreshed_review = await self._load_document_with_review(db, document_id)
        return self._build_review_response(document, extraction, refreshed_review)  # type: ignore[arg-type]

    async def get_review_history(
        self, db: AsyncSession, document_id: str, user: Optional[User] = None
    ) -> ReviewHistoryResponse:
        document, _, review = await self._ensure_review_record(db, document_id, user=user)
        return ReviewHistoryResponse(
            document_id=document.id,
            review_id=review.id,
            review_status=review.review_status,
            field_corrections=[
                FieldCorrectionResponse.model_validate(fc) for fc in review.field_corrections
            ],
            audit_events=[
                ReviewAuditEventResponse.model_validate(ev) for ev in review.audit_events
            ],
        )

    async def list_reviews_queue(
        self,
        db: AsyncSession,
        page: int = 1,
        page_size: int = 20,
        review_status: Optional[str] = None,
        extraction_method: Optional[str] = None,
        confidence_max: Optional[float] = None,
        missing_field: Optional[str] = None,
        search: Optional[str] = None,
        sort_by: str = "confidence_asc",
        user: Optional[User] = None,
    ) -> ReviewListResponse:
        query = (
            select(Document)
            .options(
                selectinload(Document.extractions),
                selectinload(Document.review),
            )
            .order_by(desc(Document.created_at))
        )
        if user is not None and user.role != UserRole.ADMIN.value:
            query = query.where(Document.owner_id == user.id)
        docs = list((await db.execute(query)).scalars().all())

        status_filter = review_status.upper().strip() if review_status else None
        method_filter = extraction_method.upper().strip() if extraction_method else None
        missing_filter = missing_field.strip() if missing_field else None
        search_lower = search.strip().lower() if search else None

        queue_items: List[ReviewQueueItem] = []
        for doc in docs:
            ext = doc.extractions[0] if doc.extractions else None
            rev = doc.review
            eff_status = rev.review_status if rev else doc.review_status
            ext_method = ext.extraction_method if ext else None
            conf = ext.confidence if ext else None
            effective_data = (
                rev.corrected_data
                if rev and rev.corrected_data is not None
                else (ext.structured_data if ext else None)
            )
            missing_critical = compute_missing_critical_fields(effective_data)

            if status_filter and status_filter != "ALL":
                if status_filter == "NEEDS_REVIEW":
                    if eff_status not in (
                        ReviewStatus.PENDING_REVIEW.value,
                        ReviewStatus.IN_REVIEW.value,
                    ):
                        continue
                elif eff_status != status_filter:
                    continue

            if method_filter and method_filter != "ALL":
                if ext_method != method_filter:
                    continue

            if confidence_max is not None:
                if conf is not None and conf > confidence_max:
                    continue

            if missing_filter:
                if missing_filter not in missing_critical:
                    continue

            if search_lower:
                if search_lower not in doc.original_filename.lower():
                    continue

            queue_items.append(
                ReviewQueueItem(
                    document_id=doc.id,
                    review_id=rev.id if rev else None,
                    original_filename=doc.original_filename,
                    content_type=doc.content_type,
                    file_size_bytes=doc.file_size_bytes,
                    document_status=doc.status,
                    review_status=eff_status,
                    needs_review=bool(doc.needs_review),
                    review_reasons=list(doc.review_reasons or []),
                    missing_critical_fields=missing_critical,
                    extraction_method=ext_method,
                    confidence=conf,
                    validation_errors_count=len(ext.validation_errors or []) if ext else 0,
                    changed_fields=list(rev.changed_fields or []) if rev else [],
                    reviewer_identifier=rev.reviewer_identifier if rev else settings.DEFAULT_REVIEWER_ID,
                    version=rev.version if rev else 0,
                    created_at=doc.created_at,
                    updated_at=rev.updated_at if rev else doc.updated_at,
                    approved_at=rev.approved_at if rev else None,
                )
            )

        if sort_by == "confidence_asc":
            queue_items.sort(
                key=lambda x: (x.confidence is not None, x.confidence if x.confidence is not None else 0.0, -x.created_at.timestamp())
            )
        elif sort_by == "confidence_desc":
            queue_items.sort(
                key=lambda x: (x.confidence is not None, x.confidence if x.confidence is not None else -1.0),
                reverse=True,
            )
        elif sort_by == "created_at_asc":
            queue_items.sort(key=lambda x: x.created_at)
        elif sort_by == "updated_at_desc":
            queue_items.sort(key=lambda x: x.updated_at, reverse=True)
        else:  # created_at_desc
            queue_items.sort(key=lambda x: x.created_at, reverse=True)

        total = len(queue_items)
        start = (page - 1) * page_size
        end = start + page_size
        return ReviewListResponse(
            items=queue_items[start:end],
            total=total,
            page=page,
            page_size=page_size,
        )

    async def get_review_metrics(self, db: AsyncSession, user: Optional[User] = None) -> ReviewMetricsResponse:
        stmt = select(Document).options(selectinload(Document.review), selectinload(Document.extractions))
        if user is not None and user.role != UserRole.ADMIN.value:
            stmt = stmt.where(Document.owner_id == user.id)
        docs_res = await db.execute(stmt)
        docs = list(docs_res.scalars().all())

        counts = {
            ReviewStatus.NOT_REQUIRED.value: 0,
            ReviewStatus.PENDING_REVIEW.value: 0,
            ReviewStatus.IN_REVIEW.value: 0,
            ReviewStatus.APPROVED.value: 0,
            ReviewStatus.REJECTED.value: 0,
        }
        changed_counts: List[int] = []
        eligible_feedback = 0

        for doc in docs:
            rev = doc.review
            st = rev.review_status if rev else doc.review_status
            if st in counts:
                counts[st] += 1

            if rev and (rev.changed_fields or st in (ReviewStatus.APPROVED.value, ReviewStatus.REJECTED.value)):
                changed_counts.append(len(rev.changed_fields or []))

            if st == ReviewStatus.APPROVED.value and rev and rev.corrected_data:
                eligible_feedback += 1

        total = len(docs)
        awaiting = counts[ReviewStatus.PENDING_REVIEW.value] + counts[ReviewStatus.IN_REVIEW.value]
        approved = counts[ReviewStatus.APPROVED.value]
        rejected = counts[ReviewStatus.REJECTED.value]

        review_denominator = awaiting + approved + rejected
        completion_rate = (approved + rejected) / review_denominator if review_denominator > 0 else 0.0
        avg_changed = sum(changed_counts) / len(changed_counts) if changed_counts else 0.0

        return ReviewMetricsResponse(
            total_documents=total,
            not_required=counts[ReviewStatus.NOT_REQUIRED.value],
            pending_review=counts[ReviewStatus.PENDING_REVIEW.value],
            in_review=counts[ReviewStatus.IN_REVIEW.value],
            awaiting_review=awaiting,
            approved=approved,
            rejected=rejected,
            avg_corrected_fields=round(avg_changed, 2),
            review_completion_rate=round(completion_rate, 4),
            eligible_for_feedback_export=eligible_feedback,
            review_confidence_threshold=settings.REVIEW_CONFIDENCE_THRESHOLD,
        )


review_service = ReviewService()

