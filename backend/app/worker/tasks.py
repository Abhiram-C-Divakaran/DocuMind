import asyncio
import concurrent.futures
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.core.exceptions import FileValidationError
from backend.app.database import get_sync_session
from backend.app.logger import logger
from backend.app.models.document import Document, DocumentStatus
from backend.app.models.extraction import ExtractionMethod, ExtractionResult
from backend.app.models.job import JobStatus, OutboxEvent, ProcessingJob
from backend.app.models.review import DocumentReview, ReviewAuditEvent, ReviewStatus
from backend.app.services.review_service import evaluate_review_eligibility
from backend.app.services.storage import storage_service
from backend.app.worker.celery_app import celery_app
from ml.extractors import PlaceholderExtractor, TesseractBaselineExtractor
from ml.extractors.layoutlmv3_extractor import get_cached_layoutlmv3_extractor

try:
    from celery.exceptions import SoftTimeLimitExceeded
except Exception:
    SoftTimeLimitExceeded = TimeoutError  # type: ignore


def _run_coro_sync(coro):
    """Runs an async extractor coroutine safely whether or not an event loop is active."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _get_worker_extractor(method: str):
    normalized = (method or settings.DEFAULT_EXTRACTION_METHOD).upper().strip()
    if normalized == ExtractionMethod.TESSERACT_BASELINE.value:
        return TesseractBaselineExtractor()
    if normalized == ExtractionMethod.LAYOUTLMV3.value:
        extractor = get_cached_layoutlmv3_extractor()
        ready, reason = extractor.validate_checkpoint()
        if not ready:
            raise RuntimeError(f"LayoutLMv3 checkpoint not ready: {reason}")
        return extractor
    if normalized == ExtractionMethod.PLACEHOLDER.value:
        return PlaceholderExtractor()
    raise ValueError(f"Unsupported extraction method '{normalized}'.")


def _sanitize_worker_error(exc: Exception) -> str:
    raw = str(exc).strip() or type(exc).__name__
    upload_prefix = str(storage_service.base_dir)
    resolved_prefix = str(storage_service.base_dir.resolve())
    cleaned = raw.replace(resolved_prefix, "[storage]").replace(upload_prefix, "[storage]")
    if len(cleaned) > 500:
        cleaned = cleaned[:497] + "..."
    return cleaned


def _is_permanent_error(exc: Exception) -> bool:
    if isinstance(exc, (ValueError, NotImplementedError, FileNotFoundError, FileValidationError)):
        return True
    msg = str(exc).lower()
    if any(
        token in msg
        for token in (
            "checkpoint not ready",
            "invalid layoutlmv3 checkpoint",
            "ocr dependency missing",
            "tesseract executable is not installed",
            "failed to invoke tesseract",
            "ml inference is disabled",
            "unsupported extraction method",
        )
    ):
        return True
    return False


def _sync_initialize_review(
    db: Session,
    document: Document,
    extraction: Optional[ExtractionResult],
    explicit_request: bool,
) -> None:
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

    existing_review = db.execute(
        select(DocumentReview).where(DocumentReview.document_id == document.id)
    ).scalar_one_or_none()

    if existing_review is None:
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
        db.flush()
        db.add(
            ReviewAuditEvent(
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
        )
    else:
        prev_status = existing_review.review_status
        existing_review.extraction_result_id = extraction.id if extraction else None
        if prev_status in (
            ReviewStatus.NOT_REQUIRED.value,
            ReviewStatus.PENDING_REVIEW.value,
        ):
            existing_review.review_status = initial_review_status
            db.add(
                ReviewAuditEvent(
                    review_id=existing_review.id,
                    action="REPROCESSED_REVIEW_UPDATED",
                    from_status=prev_status,
                    to_status=initial_review_status,
                    changed_fields=[],
                    actor_identifier="system-worker",
                    notes="Updated review state after reprocessing job completed.",
                )
            )


def execute_processing_job_sync(
    job_id: str,
    *,
    db: Optional[Session] = None,
    task_ctx: Any = None,
) -> Dict[str, Any]:
    """
    Core synchronous execution routine for a ProcessingJob.
    Used directly by Celery worker tasks and deterministic integration tests.
    """
    owns_session = db is None
    session = db or get_sync_session()
    t0 = time.perf_counter()

    try:
        job = session.get(ProcessingJob, job_id)
        if job is None:
            logger.warning("Processing job not found in database", extra={"job_id": job_id, "event": "job_missing"})
            return {"job_id": job_id, "status": "NOT_FOUND"}

        if job.status == JobStatus.CANCELLED.value:
            return {"job_id": job.id, "document_id": job.document_id, "status": JobStatus.CANCELLED.value}

        # Idempotency guard: never create duplicate ExtractionResult for the same job
        existing_ext = session.execute(
            select(ExtractionResult).where(ExtractionResult.job_id == job.id)
        ).scalar_one_or_none()
        if existing_ext is not None and job.status == JobStatus.COMPLETED.value:
            return {
                "job_id": job.id,
                "document_id": job.document_id,
                "status": JobStatus.COMPLETED.value,
                "idempotent_skip": True,
            }

        now = datetime.now(timezone.utc)
        claim = session.execute(
            update(ProcessingJob)
            .where(
                ProcessingJob.id == job.id,
                ProcessingJob.status.in_([JobStatus.QUEUED.value, JobStatus.RETRYING.value]),
            )
            .values(
                status=JobStatus.PROCESSING.value,
                attempts=ProcessingJob.attempts + 1,
                started_at=job.started_at or now,
                heartbeat_at=now,
                error_code=None,
                error_message=None,
                updated_at=now,
            )
        )
        if (claim.rowcount or 0) == 0:
            session.rollback()
            return {"job_id": job.id, "document_id": job.document_id, "status": "SKIPPED_CONCURRENT"}

        doc = session.get(Document, job.document_id)
        if doc is None:
            session.execute(
                update(ProcessingJob)
                .where(ProcessingJob.id == job.id)
                .values(status=JobStatus.FAILED.value, error_code="DOCUMENT_MISSING", completed_at=now)
            )
            session.commit()
            return {"job_id": job.id, "status": JobStatus.FAILED.value}

        doc.status = DocumentStatus.PROCESSING.value
        doc.error_message = None
        doc.updated_at = now
        session.commit()

        logger.info(
            "Started document processing job",
            extra={
                "event": "job_started",
                "job_id": job.id,
                "document_id": doc.id,
                "extraction_method": job.extraction_method,
                "status": JobStatus.PROCESSING.value,
            },
        )

        # Resolve file safely inside configured storage directory
        stored_path = storage_service.get_file_path(doc.stored_filename)
        if not stored_path.exists():
            legacy_candidate = Path(doc.file_path)
            if legacy_candidate.exists():
                stored_path = legacy_candidate

        extractor = _get_worker_extractor(job.extraction_method)
        extraction_output = _run_coro_sync(extractor.extract(stored_path))

        # Re-read inside final commit and guard idempotency again
        session.refresh(job)
        session.refresh(doc)
        if job.status == JobStatus.CANCELLED.value:
            return {"job_id": job.id, "document_id": doc.id, "status": JobStatus.CANCELLED.value}

        existing_ext = session.execute(
            select(ExtractionResult).where(ExtractionResult.job_id == job.id)
        ).scalar_one_or_none()
        if existing_ext is None:
            existing_ext = ExtractionResult(
                document_id=doc.id,
                job_id=job.id,
                extraction_method=extraction_output.extraction_method,
                structured_data=extraction_output.structured_data,
                confidence=extraction_output.confidence,
                field_confidences=extraction_output.field_confidences,
                raw_text=extraction_output.raw_text,
                validation_errors=extraction_output.validation_errors,
                extraction_metadata=extraction_output.metadata,
            )
            session.add(existing_ext)
            session.flush()

        finish_time = datetime.now(timezone.utc)
        doc.status = DocumentStatus.COMPLETED.value
        doc.error_message = None
        doc.updated_at = finish_time
        _sync_initialize_review(session, doc, existing_ext, explicit_request=bool(job.request_review))

        job.status = JobStatus.COMPLETED.value
        job.completed_at = finish_time
        job.heartbeat_at = finish_time
        job.updated_at = finish_time
        session.commit()

        duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        logger.info(
            "Document processing job completed",
            extra={
                "event": "job_completed",
                "job_id": job.id,
                "document_id": doc.id,
                "extraction_method": job.extraction_method,
                "status": JobStatus.COMPLETED.value,
                "duration_ms": duration_ms,
            },
        )
        return {
            "job_id": job.id,
            "document_id": doc.id,
            "status": JobStatus.COMPLETED.value,
            "duration_ms": duration_ms,
        }

    except Exception as exc:
        try:
            session.rollback()
        except Exception:
            pass
        if owns_session:
            from sqlalchemy.exc import DBAPIError

            if isinstance(exc, DBAPIError):
                try:
                    session.close()
                except Exception:
                    pass
                session = get_sync_session()

        duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        try:
            job = session.get(ProcessingJob, job_id)
            doc = session.get(Document, job.document_id) if job else None
        except Exception:
            if owns_session:
                session.close()
                session = get_sync_session()
            job = session.get(ProcessingJob, job_id)
            doc = session.get(Document, job.document_id) if job else None

        safe_msg = _sanitize_worker_error(exc)
        err_cat = "WORKER_SOFT_TIMEOUT" if "SoftTimeLimitExceeded" in type(exc).__name__ else type(exc).__name__

        if job is not None and job.status != JobStatus.CANCELLED.value:
            now = datetime.now(timezone.utc)
            can_retry = (
                not _is_permanent_error(exc)
                and job.attempts < job.max_attempts
            )
            if can_retry:
                job.status = JobStatus.RETRYING.value
                job.error_code = err_cat
                job.error_message = safe_msg
                job.heartbeat_at = now
                job.updated_at = now
                if doc is not None:
                    doc.status = DocumentStatus.PENDING.value
                    doc.error_message = f"Transient failure ({err_cat}); retry scheduled."
                    doc.updated_at = now
                session.commit()
                logger.warning(
                    "Document processing job encountered transient failure; scheduling retry",
                    extra={
                        "event": "job_retry_scheduled",
                        "job_id": job.id,
                        "document_id": job.document_id,
                        "extraction_method": job.extraction_method,
                        "status": JobStatus.RETRYING.value,
                        "duration_ms": duration_ms,
                        "error_category": err_cat,
                    },
                )
                if task_ctx is not None:
                    delay = min(60, 2 ** max(1, job.attempts))
                    raise task_ctx.retry(exc=exc, countdown=delay)
                return {
                    "job_id": job.id,
                    "document_id": job.document_id,
                    "status": JobStatus.RETRYING.value,
                    "error_code": err_cat,
                }

            job.status = JobStatus.FAILED.value
            job.error_code = err_cat
            job.error_message = safe_msg
            job.completed_at = now
            job.heartbeat_at = now
            job.updated_at = now
            if doc is not None:
                doc.status = DocumentStatus.FAILED.value
                doc.error_message = safe_msg
                doc.updated_at = now
                _sync_initialize_review(session, doc, None, explicit_request=bool(job.request_review))
            session.commit()

            logger.error(
                "Document processing job failed permanently",
                extra={
                    "event": "job_failed",
                    "job_id": job.id,
                    "document_id": job.document_id,
                    "extraction_method": job.extraction_method,
                    "status": JobStatus.FAILED.value,
                    "duration_ms": duration_ms,
                    "error_category": err_cat,
                },
            )
            return {
                "job_id": job.id,
                "document_id": job.document_id,
                "status": JobStatus.FAILED.value,
                "error_code": err_cat,
            }

        return {"job_id": job_id, "status": JobStatus.FAILED.value, "error_code": err_cat}
    finally:
        if owns_session:
            session.close()


@celery_app.task(
    bind=True,
    name="documind.process_document_job",
    max_retries=settings.MAX_JOB_ATTEMPTS,
)
def process_document_job(self, job_id: str) -> Dict[str, Any]:
    return execute_processing_job_sync(job_id=job_id, task_ctx=self)


def recover_stale_jobs_and_outbox_sync(db: Optional[Session] = None) -> Dict[str, int]:
    """
    Republishes pending outbox events whose broker push initially failed and
    recovers stale jobs that stayed in PROCESSING past STALE_JOB_TIMEOUT_SECONDS.
    """
    owns_session = db is None
    session = db or get_sync_session()
    republished = 0
    requeued_stale = 0
    failed_stale = 0
    now = datetime.now(timezone.utc)

    try:
        outbox_cutoff = now - timedelta(seconds=5)
        pending_events = list(
            session.execute(
                select(OutboxEvent)
                .where(
                    OutboxEvent.published_at.is_(None),
                    OutboxEvent.created_at <= outbox_cutoff,
                )
                .limit(50)
            ).scalars().all()
        )
        for ev in pending_events:
            job_id = (ev.payload or {}).get("job_id") or ev.aggregate_id
            ev.publish_attempts += 1
            try:
                async_res = process_document_job.delay(job_id)
                ev.published_at = now
                ev.last_error = None
                job = session.get(ProcessingJob, job_id)
                if job and not job.task_id:
                    job.task_id = getattr(async_res, "id", None)
                republished += 1
            except Exception as exc:
                ev.last_error = str(exc)[:500]
        session.commit()

        stale_cutoff = now - timedelta(seconds=settings.STALE_JOB_TIMEOUT_SECONDS)
        stale_jobs = list(
            session.execute(
                select(ProcessingJob).where(
                    ProcessingJob.status == JobStatus.PROCESSING.value,
                    ProcessingJob.heartbeat_at.isnot(None),
                    ProcessingJob.heartbeat_at <= stale_cutoff,
                )
            ).scalars().all()
        )
        for sjob in stale_jobs:
            sdoc = session.get(Document, sjob.document_id)
            if sjob.attempts < sjob.max_attempts:
                sjob.status = JobStatus.RETRYING.value
                sjob.error_code = "STALE_JOB_RECOVERED"
                sjob.error_message = "Worker timed out or stopped sending heartbeats; job re-queued."
                sjob.updated_at = now
                if sdoc:
                    sdoc.status = DocumentStatus.PENDING.value
                    sdoc.error_message = "Worker timeout detected; automatically retrying."
                    sdoc.updated_at = now
                session.commit()
                try:
                    async_res = process_document_job.delay(sjob.id)
                    sjob.task_id = getattr(async_res, "id", None)
                    session.commit()
                except Exception:
                    pass
                requeued_stale += 1
            else:
                sjob.status = JobStatus.FAILED.value
                sjob.error_code = "STALE_JOB_TIMEOUT"
                sjob.error_message = "Processing job timed out after maximum retry attempts."
                sjob.completed_at = now
                sjob.updated_at = now
                if sdoc:
                    sdoc.status = DocumentStatus.FAILED.value
                    sdoc.error_message = sjob.error_message
                    sdoc.updated_at = now
                session.commit()
                failed_stale += 1

        return {
            "republished_outbox": republished,
            "requeued_stale_jobs": requeued_stale,
            "failed_stale_jobs": failed_stale,
        }
    finally:
        if owns_session:
            session.close()


@celery_app.task(name="documind.recover_stale_jobs_and_outbox")
def recover_stale_jobs_and_outbox() -> Dict[str, int]:
    return recover_stale_jobs_and_outbox_sync()
