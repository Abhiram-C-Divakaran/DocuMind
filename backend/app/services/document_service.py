import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.config import settings
from backend.app.core.exceptions import (
    ConflictError,
    DocumentNotFoundError,
    FileValidationError,
    JobNotFoundError,
    PermissionDeniedError,
)
from backend.app.logger import logger
from backend.app.models.document import Document, DocumentStatus
from backend.app.models.extraction import ExtractionMethod, ExtractionResult
from backend.app.models.job import JobStatus, OutboxEvent, ProcessingJob
from backend.app.models.review import ReviewStatus
from backend.app.models.user import User, UserRole
from backend.app.services.media_security import validate_media_safety
from backend.app.services.storage import storage_service
from backend.app.services.validator import validate_file
from ml.extractors import LayoutLMv3Extractor, PlaceholderExtractor, TesseractBaselineExtractor
from ml.extractors.layoutlmv3_extractor import (
    get_cached_layoutlmv3_extractor,
    sanitize_checkpoint_identity,
)

VALID_EXTRACTION_METHODS = {
    ExtractionMethod.TESSERACT_BASELINE.value,
    ExtractionMethod.LAYOUTLMV3.value,
    ExtractionMethod.PLACEHOLDER.value,
}


def get_extractor(method_name: str):
    method = method_name.upper().strip()
    if method not in VALID_EXTRACTION_METHODS:
        raise FileValidationError(
            f"Unsupported extraction method '{method_name}'. "
            f"Allowed methods: {', '.join(sorted(VALID_EXTRACTION_METHODS))}."
        )

    if method == ExtractionMethod.LAYOUTLMV3.value:
        extractor = get_cached_layoutlmv3_extractor()
        ok, reason = extractor.validate_checkpoint()
        if not ok:
            ckpt_id = sanitize_checkpoint_identity(extractor.checkpoint_dir)
            raise FileValidationError(
                f"LayoutLMv3 multimodal extraction is unavailable in Phase 2A / without fine-tuned weights. "
                f"Model checkpoint is not available for '{ckpt_id}': {reason}. "
                "Please run the training pipeline ('python -m ml.training.train_layoutlmv3') "
                "to produce valid trained weights before selecting this extractor."
            )
        return extractor

    if method == ExtractionMethod.TESSERACT_BASELINE.value:
        return TesseractBaselineExtractor()
    return PlaceholderExtractor()


def _check_doc_ownership(document: Document, user: Optional[User]) -> None:
    if user is None:
        return
    if user.role == UserRole.ADMIN.value:
        return
    if document.owner_id is not None and document.owner_id == user.id:
        return
    raise PermissionDeniedError("You do not have access to this document.")


def _attach_latest_job(document: Document) -> Document:
    jobs = getattr(document, "processing_jobs", None) or []
    document.latest_job = jobs[0] if jobs else None
    return document


class DocumentService:
    def _dispatch_celery_job(self, job_id: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Attempts to publish a processing job to Celery/Redis.
        Returns (task_id, error_message_if_failed).
        Never leaks stack traces or sensitive paths.
        """
        try:
            from backend.app.worker.tasks import process_document_job

            async_res = process_document_job.delay(job_id)
            return getattr(async_res, "id", None), None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {str(exc)[:300]}"

    async def enqueue_document_upload(
        self,
        db: AsyncSession,
        filename: str,
        content_type: str,
        file_bytes: bytes,
        owner_id: Optional[str] = None,
        extraction_method: Optional[str] = None,
        request_review: bool = False,
    ) -> Tuple[Document, ProcessingJob, str]:
        """
        Validates file & media safety, stores artifact safely, creates Document (PENDING),
        ProcessingJob (QUEUED), and transactional OutboxEvent, commits, and dispatches Celery task.
        Returns (document, job, status_message).
        """
        method_str = (extraction_method or settings.DEFAULT_EXTRACTION_METHOD).upper().strip()
        get_extractor(method_str)

        detected_mime, ext = validate_file(
            filename=filename,
            content_type=content_type,
            file_bytes=file_bytes,
        )
        validate_media_safety(content=file_bytes, content_type=detected_mime, filename=filename)

        file_id, stored_filename, file_path = storage_service.save_file(
            file_bytes=file_bytes,
            extension=ext,
        )

        initial_review_status = (
            ReviewStatus.PENDING_REVIEW.value
            if request_review
            else ReviewStatus.NOT_REQUIRED.value
        )
        review_reasons = ["user_requested_review"] if request_review else []

        document = Document(
            id=file_id,
            owner_id=owner_id,
            original_filename=Path(filename).name,
            stored_filename=stored_filename,
            file_path=file_path,
            file_size_bytes=len(file_bytes),
            content_type=detected_mime,
            status=DocumentStatus.PENDING.value,
            review_status=initial_review_status,
            needs_review=bool(request_review),
            review_reasons=review_reasons,
        )
        job_id = str(uuid.uuid4())
        job = ProcessingJob(
            id=job_id,
            document_id=file_id,
            idempotency_key=f"{file_id}:{method_str}:1",
            extraction_method=method_str,
            status=JobStatus.QUEUED.value,
            attempts=0,
            max_attempts=settings.MAX_JOB_ATTEMPTS,
            request_review=bool(request_review),
        )
        outbox = OutboxEvent(
            aggregate_type="processing_job",
            aggregate_id=job_id,
            event_type="DOCUMENT_PROCESS_REQUESTED",
            payload={
                "job_id": job_id,
                "document_id": file_id,
                "extraction_method": method_str,
            },
        )

        try:
            db.add(document)
            db.add(job)
            db.add(outbox)
            await db.commit()
        except Exception:
            await db.rollback()
            storage_service.delete_file(stored_filename)
            raise

        task_id, dispatch_err = self._dispatch_celery_job(job_id)
        now = datetime.now(timezone.utc)
        outbox.publish_attempts = 1
        if dispatch_err is None:
            outbox.published_at = now
            job.task_id = task_id
        else:
            outbox.last_error = dispatch_err
            logger.warning(
                "Broker dispatch deferred; outbox event retained for worker recovery",
                extra={
                    "event": "broker_dispatch_deferred",
                    "document_id": file_id,
                    "job_id": job_id,
                    "extraction_method": method_str,
                    "error_category": "BrokerUnavailable",
                },
            )
        await db.commit()
        await db.refresh(job)

        refreshed_doc = await self.get_document_by_id(db, file_id)
        logger.info(
            "Document queued for asynchronous extraction",
            extra={
                "event": "document_queued",
                "document_id": file_id,
                "job_id": job_id,
                "extraction_method": method_str,
                "status": DocumentStatus.PENDING.value,
            },
        )
        return (
            refreshed_doc,
            job,
            f"Document '{refreshed_doc.original_filename}' uploaded and queued for asynchronous processing.",
        )

    async def create_document_from_upload(
        self,
        db: AsyncSession,
        filename: str,
        content_type: str,
        file_bytes: bytes,
        extraction_method: Optional[str] = None,
        request_review: bool = False,
        owner_id: Optional[str] = None,
    ) -> Tuple[Document, str]:
        """
        Enqueues an uploaded document cleanly via the async processing pipeline.
        Preserved signature for compatibility with callers passing 4-6 arguments.
        """
        doc, _, msg = await self.enqueue_document_upload(
            db=db,
            filename=filename,
            content_type=content_type,
            file_bytes=file_bytes,
            owner_id=owner_id,
            extraction_method=extraction_method,
            request_review=request_review,
        )
        return doc, msg

    async def retry_document_processing(
        self,
        db: AsyncSession,
        document_id: str,
        user: Optional[User] = None,
        extraction_method: Optional[str] = None,
    ) -> Tuple[Document, ProcessingJob]:
        doc = await self.get_document_by_id(db, document_id, user=user)
        active_job = next(
            (
                j
                for j in (doc.processing_jobs or [])
                if j.status in (JobStatus.QUEUED.value, JobStatus.PROCESSING.value, JobStatus.RETRYING.value)
            ),
            None,
        )
        if active_job is not None:
            raise ConflictError(
                f"Document '{document_id}' already has an active job in '{active_job.status}' state.",
                error_code="JOB_ALREADY_ACTIVE",
                extra={"job_id": active_job.id, "status": active_job.status},
            )

        last_method = (doc.processing_jobs[0].extraction_method if doc.processing_jobs else None)
        method_str = (extraction_method or last_method or settings.DEFAULT_EXTRACTION_METHOD).upper().strip()
        get_extractor(method_str)

        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        doc.status = DocumentStatus.PENDING.value
        doc.error_message = None
        doc.updated_at = now

        job = ProcessingJob(
            id=job_id,
            document_id=doc.id,
            idempotency_key=f"{doc.id}:{method_str}:retry-{uuid.uuid4().hex[:12]}",
            extraction_method=method_str,
            status=JobStatus.QUEUED.value,
            attempts=0,
            max_attempts=settings.MAX_JOB_ATTEMPTS,
            request_review=bool(doc.needs_review),
        )
        outbox = OutboxEvent(
            aggregate_type="processing_job",
            aggregate_id=job_id,
            event_type="DOCUMENT_PROCESS_REQUESTED",
            payload={
                "job_id": job_id,
                "document_id": doc.id,
                "extraction_method": method_str,
            },
        )
        db.add(job)
        db.add(outbox)
        await db.commit()

        task_id, dispatch_err = self._dispatch_celery_job(job_id)
        outbox.publish_attempts = 1
        if dispatch_err is None:
            outbox.published_at = datetime.now(timezone.utc)
            job.task_id = task_id
        else:
            outbox.last_error = dispatch_err
        await db.commit()

        refreshed = await self.get_document_by_id(db, doc.id, user=user)
        await db.refresh(job)
        return refreshed, job

    @staticmethod
    def _revoke_celery_task(task_id: str) -> None:
        try:
            from backend.app.worker.celery_app import celery_app

            celery_app.control.revoke(task_id, terminate=False)
        except Exception:
            pass

    async def cancel_document_processing(
        self,
        db: AsyncSession,
        document_id: str,
        user: Optional[User] = None,
    ) -> ProcessingJob:
        doc = await self.get_document_by_id(db, document_id, user=user)
        active_job = next(
            (
                j
                for j in (doc.processing_jobs or [])
                if j.status in (JobStatus.QUEUED.value, JobStatus.PROCESSING.value, JobStatus.RETRYING.value)
            ),
            None,
        )
        if active_job is None:
            raise ConflictError(
                "No active queued or running processing job exists for this document.",
                error_code="NO_ACTIVE_JOB",
            )

        now = datetime.now(timezone.utc)
        active_job.status = JobStatus.CANCELLED.value
        active_job.completed_at = now
        active_job.error_code = "CANCELLED_BY_USER"
        active_job.error_message = "Cancelled by operator."
        active_job.updated_at = now
        if doc.status != DocumentStatus.COMPLETED.value:
            doc.status = DocumentStatus.FAILED.value
            doc.error_message = "Processing cancelled by user."
            doc.updated_at = now

        await db.commit()
        if active_job.task_id:
            self._revoke_celery_task(active_job.task_id)
        await db.refresh(active_job)
        return active_job

    async def delete_document(
        self,
        db: AsyncSession,
        document_id: str,
        user: Optional[User] = None,
    ) -> str:
        doc = await self.get_document_by_id(db, document_id, user=user)
        active_job = next(
            (
                j
                for j in (doc.processing_jobs or [])
                if j.status == JobStatus.PROCESSING.value
            ),
            None,
        )
        if active_job is not None:
            raise ConflictError(
                "Cannot delete a document while extraction is actively PROCESSING.",
                error_code="DOCUMENT_BUSY",
            )
        stored_filename = doc.stored_filename
        await db.delete(doc)
        await db.commit()
        storage_service.delete_file(stored_filename)
        return document_id

    async def get_document_by_id(
        self,
        db: AsyncSession,
        document_id: str,
        user: Optional[User] = None,
    ) -> Document:
        query = (
            select(Document)
            .where(Document.id == document_id)
            .options(
                selectinload(Document.extractions),
                selectinload(Document.processing_jobs),
            )
        )
        result = await db.execute(query)
        document = result.scalar_one_or_none()
        if not document:
            raise DocumentNotFoundError(document_id=document_id)
        _check_doc_ownership(document, user)
        return _attach_latest_job(document)

    async def list_documents(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 20,
        status: Optional[str] = None,
        review_status: Optional[str] = None,
        search: Optional[str] = None,
        user: Optional[User] = None,
    ) -> Tuple[List[Document], int]:
        query = select(Document).options(
            selectinload(Document.extractions),
            selectinload(Document.processing_jobs),
        )
        count_query = select(func.count(Document.id))

        if user is not None and user.role != UserRole.ADMIN.value:
            query = query.where(Document.owner_id == user.id)
            count_query = count_query.where(Document.owner_id == user.id)

        if status:
            query = query.where(Document.status == status.upper())
            count_query = count_query.where(Document.status == status.upper())

        if review_status:
            query = query.where(Document.review_status == review_status.upper())
            count_query = count_query.where(Document.review_status == review_status.upper())

        if search:
            search_filter = Document.original_filename.ilike(f"%{search}%")
            query = query.where(search_filter)
            count_query = count_query.where(search_filter)

        query = query.order_by(desc(Document.created_at)).offset(skip).limit(limit)

        total = int((await db.execute(count_query)).scalar_one())
        documents = list((await db.execute(query)).scalars().all())
        for doc in documents:
            _attach_latest_job(doc)
        return documents, total

    async def get_dashboard_metrics(self, db: AsyncSession, user: Optional[User] = None) -> dict:
        def _scoped_count(extra_clause=None):
            stmt = select(func.count(Document.id))
            if user is not None and user.role != UserRole.ADMIN.value:
                stmt = stmt.where(Document.owner_id == user.id)
            if extra_clause is not None:
                stmt = stmt.where(extra_clause)
            return stmt

        total = int((await db.execute(_scoped_count())).scalar_one())
        pending = int((await db.execute(_scoped_count(Document.status == DocumentStatus.PENDING.value))).scalar_one())
        processing = int((await db.execute(_scoped_count(Document.status == DocumentStatus.PROCESSING.value))).scalar_one())
        completed = int((await db.execute(_scoped_count(Document.status == DocumentStatus.COMPLETED.value))).scalar_one())
        failed = int((await db.execute(_scoped_count(Document.status == DocumentStatus.FAILED.value))).scalar_one())

        return {
            "total_documents": total,
            "pending": pending,
            "processing": processing,
            "completed": completed,
            "failed": failed,
        }

    async def get_job_by_id(
        self,
        db: AsyncSession,
        job_id: str,
        user: Optional[User] = None,
    ) -> ProcessingJob:
        result = await db.execute(
            select(ProcessingJob)
            .where(ProcessingJob.id == job_id)
            .options(selectinload(ProcessingJob.document))
        )
        job = result.scalar_one_or_none()
        if not job:
            raise JobNotFoundError(job_id=job_id)
        if job.document:
            _check_doc_ownership(job.document, user)
        return job

    async def list_jobs(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 20,
        status: Optional[str] = None,
        document_id: Optional[str] = None,
        user: Optional[User] = None,
    ) -> Tuple[List[ProcessingJob], int]:
        query = select(ProcessingJob).join(Document, ProcessingJob.document_id == Document.id)
        count_query = select(func.count(ProcessingJob.id)).join(Document, ProcessingJob.document_id == Document.id)

        if user is not None and user.role != UserRole.ADMIN.value:
            query = query.where(Document.owner_id == user.id)
            count_query = count_query.where(Document.owner_id == user.id)
        if status:
            query = query.where(ProcessingJob.status == status.upper())
            count_query = count_query.where(ProcessingJob.status == status.upper())
        if document_id:
            query = query.where(ProcessingJob.document_id == document_id)
            count_query = count_query.where(ProcessingJob.document_id == document_id)

        query = query.order_by(desc(ProcessingJob.created_at)).offset(skip).limit(limit)
        total = int((await db.execute(count_query)).scalar_one())
        items = list((await db.execute(query)).scalars().all())
        return items, total

    async def get_job_metrics(self, db: AsyncSession, user: Optional[User] = None) -> dict:
        query = select(ProcessingJob).join(Document, ProcessingJob.document_id == Document.id)
        if user is not None and user.role != UserRole.ADMIN.value:
            query = query.where(Document.owner_id == user.id)

        jobs = list((await db.execute(query)).scalars().all())
        counts: Dict[str, int] = {
            JobStatus.QUEUED.value: 0,
            JobStatus.PROCESSING.value: 0,
            JobStatus.RETRYING.value: 0,
            JobStatus.COMPLETED.value: 0,
            JobStatus.FAILED.value: 0,
            JobStatus.CANCELLED.value: 0,
        }
        by_method: Dict[str, int] = {}
        durations: List[float] = []
        stale_count = 0
        now = datetime.now(timezone.utc)
        stale_cutoff = now - timedelta(seconds=settings.STALE_JOB_TIMEOUT_SECONDS)

        for job in jobs:
            counts[job.status] = counts.get(job.status, 0) + 1
            by_method[job.extraction_method] = by_method.get(job.extraction_method, 0) + 1
            if job.started_at and job.completed_at:
                s = job.started_at if job.started_at.tzinfo else job.started_at.replace(tzinfo=timezone.utc)
                c = job.completed_at if job.completed_at.tzinfo else job.completed_at.replace(tzinfo=timezone.utc)
                durations.append(max(0.0, (c - s).total_seconds()))
            if job.status == JobStatus.PROCESSING.value and job.heartbeat_at:
                hb = job.heartbeat_at if job.heartbeat_at.tzinfo else job.heartbeat_at.replace(tzinfo=timezone.utc)
                if hb <= stale_cutoff:
                    stale_count += 1

        unpublished_outbox = int(
            (await db.execute(select(func.count(OutboxEvent.id)).where(OutboxEvent.published_at.is_(None)))).scalar_one()
        )
        avg_dur = round(sum(durations) / len(durations), 3) if durations else None
        return {
            "total_jobs": len(jobs),
            "queued": counts[JobStatus.QUEUED.value],
            "processing": counts[JobStatus.PROCESSING.value],
            "retrying": counts[JobStatus.RETRYING.value],
            "completed": counts[JobStatus.COMPLETED.value],
            "failed": counts[JobStatus.FAILED.value],
            "cancelled": counts[JobStatus.CANCELLED.value],
            "stale_running_jobs": stale_count,
            "unpublished_outbox_events": unpublished_outbox,
            "average_duration_seconds": avg_dur,
            "by_extraction_method": by_method,
        }


document_service = DocumentService()
