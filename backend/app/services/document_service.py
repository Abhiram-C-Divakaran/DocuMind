from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.config import settings
from backend.app.core.exceptions import DocumentNotFoundError, FileValidationError
from backend.app.logger import logger
from backend.app.models.document import Document, DocumentStatus
from backend.app.models.extraction import ExtractionMethod, ExtractionResult
from backend.app.services.storage import storage_service
from backend.app.services.validator import validate_file
from ml.extractors import LayoutLMv3Extractor, PlaceholderExtractor, TesseractBaselineExtractor

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
        extractor = LayoutLMv3Extractor()
        ok, reason = extractor.validate_checkpoint()
        if not ok:
            raise FileValidationError(
                f"LayoutLMv3 multimodal extraction is unavailable in Phase 2A / without fine-tuned weights. "
                f"Model checkpoint is not available at '{extractor.checkpoint_dir}': {reason}. "
                "Please run the training pipeline ('python -m ml.training.train_layoutlmv3') "
                "to produce valid trained weights before selecting this extractor."
            )
        return extractor

    if method == ExtractionMethod.TESSERACT_BASELINE.value:
        return TesseractBaselineExtractor()
    return PlaceholderExtractor()


class DocumentService:
    async def create_document_from_upload(
        self,
        db: AsyncSession,
        filename: str,
        content_type: str,
        file_bytes: bytes,
        extraction_method: Optional[str] = None,
    ) -> Tuple[Document, str]:
        """
        Validates, persists, records and executes extraction for an uploaded document.
        Returns:
            Tuple of (document, status_message)
        """
        # 1. Validate requested extraction method
        method_str = (extraction_method or settings.DEFAULT_EXTRACTION_METHOD).upper().strip()
        extractor = get_extractor(method_str)

        # 2. Validate signature, MIME type, file size
        detected_mime, ext = validate_file(
            filename=filename,
            content_type=content_type,
            file_bytes=file_bytes,
        )

        # 3. Persist safely to disk outside public paths
        file_id, stored_filename, file_path = storage_service.save_file(
            file_bytes=file_bytes,
            extension=ext,
        )

        # 4. Create document record with PROCESSING or PENDING
        real_extractors = {
            ExtractionMethod.TESSERACT_BASELINE.value,
            ExtractionMethod.LAYOUTLMV3.value,
        }
        is_real_extractor = extractor.method_name in real_extractors
        initial_status = (
            DocumentStatus.PROCESSING.value
            if is_real_extractor
            else DocumentStatus.PENDING.value
        )

        document = Document(
            id=file_id,
            original_filename=filename,
            stored_filename=stored_filename,
            file_path=file_path,
            file_size_bytes=len(file_bytes),
            content_type=detected_mime,
            status=initial_status,
        )
        db.add(document)
        await db.commit()
        await db.refresh(document)

        status_message: str

        # 5. Run extraction
        if is_real_extractor:
            method_label = (
                "LayoutLMv3 multimodal extraction"
                if extractor.method_name == ExtractionMethod.LAYOUTLMV3.value
                else "Tesseract OCR"
            )
            try:
                extraction_output = await extractor.extract(Path(file_path))

                # Save successful extraction
                extraction_record = ExtractionResult(
                    document_id=document.id,
                    extraction_method=extractor.method_name,
                    structured_data=extraction_output.structured_data,
                    confidence=extraction_output.confidence,
                    field_confidences=extraction_output.field_confidences,
                    raw_text=extraction_output.raw_text,
                    validation_errors=extraction_output.validation_errors,
                    extraction_metadata=extraction_output.metadata,
                )
                db.add(extraction_record)

                document.status = DocumentStatus.COMPLETED.value
                document.updated_at = datetime.now(timezone.utc)
                await db.commit()
                await db.refresh(document)

                status_message = (
                    f"Document '{document.original_filename}' processed successfully via {method_label}."
                )
                logger.info(f"Document {document.id} successfully processed via {method_label}.")

            except Exception as e:
                sanitized_error = str(e)
                logger.error(
                    f"Error executing {method_label} on document {document.id}: {sanitized_error}"
                )

                document.status = DocumentStatus.FAILED.value
                document.error_message = sanitized_error
                document.updated_at = datetime.now(timezone.utc)

                # Persist failed extraction record with error explanation
                failed_extraction = ExtractionResult(
                    document_id=document.id,
                    extraction_method=extractor.method_name,
                    structured_data=None,
                    confidence=0.0,
                    field_confidences=None,
                    raw_text=None,
                    validation_errors=[sanitized_error],
                    extraction_metadata={"status": "FAILED", "error": sanitized_error},
                )
                db.add(failed_extraction)
                await db.commit()
                await db.refresh(document)

                status_message = (
                    f"Document '{document.original_filename}' uploaded but {method_label} failed: {sanitized_error}"
                )

        else:
            # PLACEHOLDER extraction mode: status remains PENDING
            extraction_output = await extractor.extract(Path(file_path))
            extraction_record = ExtractionResult(
                document_id=document.id,
                extraction_method=extractor.method_name,
                structured_data=None,
                confidence=None,
                field_confidences=None,
                raw_text=None,
                validation_errors=extraction_output.validation_errors,
                extraction_metadata=extraction_output.metadata,
            )
            db.add(extraction_record)

            document.status = DocumentStatus.PENDING.value
            document.updated_at = datetime.now(timezone.utc)
            await db.commit()
            await db.refresh(document)

            status_message = (
                f"Document '{document.original_filename}' uploaded in placeholder mode (status: PENDING)."
            )

        # Return document with extractions eagerly loaded
        refreshed_doc = await self.get_document_by_id(db, document.id)
        return refreshed_doc, status_message

    async def get_document_by_id(self, db: AsyncSession, document_id: str) -> Document:
        query = (
            select(Document)
            .where(Document.id == document_id)
            .options(selectinload(Document.extractions))
        )
        result = await db.execute(query)
        document = result.scalar_one_or_none()
        if not document:
            raise DocumentNotFoundError(document_id=document_id)
        return document

    async def list_documents(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 20,
        status: Optional[str] = None,
        search: Optional[str] = None,
    ) -> Tuple[List[Document], int]:
        query = select(Document).options(selectinload(Document.extractions))
        count_query = select(func.count(Document.id))

        if status:
            query = query.where(Document.status == status.upper())
            count_query = count_query.where(Document.status == status.upper())

        if search:
            search_filter = Document.original_filename.ilike(f"%{search}%")
            query = query.where(search_filter)
            count_query = count_query.where(search_filter)

        query = query.order_by(desc(Document.created_at)).offset(skip).limit(limit)

        total_result = await db.execute(count_query)
        total = total_result.scalar_one()

        docs_result = await db.execute(query)
        documents = list(docs_result.scalars().all())

        return documents, total

    async def get_dashboard_metrics(self, db: AsyncSession) -> dict:
        total_result = await db.execute(select(func.count(Document.id)))
        total = total_result.scalar_one()

        pending_result = await db.execute(
            select(func.count(Document.id)).where(Document.status == DocumentStatus.PENDING.value)
        )
        pending = pending_result.scalar_one()

        processing_result = await db.execute(
            select(func.count(Document.id)).where(Document.status == DocumentStatus.PROCESSING.value)
        )
        processing = processing_result.scalar_one()

        completed_result = await db.execute(
            select(func.count(Document.id)).where(Document.status == DocumentStatus.COMPLETED.value)
        )
        completed = completed_result.scalar_one()

        failed_result = await db.execute(
            select(func.count(Document.id)).where(Document.status == DocumentStatus.FAILED.value)
        )
        failed = failed_result.scalar_one()

        return {
            "total_documents": total,
            "pending": pending,
            "processing": processing,
            "completed": completed,
            "failed": failed,
        }


document_service = DocumentService()
