from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.config import settings
from backend.app.core.exceptions import DocumentNotFoundError
from backend.app.logger import logger
from backend.app.models.document import Document, DocumentStatus
from backend.app.models.extraction import ExtractionMethod, ExtractionResult
from backend.app.services.storage import storage_service
from backend.app.services.validator import validate_file
from ml.extractors import LayoutLMv3Extractor, PlaceholderExtractor, TesseractBaselineExtractor


def get_extractor(method_name: str):
    method = method_name.upper()
    if method == ExtractionMethod.TESSERACT_BASELINE.value:
        return TesseractBaselineExtractor()
    elif method == ExtractionMethod.LAYOUTLMV3.value:
        return LayoutLMv3Extractor()
    return PlaceholderExtractor()


class DocumentService:
    async def create_document_from_upload(
        self,
        db: AsyncSession,
        filename: str,
        content_type: str,
        file_bytes: bytes,
        extraction_method: Optional[str] = None,
    ) -> Document:
        """
        Validates, persists, records and initiates initial extraction record for an uploaded document.
        """
        # 1. Validate signature, MIME type, file size
        detected_mime, ext = validate_file(
            filename=filename,
            content_type=content_type,
            file_bytes=file_bytes,
        )

        # 2. Persist safely to disk outside public paths
        file_id, stored_filename, file_path = storage_service.save_file(
            file_bytes=file_bytes,
            extension=ext,
        )

        # 3. Create document record
        document = Document(
            id=file_id,
            original_filename=filename,
            stored_filename=stored_filename,
            file_path=file_path,
            file_size_bytes=len(file_bytes),
            content_type=detected_mime,
            status=DocumentStatus.PENDING.value,
        )
        db.add(document)
        await db.commit()
        await db.refresh(document)

        # 4. Run extractor (Phase 1 placeholder/contract)
        method = extraction_method or settings.DEFAULT_EXTRACTION_METHOD
        extractor = get_extractor(method)
        try:
            extraction_output = await extractor.extract(Path(file_path))
            
            # Record extraction result
            extraction_record = ExtractionResult(
                document_id=document.id,
                extraction_method=extractor.method_name,
                structured_data=extraction_output.structured_data,
                confidence=extraction_output.confidence,
                validation_errors=extraction_output.validation_errors,
            )
            db.add(extraction_record)

            # In Phase 1 placeholder mode, document is marked COMPLETED with placeholder result
            # or PENDING if scheduled for background worker.
            document.status = DocumentStatus.COMPLETED.value
            document.updated_at = datetime.now(timezone.utc)
            await db.commit()
            await db.refresh(document)
            logger.info(f"Created extraction result for document {document.id} via {extractor.method_name}")
        except Exception as e:
            logger.error(f"Error executing extraction on document {document.id}: {e}")
            document.status = DocumentStatus.FAILED.value
            document.error_message = str(e)
            await db.commit()
            await db.refresh(document)

        # Return document with extractions eagerly loaded
        return await self.get_document_by_id(db, document.id)

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
