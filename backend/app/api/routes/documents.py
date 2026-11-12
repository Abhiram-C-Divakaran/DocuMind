from typing import Optional
from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import get_db
from backend.app.core.exceptions import FileValidationError
from backend.app.logger import logger
from backend.app.schemas.document import (
    DocumentListResponse,
    DocumentResponse,
    DocumentUploadResponse,
)
from backend.app.services.document_service import document_service

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload (PDF, PNG, JPEG)"),
    extraction_method: Optional[str] = Form(None, description="Optional extraction method (TESSERACT_BASELINE, PLACEHOLDER)"),
    db: AsyncSession = Depends(get_db),
):
    """
    Multipart upload endpoint for receipts and invoices.
    Performs security validation:
    - Maximum file size check
    - Declared MIME type inspection
    - Magic byte file signature detection
    - Secure UUID disk storage outside public web paths
    - OCR extraction via Tesseract baseline
    """
    if not file.filename:
        raise FileValidationError("Uploaded file has no filename.")

    content = await file.read()
    content_type = file.content_type or "application/octet-stream"

    logger.info(
        f"Received upload request for '{file.filename}', claimed type '{content_type}', size {len(content)} bytes"
    )

    document, message = await document_service.create_document_from_upload(
        db=db,
        filename=file.filename,
        content_type=content_type,
        file_bytes=content,
        extraction_method=extraction_method,
    )

    return DocumentUploadResponse(
        document=DocumentResponse.model_validate(document),
        message=message,
    )


@router.get("", response_model=DocumentListResponse, status_code=status.HTTP_200_OK)
async def list_documents(
    page: int = Query(1, ge=1, description="Page number starting at 1"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    status: Optional[str] = Query(None, description="Filter by status (PENDING, PROCESSING, COMPLETED, FAILED)"),
    search: Optional[str] = Query(None, description="Search by original filename"),
    db: AsyncSession = Depends(get_db),
):
    """
    Paginated list of all stored documents with latest extractions.
    """
    skip = (page - 1) * page_size
    documents, total = await document_service.list_documents(
        db=db,
        skip=skip,
        limit=page_size,
        status=status,
        search=search,
    )

    return DocumentListResponse(
        items=[DocumentResponse.model_validate(d) for d in documents],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/metrics/summary", status_code=status.HTTP_200_OK)
async def get_metrics_summary(db: AsyncSession = Depends(get_db)):
    """
    Returns dashboard summary counts (total, pending, processing, completed, failed).
    """
    return await document_service.get_dashboard_metrics(db)


@router.get("/{document_id}", response_model=DocumentResponse, status_code=status.HTTP_200_OK)
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieves a single document with its full extraction history, field confidences, and validation errors.
    """
    document = await document_service.get_document_by_id(db, document_id)
    return DocumentResponse.model_validate(document)
