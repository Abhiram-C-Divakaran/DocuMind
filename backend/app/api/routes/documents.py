import re
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, Query, Response, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import get_db
from backend.app.config import settings
from backend.app.core.exceptions import FileValidationError
from backend.app.logger import logger
from backend.app.schemas.document import (
    DocumentListResponse,
    DocumentResponse,
    DocumentUploadResponse,
)
from backend.app.schemas.review import PreviewMetaResponse
from backend.app.services.document_service import document_service
from backend.app.services.storage import storage_service

router = APIRouter(prefix="/documents", tags=["Documents"])

SAFE_INLINE_MIME_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
}


def _sanitize_inline_filename(filename: str) -> str:
    leaf = (filename or "document").replace("\\", "/").split("/")[-1]
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", leaf)
    while ".." in cleaned:
        cleaned = cleaned.replace("..", "_")
    cleaned = cleaned.lstrip("._")
    return cleaned[:120] or "document"


@router.post("", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload (PDF, PNG, JPEG)"),
    extraction_method: Optional[str] = Form(
        None, description="Optional extraction method (TESSERACT_BASELINE, LAYOUTLMV3, PLACEHOLDER)"
    ),
    request_review: bool = Form(
        False, description="Explicitly queue this document for human review regardless of confidence"
    ),
    db: AsyncSession = Depends(get_db),
):
    """
    Multipart upload endpoint for receipts and invoices.
    Performs signature/MIME validation, secure UUID storage, OCR/ML extraction,
    and human-in-the-loop review eligibility evaluation.
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
        request_review=request_review,
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
    review_status: Optional[str] = Query(
        None,
        description="Filter by review status (NOT_REQUIRED, PENDING_REVIEW, IN_REVIEW, APPROVED, REJECTED)",
    ),
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
        review_status=review_status,
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


@router.get(
    "/{document_id}/preview/meta",
    response_model=PreviewMetaResponse,
    status_code=status.HTTP_200_OK,
)
async def get_document_preview_meta(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Returns controlled preview metadata (page count & safe preview endpoint URL) without exposing disk paths.
    """
    document = await document_service.get_document_by_id(db, document_id)
    if document.content_type not in SAFE_INLINE_MIME_TYPES:
        raise FileValidationError(
            f"Unsupported content type for inline preview: '{document.content_type}'."
        )

    resolved_path = storage_service.get_file_path(document.stored_filename)
    is_pdf = document.content_type == "application/pdf"
    total_pages = 1
    if is_pdf:
        import fitz

        try:
            with fitz.open(str(resolved_path)) as pdf_doc:
                total_pages = max(1, min(len(pdf_doc), settings.PREVIEW_MAX_PDF_PAGES))
        except Exception as exc:
            raise FileValidationError(f"Unable to inspect PDF pages for preview: {exc}")

    return PreviewMetaResponse(
        document_id=document.id,
        original_filename=document.original_filename,
        content_type=document.content_type,
        is_pdf=is_pdf,
        total_pages=total_pages,
        max_preview_pages=settings.PREVIEW_MAX_PDF_PAGES,
        preview_url=f"{settings.API_V1_STR}/documents/{document.id}/file",
    )


@router.get("/{document_id}/file", status_code=status.HTTP_200_OK)
async def preview_document_file(
    document_id: str,
    page: Optional[int] = Query(
        None,
        ge=1,
        le=50,
        description="Optional 1-based page number to render PDF pages safely as PNG images",
    ),
    db: AsyncSession = Depends(get_db),
):
    """
    Controlled inline document preview endpoint.
    Resolves stored uploads strictly by document ID within storage_service.base_dir,
    validates allowlisted inline formats, prevents path traversal, and bounds PDF rendering.
    """
    document = await document_service.get_document_by_id(db, document_id)
    if document.content_type not in SAFE_INLINE_MIME_TYPES:
        raise FileValidationError(
            f"Unsupported content type for inline preview: '{document.content_type}'."
        )

    resolved_path = storage_service.get_file_path(document.stored_filename)
    safe_name = _sanitize_inline_filename(document.original_filename)
    security_headers = {
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "private, max-age=300",
    }

    if document.content_type == "application/pdf" and page is not None:
        import fitz

        max_allowed = settings.PREVIEW_MAX_PDF_PAGES
        if page > max_allowed:
            raise FileValidationError(
                f"Requested page {page} exceeds preview limit ({max_allowed} pages)."
            )

        try:
            with fitz.open(str(resolved_path)) as pdf_doc:
                page_count = len(pdf_doc)
                if page_count == 0:
                    raise FileValidationError("PDF contains zero renderable pages.")
                if page > page_count:
                    raise FileValidationError(
                        f"Requested page {page} exceeds document page count ({page_count})."
                    )
                dpi = min(max(72, settings.PREVIEW_RENDER_DPI), 200)
                zoom = dpi / 72.0
                pix = pdf_doc[page - 1].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                png_bytes = pix.tobytes("png")
        except FileValidationError:
            raise
        except Exception as exc:
            raise FileValidationError(f"Failed to render PDF preview page {page}: {exc}")

        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={
                **security_headers,
                "Content-Disposition": f'inline; filename="{safe_name}_p{page}.png"',
                "X-DocuMind-Total-Pages": str(min(page_count, max_allowed)),
                "X-DocuMind-Rendered-Page": str(page),
            },
        )

    return FileResponse(
        path=str(resolved_path),
        media_type=document.content_type,
        headers={
            **security_headers,
            "Content-Disposition": f'inline; filename="{safe_name}"',
        },
    )
