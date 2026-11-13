from typing import Optional
from fastapi import APIRouter, Body, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import get_current_user, get_db, require_roles
from backend.app.config import settings
from backend.app.models.user import User, UserRole
from backend.app.schemas.review import (
    DocumentReviewResponse,
    ReviewDecisionRequest,
    ReviewHistoryResponse,
    ReviewListResponse,
    ReviewMetricsResponse,
    ReviewUpdateRequest,
)
from backend.app.services.review_service import review_service

router = APIRouter(tags=["Human Review"])


@router.get("/reviews", response_model=ReviewListResponse, status_code=status.HTTP_200_OK)
async def list_reviews_queue(
    page: int = Query(1, ge=1, description="Page number starting at 1"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    review_status: Optional[str] = Query(
        None,
        description="Filter by review status (ALL, NEEDS_REVIEW, NOT_REQUIRED, PENDING_REVIEW, IN_REVIEW, APPROVED, REJECTED)",
    ),
    extraction_method: Optional[str] = Query(
        None, description="Filter by extraction method (TESSERACT_BASELINE, LAYOUTLMV3, PLACEHOLDER)"
    ),
    confidence_max: Optional[float] = Query(
        None, ge=0.0, le=1.0, description="Upper heuristic confidence bound"
    ),
    missing_field: Optional[str] = Query(
        None, description="Filter documents missing a critical field (vendor_name, document_date, total_amount)"
    ),
    search: Optional[str] = Query(None, description="Search by filename"),
    sort_by: str = Query(
        "confidence_asc",
        description="Sort order (confidence_asc, confidence_desc, created_at_desc, created_at_asc, updated_at_desc)",
    ),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await review_service.list_reviews_queue(
        db=db,
        page=page,
        page_size=page_size,
        review_status=review_status,
        extraction_method=extraction_method,
        confidence_max=confidence_max,
        missing_field=missing_field,
        search=search,
        sort_by=sort_by,
        user=current_user,
    )


@router.get("/reviews/metrics", response_model=ReviewMetricsResponse, status_code=status.HTTP_200_OK)
async def get_review_metrics(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await review_service.get_review_metrics(db, user=current_user)


@router.post("/reviews/feedback-export", status_code=status.HTTP_200_OK)
async def export_approved_feedback_admin(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(UserRole.ADMIN)),
):
    """Administrator-only endpoint to export approved human corrections as active-learning JSONL."""
    from ml.feedback.export_annotations import export_feedback_from_session

    summary = await export_feedback_from_session(
        db=db,
        upload_dir=settings.UPLOAD_DIR,
    )
    return {
        "status": "exported",
        "exported_count": summary.get("exported_count", 0),
        "aligned_confident": summary.get("aligned_confident", 0),
        "needs_manual_annotation": summary.get("needs_manual_annotation", 0),
        "excluded_status_counts": summary.get("excluded_status_counts", {}),
    }


@router.get("/documents/{document_id}/review", response_model=DocumentReviewResponse, status_code=status.HTTP_200_OK)
async def get_document_review(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await review_service.get_document_review(db, document_id, user=current_user)


@router.patch("/documents/{document_id}/review", response_model=DocumentReviewResponse, status_code=status.HTTP_200_OK)
async def update_document_review_draft(
    document_id: str,
    payload: ReviewUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.REVIEWER)),
):
    return await review_service.save_review_draft(db, document_id, payload, user=current_user)


@router.post("/documents/{document_id}/review/approve", response_model=DocumentReviewResponse, status_code=status.HTTP_200_OK)
async def approve_document_review(
    document_id: str,
    payload: Optional[ReviewDecisionRequest] = Body(default=None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.REVIEWER)),
):
    return await review_service.approve_review(db, document_id, payload or ReviewDecisionRequest(), user=current_user)


@router.post("/documents/{document_id}/review/reject", response_model=DocumentReviewResponse, status_code=status.HTTP_200_OK)
async def reject_document_review(
    document_id: str,
    payload: Optional[ReviewDecisionRequest] = Body(default=None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.REVIEWER)),
):
    return await review_service.reject_review(db, document_id, payload or ReviewDecisionRequest(), user=current_user)


@router.post("/documents/{document_id}/review/reopen", response_model=DocumentReviewResponse, status_code=status.HTTP_200_OK)
async def reopen_document_review(
    document_id: str,
    payload: Optional[ReviewDecisionRequest] = Body(default=None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.REVIEWER)),
):
    return await review_service.reopen_review(db, document_id, payload or ReviewDecisionRequest(), user=current_user)


@router.get("/documents/{document_id}/review/history", response_model=ReviewHistoryResponse, status_code=status.HTTP_200_OK)
async def get_document_review_history(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await review_service.get_review_history(db, document_id, user=current_user)
