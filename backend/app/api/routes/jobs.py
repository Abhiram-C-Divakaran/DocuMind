from typing import Optional
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import get_current_user, get_db, require_roles
from backend.app.models.user import User, UserRole
from backend.app.schemas.job import (
    ProcessingJobListResponse,
    ProcessingJobMetricsResponse,
    ProcessingJobResponse,
)
from backend.app.services.document_service import document_service
from backend.app.worker.tasks import recover_stale_jobs_and_outbox_sync

router = APIRouter()


@router.get("/metrics", response_model=ProcessingJobMetricsResponse)
async def get_jobs_metrics(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Returns DB-backed processing job metrics and outbox/stale counts."""
    return await document_service.get_job_metrics(db, user=current_user)


@router.get("", response_model=ProcessingJobListResponse)
async def list_jobs(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by job status"),
    document_id: Optional[str] = Query(None, description="Filter by document ID"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    skip = (page - 1) * page_size
    items, total = await document_service.list_jobs(
        db=db,
        skip=skip,
        limit=page_size,
        status=status_filter,
        document_id=document_id,
        user=current_user,
    )
    return ProcessingJobListResponse(
        items=[ProcessingJobResponse.model_validate(item) for item in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("/recover", status_code=status.HTTP_200_OK)
async def trigger_job_recovery(
    current_user: User = Depends(require_roles(UserRole.ADMIN)),
):
    """Administrator endpoint to recover stale PROCESSING jobs and unpublished outbox events."""
    return recover_stale_jobs_and_outbox_sync()


@router.get("/{job_id}", response_model=ProcessingJobResponse)
async def get_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    job = await document_service.get_job_by_id(db, job_id, user=current_user)
    return ProcessingJobResponse.model_validate(job)

