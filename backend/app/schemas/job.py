from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict


class ProcessingJobResponse(BaseModel):
    id: str
    document_id: str
    task_id: Optional[str] = None
    idempotency_key: str
    extraction_method: str
    status: str
    attempts: int
    max_attempts: int
    request_review: bool
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    heartbeat_at: Optional[datetime] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ProcessingJobListResponse(BaseModel):
    items: List[ProcessingJobResponse]
    total: int
    page: int
    page_size: int


class DocumentStatusResponse(BaseModel):
    document_id: str
    document_status: str
    review_status: str
    needs_review: bool
    error_message: Optional[str] = None
    latest_job: Optional[ProcessingJobResponse] = None
    jobs: List[ProcessingJobResponse] = []


class ProcessingJobMetricsResponse(BaseModel):
    total_jobs: int
    queued: int
    processing: int
    retrying: int
    completed: int
    failed: int
    cancelled: int
    stale_running_jobs: int
    unpublished_outbox_events: int
    average_duration_seconds: Optional[float] = None
    by_extraction_method: Dict[str, int] = {}

