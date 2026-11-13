from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field
from backend.app.schemas.extraction import ExtractionResultResponse
from backend.app.schemas.job import ProcessingJobResponse


class DocumentBase(BaseModel):
    original_filename: str
    content_type: str
    file_size_bytes: int


class DocumentResponse(DocumentBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    owner_id: Optional[str] = None
    status: str
    review_status: str = "NOT_REQUIRED"
    needs_review: bool = False
    review_reasons: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    extractions: List[ExtractionResultResponse] = Field(default_factory=list)
    latest_job: Optional[ProcessingJobResponse] = None

    @property
    def latest_extraction(self) -> Optional[ExtractionResultResponse]:
        if self.extractions:
            return self.extractions[0]
        return None


class DocumentListResponse(BaseModel):
    items: List[DocumentResponse]
    total: int
    page: int
    page_size: int


class DocumentUploadResponse(BaseModel):
    document: DocumentResponse
    job: Optional[ProcessingJobResponse] = None
    message: str


