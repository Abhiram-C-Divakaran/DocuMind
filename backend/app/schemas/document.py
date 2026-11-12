from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field
from backend.app.schemas.extraction import ExtractionResultResponse


class DocumentBase(BaseModel):
    original_filename: str
    content_type: str
    file_size_bytes: int


class DocumentResponse(DocumentBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    extractions: List[ExtractionResultResponse] = Field(default_factory=list)

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
    message: str

