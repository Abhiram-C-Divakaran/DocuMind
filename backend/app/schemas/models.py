from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class ExtractionMethodReadiness(BaseModel):
    method: str
    label: str
    available: bool
    is_default: bool
    is_production_ready: bool
    badge: str
    description: str
    unavailable_reason: Optional[str] = None
    checkpoint_identity: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ModelReadinessResponse(BaseModel):
    default_method: str
    ml_inference_enabled: bool
    torch_runtime_available: bool
    tesseract_available: bool
    tesseract_version: Optional[str] = None
    tesseract_message: Optional[str] = None
    layoutlmv3_ready: bool
    layoutlmv3_is_production_ready: bool
    layoutlmv3_reason: Optional[str] = None
    checkpoint_identity: Optional[str] = None
    checkpoint_size_mb: Optional[float] = None
    checkpoint_metadata: Optional[Dict[str, Any]] = None
    methods: List[ExtractionMethodReadiness]

