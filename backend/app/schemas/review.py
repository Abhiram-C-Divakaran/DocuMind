import re
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _clean_optional_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _parse_decimal_strict(v: Any, field_name: str, places: int = 2) -> Optional[Decimal]:
    if v is None:
        return None
    if isinstance(v, str) and not v.strip():
        return None
    if isinstance(v, bool):
        raise ValueError(f"{field_name} must be a valid decimal number.")
    try:
        dec = Decimal(str(v).strip()) if isinstance(v, str) else Decimal(str(v))
    except (InvalidOperation, ValueError):
        raise ValueError(f"{field_name} has invalid numeric format: '{v}'.")
    if not dec.is_finite():
        raise ValueError(f"{field_name} must be a finite decimal number.")
    quant = Decimal("1") if places == 0 else Decimal(f"1.{'0' * places}")
    return dec.quantize(quant, rounding=ROUND_HALF_UP)


class LineItemCorrectionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)

    description: Optional[str] = Field(None, max_length=300)
    quantity: Optional[Decimal] = Field(None)
    unit_price: Optional[Decimal] = Field(None)
    total_price: Optional[Decimal] = Field(None)

    @field_validator("description", mode="before")
    @classmethod
    def normalize_desc(cls, v: Any) -> Optional[str]:
        return _clean_optional_str(v)

    @field_validator("quantity", mode="before")
    @classmethod
    def validate_qty(cls, v: Any) -> Optional[Decimal]:
        dec = _parse_decimal_strict(v, "quantity", places=3)
        if dec is not None and dec <= Decimal("0"):
            raise ValueError("Line item quantity must be positive.")
        return dec

    @field_validator("unit_price", "total_price", mode="before")
    @classmethod
    def validate_prices(cls, v: Any, info: Any) -> Optional[Decimal]:
        dec = _parse_decimal_strict(v, info.field_name, places=2)
        if dec is not None and dec < Decimal("0"):
            raise ValueError(f"Line item {info.field_name} cannot be negative.")
        return dec


class ReviewCorrectionPayload(BaseModel):
    """Strictly validated schema for human-in-the-loop receipt field corrections."""

    model_config = ConfigDict(extra="forbid", from_attributes=True)

    vendor_name: Optional[str] = Field(None, max_length=255)
    vendor_address: Optional[str] = Field(None, max_length=500)
    document_date: Optional[date] = Field(None)
    total_amount: Optional[Decimal] = Field(None)
    currency: Optional[str] = Field(None, max_length=10)
    tax_amount: Optional[Decimal] = Field(None)
    line_items: Optional[List[LineItemCorrectionPayload]] = Field(default_factory=list)

    @field_validator("vendor_name", "vendor_address", mode="before")
    @classmethod
    def normalize_text_fields(cls, v: Any) -> Optional[str]:
        return _clean_optional_str(v)

    @field_validator("document_date", mode="before")
    @classmethod
    def validate_date_format(cls, v: Any) -> Optional[date]:
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        if isinstance(v, datetime):
            parsed = v.date()
        elif isinstance(v, date):
            parsed = v
        elif isinstance(v, str):
            raw = v.strip()
            try:
                parsed = date.fromisoformat(raw)
            except ValueError:
                raise ValueError(
                    f"Invalid date format '{raw}'. Use ISO format YYYY-MM-DD."
                )
        else:
            raise ValueError(f"Invalid document_date value: {v}")

        if parsed < date(1990, 1, 1):
            raise ValueError("Document date cannot be earlier than 1990-01-01.")
        if parsed > date.today() + timedelta(days=1):
            raise ValueError("Document date cannot be in the future.")
        return parsed

    @field_validator("total_amount", mode="before")
    @classmethod
    def validate_total_amount(cls, v: Any) -> Optional[Decimal]:
        dec = _parse_decimal_strict(v, "total_amount", places=2)
        if dec is not None:
            if dec <= Decimal("0"):
                raise ValueError("Total amount must be greater than zero.")
            if dec > Decimal("1000000000.00"):
                raise ValueError("Total amount exceeds supported range.")
        return dec

    @field_validator("tax_amount", mode="before")
    @classmethod
    def validate_tax_amount(cls, v: Any) -> Optional[Decimal]:
        dec = _parse_decimal_strict(v, "tax_amount", places=2)
        if dec is not None:
            if dec < Decimal("0"):
                raise ValueError("Tax amount cannot be negative.")
            if dec > Decimal("1000000000.00"):
                raise ValueError("Tax amount exceeds supported range.")
        return dec

    @field_validator("currency", mode="before")
    @classmethod
    def validate_currency(cls, v: Any) -> Optional[str]:
        s = _clean_optional_str(v)
        if s is None:
            return None
        upper = s.upper()
        if not re.match(r"^[A-Z]{3}$", upper):
            raise ValueError(
                f"Invalid currency code '{s}'. Expected a 3-letter ISO 4217 code (e.g. USD, EUR, MYR, INR)."
            )
        return upper

    @field_validator("line_items", mode="before")
    @classmethod
    def normalize_line_items(cls, v: Any) -> List[Any]:
        if v is None:
            return []
        return v

    @model_validator(mode="after")
    def validate_financial_consistency(self) -> "ReviewCorrectionPayload":
        if self.tax_amount is not None and self.total_amount is not None:
            if self.tax_amount > self.total_amount:
                raise ValueError(
                    f"Tax amount ({self.tax_amount}) cannot exceed total amount ({self.total_amount})."
                )
        return self

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "vendor_name": self.vendor_name,
            "vendor_address": self.vendor_address,
            "document_date": self.document_date.isoformat() if self.document_date else None,
            "total_amount": str(self.total_amount) if self.total_amount is not None else None,
            "currency": self.currency,
            "tax_amount": str(self.tax_amount) if self.tax_amount is not None else None,
            "line_items": [
                {
                    "description": li.description,
                    "quantity": str(li.quantity) if li.quantity is not None else None,
                    "unit_price": str(li.unit_price) if li.unit_price is not None else None,
                    "total_price": str(li.total_price) if li.total_price is not None else None,
                }
                for li in (self.line_items or [])
            ],
        }


class ReviewUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corrected_data: ReviewCorrectionPayload
    reviewer_notes: Optional[str] = Field(None, max_length=2000)
    reviewer_identifier: Optional[str] = Field(None, max_length=120)
    expected_version: Optional[int] = Field(None, ge=0)


class ReviewDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corrected_data: Optional[ReviewCorrectionPayload] = None
    reviewer_notes: Optional[str] = Field(None, max_length=2000)
    reviewer_identifier: Optional[str] = Field(None, max_length=120)
    expected_version: Optional[int] = Field(None, ge=0)


class FieldCorrectionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    review_id: str
    field_name: str
    original_value: Optional[str] = None
    corrected_value: Optional[str] = None
    created_at: datetime


class ReviewAuditEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    review_id: str
    action: str
    from_status: Optional[str] = None
    to_status: Optional[str] = None
    changed_fields: List[str] = Field(default_factory=list)
    actor_identifier: str
    notes: Optional[str] = None
    event_timestamp: datetime


class DocumentReviewResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: Optional[str] = None
    document_id: str
    extraction_result_id: Optional[str] = None
    review_status: str
    document_status: str
    extraction_method: Optional[str] = None
    confidence: Optional[float] = None
    field_confidences: Optional[Dict[str, float]] = None
    validation_errors: List[str] = Field(default_factory=list)
    review_reasons: List[str] = Field(default_factory=list)
    missing_critical_fields: List[str] = Field(default_factory=list)
    original_data: Optional[Dict[str, Any]] = None
    corrected_data: Optional[Dict[str, Any]] = None
    changed_fields: List[str] = Field(default_factory=list)
    raw_text: Optional[str] = None
    reviewer_identifier: str = "local-anonymous"
    reviewer_notes: Optional[str] = None
    version: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    approved_at: Optional[datetime] = None


class ReviewQueueItem(BaseModel):
    document_id: str
    review_id: Optional[str] = None
    original_filename: str
    content_type: str
    file_size_bytes: int
    document_status: str
    review_status: str
    needs_review: bool
    review_reasons: List[str] = Field(default_factory=list)
    missing_critical_fields: List[str] = Field(default_factory=list)
    extraction_method: Optional[str] = None
    confidence: Optional[float] = None
    validation_errors_count: int = 0
    changed_fields: List[str] = Field(default_factory=list)
    reviewer_identifier: str = "local-anonymous"
    version: int = 0
    created_at: datetime
    updated_at: datetime
    approved_at: Optional[datetime] = None


class ReviewListResponse(BaseModel):
    items: List[ReviewQueueItem]
    total: int
    page: int
    page_size: int


class ReviewHistoryResponse(BaseModel):
    document_id: str
    review_id: Optional[str] = None
    review_status: str
    field_corrections: List[FieldCorrectionResponse] = Field(default_factory=list)
    audit_events: List[ReviewAuditEventResponse] = Field(default_factory=list)


class ReviewMetricsResponse(BaseModel):
    total_documents: int
    not_required: int
    pending_review: int
    in_review: int
    awaiting_review: int
    approved: int
    rejected: int
    avg_corrected_fields: float
    review_completion_rate: float
    eligible_for_feedback_export: int
    review_confidence_threshold: float


class PreviewMetaResponse(BaseModel):
    document_id: str
    original_filename: str
    content_type: str
    is_pdf: bool
    total_pages: int
    max_preview_pages: int
    preview_url: str

