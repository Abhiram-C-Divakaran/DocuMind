from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class LineItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    description: Optional[str] = Field(None, description="Description of the item or service")
    quantity: Optional[Decimal] = Field(None, description="Quantity purchased", decimal_places=3)
    unit_price: Optional[Decimal] = Field(None, description="Unit price per item", decimal_places=2)
    total_price: Optional[Decimal] = Field(None, description="Total line price", decimal_places=2)


class StructuredExtractionData(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    vendor_name: Optional[str] = Field(None, description="Extracted vendor or merchant name")
    vendor_address: Optional[str] = Field(None, description="Physical address or contact info")
    document_date: Optional[date] = Field(None, description="Date of document/invoice/receipt")
    total_amount: Optional[Decimal] = Field(None, description="Total amount due or paid", decimal_places=2)
    currency: Optional[str] = Field(None, description="ISO currency code e.g. USD, EUR, INR")
    tax_amount: Optional[Decimal] = Field(None, description="Tax or VAT amount", decimal_places=2)
    line_items: Optional[List[LineItem]] = Field(default_factory=list, description="Extracted itemized line items")


class ExtractionResultBase(BaseModel):
    extraction_method: str = Field(..., description="Method or model used (e.g. PLACEHOLDER, TESSERACT_BASELINE, LAYOUTLMV3)")
    structured_data: Optional[StructuredExtractionData] = Field(None, description="Parsed structured fields")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Overall confidence score")
    field_confidences: Optional[Dict[str, float]] = Field(default=None, description="Field-level confidence estimates")
    raw_text: Optional[str] = Field(default=None, description="Raw OCR text extracted from document")
    validation_errors: List[str] = Field(default_factory=list, description="Validation issues detected in extracted data")
    extraction_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Detailed OCR and evidence metadata")


class ExtractionResultResponse(ExtractionResultBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    created_at: datetime
