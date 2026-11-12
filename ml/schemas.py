from datetime import date
from decimal import Decimal
from typing import List, Optional
from pydantic import BaseModel, Field


class ExtractedLineItem(BaseModel):
    description: Optional[str] = None
    quantity: Optional[Decimal] = None
    unit_price: Optional[Decimal] = None
    total_price: Optional[Decimal] = None


class ExtractedDocumentData(BaseModel):
    vendor_name: Optional[str] = None
    vendor_address: Optional[str] = None
    document_date: Optional[date] = None
    total_amount: Optional[Decimal] = None
    currency: Optional[str] = None
    tax_amount: Optional[Decimal] = None
    line_items: List[ExtractedLineItem] = Field(default_factory=list)

