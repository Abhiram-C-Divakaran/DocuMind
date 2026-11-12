from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ExtractedField:
    value: Any
    confidence: float
    raw_text: Optional[str] = None
    evidence: Optional[str] = None
    norm_bbox: Optional[Tuple[int, int, int, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReceiptExtractionResult:
    vendor_name: ExtractedField
    vendor_address: ExtractedField
    document_date: ExtractedField
    total_amount: ExtractedField
    currency: ExtractedField
    tax_amount: ExtractedField
    line_items: List[Dict[str, Any]]
    overall_confidence: float
    field_confidences: Dict[str, float]
    validation_errors: List[str]
    raw_ocr_text: str
    needs_manual_review: bool
    evidence_map: Dict[str, Any]
    metadata: Dict[str, Any]
