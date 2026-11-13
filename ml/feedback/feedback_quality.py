from typing import Any, Dict, List, Set

from ml.data.annotation_alignment import find_best_matching_span
from ml.feedback.feedback_validator import ValidatedFeedbackItem

ENTITY_FIELD_TO_SROIE = {
    "vendor_name": "COMPANY",
    "vendor_address": "ADDRESS",
    "document_date": "DATE",
    "total_amount": "TOTAL",
}


def assess_feedback_quality(item: ValidatedFeedbackItem) -> Dict[str, Any]:
    """
    Evaluates quality & LayoutLMv3 annotation readiness for an approved receipt review.

    Crucially, document-level corrected strings are NEVER blindly assumed to be valid
    BIO token labels. Token spans are only marked aligned if OCR words exist and match
    unambiguously; unaligned corrections are flagged as NEEDS_MANUAL_ANNOTATION.
    """
    rec = item.record
    corrected = item.validated_corrected_values

    words: List[str] = []
    if rec.ocr_tokens:
        words = [str(t.get("text", "")).strip() for t in rec.ocr_tokens if str(t.get("text", "")).strip()]
    elif rec.raw_text:
        words = [w for w in rec.raw_text.split() if w.strip()]

    has_token_boxes = bool(
        rec.ocr_tokens and all("norm_bbox" in t or "bbox" in t for t in rec.ocr_tokens)
    )

    aligned_spans: Dict[str, Any] = {}
    unaligned_fields: List[str] = []
    occupied: Set[int] = set()

    non_empty_target_fields = 0
    for doc_field, entity_type in ENTITY_FIELD_TO_SROIE.items():
        val = corrected.get(doc_field)
        if val is None or not str(val).strip():
            continue
        non_empty_target_fields += 1

        if not words:
            unaligned_fields.append(doc_field)
            aligned_spans[doc_field] = {
                "entity": entity_type,
                "matched": False,
                "reason": "no_ocr_tokens_available",
            }
            continue

        span = find_best_matching_span(words, str(val), occupied)
        if span is not None:
            start, end = span
            for idx in range(start, end):
                occupied.add(idx)
            aligned_spans[doc_field] = {
                "entity": entity_type,
                "matched": True,
                "token_span": [start, end],
                "matched_tokens": words[start:end],
                "has_bboxes": has_token_boxes,
            }
        else:
            unaligned_fields.append(doc_field)
            aligned_spans[doc_field] = {
                "entity": entity_type,
                "matched": False,
                "reason": "corrected_value_not_matched_in_ocr_tokens",
            }

    if non_empty_target_fields == 0:
        alignment_status = "NEEDS_MANUAL_ANNOTATION"
    elif not unaligned_fields and has_token_boxes:
        alignment_status = "ALIGNED_CONFIDENT"
    else:
        alignment_status = "NEEDS_MANUAL_ANNOTATION"

    return {
        "correction_count": len(rec.changed_fields),
        "changed_fields": list(rec.changed_fields),
        "has_ocr_tokens": len(words) > 0,
        "has_token_bboxes": has_token_boxes,
        "alignment_status": alignment_status,
        "unaligned_fields": unaligned_fields,
        "aligned_token_spans": aligned_spans,
        "requires_token_annotation_verification": True,
    }

