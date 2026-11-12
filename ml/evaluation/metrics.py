from typing import Any, Dict, List, Optional
import re
try:
    import torch
except (ImportError, OSError):
    torch = None  # type: ignore
import numpy as np
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from seqeval.scheme import IOB2


def normalize_entity_value(val: Optional[str]) -> str:
    """Normalizes an entity string for exact match comparison."""
    if val is None:
        return ""
    # Lowercase, replace non-alphanumeric chars with spaces, collapse spaces
    cleaned = re.sub(r"[^\w\s]", " ", val.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def compute_seqeval_metrics(
    predictions: List[List[str]],
    references: List[List[str]],
) -> Dict[str, Any]:
    """
    Computes seqeval entity-level precision, recall, and F1 scores.

    Args:
        predictions: List of sequences of BIO tags (e.g. [["B-COMPANY", "I-COMPANY", "O"], ...])
        references: List of gold sequences of BIO tags.

    Returns:
        Dict with overall metrics and per-entity metrics.
    """
    if not predictions or not references:
        return {
            "overall_precision": 0.0,
            "overall_recall": 0.0,
            "overall_f1": 0.0,
            "entities": {},
        }

    p = float(precision_score(references, predictions, mode="strict", scheme=IOB2, zero_division=0))
    r = float(recall_score(references, predictions, mode="strict", scheme=IOB2, zero_division=0))
    f1 = float(f1_score(references, predictions, mode="strict", scheme=IOB2, zero_division=0))

    rep_dict = classification_report(
        references, predictions, mode="strict", scheme=IOB2, output_dict=True, zero_division=0
    )

    entity_metrics = {}
    for entity in ["COMPANY", "DATE", "ADDRESS", "TOTAL"]:
        if entity in rep_dict:
            entity_metrics[entity] = {
                "precision": float(rep_dict[entity]["precision"]),
                "recall": float(rep_dict[entity]["recall"]),
                "f1": float(rep_dict[entity]["f1-score"]),
                "support": int(rep_dict[entity]["support"]),
            }
        else:
            entity_metrics[entity] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "support": 0,
            }

    return {
        "overall_precision": p,
        "overall_recall": r,
        "overall_f1": f1,
        "entities": entity_metrics,
    }


def compute_document_metrics(
    predictions: List[Dict[str, Optional[str]]],
    references: List[Dict[str, Optional[str]]],
) -> Dict[str, Any]:
    """
    Computes field-level exact match and document-level exact match.

    Args:
        predictions: List of dicts with keys 'company', 'date', 'address', 'total'
        references: List of dicts with corresponding ground truth strings

    Returns:
        Dict with field exact matches, document exact match rate, and count.
    """
    n = len(references)
    if n == 0:
        return {
            "num_documents": 0,
            "document_exact_match": 0.0,
            "field_exact_match": {},
        }

    fields = ["company", "date", "address", "total"]
    field_matches = {f: 0 for f in fields}
    doc_matches = 0

    for pred, ref in zip(predictions, references):
        all_matched = True
        for f in fields:
            pred_val = normalize_entity_value(pred.get(f))
            ref_val = normalize_entity_value(ref.get(f))
            
            # Exact match after standard normalization
            if pred_val == ref_val and ref_val != "":
                field_matches[f] += 1
            elif pred_val == "" and ref_val == "":
                # Both empty
                field_matches[f] += 1
            else:
                all_matched = False

        if all_matched:
            doc_matches += 1

    return {
        "num_documents": n,
        "document_exact_match": doc_matches / n,
        "field_exact_match": {
            f: field_matches[f] / n for f in fields
        },
    }


def compute_model_footprint(model: Any) -> Dict[str, Any]:
    """Computes total and trainable parameters and memory footprint."""
    if hasattr(model, "parameters"):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        total_params = 125000000  # Default LayoutLMv3-base parameter count
        trainable_params = total_params

    # Estimate param memory in MB (assuming float32: 4 bytes per param)
    mem_mb = (total_params * 4) / (1024 * 1024)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "estimated_parameter_size_mb": round(mem_mb, 2),
    }
