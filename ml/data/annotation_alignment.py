import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# Canonical BIO Labels for DocuMind SROIE Task
LABEL_LIST: List[str] = [
    "O",
    "B-COMPANY",
    "I-COMPANY",
    "B-DATE",
    "I-DATE",
    "B-ADDRESS",
    "I-ADDRESS",
    "B-TOTAL",
    "I-TOTAL",
]

ID2LABEL: Dict[int, str] = {i: label for i, label in enumerate(LABEL_LIST)}
LABEL2ID: Dict[str, int] = {label: i for i, label in enumerate(LABEL_LIST)}

ENTITIES = ["COMPANY", "DATE", "ADDRESS", "TOTAL"]


@dataclass
class AlignedReceipt:
    receipt_id: str
    words: List[str]
    boxes: List[List[int]]  # Normalized 0-1000 coordinates [x0, y0, x1, y1]
    labels: List[str]  # BIO string tags
    ner_tags: List[int]  # Integer tag IDs
    ground_truth: Dict[str, str]
    alignment_stats: Dict[str, bool]  # Whether each entity was successfully aligned
    unmatched_fields: List[str] = field(default_factory=list)
    image_path: Optional[str] = None
    width: int = 1000
    height: int = 1000
    image_sha256: Optional[str] = None
    is_synthetic: bool = False
    source: str = "unknown"
    alignment_audit: Dict[str, Any] = field(default_factory=dict)


def normalize_text(text: str) -> str:
    """Normalizes text for fuzzy whitespace/punctuation matching."""
    if not text:
        return ""
    cleaned = re.sub(r"[^\w\d]", "", text.lower())
    return cleaned


def split_segment_to_words_and_boxes(
    segment_text: str, segment_box: List[int]
) -> Tuple[List[str], List[List[int]]]:
    """
    Splits a multi-word OCR line/segment into word tokens and linearly interpolates
    their horizontal bounding boxes within [x0, y0, x1, y1] in 0-1000 coordinates.
    """
    stripped = segment_text.strip()
    if not stripped:
        return [], []

    x0, y0, x1, y1 = [int(c) for c in segment_box]
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)

    words_with_spans = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", stripped)]
    if len(words_with_spans) <= 1:
        return [stripped], [[x0, y0, x1, y1]]

    total_len = max(1, len(stripped))
    box_w = max(1, x1 - x0)

    out_words: List[str] = []
    out_boxes: List[List[int]] = []
    for w, start_c, end_c in words_with_spans:
        wx0 = max(0, min(1000, int(x0 + (box_w * start_c) / total_len)))
        wx1 = max(wx0, min(1000, int(x0 + (box_w * end_c) / total_len)))
        out_words.append(w)
        out_boxes.append([wx0, y0, wx1, y1])

    return out_words, out_boxes


def find_best_matching_span(
    words: List[str], target: str, occupied_indices: Set[int]
) -> Optional[Tuple[int, int]]:
    """
    Finds a consecutive subsequence of words [start, end) that best matches target.
    Returns (start_idx, end_idx) or None if no confident match is found.
    Never overwrites already-occupied entity tokens and requires >= 0.85 similarity overlap.
    """
    if not target or not words:
        return None

    clean_target = normalize_text(target)
    if not clean_target:
        return None

    target_tokens = target.split()
    n_tokens = len(target_tokens)
    n_words = len(words)

    min_len = max(1, n_tokens - 1)
    max_len = min(n_words, n_tokens + 5)

    best_span: Optional[Tuple[int, int]] = None
    best_overlap = 0.0

    for length in range(min_len, max_len + 1):
        for i in range(n_words - length + 1):
            span_indices = set(range(i, i + length))
            if span_indices & occupied_indices:
                continue

            span_text = " ".join(words[i : i + length])
            clean_span = normalize_text(span_text)
            if not clean_span:
                continue

            if clean_span == clean_target:
                return (i, i + length)

            if clean_target in clean_span or clean_span in clean_target:
                overlap = len(clean_span) / max(len(clean_target), 1)
                if 0.85 <= overlap <= 1.15:
                    # Prefer score closest to 1.0
                    closeness = 1.0 - abs(1.0 - overlap)
                    if closeness > best_overlap:
                        best_overlap = closeness
                        best_span = (i, i + length)

    if best_span and best_overlap >= 0.85:
        return best_span

    # Single token exact match fallback (crucial for TOTAL or DATE)
    for i, word in enumerate(words):
        if i in occupied_indices:
            continue
        clean_word = normalize_text(word)
        if clean_word == clean_target:
            return (i, i + 1)

    return None


def align_receipt_annotations(
    receipt_id: str,
    words: List[str],
    boxes: List[List[int]],
    ground_truth: Dict[str, str],
    image_path: Optional[str] = None,
    width: int = 1000,
    height: int = 1000,
    image_sha256: Optional[str] = None,
    is_synthetic: bool = False,
    source: str = "unknown",
    pre_assigned_labels: Optional[List[str]] = None,
) -> AlignedReceipt:
    """
    Aligns SROIE ground truth fields (COMPANY, DATE, ADDRESS, TOTAL) with OCR word tokens.
    If pre_assigned_labels (e.g., from verified BIO/segment annotations) are provided and match len(words),
    validates them directly and audits alignment against ground_truth.
    Otherwise runs span matching and explicitly records unmatched/ambiguous entities.
    """
    n_words = len(words)
    alignment_stats: Dict[str, bool] = {}
    unmatched: List[str] = []
    audit_details: Dict[str, Any] = {}

    key_mapping = {
        "company": "COMPANY",
        "date": "DATE",
        "address": "ADDRESS",
        "total": "TOTAL",
    }

    if pre_assigned_labels and len(pre_assigned_labels) == n_words:
        labels = [l if l in LABEL2ID else "O" for l in pre_assigned_labels]
        for key, canonical_entity in key_mapping.items():
            raw_val = str(ground_truth.get(key) or ground_truth.get(canonical_entity) or "").strip()
            has_tag = any(l in (f"B-{canonical_entity}", f"I-{canonical_entity}") for l in labels)
            matched_words = [
                w for w, l in zip(words, labels)
                if l in (f"B-{canonical_entity}", f"I-{canonical_entity}")
            ]
            alignment_stats[canonical_entity] = has_tag
            if not has_tag:
                unmatched.append(canonical_entity)
                audit_details[canonical_entity] = {
                    "matched": False,
                    "reason": "missing_ground_truth" if not raw_val else "no_labeled_tokens",
                    "ground_truth": raw_val,
                }
            else:
                audit_details[canonical_entity] = {
                    "matched": True,
                    "method": "dataset_segment_labels",
                    "matched_text": " ".join(matched_words),
                    "ground_truth": raw_val,
                }
    else:
        labels = ["O"] * n_words
        occupied: Set[int] = set()
        # Match shortest/most specific fields first to avoid greedy absorption
        match_order = ["total", "date", "company", "address"]

        for key in match_order:
            canonical_entity = key_mapping[key]
            raw_val = ground_truth.get(key) or ground_truth.get(canonical_entity) or ""
            raw_val = str(raw_val).strip()

            if not raw_val:
                alignment_stats[canonical_entity] = False
                unmatched.append(canonical_entity)
                audit_details[canonical_entity] = {
                    "matched": False,
                    "reason": "missing_ground_truth",
                    "ground_truth": "",
                }
                continue

            span = find_best_matching_span(words, raw_val, occupied)
            if span:
                start, end = span
                labels[start] = f"B-{canonical_entity}"
                occupied.add(start)
                for k in range(start + 1, end):
                    labels[k] = f"I-{canonical_entity}"
                    occupied.add(k)
                alignment_stats[canonical_entity] = True
                audit_details[canonical_entity] = {
                    "matched": True,
                    "method": "span_alignment",
                    "span": [start, end],
                    "matched_text": " ".join(words[start:end]),
                    "ground_truth": raw_val,
                }
            else:
                alignment_stats[canonical_entity] = False
                unmatched.append(canonical_entity)
                audit_details[canonical_entity] = {
                    "matched": False,
                    "reason": "unmatched_in_ocr_tokens",
                    "ground_truth": raw_val,
                }

    ner_tags = [LABEL2ID[l] for l in labels]

    return AlignedReceipt(
        receipt_id=receipt_id,
        words=words,
        boxes=boxes,
        labels=labels,
        ner_tags=ner_tags,
        ground_truth=ground_truth,
        alignment_stats=alignment_stats,
        unmatched_fields=unmatched,
        image_path=image_path,
        width=width,
        height=height,
        image_sha256=image_sha256,
        is_synthetic=is_synthetic,
        source=source,
        alignment_audit=audit_details,
    )
