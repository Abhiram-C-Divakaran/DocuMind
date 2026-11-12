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
    boxes: List[List[int]]  # Normalized 0-1000 coordinates
    labels: List[str]  # BIO string tags
    ner_tags: List[int]  # Integer tag IDs
    ground_truth: Dict[str, str]
    alignment_stats: Dict[str, bool]  # Whether each entity was successfully aligned
    unmatched_fields: List[str] = field(default_factory=list)
    image_path: Optional[str] = None
    width: int = 1000
    height: int = 1000


def normalize_text(text: str) -> str:
    """Normalizes text for fuzzy whitespace/punctuation matching."""
    if not text:
        return ""
    # Strip whitespace, convert to lower, normalize spaces and punctuation
    cleaned = re.sub(r"[^\w\d]", "", text.lower())
    return cleaned


def find_best_matching_span(
    words: List[str], target: str, occupied_indices: Set[int]
) -> Optional[Tuple[int, int]]:
    """
    Finds a consecutive subsequence of words [start, end) that best matches target.
    Returns:
        (start_idx, end_idx) or None if no confident match is found.
    """
    if not target or not words:
        return None

    clean_target = normalize_text(target)
    if not clean_target:
        return None

    target_tokens = target.split()
    n_tokens = len(target_tokens)
    n_words = len(words)

    # 1. Exact string search across n-gram windows
    # Window size between max(1, n_tokens - 2) and min(n_words, n_tokens + 4)
    min_len = max(1, n_tokens - 1)
    max_len = min(n_words, n_tokens + 5)

    best_span = None
    best_overlap = 0.0

    for length in range(min_len, max_len + 1):
        for i in range(n_words - length + 1):
            span_indices = set(range(i, i + length))
            # Avoid overwriting already matched tokens
            if span_indices & occupied_indices:
                continue

            span_text = " ".join(words[i : i + length])
            clean_span = normalize_text(span_text)

            # Exact match
            if clean_span == clean_target:
                return (i, i + length)

            # Substring exact matches
            if clean_target in clean_span or clean_span in clean_target:
                overlap = len(clean_span) / max(len(clean_target), 1)
                if 0.85 <= overlap <= 1.15 and overlap > best_overlap:
                    best_overlap = overlap
                    best_span = (i, i + length)

    if best_span and best_overlap >= 0.85:
        return best_span

    # 2. Single token exact match fallback (crucial for TOTAL or DATE)
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
) -> AlignedReceipt:
    """
    Aligns SROIE ground truth fields (COMPANY, DATE, ADDRESS, TOTAL) with OCR word tokens.
    Assigns BIO labels, records unmatched entities, and avoids corrupting ambiguous matches.
    """
    n_words = len(words)
    labels = ["O"] * n_words
    occupied: Set[int] = set()
    alignment_stats: Dict[str, bool] = {}
    unmatched: List[str] = []

    # Map possible ground truth keys to canonical ENTITIES
    key_mapping = {
        "company": "COMPANY",
        "date": "DATE",
        "address": "ADDRESS",
        "total": "TOTAL",
    }

    # Match in priority order: TOTAL, DATE, COMPANY, ADDRESS
    # (Total and Date are shorter and more specific, reducing false positives)
    match_order = ["total", "date", "company", "address"]

    for key in match_order:
        canonical_entity = key_mapping[key]
        raw_val = ground_truth.get(key) or ground_truth.get(canonical_entity) or ""
        raw_val = str(raw_val).strip()

        if not raw_val:
            alignment_stats[canonical_entity] = False
            unmatched.append(canonical_entity)
            continue

        span = find_best_matching_span(words, raw_val, occupied)

        if span:
            start, end = span
            # Assign BIO tags
            labels[start] = f"B-{canonical_entity}"
            for k in range(start + 1, end):
                labels[k] = f"I-{canonical_entity}"
                occupied.add(k)
            occupied.add(start)
            alignment_stats[canonical_entity] = True
        else:
            alignment_stats[canonical_entity] = False
            unmatched.append(canonical_entity)

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
    )
