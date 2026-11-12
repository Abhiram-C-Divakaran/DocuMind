from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    HAS_TORCH = True
except (ImportError, OSError):
    torch = None  # type: ignore
    HAS_TORCH = False

from ml.data.annotation_alignment import ID2LABEL


def decode_layoutlmv3_predictions(
    words: List[str],
    boxes: List[List[int]],
    word_ids: List[Optional[int]],
    logits: Any,
) -> Dict[str, Any]:
    """
    Decodes token-level classification logits into structured entity fields:
    COMPANY, DATE, ADDRESS, and TOTAL.

    Args:
        words: List of original OCR words.
        boxes: List of original bounding boxes.
        word_ids: List mapping each sequence token to original word index (from tokenizer.word_ids()).
        logits: Tensor or numpy array of shape (seq_len, num_labels) with unnormalized scores.

    Returns:
        Dict containing:
            - fields: {"company": str, "date": str, "address": str, "total": str}
            - confidences: {"company": float, "date": float, ...}
            - word_labels: List of predicted labels for each word in `words`
            - word_confidences: List of float confidences for each word
    """
    if HAS_TORCH and torch is not None and hasattr(logits, "detach"):
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    else:
        l_arr = np.array(logits, dtype=np.float32)
        exp_logits = np.exp(l_arr - np.max(l_arr, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    preds = np.argmax(probs, axis=-1)

    num_words = len(words)
    word_predicted_labels = ["O"] * num_words
    word_confidences = [1.0] * num_words

    # Assign predicted label from the first subword of each word
    seen_words = set()
    for tok_idx, w_idx in enumerate(word_ids):
        if w_idx is not None and w_idx < num_words:
            if w_idx not in seen_words:
                seen_words.add(w_idx)
                pred_label_id = int(preds[tok_idx])
                conf = float(probs[tok_idx, pred_label_id])
                word_predicted_labels[w_idx] = ID2LABEL.get(pred_label_id, "O")
                word_confidences[w_idx] = conf

    # Group words into contiguous entity chunks
    # Entities: COMPANY, DATE, ADDRESS, TOTAL
    entities: Dict[str, List[Tuple[str, float]]] = {
        "COMPANY": [],
        "DATE": [],
        "ADDRESS": [],
        "TOTAL": [],
    }

    current_entity: Optional[str] = None
    current_chunk: List[str] = []
    current_confs: List[float] = []

    def commit_chunk():
        nonlocal current_entity, current_chunk, current_confs
        if current_entity and current_chunk:
            text = " ".join(current_chunk).strip()
            avg_conf = sum(current_confs) / len(current_confs) if current_confs else 0.5
            if text:
                entities[current_entity].append((text, avg_conf))
        current_entity = None
        current_chunk = []
        current_confs = []

    for w, lbl, conf in zip(words, word_predicted_labels, word_confidences):
        if lbl.startswith("B-"):
            commit_chunk()
            current_entity = lbl[2:]
            current_chunk.append(w)
            current_confs.append(conf)
        elif lbl.startswith("I-"):
            ent_type = lbl[2:]
            if current_entity == ent_type:
                current_chunk.append(w)
                current_confs.append(conf)
            else:
                # Mismatched I- tag, start fresh
                commit_chunk()
                current_entity = ent_type
                current_chunk.append(w)
                current_confs.append(conf)
        else:
            commit_chunk()

    commit_chunk()

    # Aggregate extracted fields
    extracted_fields: Dict[str, Optional[str]] = {
        "company": None,
        "date": None,
        "address": None,
        "total": None,
    }
    extracted_confs: Dict[str, float] = {
        "company": 0.0,
        "date": 0.0,
        "address": 0.0,
        "total": 0.0,
    }

    # COMPANY: choose first or highest confidence chunk
    if entities["COMPANY"]:
        # Company usually at the top, prefer the first detected block
        best_text, best_c = entities["COMPANY"][0]
        extracted_fields["company"] = best_text
        extracted_confs["company"] = round(best_c, 3)

    # ADDRESS: join multiple address chunks or pick longest
    if entities["ADDRESS"]:
        # Combine multi-line address blocks if present
        all_addr = ", ".join([txt for txt, _ in entities["ADDRESS"]])
        avg_c = sum([c for _, c in entities["ADDRESS"]]) / len(entities["ADDRESS"])
        extracted_fields["address"] = all_addr
        extracted_confs["address"] = round(avg_c, 3)

    # DATE: pick chunk with highest confidence
    if entities["DATE"]:
        sorted_dates = sorted(entities["DATE"], key=lambda x: x[1], reverse=True)
        extracted_fields["date"] = sorted_dates[0][0]
        extracted_confs["date"] = round(sorted_dates[0][1], 3)

    # TOTAL: pick highest confidence or last chunk (totals appear at the bottom)
    if entities["TOTAL"]:
        # Totals usually appear toward bottom of receipt
        best_text, best_c = entities["TOTAL"][-1]
        extracted_fields["total"] = best_text
        extracted_confs["total"] = round(best_c, 3)

    return {
        "fields": extracted_fields,
        "confidences": extracted_confs,
        "word_labels": word_predicted_labels,
        "word_confidences": word_confidences,
    }
