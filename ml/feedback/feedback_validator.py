import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.app.config import settings
from backend.app.schemas.review import ReviewCorrectionPayload
from ml.feedback.feedback_loader import FeedbackRecord


@dataclass
class ValidatedFeedbackItem:
    record: FeedbackRecord
    image_ref: str
    image_sha256: str
    text_sha256: str
    validated_corrected_values: Dict[str, Any]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _text_fingerprint(text: Optional[str], values: Dict[str, Any]) -> str:
    norm_text = " ".join((text or "").lower().split())
    payload = f"{norm_text}||{json.dumps(values, sort_keys=True, ensure_ascii=False)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_contamination_hashes(
    reference_split_paths: Optional[List[str]],
) -> Tuple[Set[str], Set[str]]:
    """
    Collects image SHA-256 and normalized receipt text/entity hashes from held-out splits
    (e.g. SROIE test.jsonl / validation.jsonl) to guarantee zero train/test contamination.
    """
    image_hashes: Set[str] = set()
    text_hashes: Set[str] = set()
    for p_str in reference_split_paths or []:
        p = Path(p_str)
        if not p.exists() or not p.is_file():
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("image_sha256"):
                    image_hashes.add(str(row["image_sha256"]))
                words = row.get("words") or []
                gt = row.get("ground_truth") or {}
                if words:
                    fp = hashlib.sha256(
                        (
                            " ".join(str(w).lower() for w in words)
                            + "||"
                            + json.dumps(gt, sort_keys=True, ensure_ascii=False)
                        ).encode("utf-8")
                    ).hexdigest()
                    text_hashes.add(fp)
    return image_hashes, text_hashes


def validate_and_deduplicate_feedback(
    records: List[FeedbackRecord],
    upload_dir: Optional[str] = None,
    reference_split_paths: Optional[List[str]] = None,
) -> Tuple[List[ValidatedFeedbackItem], List[Dict[str, str]]]:
    """
    Validates approved feedback records against Pydantic schema, file existence,
    image/content duplicate detection, and held-out split contamination checks.
    """
    storage_root = Path(upload_dir or settings.UPLOAD_DIR).resolve()
    contam_img_hashes, contam_txt_hashes = _load_contamination_hashes(reference_split_paths)

    validated: List[ValidatedFeedbackItem] = []
    rejected_invalid: List[Dict[str, str]] = []

    seen_doc_ids: Set[str] = set()
    seen_img_hashes: Set[str] = set()

    for rec in records:
        if rec.review_status != "APPROVED":
            rejected_invalid.append(
                {
                    "document_id": rec.document_id,
                    "reason": f"non_approved_status:{rec.review_status}",
                }
            )
            continue

        # Schema & financial/date integrity validation
        try:
            parsed_corr = ReviewCorrectionPayload.model_validate(rec.corrected_values)
            clean_corr = parsed_corr.to_json_dict()
        except Exception as exc:
            rejected_invalid.append(
                {
                    "document_id": rec.document_id,
                    "reason": f"schema_validation_failed:{exc}",
                }
            )
            continue

        # Verify source document file safely inside storage_root
        stored = rec.stored_filename
        if not stored or "/" in stored or "\\" in stored or ".." in stored:
            rejected_invalid.append(
                {
                    "document_id": rec.document_id,
                    "reason": "invalid_stored_filename",
                }
            )
            continue

        file_path = (storage_root / stored).resolve()
        try:
            file_path.relative_to(storage_root)
        except ValueError:
            rejected_invalid.append(
                {
                    "document_id": rec.document_id,
                    "reason": "path_traversal_detected",
                }
            )
            continue

        if not file_path.exists() or not file_path.is_file() or file_path.stat().st_size == 0:
            rejected_invalid.append(
                {
                    "document_id": rec.document_id,
                    "reason": "missing_source_file",
                }
            )
            continue

        img_sha = _sha256_file(file_path)
        txt_sha = _text_fingerprint(rec.raw_text, clean_corr)

        # Duplicate detection
        if rec.document_id in seen_doc_ids or img_sha in seen_img_hashes:
            rejected_invalid.append(
                {
                    "document_id": rec.document_id,
                    "reason": "duplicate_document_or_image_hash",
                }
            )
            continue

        # Held-out test/val contamination guard
        if img_sha in contam_img_hashes or txt_sha in contam_txt_hashes:
            rejected_invalid.append(
                {
                    "document_id": rec.document_id,
                    "reason": "held_out_split_contamination",
                }
            )
            continue

        seen_doc_ids.add(rec.document_id)
        seen_img_hashes.add(img_sha)

        # Store non-filesystem-leaking relative reference within upload store
        image_ref = f"uploads/{stored}"
        validated.append(
            ValidatedFeedbackItem(
                record=rec,
                image_ref=image_ref,
                image_sha256=img_sha,
                text_sha256=txt_sha,
                validated_corrected_values=clean_corr,
            )
        )

    return validated, rejected_invalid

