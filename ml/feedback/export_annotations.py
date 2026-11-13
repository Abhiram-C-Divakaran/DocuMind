import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from ml.feedback.feedback_loader import (
    load_approved_feedback_records,
    load_feedback_from_session,
)
from ml.feedback.feedback_quality import assess_feedback_quality
from ml.feedback.feedback_validator import validate_and_deduplicate_feedback

logger = logging.getLogger("documind.ml.feedback.export")


async def export_feedback_from_session(
    db: AsyncSession,
    output_dir: str = "data/feedback_exports",
    upload_dir: Optional[str] = None,
    reference_splits: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Exports validated human-reviewed records using an active AsyncSession."""
    approved_records, excluded_status_counts = await load_feedback_from_session(db)
    return _finalize_export(
        approved_records=approved_records,
        excluded_status_counts=excluded_status_counts,
        output_dir=output_dir,
        upload_dir=upload_dir,
        reference_splits=reference_splits,
    )


def export_feedback_dataset(
    db_url: Optional[str] = None,
    output_dir: str = "data/feedback_exports",
    upload_dir: Optional[str] = None,
    reference_splits: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Synchronous CLI & pipeline helper exporting approved feedback JSONL."""
    approved_records, excluded_status_counts = load_approved_feedback_records(db_url=db_url)
    return _finalize_export(
        approved_records=approved_records,
        excluded_status_counts=excluded_status_counts,
        output_dir=output_dir,
        upload_dir=upload_dir,
        reference_splits=reference_splits,
    )


def _finalize_export(
    approved_records: List[Any],
    excluded_status_counts: Dict[str, int],
    output_dir: str,
    upload_dir: Optional[str],
    reference_splits: Optional[List[str]],
) -> Dict[str, Any]:
    validated_items, excluded_validation = validate_and_deduplicate_feedback(
        records=approved_records,
        upload_dir=upload_dir,
        reference_split_paths=reference_splits,
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_path / "feedback_export.jsonl"

    aligned_confident = 0
    needs_manual_annotation = 0

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in validated_items:
            rec = item.record
            quality = assess_feedback_quality(item)
            if quality["alignment_status"] == "ALIGNED_CONFIDENT":
                aligned_confident += 1
            else:
                needs_manual_annotation += 1

            payload = {
                "schema_version": "3a-1.0",
                "record_id": f"fb_{rec.review_id}",
                "document_id": rec.document_id,
                "extraction_id": rec.extraction_id,
                "review_id": rec.review_id,
                "original_filename": rec.original_filename,
                "image_ref": item.image_ref,
                "image_sha256": item.image_sha256,
                "text_sha256": item.text_sha256,
                "content_type": rec.content_type,
                "extraction_method": rec.extraction_method,
                "model_confidence_heuristic": rec.heuristic_confidence,
                "original_predictions": rec.original_predictions,
                "corrected_values": item.validated_corrected_values,
                "changed_fields": rec.changed_fields,
                "provenance": {
                    "review_status": rec.review_status,
                    "reviewer_identifier": rec.reviewer_identifier,
                    "reviewer_notes": rec.reviewer_notes,
                    "approved_at": rec.approved_at,
                    "review_version": rec.review_version,
                },
                "quality": quality,
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    summary: Dict[str, Any] = {
        "schema_version": "3a-1.0",
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_jsonl": str(jsonl_path),
        "approved_loaded": len(approved_records),
        "exported_count": len(validated_items),
        "excluded_by_review_status": excluded_status_counts,
        "excluded_by_validation_or_contamination": excluded_validation,
        "alignment_breakdown": {
            "aligned_confident": aligned_confident,
            "needs_manual_annotation": needs_manual_annotation,
        },
        "automatic_retraining_launched": False,
    }

    summary_json_path = out_path / "feedback_export_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2)

    md_lines = [
        "# DocuMind Active Learning Feedback Export Summary",
        "",
        f"- **Exported At (UTC)**: `{summary['exported_at_utc']}`",
        f"- **Output JSONL**: `{jsonl_path}`",
        f"- **Approved Reviews Loaded**: {summary['approved_loaded']}",
        f"- **Valid Unique Records Exported**: {summary['exported_count']}",
        f"- **Automatic Retraining Launched**: `False` (human/token alignment verification required first)",
        "",
        "## Token Alignment Readiness",
        f"- `ALIGNED_CONFIDENT` (OCR tokens + boxes matched): {aligned_confident}",
        f"- `NEEDS_MANUAL_ANNOTATION` (unmatched or missing boxes): {needs_manual_annotation}",
        "",
        "## Excluded Non-Approved Records",
    ]
    for reason, cnt in excluded_status_counts.items():
        md_lines.append(f"- `{reason}`: {cnt}")
    if excluded_validation:
        md_lines.append("")
        md_lines.append("## Excluded During Validation / Deduplication / Contamination Check")
        for ex in excluded_validation:
            md_lines.append(f"- `{ex['document_id']}`: `{ex['reason']}`")

    with open(out_path / "feedback_export_summary.md", "w", encoding="utf-8") as mf:
        mf.write("\n".join(md_lines) + "\n")

    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(
        description="Export validated human-reviewed document corrections to JSONL feedback dataset"
    )
    parser.add_argument("--db-url", type=str, default=None, help="Database URL override")
    parser.add_argument("--output-dir", type=str, default="data/feedback_exports")
    parser.add_argument("--upload-dir", type=str, default=None)
    parser.add_argument(
        "--reference-splits",
        nargs="*",
        default=["data/sroie_processed/test.jsonl", "data/sroie_processed/validation.jsonl"],
        help="JSONL splits checked for train/test contamination",
    )
    args = parser.parse_args()

    res = export_feedback_dataset(
        db_url=args.db_url,
        output_dir=args.output_dir,
        upload_dir=args.upload_dir,
        reference_splits=args.reference_splits,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

