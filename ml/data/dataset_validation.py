import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from ml.data.annotation_alignment import AlignedReceipt


def generate_dataset_validation_report(
    splits: Dict[str, List[AlignedReceipt]],
    output_dir: Optional[Path] = None,
    excluded_samples: Optional[List[Dict[str, str]]] = None,
    leakage_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Computes dataset statistics across splits:
    - Record counts, synthetic vs real source breakdown
    - Image integrity and SHA-256 coverage
    - BIO label distribution
    - Entity alignment success rates and unmatched audit examples
    - Bounding box integrity validation
    - Excluded sample tracking and leakage summary
    """
    excluded = excluded_samples or []
    report: Dict[str, Any] = {
        "record_counts": {},
        "total_records": 0,
        "total_words": 0,
        "is_synthetic": False,
        "sources": Counter(),
        "label_distribution": Counter(),
        "alignment_rates": {},
        "missing_ground_truth": Counter(),
        "unmatched_examples": [],
        "bbox_integrity": {"valid_boxes": 0, "invalid_boxes": 0},
        "image_integrity": {
            "present_paths": 0,
            "missing_or_null_paths": 0,
            "with_sha256": 0,
        },
        "excluded_samples_count": len(excluded),
        "excluded_samples": excluded[:50],
        "leakage_report": leakage_report or {},
    }

    entity_attempts: Counter = Counter()
    entity_successes: Counter = Counter()

    for split_name, receipts in splits.items():
        report["record_counts"][split_name] = len(receipts)

        for r in receipts:
            report["total_words"] += len(r.words)
            if r.is_synthetic:
                report["is_synthetic"] = True
            report["sources"][r.source or "unknown"] += 1

            if r.image_path and Path(r.image_path).exists():
                report["image_integrity"]["present_paths"] += 1
            else:
                report["image_integrity"]["missing_or_null_paths"] += 1

            if r.image_sha256:
                report["image_integrity"]["with_sha256"] += 1

            for lbl in r.labels:
                report["label_distribution"][lbl] += 1

            for box in r.boxes:
                if (
                    len(box) == 4
                    and 0 <= box[0] <= box[2] <= 1000
                    and 0 <= box[1] <= box[3] <= 1000
                ):
                    report["bbox_integrity"]["valid_boxes"] += 1
                else:
                    report["bbox_integrity"]["invalid_boxes"] += 1

            for field_name in ["COMPANY", "DATE", "ADDRESS", "TOTAL"]:
                entity_attempts[field_name] += 1
                if r.alignment_stats.get(field_name, False):
                    entity_successes[field_name] += 1
                else:
                    gt_key = field_name.lower()
                    gt_val = r.ground_truth.get(gt_key)
                    if not gt_val:
                        report["missing_ground_truth"][field_name] += 1
                    elif len(report["unmatched_examples"]) < 25:
                        report["unmatched_examples"].append(
                            {
                                "split": split_name,
                                "receipt_id": r.receipt_id,
                                "entity": field_name,
                                "ground_truth": str(gt_val),
                            }
                        )

    for field_name in ["COMPANY", "DATE", "ADDRESS", "TOTAL"]:
        att = entity_attempts[field_name]
        succ = entity_successes[field_name]
        report["alignment_rates"][field_name] = {
            "attempted": att,
            "aligned": succ,
            "rate_percent": round((succ / att * 100.0) if att > 0 else 0.0, 2),
        }

    report["total_records"] = sum(report["record_counts"].values())
    report["sources"] = dict(report["sources"])
    report["label_distribution"] = dict(report["label_distribution"])
    report["missing_ground_truth"] = dict(report["missing_ground_truth"])

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        json_path = out / "validation_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        md_path = out / "validation_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            title_suffix = " (SYNTHETIC SMOKE TEST)" if report["is_synthetic"] else ""
            f.write(f"# SROIE Dataset Validation Report{title_suffix}\n\n")
            if report["is_synthetic"]:
                f.write(
                    "> **WARNING:** This dataset contains synthetic smoke-test receipts, not real SROIE benchmark images.\n\n"
                )
            f.write(f"- **Sources**: `{report['sources']}`\n")
            f.write(f"- **Excluded Invalid/Missing Samples**: {report['excluded_samples_count']}\n\n")

            f.write("## Record Counts\n")
            for sp, cnt in report["record_counts"].items():
                f.write(f"- **{sp}**: {cnt} receipts\n")
            f.write(f"- **Total Words**: {report['total_words']:,}\n\n")

            f.write("## Image & Bounding Box Integrity\n")
            f.write(
                f"- Present image files: {report['image_integrity']['present_paths']} "
                f"(missing/null: {report['image_integrity']['missing_or_null_paths']})\n"
            )
            f.write(f"- Verified SHA-256 hashes: {report['image_integrity']['with_sha256']}\n")
            f.write(
                f"- Valid bounding boxes: {report['bbox_integrity']['valid_boxes']} "
                f"(invalid: {report['bbox_integrity']['invalid_boxes']})\n\n"
            )

            f.write("## Alignment Success Rates\n")
            f.write("| Entity | Attempted | Aligned | Success Rate |\n")
            f.write("|---|---|---|---|\n")
            for ent, stats in report["alignment_rates"].items():
                f.write(
                    f"| {ent} | {stats['attempted']} | {stats['aligned']} | {stats['rate_percent']}% |\n"
                )

            if report["unmatched_examples"]:
                f.write("\n## Sample Unmatched Ground-Truth Annotations\n")
                for ex in report["unmatched_examples"][:10]:
                    f.write(
                        f"- `{ex['receipt_id']}` ({ex['split']}) **{ex['entity']}**: `{ex['ground_truth']}`\n"
                    )

    return report
