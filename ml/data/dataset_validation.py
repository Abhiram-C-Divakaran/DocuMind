import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional
from ml.data.annotation_alignment import AlignedReceipt, LABEL_LIST


def generate_dataset_validation_report(
    splits: Dict[str, List[AlignedReceipt]],
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Computes dataset statistics across splits:
    - Record counts
    - BIO label distribution
    - Entity alignment success rates
    - Missing label counts
    - Bounding box integrity validation
    """
    report: Dict[str, Any] = {
        "record_counts": {},
        "total_records": 0,
        "total_words": 0,
        "label_distribution": Counter(),
        "alignment_rates": {},
        "missing_ground_truth": Counter(),
        "bbox_integrity": {"valid_boxes": 0, "invalid_boxes": 0},
    }

    entity_attempts: Counter = Counter()
    entity_successes: Counter = Counter()

    for split_name, receipts in splits.items():
        report["record_counts"][split_name] = len(receipts)

        for r in receipts:
            report["total_words"] += len(r.words)

            # Label counts
            for l in r.labels:
                report["label_distribution"][l] += 1

            # Bounding box sanity
            for box in r.boxes:
                if (
                    len(box) == 4
                    and 0 <= box[0] <= box[2] <= 1000
                    and 0 <= box[1] <= box[3] <= 1000
                ):
                    report["bbox_integrity"]["valid_boxes"] += 1
                else:
                    report["bbox_integrity"]["invalid_boxes"] += 1

            # Alignment stats
            for field_name in ["COMPANY", "DATE", "ADDRESS", "TOTAL"]:
                entity_attempts[field_name] += 1
                if r.alignment_stats.get(field_name, False):
                    entity_successes[field_name] += 1
                else:
                    # Check if ground truth was missing or alignment failed
                    gt_key = field_name.lower()
                    if not r.ground_truth.get(gt_key):
                        report["missing_ground_truth"][field_name] += 1

    # Calculate percentages
    for field_name in ["COMPANY", "DATE", "ADDRESS", "TOTAL"]:
        att = entity_attempts[field_name]
        succ = entity_successes[field_name]
        report["alignment_rates"][field_name] = {
            "attempted": att,
            "aligned": succ,
            "rate_percent": round((succ / att * 100.0) if att > 0 else 0.0, 2),
        }

    report["total_records"] = sum(report["record_counts"].values())
    # Convert Counters to standard dicts
    report["label_distribution"] = dict(report["label_distribution"])
    report["missing_ground_truth"] = dict(report["missing_ground_truth"])

    # Persist reports if output_dir is given
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        json_path = out / "validation_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        md_path = out / "validation_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# SROIE Dataset Validation Report\n\n")
            f.write("## Record Counts\n")
            for sp, cnt in report["record_counts"].items():
                f.write(f"- **{sp}**: {cnt} receipts\n")
            f.write(f"- **Total Words**: {report['total_words']:,}\n\n")

            f.write("## Alignment Success Rates\n")
            f.write("| Entity | Attempted | Aligned | Success Rate |\n")
            f.write("|---|---|---|---|\n")
            for ent, stats in report["alignment_rates"].items():
                f.write(f"| {ent} | {stats['attempted']} | {stats['aligned']} | {stats['rate_percent']}% |\n")
            f.write("\n")

            f.write("## BIO Label Distribution\n")
            f.write("| Label | Count |\n|---|---|\n")
            for lbl in LABEL_LIST:
                f.write(f"| `{lbl}` | {report['label_distribution'].get(lbl, 0)} |\n")
            f.write("\n")

            f.write("## Bounding Box Integrity\n")
            f.write(f"- Valid BBoxes in [0, 1000]: {report['bbox_integrity']['valid_boxes']:,}\n")
            f.write(f"- Invalid BBoxes: {report['bbox_integrity']['invalid_boxes']}\n")

    return report
