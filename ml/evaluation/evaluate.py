import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        AutoImageProcessor,
        AutoProcessor,
        AutoTokenizer,
        LayoutLMv3ForTokenClassification,
    )

    HAS_TORCH = True
except (ImportError, OSError):
    torch = None  # type: ignore
    DataLoader = None  # type: ignore
    AutoImageProcessor = None  # type: ignore
    AutoProcessor = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    LayoutLMv3ForTokenClassification = None  # type: ignore
    HAS_TORCH = False

from ml.data.annotation_alignment import ID2LABEL
from ml.data.prepare_dataset import compute_split_hash, load_aligned_receipts
from ml.evaluation.metrics import (
    compute_checkpoint_disk_size_mb,
    compute_document_metrics,
    compute_latency_summary,
    compute_model_footprint,
    compute_seqeval_metrics,
)
from ml.extractors.layoutlmv3_extractor import validate_checkpoint_directory
from ml.inference.decoder import decode_layoutlmv3_predictions
from ml.training.config import detect_default_device
from ml.training.dataset import SROIELayoutLMv3Dataset, align_tokens_and_boxes_arrays, collate_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DocuMind.Evaluate")


def _write_evaluation_outputs(out_dir: Path, results: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "evaluation_report.json"
    md_path = out_dir / "evaluation_report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    is_synthetic = bool(results.get("is_synthetic", False))
    status = results.get("status", "UNKNOWN")
    title_tag = " (SYNTHETIC SMOKE TEST)" if is_synthetic else ""
    md_lines = [f"# DocuMind LayoutLMv3 Evaluation Report{title_tag}", ""]

    if is_synthetic:
        md_lines.append(
            "> **WARNING:** Evaluated on synthetic smoke-test receipts. Numbers do NOT reflect real SROIE benchmark accuracy.\n"
        )

    md_lines.extend(
        [
            f"- **Status**: `{status}`",
            f"- **Checkpoint**: `{results.get('checkpoint')}`",
            f"- **Dataset Split**: `{results.get('test_dataset')}` (`split_hash={results.get('split_hash')}`)",
            f"- **Samples Evaluated**: {results.get('num_test_samples')}",
            f"- **Device**: `{results.get('device')}`",
            f"- **Evaluated At (UTC)**: `{results.get('evaluated_at_utc')}`",
        ]
    )

    if status != "EVALUATED":
        md_lines.append(f"- **Reason**: {results.get('reason', 'Unavailable')}")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")
        return

    footprint = results.get("footprint") or {}
    lat_pct = results.get("latency_percentiles_ms") or {}
    ckpt_size_mb = results.get("checkpoint_size_mb")
    seqeval_res = results["seqeval"]
    doc_res = results["document_metrics"]
    md_lines.extend(
        [
            f"- **Model-Only Latency (avg / p50 / p95 / p99)**: {results['latency_ms_per_doc']} ms / {lat_pct.get('p50_ms', 'N/A')} ms / {lat_pct.get('p95_ms', 'N/A')} ms / {lat_pct.get('p99_ms', 'N/A')} ms ({results['throughput_docs_per_sec']} docs/sec)",
            f"- **Total Parameters**: {footprint.get('total_parameters', 0):,} ({footprint.get('estimated_parameter_size_mb', 0)} MB param footprint; checkpoint on disk: {ckpt_size_mb if ckpt_size_mb is not None else 'N/A'} MB)",
            "",
            "## Entity Extraction Performance (Seqeval strict IOB2)",
            "| Entity | Precision | Recall | F1 Score | Support |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
    )
    for ent, met in seqeval_res["entities"].items():
        md_lines.append(
            f"| **{ent}** | {met['precision']:.4f} | {met['recall']:.4f} | {met['f1']:.4f} | {met['support']} |"
        )
    md_lines.append(
        f"| **OVERALL (Micro)** | {seqeval_res['overall_precision']:.4f} | {seqeval_res['overall_recall']:.4f} | {seqeval_res['overall_f1']:.4f} | - |"
    )
    if "macro_f1" in seqeval_res:
        md_lines.append(
            f"| **MACRO F1 (4 Entities)** | - | - | {seqeval_res['macro_f1']:.4f} | - |"
        )

    md_lines.extend(
        [
            "",
            "## Document & Field Level Exact Match",
            f"- **Complete Document Exact Match**: `{doc_res['document_exact_match']:.2%}`",
            "",
            "| Field | Exact Match Rate |",
            "| :--- | :--- |",
        ]
    )
    for field_name, em in doc_res["field_exact_match"].items():
        md_lines.append(f"| **{field_name.upper()}** | {em:.2%} |")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")


def evaluate_checkpoint(
    checkpoint_dir: str,
    test_jsonl: str,
    output_dir: str = "ml/evaluation/results",
    device: Optional[str] = None,
    batch_size: int = 2,
    allow_synthetic: bool = False,
) -> Dict[str, Any]:
    """Runs held-out evaluation for a fine-tuned LayoutLMv3 checkpoint using unified preprocessing."""
    resolved_device = device or detect_default_device()
    test_path = Path(test_jsonl)
    if not test_path.exists():
        raise FileNotFoundError(f"Evaluation split manifest does not exist: {test_jsonl}")

    receipts = load_aligned_receipts(test_path)
    split_hash = compute_split_hash(receipts)
    is_synthetic = any(r.is_synthetic for r in receipts)
    out_dir = Path(output_dir)

    if is_synthetic and not allow_synthetic:
        raise RuntimeError(
            f"Refusing scientific evaluation on synthetic dataset '{test_jsonl}' without --allow-synthetic."
        )

    checkpoint_path = Path(checkpoint_dir)
    is_valid_ckpt, ckpt_reason = validate_checkpoint_directory(checkpoint_path)
    if not is_valid_ckpt or not HAS_TORCH or torch is None:
        reason = ckpt_reason if not is_valid_ckpt else "PyTorch/Transformers runtime unavailable"
        not_eval_res = {
            "status": "NOT_EVALUATED",
            "reason": reason,
            "checkpoint": str(checkpoint_path),
            "checkpoint_metadata": None,
            "test_dataset": str(test_path),
            "split_hash": split_hash,
            "num_test_samples": len(receipts),
            "is_synthetic": is_synthetic,
            "device": resolved_device,
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "seqeval": None,
            "document_metrics": None,
            "latency_ms_per_doc": None,
            "throughput_docs_per_sec": None,
            "footprint": None,
        }
        _write_evaluation_outputs(out_dir, not_eval_res)
        return not_eval_res

    ckpt_meta = None
    meta_file = checkpoint_path / "checkpoint_metadata.json"
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as mf:
            ckpt_meta = json.load(mf)

    try:
        processor = AutoProcessor.from_pretrained(str(checkpoint_path), apply_ocr=False)
        tokenizer = processor.tokenizer
        image_processor = getattr(processor, "image_processor", None)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
        try:
            image_processor = AutoImageProcessor.from_pretrained(str(checkpoint_path), apply_ocr=False)
        except Exception:
            image_processor = None

    model = LayoutLMv3ForTokenClassification.from_pretrained(str(checkpoint_path))
    model.to(resolved_device)
    model.eval()

    test_dataset = SROIELayoutLMv3Dataset(
        receipts,
        tokenizer=tokenizer,
        max_seq_length=512,
        image_processor=image_processor,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_preds: List[List[str]] = []
    all_refs: List[List[str]] = []
    latencies: List[float] = []
    decoded_predictions: List[Dict[str, Any]] = []
    ground_truth_docs: List[Dict[str, Any]] = []

    receipt_idx = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(resolved_device)
            attention_mask = batch["attention_mask"].to(resolved_device)
            bbox = batch["bbox"].to(resolved_device)
            pixel_values = batch["pixel_values"].to(resolved_device)
            labels = batch["labels"].to(resolved_device)

            start_t = time.perf_counter()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
                labels=labels,
            )
            elapsed = (time.perf_counter() - start_t) / max(1, len(input_ids))
            for _ in range(len(input_ids)):
                latencies.append(elapsed)

            logits = outputs.logits.detach().cpu().numpy()
            label_ids = labels.detach().cpu().numpy()
            pred_ids = np.argmax(logits, axis=-1)

            for b_idx in range(len(label_ids)):
                seq_preds: List[str] = []
                seq_refs: List[str] = []
                for t_idx in range(len(label_ids[b_idx])):
                    if label_ids[b_idx, t_idx] != -100:
                        seq_refs.append(ID2LABEL.get(int(label_ids[b_idx, t_idx]), "O"))
                        seq_preds.append(ID2LABEL.get(int(pred_ids[b_idx, t_idx]), "O"))

                all_preds.append(seq_preds)
                all_refs.append(seq_refs)

                current_receipt = receipts[receipt_idx]
                token_prep = align_tokens_and_boxes_arrays(
                    tokenizer=tokenizer,
                    words=current_receipt.words,
                    boxes=current_receipt.boxes,
                    word_labels=None,
                    max_seq_length=512,
                    return_word_ids=True,
                )
                decoded = decode_layoutlmv3_predictions(
                    words=current_receipt.words,
                    boxes=current_receipt.boxes,
                    word_ids=token_prep["word_ids"],
                    logits=logits[b_idx],
                )
                decoded_predictions.append(decoded["fields"])
                ground_truth_docs.append(current_receipt.ground_truth)
                receipt_idx += 1

    seqeval_res = compute_seqeval_metrics(all_preds, all_refs)
    doc_res = compute_document_metrics(decoded_predictions, ground_truth_docs)
    footprint = compute_model_footprint(model)
    latency_percentiles_ms = compute_latency_summary([lat * 1000.0 for lat in latencies])
    checkpoint_size_mb = compute_checkpoint_disk_size_mb(checkpoint_path)

    avg_latency_ms = latency_percentiles_ms["avg_ms"]
    throughput = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

    results = {
        "status": "EVALUATED",
        "checkpoint": str(checkpoint_path),
        "checkpoint_size_mb": checkpoint_size_mb,
        "checkpoint_metadata": ckpt_meta,
        "test_dataset": str(test_path),
        "split_hash": split_hash,
        "num_test_samples": len(receipts),
        "is_synthetic": is_synthetic,
        "device": resolved_device,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seqeval": seqeval_res,
        "document_metrics": doc_res,
        "latency_ms_per_doc": avg_latency_ms,
        "latency_percentiles_ms": latency_percentiles_ms,
        "throughput_docs_per_sec": round(throughput, 2),
        "footprint": footprint,
    }
    _write_evaluation_outputs(out_dir, results)
    logger.info("Evaluation report written to %s", out_dir)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned LayoutLMv3 checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/layoutlmv3-sroie-best")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--test-jsonl", type=str, default="data/sroie_processed/test.jsonl")
    parser.add_argument("--output-dir", type=str, default="ml/evaluation/results")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()

    split_jsonl = args.test_jsonl
    if args.data_dir:
        cand = Path(args.data_dir) / f"{args.split}.jsonl"
        if args.split == "validation" and not cand.exists():
            cand = Path(args.data_dir) / "val.jsonl"
        split_jsonl = str(cand)

    res = evaluate_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        test_jsonl=split_jsonl,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        allow_synthetic=args.allow_synthetic,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
