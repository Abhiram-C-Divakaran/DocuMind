import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, LayoutLMv3ForTokenClassification
    HAS_TORCH = True
except (ImportError, OSError):
    torch = None  # type: ignore
    DataLoader = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    LayoutLMv3ForTokenClassification = None  # type: ignore
    HAS_TORCH = False

from ml.data.annotation_alignment import ID2LABEL, AlignedReceipt
from ml.data.prepare_dataset import load_aligned_receipts
from ml.evaluation.metrics import (
    compute_document_metrics,
    compute_model_footprint,
    compute_seqeval_metrics,
)
from ml.inference.decoder import decode_layoutlmv3_predictions
from ml.training.dataset import SROIELayoutLMv3Dataset, collate_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DocuMind.Evaluate")


def evaluate_checkpoint(
    checkpoint_dir: str,
    test_jsonl: str,
    output_dir: str = "ml/evaluation/results",
    device: str = "cpu",
    batch_size: int = 2,
) -> Dict[str, Any]:
    """Runs full evaluation on the test set for a fine-tuned LayoutLMv3 checkpoint."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    test_path = Path(test_jsonl)
    if not test_path.exists():
        raise FileNotFoundError(f"Test split manifest does not exist: {test_jsonl}")

    logger.info("Loading checkpoint from %s", checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
    model = LayoutLMv3ForTokenClassification.from_pretrained(str(checkpoint_path))
    model.to(device)
    model.eval()

    receipts = load_aligned_receipts(test_path)
    logger.info("Evaluating on %d test receipts", len(receipts))

    test_dataset = SROIELayoutLMv3Dataset(receipts, tokenizer=tokenizer, max_seq_length=512)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 1. Token & Entity Level Evaluation with Seqeval
    all_preds: List[List[str]] = []
    all_refs: List[List[str]] = []
    latencies: List[float] = []

    decoded_predictions: List[Dict[str, Any]] = []
    ground_truth_docs: List[Dict[str, Any]] = []

    receipt_idx = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bbox = batch["bbox"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            start_t = time.perf_counter()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
                labels=labels,
            )
            elapsed = (time.perf_counter() - start_t) / len(input_ids)
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
                        ref_label = ID2LABEL.get(int(label_ids[b_idx, t_idx]), "O")
                        pred_label = ID2LABEL.get(int(pred_ids[b_idx, t_idx]), "O")
                        seq_refs.append(ref_label)
                        seq_preds.append(pred_label)

                all_preds.append(seq_preds)
                all_refs.append(seq_refs)

                # Decode into structured document fields
                current_receipt = receipts[receipt_idx]
                encoding = tokenizer(
                    current_receipt.words if current_receipt.words else [""],
                    is_split_into_words=True,
                    truncation=True,
                    max_length=512,
                    padding="max_length",
                )
                decoded = decode_layoutlmv3_predictions(
                    words=current_receipt.words,
                    boxes=current_receipt.boxes,
                    word_ids=encoding.word_ids(),
                    logits=torch.tensor(logits[b_idx]),
                )
                decoded_predictions.append(decoded["fields"])
                ground_truth_docs.append(current_receipt.ground_truth)
                receipt_idx += 1

    seqeval_res = compute_seqeval_metrics(all_preds, all_refs)
    doc_res = compute_document_metrics(decoded_predictions, ground_truth_docs)
    footprint = compute_model_footprint(model)

    avg_latency_ms = (sum(latencies) / len(latencies)) * 1000.0 if latencies else 0.0
    throughput = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

    results = {
        "checkpoint": str(checkpoint_path),
        "test_dataset": str(test_path),
        "num_test_samples": len(receipts),
        "seqeval": seqeval_res,
        "document_metrics": doc_res,
        "latency_ms_per_doc": round(avg_latency_ms, 2),
        "throughput_docs_per_sec": round(throughput, 2),
        "footprint": footprint,
    }

    # Save to disk
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "evaluation_results.json"
    md_path = out_dir / "evaluation_summary.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Build markdown report
    md_lines = [
        "# DocuMind LayoutLMv3 Evaluation Report",
        f"- **Checkpoint**: `{checkpoint_path}`",
        f"- **Samples Evaluated**: {len(receipts)}",
        f"- **Latency**: {results['latency_ms_per_doc']} ms/doc ({results['throughput_docs_per_sec']} docs/sec)",
        f"- **Total Parameters**: {footprint['total_parameters']:,} ({footprint['estimated_parameter_size_mb']} MB)",
        "",
        "## Entity Extraction Performance (Seqeval strict IOB2)",
        "| Entity | Precision | Recall | F1 Score | Support |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for ent, met in seqeval_res["entities"].items():
        md_lines.append(
            f"| **{ent}** | {met['precision']:.4f} | {met['recall']:.4f} | {met['f1']:.4f} | {met['support']} |"
        )
    md_lines.append(
        f"| **OVERALL** | {seqeval_res['overall_precision']:.4f} | {seqeval_res['overall_recall']:.4f} | {seqeval_res['overall_f1']:.4f} | - |"
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
        f.write("\n".join(md_lines))

    logger.info("Evaluation results written to %s and %s", json_path, md_path)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned LayoutLMv3 checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/layoutlmv3-sroie-best")
    parser.add_argument("--test-jsonl", type=str, default="data/sroie_processed/test.jsonl")
    parser.add_argument("--output-dir", type=str, default="ml/evaluation/results")
    parser.add_argument("--device", type=str, default="cuda" if (HAS_TORCH and torch and torch.cuda.is_available()) else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)

    args = parser.parse_args()
    res = evaluate_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        test_jsonl=args.test_jsonl,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
    )
    print("\n=== Evaluation Results ===")
    print(f"Overall F1: {res['seqeval']['overall_f1']:.4f}")
    print(f"Document Exact Match: {res['document_metrics']['document_exact_match']:.2%}")
    print(f"Latency: {res['latency_ms_per_doc']} ms/doc")


if __name__ == "__main__":
    main()
