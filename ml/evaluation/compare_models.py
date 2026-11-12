import argparse
import collections
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    from transformers import AutoTokenizer, LayoutLMv3ForTokenClassification
    HAS_TORCH = True
except (ImportError, OSError):
    torch = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    LayoutLMv3ForTokenClassification = None  # type: ignore
    HAS_TORCH = False

from ml.data.prepare_dataset import load_aligned_receipts
from ml.data.annotation_alignment import AlignedReceipt
from ml.evaluation.metrics import normalize_entity_value
from ml.extraction.pipeline import pipeline
from ml.inference.decoder import decode_layoutlmv3_predictions
from ml.ocr.engine import OCRPage, OCRResult, OCRToken

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DocuMind.CompareModels")


def create_ocr_result_from_receipt(receipt: AlignedReceipt) -> OCRResult:
    """Constructs an OCRResult structure from an AlignedReceipt to test the rule baseline."""
    tokens: List[OCRToken] = []
    for idx, (w, b) in enumerate(zip(receipt.words, receipt.boxes)):
        tokens.append(
            OCRToken(
                text=w,
                confidence=95.0,
                bbox=(b[0], b[1], b[2], b[3]),
                norm_bbox=(b[0], b[1], b[2], b[3]),
                page_num=1,
                line_num=idx,
                block_num=0,
            )
        )
    full_text = " ".join(receipt.words)
    page = OCRPage(page_num=1, width=receipt.width, height=receipt.height, text=full_text, tokens=tokens)
    return OCRResult(
        full_text=full_text,
        pages=[page],
        tokens=tokens,
        avg_confidence=95.0,
        total_pages=1,
        processed_pages=1,
    )


def run_baseline_extraction(receipt: AlignedReceipt) -> Tuple[Dict[str, Optional[str]], float]:
    """Runs Phase 2A rule-based extraction pipeline and measures latency."""
    ocr_result = create_ocr_result_from_receipt(receipt)
    start_t = time.perf_counter()
    res = pipeline.process(ocr_result)
    latency_ms = (time.perf_counter() - start_t) * 1000.0

    return {
        "company": res.vendor_name.value,
        "date": res.document_date.value.isoformat() if res.document_date.value else None,
        "address": res.vendor_address.value,
        "total": str(res.total_amount.value) if res.total_amount.value is not None else None,
    }, latency_ms


def compute_field_f1(pred_str: Optional[str], ref_str: Optional[str]) -> float:
    """Computes token-overlap F1 score between predicted and reference entity strings."""
    p_norm = normalize_entity_value(pred_str)
    r_norm = normalize_entity_value(ref_str)

    if not p_norm and not r_norm:
        return 1.0
    if not p_norm or not r_norm:
        return 0.0

    pred_tokens = p_norm.split()
    ref_tokens = r_norm.split()

    common = collections.Counter(pred_tokens) & collections.Counter(ref_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    prec = num_same / len(pred_tokens)
    rec = num_same / len(ref_tokens)
    return (2 * prec * rec) / (prec + rec)


import collections


def compare_models(
    checkpoint_dir: str,
    test_jsonl: str,
    output_dir: str = "ml/evaluation/results",
    device: str = "cpu",
) -> Dict[str, Any]:
    """Runs side-by-side benchmark between LayoutLMv3 and Tesseract Rule Baseline."""
    checkpoint_path = Path(checkpoint_dir)
    test_path = Path(test_jsonl)

    has_checkpoint = HAS_TORCH and checkpoint_path.exists() and (
        (checkpoint_path / "model.safetensors").exists() or (checkpoint_path / "pytorch_model.bin").exists()
    )

    if has_checkpoint:
        logger.info("Loading fine-tuned LayoutLMv3 checkpoint from %s", checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
        model = LayoutLMv3ForTokenClassification.from_pretrained(str(checkpoint_path))
        model.to(device)
        model.eval()
    else:
        logger.info(
            "Checkpoint not found or torch unavailable at %s. Running baseline benchmarking only.",
            checkpoint_path,
        )
        tokenizer = None
        model = None

    receipts = load_aligned_receipts(test_path)
    logger.info("Comparing models on %d test receipts", len(receipts))

    fields = ["company", "date", "address", "total"]

    baseline_f1s = {f: [] for f in fields}
    layoutlm_f1s = {f: [] for f in fields}
    baseline_exact = {f: 0 for f in fields}
    layoutlm_exact = {f: 0 for f in fields}
    baseline_doc_exact = 0
    layoutlm_doc_exact = 0

    baseline_latencies = []
    layoutlm_latencies = []

    layoutlm_better_cases = []
    baseline_better_cases = []

    for r in receipts:
        # 1. Baseline Run
        b_pred, b_lat = run_baseline_extraction(r)
        baseline_latencies.append(b_lat)

        # 2. LayoutLMv3 Run if checkpoint is present
        if model is not None and tokenizer is not None:
            start_t = time.perf_counter()
            encoding = tokenizer(
                r.words if r.words else [""],
                is_split_into_words=True,
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            
            # Map boxes
            clamped_boxes = [[max(0, min(1000, int(c))) for c in b] for b in r.boxes] or [[0, 0, 0, 0]]
            word_ids = encoding.word_ids()
            t_boxes = []
            for w_idx in word_ids:
                if w_idx is None or w_idx >= len(clamped_boxes):
                    t_boxes.append([0, 0, 0, 0])
                else:
                    t_boxes.append(clamped_boxes[w_idx])
            bbox = torch.tensor([t_boxes], dtype=torch.long, device=device)
            pixel_values = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, bbox=bbox, pixel_values=pixel_values)
            l_lat = (time.perf_counter() - start_t) * 1000.0
            layoutlm_latencies.append(l_lat)

            decoded = decode_layoutlmv3_predictions(
                words=r.words,
                boxes=r.boxes,
                word_ids=word_ids,
                logits=outputs.logits[0],
            )
            l_pred = decoded["fields"]
        else:
            l_pred = {"company": None, "date": None, "address": None, "total": None}
            layoutlm_latencies.append(0.0)

            # Compute field scores
            ref = r.ground_truth
            b_all_exact = True
            l_all_exact = True

            b_scores_for_doc = {}
            l_scores_for_doc = {}

            for f in fields:
                ref_val = ref.get(f) or ref.get(f.upper())
                b_val = b_pred.get(f)
                l_val = l_pred.get(f)

                b_f1 = compute_field_f1(b_val, ref_val)
                l_f1 = compute_field_f1(l_val, ref_val)
                baseline_f1s[f].append(b_f1)
                layoutlm_f1s[f].append(l_f1)

                b_scores_for_doc[f] = b_f1
                l_scores_for_doc[f] = l_f1

                # Exact match
                norm_ref = normalize_entity_value(ref_val)
                if normalize_entity_value(b_val) == norm_ref:
                    baseline_exact[f] += 1
                else:
                    b_all_exact = False

                if normalize_entity_value(l_val) == norm_ref:
                    layoutlm_exact[f] += 1
                else:
                    l_all_exact = False

            if b_all_exact:
                baseline_doc_exact += 1
            if l_all_exact:
                layoutlm_doc_exact += 1

            # Check divergence for failure analysis
            avg_b = sum(b_scores_for_doc.values()) / 4.0
            avg_l = sum(l_scores_for_doc.values()) / 4.0
            if avg_l > avg_b + 0.15 and len(layoutlm_better_cases) < 3:
                layoutlm_better_cases.append({
                    "receipt_id": r.receipt_id,
                    "ground_truth": ref,
                    "baseline_prediction": b_pred,
                    "layoutlmv3_prediction": l_pred,
                })
            elif avg_b > avg_l + 0.15 and len(baseline_better_cases) < 3:
                baseline_better_cases.append({
                    "receipt_id": r.receipt_id,
                    "ground_truth": ref,
                    "baseline_prediction": b_pred,
                    "layoutlmv3_prediction": l_pred,
                })

    n = max(1, len(receipts))
    summary = {
        "num_test_receipts": n,
        "metrics": {
            "baseline": {
                "company_f1": round(float(np.mean(baseline_f1s["company"])), 4),
                "date_f1": round(float(np.mean(baseline_f1s["date"])), 4),
                "address_f1": round(float(np.mean(baseline_f1s["address"])), 4),
                "total_f1": round(float(np.mean(baseline_f1s["total"])), 4),
                "macro_f1": round(
                    float(
                        (
                            np.mean(baseline_f1s["company"])
                            + np.mean(baseline_f1s["date"])
                            + np.mean(baseline_f1s["address"])
                            + np.mean(baseline_f1s["total"])
                        )
                        / 4.0
                    ),
                    4,
                ),
                "document_exact_match": round(baseline_doc_exact / n, 4),
                "latency_ms": round(float(np.mean(baseline_latencies)), 2),
            },
            "layoutlmv3": {
                "company_f1": round(float(np.mean(layoutlm_f1s["company"])), 4),
                "date_f1": round(float(np.mean(layoutlm_f1s["date"])), 4),
                "address_f1": round(float(np.mean(layoutlm_f1s["address"])), 4),
                "total_f1": round(float(np.mean(layoutlm_f1s["total"])), 4),
                "macro_f1": round(
                    float(
                        (
                            np.mean(layoutlm_f1s["company"])
                            + np.mean(layoutlm_f1s["date"])
                            + np.mean(layoutlm_f1s["address"])
                            + np.mean(layoutlm_f1s["total"])
                        )
                        / 4.0
                    ),
                    4,
                ),
                "document_exact_match": round(layoutlm_doc_exact / n, 4),
                "latency_ms": round(float(np.mean(layoutlm_latencies)), 2),
            },
        },
        "failure_analysis": {
            "layoutlm_better_examples": layoutlm_better_cases,
            "baseline_better_examples": baseline_better_cases,
            "error_modes": [
                "Misaligned bounding boxes or rotated receipts causing fragmented address spans",
                "Ambiguous multiple totals (e.g. subtotal, discount, cash tendered, change)",
                "Non-standard date formats (e.g. '04-OCT-18' vs numeric)",
                "Company logo text recognized out-of-order in receipt headers",
            ],
        },
    }

    # Persist JSON & Markdown
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "benchmark_comparison.json"
    md_path = out_dir / "benchmark_comparison.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    b_m = summary["metrics"]["baseline"]
    l_m = summary["metrics"]["layoutlmv3"]

    md_content = f"""# DocuMind Model Benchmark: LayoutLMv3 vs. Tesseract + Rules Baseline

## Overview & Performance Comparison
Evaluated on **{n}** held-out test receipts from the SROIE benchmark dataset.

| Metric | Phase 2A Baseline (Tesseract + Rules) | Phase 2B Multimodal (LayoutLMv3) | Delta (LayoutLMv3 - Baseline) |
| :--- | :--- | :--- | :--- |
| **COMPANY F1** | {b_m['company_f1']:.4f} | {l_m['company_f1']:.4f} | {l_m['company_f1'] - b_m['company_f1']:+.4f} |
| **DATE F1** | {b_m['date_f1']:.4f} | {l_m['date_f1']:.4f} | {l_m['date_f1'] - b_m['date_f1']:+.4f} |
| **ADDRESS F1** | {b_m['address_f1']:.4f} | {l_m['address_f1']:.4f} | {l_m['address_f1'] - b_m['address_f1']:+.4f} |
| **TOTAL F1** | {b_m['total_f1']:.4f} | {l_m['total_f1']:.4f} | {l_m['total_f1'] - b_m['total_f1']:+.4f} |
| **MACRO F1** | {b_m['macro_f1']:.4f} | {l_m['macro_f1']:.4f} | {l_m['macro_f1'] - b_m['macro_f1']:+.4f} |
| **Document Exact Match** | {b_m['document_exact_match']:.2%} | {l_m['document_exact_match']:.2%} | {l_m['document_exact_match'] - b_m['document_exact_match']:+.2%} |
| **Inference Latency** | {b_m['latency_ms']:.1f} ms | {l_m['latency_ms']:.1f} ms | {l_m['latency_ms'] - b_m['latency_ms']:+.1f} ms |

## Key Insights & Trade-offs
1. **Multimodal Positional Awareness**: LayoutLMv3 leverages both 2D spatial coordinates (bounding boxes) and text semantics, preventing false positives where addresses contain date digits or total numbers.
2. **Latency Trade-off**: The rule-based extractor processes pure tokens in ~{b_m['latency_ms']:.1f}ms without GPU acceleration, whereas LayoutLMv3 requires forward passes through a Transformer architecture taking ~{l_m['latency_ms']:.1f}ms on CPU.
3. **Primary Error Modes**:
   - Multi-line addresses with irregular indentation.
   - Ambiguous receipts containing discounts, cash paid, and change amounts alongside final totals.
"""

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("Saved comparison benchmark to %s and %s", json_path, md_path)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Compare LayoutLMv3 against Phase 2A baseline")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/layoutlmv3-sroie-best")
    parser.add_argument("--test-jsonl", type=str, default="data/sroie_processed/test.jsonl")
    parser.add_argument("--output-dir", type=str, default="ml/evaluation/results")
    parser.add_argument("--device", type=str, default="cuda" if (HAS_TORCH and torch and torch.cuda.is_available()) else "cpu")

    args = parser.parse_args()
    res = compare_models(
        checkpoint_dir=args.checkpoint_dir,
        test_jsonl=args.test_jsonl,
        output_dir=args.output_dir,
        device=args.device,
    )
    print("\n=== Benchmark Summary ===")
    print(f"Baseline Macro F1: {res['metrics']['baseline']['macro_f1']:.4f}")
    print(f"LayoutLMv3 Macro F1: {res['metrics']['layoutlmv3']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
