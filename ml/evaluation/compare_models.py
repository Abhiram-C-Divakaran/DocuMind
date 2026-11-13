import argparse
import collections
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import torch
    from transformers import (
        AutoImageProcessor,
        AutoProcessor,
        AutoTokenizer,
        LayoutLMv3ForTokenClassification,
    )

    HAS_TORCH = True
except (ImportError, OSError):
    torch = None  # type: ignore
    AutoImageProcessor = None  # type: ignore
    AutoProcessor = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    LayoutLMv3ForTokenClassification = None  # type: ignore
    HAS_TORCH = False

from ml.data.annotation_alignment import AlignedReceipt
from ml.data.prepare_dataset import compute_split_hash, load_aligned_receipts
from ml.evaluation.metrics import (
    compute_checkpoint_disk_size_mb,
    compute_latency_summary,
    normalize_entity_value,
)
from ml.extraction.pipeline import ReceiptExtractionPipeline
from ml.extractors.layoutlmv3_extractor import (
    parse_extracted_amount,
    parse_extracted_date,
    validate_checkpoint_directory,
)
from ml.inference.decoder import decode_layoutlmv3_predictions
from ml.ocr.engine import OCRPage, OCRResult, OCRToken, execute_tesseract_ocr, get_tesseract_status
from ml.training.config import detect_default_device
from ml.training.dataset import prepare_layoutlmv3_inputs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DocuMind.CompareModels")

pipeline = ReceiptExtractionPipeline()
BENCHMARK_FIELDS = ["company", "date", "address", "total"]


def create_ocr_result_from_receipt(receipt: AlignedReceipt) -> OCRResult:
    """
    Constructs a valid normalized OCRResult from an AlignedReceipt using 0.0-1.0 confidences
    (matching ml.ocr.engine.OCRToken scale), grouping words by y-coordinate proximity into
    visual receipt lines so line-based baseline heuristics operate fairly.
    """
    items: List[Tuple[float, int, str, List[int]]] = []
    for w, b in zip(receipt.words, receipt.boxes):
        y_center = (b[1] + b[3]) / 2.0
        items.append((y_center, int(b[0]), w, b))

    # Group tokens into visual horizontal lines in reading order
    items_sorted = sorted(items, key=lambda it: (it[0], it[1]))
    lines_grouped: List[List[Tuple[float, int, str, List[int]]]] = []
    for it in items_sorted:
        if not lines_grouped:
            lines_grouped.append([it])
            continue
        cur_line = lines_grouped[-1]
        cur_y = sum(x[0] for x in cur_line) / len(cur_line)
        if abs(it[0] - cur_y) <= 12.0:
            cur_line.append(it)
        else:
            lines_grouped.append([it])

    tokens: List[OCRToken] = []
    text_lines: List[str] = []
    for line_idx, line_items in enumerate(lines_grouped):
        line_items_lr = sorted(line_items, key=lambda it: it[1])
        line_words: List[str] = []
        for _, _, w, b in line_items_lr:
            tup_box = (b[0], b[1], b[2], b[3])
            tokens.append(
                OCRToken(
                    text=w,
                    confidence=0.95,
                    bbox=tup_box,
                    norm_bbox=tup_box,
                    page_num=1,
                    line_num=line_idx,
                    block_num=0,
                )
            )
            line_words.append(w)
        text_lines.append(" ".join(line_words))

    reconstructed_text = "\n".join(text_lines)
    page = OCRPage(
        page_num=1,
        width=receipt.width or 1000,
        height=receipt.height or 1000,
        text=reconstructed_text,
        tokens=tokens,
    )

    return OCRResult(
        full_text=reconstructed_text,
        pages=[page],
        tokens=tokens,
        avg_confidence=0.95,
        total_pages=1,
        processed_pages=1,
    )


def canonicalize_field_for_eval(field_name: str, value: Optional[str]) -> Optional[str]:
    """Normalizes structured dates/amounts fairly across rule baseline, LayoutLMv3, and SROIE refs."""
    if value is None:
        return None
    if field_name == "date":
        parsed = parse_extracted_date(str(value))
        return parsed.isoformat() if parsed else str(value)
    if field_name == "total":
        parsed_amt = parse_extracted_amount(str(value))
        return str(parsed_amt) if parsed_amt is not None else str(value)
    return str(value)


def run_baseline_extraction(
    receipt: AlignedReceipt,
    run_full_ocr: bool = False,
) -> Tuple[Dict[str, Optional[str]], Dict[str, float]]:
    """
    Runs Phase 2A rule-based baseline extraction.
    Separates full OCR latency (when run_full_ocr=True) from rule parsing latency.
    """
    ocr_latency_ms = 0.0
    if run_full_ocr:
        if not receipt.image_path or not Path(receipt.image_path).exists():
            raise FileNotFoundError(
                f"Cannot execute full OCR baseline: image missing for receipt '{receipt.receipt_id}'."
            )
        ocr_start = time.perf_counter()
        ocr_result = execute_tesseract_ocr(Path(receipt.image_path))
        ocr_latency_ms = (time.perf_counter() - ocr_start) * 1000.0
    else:
        ocr_result = create_ocr_result_from_receipt(receipt)

    parse_start = time.perf_counter()
    res = pipeline.process(ocr_result)
    parse_latency_ms = (time.perf_counter() - parse_start) * 1000.0

    fields = {
        "company": res.vendor_name.value,
        "date": res.document_date.value.isoformat() if res.document_date.value else None,
        "address": res.vendor_address.value,
        "total": str(res.total_amount.value) if res.total_amount.value is not None else None,
    }
    timings = {
        "ocr_latency_ms": ocr_latency_ms,
        "rule_parse_latency_ms": parse_latency_ms,
        "total_pipeline_latency_ms": ocr_latency_ms + parse_latency_ms,
    }
    return fields, timings


def compute_field_f1(
    pred_str: Optional[str],
    ref_str: Optional[str],
    field_name: Optional[str] = None,
) -> float:
    """Computes token-overlap F1 score between predicted and reference entity strings."""
    if field_name:
        pred_str = canonicalize_field_for_eval(field_name, pred_str)
        ref_str = canonicalize_field_for_eval(field_name, ref_str)

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


def _run_layoutlmv3_on_receipt(
    receipt: AlignedReceipt,
    model: Any,
    tokenizer: Any,
    image_processor: Any,
    device: str,
    run_full_ocr: bool = False,
) -> Tuple[Dict[str, Optional[str]], Dict[str, float]]:
    """Runs LayoutLMv3 using shared preprocessing helper and genuine receipt images."""
    if not receipt.image_path or not Path(receipt.image_path).exists():
        raise FileNotFoundError(
            f"Real receipt image is required for LayoutLMv3 benchmark (missing for '{receipt.receipt_id}')."
        )

    ocr_latency_ms = 0.0
    words = receipt.words
    boxes = receipt.boxes
    if run_full_ocr:
        ocr_start = time.perf_counter()
        ocr_res = execute_tesseract_ocr(Path(receipt.image_path))
        ocr_latency_ms = (time.perf_counter() - ocr_start) * 1000.0
        words = [t.text for t in ocr_res.tokens]
        boxes = [[t.norm_bbox[0], t.norm_bbox[1], t.norm_bbox[2], t.norm_bbox[3]] for t in ocr_res.tokens]
        if not words:
            return {f: None for f in BENCHMARK_FIELDS}, {
                "ocr_latency_ms": ocr_latency_ms,
                "feature_prep_latency_ms": 0.0,
                "model_inference_latency_ms": 0.0,
                "total_pipeline_latency_ms": ocr_latency_ms,
            }

    prep_start = time.perf_counter()
    prepared = prepare_layoutlmv3_inputs(
        image=receipt.image_path,
        words=words,
        boxes=boxes,
        tokenizer=tokenizer,
        image_processor=image_processor,
        word_labels=None,
        max_seq_length=512,
        return_word_ids=True,
        as_numpy=False,
    )
    input_ids = prepared["input_ids"].unsqueeze(0).to(device)
    attention_mask = prepared["attention_mask"].unsqueeze(0).to(device)
    bbox = prepared["bbox"].unsqueeze(0).to(device)
    pixel_values = prepared["pixel_values"].unsqueeze(0).to(device)
    word_ids = prepared["word_ids"]
    feature_prep_latency_ms = (time.perf_counter() - prep_start) * 1000.0

    infer_start = time.perf_counter()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
        )
    logits = outputs.logits[0].detach().cpu()
    model_inference_latency_ms = (time.perf_counter() - infer_start) * 1000.0

    decoded = decode_layoutlmv3_predictions(
        words=words,
        boxes=boxes,
        word_ids=word_ids,
        logits=logits,
    )
    return decoded["fields"], {
        "ocr_latency_ms": ocr_latency_ms,
        "feature_prep_latency_ms": feature_prep_latency_ms,
        "model_inference_latency_ms": model_inference_latency_ms,
        "total_pipeline_latency_ms": ocr_latency_ms + feature_prep_latency_ms + model_inference_latency_ms,
    }


def _build_not_evaluated_summary(reason: str) -> Dict[str, Any]:
    return {
        "status": "NOT_EVALUATED",
        "reason": reason,
        "field_f1": {f: None for f in BENCHMARK_FIELDS},
        "field_exact_match": {f: None for f in BENCHMARK_FIELDS},
        "macro_f1": None,
        "document_exact_match": None,
        "ocr_latency_ms": None,
        "feature_prep_latency_ms": None,
        "model_inference_latency_ms": None,
        "total_pipeline_latency_ms": None,
        "avg_latency_ms": None,
        "latency_percentiles_ms": None,
        "checkpoint_size_mb": None,
    }


def compare_models(
    checkpoint_dir: str,
    test_jsonl: str,
    output_dir: str = "ml/evaluation/results",
    device: Optional[str] = None,
    run_full_ocr: bool = False,
    allow_synthetic: bool = False,
    seed: int = 42,
    layoutlm_runner: Optional[
        Callable[[AlignedReceipt], Tuple[Dict[str, Optional[str]], Dict[str, float]]]
    ] = None,
) -> Dict[str, Any]:
    """
    Runs honest side-by-side benchmark comparison between Tesseract+Rules Baseline
    and LayoutLMv3. Never fabricates metrics when LayoutLMv3 weights or runtime are missing.
    """
    resolved_device = device or detect_default_device()
    checkpoint_path = Path(checkpoint_dir)
    test_path = Path(test_jsonl)
    if not test_path.exists():
        raise FileNotFoundError(f"Benchmark split manifest not found: {test_jsonl}")

    receipts = load_aligned_receipts(test_path)
    split_hash = compute_split_hash(receipts)
    is_synthetic = any(r.is_synthetic for r in receipts)

    if is_synthetic and not allow_synthetic:
        raise RuntimeError(
            f"Refusing to publish SROIE benchmark numbers on synthetic dataset '{test_jsonl}' "
            "without allow_synthetic=True (--allow-synthetic)."
        )

    effective_full_ocr = False
    if run_full_ocr:
        ocr_ok, _, ocr_msg = get_tesseract_status()
        if not ocr_ok:
            raise RuntimeError(f"--run-full-ocr requested but Tesseract is unavailable: {ocr_msg}")
        effective_full_ocr = True

    baseline_mode = (
        "full_tesseract_ocr_plus_rules"
        if effective_full_ocr
        else "prepared_sroie_ocr_tokens_rules_only"
    )

    ckpt_meta = None
    meta_file = checkpoint_path / "checkpoint_metadata.json"
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as mf:
                ckpt_meta = json.load(mf)
        except Exception:
            ckpt_meta = None

    # Resolve LayoutLMv3 evaluator
    layoutlm_evaluated = False
    layoutlm_unavailable_reason = ""
    tokenizer = None
    image_processor = None
    model = None

    if layoutlm_runner is not None:
        layoutlm_evaluated = True
    else:
        valid_ckpt, ckpt_reason = validate_checkpoint_directory(checkpoint_path)
        if not valid_ckpt:
            layoutlm_unavailable_reason = ckpt_reason
        elif not HAS_TORCH or torch is None:
            layoutlm_unavailable_reason = "PyTorch/Transformers runtime unavailable on this host."
        else:
            try:
                processor = AutoProcessor.from_pretrained(str(checkpoint_path), apply_ocr=False)
                tokenizer = processor.tokenizer
                image_processor = getattr(processor, "image_processor", None)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path), add_prefix_space=True)
                try:
                    image_processor = AutoImageProcessor.from_pretrained(
                        str(checkpoint_path), apply_ocr=False
                    )
                except Exception:
                    image_processor = None

            model = LayoutLMv3ForTokenClassification.from_pretrained(str(checkpoint_path))
            model.to(resolved_device)
            model.eval()
            layoutlm_evaluated = True

    baseline_f1s: Dict[str, List[float]] = {f: [] for f in BENCHMARK_FIELDS}
    baseline_exact: Dict[str, int] = {f: 0 for f in BENCHMARK_FIELDS}
    baseline_doc_exact = 0
    baseline_ocr_lats: List[float] = []
    baseline_parse_lats: List[float] = []
    baseline_total_lats: List[float] = []

    layoutlm_f1s: Dict[str, List[float]] = {f: [] for f in BENCHMARK_FIELDS}
    layoutlm_exact: Dict[str, int] = {f: 0 for f in BENCHMARK_FIELDS}
    layoutlm_doc_exact = 0
    layoutlm_ocr_lats: List[float] = []
    layoutlm_prep_lats: List[float] = []
    layoutlm_infer_lats: List[float] = []
    layoutlm_total_lats: List[float] = []

    layoutlm_better_cases: List[Dict[str, Any]] = []
    baseline_better_cases: List[Dict[str, Any]] = []

    n = max(1, len(receipts))
    for r in receipts:
        ref = r.ground_truth

        # 1. Baseline evaluation (always runs cleanly)
        b_pred, b_timings = run_baseline_extraction(r, run_full_ocr=effective_full_ocr)
        baseline_ocr_lats.append(b_timings["ocr_latency_ms"])
        baseline_parse_lats.append(b_timings["rule_parse_latency_ms"])
        baseline_total_lats.append(b_timings["total_pipeline_latency_ms"])

        b_doc_match = True
        b_scores: List[float] = []
        for f in BENCHMARK_FIELDS:
            b_f1 = compute_field_f1(b_pred.get(f), ref.get(f), field_name=f)
            baseline_f1s[f].append(b_f1)
            b_scores.append(b_f1)

            norm_b = normalize_entity_value(canonicalize_field_for_eval(f, b_pred.get(f)))
            norm_ref = normalize_entity_value(canonicalize_field_for_eval(f, ref.get(f)))
            if norm_b == norm_ref:
                baseline_exact[f] += 1
            else:
                b_doc_match = False
        if b_doc_match:
            baseline_doc_exact += 1

        # 2. LayoutLMv3 evaluation (ONLY when valid checkpoint/runner exists)
        if layoutlm_evaluated:
            if layoutlm_runner is not None:
                l_pred, l_timings = layoutlm_runner(r)
            else:
                l_pred, l_timings = _run_layoutlmv3_on_receipt(
                    r,
                    model=model,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    device=resolved_device,
                    run_full_ocr=effective_full_ocr,
                )

            layoutlm_ocr_lats.append(float(l_timings.get("ocr_latency_ms", 0.0)))
            layoutlm_prep_lats.append(float(l_timings.get("feature_prep_latency_ms", 0.0)))
            layoutlm_infer_lats.append(float(l_timings.get("model_inference_latency_ms", 0.0)))
            layoutlm_total_lats.append(float(l_timings.get("total_pipeline_latency_ms", 0.0)))

            l_doc_match = True
            l_scores: List[float] = []
            for f in BENCHMARK_FIELDS:
                l_f1 = compute_field_f1(l_pred.get(f), ref.get(f), field_name=f)
                layoutlm_f1s[f].append(l_f1)
                l_scores.append(l_f1)

                norm_l = normalize_entity_value(canonicalize_field_for_eval(f, l_pred.get(f)))
                norm_ref = normalize_entity_value(canonicalize_field_for_eval(f, ref.get(f)))
                if norm_l == norm_ref:
                    layoutlm_exact[f] += 1
                else:
                    l_doc_match = False
            if l_doc_match:
                layoutlm_doc_exact += 1

            avg_b = sum(b_scores) / len(BENCHMARK_FIELDS)
            avg_l = sum(l_scores) / len(BENCHMARK_FIELDS)
            if avg_l > avg_b + 0.15 and len(layoutlm_better_cases) < 5:
                layoutlm_better_cases.append(
                    {
                        "receipt_id": r.receipt_id,
                        "ground_truth": ref,
                        "baseline_pred": b_pred,
                        "layoutlm_pred": l_pred,
                    }
                )
            elif avg_b > avg_l + 0.15 and len(baseline_better_cases) < 5:
                baseline_better_cases.append(
                    {
                        "receipt_id": r.receipt_id,
                        "ground_truth": ref,
                        "baseline_pred": b_pred,
                        "layoutlm_pred": l_pred,
                    }
                )

    baseline_lat_summary = compute_latency_summary(baseline_total_lats)
    baseline_summary = {
        "status": "EVALUATED",
        "evaluation_mode": baseline_mode,
        "field_f1": {f: round(sum(baseline_f1s[f]) / n, 4) for f in BENCHMARK_FIELDS},
        "field_exact_match": {f: round(baseline_exact[f] / n, 4) for f in BENCHMARK_FIELDS},
        "macro_f1": round(sum(sum(baseline_f1s[f]) for f in BENCHMARK_FIELDS) / (4 * n), 4),
        "document_exact_match": round(baseline_doc_exact / n, 4),
        "ocr_latency_ms": round(sum(baseline_ocr_lats) / n, 2),
        "rule_parse_latency_ms": round(sum(baseline_parse_lats) / n, 2),
        "total_pipeline_latency_ms": baseline_lat_summary["avg_ms"],
        "avg_latency_ms": baseline_lat_summary["avg_ms"],
        "latency_percentiles_ms": baseline_lat_summary,
    }

    ckpt_size_mb = compute_checkpoint_disk_size_mb(checkpoint_path)

    if layoutlm_evaluated:
        lm_pipeline_lat = compute_latency_summary(layoutlm_total_lats)
        lm_model_lat = compute_latency_summary(layoutlm_infer_lats)
        layoutlm_summary = {
            "status": "EVALUATED",
            "evaluation_mode": (
                "full_tesseract_ocr_plus_layoutlmv3"
                if effective_full_ocr
                else "prepared_sroie_ocr_tokens_plus_image_layoutlmv3"
            ),
            "field_f1": {f: round(sum(layoutlm_f1s[f]) / n, 4) for f in BENCHMARK_FIELDS},
            "field_exact_match": {f: round(layoutlm_exact[f] / n, 4) for f in BENCHMARK_FIELDS},
            "macro_f1": round(sum(sum(layoutlm_f1s[f]) for f in BENCHMARK_FIELDS) / (4 * n), 4),
            "document_exact_match": round(layoutlm_doc_exact / n, 4),
            "ocr_latency_ms": round(sum(layoutlm_ocr_lats) / n, 2),
            "feature_prep_latency_ms": round(sum(layoutlm_prep_lats) / n, 2),
            "model_inference_latency_ms": lm_model_lat["avg_ms"],
            "model_latency_percentiles_ms": lm_model_lat,
            "total_pipeline_latency_ms": lm_pipeline_lat["avg_ms"],
            "avg_latency_ms": lm_pipeline_lat["avg_ms"],
            "latency_percentiles_ms": lm_pipeline_lat,
            "checkpoint_size_mb": ckpt_size_mb,
        }
        comparison_status = "COMPLETE"
    else:
        layoutlm_summary = _build_not_evaluated_summary(layoutlm_unavailable_reason)
        layoutlm_summary["checkpoint_size_mb"] = ckpt_size_mb
        comparison_status = "INCOMPLETE_MISSING_LAYOUTLMV3"

    comparison_res: Dict[str, Any] = {
        "benchmark_status": comparison_status,
        "dataset": str(test_path),
        "split_hash": split_hash,
        "is_synthetic": is_synthetic,
        "seed": seed,
        "device": resolved_device,
        "checkpoint": str(checkpoint_path),
        "checkpoint_metadata": ckpt_meta,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_samples": len(receipts),
        "baseline": baseline_summary,
        "layoutlmv3": layoutlm_summary,
        "error_analysis": {
            "layoutlmv3_wins": layoutlm_better_cases,
            "baseline_wins": baseline_better_cases,
        },
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "benchmark_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison_res, f, indent=2)

    title_suffix = " (SYNTHETIC SMOKE TEST)" if is_synthetic else " (SROIE)"
    md_lines = [
        f"# DocuMind Model Comparison: Tesseract Rule Baseline vs LayoutLMv3{title_suffix}",
        "",
    ]
    if is_synthetic:
        md_lines.append(
            "> **WARNING:** Benchmark executed on synthetic smoke-test receipts. Numbers do NOT represent official SROIE performance.\n"
        )
    if not layoutlm_evaluated:
        md_lines.append(
            f"> **INCOMPLETE BENCHMARK (`NOT_EVALUATED`):** LayoutLMv3 was not evaluated ({layoutlm_unavailable_reason}). No LayoutLMv3 metrics were fabricated.\n"
        )

    md_lines.extend(
        [
            f"- **Benchmark Status**: `{comparison_status}`",
            f"- **Dataset Split**: `{test_path}` (`split_hash={split_hash}`)",
            f"- **Samples Evaluated**: {len(receipts)}",
            f"- **Baseline Mode**: `{baseline_mode}`",
            f"- **Device**: `{resolved_device}`",
            f"- **Seed**: `{seed}`",
            f"- **Checkpoint**: `{checkpoint_path}`",
            "",
            "| Field / Metric | Tesseract + Rules (Phase 2A) | LayoutLMv3 (Fine-Tuned) | Delta (LMv3 - Baseline) |",
            "| :--- | :--- | :--- | :--- |",
        ]
    )

    for f in BENCHMARK_FIELDS:
        b_v = baseline_summary["field_f1"][f]
        l_v = layoutlm_summary["field_f1"][f]
        if l_v is None:
            md_lines.append(f"| **{f.upper()} F1** | {b_v:.4f} | `NOT_EVALUATED` | N/A |")
        else:
            diff = l_v - b_v
            md_lines.append(f"| **{f.upper()} F1** | {b_v:.4f} | {l_v:.4f} | {diff:+.4f} |")

    b_m = baseline_summary["macro_f1"]
    l_m = layoutlm_summary["macro_f1"]
    if l_m is None:
        md_lines.append(f"| **Macro F1** | {b_m:.4f} | `NOT_EVALUATED` | N/A |")
    else:
        md_lines.append(f"| **Macro F1** | {b_m:.4f} | {l_m:.4f} | {l_m - b_m:+.4f} |")

    b_d = baseline_summary["document_exact_match"]
    l_d = layoutlm_summary["document_exact_match"]
    if l_d is None:
        md_lines.append(f"| **Document Exact Match** | {b_d:.2%} | `NOT_EVALUATED` | N/A |")
    else:
        md_lines.append(
            f"| **Document Exact Match** | {b_d:.2%} | {l_d:.2%} | {(l_d - b_d) * 100:+.2f} pp |"
        )

    b_lat = baseline_summary["total_pipeline_latency_ms"]
    b_pct = baseline_summary.get("latency_percentiles_ms") or {}
    l_lat = layoutlm_summary["total_pipeline_latency_ms"]
    l_pct = layoutlm_summary.get("latency_percentiles_ms") or {}
    if l_lat is None:
        md_lines.append(
            f"| **Total Pipeline Latency (avg / p95)** | {b_lat:.2f} ms (p95: {b_pct.get('p95_ms', b_lat):.2f} ms) | `NOT_EVALUATED` | N/A |"
        )
    else:
        md_lines.append(
            f"| **Total Pipeline Latency (avg / p95)** | {b_lat:.2f} ms (p95: {b_pct.get('p95_ms', b_lat):.2f} ms) | {l_lat:.2f} ms (p95: {l_pct.get('p95_ms', l_lat):.2f} ms; model-only: {layoutlm_summary['model_inference_latency_ms']:.2f} ms) | {l_lat - b_lat:+.2f} ms |"
        )

    md_lines.append(
        f"| **Checkpoint Disk Size** | Rules (~0 MB) | {f'{ckpt_size_mb:.2f} MB' if ckpt_size_mb is not None else 'N/A'} | - |"
    )

    if layoutlm_better_cases:
        md_lines.extend(["", "## Qualitative Inspection: LayoutLMv3 Advantages"])
        for case in layoutlm_better_cases[:3]:
            md_lines.append(f"### Receipt `{case['receipt_id']}`")
            md_lines.append(f"- **Ground Truth**: `{case['ground_truth']}`")
            md_lines.append(f"- **Baseline Pred**: `{case['baseline_pred']}`")
            md_lines.append(f"- **LayoutLMv3 Pred**: `{case['layoutlm_pred']}`")

    if baseline_better_cases:
        md_lines.extend(["", "## Qualitative Error Analysis: Baseline Advantages / LayoutLMv3 Misses"])
        for case in baseline_better_cases[:3]:
            md_lines.append(f"### Receipt `{case['receipt_id']}`")
            md_lines.append(f"- **Ground Truth**: `{case['ground_truth']}`")
            md_lines.append(f"- **Baseline Pred**: `{case['baseline_pred']}`")
            md_lines.append(f"- **LayoutLMv3 Pred**: `{case['layoutlm_pred']}`")

    with open(out_dir / "benchmark_comparison.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    logger.info("Benchmark comparison written cleanly to %s", out_dir)
    return comparison_res


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LayoutLMv3 vs Tesseract Rule Baseline"
    )
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/layoutlmv3-sroie-best")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--test-jsonl", type=str, default="data/sroie_processed/test.jsonl")
    parser.add_argument("--output-dir", type=str, default="ml/evaluation/results")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-full-ocr",
        action="store_true",
        help="Execute live Tesseract OCR on images rather than prepared SROIE OCR tokens",
    )
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()

    split_jsonl = args.test_jsonl
    if args.data_dir:
        cand = Path(args.data_dir) / f"{args.split}.jsonl"
        if args.split == "validation" and not cand.exists():
            cand = Path(args.data_dir) / "val.jsonl"
        split_jsonl = str(cand)

    res = compare_models(
        checkpoint_dir=args.checkpoint_dir,
        test_jsonl=split_jsonl,
        output_dir=args.output_dir,
        device=args.device,
        run_full_ocr=args.run_full_ocr,
        allow_synthetic=args.allow_synthetic,
        seed=args.seed,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
