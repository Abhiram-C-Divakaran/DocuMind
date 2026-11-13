# DocuMind ML Subsystem (Phase 3B — SROIE, LayoutLMv3 Training, Tracking & Readiness)

This package contains the machine learning subsystem for DocuMind (see also [`MODEL_CARD.md`](../MODEL_CARD.md) for architecture, dataset schema, training status, and limitations):
1. **Phase 2A**: Tesseract OCR and rule-based receipt extraction pipeline (`TesseractBaselineExtractor`).
2. **Phase 2B/2C**: Reproducible deep learning data ingestion, unified multimodal preprocessing, LayoutLMv3 fine-tuning on real SROIE receipt images, held-out evaluation, and comparative benchmarking.
3. **Phase 3A**: Human-in-the-loop validated active-learning feedback export pipeline (`ml/feedback/`) producing audited JSONL datasets exclusively from `APPROVED` reviews without triggering unsafe automatic retraining.
4. **Phase 3B**: Structured JSON/optional MLflow experiment tracking (`ml/training/experiment_tracker.py`), macro F1 & percentile latency (`p50`/`p95`/`p99`) reporting, process-level model weight caching (`get_cached_layoutlmv3_extractor`), and FastAPI readiness gating (`/api/v1/models/readiness`).

---

## 1. Directory Structure

```text
ml/
├── data/
│   ├── annotation_alignment.py  # SROIE word-splitting, span alignment & BIO tagging
│   ├── sroie_loader.py          # Verified SROIE loader (HF / local split dirs) & PNG smoke generator
│   ├── prepare_dataset.py       # Reproducible splitting, SHA-256 deduplication & leakage checks
│   └── dataset_validation.py    # Audit statistics, image integrity & alignment reports
├── feedback/
│   ├── feedback_loader.py       # Queries DB for APPROVED DocumentReviews (tracks excluded counts)
│   ├── feedback_validator.py    # Pydantic validation, SHA-256 deduplication & split contamination guard
│   ├── feedback_quality.py      # Conservative OCR token alignment check (ALIGNED_CONFIDENT vs manual)
│   └── export_annotations.py    # CLI & async helper generating feedback_export.jsonl & summaries
├── training/
│   ├── config.py                # Hyperparameters dataclass & CUDA/MPS/CPU device detection
│   ├── dataset.py               # Shared prepare_layoutlmv3_inputs() preprocessing & PyTorch Dataset
│   ├── experiment_tracker.py    # Structured JSON run tracker & optional MLflow mirroring
│   ├── train_layoutlmv3.py      # Reproducible fine-tuning (AMP fp16, early stopping, finite loss check)
│   └── DocuMind_LayoutLMv3_Training.ipynb  # Ready-to-run Google Colab T4 GPU notebook
├── evaluation/
│   ├── metrics.py               # Seqeval strict IOB2 entity F1, document exact match & footprint
│   ├── evaluate.py              # Held-out split checkpoint evaluation (evaluation_report.json/.md)
│   └── compare_models.py        # Honest side-by-side comparison vs. Tesseract+Rules baseline
├── inference/
│   └── decoder.py               # Token logit BIO decoding & entity confidence scoring
├── extractors/
│   ├── layoutlmv3_extractor.py  # Strict checkpoint validation & multimodal runtime extractor
│   ├── tesseract_baseline.py    # Phase 2A OCR + Rules baseline extractor
│   └── placeholder.py           # Phase 1 transparent placeholder stub
├── ocr/
│   ├── engine.py                # Tesseract OCR runner, tokenization, 0-1000 bbox normalization
│   ├── pdf_converter.py         # PDF page rasterizer via PyMuPDF
│   └── preprocessor.py          # Deskew, CLAHE contrast boost, adaptive thresholding
└── extraction/
    ├── pipeline.py              # Rule-based receipt extraction orchestrator
    ├── vendor.py                # Vendor name extraction heuristics
    ├── address.py               # Spatial address heuristics
    ├── date.py                  # Multi-format date regex parser
    └── financial.py             # Monetary values, taxes, and currency codes
```

---

## 2. Canonical BIO Annotation Schema

The SROIE token classification task maps document tokens into 9 canonical classes:

| Class ID | BIO Label | Entity Description |
| :---: | :--- | :--- |
| `0` | `O` | Outside / Non-entity token |
| `1` | `B-COMPANY` | Beginning of Merchant / Vendor Name |
| `2` | `I-COMPANY` | Inside continuation of Merchant / Vendor Name |
| `3` | `B-DATE` | Beginning of Receipt Date |
| `4` | `I-DATE` | Inside continuation of Receipt Date |
| `5` | `B-ADDRESS` | Beginning of Merchant Address |
| `6` | `I-ADDRESS` | Inside continuation of Merchant Address |
| `7` | `B-TOTAL` | Beginning of Final Total Amount |
| `8` | `I-TOTAL` | Inside continuation of Final Total Amount |

### Unified Preprocessing (`prepare_layoutlmv3_inputs`)
Training, held-out evaluation, benchmark comparison, and FastAPI inference all share `prepare_layoutlmv3_inputs()` in `ml/training/dataset.py`:
- Real document images are resized to `224x224` RGB and normalized (`mean=0.5`, `std=0.5`). Zero-filled or blank substitute canvases are never used when an image fails to load.
- Special tokens receive `[0, 0, 0, 0]` bounding boxes and `-100` labels; subword continuations share the parent word's `0–1000` bounding box and receive `-100` labels.

---

## 3. Dataset Ingestion & Preparation

By default, `prepare_dataset.py` ingests verified SROIE receipts with genuine images from Hugging Face (`arvindrajan92/sroie_document_understanding`) or a local SROIE directory (`--local-dir`). Missing or corrupted images are explicitly excluded and audited rather than replaced with blank images.

```bash
# 1. Prepare real SROIE dataset from Hugging Face Hub:
python -m ml.data.prepare_dataset \
    --hf-dataset arvindrajan92/sroie_document_understanding \
    --output-dir data/sroie_processed \
    --seed 42

# 2. Or prepare from a local SROIE benchmark folder (flat or train/+test/ layout):
python -m ml.data.prepare_dataset \
    --local-dir /path/to/SROIE2019 \
    --output-dir data/sroie_processed \
    --seed 42
```

Outputs generated in `data/sroie_processed/`:
- `train.jsonl`, `validation.jsonl`, `test.jsonl`
- `dataset_manifest.json` (source, SHA-256 split hashes, leakage audit, excluded counts)
- `validation_report.json` and `validation_report.md`

Full dataset preparation and full training refuse to silently fall back to synthetic data. Synthetic generation (`--allow-synthetic`) is strictly reserved for unit tests and explicit `--smoke-test` execution.

---

## 4. Training Pipeline (Colab GPU or Local Linux/macOS)

Open `ml/training/DocuMind_LayoutLMv3_Training.ipynb` in Google Colab (T4 GPU runtime) or run directly from CLI:

```bash
# Full GPU fine-tuning on prepared real SROIE data (default --epochs 3, configurable):
python -m ml.training.train_layoutlmv3 \
    --data-dir data/sroie_processed \
    --output-dir checkpoints/layoutlmv3-sroie \
    --best-model-dir checkpoints/layoutlmv3-sroie-best \
    --model-name microsoft/layoutlmv3-base \
    --epochs 3 \
    --batch-size 4 \
    --grad-accum 2 \
    --lr 2e-5 \
    --patience 3 \
    --experiment-dir ml/experiments \
    --seed 42
```

Features:
- Fixed random seeding across Python, NumPy, and PyTorch (`--seed`)
- CUDA AMP (`fp16`) enabled automatically on CUDA devices (disable with `--no-fp16`)
- Gradient accumulation, linear warmup decay, gradient clipping, finite loss guard, and early stopping
- Structured run tracking (`ml/experiments/<experiment_id>/experiment_run.json` + optional `--enable-mlflow`)
- Best checkpoint selection on validation strict BIO F1, saving `checkpoint_metadata.json`, tokenizer/processor configs, and `training_state.pt` (`--resume-from-checkpoint`)

For quick pipeline verification with synthetic rendered PNGs:

```bash
python -m ml.training.train_layoutlmv3 --smoke-test --epochs 1 --batch-size 2
```

---

## 5. Held-Out Evaluation & Scientific Benchmarking

```bash
# Held-out test evaluation (writes evaluation_report.json and evaluation_report.md):
python -m ml.evaluation.evaluate \
    --checkpoint-dir checkpoints/layoutlmv3-sroie-best \
    --data-dir data/sroie_processed \
    --split test \
    --output-dir ml/evaluation/results

# Side-by-side comparison vs. Phase 2A Tesseract+Rules baseline:
# Add --run-full-ocr when tesseract-ocr is installed to measure true end-to-end OCR+extraction latency.
python -m ml.evaluation.compare_models \
    --checkpoint-dir checkpoints/layoutlmv3-sroie-best \
    --data-dir data/sroie_processed \
    --split test \
    --output-dir ml/evaluation/results \
    --run-full-ocr
```

### Reporting Integrity Rules
- If the LayoutLMv3 checkpoint is missing, invalid, or cannot be loaded, `evaluate.py` and `compare_models.py` mark LayoutLMv3 as `NOT_EVALUATED` with `null` metrics rather than reporting fabricated `0.0` F1 or `0.0 ms` latency.
- Full-pipeline latency (`ocr_latency_ms` + `feature_prep_latency_ms` + `model_inference_latency_ms`) is reported separately from model-only or rule-parse-only latency.
- Synthetic smoke datasets are explicitly rejected by `evaluate.py` and `compare_models.py` unless `--allow-synthetic` is passed, in which case all reports are labeled `SYNTHETIC SMOKE TEST`.

---

## 6. Phase 3A Active-Learning Feedback Export (`ml/feedback/`)

Only human-reviewed records with `review_status == APPROVED` are eligible for export:

```bash
python -m ml.feedback.export_annotations \
    --output-dir data/feedback_exports \
    --reference-splits data/sroie_processed/test.jsonl data/sroie_processed/validation.jsonl
```

Safeguards enforced automatically:
1. **Approval Gating**: `PENDING_REVIEW`, `IN_REVIEW`, `REJECTED`, and `NOT_REQUIRED` documents are strictly excluded and reported in `excluded_by_review_status`.
2. **Integrity & Deduplication**: Validates `corrected_values` via `ReviewCorrectionPayload`, verifies source file existence inside `UPLOAD_DIR`, and removes duplicate image SHA-256 hashes.
3. **Train/Test Contamination Prevention**: Hashes are cross-checked against held-out `test.jsonl` and `validation.jsonl` splits before writing `feedback_export.jsonl`.
4. **Conservative Token Alignment**: Records are labeled `ALIGNED_CONFIDENT` only when OCR words + bounding boxes unambiguously match corrected values; otherwise they are marked `NEEDS_MANUAL_ANNOTATION`. Document-level strings are never blindly treated as BIO token labels, and automatic retraining is disabled (`automatic_retraining_launched: false`).
