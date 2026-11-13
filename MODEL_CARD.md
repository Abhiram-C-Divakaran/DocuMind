# DocuMind LayoutLMv3 Model Card (`microsoft/layoutlmv3-base` — SROIE Receipt KIE)

## 1. Model Overview

- **Task**: Token classification / Key Information Extraction (KIE) on scanned receipts
- **Base Architecture**: `microsoft/layoutlmv3-base` (`LayoutLMv3ForTokenClassification`, multimodal vision + text + 2D spatial layout transformer)
- **Label Schema (9 Canonical IOB2 Classes)**:
  - `0`: `O`
  - `1`: `B-COMPANY`, `2`: `I-COMPANY`
  - `3`: `B-DATE`, `4`: `I-DATE`
  - `5`: `B-ADDRESS`, `6`: `I-ADDRESS`
  - `7`: `B-TOTAL`, `8`: `I-TOTAL`
- **Unified Feature Pipeline**: [`ml/training/dataset.py`](ml/training/dataset.py) (`prepare_layoutlmv3_inputs`) shared across training, held-out evaluation, comparative benchmarking, and FastAPI inference (`ml/extractors/layoutlmv3_extractor.py`).
  - Document images resized to `224x224` RGB (never substituted with blank/zero images when decoding fails).
  - Token bounding boxes normalized to integer `0–1000` coordinates (`[x0, y0, x1, y1]`).
  - Special tokens receive `[0, 0, 0, 0]` boxes and `-100` label IDs; subword continuation tokens inherit the parent word bounding box and receive `-100` label IDs.

---

## 2. Dataset & Split Hygiene

- **Primary Benchmark Dataset**: `arvindrajan92/sroie_document_understanding` on Hugging Face Hub (or local SROIE2019 benchmark directory via `--local-dir`).
- **Inspected HF Hub Schema (Verified Directly via HF API/Datasets Server)**: Single `train` split containing **652 examples** (`MIT` license, `download_size=217,146,103` bytes, `dataset_size=267,317,016` bytes; no official test split published on HF). Each record contains `image` RGB scan + `ocr` list of segments with `text`, 4-corner pixel `box`, and `label` in `['address', 'company', 'date', 'line_description', 'line_total', 'other', 'total']`.
- **Deduplication & Splitting (`ml/data/prepare_dataset.py`)**:
  - Single-split sources are deduplicated by image SHA-256 hash prior to splitting (`seed=42`, `80% train / 10% validation / 10% test`) and explicitly record `has_official_test_split: false`.
  - Cross-split SHA-256 image & receipt-ID leakage checks are recorded in `dataset_manifest.json` and `validation_report.md`.
  - Corrupted or missing images are recorded in `excluded_samples` and omitted rather than replaced.
  - Synthetic receipt generation (`--allow-synthetic`) is strictly isolated to automated unit tests / explicit `--smoke-test` checks and flagged with `is_synthetic: true`.

---

## 3. Training Configuration & Experiment Tracking

- **Default Training Config (`ml/training/config.py`)**:
  - Optimizer: `AdamW` (`lr=2e-5`, `weight_decay=0.01`, `eps=1e-8`)
  - Scheduler: Linear warmup (`warmup_ratio=0.1`) + linear decay
  - Epochs: `3` default (configurable via `--epochs`)
  - Sequence Length / Image Size: `max_seq_length=512`, `image_size=224`
  - Gradient Clipping: `max_grad_norm=1.0`, finite loss verification (`torch.isfinite`)
  - Mixed Precision: CUDA AMP `fp16` when CUDA is available
  - Early Stopping & Model Selection: Validation strict IOB2 F1 evaluated each epoch (`early_stopping_patience=3`)
- **Experiment Tracking (`ml/training/experiment_tracker.py`)**:
  - Logs `experiment_id`, dataset provenance & split hashes, hyperparameters, seed, hardware & dependency versions, per-epoch train/val loss and per-entity F1, best checkpoint location, and total duration to `ml/experiments/<experiment_id>/experiment_run.json` and `ml/experiments/latest_experiment.json`.
  - Optional MLflow tracking supported via `--enable-mlflow`.

---

## 4. Current Training & Evaluation Status (Honest Reporting)

> **Status: Not Yet Trained / Evaluated on Held-Out SROIE Test Split (`NOT_EVALUATED`)**

No real fine-tuned `LayoutLMv3` weights (`model.safetensors`) are included in this repository, and no held-out LayoutLMv3 accuracy or latency numbers are claimed.

- **Local Environment Blockers Observed**:
  - Host GPU is an **NVIDIA GeForce RTX 4050 Laptop GPU** (`6141 MiB` VRAM, CUDA `13.1`), running **Python 3.14.3**, where Windows Application Control policy (`WinError 4551`) blocks native extension DLLs required by `torch` (`torch_python.dll`), `pandas`, and `pyarrow` / `datasets`.
  - WSL is not installed, Docker Desktop daemon is stopped, and `tesseract.exe` is not installed on the local Windows `PATH`.
- **Reproducible Cloud GPU & Activation Path**:
  - [`ml/training/DocuMind_LayoutLMv3_Training.ipynb`](ml/training/DocuMind_LayoutLMv3_Training.ipynb) executes the full workflow end-to-end on Google Colab (T4 GPU): CUDA/disk checks, real SROIE data preparation, isolated `/tmp` `--smoke-test`, 5-epoch LayoutLMv3 fine-tuning with experiment tracking, held-out test evaluation, `--run-full-ocr` baseline comparison, `python -m ml.training.import_checkpoint` sanity checks, and archive export (`documind_layoutlmv3_best.zip`).
  - See [`docs/TRAINING_RESULTS.md`](docs/TRAINING_RESULTS.md) for the full Phase 5A execution & blocker report plus exact post-Colab activation commands (`import_checkpoint --target-dir checkpoints/layoutlmv3-sroie-best --apply-env`).

### Evaluation Metrics Tracked Once Executed (`ml/evaluation/evaluate.py` & `compare_models.py`)
- Strict IOB2 per-entity precision, recall, F1, support (`COMPANY`, `DATE`, `ADDRESS`, `TOTAL`), overall micro F1, and 4-entity macro F1 (`seqeval`).
- Canonicalized field exact-match and complete-document exact-match rates.
- Latency mean plus `p50`, `p95`, and `p99` percentiles, separating OCR latency, feature preparation latency, and model inference latency.
- Checkpoint disk size (`MB`) and parameter footprint (`total_parameters`, `trainable_parameters`, estimated `fp32` size).

---

## 5. Serving Safeguards, Limitations & Intended Use

1. **Default Extraction Method**: `TESSERACT_BASELINE` (OCR + deterministic rules) remains the active default until a verified trained checkpoint and runtime are present.
2. **Model Readiness Gating (`GET /api/v1/models/readiness`)**:
   - Validates `ENABLE_ML_INFERENCE`, PyTorch runtime availability, checkpoint file presence (`config.json`, tokenizer files, non-empty `model.safetensors`/`pytorch_model.bin`), `model_type == "layoutlmv3"`, and exact 9-class `id2label` parity.
   - Checkpoints marked `is_synthetic: true` are never presented as production-ready (`layoutlmv3_is_production_ready: false`).
   - UI upload selector (`DocumentUploader.tsx`) disables `LayoutLMv3 Transformer` with an explicit reason whenever readiness checks fail.
3. **Failure Isolation**: Missing/invalid LayoutLMv3 checkpoints or PyTorch runtime errors do not interfere with `TESSERACT_BASELINE` or `PLACEHOLDER` pipelines.
4. **Known Limitations**:
   - **Sequence Truncation**: Receipts exceeding 512 WordPiece/BPE tokens are truncated at `max_seq_length=512`.
   - **Single-Page Inference Limit**: `LayoutLMv3Extractor` renders page 1 for PDFs during transformer inference (`max_pages=1`).
   - **Domain Shift**: SROIE primarily represents Malaysian retail thermal receipts; invoices or non-English receipts require human review (`PENDING_REVIEW` via Phase 3A) and domain fine-tuning.

