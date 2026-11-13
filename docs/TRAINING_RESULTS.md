# DocuMind Phase 5A — LayoutLMv3 Execution, Blocker & Activation Report

> **Honest Training & Benchmark Status:** `NOT_EVALUATED`  
> Genuine `microsoft/layoutlmv3-base` fine-tuning, held-out SROIE evaluation, and trained weight generation **could not execute on the local Windows host** due to OS binary code-integrity blocks (`WinError 4551`). No synthetic metrics or unverified checkpoints were fabricated. All ML training, evaluation, comparison, checkpoint-import, and CI execution blockers uncovered during audit were repaired and verified via automated tests.

---

## 1. Environment & Execution Reality Audit

| Component | Observed State | Impact |
| :--- | :--- | :--- |
| **OS / Architecture** | Windows 11 (`Windows-11-10.0.26300-SP0`), `194.31 GB` free disk | Local host runtime |
| **Python Runtime** | `Python 3.14.3` (`C:\Python314\python.exe`, `.venv\Scripts\python.exe`; no other installed Python via `py -0p`) | Native wheel compatibility limited |
| **GPU Hardware** | **NVIDIA GeForce RTX 4050 Laptop GPU** (`6141 MiB` VRAM, Driver `591.74`, CUDA `13.1`) | Physical GPU present |
| **PyTorch / PyArrow / Datasets** | `torch 2.15.0`, `pyarrow 26.0.1`, `datasets 5.2.0` -> **`OSError: [WinError 4551] An Application Control policy has blocked this file`** (`torch_python.dll`, `pyarrow.lib`) | **Blocked** loading PyTorch & Hugging Face `datasets` locally without bypassing OS security controls (prohibited) |
| **Container / Linux Subsystems** | WSL not installed (`wsl -l -v`); Docker Desktop engine daemon stopped | Local Linux container GPU execution unavailable |
| **Tesseract OCR** | Binary absent on Windows `PATH` (`get_tesseract_status() -> False`) | `--run-full-ocr` baseline benchmark unavailable locally; tested cleanly in Ubuntu CI |
| **Transformers / Seqeval / OpenCV / PIL** | `transformers 5.22.2`, `seqeval 1.2.2`, `huggingface_hub 2.4.0`, `cv2 5.0.0`, `PIL 13.0.0`, `fitz 2.1` import cleanly | Full non-torch test suite, static checks, API & Playwright E2E verified locally |

---

## 2. Verified Hugging Face SROIE Dataset Inspection

Directly verified against Hugging Face Hub API & Datasets Server (`arvindrajan92/sroie_document_understanding`):

- **Repository & License**: `arvindrajan92/sroie_document_understanding` (`MIT` license, `download_size=217,146,103` bytes, `dataset_size=267,317,016` bytes)
- **Splits Present**: **Single `train` split only (`652` examples)** — no official held-out `validation` or `test` split is published in this HF dataset repository. Accordingly, [`ml/data/prepare_dataset.py`](../ml/data/prepare_dataset.py) performs exact `image_sha256` deduplication first (`seed=42`, `80% train / 10% validation / 10% test`), verifies zero cross-split SHA-256 leakage, and records `has_official_test_split: false`.
- **Actual Record Schema**:
  - `image`: RGB image (e.g., row `0` is `463x1013`)
  - `ocr`: list of segments containing 4-corner polygon `box` (`[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]` float pixel coordinates), `text`, and `label`.
- **Observed Segment Labels (first 50 rows)**:
  `['address', 'company', 'date', 'line_description', 'line_total', 'other', 'total']` (`line_description`, `line_total`, and `other` map cleanly to `O` under our canonical 9-class IOB2 schema).
- **Observed Receipt Ground-Truth Format Quirks**:
  - Row `0`: `company='BOOK TA .K(TAMAN DAYA) SDN BHD'`, `date='25/12/2018 8:13:39 PM'`, `total='9.00'`
  - Row `1`: `company='INDAH GIFT & HOME DECO'`, `date='19/10/2018 20:49:59 #01'`, `total='60.30'`
  - Row `2`: `company='MR D.I.Y. (JOHOR) SDN BHD'`, `date='12-01-19 21:13 SH01 ZK09'`, `total='RM 33.90'`

---

## 3. Execution Blockers & Pipeline Defects Fixed in Phase 5A

Auditing the full end-to-end training, evaluation, comparison, checkpoint-import, and CI pipelines uncovered the following concrete defects that were fixed and covered by unit tests:

1. **Checkpoint Import Crash on Real PyTorch Environments ([`ml/training/import_checkpoint.py`](../ml/training/import_checkpoint.py))**:
   - `prepare_layoutlmv3_inputs(..., max_length=64)` passed a nonexistent keyword (`max_seq_length=64` required).
   - `decode_layoutlmv3_predictions(..., pred_label_ids=..., token_confidences=...)` passed nonexistent keywords (`logits=logits` required).
   - `_safe_extract_zip()` only checked depth `0` and depth `1` for `config.json`, failing on Colab export bundles (`checkpoints/layoutlmv3-sroie-best/config.json`); updated with recursive `rglob("config.json")` lookup plus isolated staging cleanup when `--target-dir` is given.
2. **SROIE Date+Timestamp Parsing Failure ([`ml/extractors/layoutlmv3_extractor.py`](../ml/extractors/layoutlmv3_extractor.py))**:
   - Real SROIE annotations frequently append timestamps or POS tags (`"25/12/2018 8:13:39 PM"`, `"12-01-19 21:13 SH01 ZK09"`), causing exact `strptime` matching in `parse_extracted_date()` to return `None` during LayoutLMv3 serving and `compare_models.py` canonicalization. Added `%d-%m-%y`/`%m-%d-%y` and embedded date extraction.
3. **Prepared-Token Baseline Flattening ([`ml/evaluation/compare_models.py`](../ml/evaluation/compare_models.py))**:
   - `create_ocr_result_from_receipt()` collapsed all words into a single space-joined line, breaking line-oriented heuristics (`VendorExtractor`, `FinancialExtractor`). Reconstructed reading-order visual lines using bounding-box vertical proximity.
4. **Percentage Token False Positive in Tax Heuristic ([`ml/extraction/financial.py`](../ml/extraction/financial.py))**:
   - On receipts formatted like `TAX (8%)   USD 1.92`, `8` before `%` was extracted as `tax_amount = 8.00`. Filtered unprefixed numbers immediately followed by `%`.
5. **Colab Notebook Hardening ([`ml/training/DocuMind_LayoutLMv3_Training.ipynb`](../ml/training/DocuMind_LayoutLMv3_Training.ipynb))**:
   - Added explicit CUDA GPU + VRAM + free-disk assertions (`>= 8 GB`), `zip` + `python-dateutil` + `scikit-learn` install coverage, an isolated `/tmp` `--smoke-test` cell before full training, `EPOCHS = 5` default (`patience=3`), `python -m ml.training.import_checkpoint` end-to-end validation, and exclusion of optimizer state bloat (`*/training_state.pt`) from the download bundle.
6. **GitHub Actions CI Fixes ([`.github/workflows/ci.yml`](../.github/workflows/ci.yml), [`frontend/playwright.config.ts`](../frontend/playwright.config.ts))**:
   - `gh run view 43474987430` showed `backend` failing at `Verify Alembic clean upgrade and downgrade` because `alembic` lacked `-c backend/alembic.ini`. Fixed all three commands.
   - Switched `frontend-and-e2e` Python setup to CPU PyTorch wheels (`--extra-index-url https://download.pytorch.org/whl/cpu`) and raised Playwright's backend startup timeout to `90_000 ms` to avoid cold-import CI timeouts.

---

## 4. How to Run Real GPU Training on Google Colab & Activate Locally

When executing on a GPU runtime (Google Colab T4/L4/A100 or a clean Linux host):

1. Open [`ml/training/DocuMind_LayoutLMv3_Training.ipynb`](../ml/training/DocuMind_LayoutLMv3_Training.ipynb) in Google Colab and select **Runtime -> Change runtime type -> T4 GPU**.
2. Execute all cells sequentially:
   - Installs CUDA PyTorch/Transformers/Seqeval/Tesseract/zip and asserts `torch.cuda.is_available()`.
   - Prepares deduplicated splits from `arvindrajan92/sroie_document_understanding`.
   - Runs an isolated `/tmp` `--smoke-test`.
   - Fine-tunes `microsoft/layoutlmv3-base` (`5` epochs, `lr=2e-5`, `fp16` AMP, seed `42`, early stopping `patience=3`) and logs to `ml/experiments/latest_experiment.json`.
   - Runs `ml.evaluation.evaluate`, `ml.evaluation.compare_models --run-full-ocr`, and `ml.training.import_checkpoint`.
   - Downloads `documind_layoutlmv3_best.zip`.
3. Import and verify the downloaded bundle into your local Docker/Linux deployment environment (where PyTorch + Tesseract are available):

```bash
python -m ml.training.import_checkpoint \
  --checkpoint documind_layoutlmv3_best.zip \
  --target-dir checkpoints/layoutlmv3-sroie-best \
  --apply-env \
  --env-file .env
```

4. Start/restart the API and Celery worker (`docker compose up -d --build backend worker`), then confirm:
   - `GET /api/v1/models/readiness` reports `"ready": true` and `"is_production_ready": true`.
   - Submitting a receipt with `extraction_method=LAYOUTLMV3` succeeds in the UI and worker logs.
   - Copy `latest_experiment.json`, `evaluation_report.*`, and `benchmark_comparison.*` into the repository and update [`MODEL_CARD.md`](../MODEL_CARD.md) with those observed outputs.

