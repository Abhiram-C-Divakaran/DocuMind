# DocuMind ML Subsystem (Phase 2B — SROIE & LayoutLMv3 Multimodal Intelligence)

This package contains the complete machine learning subsystem for DocuMind:
1. **Phase 2A**: Production OCR and rule-based receipt extraction pipeline (`TesseractBaselineExtractor`).
2. **Phase 2B**: Multimodal document deep learning pipeline fine-tuning `microsoft/layoutlmv3-base` on the SROIE benchmark with token/entity-level evaluation and comparative benchmarking (`LayoutLMv3Extractor`).

---

## 1. Directory Structure

```
ml/
├── data/
│   ├── annotation_alignment.py  # SROIE ground truth to OCR span alignment & BIO tagging
│   ├── sroie_loader.py          # SROIE dataset loader (local, Hugging Face, or synthetic)
│   ├── prepare_dataset.py       # Reproducible train/val/test splitting & data leakage checks
│   └── dataset_validation.py    # Manifest statistical validation (JSON & Markdown reports)
├── training/
│   ├── config.py                # LayoutLMv3 training hyperparameters dataclass
│   ├── dataset.py               # PyTorch Dataset, token/bbox alignment, subword -100 masking
│   ├── train_layoutlmv3.py      # End-to-end fine-tuning script with smoke-test mode
│   └── colab_guide.py           # Google Colab GPU training instructions & notebook runner
├── evaluation/
│   ├── metrics.py               # Seqeval strict IOB2 entity F1, exact match, & model footprint
│   ├── evaluate.py              # Single checkpoint test evaluation CLI
│   ├── compare_models.py        # Side-by-side benchmark comparing LayoutLMv3 vs. Baseline
│   └── results/                 # Evaluation JSON artifacts and Markdown comparison tables
├── inference/
│   └── decoder.py               # Token logit BIO decoding to canonical entity fields
├── extractors/
│   ├── layoutlmv3_extractor.py  # Production LayoutLMv3Extractor integrating with FastAPI
│   ├── tesseract_baseline.py    # Active Phase 2A OCR + Rules baseline extractor
│   └── placeholder.py           # Phase 1 transparent placeholder stub
├── ocr/
│   ├── engine.py                # Tesseract OCR runner, tokenization, 0-1000 bbox normalization
│   ├── pdf_converter.py         # High-DPI PDF page rasterizer via PyMuPDF
│   └── preprocessor.py          # Orientation deskew, CLAHE contrast boost, adaptive thresholding
└── extraction/
    ├── pipeline.py              # Phase 2A rule-based receipt extraction orchestrator
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

### Subword Masking & Alignment
LayoutLMv3 uses a Byte-Pair Encoding (BPE) subword tokenizer.
- The **first subword** of each OCR word receives the assigned entity label (`B-*` or `I-*`).
- All **subsequent subwords** of that word, as well as special tokens (`<s>`, `</s>`, `<pad>`), are masked with `-100` so that PyTorch `CrossEntropyLoss` ignores them during backpropagation and `seqeval` does not penalize subword divisions.

---

## 3. Dataset Ingestion & Preparation

To download or ingest SROIE, run annotation alignment, verify zero cross-split data leakage, and persist processed manifests to `data/sroie_processed/`:

```bash
# Ingest SROIE with reproducible train (70%) / val (15%) / test (15%) splits:
python -m ml.data.prepare_dataset --output-dir data/sroie_processed --seed 42
```

Outputs generated in `data/sroie_processed/`:
- `train.jsonl`
- `val.jsonl`
- `test.jsonl`
- `validation_report.json`
- `validation_report.md`

---

## 4. LayoutLMv3 Fine-Tuning Pipeline

### Fast Local CPU Smoke Test
To verify the entire training loop, gradient accumulation, validation step, and checkpoint serialization without downloading massive datasets:

```bash
python -m ml.training.train_layoutlmv3 --smoke-test --epochs 1 --batch-size 2
```

### Full GPU Training (Local or Cloud Server)
```bash
python -m ml.training.train_layoutlmv3 \
    --data-dir data/sroie_processed \
    --output-dir checkpoints/layoutlmv3-sroie \
    --best-model-dir checkpoints/layoutlmv3-sroie-best \
    --epochs 3 \
    --batch-size 4 \
    --lr 2e-5 \
    --seed 42
```

---

## 5. Google Colab Free GPU Guide

For training with a free Google Colab T4 GPU:
1. Open a new notebook in [Google Colab](https://colab.research.google.com).
2. Set runtime type to GPU: `Runtime` > `Change runtime type` > `T4 GPU`.
3. Follow the instructions in `ml/training/colab_guide.py`:

```python
# Cell 1: Install dependencies
!pip install -q torch transformers datasets seqeval pillow pytesseract accelerate

# Cell 2: Clone repository
!git clone https://github.com/Abhiram-C-Divakaran/DocuMind.git
%cd DocuMind

# Cell 3: Prepare dataset
!python -m ml.data.prepare_dataset --output-dir data/sroie_processed --use-huggingface --seed 42

# Cell 4: Train model
!python -m ml.training.train_layoutlmv3 \
    --data-dir data/sroie_processed \
    --output-dir checkpoints/layoutlmv3-sroie \
    --best-model-dir checkpoints/layoutlmv3-sroie-best \
    --epochs 3 \
    --batch-size 4 \
    --lr 2e-5

# Cell 5: Run Comparative Benchmark
!python -m ml.evaluation.compare_models \
    --checkpoint-dir checkpoints/layoutlmv3-sroie-best \
    --test-jsonl data/sroie_processed/test.jsonl

# Cell 6: Download trained weights to local machine
from google.colab import files
!zip -r layoutlmv3_sroie_best.zip checkpoints/layoutlmv3-sroie-best/
files.download('layoutlmv3_sroie_best.zip')
```

Extract the downloaded zip into `checkpoints/layoutlmv3-sroie-best/` in your local project workspace.

---

## 6. Evaluation & Comparative Benchmark

To run evaluation on the held-out test split:

```bash
python -m ml.evaluation.evaluate \
    --checkpoint-dir checkpoints/layoutlmv3-sroie-best \
    --test-jsonl data/sroie_processed/test.jsonl \
    --output-dir ml/evaluation/results
```

To run a side-by-side benchmark comparison between the Phase 2A Tesseract baseline and fine-tuned LayoutLMv3:

```bash
python -m ml.evaluation.compare_models \
    --checkpoint-dir checkpoints/layoutlmv3-sroie-best \
    --test-jsonl data/sroie_processed/test.jsonl \
    --output-dir ml/evaluation/results
```

Outputs:
- `ml/evaluation/results/benchmark_comparison.json`
- `ml/evaluation/results/benchmark_comparison.md`

### Benchmark Insights & Error Analysis
1. **Address & Postal Code Disambiguation**: In rule-based extraction, 5-digit postal codes (e.g., `55100 KUALA LUMPUR`) frequently trigger false positives for `total_amount`. Multimodal LayoutLMv3 uses 2D spatial coordinates and header/footer proximity to distinguish addresses from totals.
2. **Multi-line Addresses**: LayoutLMv3 connects sequential tokens using `B-ADDRESS` and `I-ADDRESS` spanning multiple receipt lines, eliminating fragmented address extractions.
3. **Inference Latency Trade-off**: The pure CPU rule-based pipeline extracts fields in **~2ms**; LayoutLMv3 requires transformer forward passes taking **~80-120ms** on GPU (~400ms on CPU).
