# DocuMind ML Subsystem (Phase 2A — OCR & Rule-Based Extraction)

This package contains the functioning OCR and rule-based receipt extraction pipeline for DocuMind.

---

## Architecture Overview

```
ml/
├── ocr/
│   ├── engine.py              # Tesseract OCR runner, tokenization, 0-1000 bbox normalization
│   ├── pdf_converter.py       # High-DPI PDF page rasterizer via PyMuPDF
│   └── preprocessor.py        # Orientation deskew, CLAHE contrast boost, adaptive thresholding
├── extraction/
│   ├── schemas.py             # Internal extraction models, token evidence, and field candidates
│   ├── vendor.py              # Vendor name extraction via header geometry & header exclusion filters
│   ├── date.py                # Regex parser supporting DD/MM/YYYY, MM/DD/YYYY, YMD & textual dates
│   ├── financial.py           # Total, subtotal, tax and ISO currency normalization
│   ├── address.py             # Spatial address extractor below vendor headers
│   ├── line_items.py          # Experimental tabular line item pattern scanner
│   ├── validator.py           # Decimal-safe business rules, sanity checks & confidence estimation
│   └── pipeline.py            # Extraction orchestrator producing ReceiptExtractionResult
├── extractors/
│   ├── tesseract_baseline.py  # Active Phase 2A Tesseract Baseline Extractor
│   ├── layoutlmv3_extractor.py# Phase 2B Multimodal Transformer (scheduled for fine-tuning)
│   └── placeholder.py         # Phase 1 transparent placeholder stub
├── base.py                    # BaseExtractor ABC & ExtractionOutput container
└── schemas.py                 # Pydantic schemas for extracted entities
```

---

## System Requirements & Installation

### 1. Tesseract OCR Executable

Tesseract is a native C++ system executable required for optical character recognition.

#### Windows Installation:
1. Download the official installer from the [UB-Mannheim Tesseract Wiki](https://github.com/UB-Mannheim/tesseract/wiki) (e.g. `tesseract-ocr-w64-setup-5.4.0.exe`).
2. Run the installer and install to the default location: `C:\Program Files\Tesseract-OCR`.
3. Add `C:\Program Files\Tesseract-OCR` to your Windows System `PATH`, or set `TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe` in your `.env` file.

#### Linux (Ubuntu / Debian):
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr libtesseract-dev
```

#### macOS:
```bash
brew install tesseract
```

---

## Pipeline Features

1. **Multi-Format Support**: Ingests PDF, PNG, and JPEG documents.
2. **High-Resolution Rasterization**: Renders PDF pages to RGB matrices at configurable DPI (default `300 DPI`) with configurable page limit (`OCR_MAX_PAGES`, default 5).
3. **Computer Vision Pre-processing**:
   - Automatic orientation deskewing using contour minimum area bounding boxes.
   - Contrast Limited Adaptive Histogram Equalization (CLAHE) for faded thermal paper.
   - Gaussian denoising and adaptive Gaussian thresholding.
4. **Bounding Box Normalization**:
   - Word bounding boxes are normalized to a `[0, 1000]` integer coordinate grid:
     $$x_0 = \mathrm{int}(1000 \times \mathrm{left} / W)$$
     $$y_0 = \mathrm{int}(1000 \times \mathrm{top} / H)$$
     $$x_1 = \mathrm{int}(1000 \times (\mathrm{left} + w) / W)$$
     $$y_1 = \mathrm{int}(1000 \times (\mathrm{top} + h) / H)$$
   - Ready for direct feature input into LayoutLMv3 in Phase 2B.
5. **Deterministic Rule-Based Extraction**:
   - **Vendor Name**: Analyzes top 35% of the receipt, filtering out invoice/tax headers, websites, and phone numbers.
   - **Dates**: Robust regex capturing ISO dates, DD/MM/YYYY, MM/DD/YYYY, and textual dates ("15 Oct 2024"). Ambiguous dates (e.g., `05/06/2024`) use `DEFAULT_DATE_FORMAT_ORDER` (default `DMY`) and record ambiguity warnings.
   - **Financials**: Positional keyword matching (`GRAND TOTAL`, `TOTAL DUE`, `AMOUNT PAID`, `TAX`, `VAT`). Discards subtotal/cash/change lines. Decimal-safe conversion.
   - **Currency**: Automatically maps symbols (`$`, `₹`, `€`, `£`) and codes (`USD`, `INR`, `EUR`, `GBP`) to ISO 4217 standard.
   - **Address**: Captures street, road, avenue, postal code, and city lines below vendor names.
6. **Heuristic Confidence Estimation & Manual Review**:
   - Computes weighted field-level confidences based on OCR token confidence, keyword proximity, and candidate competition.
   - Computes overall document confidence score.
   - Enforces business rules: Total $> 0$, Tax $\le$ Total, reasonable date bounds ($1990 \le \mathrm{year} \le \mathrm{current} + 1$).
   - Flags `needs_manual_review` if confidence is below threshold (`0.60`) or validation anomalies exist.

---

## Known OCR Limitations

- **Thermal Receipt Degradation**: Severe physical fading, crumpling, or tearing can cause character misrecognition (e.g., `8` misread as `B`, `0` as `O`).
- **Complex Multi-Column Tables**: Rule-based extraction uses line-oriented heuristics; intricate multi-column layouts without clear dividers are better served by the upcoming Phase 2B LayoutLMv3 model.
- **Handwritten Receipts**: Tesseract's default model is optimized for printed text; handwriting requires custom model fine-tuning.
- **Non-Latin Scripts**: Currently configured for English/Latin scripts (`eng`). Additional languages require installing the corresponding `.traineddata` files.

---

## Phase 2B Roadmap: SROIE & LayoutLMv3

1. **SROIE Dataset Ingestion**: Load 1,000 scanned receipts with ground-truth bounding box coordinates and entity labels (`COMPANY`, `DATE`, `ADDRESS`, `TOTAL`).
2. **Tokenizer & Feature Extraction**: Implement `LayoutLMv3Processor` generating image patches, token embeddings, and 2D normalized spatial coordinates.
3. **Fine-Tuning**: Train `microsoft/layoutlmv3-base` on token classification with BIO tags.
4. **Comparative Benchmark**: Produce side-by-side evaluation metrics comparing Tesseract Baseline vs. Fine-tuned LayoutLMv3 (Entity F1, Latency, VRAM footprint).
