# DocuMind — Multimodal Document Intelligence Platform

DocuMind is a production-oriented multimodal document intelligence platform designed to ingest scanned receipts and invoices, validate binary file integrity, perform high-resolution OCR, extract structured tabular and entity data, and maintain an immutable processing ledger.

**Current Release**: **Phase 2A (Functioning Tesseract OCR & Rule-Based Extraction)**

---

## What's New in Phase 2A

- **Functioning Tesseract OCR Engine**: Real text recognition and word tokenization using `pytesseract` and OpenCV (headless).
- **PDF-to-Image Rasterization**: High-resolution multi-page PDF rendering via `PyMuPDF` with configurable DPI (default `300 DPI`) and page resource bounds.
- **Computer Vision Pre-processing**: Grayscale conversion, CLAHE contrast enhancement, Gaussian denoising, and automated orientation deskewing.
- **Multimodal Coordinates**: Normalizes all word bounding boxes to a `0–1000` integer coordinate grid ready for LayoutLMv3 integration.
- **Deterministic Field Extraction**:
  - **Vendor / Merchant**: Header layout analysis, font-size weighting, exclusion filter for tax/invoice headers.
  - **Document Date**: Parsing for ISO, DD/MM/YYYY, MM/DD/YYYY, and textual dates with configurable ambiguity handling (`DEFAULT_DATE_FORMAT_ORDER=DMY`).
  - **Financial Totals**: Keyword proximity parsing (`TOTAL`, `GRAND TOTAL`, `AMOUNT PAID`, `BALANCE DUE`), decimal-safe conversion, subtotal/cash separation.
  - **Taxes**: Extraction and validation ensuring $\mathrm{Tax} \le \mathrm{Total}$.
  - **Currency Normalization**: Automatic mapping of symbols (`$`, `₹`, `€`, `£`, `¥`) and ISO codes (`USD`, `INR`, `EUR`, `GBP`).
  - **Vendor Address**: Contextual line extraction below vendor names.
  - **Line Items**: Experimental tabular itemization.
- **Heuristic Confidence & Integrity Checks**: Weighted field-level confidences, overall confidence scores, and automatic flags for manual review.
- **Frontend Upgrades**: Document details modal displaying structured entities, confidence badges, validation warnings, and a collapsible raw OCR text viewer.

---

## Repository Structure

```
Doc/
├── backend/                  # FastAPI backend service
│   ├── alembic/              # Database migration definitions (001, 002)
│   ├── app/
│   │   ├── api/              # API routes (/health, /api/v1/documents)
│   │   ├── core/             # Central error handlers and middleware
│   │   ├── models/           # SQLAlchemy ORM models (Document, ExtractionResult)
│   │   ├── schemas/          # Pydantic v2 schemas (decimal-safe currency & line items)
│   │   ├── services/         # Storage and cryptographic signature validator
│   │   ├── config.py         # Pydantic BaseSettings environment loader
│   │   ├── database.py       # Async SQLAlchemy session and connection engine
│   │   ├── logger.py         # Structured logging formatter
│   │   └── main.py           # Application entrypoint with CORS & lifecycle
│   ├── requirements.txt      # Python dependencies (OpenCV, PyMuPDF, pytesseract)
│   └── pyproject.toml        # Backend build & pytest configuration
├── frontend/                 # React 18 + TypeScript + Vite + Tailwind CSS
│   ├── src/
│   │   ├── api/client.ts     # Typed API client for FastAPI
│   │   ├── components/       # Modern responsive components
│   │   │   ├── Common/       # Skeleton loaders, Empty states, Toast alerts
│   │   │   ├── Dashboard/    # Metrics overview & ML roadmap view
│   │   │   ├── Documents/    # Document table, uploader, details modal
│   │   │   └── Layout/       # Sidebar navigation, Header
│   │   ├── types/            # TypeScript interfaces aligned with Pydantic schemas
│   │   ├── App.tsx           # Main application state and layout
│   │   └── main.tsx          # React DOM entry
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.js
├── ml/                       # Extensible ML extraction subsystem
│   ├── ocr/                  # Real OCR engine, PDF converter, pre-processing
│   ├── extraction/           # Rule-based field extraction & validation pipelines
│   ├── extractors/
│   │   ├── tesseract_baseline.py # Functioning Tesseract OCR baseline extractor
│   │   ├── layoutlmv3_extractor.py # Multimodal transformer interface for Phase 2B
│   │   └── placeholder.py    # Phase 1 transparent stub
│   ├── base.py               # BaseExtractor abstract contract & ExtractionOutput
│   └── README.md             # Detailed ML subsystem documentation
├── storage/
│   └── uploads/              # Local storage isolated from web root (safe UUIDs)
├── tests/                    # Pytest suite (32 unit & integration tests)
├── docker-compose.yml        # PostgreSQL 16 container definition
├── .env.example              # Documented configuration template
└── README.md
```

---

## Installation & System Prerequisites

### 1. Tesseract OCR Executable (System Dependency)

Tesseract is a real system dependency:

- **Windows**:
  1. Download the installer from the [UB-Mannheim Tesseract Wiki](https://github.com/UB-Mannheim/tesseract/wiki).
  2. Install to `C:\Program Files\Tesseract-OCR`.
  3. Add `C:\Program Files\Tesseract-OCR` to your System `PATH`, or set `TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe` in your `.env`.
- **Linux (Ubuntu / Debian)**:
  ```bash
  sudo apt-get update && sudo apt-get install -y tesseract-ocr libtesseract-dev
  ```
- **macOS**:
  ```bash
  brew install tesseract
  ```

### 2. Python Backend Setup

```bash
# 1. Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Run database migrations
python -m alembic -c backend/alembic.ini upgrade head

# 4. Start FastAPI server
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

The backend is accessible at:
- **API Root**: `http://localhost:8000`
- **Swagger Documentation**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`

### 3. Frontend Dashboard Setup

```bash
cd frontend

# Install dependencies
npm install

# Start Vite development server
npm run dev
```

The frontend dashboard is accessible at `http://localhost:5173`.

---

## Sample Document Upload via CLI

Upload a receipt or invoice directly to the FastAPI API using curl or PowerShell:

```bash
# Using curl (default method is TESSERACT_BASELINE)
curl -X POST "http://localhost:8000/api/v1/documents" \
  -F "file=@receipt_sample.png" \
  -F "extraction_method=TESSERACT_BASELINE"
```

In PowerShell:
```powershell
$form = @{
    file = Get-Item -Path "receipt_sample.png"
    extraction_method = "TESSERACT_BASELINE"
}
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/documents" -Method Post -Form $form
```

---

## Running the Automated Test Suite

Run the 32-test automated suite exercising OCR logic, date parsing, financial normalization, PDF conversion, persistence, and error handling:

```bash
python -m pytest -v tests
```

---

## Known OCR Limitations

1. **Physical Receipt Degradation**: Severe physical crumpling, heavy shadows, or faded thermal paper may degrade OCR character recognition.
2. **Dense Multi-Column Receipts**: Complex column structures without clear delimiters can cause horizontal text merging, best handled by multimodal models (Phase 2B).
3. **Handwritten Annotations**: Tesseract's standard English model is trained for printed typography; handwriting recognition is limited.

---

## Phase 2B Roadmap: SROIE Benchmark & LayoutLMv3

1. **SROIE Dataset Curation**: Parse and split 1,000 scanned receipts with ground-truth bounding box coordinates (`data/sroie/`).
2. **LayoutLMv3 Fine-Tuning**: Fine-tune `microsoft/layoutlmv3-base` on token classification with BIO labels (`B-VENDOR`, `I-VENDOR`, `B-DATE`, `I-DATE`, `B-TOTAL`, `I-TOTAL`, `B-ADDRESS`, `I-ADDRESS`).
3. **Side-by-Side Benchmarking**: Evaluate Tesseract baseline vs. fine-tuned LayoutLMv3 measuring Entity F1 Score, Character Error Rate (CER), inference latency (ms), and VRAM footprint.
