# DocuMind — Multimodal Document Intelligence Platform

DocuMind is a production-oriented multimodal document intelligence platform designed to ingest scanned receipts and invoices, validate binary file integrity, perform high-resolution OCR, extract structured tabular and entity data, and maintain an immutable processing ledger.

**Current Release**: **Phase 3A (Human-in-the-Loop Document Review & Active Learning Foundation)**

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
│   ├── alembic/              # Database migration definitions (001, 002, 003)
│   ├── app/
│   │   ├── api/              # API routes (/health, /api/v1/documents, /api/v1/reviews)
│   │   ├── core/             # Central error handlers and middleware
│   │   ├── models/           # SQLAlchemy ORM models (Document, ExtractionResult, DocumentReview)
│   │   ├── schemas/          # Pydantic v2 schemas (extraction, review, financial validation)
│   │   ├── services/         # Document ingestion, review workflow, cryptographic validation, storage
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
│   │   │   ├── Dashboard/    # Pipeline & HITL review analytics overview
│   │   │   ├── Documents/    # Document table, uploader, badges, details modal
│   │   │   ├── Reviews/      # Review Queue & Split-Screen Review Workspace
│   │   │   └── Layout/       # Sidebar navigation, Header
│   │   ├── types/            # TypeScript interfaces aligned with Pydantic schemas
│   │   ├── App.tsx           # Main application state and layout
│   │   └── main.tsx          # React DOM entry
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.js
├── ml/                       # Extensible ML extraction subsystem
│   ├── data/                 # Verified SROIE loader, alignment, splitting & validation
│   ├── feedback/             # Active-learning feedback loader, validator & JSONL exporter
│   ├── training/             # Shared LayoutLMv3 preprocessing, GPU training & Colab notebook
│   ├── evaluation/           # Held-out evaluation & scientific model comparison
│   ├── inference/            # BIO tag decoder & confidence aggregation
│   ├── ocr/                  # Real OCR engine, PDF converter, pre-processing
│   ├── extraction/           # Rule-based field extraction & validation pipelines
│   ├── extractors/
│   │   ├── tesseract_baseline.py # Functioning Tesseract OCR baseline extractor
│   │   ├── layoutlmv3_extractor.py # Validated LayoutLMv3 multimodal extractor
│   │   └── placeholder.py    # Phase 1 transparent stub
│   ├── base.py               # BaseExtractor abstract contract & ExtractionOutput
│   └── README.md             # Detailed ML subsystem documentation
├── storage/
│   └── uploads/              # Local storage isolated from web root (safe UUIDs)
├── tests/                    # Pytest suite (60 unit & integration tests)
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

Run the full automated pytest suite (60 tests covering upload validation, OCR logic, SROIE data & LayoutLMv3 evaluation safeguards, HITL review workflows, controlled preview security, and active-learning feedback export):

```bash
python -m pytest -v tests
```

---

## Phase 3A: Human-in-the-Loop Document Review & Active Learning

Phase 3A upgrades DocuMind into a complete document review platform while keeping extraction processing state (`PENDING`, `PROCESSING`, `COMPLETED`, `FAILED`) cleanly separated from human review state (`NOT_REQUIRED`, `PENDING_REVIEW`, `IN_REVIEW`, `APPROVED`, `REJECTED`).

### Key Capabilities
1. **Automatic & Voluntary Review Routing**: Documents are automatically flagged as `PENDING_REVIEW` when heuristic confidence falls below `REVIEW_CONFIDENCE_THRESHOLD` (`0.75`), critical fields (`vendor_name`, `document_date`, `total_amount`) are missing, financial/date integrity warnings occur, or `request_review=true` is passed at upload. High-confidence documents start in `NOT_REQUIRED` and can still be voluntarily reviewed anytime.
2. **Immutable Original Predictions & Field Diff Audit Trail**:
   - `ExtractionResult.structured_data` and `raw_text` are never overwritten by human edits.
   - `DocumentReview`, `FieldCorrection`, and `ReviewAuditEvent` tables (migration `003_add_document_review_tables`) track draft saves, approvals, rejections, reopens, field-level diffs, notes, and optimistic concurrency `version` (`409 Conflict` protection).
3. **Human Review & Controlled Preview Endpoints**:
   - `GET /api/v1/reviews` & `GET /api/v1/reviews/metrics` — Review queue filtering/sorting and real DB-backed completion telemetry.
   - `GET /api/v1/documents/{id}/review`, `PATCH /api/v1/documents/{id}/review`, `POST .../approve`, `POST .../reject`, `POST .../reopen`, `GET .../history`.
   - `GET /api/v1/documents/{id}/preview/meta` & `GET /api/v1/documents/{id}/file` — Controlled inline preview resolving safe UUID filenames inside `storage/uploads/`, setting `X-Content-Type-Options: nosniff`, and bounding PDF page rasterization (`PREVIEW_MAX_PDF_PAGES=20`).
4. **Split-Screen Review Workspace & Dashboard UI**:
   - **Review Queue** tab prioritized by lowest heuristic confidence, missing fields, engine, and status.
   - **Review Workspace**: side-by-side zoomed/paginated receipt preview + copyable OCR text vs. editable Decimal/ISO-validated fields, original prediction captions, unsaved-change protection, and chronological audit timeline.
5. **Active-Learning Feedback JSONL Export (`ml/feedback/`)**:
   - Exports **only `APPROVED`** human-verified records (strictly excluding `REJECTED`, `PENDING_REVIEW`, `IN_REVIEW`, duplicates by SHA-256, and held-out `test.jsonl`/`validation.jsonl` overlaps).
   - Never launches automatic model retraining; flags samples conservatively as `ALIGNED_CONFIDENT` vs `NEEDS_MANUAL_ANNOTATION` (`ml/feedback/feedback_quality.py`).

```bash
# Export verified APPROVED reviews to JSONL feedback dataset:
python -m ml.feedback.export_annotations \
    --output-dir data/feedback_exports \
    --reference-splits data/sroie_processed/test.jsonl data/sroie_processed/validation.jsonl
```

> [!WARNING]
> **Security & Privacy Limitation (Local Development Mode)**: Authentication and user sessions are intentionally omitted in Phase 3A local mode. Review edits and audit events default to `DEFAULT_REVIEWER_ID=local-anonymous`, and controlled preview routes do not require bearer credentials. **Do not deploy this build onto an untrusted network or store sensitive financial/PII documents without adding authentication, RBAC, TLS, and access-controlled object storage.**

---

## Phase 3B Roadmap: Verified Retraining & Edge Inference

1. **Human-Reviewed Bio Token Verification & Fine-Tuning**: Promote `ALIGNED_CONFIDENT` (and manually verified `NEEDS_MANUAL_ANNOTATION`) JSONL records into scheduled LayoutLMv3 fine-tuning experiments tracked against held-out SROIE baselines.
2. **Authentication & Multi-Reviewer RBAC**: Replace `local-anonymous` attribution with authenticated reviewer identities and tenant isolation.
3. **ONNX Runtime / Multi-Page Itemization**: Quantize LayoutLMv3 for fast CPU inference and structured line-item relation extraction.
