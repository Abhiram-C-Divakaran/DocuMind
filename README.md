# DocuMind - Multimodal Document Intelligence Platform (Phase 1)

DocuMind is a production-oriented multimodal document intelligence platform designed to ingest scanned receipts and invoices, validate integrity via binary signatures, extract structured tabular and entity data, and maintain an immutable processing ledger.

Phase 1 establishes the enterprise-grade foundation: secure multipart ingestion, SQLAlchemy ORM with Alembic migrations, Pydantic v2 validation contracts, a modern React + Vite + Tailwind CSS frontend dashboard, typed API client, and extensible ML baseline interfaces ready for Phase 2 fine-tuning on the SROIE benchmark.

---

## Architecture Overview

```
Doc/
├── backend/                  # FastAPI backend service
│   ├── alembic/              # Database migration definitions
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
│   ├── requirements.txt      # Python dependencies
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
│   ├── extractors/
│   │   ├── placeholder.py    # Phase 1 transparent stub (no fake inferences)
│   │   ├── tesseract_baseline.py # Tesseract OCR + regex rules interface
│   │   └── layoutlmv3_extractor.py # Multimodal transformer interface for SROIE
│   ├── base.py               # BaseExtractor abstract contract & ExtractionOutput
│   └── schemas.py            # ML data schemas
├── storage/
│   └── uploads/              # Local storage isolated from web root (safe UUIDs)
├── tests/                    # Pytest suite
│   ├── conftest.py           # In-memory async SQLite engine & test client
│   ├── test_health.py        # Health and component verification
│   ├── test_upload_validation.py # Magic bytes, MIME types & size limit tests
│   ├── test_document_persistence.py # Document and extraction lifecycle
│   └── test_ml_interfaces.py # Extractor contracts & provenance
├── docker-compose.yml        # PostgreSQL 16 container definition
├── .env.example              # Documented configuration template
└── README.md
```

---

## Security & Validation Architecture

1. **Dual Signature & MIME Verification**:
   - Files are validated using both the client-declared `Content-Type` header and actual byte headers (**magic bytes**):
     - PDF: `%PDF-` (`0x25, 0x50, 0x44, 0x46, 0x2D`)
     - PNG: `\x89PNG\r\n\x1a\n` (`0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A`)
     - JPEG: `\xff\xd8\xff` SOI marker
   - Rejects corrupted files, mismatched headers (e.g. executable or text files disguised as PDF), and unsupported extensions.
2. **File Size Enforcement**: Default limit of 15 MB (configurable via `MAX_FILE_SIZE_MB`), rejecting oversized payloads with HTTP 413.
3. **Directory Traversal Protection**: Files are stored using cryptographic UUIDv4 filenames (`<uuid4>.<ext>`) in a dedicated directory outside publicly accessible web roots.
4. **Decimal-Safe Financial Arithmetic**: All financial values (`total_amount`, `tax_amount`, `unit_price`) utilize Python's `Decimal` type to prevent IEEE 754 floating-point rounding errors.

---

## Quickstart & Installation

### 1. Prerequisites
- **Python 3.11+**
- **Node.js 18+** and **npm**
- **Docker & Docker Compose** (Optional for PostgreSQL; SQLite runs out of the box)

### 2. Clone and Configure Environment

```bash
# Copy environment configuration
cp .env.example .env
```

### 3. Backend Setup

```bash
# Create and activate Python virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Run database migrations (or rely on auto-table creation on startup)
alembic -c backend/alembic.ini upgrade head

# Start FastAPI development server
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

FastAPI will be available at:
- **API Root**: `http://localhost:8000`
- **Swagger Interactive Docs**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`

### 4. Database via Docker Compose (PostgreSQL)

If using PostgreSQL rather than the default SQLite:

```bash
# Start PostgreSQL container
docker-compose up -d

# Update .env:
# DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/documind
```

### 5. Frontend Setup

```bash
cd frontend

# Install npm dependencies
npm install

# Start Vite development server
npm run dev
```

The frontend application will be running at `http://localhost:5173`.

---

## API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health, database connectivity & storage verification |
| `POST` | `/api/v1/documents` | Multipart document upload (PDF, PNG, JPEG) with signature check |
| `GET` | `/api/v1/documents` | Paginated document ledger with filtering (`status`, `search`) |
| `GET` | `/api/v1/documents/{id}` | Detailed document view with structured extractions and error log |
| `GET` | `/api/v1/documents/metrics/summary` | Telemetry counts (Total, Completed, Pending, Failed) |

---

## Running the Automated Test Suite

The test suite exercises the health probe, magic byte validation, MIME rejection, file size limits, document persistence, and ML interface contracts:

```bash
# Run tests with pytest
.venv\Scripts\pytest -v tests
```

---

## Phase 2 Machine Learning Plan

In Phase 2, DocuMind will advance from structural placeholder contracts to production models:

1. **Benchmark Dataset**:
   - Ingest the **SROIE** (Scanned Receipts OCR and Information Extraction) benchmark (1,000 scanned receipts).
   - Establish evaluation splits (Train: 600, Validation: 100, Test: 300).
2. **Classical Baseline**:
   - Calibrate the `TesseractBaselineExtractor` using PyTesseract OCR with bounding box extraction (`hOCR`).
   - Implement regex-based heuristic field parsing for vendor, date, address, and totals.
3. **Multimodal Transformer**:
   - Fine-tune `microsoft/layoutlmv3-base` on token classification utilizing visual tokens, 2D normalized bounding boxes, and OCR text tokens.
   - Target entity tags: `COMPANY`, `DATE`, `ADDRESS`, `TOTAL`.
4. **Evaluation & Comparison**:
   - Run side-by-side benchmarking measuring **Entity F1 Score**, **Character Error Rate (CER)**, **Inference Latency (ms)**, and **GPU Memory Consumption**.
   - Deploy as an asynchronous background worker using Celery or Redis Queue.

