# DocuMind — Multimodal Document Intelligence Platform

DocuMind is a production-oriented multimodal document intelligence platform designed to ingest scanned receipts and invoices, validate binary file integrity, perform high-resolution OCR, extract structured tabular and entity data, and maintain an immutable processing ledger.

**Current Release**: **Phase 4A (Asynchronous Celery + Redis Pipeline, Argon2/JWT RBAC Security, Hardened Docker Compose & Observability)** — *Note: see [`MODEL_CARD.md`](MODEL_CARD.md) for LayoutLMv3 checkpoint readiness and Google Colab GPU training instructions.*

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
├── backend/                  # FastAPI backend service + Celery worker
│   ├── alembic/              # Database migration definitions (001–004)
│   ├── app/
│   │   ├── api/              # API routes (/health, /api/v1/auth, /documents, /jobs, /models, /reviews)
│   │   ├── core/             # Auth security (Argon2/JWT/rate limiter), error handlers, exceptions
│   │   ├── models/           # SQLAlchemy ORM models (User, RefreshToken, Document, ProcessingJob, OutboxEvent, DocumentReview)
│   │   ├── schemas/          # Pydantic v2 schemas (auth, documents, jobs, extraction, models readiness, review)
│   │   ├── services/         # Auth service, async queueing, media security, review workflow, storage
│   │   ├── worker/           # Celery app, background extraction task, outbox & stale job recovery beat
│   │   ├── cli.py            # Admin provisioning & maintenance CLI (create-admin)
│   │   ├── config.py         # Pydantic BaseSettings environment loader & runtime security guard
│   │   ├── database.py       # Async & worker sync SQLAlchemy session factories
│   │   ├── logger.py         # Structured JSON logging with sensitive-field redaction
│   │   └── main.py           # FastAPI entrypoint, security headers, CORS & lifecycle
│   ├── Dockerfile            # Non-root multi-stage FastAPI & Celery worker image
│   ├── docker-entrypoint.sh  # Safe DB wait & optional migration entrypoint
│   ├── requirements.txt      # Python dependencies (FastAPI, Celery, Redis, Argon2, PyJWT, OpenCV, pytesseract)
│   └── pyproject.toml        # Backend build & pytest configuration
├── frontend/                 # React 18 + TypeScript + Vite + Tailwind CSS + Nginx SPA
│   ├── src/
│   │   ├── api/              # Typed API client (single-flight token refresh) & auth token storage
│   │   ├── components/       # Responsive UI components
│   │   │   ├── Auth/         # Login panel & Admin user management view
│   │   │   ├── Common/       # Skeleton loaders, Empty states, Toast alerts
│   │   │   ├── Dashboard/    # Pipeline & HITL review analytics overview
│   │   │   ├── Documents/    # Document table, async uploader, status polling, details modal
│   │   │   ├── Reviews/      # Review Queue & Split-Screen Review Workspace
│   │   │   └── Layout/       # Sidebar navigation, Header, Role badge
│   │   ├── types/            # TypeScript interfaces aligned with Pydantic schemas
│   │   ├── App.tsx           # Authenticated application state & navigation
│   │   └── main.tsx          # React DOM entry
│   ├── Dockerfile            # Multi-stage Vite build + non-root nginx:1.27-alpine runtime
│   ├── nginx.conf            # SPA routing, API reverse proxy, upload caps & security headers
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
│   │   ├── layoutlmv3_extractor.py # Validated LayoutLMv3 multimodal extractor (GPU semaphore + CPU fallback)
│   │   └── placeholder.py    # Phase 1 transparent stub
│   ├── base.py               # BaseExtractor abstract contract & ExtractionOutput
│   └── README.md             # Detailed ML subsystem documentation
├── storage/
│   └── uploads/              # Local storage isolated from web root (safe UUIDs)
├── tests/                    # Pytest suite (Phases 1–4A: 87 unit & integration tests)
├── MODEL_CARD.md             # LayoutLMv3 architecture, dataset hygiene & honest status
├── docker-compose.yml        # Full stack (postgres, redis, api, worker, beat, frontend + optional flower)
├── docker-compose.prod.yml   # Production overrides (restart policies, internal DB/Redis ports, TLS note)
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

# 4. Provision an initial ADMIN user (or bootstrap via CLI anytime)
python -m backend.app.cli create-admin --email admin@documind.local --full-name "DocuMind Admin" --password "ChangeMeStrong123!"

# 5. Start FastAPI server
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

# 6. Start Celery worker & beat scheduler (requires Redis)
celery -A backend.app.worker.celery_app worker --loglevel=info --concurrency=2
celery -A backend.app.worker.celery_app beat --loglevel=info
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

First authenticate to obtain a JWT bearer token, then submit a document (returns `202 Accepted` with queued job ID):

```bash
# 1. Obtain access token
TOKEN=$(curl -s -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@documind.local","password":"ChangeMeStrong123!"}' | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# 2. Enqueue document for async processing (returns 202 Accepted)
curl -X POST "http://localhost:8000/api/v1/documents" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@receipt_sample.png" \
  -F "extraction_method=TESSERACT_BASELINE"
```

In PowerShell:
```powershell
$login = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/auth/login" `
  -Method Post -ContentType "application/json" `
  -Body '{"email":"admin@documind.local","password":"ChangeMeStrong123!"}'
$headers = @{ Authorization = "Bearer $($login.access_token)" }
$form = @{
    file = Get-Item -Path "receipt_sample.png"
    extraction_method = "TESSERACT_BASELINE"
}
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/documents" -Method Post -Headers $headers -Form $form
```

---

## Running the Automated Test Suite

Run the full automated pytest suite covering upload validation, OCR logic, SROIE data & LayoutLMv3 evaluation safeguards, HITL review workflows, controlled preview security, active-learning feedback export, experiment tracking, and `/api/v1/models/readiness`:

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

> [!NOTE]
> **Authentication & RBAC Enabled (Phase 4A)**: All document, review, job, model-readiness, and preview endpoints enforce JWT bearer authentication and role checks (`ADMIN`, `REVIEWER`, `VIEWER`) with account ownership scoping. See [Phase 4A Architecture & Production Deployment](#phase-4a-production-backend-async-workers-security--docker) below before deploying externally.

---

## Phase 3B Status: Training Infrastructure, Model Readiness & Deployment Gating

1. **Experiment Tracking (`ml/training/experiment_tracker.py`)**: Records `experiment_id`, dataset split hashes, hyperparameters, hardware/dependency metadata, per-epoch loss/entity F1, and best checkpoint paths (with optional MLflow mirroring).
2. **Evaluation & Benchmarking Upgrades**: `evaluate.py` and `compare_models.py` now compute 4-entity macro F1, `p50`/`p95`/`p99` inference latency percentiles, checkpoint disk footprint (`MB`), and bidirectional error analysis while refusing to fabricate metrics when weights or runtime are unavailable (`NOT_EVALUATED`).
3. **Model Caching & Readiness API (`GET /api/v1/models/readiness`)**: Caches loaded LayoutLMv3 weights by `(resolved_checkpoint_dir, device)` in FastAPI and exposes sanitized readiness status to gate `DocumentUploader.tsx` and `MlArchitectureView.tsx` (defaulting safely to `TESSERACT_BASELINE`).
4. **Next Execution Step (Cloud GPU)**: Run [`ml/training/DocuMind_LayoutLMv3_Training.ipynb`](ml/training/DocuMind_LayoutLMv3_Training.ipynb) on Google Colab T4 GPU to generate real fine-tuned weights (`checkpoints/layoutlmv3-sroie-best`) and held-out SROIE benchmark results (local Windows PyTorch/Pandas/PyArrow execution is blocked by Windows App Control `WinError 4551`).

---

## Phase 4A: Production Backend, Async Workers, Security & Docker

Phase 4A converts DocuMind into a production-ready asynchronous document processing system.

### 1. Asynchronous Processing Architecture (Celery + Redis + Outbox)
- **Non-blocking Ingestion (`POST /api/v1/documents`)**: Streams upload chunks cleanly within `MAX_FILE_SIZE_MB`, validates magic bytes, image pixels (`MAX_IMAGE_PIXELS=40000000`), and PDF page counts (`MAX_PDF_PAGES_HARD_LIMIT=50`), persists `Document`, `ProcessingJob` (`QUEUED`), and `OutboxEvent` inside a single DB transaction, and returns **`202 Accepted`**.
- **Bounded Retries & Idempotency**: Celery workers atomically claim `QUEUED`/`RETRYING` jobs, prevent duplicate `ExtractionResult` rows per `job_id`, classify permanent failures (missing checkpoints, missing OCR binaries, invalid input) cleanly without silent fallback, and retry transient `OSError`/`TimeoutError`/`ConnectionError` failures up to `MAX_JOB_ATTEMPTS`.
- **Outbox & Stale Job Recovery (`celery_beat`)**: Periodically republishes unpublished `OutboxEvent` rows and recovers jobs stuck in `PROCESSING` past `STALE_JOB_TIMEOUT_SECONDS`.
- **Job Telemetry & Operator Controls**: `GET /api/v1/documents/{id}/status`, `POST /api/v1/documents/{id}/retry`, `POST /api/v1/documents/{id}/cancel`, `DELETE /api/v1/documents/{id}`, `GET /api/v1/jobs`, `GET /api/v1/jobs/metrics`, and `ADMIN`-only `POST /api/v1/jobs/recover`.
- **Worker ML Lifecycle**: Bounded GPU inference semaphore (`MAX_CONCURRENT_GPU_INFERENCE=1`), optional preloading (`PRELOAD_LAYOUTLMV3_ON_WORKER_START`), and explicit CUDA OOM -> CPU fallback metadata (`device_fallback`).

### 2. Authentication, RBAC & Multi-User Isolation
- **Argon2 + JWT Access/Refresh Rotation**: Short-lived JWT access tokens (`15m`), rotated single-use SHA-256-hashed refresh tokens (`7d`) with replay-attack session revocation, and Redis-backed / in-memory login rate limiting (`429 Too Many Requests`).
- **Role-Based Access Control (`ADMIN`, `REVIEWER`, `VIEWER`)**:
  - `ADMIN`: Full access across users, documents, reviews, job recovery, and feedback exports (`GET /api/v1/users`, `POST /api/v1/users`, `PATCH /api/v1/users/{id}`, `GET /api/v1/reviews/feedback-export`).
  - `REVIEWER`: Can upload, review, approve/reject/reopen, retry/cancel/delete **own** documents.
  - `VIEWER`: Read-only access to **own** documents/jobs/previews.
  - `ALLOW_VIEWER_SELF_REGISTRATION` defaults to `false`; unrestricted self-registration with elevated roles is never permitted.

### 3. Containerization & Production Operations

```bash
# 1. Configure .env from .env.example (set strong POSTGRES_PASSWORD, REDIS_PASSWORD, JWT_SECRET_KEY)
cp .env.example .env

# 2. Build and launch core stack (postgres, redis, api, worker, beat, frontend)
docker compose up --build

# 3. Provision an initial admin inside the API container
docker compose exec api python -m backend.app.cli create-admin \
  --email admin@documind.local --full-name "DocuMind Admin" --password "ChangeMeStrong123!"

# Optional: enable Flower monitoring UI (profile: monitoring)
docker compose --profile monitoring up --build
```

For production hosts, combine with [`docker-compose.prod.yml`](docker-compose.prod.yml) and terminate TLS in an upstream reverse proxy (Caddy/Traefik/ALB) forwarding to `frontend`:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

### 4. Production Security Checklist
1. Set `ENVIRONMENT=production` and `DEBUG=false` (FastAPI startup refuses to boot in `production` if `JWT_SECRET_KEY` is missing, `< 32` characters, or a placeholder value, or if `DEBUG=true`).
2. Restrict `CORS_ORIGINS` to exact trusted HTTPS frontend origins (never `*`).
3. Keep `POSTGRES_PASSWORD`, `REDIS_PASSWORD`, `FLOWER_BASIC_AUTH`, and `JWT_SECRET_KEY` outside source control.
4. Put TLS/HTTPS termination in front of Nginx (`frontend` service) and monitor `/health` (`postgres`, `redis`, `celery_worker`, `tesseract`, `layoutlmv3` status).
