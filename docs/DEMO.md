# DocuMind Reproducible Portfolio Demo

This walkthrough demonstrates the complete end-to-end DocuMind workflow using a clean synthetic receipt fixture, asynchronous extraction workers, human-in-the-loop review, immutable audit provenance, active-learning feedback export, and RBAC isolation.

---

## 1. Prerequisites & Stack Startup

Start PostgreSQL, Redis, FastAPI, Celery worker, Celery Beat, and the React/Nginx frontend via Docker Compose:

```bash
cp .env.example .env
# Edit .env to set POSTGRES_PASSWORD, REDIS_PASSWORD, and JWT_SECRET_KEY
docker compose up -d --build
docker compose exec backend alembic upgrade head
docker compose exec backend python -m backend.app.cli create-admin \
  --email admin@documind.local --password "AdminStrongPassword123!"
```

Generate a crisp synthetic receipt image for reproducible local testing:

```bash
python scripts/generate_sample_receipt.py
# Generates docs/demo_sample_receipt.png ("SUNRISE ARTISAN BAKERY")
```

Open the application at `http://localhost:3000`.

---

## 2. Step-by-Step Portfolio Walkthrough

### Step 1 — Authenticated Access & Cookie Session Security
- Sign in with your seeded `ADMIN` credentials (`POST /api/v1/auth/login` with `X-Auth-Transport: cookie`).
- Access tokens remain in JS memory only; refresh tokens reside in an `HttpOnly`, `SameSite=lax` cookie scoped to `/api/v1/auth`, paired with a double-submit `X-CSRF-Token`.

![Login Screen](file:///c:/Users/Abhiram/Documents/Projects/Doc/docs/screenshots/01_login.png)

### Step 2 — Provision RBAC Accounts
- Navigate to **Admin & Jobs Ops**.
- Create a `REVIEWER` account (can upload, review, draft-save, approve, and reject) and a `VIEWER` account (read-only access, prevented from UI/API mutations).

### Step 3 — Queue Receipt Asynchronously & Track Worker Progress
- Sign in as the `REVIEWER`, navigate to **Upload Documents**, select [demo_sample_receipt.png](file:///c:/Users/Abhiram/Documents/Projects/Doc/docs/demo_sample_receipt.png), and click **Submit for Ingestion**.
- FastAPI validates the file signature (`image/png`), persists a `PENDING` document + `OutboxEvent`, dispatches a Celery job, and returns HTTP `202 Accepted`.
- The **Document Ledger** polls asynchronously until the Celery worker finishes Tesseract OCR extraction and sets the document status to `COMPLETED`.

![Document Ledger](file:///c:/Users/Abhiram/Documents/Projects/Doc/docs/screenshots/02_document_ledger.png)

### Step 4 — Inspect Structured Extraction & Validation Findings
- Click the completed document row to open the extraction modal displaying per-field confidence badges, extracted merchant metadata, line items, and validation notes.

![Extraction Detail Modal](file:///c:/Users/Abhiram/Documents/Projects/Doc/docs/screenshots/03_extraction_modal.png)

### Step 5 — Human-in-the-Loop Review, Draft Persistence & Audit Approval
- Click **Open Review Workspace** to view the controlled receipt preview side-by-side with the correction form.
- Edit `Vendor Name` (e.g., `"SUNRISE ARTISAN BAKERY LLC"`), add reviewer notes, click **Save Draft**, and then click **Approve & Verify**.
- Original OCR predictions remain immutable alongside field-level diff entries and chronological audit events (`QUEUE_FLAGGED` -> `REVIEW_STARTED` -> `APPROVED`).

![Review Workspace](file:///c:/Users/Abhiram/Documents/Projects/Doc/docs/screenshots/04_review_workspace.png)

### Step 6 — Platform Overview, Queue Telemetry & Feedback Export
- Sign back in as `ADMIN` to inspect aggregate metrics in **Dashboard** and live Dependency Health, Job Queue Telemetry, Stale Recovery triggers, and **Export Approved Feedback** in **Admin & Jobs Ops**.
- Approved corrections export to active-learning JSONL (`feedback_export.jsonl`) with image SHA-256 deduplication and held-out split contamination filtering.

![Dashboard Overview](file:///c:/Users/Abhiram/Documents/Projects/Doc/docs/screenshots/05_dashboard_overview.png)

![Admin & Jobs Ops](file:///c:/Users/Abhiram/Documents/Projects/Doc/docs/screenshots/06_admin_jobs_ops.png)

---

## 3. Reproducible Colab GPU Training Path (`LayoutLMv3`)

Genuine fine-tuning of `microsoft/layoutlmv3-base` requires PyTorch and GPU acceleration not assumed on local CPU machines. To train and evaluate on real SROIE receipts:

1. Open [DocuMind_LayoutLMv3_Training.ipynb](file:///c:/Users/Abhiram/Documents/Projects/Doc/ml/training/DocuMind_LayoutLMv3_Training.ipynb) in Google Colab with a T4 GPU runtime.
2. Execute the dataset preparation (`ml/data/prepare_dataset.py`), LayoutLMv3 fine-tuning (`ml/training/train_layoutlmv3.py`), held-out evaluation (`ml/evaluation/evaluate.py`), and `--run-full-ocr` baseline comparison (`ml/evaluation/compare_models.py`).
3. Download the verified checkpoint zip and inspect/validate it locally via `ml/training/import_checkpoint.py` before enabling `ENABLE_ML_INFERENCE=true`.

