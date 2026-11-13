# DocuMind Testing & Verification Guide

This guide covers unit, integration, browser E2E, database migration, and security regression testing across the DocuMind stack.

---

## 1. Backend Test Suite (`pytest`)

Run the full backend + ML pipeline suite from the repository root:

```bash
python -m pytest -v tests
```

Targeted suites:

- **Upload validation & magic-byte checks**: `python -m pytest -v tests/test_upload_validation.py`
- **Persistence & health endpoint**: `python -m pytest -v tests/test_document_persistence.py tests/test_health.py`
- **OCR rules & PDF rendering**: `python -m pytest -v tests/test_phase2a_ocr.py`
- **LayoutLMv3 data prep & inference contract**: `python -m pytest -v tests/test_phase2b_layoutlmv3.py tests/test_phase2c_ml_pipeline.py`
- **Review state machine & active learning export**: `python -m pytest -v tests/test_phase3a_review_and_feedback.py`
- **JWT Auth, RBAC, worker retry/outbox, upload bomb & log redaction**: `python -m pytest -v tests/test_phase4a_auth_rbac.py tests/test_phase4a_jobs_and_worker.py tests/test_phase4a_security_and_ops.py`
- **HttpOnly CSRF cookies, fail-closed rate limiting, broker outage & checkpoint safety**: `python -m pytest -v tests/test_phase4b_hardening_and_resilience.py`
- **End-to-end unmocked Tesseract OCR & review integration**: `python -m pytest -v tests/test_phase4b_ocr_and_review_integration.py`

> [!NOTE]
> `test_real_tesseract_ocr_and_review_workflow_unmocked` requires `tesseract` on `PATH` (as provided inside the backend/worker Docker container and on Ubuntu CI runners). On hosts without `tesseract`, it skips explicitly rather than faking OCR.

---

## 2. Database Migration Verification (Alembic)

Validate clean schema upgrade, teardown, and re-upgrade:

```bash
alembic upgrade head
alembic downgrade base
alembic upgrade head
```

This lifecycle is also automated inside `tests/test_phase4a_security_and_ops.py::test_alembic_migrations_upgrade_and_downgrade_clean_db` and in GitHub Actions CI.

---

## 3. Frontend Static Checks & Production Build

```bash
cd frontend
npm ci
npm audit --omit=dev
npm run lint
npm run build
```

- `npm run lint` runs TypeScript type checking (`tsc --noEmit`).
- `npm run build` compiles the Vite production bundle.

---

## 4. Playwright Real-Browser End-to-End Tests

Playwright exercises the compiled/served React app against a live FastAPI backend (`scripts/e2e_test_server.py`) with real cookies, CSRF tokens, RBAC accounts (`ADMIN`, `REVIEWER`, `VIEWER`), async polling, draft corrections, approvals, audit diffs, feedback JSONL export, and automated UI screenshots saved under `docs/screenshots/`.

Generate the test fixture if not present and run the E2E suite:

```bash
python scripts/generate_sample_receipt.py
cd frontend
npm run test:e2e
```

If running behind an HTTP proxy environment (e.g., corporate proxy or sandbox), ensure localhost bypasses the proxy (`NO_PROXY=127.0.0.1,localhost`). `frontend/playwright.config.ts` automatically clears proxy variables for its spawned `webServer` processes.

---

## 5. Docker Compose Validation

```bash
docker compose -f docker-compose.yml config -q
docker compose -f docker-compose.yml -f docker-compose.prod.yml config -q
docker compose build
```

---

## 6. Automated CI/CD Pipeline

[.github/workflows/ci.yml](file:///c:/Users/Abhiram/Documents/Projects/Doc/.github/workflows/ci.yml) runs on every push and pull request to `main`:
1. `backend`: Installs `tesseract-ocr`, Python dependencies, verifies Alembic `upgrade/downgrade/upgrade`, and runs `pytest -v tests` (including unmocked Tesseract OCR integration).
2. `frontend-and-e2e`: Audits production npm dependencies, type-checks, builds, installs Playwright Chromium, and executes `frontend/e2e/documind.spec.ts`.
3. `docker-compose-validate`: Validates development & production Compose manifests and builds backend, worker, and frontend images.

