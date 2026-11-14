"""
Self-Contained FastAPI Backend Runner for Playwright E2E Verification
---------------------------------------------------------------------
Launches FastAPI on an ephemeral port using temporary SQLite DB, temporary file
storage, an in-process background worker thread triggering real worker execution
(execute_processing_job_sync), and a seeded ADMIN account.

Intended strictly for browser E2E verification on hosts where Docker/Redis/Tesseract
are unavailable; does NOT alter production server defaults.
"""

import asyncio
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import uvicorn

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main() -> None:
    port = int(os.environ.get("E2E_BACKEND_PORT", "18011"))
    frontend_port = int(os.environ.get("E2E_FRONTEND_PORT", "14173"))
    admin_email = os.environ.get("E2E_ADMIN_EMAIL", "admin@documind.e2e")
    admin_password = os.environ.get("E2E_ADMIN_PASSWORD", "AdminStrong123!")

    tmp_root = Path(tempfile.mkdtemp(prefix="documind_e2e_"))
    db_path = tmp_root / "e2e.db"
    upload_dir = tmp_root / "uploads"
    feedback_dir = tmp_root / "feedback_exports"
    upload_dir.mkdir(parents=True, exist_ok=True)
    feedback_dir.mkdir(parents=True, exist_ok=True)

    # Configure isolated runtime settings before importing DB/app modules
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{db_path}"
    os.environ["SYNC_DATABASE_URL"] = f"sqlite:///{db_path}"
    os.environ["UPLOAD_DIR"] = str(upload_dir)
    os.environ["ENVIRONMENT"] = "development"
    os.environ["DEBUG"] = "false"
    os.environ["JWT_SECRET_KEY"] = "e2e_test_jwt_secret_key_over_32_characters_long_1234"
    os.environ["CORS_ORIGINS"] = (
        f"http://127.0.0.1:{frontend_port},http://localhost:{frontend_port},http://127.0.0.1:5173"
    )

    from backend.app.config import settings
    from backend.app.core.security import hash_password
    from backend.app.database import AsyncSessionLocal, Base, engine, reset_sync_engine
    from backend.app.main import app
    from backend.app.models.user import User, UserRole
    from backend.app.services.document_service import DocumentService
    from backend.app.services.storage import storage_service
    from backend.app.worker.tasks import execute_processing_job_sync
    from ml.base import ExtractionOutput
    from ml.ocr.engine import get_tesseract_status

    settings.DATABASE_URL = os.environ["DATABASE_URL"]
    settings.SYNC_DATABASE_URL = os.environ["SYNC_DATABASE_URL"]
    settings.UPLOAD_DIR = str(upload_dir)
    storage_service.base_dir = upload_dir
    reset_sync_engine()

    async def init_db_and_seed():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with AsyncSessionLocal() as session:
            admin = User(
                email=admin_email,
                full_name="E2E Administrator",
                password_hash=hash_password(admin_password),
                role=UserRole.ADMIN.value,
                is_active=True,
            )
            session.add(admin)
            await session.commit()

    asyncio.run(init_db_and_seed())

    def in_process_dispatch(self, job_id: str):
        def _run():
            time.sleep(0.5)
            execute_processing_job_sync(job_id)

        threading.Thread(target=_run, daemon=True).start()
        return f"e2e-thread-{job_id[:8]}", None

    # Realistic fallback extractor ONLY if host lacks Tesseract binary during local E2E
    tess_ok, _, tess_reason = get_tesseract_status()
    require_real_ocr = os.environ.get("REQUIRE_REAL_TESSERACT", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if not tess_ok and require_real_ocr:
        raise RuntimeError(
            f"REQUIRE_REAL_TESSERACT is enabled, but Tesseract is unavailable: {tess_reason}"
        )

    async def synthetic_e2e_extract(self, file_path):
        return ExtractionOutput(
            extraction_method="TESSERACT_BASELINE",
            structured_data={
                "vendor_name": "SUNRISE BAKERY",
                "vendor_address": "742 Evergreen Terrace, Austin, TX 78701",
                "document_date": "2026-10-18",
                "total_amount": 25.92,
                "tax_amount": 1.92,
                "currency": "USD",
                "line_items": [
                    {"description": "Sourdough Loaf x2", "quantity": 2, "unit_price": 6.00, "total_price": 12.00},
                    {"description": "Espresso Roast 500g", "quantity": 1, "unit_price": 8.50, "total_price": 8.50},
                ],
            },
            confidence=0.68,  # triggers PENDING_REVIEW automatically so review workflow works organically
            field_confidences={
                "vendor_name": 0.62,
                "document_date": 0.91,
                "total_amount": 0.88,
            },
            raw_text="SUNRISE ARTISAN BAKERY\nDATE: 2026-10-18\nSUBTOTAL USD 24.00\nTAX USD 1.92\nTOTAL PAID USD 25.92",
            validation_errors=["Vendor header confidence below 0.75 threshold (E2E sample)"],
            is_placeholder=False,
            metadata={"ocr_engine": "e2e_synthetic_fixture", "is_e2e_fixture": True},
        )

    from ml.feedback import export_annotations as fb_mod

    orig_export_from_session = fb_mod.export_feedback_from_session

    async def isolated_export_from_session(*, db, upload_dir, **kwargs):
        kwargs["output_dir"] = str(feedback_dir)
        return await orig_export_from_session(db=db, upload_dir=upload_dir, **kwargs)

    patches = [
        patch.object(DocumentService, "_dispatch_celery_job", in_process_dispatch),
        patch("backend.app.worker.tasks.process_document_job.delay", return_value=SimpleNamespace(id="e2e-delay")),
        patch("backend.app.api.routes.models._check_redis_status", return_value="connected"),
        patch("backend.app.api.routes.models._check_worker_status", return_value="available"),
        patch("backend.app.api.routes.health._check_redis_status", return_value="connected"),
        patch("backend.app.api.routes.health._check_worker_status", return_value="available"),
        patch("ml.feedback.export_annotations.export_feedback_from_session", isolated_export_from_session),
    ]
    if not tess_ok:
        patches.append(patch("ml.extractors.tesseract_baseline.TesseractBaselineExtractor.extract", synthetic_e2e_extract))

    for p in patches:
        p.start()

    try:
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", timeout_keep_alive=1)
    finally:
        for p in reversed(patches):
            p.stop()
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
