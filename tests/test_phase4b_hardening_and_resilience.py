import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import Request
from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import get_client_ip
from backend.app.config import settings
from backend.app.core.security import SlidingWindowRateLimiter
from backend.app.models.document import Document, DocumentStatus
from backend.app.models.extraction import ExtractionResult
from backend.app.models.job import JobStatus, OutboxEvent, ProcessingJob
from backend.app.services.document_service import DocumentService
from backend.app.worker.tasks import (
    execute_processing_job_sync,
    recover_stale_jobs_and_outbox_sync,
)
from ml.base import ExtractionOutput
from ml.data.annotation_alignment import ID2LABEL
from ml.training.import_checkpoint import (
    main as import_checkpoint_main,
    validate_and_smoke_test_checkpoint,
)
from tests.conftest import MINIMAL_PNG_BYTES


def _sample_output() -> ExtractionOutput:
    return ExtractionOutput(
        extraction_method="TESSERACT_BASELINE",
        structured_data={"vendor_name": "RESILIENCE MART", "total_amount": 42.50},
        confidence=0.92,
        field_confidences={"vendor_name": 0.95, "total_amount": 0.90},
        raw_text="RESILIENCE MART\nTOTAL 42.50",
        validation_errors=[],
        is_placeholder=False,
        metadata={"ocr_engine": "tesseract"},
    )


@pytest.mark.asyncio
async def test_cookie_refresh_csrf_protection_replay_and_logout(
    anon_client: AsyncClient,
    test_users: dict,
):
    # Login in cookie transport mode: omits refresh_token from JSON body and sets HttpOnly cookie + CSRF cookie
    res = await anon_client.post(
        "/api/v1/auth/login",
        headers={"X-Auth-Transport": "cookie"},
        json={"email": "reviewer@test.local", "password": "StrongPass123!"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["refresh_token"] is None
    assert res.headers.get("Cache-Control") == "no-store"

    set_cookies = res.headers.get_list("set-cookie")
    refresh_cookie_header = next(c for c in set_cookies if "documind_refresh_token=" in c)
    assert "httponly" in refresh_cookie_header.lower()
    assert "samesite=lax" in refresh_cookie_header.lower()
    assert "path=/api/v1/auth" in refresh_cookie_header.lower()

    csrf_token = anon_client.cookies.get(settings.CSRF_COOKIE_NAME)
    assert csrf_token

    # Cookie-based refresh WITHOUT X-CSRF-Token must be blocked (403 CSRF_INVALID)
    no_csrf = await anon_client.post(
        "/api/v1/auth/refresh",
        headers={"X-Auth-Transport": "cookie"},
        json={},
    )
    assert no_csrf.status_code == 403
    assert no_csrf.json()["error"]["code"] == "CSRF_INVALID"

    # Cookie-based refresh WITH valid X-CSRF-Token succeeds and rotates both cookies
    old_refresh_cookie = anon_client.cookies.get(settings.REFRESH_COOKIE_NAME)
    ok_refresh = await anon_client.post(
        "/api/v1/auth/refresh",
        headers={"X-Auth-Transport": "cookie", "X-CSRF-Token": csrf_token},
        json={},
    )
    assert ok_refresh.status_code == 200
    assert ok_refresh.json()["refresh_token"] is None
    rotated_csrf = anon_client.cookies.get(settings.CSRF_COOKIE_NAME)
    rotated_refresh = anon_client.cookies.get(settings.REFRESH_COOKIE_NAME)
    assert rotated_refresh and rotated_refresh != old_refresh_cookie
    assert rotated_csrf

    # Logout with cookie + CSRF revokes server session and clears cookies
    access_token = ok_refresh.json()["access_token"]
    logout_res = await anon_client.post(
        "/api/v1/auth/logout",
        headers={
            "Authorization": f"Bearer {access_token}",
            "X-CSRF-Token": rotated_csrf,
        },
        json={},
    )
    assert logout_res.status_code == 200
    assert logout_res.json()["revoked_tokens"] >= 1


@pytest.mark.asyncio
async def test_proxy_ip_spoofing_ignored_unless_trusted_and_fail_closed_rate_limiter(
    anon_client: AsyncClient,
):
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/auth/login",
        "headers": [(b"x-forwarded-for", b"203.0.113.99")],
        "client": ("127.0.0.10", 54321),
    }
    req = Request(scope)

    with patch.object(settings, "TRUST_PROXY_HEADERS", False):
        assert get_client_ip(req) == "127.0.0.10"

    with patch.object(settings, "TRUST_PROXY_HEADERS", True):
        assert get_client_ip(req) == "203.0.113.99"

    # Fail-closed rate limiter in production rejects login with 503 when Redis is unavailable
    rl = SlidingWindowRateLimiter()
    with (
        patch.object(settings, "ENVIRONMENT", "production"),
        patch.object(settings, "RATE_LIMIT_FAIL_CLOSED", True),
        patch.object(rl, "_get_redis", return_value=None),
        patch("backend.app.api.routes.auth.rate_limiter", rl),
    ):
        res = await anon_client.post(
            "/api/v1/auth/login",
            json={"email": "reviewer@test.local", "password": "StrongPass123!"},
        )
        assert res.status_code == 503
        assert res.json()["error"]["code"] == "RATE_LIMITER_UNAVAILABLE"


@pytest.mark.asyncio
async def test_health_reports_unknown_worker_when_liveness_unproven_and_no_secret_leaks(
    anon_client: AsyncClient,
):
    with (
        patch("backend.app.api.routes.health._check_redis_status", return_value="connected"),
        patch("backend.app.api.routes.health._check_worker_status", return_value="unknown"),
    ):
        res = await anon_client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["worker"] == "unknown"
        serialized = json.dumps(data)
        assert "postgresql" not in serialized
        assert "redis://" not in serialized
        assert "JWT_SECRET" not in serialized
        assert res.headers.get("Permissions-Policy") is not None


@pytest.mark.asyncio
async def test_broker_outage_outbox_recovery_duplicate_delivery_retry_exhaustion_and_soft_timeout(
    client: AsyncClient,
    test_db_session: AsyncSession,
):
    # 1. Broker down during upload still persists Document, ProcessingJob & unpublished OutboxEvent cleanly
    with patch.object(
        DocumentService,
        "_dispatch_celery_job",
        return_value=(None, "Broker ConnectionRefusedError"),
    ):
        up = await client.post(
            "/api/v1/documents",
            files={"file": ("outbox_broker_down.png", MINIMAL_PNG_BYTES, "image/png")},
            data={"extraction_method": "TESSERACT_BASELINE"},
        )
        assert up.status_code == 202
        doc_id = up.json()["document"]["id"]
        job_id = up.json()["job"]["id"]

    ev = (
        await test_db_session.execute(
            select(OutboxEvent).where(OutboxEvent.aggregate_id == job_id)
        )
    ).scalar_one()
    assert ev.published_at is None
    assert "ConnectionRefused" in (ev.last_error or "")

    # Force outbox event age so recovery immediately picks it up
    from datetime import datetime, timedelta, timezone

    ev.created_at = datetime.now(timezone.utc) - timedelta(seconds=30)
    await test_db_session.commit()

    with patch("backend.app.worker.tasks.process_document_job.delay", return_value=SimpleNamespace(id="rec-task-1")):
        rec = recover_stale_jobs_and_outbox_sync()
        assert rec["republished_outbox"] >= 1

    # 2. Duplicate delivery / idempotent re-execution must NEVER create duplicate ExtractionResult rows
    with patch("backend.app.worker.tasks.TesseractBaselineExtractor.extract", return_value=_sample_output()) as mock_ext:
        res1 = execute_processing_job_sync(job_id)
        assert res1["status"] == "COMPLETED"
        res2 = execute_processing_job_sync(job_id)
        assert res2["status"] == "COMPLETED"
        assert res2.get("idempotent_skip") is True
        assert mock_ext.call_count == 1

    ext_count = (
        await test_db_session.execute(
            select(func.count(ExtractionResult.id)).where(ExtractionResult.document_id == doc_id)
        )
    ).scalar_one()
    assert ext_count == 1

    # 3. Retry exhaustion & SoftTimeLimitExceeded handling
    class SoftTimeLimitExceeded(Exception):
        pass

    up_timeout = await client.post(
        "/api/v1/documents",
        files={"file": ("timeout_doc.png", MINIMAL_PNG_BYTES, "image/png")},
        data={"extraction_method": "TESSERACT_BASELINE"},
    )
    job_t_id = up_timeout.json()["job"]["id"]

    with patch(
        "backend.app.worker.tasks.TesseractBaselineExtractor.extract",
        side_effect=SoftTimeLimitExceeded("Task exceeded 120s soft limit"),
    ):
        for _ in range(settings.MAX_JOB_ATTEMPTS):
            out_t = execute_processing_job_sync(job_t_id)

    assert out_t["status"] == "FAILED"
    assert out_t["error_code"] == "WORKER_SOFT_TIMEOUT"

    # 4. DB transient OperationalError during extraction marks job retrying cleanly using fresh session
    up_db = await client.post(
        "/api/v1/documents",
        files={"file": ("db_drop.png", MINIMAL_PNG_BYTES, "image/png")},
        data={"extraction_method": "TESSERACT_BASELINE"},
    )
    job_db_id = up_db.json()["job"]["id"]
    with patch(
        "backend.app.worker.tasks.TesseractBaselineExtractor.extract",
        side_effect=OperationalError("SELECT 1", {}, Exception("connection reset by peer")),
    ):
        db_fail = execute_processing_job_sync(job_db_id)
        assert db_fail["status"] == "RETRYING"


def test_checkpoint_import_utility_rejects_zip_slip_missing_and_synthetic_promotion(tmp_path: Path):
    # 1. Zip Slip path traversal archive rejected cleanly
    evil_zip = tmp_path / "evil_ckpt.zip"
    with zipfile.ZipFile(evil_zip, "w") as zf:
        zf.writestr("../../outside.txt", "pwned")

    zip_report = validate_and_smoke_test_checkpoint(evil_zip, target_dir=tmp_path / "unpacked")
    assert zip_report["compatible"] is False
    assert zip_report["checks"]["archive_extraction"] == "FAIL"

    # 2. Synthetic smoke checkpoint must NEVER be marked production_ready or applied to .env
    ckpt_dir = tmp_path / "synthetic_ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "layoutlmv3",
                "id2label": {str(k): v for k, v in ID2LABEL.items()},
                "label2id": {v: k for k, v in ID2LABEL.items()},
            }
        ),
        encoding="utf-8",
    )
    (ckpt_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (ckpt_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (ckpt_dir / "model.safetensors").write_bytes(b"fake_weights_bytes")
    (ckpt_dir / "checkpoint_metadata.json").write_text(
        json.dumps({"is_synthetic": True, "smoke_test": True}),
        encoding="utf-8",
    )

    env_file = tmp_path / ".env"
    env_file.write_text("LAYOUTLMV3_CHECKPOINT_DIR=checkpoints/original\n", encoding="utf-8")

    # Mock HAS_TORCH + forward pass to verify synthetic flag STILL blocks promotion to .env
    mock_logits = MagicMock()
    mock_logits.shape = (1, 10, len(ID2LABEL))
    fake_probs = MagicMock()
    fake_probs.argmax.return_value.tolist.return_value = [0] * 10
    fake_probs.max.return_value.values.tolist.return_value = [0.99] * 10

    with (
        patch("ml.training.import_checkpoint.HAS_TORCH", True),
        patch(
            "ml.training.import_checkpoint.prepare_layoutlmv3_inputs",
            return_value={"word_ids": [None, 0, None]},
        ),
        patch("ml.training.import_checkpoint.decode_layoutlmv3_predictions", return_value=({}, {}, {})),
    ):
        with patch.dict("sys.modules", {"torch": MagicMock()}):
            import sys as _sys

            _torch = _sys.modules["torch"]
            _torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
            _torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
            _torch.softmax.return_value = fake_probs

            # Fake model returns object with .logits[0] having shape[-1] == 9
            fake_model = MagicMock()
            fake_model.return_value.logits.__getitem__.return_value.shape = (10, len(ID2LABEL))

            def _fake_load(self):
                self._tokenizer = MagicMock()
                self._image_processor = MagicMock()
                self._model = fake_model

            with patch("ml.training.import_checkpoint.LayoutLMv3Extractor.load_model", _fake_load):
                exit_code = import_checkpoint_main(
                    [
                        "--checkpoint",
                        str(ckpt_dir),
                        "--apply-env",
                        "--env-file",
                        str(env_file),
                    ]
                )
                assert exit_code == 2  # compatible smoke checkpoint, NOT production ready
                assert "LAYOUTLMV3_CHECKPOINT_DIR=checkpoints/original" in env_file.read_text(encoding="utf-8")

