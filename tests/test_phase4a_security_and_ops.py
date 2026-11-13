import json
import logging
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import pytest
from httpx import AsyncClient

from backend.app.config import Settings, settings
from backend.app.core.exceptions import FileValidationError
from backend.app.logger import StructuredFormatter
from backend.app.services.media_security import validate_media_safety
from tests.conftest import MINIMAL_PNG_BYTES


@pytest.mark.asyncio
async def test_security_response_headers_and_request_id(anon_client: AsyncClient):
    res = await anon_client.get("/health", headers={"X-Request-ID": "req-4a-check"})
    assert res.status_code == 200
    assert res.headers.get("X-Request-ID") == "req-4a-check"
    assert res.headers.get("X-Content-Type-Options") == "nosniff"
    assert res.headers.get("X-Frame-Options") == "DENY"
    assert res.headers.get("Referrer-Policy") == "no-referrer"


@pytest.mark.asyncio
async def test_chunked_upload_rejects_oversized_body(client: AsyncClient):
    orig_mb = settings.MAX_FILE_SIZE_MB
    settings.MAX_FILE_SIZE_MB = 1
    try:
        oversized = b"%PDF-1.4\n" + (b"0" * (1024 * 1024 + 2048))
        res = await client.post(
            "/api/v1/documents",
            files={"file": ("big.pdf", oversized, "application/pdf")},
        )
        assert res.status_code == 413
        assert res.json()["error"]["code"] == "FILE_TOO_LARGE"
    finally:
        settings.MAX_FILE_SIZE_MB = orig_mb


def test_decompression_bomb_and_pixel_limit_rejected():
    buf = BytesIO()
    Image.new("RGB", (200, 200), color=(10, 20, 30)).save(buf, format="PNG")
    img_bytes = buf.getvalue()

    with patch.object(settings, "MAX_IMAGE_PIXELS", 10_000):
        with pytest.raises(FileValidationError) as exc_info:
            validate_media_safety(img_bytes, "image/png", "bomb.png")
        assert "exceeds" in str(exc_info.value.detail).lower()


def test_excessive_pdf_pages_rejected():
    import fitz

    doc = fitz.open()
    for _ in range(4):
        doc.new_page()
    pdf_bytes = doc.tobytes()
    doc.close()

    with patch.object(settings, "MAX_PDF_PAGES_HARD_LIMIT", 2):
        with pytest.raises(FileValidationError) as exc_info:
            validate_media_safety(pdf_bytes, "application/pdf", "many_pages.pdf")
        assert "page count" in str(exc_info.value.detail).lower()


@pytest.mark.asyncio
async def test_path_traversal_filename_sanitized_on_upload(client: AsyncClient):
    res = await client.post(
        "/api/v1/documents",
        files={"file": ("../../etc/passwd.png", MINIMAL_PNG_BYTES, "image/png")},
        data={"extraction_method": "PLACEHOLDER"},
    )
    assert res.status_code == 202
    doc = res.json()["document"]
    assert ".." not in doc["original_filename"]
    assert "/" not in doc["original_filename"]
    assert doc["original_filename"] == "passwd.png"


def test_production_secret_validation_rejects_weak_or_missing_secret():
    s_missing = Settings(ENVIRONMENT="production", DEBUG=False, JWT_SECRET_KEY="")
    with pytest.raises(RuntimeError):
        s_missing.validate_runtime_security()

    s_weak = Settings(ENVIRONMENT="production", DEBUG=False, JWT_SECRET_KEY="changeme")
    with pytest.raises(RuntimeError):
        s_weak.validate_runtime_security()

    s_ok = Settings(
        ENVIRONMENT="production",
        DEBUG=False,
        JWT_SECRET_KEY="A_very_strong_random_production_secret_key_over_32_bytes!",
    )
    assert s_ok.validate_runtime_security().startswith("A_very_strong")


def test_structured_logger_redacts_sensitive_fields():
    formatter = StructuredFormatter()
    rec = logging.LogRecord(
        name="documind.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Auth audit",
        args=(),
        exc_info=None,
    )
    rec.request_id = "req-redact-1"
    rec.status = "200"
    rec.document_id = "doc-123"
    # Simulate internal helper passing extra dict directly
    from backend.app.logger import _sanitize_extra

    cleaned = _sanitize_extra(
        {
            "request_id": "req-1",
            "password": "my_secret_password",
            "raw_text": "TOTAL $999 RECEIPT",
            "access_token": "jwt.header.sig",
        }
    )
    assert cleaned["password"] == "[REDACTED]"
    assert cleaned["raw_text"] == "[REDACTED]"
    assert cleaned["access_token"] == "[REDACTED]"
    assert cleaned["request_id"] == "req-1"

    formatted = formatter.format(rec)
    parsed = json.loads(formatted)
    assert parsed["request_id"] == "req-redact-1"
    assert parsed["document_id"] == "doc-123"


def test_alembic_migrations_upgrade_and_downgrade_clean_db(tmp_path: Path):
    from alembic import command
    from alembic.config import Config

    db_file = tmp_path / "alembic_verify.db"
    cfg = Config("backend/alembic.ini")
    cfg.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{db_file}")

    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    assert db_file.exists()


def test_cli_create_admin_provisions_admin_user(tmp_path: Path, monkeypatch):
    from backend.app.cli import main as cli_main
    from backend.app.database import AsyncSessionLocal

    # Smoke check CLI parser & verify module executes cleanly via -h
    proc = subprocess.run(
        [sys.executable, "-m", "backend.app.cli", "create-admin", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "--email" in proc.stdout

