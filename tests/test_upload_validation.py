import io
import pytest
from httpx import AsyncClient
from tests.conftest import MINIMAL_JPEG_BYTES, MINIMAL_PDF_BYTES, MINIMAL_PNG_BYTES


@pytest.mark.asyncio
async def test_upload_valid_pdf(client: AsyncClient):
    files = {"file": ("test_invoice.pdf", MINIMAL_PDF_BYTES, "application/pdf")}
    response = await client.post("/api/v1/documents", files=files)
    assert response.status_code == 201
    data = response.json()
    assert "document" in data
    doc = data["document"]
    assert doc["original_filename"] == "test_invoice.pdf"
    assert doc["content_type"] == "application/pdf"
    assert doc["file_size_bytes"] == len(MINIMAL_PDF_BYTES)
    assert len(doc["id"]) == 36  # Valid UUID
    assert doc["status"] in ["COMPLETED", "PENDING"]


@pytest.mark.asyncio
async def test_upload_valid_png(client: AsyncClient):
    files = {"file": ("receipt_scan.png", MINIMAL_PNG_BYTES, "image/png")}
    response = await client.post("/api/v1/documents", files=files)
    assert response.status_code == 201
    doc = response.json()["document"]
    assert doc["original_filename"] == "receipt_scan.png"
    assert doc["content_type"] == "image/png"


@pytest.mark.asyncio
async def test_upload_valid_jpeg(client: AsyncClient):
    files = {"file": ("receipt_photo.jpg", MINIMAL_JPEG_BYTES, "image/jpeg")}
    response = await client.post("/api/v1/documents", files=files)
    assert response.status_code == 201
    doc = response.json()["document"]
    assert doc["original_filename"] == "receipt_photo.jpg"
    assert doc["content_type"] == "image/jpeg"


@pytest.mark.asyncio
async def test_upload_empty_file(client: AsyncClient):
    files = {"file": ("empty.pdf", b"", "application/pdf")}
    response = await client.post("/api/v1/documents", files=files)
    assert response.status_code == 400
    err = response.json()
    assert err["error"]["code"] == "INVALID_FILE"
    assert "empty" in err["error"]["message"].lower()


@pytest.mark.asyncio
async def test_upload_disallowed_extension(client: AsyncClient):
    files = {"file": ("script.exe", b"%PDF-dummy", "application/pdf")}
    response = await client.post("/api/v1/documents", files=files)
    assert response.status_code == 400
    err = response.json()
    assert err["error"]["code"] == "INVALID_FILE"
    assert "extension" in err["error"]["message"].lower()


@pytest.mark.asyncio
async def test_upload_mismatched_magic_bytes(client: AsyncClient):
    # Declares PDF MIME and extension, but content is garbage text
    files = {"file": ("fake_invoice.pdf", b"This is plain text pretending to be PDF", "application/pdf")}
    response = await client.post("/api/v1/documents", files=files)
    assert response.status_code == 400
    err = response.json()
    assert err["error"]["code"] == "INVALID_FILE"
    assert "signature" in err["error"]["message"].lower()


@pytest.mark.asyncio
async def test_upload_mismatched_mime_and_header(client: AsyncClient):
    # Valid PNG bytes, but declared as application/pdf and .pdf extension
    files = {"file": ("inconsistent.pdf", MINIMAL_PNG_BYTES, "application/pdf")}
    response = await client.post("/api/v1/documents", files=files)
    assert response.status_code == 400
    err = response.json()
    assert err["error"]["code"] == "INVALID_FILE"


@pytest.mark.asyncio
async def test_upload_oversized_file(client: AsyncClient):
    # Simulate oversized file by passing a byte array larger than 15MB or mocking settings
    from backend.app.config import settings
    # Test validator directly or with temporary limit
    from backend.app.services.validator import validate_file
    from backend.app.core.exceptions import FileSizeLimitExceededError

    with pytest.raises(FileSizeLimitExceededError):
        # Limit set to 100 bytes for test
        validate_file("test.pdf", "application/pdf", MINIMAL_PDF_BYTES, max_size_bytes=50)

