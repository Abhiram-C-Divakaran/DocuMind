import pytest
from httpx import AsyncClient
from tests.conftest import MINIMAL_PDF_BYTES


@pytest.mark.asyncio
async def test_document_persistence_and_retrieval(client: AsyncClient):
    # 1. Upload a document with explicit placeholder method
    files = {"file": ("vendor_bill.pdf", MINIMAL_PDF_BYTES, "application/pdf")}
    data = {"extraction_method": "PLACEHOLDER"}
    upload_res = await client.post("/api/v1/documents", files=files, data=data)
    assert upload_res.status_code == 201
    doc = upload_res.json()["document"]
    doc_id = doc["id"]
    assert doc["status"] == "PENDING"

    # 2. Retrieve document list
    list_res = await client.get("/api/v1/documents")
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert list_data["total"] >= 1
    found_doc = next((d for d in list_data["items"] if d["id"] == doc_id), None)
    assert found_doc is not None
    assert found_doc["original_filename"] == "vendor_bill.pdf"

    # 3. Retrieve single document by ID
    single_res = await client.get(f"/api/v1/documents/{doc_id}")
    assert single_res.status_code == 200
    single_data = single_res.json()
    assert single_data["id"] == doc_id
    assert single_data["original_filename"] == "vendor_bill.pdf"
    assert "extractions" in single_data
    assert len(single_data["extractions"]) > 0
    extraction = single_data["extractions"][0]
    assert extraction["extraction_method"] == "PLACEHOLDER"
    assert "validation_errors" in extraction


@pytest.mark.asyncio
async def test_tesseract_processing_transition_and_persistence(client: AsyncClient):
    # Upload with TESSERACT_BASELINE
    files = {"file": ("receipt_scan.pdf", MINIMAL_PDF_BYTES, "application/pdf")}
    data = {"extraction_method": "TESSERACT_BASELINE"}
    upload_res = await client.post("/api/v1/documents", files=files, data=data)
    assert upload_res.status_code == 201
    doc = upload_res.json()["document"]
    # Depending on whether tesseract is installed on host, status is COMPLETED or FAILED
    assert doc["status"] in ["COMPLETED", "FAILED"]
    assert len(doc["extractions"]) > 0
    assert doc["extractions"][0]["extraction_method"] == "TESSERACT_BASELINE"


@pytest.mark.asyncio
async def test_unsupported_method_rejected(client: AsyncClient):
    files = {"file": ("test.pdf", MINIMAL_PDF_BYTES, "application/pdf")}
    data = {"extraction_method": "LAYOUTLMV3"}
    res = await client.post("/api/v1/documents", files=files, data=data)
    assert res.status_code == 400
    err = res.json()
    assert "unavailable in Phase 2A" in err["error"]["message"]


@pytest.mark.asyncio
async def test_document_not_found(client: AsyncClient):
    fake_id = "00000000-0000-0000-0000-000000000000"
    res = await client.get(f"/api/v1/documents/{fake_id}")
    assert res.status_code == 404
    data = res.json()
    assert data["error"]["code"] == "DOCUMENT_NOT_FOUND"


@pytest.mark.asyncio
async def test_metrics_summary_endpoint(client: AsyncClient):
    res = await client.get("/api/v1/documents/metrics/summary")
    assert res.status_code == 200
    metrics = res.json()
    assert "total_documents" in metrics
    assert "completed" in metrics
    assert "failed" in metrics
    assert "pending" in metrics
