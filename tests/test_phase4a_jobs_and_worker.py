from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.document import Document, DocumentStatus
from backend.app.models.extraction import ExtractionResult
from backend.app.models.job import JobStatus, OutboxEvent, ProcessingJob
from backend.app.services.document_service import DocumentService
from backend.app.worker.tasks import execute_processing_job_sync, recover_stale_jobs_and_outbox_sync
from ml.base import ExtractionOutput
from tests.conftest import MINIMAL_PNG_BYTES


def _sample_ocr_output() -> ExtractionOutput:
    return ExtractionOutput(
        extraction_method="TESSERACT_BASELINE",
        is_placeholder=False,
        structured_data={
            "vendor_name": "KOPITIAM HUB",
            "vendor_address": "12 JALAN TUN",
            "document_date": "2025-02-01",
            "total_amount": "24.00",
            "currency": "MYR",
            "tax_amount": "1.20",
            "line_items": [],
        },
        confidence=0.88,
        field_confidences={"vendor_name": 0.9, "total_amount": 0.95},
        raw_text="KOPITIAM HUB TOTAL 24.00",
        validation_errors=[],
        metadata={"engine": "tesseract"},
    )


@pytest.mark.asyncio
async def test_async_upload_queues_202_and_worker_completes_idempotently(
    client: AsyncClient,
    test_db_session: AsyncSession,
):
    up = await client.post(
        "/api/v1/documents",
        files={"file": ("queue_test.png", MINIMAL_PNG_BYTES, "image/png")},
        data={"extraction_method": "TESSERACT_BASELINE"},
    )
    assert up.status_code == 202
    payload = up.json()
    doc_id = payload["document"]["id"]
    job_id = payload["job"]["id"]

    # Document must NOT be prematurely COMPLETED before worker runs
    assert payload["document"]["status"] == DocumentStatus.PENDING.value
    assert payload["job"]["status"] == JobStatus.QUEUED.value

    st_before = await client.get(f"/api/v1/documents/{doc_id}/status")
    assert st_before.status_code == 200
    assert st_before.json()["document_status"] == "PENDING"
    assert st_before.json()["latest_job"]["status"] == "QUEUED"

    with patch("backend.app.worker.tasks.TesseractBaselineExtractor.extract", return_value=_sample_ocr_output()):
        res1 = execute_processing_job_sync(job_id)
        assert res1["status"] == "COMPLETED"
        # Re-delivering the exact same job must be idempotent and never create duplicate ExtractionResult rows
        res2 = execute_processing_job_sync(job_id)
        assert res2["status"] == "COMPLETED"
        assert res2.get("idempotent_skip") is True

    test_db_session.expire_all()
    ext_rows = list(
        (await test_db_session.execute(select(ExtractionResult).where(ExtractionResult.document_id == doc_id)))
        .scalars()
        .all()
    )
    assert len(ext_rows) == 1
    assert ext_rows[0].structured_data["vendor_name"] == "KOPITIAM HUB"

    st_after = await client.get(f"/api/v1/documents/{doc_id}/status")
    assert st_after.status_code == 200
    assert st_after.json()["document_status"] == "COMPLETED"
    assert st_after.json()["latest_job"]["status"] == "COMPLETED"
    assert st_after.json()["latest_job"]["attempts"] == 1


@pytest.mark.asyncio
async def test_transient_retry_then_success_and_permanent_failure_no_fallback(
    client: AsyncClient,
    admin_client: AsyncClient,
    test_db_session: AsyncSession,
):
    # 1. Transient OSError schedules RETRYING cleanly, then succeeds on retry
    up1 = await client.post(
        "/api/v1/documents",
        files={"file": ("retry_ok.png", MINIMAL_PNG_BYTES, "image/png")},
        data={"extraction_method": "TESSERACT_BASELINE"},
    )
    assert up1.status_code == 202
    job1_id = up1.json()["job"]["id"]
    doc1_id = up1.json()["document"]["id"]

    with patch("backend.app.worker.tasks.TesseractBaselineExtractor.extract", side_effect=OSError("Temporary I/O glitch")):
        out1 = execute_processing_job_sync(job1_id)
        assert out1["status"] == "RETRYING"

    test_db_session.expire_all()
    job1_row = await test_db_session.get(ProcessingJob, job1_id)
    assert job1_row.status == "RETRYING"
    assert job1_row.attempts == 1

    with patch("backend.app.worker.tasks.TesseractBaselineExtractor.extract", return_value=_sample_ocr_output()):
        out2 = execute_processing_job_sync(job1_id)
        assert out2["status"] == "COMPLETED"
    test_db_session.expire_all()

    # 2. Permanent failure (ValueError / missing checkpoint) fails immediately WITHOUT silent fallback
    up2 = await client.post(
        "/api/v1/documents",
        files={"file": ("perm_fail.png", MINIMAL_PNG_BYTES, "image/png")},
        data={"extraction_method": "TESSERACT_BASELINE"},
    )
    job2_id = up2.json()["job"]["id"]
    doc2_id = up2.json()["document"]["id"]

    with patch(
        "backend.app.worker.tasks.TesseractBaselineExtractor.extract",
        side_effect=ValueError("OCR dependency missing on host"),
    ):
        fail_out = execute_processing_job_sync(job2_id)
        assert fail_out["status"] == "FAILED"

    doc2_res = await client.get(f"/api/v1/documents/{doc2_id}")
    assert doc2_res.json()["status"] == "FAILED"

    # Retry endpoint re-queues failed doc -> 202 Accepted
    retry_res = await client.post(f"/api/v1/documents/{doc2_id}/retry")
    assert retry_res.status_code == 202
    assert retry_res.json()["document"]["status"] == "PENDING"
    new_job_id = retry_res.json()["job"]["id"]
    assert new_job_id != job2_id

    # Conflict if retrying while active QUEUED job exists
    conflict = await client.post(f"/api/v1/documents/{doc2_id}/retry")
    assert conflict.status_code == 409

    # Cancel active job
    cancel_res = await client.post(f"/api/v1/documents/{doc2_id}/cancel")
    assert cancel_res.status_code == 200
    assert cancel_res.json()["status"] == "CANCELLED"

    # Metrics reflect real DB job counts
    metrics_res = await admin_client.get("/api/v1/jobs/metrics")
    assert metrics_res.status_code == 200
    m = metrics_res.json()
    assert m["total_jobs"] >= 3
    assert m["completed"] >= 1
    assert m["failed"] >= 1
    assert m["cancelled"] >= 1

    # Delete completed document removes record cleanly
    del_res = await client.delete(f"/api/v1/documents/{doc1_id}")
    assert del_res.status_code == 200
    assert (await client.get(f"/api/v1/documents/{doc1_id}")).status_code == 404


@pytest.mark.asyncio
async def test_outbox_recovery_and_stale_job_recovery(
    client: AsyncClient,
    test_db_session: AsyncSession,
):
    # Simulate broker failure during upload -> OutboxEvent stays unpublished
    with patch.object(DocumentService, "_dispatch_celery_job", return_value=(None, "Broker offline")):
        up = await client.post(
            "/api/v1/documents",
            files={"file": ("outbox.png", MINIMAL_PNG_BYTES, "image/png")},
            data={"extraction_method": "TESSERACT_BASELINE"},
        )
    assert up.status_code == 202
    job_id = up.json()["job"]["id"]

    # Backdate OutboxEvent and create a stale RUNNING job to exercise recovery
    test_db_session.expire_all()
    ev = (
        await test_db_session.execute(select(OutboxEvent).where(OutboxEvent.aggregate_id == job_id))
    ).scalar_one()
    ev.created_at = datetime.now(timezone.utc) - timedelta(seconds=30)

    stale_doc = Document(
        original_filename="stale.png",
        stored_filename="stale.png",
        file_path="storage/uploads/stale.png",
        file_size_bytes=50,
        content_type="image/png",
        status=DocumentStatus.PROCESSING.value,
    )
    test_db_session.add(stale_doc)
    await test_db_session.flush()
    stale_job = ProcessingJob(
        document_id=stale_doc.id,
        idempotency_key="stale:test:1",
        extraction_method="TESSERACT_BASELINE",
        status=JobStatus.PROCESSING.value,
        attempts=1,
        max_attempts=3,
        started_at=datetime.now(timezone.utc) - timedelta(seconds=600),
        heartbeat_at=datetime.now(timezone.utc) - timedelta(seconds=600),
    )
    test_db_session.add(stale_job)
    await test_db_session.commit()

    with patch("backend.app.worker.tasks.process_document_job.delay", return_value=type("R", (), {"id": "rec-task"})()):
        summary = recover_stale_jobs_and_outbox_sync()

    assert summary["republished_outbox"] >= 1
    assert summary["requeued_stale_jobs"] >= 1

