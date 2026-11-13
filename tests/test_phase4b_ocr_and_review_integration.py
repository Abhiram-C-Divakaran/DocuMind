import json
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest
from httpx import AsyncClient
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.document import DocumentStatus
from backend.app.models.job import JobStatus
from backend.app.services.storage import storage_service
from backend.app.worker.tasks import execute_processing_job_sync
from ml.base import ExtractionOutput
from ml.feedback.export_annotations import export_feedback_from_session
from ml.ocr.engine import get_tesseract_status


def _render_crisp_receipt_bytes() -> bytes:
    """Renders a crisp, high-contrast synthetic receipt PNG legible to Tesseract OCR."""
    img = Image.new("RGB", (900, 1100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.load_default(size=36)
        body_font = ImageFont.load_default(size=26)
    except TypeError:  # pragma: no cover - Pillow <10.1 fallback
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    lines = [
        (70, 70, "SUNRISE ARTISAN BAKERY", title_font),
        (70, 140, "742 Evergreen Terrace, Austin, TX", body_font),
        (70, 210, "Date: 2026-10-18", body_font),
        (70, 270, "1x Sourdough Loaf       8.50", body_font),
        (70, 320, "2x Cold Brew Coffee    11.00", body_font),
        (70, 370, "1x Almond Croissant     4.50", body_font),
        (70, 440, "Subtotal               24.00", body_font),
        (70, 490, "Tax                     1.92", body_font),
        (70, 560, "TOTAL USD              25.92", title_font),
    ]
    for x, y, text, font in lines:
        draw.text((x, y), text, fill=(0, 0, 0), font=font)

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


async def _login_bearer_headers(anon_client: AsyncClient, email: str, password: str) -> dict[str, str]:
    res = await anon_client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
    )
    assert res.status_code == 200, res.text
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_real_tesseract_ocr_and_review_workflow_unmocked(
    anon_client: AsyncClient,
    test_users: dict,
    test_db_session: AsyncSession,
    tmp_path: Path,
):
    """Unmocked integration test: login -> 202 upload -> Celery worker -> real Tesseract OCR -> review -> export."""
    available, _, reason = get_tesseract_status()
    if not available:
        pytest.skip(f"Real Tesseract OCR binary unavailable on host: {reason}")

    headers = await _login_bearer_headers(anon_client, "reviewer@test.local", "StrongPass123!")
    receipt_bytes = _render_crisp_receipt_bytes()

    up = await anon_client.post(
        "/api/v1/documents",
        headers=headers,
        files={"file": ("synthetic_ocr_test.png", receipt_bytes, "image/png")},
        data={"extraction_method": "TESSERACT_BASELINE", "request_review": "true"},
    )
    assert up.status_code == 202, up.text
    payload = up.json()
    doc_id = payload["document"]["id"]
    job_id = payload["job"]["id"]
    assert payload["document"]["status"] == DocumentStatus.PENDING.value
    assert payload["job"]["status"] == JobStatus.QUEUED.value

    # Run Celery task synchronously with NO OCR mocks
    worker_result = execute_processing_job_sync(job_id)
    assert worker_result["status"] == "COMPLETED"

    doc_res = await anon_client.get(f"/api/v1/documents/{doc_id}", headers=headers)
    assert doc_res.status_code == 200, doc_res.text
    doc_data = doc_res.json()
    assert doc_data["status"] == DocumentStatus.COMPLETED.value
    assert len(doc_data["extractions"]) >= 1

    latest_ext = doc_data["extractions"][-1]
    assert latest_ext["extraction_method"] == "TESSERACT_BASELINE"
    assert latest_ext["is_placeholder"] is False
    assert latest_ext["raw_text"] and "SUNRISE" in latest_ext["raw_text"].upper()
    assert latest_ext["structured_data"]["total_amount"] is not None

    # Review workflow: draft edit -> reload -> approve -> history -> active learning export
    draft = await anon_client.patch(
        f"/api/v1/documents/{doc_id}/review",
        headers=headers,
        json={
            "corrected_data": {"vendor_name": "Sunrise Artisan Bakery Verified"},
            "reviewer_notes": "Verified against unmocked Tesseract OCR output.",
            "expected_version": 1,
        },
    )
    assert draft.status_code == 200, draft.text
    assert draft.json()["review_status"] == "IN_REVIEW"

    reloaded_review = await anon_client.get(f"/api/v1/documents/{doc_id}/review", headers=headers)
    assert reloaded_review.status_code == 200
    assert reloaded_review.json()["corrected_data"]["vendor_name"] == "Sunrise Artisan Bakery Verified"

    approved = await anon_client.post(
        f"/api/v1/documents/{doc_id}/review/approve",
        headers=headers,
        json={"expected_version": draft.json()["version"]},
    )
    assert approved.status_code == 200, approved.text
    assert approved.json()["review_status"] == "APPROVED"

    hist = await anon_client.get(f"/api/v1/documents/{doc_id}/review/history", headers=headers)
    assert hist.status_code == 200
    assert [ev["action"] for ev in hist.json()["audit_events"]][0] == "APPROVED"

    out_dir = tmp_path / "unmocked_feedback_export"
    summary = await export_feedback_from_session(
        db=test_db_session,
        output_dir=str(out_dir),
        upload_dir=str(storage_service.base_dir),
    )
    assert summary["exported_count"] == 1
    assert (out_dir / "feedback_export.jsonl").exists()


@pytest.mark.asyncio
async def test_end_to_end_authenticated_document_processing_review_and_feedback_export(
    anon_client: AsyncClient,
    test_users: dict,
    test_db_session: AsyncSession,
    tmp_path: Path,
):
    """Verifies the complete auth -> 202 queue -> worker persistence -> preview -> review -> feedback API flow."""
    headers = await _login_bearer_headers(anon_client, "reviewer@test.local", "StrongPass123!")
    receipt_bytes = _render_crisp_receipt_bytes()

    up = await anon_client.post(
        "/api/v1/documents",
        headers=headers,
        files={"file": ("portfolio_demo_receipt.png", receipt_bytes, "image/png")},
        data={"extraction_method": "TESSERACT_BASELINE", "request_review": "true"},
    )
    assert up.status_code == 202, up.text
    payload = up.json()
    doc_id = payload["document"]["id"]
    job_id = payload["job"]["id"]
    assert payload["document"]["status"] == "PENDING"
    assert payload["job"]["status"] == "QUEUED"

    simulated_output = ExtractionOutput(
        extraction_method="TESSERACT_BASELINE",
        is_placeholder=False,
        structured_data={
            "vendor_name": "SUNRISE ARTISAN BAKERY",
            "vendor_address": "742 Evergreen Terrace, Austin, TX",
            "document_date": "2026-10-18",
            "total_amount": "25.92",
            "currency": "USD",
            "tax_amount": "1.92",
            "line_items": [],
        },
        confidence=0.72,
        field_confidences={"vendor_name": 0.78, "total_amount": 0.91},
        raw_text="SUNRISE ARTISAN BAKERY TOTAL USD 25.92",
        validation_errors=[],
        metadata={
            "ocr_words": ["SUNRISE", "ARTISAN", "BAKERY", "2026-10-18", "25.92"],
            "ocr_boxes": [
                [70, 70, 260, 110],
                [270, 70, 440, 110],
                [450, 70, 620, 110],
                [70, 210, 250, 240],
                [480, 560, 620, 600],
            ],
            "image_width": 900,
            "image_height": 1100,
        },
    )

    with patch("backend.app.worker.tasks.TesseractBaselineExtractor.extract", return_value=simulated_output):
        worker_res = execute_processing_job_sync(job_id)
    assert worker_res["status"] == "COMPLETED"

    # Fetch document & verify original extraction is persisted while pending review
    doc_res = await anon_client.get(f"/api/v1/documents/{doc_id}", headers=headers)
    assert doc_res.status_code == 200
    doc_body = doc_res.json()
    assert doc_body["status"] == "COMPLETED"
    assert doc_body["review_status"] == "PENDING_REVIEW"

    # Preview binary streams cleanly with nosniff header
    preview_res = await anon_client.get(f"/api/v1/documents/{doc_id}/file", headers=headers)
    assert preview_res.status_code == 200
    assert preview_res.headers.get("x-content-type-options") == "nosniff"
    assert preview_res.content.startswith(b"\x89PNG\r\n\x1a\n")

    # Save draft correction -> approve -> verify immutable original vs corrected -> audit -> export metrics
    draft_res = await anon_client.patch(
        f"/api/v1/documents/{doc_id}/review",
        headers=headers,
        json={
            "corrected_data": {"vendor_name": "Sunrise Artisan Bakery LLC"},
            "reviewer_notes": "Normalized legal suffix.",
            "expected_version": 1,
        },
    )
    assert draft_res.status_code == 200
    assert draft_res.json()["review_status"] == "IN_REVIEW"

    approve_res = await anon_client.post(
        f"/api/v1/documents/{doc_id}/review/approve",
        headers=headers,
        json={"expected_version": draft_res.json()["version"]},
    )
    assert approve_res.status_code == 200
    assert approve_res.json()["review_status"] == "APPROVED"

    final_doc = (await anon_client.get(f"/api/v1/documents/{doc_id}", headers=headers)).json()
    final_review = (await anon_client.get(f"/api/v1/documents/{doc_id}/review", headers=headers)).json()
    assert final_review["original_data"]["vendor_name"] == "SUNRISE ARTISAN BAKERY"
    assert final_review["corrected_data"]["vendor_name"] == "Sunrise Artisan Bakery LLC"
    assert final_doc["extractions"][-1]["structured_data"]["vendor_name"] == "SUNRISE ARTISAN BAKERY"

    hist_res = await anon_client.get(f"/api/v1/documents/{doc_id}/review/history", headers=headers)
    assert hist_res.status_code == 200
    corr_fields = {fc["field_name"]: fc for fc in hist_res.json()["field_corrections"]}
    assert corr_fields["vendor_name"]["original_value"] == "SUNRISE ARTISAN BAKERY"
    assert corr_fields["vendor_name"]["corrected_value"] == "Sunrise Artisan Bakery LLC"

    out_dir = tmp_path / "feedback_out"
    export_summary = await export_feedback_from_session(
        db=test_db_session,
        output_dir=str(out_dir),
        upload_dir=str(storage_service.base_dir),
    )
    assert export_summary["exported_count"] == 1
    jsonl_path = out_dir / "feedback_export.jsonl"
    assert jsonl_path.exists()
    record = json.loads(jsonl_path.read_text(encoding="utf-8").strip().splitlines()[0])
    assert record["corrected_values"]["vendor_name"] == "Sunrise Artisan Bakery LLC"
