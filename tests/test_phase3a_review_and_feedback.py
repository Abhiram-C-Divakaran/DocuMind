import hashlib
import json
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.exceptions import StorageError
from backend.app.models.document import DocumentStatus
from backend.app.models.extraction import ExtractionResult
from backend.app.models.review import ReviewStatus
from backend.app.services.storage import storage_service
from ml.base import ExtractionOutput
from ml.feedback.export_annotations import export_feedback_from_session
from tests.conftest import MINIMAL_PDF_BYTES, MINIMAL_PNG_BYTES


def _make_output(
    *,
    structured_data: dict,
    confidence: float = 0.7,
    field_confidences: dict | None = None,
    raw_text: str = "SAMPLE",
    validation_errors: list[str] | None = None,
    metadata: dict | None = None,
) -> ExtractionOutput:
    return ExtractionOutput(
        extraction_method="TESSERACT_BASELINE",
        is_placeholder=False,
        structured_data=structured_data,
        confidence=confidence,
        field_confidences=field_confidences if field_confidences is not None else {},
        raw_text=raw_text,
        validation_errors=validation_errors if validation_errors is not None else [],
        metadata=metadata if metadata is not None else {},
    )


async def _upload_with_mocked_ocr(
    client: AsyncClient,
    *,
    filename: str = "receipt.png",
    file_bytes: bytes = MINIMAL_PNG_BYTES,
    content_type: str = "image/png",
    request_review: bool = False,
    output: ExtractionOutput,
) -> dict:
    with patch(
        "backend.app.services.document_service.TesseractBaselineExtractor.extract",
        return_value=output,
    ):
        data = {"extraction_method": "TESSERACT_BASELINE"}
        if request_review:
            data["request_review"] = "true"
        res = await client.post(
            "/api/v1/documents",
            files={"file": (filename, file_bytes, content_type)},
            data=data,
        )
    assert res.status_code == 201, res.text
    return res.json()["document"]


@pytest.mark.asyncio
async def test_review_eligibility_triggers_and_high_confidence_bypass(client: AsyncClient):
    # 1. Low heuristic confidence -> COMPLETED extraction, PENDING_REVIEW review state
    low_conf_doc = await _upload_with_mocked_ocr(
        client,
        filename="low_conf.png",
        output=_make_output(
            structured_data={
                "vendor_name": "Cafe Nero",
                "vendor_address": "10 High St",
                "document_date": "2024-03-10",
                "total_amount": "18.50",
                "currency": "USD",
                "tax_amount": "1.50",
                "line_items": [],
            },
            confidence=0.52,
            field_confidences={"vendor_name": 0.55, "total_amount": 0.50},
            raw_text="CAFE NERO\nTOTAL 18.50",
        ),
    )
    assert low_conf_doc["status"] == DocumentStatus.COMPLETED.value
    assert low_conf_doc["review_status"] == ReviewStatus.PENDING_REVIEW.value
    assert low_conf_doc["needs_review"] is True
    assert any("low_heuristic_confidence" in r for r in low_conf_doc["review_reasons"])

    # 2. High confidence + complete valid fields -> NOT_REQUIRED
    high_conf_doc = await _upload_with_mocked_ocr(
        client,
        filename="clean_receipt.png",
        output=_make_output(
            structured_data={
                "vendor_name": "Clean Mart",
                "vendor_address": "99 Market Ave",
                "document_date": "2024-04-01",
                "total_amount": "45.00",
                "currency": "USD",
                "tax_amount": "3.00",
                "line_items": [],
            },
            confidence=0.92,
            field_confidences={"vendor_name": 0.94, "document_date": 0.90, "total_amount": 0.92},
            raw_text="CLEAN MART\nTOTAL 45.00",
        ),
    )
    assert high_conf_doc["status"] == DocumentStatus.COMPLETED.value
    assert high_conf_doc["review_status"] == ReviewStatus.NOT_REQUIRED.value
    assert high_conf_doc["needs_review"] is False

    # 3. High confidence + explicit request_review=True -> PENDING_REVIEW
    explicit_doc = await _upload_with_mocked_ocr(
        client,
        filename="voluntary_review.png",
        request_review=True,
        output=_make_output(
            structured_data={
                "vendor_name": "Clean Mart",
                "vendor_address": "99 Market Ave",
                "document_date": "2024-04-01",
                "total_amount": "45.00",
                "currency": "USD",
                "tax_amount": "3.00",
                "line_items": [],
            },
            confidence=0.95,
            field_confidences={"vendor_name": 0.95},
            raw_text="CLEAN MART\nTOTAL 45.00",
        ),
    )
    assert explicit_doc["review_status"] == ReviewStatus.PENDING_REVIEW.value
    assert explicit_doc["needs_review"] is True

    # 4. High confidence with missing critical field & ambiguous date warning -> PENDING_REVIEW
    missing_field_doc = await _upload_with_mocked_ocr(
        client,
        filename="missing_date.png",
        output=_make_output(
            structured_data={
                "vendor_name": "Bookshop",
                "vendor_address": "5 Elm Rd",
                "document_date": None,
                "total_amount": "12.00",
                "currency": "USD",
                "tax_amount": "1.00",
                "line_items": [],
            },
            confidence=0.88,
            field_confidences={"vendor_name": 0.90},
            raw_text="BOOKSHOP 12.00",
            validation_errors=["Date ambiguity detected across numeric tokens"],
        ),
    )
    assert missing_field_doc["review_status"] == ReviewStatus.PENDING_REVIEW.value
    assert missing_field_doc["needs_review"] is True


@pytest.mark.asyncio
async def test_draft_persistence_original_immutability_and_audit_history(
    client: AsyncClient,
    test_db_session: AsyncSession,
):
    doc = await _upload_with_mocked_ocr(
        client,
        filename="coffee_edit.png",
        output=_make_output(
            structured_data={
                "vendor_name": "C0ffee Sh0p",
                "vendor_address": "12 Main St",
                "document_date": "2024-05-02",
                "total_amount": "14.20",
                "currency": "USD",
                "tax_amount": "1.20",
                "line_items": [],
            },
            confidence=0.61,
            field_confidences={"vendor_name": 0.58, "total_amount": 0.65},
            raw_text="Coffee Shop\n12 Main St\n2024-05-02\nTotal 14.50",
        ),
    )
    doc_id = doc["id"]

    # Fetch review workspace state
    get_res = await client.get(f"/api/v1/documents/{doc_id}/review")
    assert get_res.status_code == 200
    review_state = get_res.json()
    assert review_state["review_status"] == "PENDING_REVIEW"
    assert review_state["version"] == 1
    assert review_state["original_data"]["vendor_name"] == "C0ffee Sh0p"

    # Save draft correction -> transitions PENDING_REVIEW -> IN_REVIEW
    patch_res = await client.patch(
        f"/api/v1/documents/{doc_id}/review",
        json={
            "corrected_data": {
                "vendor_name": "Coffee Shop",
                "vendor_address": "12 Main St",
                "document_date": "2024-05-02",
                "total_amount": "14.50",
                "currency": "USD",
                "tax_amount": "1.20",
                "line_items": [],
            },
            "reviewer_notes": "Fixed OCR typo in vendor name and total.",
            "expected_version": review_state["version"],
        },
    )
    assert patch_res.status_code == 200, patch_res.text
    draft = patch_res.json()
    assert draft["review_status"] == "IN_REVIEW"
    assert draft["version"] == 2
    assert set(draft["changed_fields"]) == {"vendor_name", "total_amount"}
    assert draft["corrected_data"]["vendor_name"] == "Coffee Shop"
    assert draft["corrected_data"]["total_amount"] == "14.50"
    assert draft["original_data"]["vendor_name"] == "C0ffee Sh0p"

    # Reload from API & verify raw ExtractionResult in DB was NOT overwritten
    reloaded = (await client.get(f"/api/v1/documents/{doc_id}/review")).json()
    assert reloaded["corrected_data"]["vendor_name"] == "Coffee Shop"
    assert reloaded["original_data"]["vendor_name"] == "C0ffee Sh0p"

    ext_row = (
        await test_db_session.execute(
            select(ExtractionResult).where(ExtractionResult.document_id == doc_id)
        )
    ).scalar_one()
    assert ext_row.structured_data["vendor_name"] == "C0ffee Sh0p"
    assert ext_row.structured_data["total_amount"] == "14.20"

    # Approve review -> IN_REVIEW -> APPROVED
    approve_res = await client.post(
        f"/api/v1/documents/{doc_id}/review/approve",
        json={"expected_version": draft["version"], "reviewer_notes": "Verified against scan."},
    )
    assert approve_res.status_code == 200
    approved = approve_res.json()
    assert approved["review_status"] == "APPROVED"
    assert approved["approved_at"] is not None
    assert approved["version"] == 3

    # Verify history captures FieldCorrection diffs and chronological AuditEvents
    hist_res = await client.get(f"/api/v1/documents/{doc_id}/review/history")
    assert hist_res.status_code == 200
    hist = hist_res.json()
    corr_fields = {fc["field_name"]: fc for fc in hist["field_corrections"]}
    assert "vendor_name" in corr_fields
    assert corr_fields["vendor_name"]["original_value"] == "C0ffee Sh0p"
    assert corr_fields["vendor_name"]["corrected_value"] == "Coffee Shop"
    actions = [ev["action"] for ev in hist["audit_events"]]
    assert actions == ["APPROVED", "REVIEW_STARTED", "QUEUE_FLAGGED"]


@pytest.mark.asyncio
async def test_state_machine_conflicts_idempotency_and_validation_errors(client: AsyncClient):
    doc = await _upload_with_mocked_ocr(
        client,
        filename="state_machine.png",
        output=_make_output(
            structured_data={
                "vendor_name": "Bakery",
                "document_date": "2024-06-01",
                "total_amount": "20.00",
                "currency": "USD",
                "tax_amount": "2.00",
            },
            confidence=0.60,
            raw_text="BAKERY 20.00",
        ),
    )
    doc_id = doc["id"]

    # Reject invalid date, currency, tax > total, and extra payload field
    bad_date = await client.patch(
        f"/api/v1/documents/{doc_id}/review",
        json={"corrected_data": {"document_date": "01/06/2024"}},
    )
    assert bad_date.status_code == 422

    bad_tax = await client.patch(
        f"/api/v1/documents/{doc_id}/review",
        json={
            "corrected_data": {
                "total_amount": "10.00",
                "tax_amount": "15.00",
            }
        },
    )
    assert bad_tax.status_code == 422

    bad_currency = await client.patch(
        f"/api/v1/documents/{doc_id}/review",
        json={"corrected_data": {"currency": "US"}},
    )
    assert bad_currency.status_code == 422

    extra_field = await client.patch(
        f"/api/v1/documents/{doc_id}/review",
        json={"corrected_data": {"vendor_name": "Bakery", "unexpected": 123}},
    )
    assert extra_field.status_code == 422

    # Stale expected_version -> 409 Conflict
    stale_res = await client.patch(
        f"/api/v1/documents/{doc_id}/review",
        json={
            "corrected_data": {"vendor_name": "Bakery Plus"},
            "expected_version": 99,
        },
    )
    assert stale_res.status_code == 409

    # Approve directly from PENDING_REVIEW
    approve1 = await client.post(
        f"/api/v1/documents/{doc_id}/review/approve",
        json={"expected_version": 1},
    )
    assert approve1.status_code == 200
    assert approve1.json()["review_status"] == "APPROVED"

    # Idempotent repeated approve without edits returns 200 without incrementing version
    approve2 = await client.post(
        f"/api/v1/documents/{doc_id}/review/approve",
        json={},
    )
    assert approve2.status_code == 200
    assert approve2.json()["version"] == approve1.json()["version"]

    # Invalid direct transition APPROVED -> REJECTED blocked with 409 Conflict
    invalid_reject = await client.post(
        f"/api/v1/documents/{doc_id}/review/reject",
        json={},
    )
    assert invalid_reject.status_code == 409

    # Reopen APPROVED -> IN_REVIEW, then REJECT
    reopen_res = await client.post(
        f"/api/v1/documents/{doc_id}/review/reopen",
        json={"reviewer_notes": "Need to reject blurry crop"},
    )
    assert reopen_res.status_code == 200
    assert reopen_res.json()["review_status"] == "IN_REVIEW"

    reject_res = await client.post(
        f"/api/v1/documents/{doc_id}/review/reject",
        json={"reviewer_notes": "Unreadable receipt stamp"},
    )
    assert reject_res.status_code == 200
    assert reject_res.json()["review_status"] == "REJECTED"

    # Unknown doc returns 404
    missing_res = await client.get("/api/v1/documents/00000000-0000-0000-0000-000000000000/review")
    assert missing_res.status_code == 404


@pytest.mark.asyncio
async def test_review_queue_filters_sorting_and_metrics(client: AsyncClient):
    doc1 = await _upload_with_mocked_ocr(
        client,
        filename="alpha_low.png",
        output=_make_output(
            structured_data={"vendor_name": None, "total_amount": "10.00", "document_date": "2024-03-01"},
            confidence=0.40,
            raw_text="10.00",
            validation_errors=["Missing vendor"],
        ),
    )
    doc2 = await _upload_with_mocked_ocr(
        client,
        filename="beta_mid.png",
        output=_make_output(
            structured_data={"vendor_name": "Beta Store", "total_amount": "25.00", "document_date": "2024-03-02"},
            confidence=0.65,
            raw_text="Beta Store 25.00",
        ),
    )

    # Approve doc2 with 1 edited field
    await client.post(
        f"/api/v1/documents/{doc2['id']}/review/approve",
        json={
            "corrected_data": {
                "vendor_name": "Beta Superstore",
                "total_amount": "25.00",
                "document_date": "2024-03-02",
            }
        },
    )

    # Filter NEEDS_REVIEW -> returns only doc1
    q_needs = (await client.get("/api/v1/reviews?review_status=NEEDS_REVIEW")).json()
    assert q_needs["total"] == 1
    assert q_needs["items"][0]["document_id"] == doc1["id"]

    # Filter missing_field=vendor_name
    q_missing = (await client.get("/api/v1/reviews?review_status=ALL&missing_field=vendor_name")).json()
    assert q_missing["total"] == 1
    assert q_missing["items"][0]["document_id"] == doc1["id"]

    # Sort confidence_desc on ALL
    q_desc = (await client.get("/api/v1/reviews?review_status=ALL&sort_by=confidence_desc")).json()
    assert [it["document_id"] for it in q_desc["items"]] == [doc2["id"], doc1["id"]]

    # Review metrics
    metrics = (await client.get("/api/v1/reviews/metrics")).json()
    assert metrics["total_documents"] == 2
    assert metrics["awaiting_review"] == 1
    assert metrics["approved"] == 1
    assert metrics["rejected"] == 0
    assert metrics["eligible_for_feedback_export"] == 1
    assert metrics["avg_corrected_fields"] == 1.0


@pytest.mark.asyncio
async def test_controlled_preview_endpoints_and_security(client: AsyncClient):
    pdf_doc = await _upload_with_mocked_ocr(
        client,
        filename="../evil_path_attempt_invoice.pdf",
        file_bytes=MINIMAL_PDF_BYTES,
        content_type="application/pdf",
        output=_make_output(
            structured_data={"vendor_name": "PDF Co", "total_amount": "30.00", "document_date": "2024-03-11"},
            confidence=0.80,
            raw_text="PDF Co 30.00",
        ),
    )
    doc_id = pdf_doc["id"]

    meta_res = await client.get(f"/api/v1/documents/{doc_id}/preview/meta")
    assert meta_res.status_code == 200
    meta = meta_res.json()
    assert meta["is_pdf"] is True
    assert meta["total_pages"] == 1

    # Render page 1 safely to PNG with security headers and sanitized filename
    page_res = await client.get(f"/api/v1/documents/{doc_id}/file?page=1")
    assert page_res.status_code == 200
    assert page_res.headers["content-type"] == "image/png"
    assert page_res.headers["x-content-type-options"] == "nosniff"
    assert "inline" in page_res.headers["content-disposition"]
    assert ".." not in page_res.headers["content-disposition"]
    assert page_res.content.startswith(b"\x89PNG\r\n\x1a\n")

    # Requesting non-existent PDF page fails cleanly with 400 Bad Request
    bad_page = await client.get(f"/api/v1/documents/{doc_id}/file?page=9")
    assert bad_page.status_code == 400

    # Direct storage path traversal check raises StorageError
    with pytest.raises(StorageError):
        storage_service.get_file_path("../secret.txt")
    with pytest.raises(StorageError):
        storage_service.get_file_path("subdir/file.png")


@pytest.mark.asyncio
async def test_feedback_export_filters_dedup_contamination_and_alignment(
    client: AsyncClient,
    test_db_session: AsyncSession,
    tmp_path: Path,
):
    def make_png_bytes(rgb: tuple[int, int, int]) -> bytes:
        buf = BytesIO()
        Image.new("RGB", (32, 32), color=rgb).save(buf, format="PNG")
        return buf.getvalue()

    img1 = make_png_bytes((255, 0, 0))
    img2 = make_png_bytes((0, 255, 0))
    img3 = make_png_bytes((0, 0, 255))
    img4 = make_png_bytes((128, 128, 0))

    # 1. Approved WITH matched OCR token bounding boxes -> ALIGNED_CONFIDENT
    doc_aligned = await _upload_with_mocked_ocr(
        client,
        filename="aligned.png",
        file_bytes=img1,
        output=_make_output(
            structured_data={
                "vendor_name": "ACME MART",
                "vendor_address": "10 OAK RD",
                "document_date": "2024-05-01",
                "total_amount": "55.00",
                "currency": "USD",
                "tax_amount": "5.00",
            },
            confidence=0.70,
            raw_text="ACME MART 10 OAK RD 2024-05-01 55.00",
            metadata={
                "ocr_tokens": [
                    {"text": "ACME", "norm_bbox": [10, 10, 100, 30], "page_num": 1},
                    {"text": "MART", "norm_bbox": [110, 10, 200, 30], "page_num": 1},
                    {"text": "10", "norm_bbox": [10, 40, 40, 60], "page_num": 1},
                    {"text": "OAK", "norm_bbox": [45, 40, 90, 60], "page_num": 1},
                    {"text": "RD", "norm_bbox": [95, 40, 130, 60], "page_num": 1},
                    {"text": "2024-05-01", "norm_bbox": [10, 70, 140, 90], "page_num": 1},
                    {"text": "55.00", "norm_bbox": [200, 100, 280, 120], "page_num": 1},
                ]
            },
        ),
    )
    await client.post(f"/api/v1/documents/{doc_aligned['id']}/review/approve", json={})

    # 2. Approved WITHOUT OCR token boxes or unmatched edit -> NEEDS_MANUAL_ANNOTATION
    doc_unaligned = await _upload_with_mocked_ocr(
        client,
        filename="unaligned.png",
        file_bytes=img2,
        output=_make_output(
            structured_data={
                "vendor_name": "Blurred Store",
                "document_date": "2024-05-02",
                "total_amount": "19.99",
            },
            confidence=0.60,
            raw_text="Blrr Str",
        ),
    )
    await client.post(
        f"/api/v1/documents/{doc_unaligned['id']}/review/approve",
        json={
            "corrected_data": {
                "vendor_name": "Green Valley Books",
                "document_date": "2024-05-02",
                "total_amount": "19.99",
            }
        },
    )

    # 3. Duplicate image hash of doc_aligned -> excluded during export
    doc_dup = await _upload_with_mocked_ocr(
        client,
        filename="duplicate_aligned.png",
        file_bytes=img1,
        output=_make_output(
            structured_data={"vendor_name": "ACME MART", "total_amount": "55.00", "document_date": "2024-05-01"},
            confidence=0.62,
            raw_text="ACME MART 55.00",
        ),
    )
    await client.post(f"/api/v1/documents/{doc_dup['id']}/review/approve", json={})

    # 4. Rejected document -> MUST be excluded
    doc_rejected = await _upload_with_mocked_ocr(
        client,
        filename="rejected.png",
        file_bytes=img3,
        output=_make_output(
            structured_data={"vendor_name": "Bad", "total_amount": "9.00", "document_date": "2024-05-03"},
            confidence=0.50,
            raw_text="Bad 9.00",
        ),
    )
    await client.post(
        f"/api/v1/documents/{doc_rejected['id']}/review/reject",
        json={"reviewer_notes": "Invalid scan"},
    )

    # 5. Held-out test split contaminated image -> excluded
    contam_sha = hashlib.sha256(img4).hexdigest()
    split_file = tmp_path / "heldout_test.jsonl"
    split_file.write_text(
        json.dumps({"image_sha256": contam_sha, "words": ["CONTAM"], "ground_truth": {}}) + "\n",
        encoding="utf-8",
    )
    doc_contam = await _upload_with_mocked_ocr(
        client,
        filename="contam.png",
        file_bytes=img4,
        output=_make_output(
            structured_data={"vendor_name": "Contam Co", "total_amount": "33.00", "document_date": "2024-05-04"},
            confidence=0.60,
            raw_text="Contam Co 33.00",
        ),
    )
    await client.post(f"/api/v1/documents/{doc_contam['id']}/review/approve", json={})

    # Export feedback from async test DB session using client's temp storage_service.base_dir
    out_dir = tmp_path / "feedback_out"
    summary = await export_feedback_from_session(
        db=test_db_session,
        output_dir=str(out_dir),
        upload_dir=str(storage_service.base_dir),
        reference_splits=[str(split_file)],
    )

    assert summary["automatic_retraining_launched"] is False
    assert summary["approved_loaded"] == 4
    assert summary["exported_count"] == 2
    assert summary["excluded_by_review_status"].get("rejected") == 1
    assert summary["alignment_breakdown"]["aligned_confident"] == 1
    assert summary["alignment_breakdown"]["needs_manual_annotation"] == 1

    reasons = {ex["reason"] for ex in summary["excluded_by_validation_or_contamination"]}
    assert "duplicate_document_or_image_hash" in reasons
    assert "held_out_split_contamination" in reasons

    jsonl_lines = (out_dir / "feedback_export.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(jsonl_lines) == 2
    records_by_doc = {json.loads(line)["document_id"]: json.loads(line) for line in jsonl_lines}
    assert records_by_doc[doc_aligned["id"]]["quality"]["alignment_status"] == "ALIGNED_CONFIDENT"
    assert records_by_doc[doc_unaligned["id"]]["quality"]["alignment_status"] == "NEEDS_MANUAL_ANNOTATION"
    assert records_by_doc[doc_unaligned["id"]]["corrected_values"]["vendor_name"] == "Green Valley Books"
