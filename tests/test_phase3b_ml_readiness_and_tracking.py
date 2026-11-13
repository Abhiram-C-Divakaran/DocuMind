import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image, ImageDraw
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.app.config import settings
from backend.app.database import Base, get_db
from backend.app.main import app
from backend.app.models.document import DocumentStatus
from backend.app.models.review import ReviewStatus
from backend.app.services.document_service import get_extractor
from ml.base import ExtractionOutput
from ml.data.annotation_alignment import ID2LABEL
from ml.data.prepare_dataset import prepare_sroie_dataset
from ml.data.sroie_loader import SROIELoader
from ml.feedback.export_annotations import export_feedback_from_session
from ml.evaluation.compare_models import compare_models
from ml.evaluation.metrics import (
    compute_checkpoint_disk_size_mb,
    compute_latency_summary,
    compute_seqeval_metrics,
)
import ml.extractors.layoutlmv3_extractor as lm_mod
from ml.extractors.layoutlmv3_extractor import (
    LayoutLMv3Extractor,
    clear_layoutlmv3_cache,
    get_cached_layoutlmv3_extractor,
    sanitize_checkpoint_identity,
    validate_checkpoint_directory,
)
from ml.training.experiment_tracker import ExperimentTracker


def _create_sample_png(path: Path, text: str = "99 SPEED MART TOTAL 15.50") -> bytes:
    img = Image.new("RGB", (320, 480), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), text, fill=(0, 0, 0))
    img.save(path, format="PNG")
    return path.read_bytes()


def _create_valid_checkpoint_files(ckpt_dir: Path, metadata: dict | None = None) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "model_type": "layoutlmv3",
        "id2label": {str(k): v for k, v in ID2LABEL.items()},
        "label2id": {v: k for k, v in ID2LABEL.items()},
    }
    (ckpt_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (ckpt_dir / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 512}), encoding="utf-8")
    (ckpt_dir / "model.safetensors").write_bytes(b"mock-safetensors-payload-12345678")
    if metadata is not None:
        (ckpt_dir / "checkpoint_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_sroie_arvindrajan_contiguous_and_interleaved_bio_tags(tmp_path: Path):
    loader = SROIELoader(cache_dir=str(tmp_path / "cache"))
    row = {
        "image": Image.new("RGB", (500, 1000), color=(255, 255, 255)),
        "ocr": [
            {"text": "ACME MART", "box": [[10, 10], [200, 10], [200, 40], [10, 40]], "label": "company"},
            {"text": "HEADER", "box": [[10, 50], [120, 50], [120, 70], [10, 70]], "label": "other"},
            {"text": "SDN BHD", "box": [[10, 80], [150, 80], [150, 100], [10, 100]], "label": "company"},
            {"text": "NO 12 JALAN", "box": [[10, 120], [200, 120], [200, 140], [10, 140]], "label": "address"},
            {"text": "KUALA LUMPUR", "box": [[10, 150], [200, 150], [200, 170], [10, 170]], "label": "address"},
            {"text": "18/05/2020", "box": [[10, 200], [150, 200], [150, 220], [10, 220]], "label": "date"},
            {"text": "45.00", "box": [[200, 300], [280, 300], [280, 320], [200, 320]], "label": "total"},
        ],
    }

    parsed = loader._parse_arvindrajan_row(row, "r001", "train", tmp_path / "imgs")
    assert parsed is not None
    assert parsed.pre_assigned_labels == [
        "B-COMPANY",
        "I-COMPANY",
        "O",
        "B-COMPANY",  # Restarts with B- after intervening 'O' segment
        "I-COMPANY",
        "B-ADDRESS",
        "I-ADDRESS",
        "I-ADDRESS",
        "I-ADDRESS",  # Contiguous multi-line address remains single span
        "I-ADDRESS",
        "B-DATE",
        "B-TOTAL",
    ]


def test_experiment_tracker_records_lineage_and_failure(tmp_path: Path):
    tracker = ExperimentTracker(experiment_dir=str(tmp_path / "exp"), enable_mlflow=True)
    exp_id = tracker.start_run(
        model_architecture="microsoft/layoutlmv3-base",
        label_list=list(ID2LABEL.values()),
        hyperparameters={"learning_rate": 2e-5, "num_epochs": 3, "batch_size": 4},
        dataset_info={"source": "sroie", "train_samples": 10, "validation_samples": 2},
        seed=42,
        resolved_device="cpu",
    )
    tracker.log_epoch(
        {
            "epoch": 1,
            "global_step": 3,
            "train_loss": 1.25,
            "val_loss": 0.98,
            "val_f1": 0.61,
            "val_metrics": {"overall_f1": 0.61, "macro_f1": 0.58},
        }
    )
    run_file = tracker.finish_run(
        status="COMPLETED",
        best_checkpoint={"path": "checkpoints/best", "best_val_f1": 0.61, "best_epoch": 1},
    )

    data = json.loads(run_file.read_text(encoding="utf-8"))
    assert data["experiment_id"] == exp_id
    assert data["status"] == "COMPLETED"
    assert data["random_seed"] == 42
    assert data["environment"]["dependencies"]["transformers"] is not None
    assert len(data["epochs"]) == 1
    assert data["best_checkpoint"]["best_epoch"] == 1
    assert (tmp_path / "exp" / "latest_experiment.json").exists()

    fail_tracker = ExperimentTracker(experiment_dir=str(tmp_path / "exp_fail"), enable_mlflow=False)
    fail_tracker.start_run("microsoft/layoutlmv3-base", [], {}, {}, 42, "cpu")
    fail_file = fail_tracker.finish_run(status="FAILED", extra={"error": "OOM"})
    assert json.loads(fail_file.read_text(encoding="utf-8"))["status"] == "FAILED"


def test_metrics_macro_f1_latency_percentiles_and_checkpoint_size(tmp_path: Path):
    seq = compute_seqeval_metrics(
        predictions=[["B-COMPANY", "O", "B-DATE", "B-TOTAL"]],
        references=[["B-COMPANY", "O", "B-DATE", "O"]],
    )
    assert "macro_f1" in seq
    assert 0.0 <= seq["macro_f1"] <= 1.0

    lat = compute_latency_summary([10.0, 20.0, 30.0, 40.0, 100.0])
    assert lat["avg_ms"] == pytest.approx(40.0)
    assert lat["p50_ms"] == pytest.approx(30.0)
    assert lat["p95_ms"] >= lat["p50_ms"]
    assert lat["p99_ms"] >= lat["p95_ms"]

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "weights.bin").write_bytes(b"a" * (2 * 1024 * 1024))
    assert compute_checkpoint_disk_size_mb(ckpt) == pytest.approx(2.0)
    assert compute_checkpoint_disk_size_mb(tmp_path / "does_not_exist") is None


def test_compare_models_outputs_percentiles_size_and_error_analysis(tmp_path: Path):
    data_dir = tmp_path / "sroie_syn"
    prepare_sroie_dataset(
        output_dir=str(data_dir),
        seed=42,
        allow_synthetic=True,
        synthetic_count=8,
    )
    out_dir = tmp_path / "bench"

    def fake_lm_runner(receipt):
        # Deliberately underperform on all fields to test baseline_wins error analysis + honest comparison
        return (
            {
                "company": "WRONG VENDOR",
                "date": "WRONG DATE",
                "address": "WRONG ADDR",
                "total": "0.00",
            },
            {
                "ocr_latency_ms": 0.0,
                "feature_prep_latency_ms": 2.0,
                "model_inference_latency_ms": 8.0,
                "total_pipeline_latency_ms": 10.0,
            },
        )

    res = compare_models(
        checkpoint_dir=str(tmp_path / "unused_ckpt"),
        test_jsonl=str(data_dir / "test.jsonl"),
        output_dir=str(out_dir),
        allow_synthetic=True,
        layoutlm_runner=fake_lm_runner,
    )
    assert res["benchmark_status"] == "COMPLETE"
    assert "latency_percentiles_ms" in res["baseline"]
    assert "latency_percentiles_ms" in res["layoutlmv3"]
    assert "p95_ms" in res["layoutlmv3"]["latency_percentiles_ms"]
    assert "checkpoint_size_mb" in res["layoutlmv3"]
    md_text = (out_dir / "benchmark_comparison.md").read_text(encoding="utf-8")
    assert "Checkpoint Disk Size" in md_text
    assert "Error Analysis" in md_text


def test_layoutlmv3_weight_caching_and_isolation(tmp_path: Path):
    clear_layoutlmv3_cache()
    ckpt_dir = tmp_path / "valid_ckpt"
    _create_valid_checkpoint_files(
        ckpt_dir,
        metadata={"base_model": "microsoft/layoutlmv3-base", "best_val_f1": 0.88, "is_synthetic": False},
    )

    fake_tok = MagicMock()
    fake_proc = MagicMock(tokenizer=fake_tok, image_processor=MagicMock())
    fake_mdl = MagicMock()
    fake_mdl.to.return_value = fake_mdl

    fake_auto_proc = MagicMock()
    fake_auto_proc.from_pretrained.return_value = fake_proc
    fake_lm_cls = MagicMock()
    fake_lm_cls.from_pretrained.return_value = fake_mdl

    with patch.object(lm_mod, "HAS_TORCH", True), \
         patch.object(lm_mod, "torch", MagicMock()), \
         patch.object(lm_mod, "AutoProcessor", fake_auto_proc), \
         patch.object(lm_mod, "LayoutLMv3ForTokenClassification", fake_lm_cls):
        ext1 = get_cached_layoutlmv3_extractor(str(ckpt_dir), device="cpu")
        ext2 = get_cached_layoutlmv3_extractor(str(ckpt_dir), device="cpu")
        assert ext1 is ext2

        ext1.load_model()
        ext2.load_model()
        # Even constructing a fresh extractor pointing at same checkpoint reuses weight cache
        ext3 = LayoutLMv3Extractor(checkpoint_dir=str(ckpt_dir), device="cpu")
        ext3.load_model()

        assert fake_lm_cls.from_pretrained.call_count == 1
        assert fake_auto_proc.from_pretrained.call_count == 1

    clear_layoutlmv3_cache()

    # Invalid label map must fail validation without breaking TesseractBaselineExtractor
    bad_ckpt = tmp_path / "bad_ckpt"
    bad_ckpt.mkdir()
    (bad_ckpt / "config.json").write_text(
        json.dumps({"model_type": "layoutlmv3", "id2label": {"0": "O"}}),
        encoding="utf-8",
    )
    (bad_ckpt / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (bad_ckpt / "model.safetensors").write_bytes(b"weights")
    ok, reason = validate_checkpoint_directory(bad_ckpt)
    assert not ok
    assert "Label count mismatch" in reason

    tess_ext = get_extractor("TESSERACT_BASELINE")
    assert tess_ext.method_name == "TESSERACT_BASELINE"


@pytest.mark.asyncio
async def test_models_readiness_endpoint_and_path_sanitization(tmp_path: Path):
    clear_layoutlmv3_cache()
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'readiness.db'}"
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def override_get_db():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    try:
        missing_abs = tmp_path / "secret_host_folder" / "missing_ckpt"
        with patch.object(settings, "LAYOUTLMV3_CHECKPOINT_DIR", str(missing_abs)):
            clear_layoutlmv3_cache()
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.get("/api/v1/models/readiness")
            assert resp.status_code == 200
            payload = resp.json()
            assert payload["default_method"] == "TESSERACT_BASELINE"
            assert payload["layoutlmv3_ready"] is False
            assert payload["layoutlmv3_is_production_ready"] is False
            assert payload["checkpoint_identity"] == "missing_ckpt"
            # Must not expose parent host path like tmp_path / secret_host_folder
            assert "secret_host_folder" not in (payload["layoutlmv3_reason"] or "")

        # Now simulate valid non-synthetic checkpoint + available runtime
        valid_abs = tmp_path / "secret_host_folder" / "real_ckpt"
        _create_valid_checkpoint_files(
            valid_abs,
            metadata={
                "experiment_id": "exp-123",
                "base_model": "microsoft/layoutlmv3-base",
                "best_val_f1": 0.91,
                "is_synthetic": False,
            },
        )
        with patch.object(settings, "LAYOUTLMV3_CHECKPOINT_DIR", str(valid_abs)), \
             patch.object(lm_mod, "HAS_TORCH", True), \
             patch.object(lm_mod, "torch", MagicMock()):
            clear_layoutlmv3_cache()
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.get("/api/v1/models/readiness")
            assert resp.status_code == 200
            payload = resp.json()
            assert payload["layoutlmv3_ready"] is True
            assert payload["layoutlmv3_is_production_ready"] is True
            assert payload["checkpoint_identity"] == "real_ckpt"
            assert payload["checkpoint_metadata"]["best_val_f1"] == pytest.approx(0.91)

        # Synthetic checkpoint must NEVER claim production readiness
        syn_abs = tmp_path / "secret_host_folder" / "syn_ckpt"
        _create_valid_checkpoint_files(
            syn_abs,
            metadata={"base_model": "microsoft/layoutlmv3-base", "is_synthetic": True},
        )
        with patch.object(settings, "LAYOUTLMV3_CHECKPOINT_DIR", str(syn_abs)), \
             patch.object(lm_mod, "HAS_TORCH", True), \
             patch.object(lm_mod, "torch", MagicMock()):
            clear_layoutlmv3_cache()
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.get("/api/v1/models/readiness")
            assert resp.status_code == 200
            payload = resp.json()
            assert payload["layoutlmv3_ready"] is True
            assert payload["layoutlmv3_is_production_ready"] is False
    finally:
        app.dependency_overrides.clear()
        clear_layoutlmv3_cache()
        await engine.dispose()


@pytest.mark.asyncio
async def test_end_to_end_layoutlmv3_upload_review_and_feedback_export(tmp_path: Path):
    from backend.app.core.security import create_access_token, hash_password
    from backend.app.database import reset_sync_engine
    from backend.app.models.user import User, UserRole
    from backend.app.worker.tasks import execute_processing_job_sync

    clear_layoutlmv3_cache()
    db_file = tmp_path / "e2e_3b.db"
    db_url = f"sqlite+aiosqlite:///{db_file}"
    orig_sync_url = settings.SYNC_DATABASE_URL
    settings.SYNC_DATABASE_URL = f"sqlite:///{db_file}"
    reset_sync_engine()

    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def override_get_db():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    img_bytes = _create_sample_png(tmp_path / "receipt_e2e.png")

    async with session_factory() as db:
        user = User(
            email="e2e3b@test.local",
            full_name="QA E2E",
            password_hash=hash_password("StrongPass123!"),
            role=UserRole.REVIEWER.value,
            is_active=True,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        token, _ = create_access_token(subject=user.id, email=user.email, role=user.role)
    auth_headers = {"Authorization": f"Bearer {token}"}

    async def fake_lm_extract(self, file_path, **kwargs):
        return ExtractionOutput(
            extraction_method="LAYOUTLMV3",
            is_placeholder=False,
            structured_data={
                "vendor_name": "99 SPEED MART",
                "vendor_address": "JALAN ANGSA KL",
                "document_date": "2024-05-01",
                "total_amount": "15.50",
                "currency": "MYR",
                "tax_amount": None,
                "line_items": [],
            },
            raw_text="99 SPEED MART JALAN ANGSA KL TOTAL 15.50",
            confidence=0.62,  # Below 0.75 threshold -> routes into PENDING_REVIEW
            field_confidences={
                "vendor_name": 0.9,
                "vendor_address": 0.5,
                "document_date": 0.6,
                "total_amount": 0.48,
            },
            validation_errors=[],
            metadata={"model": "microsoft/layoutlmv3-base", "checkpoint": "best_ckpt"},
        )

    try:
        with patch("backend.app.services.document_service.settings.UPLOAD_DIR", str(tmp_path / "uploads")), \
             patch("backend.app.services.storage.storage_service.base_dir", tmp_path / "uploads"), \
             patch.object(LayoutLMv3Extractor, "validate_checkpoint", return_value=(True, "Valid")), \
             patch.object(LayoutLMv3Extractor, "extract", new=fake_lm_extract):
            (tmp_path / "uploads").mkdir(parents=True, exist_ok=True)
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", headers=auth_headers) as client:
                up = await client.post(
                    "/api/v1/documents",
                    files={"file": ("receipt_e2e.png", img_bytes, "image/png")},
                    data={"extraction_method": "LAYOUTLMV3"},
                )
                assert up.status_code == 202, up.text
                doc = up.json()["document"]
                job = up.json()["job"]
                doc_id = doc["id"]
                assert doc["status"] == DocumentStatus.PENDING.value

                execute_processing_job_sync(job["id"])

                doc_res = await client.get(f"/api/v1/documents/{doc_id}")
                assert doc_res.status_code == 200
                processed_doc = doc_res.json()
                assert processed_doc["status"] == DocumentStatus.COMPLETED.value
                assert processed_doc["review_status"] == ReviewStatus.PENDING_REVIEW.value
                assert processed_doc["extractions"][0]["extraction_method"] == "LAYOUTLMV3"

                # Reviewer edits and approves -> eligible for feedback export
                approve = await client.post(
                    f"/api/v1/documents/{doc_id}/review/approve",
                    json={
                        "corrected_data": {"total_amount": "16.50"},
                        "reviewer_notes": "Verified total line on receipt",
                        "reviewer_identifier": "qa-engineer",
                    },
                )
                assert approve.status_code == 200, approve.text
                assert approve.json()["review_status"] == ReviewStatus.APPROVED.value

            async with session_factory() as db:
                export_res = await export_feedback_from_session(
                    db=db,
                    output_dir=str(tmp_path / "feedback_out"),
                    upload_dir=str(tmp_path / "uploads"),
                )
            assert export_res["exported_count"] == 1
            jsonl_lines = [
                json.loads(line)
                for line in (tmp_path / "feedback_out" / "feedback_export.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            assert len(jsonl_lines) == 1
            assert jsonl_lines[0]["extraction_method"] == "LAYOUTLMV3"
            assert jsonl_lines[0]["corrected_values"]["total_amount"] == "16.50"
    finally:
        app.dependency_overrides.clear()
        clear_layoutlmv3_cache()
        reset_sync_engine()
        settings.SYNC_DATABASE_URL = orig_sync_url
        await engine.dispose()


def test_colab_notebook_structure_and_python_syntax():
    nb_path = Path("ml/training/DocuMind_LayoutLMv3_Training.ipynb")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 5
    full_nb_src = "\n".join("".join(c["source"]) for c in code_cells)
    assert "--smoke-test" in full_nb_src
    assert "ml.training.import_checkpoint" in full_nb_src
    for idx, cell in enumerate(code_cells):
        src = "".join(cell["source"])
        # Replace IPython shell/magic commands (including multi-line '\' continuations) cleanly before AST compile verification
        py_lines = []
        in_shell_continuation = False
        for ln in src.splitlines():
            stripped = ln.lstrip()
            if in_shell_continuation:
                in_shell_continuation = ln.rstrip().endswith("\\")
                continue
            if stripped.startswith("!") or stripped.startswith("%"):
                indent = ln[: len(ln) - len(stripped)]
                py_lines.append(f"{indent}pass")
                in_shell_continuation = ln.rstrip().endswith("\\")
            else:
                py_lines.append(ln)
        compile("\n".join(py_lines), f"cell_{idx}", "exec")


def test_parse_extracted_date_handles_sroie_timestamps_and_short_years():
    from datetime import date
    from ml.extractors.layoutlmv3_extractor import parse_extracted_date

    assert parse_extracted_date("25/12/2018 8:13:39 PM") == date(2018, 12, 25)
    assert parse_extracted_date("19/10/2018 20:49:59 #01") == date(2018, 10, 19)
    assert parse_extracted_date("12-01-19 21:13 SH01 ZK09") == date(2019, 1, 12)
    assert parse_extracted_date("2026-10-18") == date(2026, 10, 18)


def test_multiline_ocr_reconstruction_and_percentage_tax_filtering():
    from ml.data.annotation_alignment import LABEL2ID, AlignedReceipt
    from ml.evaluation.compare_models import create_ocr_result_from_receipt
    from ml.extraction.pipeline import ReceiptExtractionPipeline

    labels = [
        "B-COMPANY", "I-COMPANY",
        "O", "B-DATE",
        "O", "O", "O", "O",
        "O", "O", "O", "B-TOTAL",
    ]
    receipt = AlignedReceipt(
        receipt_id="r_multiline",
        words=[
            "SUNRISE", "BAKERY",
            "DATE:", "2026-10-18",
            "TAX", "(8%)", "USD", "1.92",
            "TOTAL", "PAID", "USD", "25.92",
        ],
        boxes=[
            [50, 50, 200, 80], [210, 50, 350, 80],
            [50, 150, 140, 180], [150, 150, 300, 180],
            [50, 400, 110, 430], [120, 400, 190, 430], [450, 400, 520, 430], [530, 400, 620, 430],
            [50, 460, 150, 490], [160, 460, 250, 490], [450, 460, 520, 490], [530, 460, 620, 490],
        ],
        labels=labels,
        ner_tags=[LABEL2ID[lbl] for lbl in labels],
        ground_truth={"company": "SUNRISE BAKERY", "date": "2026-10-18", "total": "25.92"},
        alignment_stats={"COMPANY": True, "DATE": True, "ADDRESS": False, "TOTAL": True},
        width=760,
        height=1020,
    )
    ocr_res = create_ocr_result_from_receipt(receipt)
    assert "\n" in ocr_res.full_text
    assert len(ocr_res.full_text.splitlines()) == 4

    extracted = ReceiptExtractionPipeline().process(ocr_res)
    assert extracted.vendor_name.value == "SUNRISE BAKERY"
    assert str(extracted.total_amount.value) == "25.92"
    assert str(extracted.tax_amount.value) == "1.92"


def test_import_checkpoint_uses_real_signatures_and_unpacks_colab_nested_zip(tmp_path: Path):
    import inspect
    import zipfile
    import ml.training.import_checkpoint as imp_mod
    from ml.training.dataset import prepare_layoutlmv3_inputs
    from ml.inference.decoder import decode_layoutlmv3_predictions

    # Build a Colab-style export archive where checkpoint resides at checkpoints/layoutlmv3-sroie-best/
    bundle_zip = tmp_path / "documind_layoutlmv3_best.zip"
    cfg = {
        "model_type": "layoutlmv3",
        "id2label": {str(k): v for k, v in ID2LABEL.items()},
        "label2id": {v: k for k, v in ID2LABEL.items()},
    }
    with zipfile.ZipFile(bundle_zip, "w") as zf:
        zf.writestr("checkpoints/layoutlmv3-sroie-best/config.json", json.dumps(cfg))
        zf.writestr("checkpoints/layoutlmv3-sroie-best/tokenizer_config.json", "{}")
        zf.writestr("checkpoints/layoutlmv3-sroie-best/tokenizer.json", "{}")
        zf.writestr("checkpoints/layoutlmv3-sroie-best/model.safetensors", b"real-weights-bytes")
        zf.writestr(
            "checkpoints/layoutlmv3-sroie-best/checkpoint_metadata.json",
            json.dumps({"is_synthetic": False, "smoke_test": False, "epoch": 5}),
        )
        zf.writestr("ml/experiments/latest_experiment.json", "{}")

    prep_sig = inspect.signature(prepare_layoutlmv3_inputs)
    dec_sig = inspect.signature(decode_layoutlmv3_predictions)

    def _checked_prep(*args, **kwargs):
        prep_sig.bind(*args, **kwargs)
        return {"word_ids": [None, 0, None]}

    def _checked_decode(*args, **kwargs):
        dec_sig.bind(*args, **kwargs)
        return {"fields": {}, "confidences": {}}

    fake_model = MagicMock()
    fake_model.return_value.logits.__getitem__.return_value.shape = (10, len(ID2LABEL))

    def _fake_load(self):
        self._tokenizer = MagicMock()
        self._image_processor = MagicMock()
        self._model = fake_model

    target_dir = tmp_path / "installed" / "layoutlmv3-sroie-best"
    with (
        patch.object(imp_mod, "HAS_TORCH", True),
        patch.object(imp_mod, "prepare_layoutlmv3_inputs", side_effect=_checked_prep),
        patch.object(imp_mod, "decode_layoutlmv3_predictions", side_effect=_checked_decode),
        patch.object(imp_mod.LayoutLMv3Extractor, "load_model", _fake_load),
        patch.dict("sys.modules", {"torch": MagicMock()}),
    ):
        report = imp_mod.validate_and_smoke_test_checkpoint(bundle_zip, target_dir=target_dir)

    assert report["compatible"] is True
    assert report["production_ready"] is True
    assert report["smoke_test_passed"] is True
    assert (target_dir / "config.json").exists()
    assert (target_dir / "model.safetensors").exists()
