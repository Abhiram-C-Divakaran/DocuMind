import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image, ImageDraw

from ml.base import ExtractionOutput
from ml.data.annotation_alignment import ID2LABEL, LABEL_LIST, align_receipt_annotations
from ml.data.prepare_dataset import (
    check_data_leakage,
    load_aligned_receipts,
    prepare_sroie_dataset,
)
from ml.data.sroie_loader import SROIELoader
from ml.evaluation.compare_models import compare_models, create_ocr_result_from_receipt
from ml.evaluation.evaluate import evaluate_checkpoint
from ml.extractors.layoutlmv3_extractor import validate_checkpoint_directory
from ml.training.config import LayoutLMv3TrainingConfig
from ml.training.dataset import prepare_layoutlmv3_inputs, preprocess_image_array
from ml.training.train_layoutlmv3 import load_training_splits


class DummyBatchEncoding(dict):
    def __init__(self, data, word_id_list):
        super().__init__(data)
        self._word_ids = word_id_list

    def word_ids(self, batch_index: int = 0):
        return self._word_ids


class DummyFastTokenizer:
    """Minimal fast tokenizer stand-in producing deterministic subwords and word_ids."""

    def __call__(
        self,
        words,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=16,
        return_attention_mask=True,
    ):
        input_ids = [0]
        word_ids = [None]
        for w_idx, w in enumerate(words):
            pieces = max(1, min(2, len(w) // 4 or 1))
            for p_idx in range(pieces):
                if len(input_ids) >= max_length - 1:
                    break
                input_ids.append(100 + w_idx * 10 + p_idx)
                word_ids.append(w_idx)
        input_ids.append(2)
        word_ids.append(None)

        attn = [1] * len(input_ids)
        while len(input_ids) < max_length:
            input_ids.append(1)
            attn.append(0)
            word_ids.append(None)

        return DummyBatchEncoding(
            {"input_ids": input_ids[:max_length], "attention_mask": attn[:max_length]},
            word_ids[:max_length],
        )


def _create_receipt_png(path: Path, text: str = "ACME MART TOTAL 15.00", color=(248, 248, 248)):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (400, 600), color=color)
    draw = ImageDraw.Draw(img)
    draw.text((30, 40), text, fill=(10, 10, 10))
    img.save(path, format="PNG")


def test_create_ocr_result_normalized_confidence_scale(tmp_path):
    receipt = align_receipt_annotations(
        receipt_id="r1",
        words=["ACME", "12.50"],
        boxes=[[10, 10, 50, 30], [60, 10, 100, 30]],
        ground_truth={"company": "ACME", "total": "12.50"},
    )
    ocr_res = create_ocr_result_from_receipt(receipt)
    assert ocr_res.avg_confidence == pytest.approx(0.95)
    assert all(0.0 <= t.confidence <= 1.0 for t in ocr_res.tokens)


def test_compare_models_missing_checkpoint_marks_not_evaluated(tmp_path):
    data_dir = tmp_path / "processed"
    prepare_sroie_dataset(
        output_dir=str(data_dir),
        seed=42,
        allow_synthetic=True,
        synthetic_count=10,
    )

    out_dir = tmp_path / "eval_out"
    res = compare_models(
        checkpoint_dir=str(tmp_path / "nonexistent_ckpt"),
        test_jsonl=str(data_dir / "test.jsonl"),
        output_dir=str(out_dir),
        allow_synthetic=True,
    )

    assert res["benchmark_status"] == "INCOMPLETE_MISSING_LAYOUTLMV3"
    assert res["is_synthetic"] is True
    assert res["baseline"]["status"] == "EVALUATED"
    assert res["layoutlmv3"]["status"] == "NOT_EVALUATED"
    assert res["layoutlmv3"]["macro_f1"] is None
    assert res["layoutlmv3"]["document_exact_match"] is None
    assert res["layoutlmv3"]["total_pipeline_latency_ms"] is None
    for f in ["company", "date", "address", "total"]:
        assert res["layoutlmv3"]["field_f1"][f] is None

    md_text = (out_dir / "benchmark_comparison.md").read_text(encoding="utf-8")
    assert "SYNTHETIC SMOKE TEST" in md_text
    assert "NOT_EVALUATED" in md_text


def test_compare_models_scores_both_when_layoutlm_runs(tmp_path):
    data_dir = tmp_path / "processed"
    prepare_sroie_dataset(
        output_dir=str(data_dir),
        seed=42,
        allow_synthetic=True,
        synthetic_count=10,
    )

    def perfect_runner(receipt):
        return dict(receipt.ground_truth), {
            "ocr_latency_ms": 0.0,
            "feature_prep_latency_ms": 2.5,
            "model_inference_latency_ms": 12.5,
            "total_pipeline_latency_ms": 15.0,
        }

    out_dir = tmp_path / "eval_complete"
    res = compare_models(
        checkpoint_dir=str(tmp_path / "mock_ckpt"),
        test_jsonl=str(data_dir / "test.jsonl"),
        output_dir=str(out_dir),
        allow_synthetic=True,
        layoutlm_runner=perfect_runner,
    )

    assert res["benchmark_status"] == "COMPLETE"
    assert res["baseline"]["status"] == "EVALUATED"
    assert res["layoutlmv3"]["status"] == "EVALUATED"
    assert res["layoutlmv3"]["macro_f1"] == pytest.approx(1.0)
    assert res["layoutlmv3"]["document_exact_match"] == pytest.approx(1.0)
    assert res["layoutlmv3"]["model_inference_latency_ms"] == pytest.approx(12.5)
    assert res["layoutlmv3"]["total_pipeline_latency_ms"] == pytest.approx(15.0)


def test_compare_and_eval_refuse_synthetic_without_flag(tmp_path):
    data_dir = tmp_path / "processed"
    prepare_sroie_dataset(
        output_dir=str(data_dir),
        seed=42,
        allow_synthetic=True,
        synthetic_count=8,
    )
    with pytest.raises(RuntimeError, match="synthetic"):
        compare_models(
            checkpoint_dir=str(tmp_path / "ckpt"),
            test_jsonl=str(data_dir / "test.jsonl"),
            output_dir=str(tmp_path / "out"),
            allow_synthetic=False,
        )
    with pytest.raises(RuntimeError, match="synthetic"):
        evaluate_checkpoint(
            checkpoint_dir=str(tmp_path / "ckpt"),
            test_jsonl=str(data_dir / "test.jsonl"),
            output_dir=str(tmp_path / "out"),
            allow_synthetic=False,
        )


def test_prepare_and_training_refuse_silent_synthetic_fallback(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare_sroie_dataset(
            local_dir=str(tmp_path / "missing_sroie"),
            output_dir=str(tmp_path / "out"),
            allow_synthetic=False,
        )

    cfg_missing = LayoutLMv3TrainingConfig(
        data_dir=str(tmp_path / "no_splits"),
        smoke_test=False,
    )
    with pytest.raises(FileNotFoundError, match="refuses to auto-generate synthetic"):
        load_training_splits(cfg_missing)

    synth_dir = tmp_path / "synth_splits"
    prepare_sroie_dataset(
        output_dir=str(synth_dir),
        allow_synthetic=True,
        synthetic_count=10,
    )
    cfg_real_on_synth = LayoutLMv3TrainingConfig(
        data_dir=str(synth_dir),
        smoke_test=False,
    )
    with pytest.raises(RuntimeError, match="synthetic records"):
        load_training_splits(cfg_real_on_synth)


def test_loader_excludes_corrupted_images_and_detects_hash_leakage(tmp_path):
    root = tmp_path / "local_sroie"
    (root / "img").mkdir(parents=True)
    (root / "box").mkdir(parents=True)
    (root / "entities").mkdir(parents=True)

    # 1. Valid receipt
    _create_receipt_png(root / "img" / "r_valid.png", "KOPITIAM")
    (root / "box" / "r_valid.txt").write_text(
        "10,10,200,10,200,40,10,40,KOPITIAM CAFE\n"
        "10,60,180,60,180,90,10,90,12.50\n",
        encoding="utf-8",
    )
    (root / "entities" / "r_valid.txt").write_text(
        json.dumps({"company": "KOPITIAM CAFE", "total": "12.50"}),
        encoding="utf-8",
    )

    # 2. Corrupted empty image
    (root / "img" / "r_corrupt.png").write_bytes(b"not-a-valid-png")
    (root / "box" / "r_corrupt.txt").write_text("10,10,50,10,50,30,10,30,TEST\n", encoding="utf-8")

    loader = SROIELoader(cache_dir=str(tmp_path / "cache"))
    loaded = loader.load_from_local_directory(root)
    assert len(loaded) == 1
    assert loaded[0].receipt_id == "r_valid"
    assert loaded[0].is_synthetic is False
    assert loaded[0].image_sha256 is not None
    assert len(loader.excluded_samples) == 1
    assert loader.excluded_samples[0]["receipt_id"] == "r_corrupt"
    assert loader.excluded_samples[0]["reason"] == "missing_or_corrupted_image"

    # Identical image hash across splits triggers leakage flag
    a = align_receipt_annotations(
        "id_train",
        ["A"],
        [[0, 0, 10, 10]],
        {},
        image_sha256=loaded[0].image_sha256,
    )
    b = align_receipt_annotations(
        "id_val",
        ["B"],
        [[0, 0, 10, 10]],
        {},
        image_sha256=loaded[0].image_sha256,
    )
    leak = check_data_leakage([a], [b], [])
    assert leak["train_val_image_hash_overlaps"] == 1
    assert leak["has_leakage"] is True


def test_prepare_dataset_preserves_official_test_split(tmp_path):
    root = tmp_path / "official_sroie"
    for split, count, prefix in [("train", 6, "tr"), ("test", 3, "te")]:
        (root / split / "img").mkdir(parents=True)
        (root / split / "box").mkdir(parents=True)
        (root / split / "entities").mkdir(parents=True)
        for i in range(count):
            name = f"{prefix}_{i}"
            _create_receipt_png(root / split / "img" / f"{name}.png", f"MART {name} TOTAL {10+i}.00")
            (root / split / "box" / f"{name}.txt").write_text(
                f"10,10,150,10,150,40,10,40,MART {name}\n"
                f"10,80,150,80,150,110,10,110,{10+i}.00\n",
                encoding="utf-8",
            )
            (root / split / "entities" / f"{name}.txt").write_text(
                json.dumps({"company": f"MART {name}", "total": f"{10+i}.00"}),
                encoding="utf-8",
            )

    out_dir = tmp_path / "processed_official"
    summary = prepare_sroie_dataset(local_dir=str(root), output_dir=str(out_dir), seed=42)
    test_recs = load_aligned_receipts(out_dir / "test.jsonl")
    assert summary["counts"]["test"] == 3
    assert all(r.receipt_id.startswith("test_") for r in test_recs)
    assert summary["is_synthetic"] is False


def test_unified_preprocessing_training_vs_inference_equivalence(tmp_path):
    img_path = tmp_path / "sample_receipt.png"
    _create_receipt_png(img_path, "SUNWAY MART 45.00", color=(200, 210, 220))
    words = ["SUNWAY", "MART", "45.00"]
    boxes = [[10, 20, 80, 50], [90, 20, 150, 50], [160, 20, 220, 50]]
    labels = ["B-COMPANY", "I-COMPANY", "B-TOTAL"]
    tok = DummyFastTokenizer()

    train_prep = prepare_layoutlmv3_inputs(
        image=str(img_path),
        words=words,
        boxes=boxes,
        tokenizer=tok,
        word_labels=labels,
        max_seq_length=16,
        return_word_ids=False,
        as_numpy=True,
    )
    infer_prep = prepare_layoutlmv3_inputs(
        image=Image.open(img_path),
        words=words,
        boxes=boxes,
        tokenizer=tok,
        word_labels=None,
        max_seq_length=16,
        return_word_ids=True,
        as_numpy=True,
    )

    np.testing.assert_allclose(train_prep["pixel_values"], infer_prep["pixel_values"], atol=1e-6)
    np.testing.assert_array_equal(train_prep["input_ids"], infer_prep["input_ids"])
    np.testing.assert_array_equal(train_prep["attention_mask"], infer_prep["attention_mask"])
    np.testing.assert_array_equal(train_prep["bbox"], infer_prep["bbox"])
    assert train_prep["bbox"][0].tolist() == [0, 0, 0, 0]
    assert train_prep["labels"][0] == -100
    assert -1.05 <= float(train_prep["pixel_values"].min()) <= float(train_prep["pixel_values"].max()) <= 1.05

    # Missing image must fail loudly instead of silently generating a blank tensor
    with pytest.raises(FileNotFoundError):
        preprocess_image_array(tmp_path / "does_not_exist.png")


def test_validate_checkpoint_directory_strict_checks(tmp_path):
    ckpt = tmp_path / "ckpt"
    ok, _ = validate_checkpoint_directory(ckpt)
    assert ok is False

    ckpt.mkdir()
    # Wrong model_type
    (ckpt / "config.json").write_text(json.dumps({"model_type": "bert"}), encoding="utf-8")
    (ckpt / "tokenizer.json").write_text("{}", encoding="utf-8")
    (ckpt / "model.safetensors").write_bytes(b"weights")
    ok, reason = validate_checkpoint_directory(ckpt)
    assert ok is False and "model_type" in reason

    # Wrong label count
    (ckpt / "config.json").write_text(
        json.dumps({"model_type": "layoutlmv3", "id2label": {"0": "O"}}),
        encoding="utf-8",
    )
    ok, reason = validate_checkpoint_directory(ckpt)
    assert ok is False and "Label count" in reason

    # Valid complete checkpoint structure
    (ckpt / "config.json").write_text(
        json.dumps(
            {
                "model_type": "layoutlmv3",
                "id2label": {str(i): lbl for i, lbl in ID2LABEL.items()},
                "label2id": {lbl: i for i, lbl in ID2LABEL.items()},
            }
        ),
        encoding="utf-8",
    )
    ok, reason = validate_checkpoint_directory(ckpt)
    assert ok is True, reason


@pytest.mark.asyncio
async def test_document_service_executes_layoutlmv3_end_to_end(tmp_path):
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from backend.app.database import Base
    from backend.app.models.document import Document, DocumentStatus
    from backend.app.models.extraction import ExtractionResult
    from backend.app.services.document_service import DocumentService
    from ml.extractors.layoutlmv3_extractor import LayoutLMv3Extractor

    db_url = f"sqlite+aiosqlite:///{tmp_path / 'test_lmv3.db'}"
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    img_bytes_path = tmp_path / "upload_receipt.png"
    _create_receipt_png(img_bytes_path, "SUNWAY MART TOTAL 29.90")
    img_bytes = img_bytes_path.read_bytes()

    async def fake_extract(self, file_path, **kwargs):
        return ExtractionOutput(
            extraction_method="LAYOUTLMV3",
            is_placeholder=False,
            structured_data={
                "vendor_name": "SUNWAY MART",
                "vendor_address": "KL SENTRAL",
                "document_date": "2024-03-10",
                "total_amount": "29.90",
                "currency": "MYR",
                "tax_amount": None,
                "line_items": [],
            },
            raw_text="SUNWAY MART TOTAL 29.90",
            confidence=0.92,
            field_confidences={"vendor_name": 0.95, "total_amount": 0.90},
            validation_errors=[],
            metadata={"device": "cpu", "model": "microsoft/layoutlmv3-base"},
        )

    service = DocumentService()
    with patch("backend.app.services.document_service.settings.UPLOAD_DIR", str(tmp_path / "uploads")), \
         patch.object(LayoutLMv3Extractor, "validate_checkpoint", return_value=(True, "Valid mock checkpoint")), \
         patch.object(LayoutLMv3Extractor, "extract", new=fake_extract):
        async with session_factory() as db:
            res, status_msg = await service.create_document_from_upload(
                db=db,
                filename="receipt.png",
                content_type="image/png",
                file_bytes=img_bytes,
                extraction_method="LAYOUTLMV3",
            )
            assert res.status == DocumentStatus.COMPLETED.value
            assert "LayoutLMv3" in status_msg

            doc_row = (await db.execute(select(Document).where(Document.id == res.id))).scalar_one()
            assert doc_row.status == DocumentStatus.COMPLETED.value

            ext_row = (
                await db.execute(select(ExtractionResult).where(ExtractionResult.document_id == res.id))
            ).scalar_one()
            assert ext_row.extraction_method == "LAYOUTLMV3"
            assert ext_row.structured_data["vendor_name"] == "SUNWAY MART"
            assert ext_row.structured_data["total_amount"] == "29.90"
            assert ext_row.confidence == pytest.approx(0.92)

    await engine.dispose()
