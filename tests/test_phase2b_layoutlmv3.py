import json
from decimal import Decimal
from pathlib import Path
import numpy as np
import pytest

try:
    import torch
    HAS_TORCH = True
except (ImportError, OSError):
    torch = None  # type: ignore
    HAS_TORCH = False

from PIL import Image

from backend.app.core.exceptions import FileValidationError
from backend.app.services.document_service import get_extractor
from ml.data.annotation_alignment import (
    ID2LABEL,
    LABEL2ID,
    LABEL_LIST,
    align_receipt_annotations,
    find_best_matching_span,
)
from ml.data.dataset_validation import generate_dataset_validation_report
from ml.data.prepare_dataset import (
    check_data_leakage,
    load_aligned_receipts,
    prepare_sroie_dataset,
)
from ml.data.sroie_loader import SROIELoader, parse_quad_bbox
from ml.evaluation.metrics import (
    compute_document_metrics,
    compute_model_footprint,
    compute_seqeval_metrics,
    normalize_entity_value,
)
from ml.extractors.layoutlmv3_extractor import (
    LayoutLMv3Extractor,
    parse_extracted_amount,
    parse_extracted_date,
)
from ml.inference.decoder import decode_layoutlmv3_predictions
from ml.training.config import LayoutLMv3TrainingConfig
from ml.training.dataset import (
    SROIELayoutLMv3Dataset,
    align_tokens_and_boxes,
    collate_fn,
    preprocess_image_tensor,
)


class DummyTokenizer:
    """Lightweight mock tokenizer for fast unit testing without downloading HF models."""

    def __init__(self):
        self.pad_token_id = 1

    def __call__(
        self,
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_offsets_mapping=False,
    ):
        word_ids = []
        input_ids = [0]  # BOS
        word_ids.append(None)

        for w_idx, w in enumerate(words):
            # Split into pseudo subwords
            subwords = [w[:2], w[2:]] if len(w) > 4 else [w]
            for _ in subwords:
                input_ids.append(100 + w_idx)
                word_ids.append(w_idx)

        input_ids.append(2)  # EOS
        word_ids.append(None)

        # Pad to max_length
        curr_len = len(input_ids)
        if curr_len < max_length:
            input_ids.extend([self.pad_token_id] * (max_length - curr_len))
            word_ids.extend([None] * (max_length - curr_len))
        else:
            input_ids = input_ids[:max_length]
            word_ids = word_ids[:max_length]

        class Encoding:
            def __init__(self, ids, w_ids):
                self._ids = ids
                self._w_ids = w_ids

            def __getitem__(self, item):
                if item == "input_ids":
                    return self._ids
                if item == "attention_mask":
                    return [1 if x != 1 else 0 for x in self._ids]
                raise KeyError(item)

            def word_ids(self):
                return self._w_ids

        return Encoding(input_ids, word_ids)


def test_bio_label_schema():
    assert len(LABEL_LIST) == 9
    assert LABEL_LIST[0] == "O"
    assert "B-COMPANY" in LABEL_LIST
    assert "I-COMPANY" in LABEL_LIST
    assert "B-DATE" in LABEL_LIST
    assert "I-DATE" in LABEL_LIST
    assert "B-ADDRESS" in LABEL_LIST
    assert "I-ADDRESS" in LABEL_LIST
    assert "B-TOTAL" in LABEL_LIST
    assert "I-TOTAL" in LABEL_LIST
    for i, l in enumerate(LABEL_LIST):
        assert ID2LABEL[i] == l
        assert LABEL2ID[l] == i


def test_parse_quad_bbox():
    quad = [100, 200, 300, 200, 300, 400, 100, 400]
    box = parse_quad_bbox(quad, img_width=1000, img_height=1000)
    assert list(box) == [100, 200, 300, 400]

    # Test clamping
    quad_overflow = [-50, -50, 1500, -50, 1500, 1500, -50, 1500]
    box_clamped = parse_quad_bbox(quad_overflow, img_width=1000, img_height=1000)
    assert list(box_clamped) == [0, 0, 1000, 1000]


def test_annotation_alignment():
    words = ["TARGET", "STORE", "#123", "123", "MAIN", "ST", "DATE:", "15/08/2023", "TOTAL", "45.50"]
    boxes = [[100, 100, 200, 120] for _ in words]
    ground_truth = {
        "company": "TARGET STORE",
        "address": "123 MAIN ST",
        "date": "15/08/2023",
        "total": "45.50",
    }

    aligned = align_receipt_annotations(
        receipt_id="rec_001",
        words=words,
        boxes=boxes,
        ground_truth=ground_truth,
    )

    assert aligned.receipt_id == "rec_001"
    assert aligned.alignment_stats["COMPANY"] is True
    assert aligned.alignment_stats["ADDRESS"] is True
    assert aligned.alignment_stats["DATE"] is True
    assert aligned.alignment_stats["TOTAL"] is True
    assert aligned.labels[0] == "B-COMPANY"
    assert aligned.labels[1] == "I-COMPANY"
    assert aligned.labels[3] == "B-ADDRESS"
    assert aligned.labels[7] == "B-DATE"
    assert aligned.labels[9] == "B-TOTAL"
    assert len(aligned.unmatched_fields) == 0


def test_annotation_alignment_unmatched():
    words = ["STORE", "ONLY"]
    boxes = [[0, 0, 10, 10], [10, 0, 20, 10]]
    ground_truth = {
        "company": "UNKNOWN ENTERPRISES",
        "total": "999.99",
    }
    aligned = align_receipt_annotations(
        receipt_id="rec_unmatched",
        words=words,
        boxes=boxes,
        ground_truth=ground_truth,
    )
    assert aligned.alignment_stats["COMPANY"] is False
    assert "COMPANY" in aligned.unmatched_fields
    assert aligned.alignment_stats["TOTAL"] is False
    assert "TOTAL" in aligned.unmatched_fields


def test_dataset_preparation_and_zero_leakage(tmp_path: Path):
    data_dir = tmp_path / "sroie_test"
    splits = prepare_sroie_dataset(
        output_dir=data_dir,
        generate_synthetic_if_missing=True,
        seed=123,
    )

    assert "train" in splits
    assert "val" in splits
    assert "test" in splits
    assert len(splits["train"]) > 0
    assert len(splits["val"]) > 0
    assert len(splits["test"]) > 0

    # Test data leakage checker
    violations = check_data_leakage(splits)
    assert len(violations) == 0

    # Verify JSONL persistence and reloading
    train_loaded = load_aligned_receipts(data_dir / "train.jsonl")
    assert len(train_loaded) == len(splits["train"])
    assert train_loaded[0].receipt_id == splits["train"][0].receipt_id

    # Test validation report generation
    report = generate_dataset_validation_report(splits, output_dir=data_dir)
    assert report["total_records"] == len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert (data_dir / "validation_report.json").exists()
    assert (data_dir / "validation_report.md").exists()


def test_token_and_box_alignment_with_subword_masking():
    tokenizer = DummyTokenizer()
    words = ["SUPERMARKET", "DATE", "12/12/2023"]
    boxes = [[50, 50, 200, 100], [50, 120, 100, 150], [110, 120, 250, 150]]
    word_labels = ["B-COMPANY", "O", "B-DATE"]

    input_ids, attention_mask, bbox, labels = align_tokens_and_boxes(
        tokenizer=tokenizer,
        words=words,
        boxes=boxes,
        word_labels=word_labels,
        max_seq_length=16,
    )

    assert len(input_ids) == 16
    assert len(attention_mask) == 16
    assert len(bbox) == 16
    assert len(labels) == 16

    # Special token at index 0 should have box [0, 0, 0, 0] and label -100
    b0 = bbox[0].tolist() if hasattr(bbox[0], "tolist") else list(bbox[0])
    l0 = int(labels[0].item()) if hasattr(labels[0], "item") else int(labels[0])
    assert b0 == [0, 0, 0, 0]
    assert l0 == -100

    # First subword of word 0 should get B-COMPANY ID
    l1 = int(labels[1].item()) if hasattr(labels[1], "item") else int(labels[1])
    assert l1 == LABEL2ID["B-COMPANY"]
    # Subsequent subword should get -100
    l2 = int(labels[2].item()) if hasattr(labels[2], "item") else int(labels[2])
    assert l2 == -100

    # Padding tokens should have box [0,0,0,0] and label -100
    blast = bbox[-1].tolist() if hasattr(bbox[-1], "tolist") else list(bbox[-1])
    llast = int(labels[-1].item()) if hasattr(labels[-1], "item") else int(labels[-1])
    assert blast == [0, 0, 0, 0]
    assert llast == -100


def test_image_preprocessing_tensor():
    img = Image.new("RGB", (600, 800), color=(128, 128, 128))
    tensor = preprocess_image_tensor(img)
    assert tuple(tensor.shape) == (3, 224, 224)
    # Check normalization range approx [-1, 1]
    min_val = float(tensor.min().item()) if hasattr(tensor.min(), "item") else float(tensor.min())
    max_val = float(tensor.max().item()) if hasattr(tensor.max(), "item") else float(tensor.max())
    assert min_val >= -1.1
    assert max_val <= 1.1


def test_collate_fn():
    if HAS_TORCH and torch is not None:
        sample1 = {
            "input_ids": torch.zeros(8, dtype=torch.long),
            "attention_mask": torch.ones(8, dtype=torch.long),
            "bbox": torch.zeros(8, 4, dtype=torch.long),
            "labels": torch.full((8,), -100, dtype=torch.long),
            "pixel_values": torch.zeros(3, 224, 224, dtype=torch.float32),
        }
        sample2 = {
            "input_ids": torch.ones(8, dtype=torch.long),
            "attention_mask": torch.ones(8, dtype=torch.long),
            "bbox": torch.ones(8, 4, dtype=torch.long),
            "labels": torch.zeros(8, dtype=torch.long),
            "pixel_values": torch.ones(3, 224, 224, dtype=torch.float32),
        }
    else:
        sample1 = {
            "input_ids": np.zeros(8, dtype=np.int64),
            "attention_mask": np.ones(8, dtype=np.int64),
            "bbox": np.zeros((8, 4), dtype=np.int64),
            "labels": np.full((8,), -100, dtype=np.int64),
            "pixel_values": np.zeros((3, 224, 224), dtype=np.float32),
        }
        sample2 = {
            "input_ids": np.ones(8, dtype=np.int64),
            "attention_mask": np.ones(8, dtype=np.int64),
            "bbox": np.ones((8, 4), dtype=np.int64),
            "labels": np.zeros(8, dtype=np.int64),
            "pixel_values": np.ones((3, 224, 224), dtype=np.float32),
        }

    batch = collate_fn([sample1, sample2])
    assert tuple(batch["input_ids"].shape) == (2, 8)
    assert tuple(batch["attention_mask"].shape) == (2, 8)
    assert tuple(batch["bbox"].shape) == (2, 8, 4)
    assert tuple(batch["labels"].shape) == (2, 8)
    assert tuple(batch["pixel_values"].shape) == (2, 3, 224, 224)


def test_seqeval_and_document_metrics():
    predictions = [["B-COMPANY", "I-COMPANY", "O", "B-TOTAL"]]
    references = [["B-COMPANY", "I-COMPANY", "O", "B-TOTAL"]]

    seqeval_res = compute_seqeval_metrics(predictions, references)
    assert seqeval_res["overall_f1"] == 1.0
    assert seqeval_res["entities"]["COMPANY"]["f1"] == 1.0
    assert seqeval_res["entities"]["TOTAL"]["f1"] == 1.0

    doc_preds = [{"company": "ABC LTD", "date": "2023-01-01", "address": "123 ROAD", "total": "50.00"}]
    doc_refs = [{"company": "ABC Ltd.", "date": "2023-01-01", "address": "123 ROAD", "total": "50.00"}]

    doc_res = compute_document_metrics(doc_preds, doc_refs)
    assert doc_res["document_exact_match"] == 1.0
    assert doc_res["field_exact_match"]["company"] == 1.0


def test_decoder_logic():
    words = ["WALMART", "STORE", "101", "DATE", "01/01/2024", "TOTAL", "$85.00"]
    boxes = [[0, 0, 10, 10]] * len(words)
    word_ids = [None, 0, 1, 2, 3, 4, 5, 6, None]  # length 9

    # Build dummy logits: length 9, num_labels 9
    if HAS_TORCH and torch is not None:
        logits = torch.zeros(9, 9)
    else:
        logits = np.zeros((9, 9), dtype=np.float32)

    logits[1, LABEL2ID["B-COMPANY"]] = 10.0
    logits[2, LABEL2ID["I-COMPANY"]] = 10.0
    logits[4, LABEL2ID["B-DATE"]] = 10.0
    logits[7, LABEL2ID["B-TOTAL"]] = 10.0

    decoded = decode_layoutlmv3_predictions(
        words=words,
        boxes=boxes,
        word_ids=word_ids,
        logits=logits,
    )

    fields = decoded["fields"]
    assert fields["company"] == "WALMART STORE"
    assert fields["date"] == "DATE"
    assert fields["total"] == "$85.00"


def test_date_and_amount_parsers():
    assert parse_extracted_date("25/12/2023") == parse_extracted_date("2023-12-25")
    assert parse_extracted_date("04-OCT-18") is not None
    assert parse_extracted_date("invalid-text") is None

    assert parse_extracted_amount("$1,234.56") == Decimal("1234.56")
    assert parse_extracted_amount("EUR 45,90") == Decimal("45.90")
    assert parse_extracted_amount("not an amount") is None


def test_layoutlmv3_extractor_missing_checkpoint():
    extractor = LayoutLMv3Extractor(checkpoint_dir="non_existent_checkpoint_directory")
    assert extractor.is_checkpoint_available() is False

    with pytest.raises(FileValidationError) as exc_info:
        get_extractor("LAYOUTLMV3")
    assert "Model checkpoint is not available" in str(exc_info.value)
