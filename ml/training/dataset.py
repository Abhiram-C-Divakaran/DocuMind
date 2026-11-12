from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except (ImportError, OSError):
    torch = None  # type: ignore
    class Dataset:  # type: ignore
        pass
    HAS_TORCH = False
from PIL import Image

from ml.data.annotation_alignment import AlignedReceipt, LABEL2ID


def align_tokens_and_boxes(
    tokenizer: Any,
    words: List[str],
    boxes: List[List[int]],
    word_labels: List[str],
    max_seq_length: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tokenizes words using LayoutLMv3 tokenizer, maps subword tokens to parent boxes,
    and assigns token-level BIO labels (first subword gets label, subsequent subwords get -100).

    Returns:
        input_ids: (max_seq_length,)
        attention_mask: (max_seq_length,)
        bbox: (max_seq_length, 4)
        labels: (max_seq_length,)
    """
    if not words:
        words = [""]
        boxes = [[0, 0, 0, 0]]
        word_labels = ["O"]

    # Clamp boxes to 0-1000
    clamped_boxes = []
    for b in boxes:
        clamped_boxes.append(
            [
                max(0, min(1000, int(b[0]))),
                max(0, min(1000, int(b[1]))),
                max(0, min(1000, int(b[2]))),
                max(0, min(1000, int(b[3]))),
            ]
        )

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
        return_offsets_mapping=False,
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    # Retrieve word index for each token
    word_ids = encoding.word_ids()

    token_boxes: List[List[int]] = []
    token_labels: List[int] = []
    prev_word_idx = None

    for i, word_idx in enumerate(word_ids):
        if word_idx is None:
            # Special token (<s>, </s>, <pad>)
            token_boxes.append([0, 0, 0, 0])
            token_labels.append(-100)
        else:
            token_boxes.append(clamped_boxes[word_idx])
            if word_idx != prev_word_idx:
                # First subword: assign word BIO label
                lbl_str = word_labels[word_idx] if word_idx < len(word_labels) else "O"
                token_labels.append(LABEL2ID.get(lbl_str, 0))
            else:
                # Subsequent subword: mask loss with -100
                token_labels.append(-100)
        prev_word_idx = word_idx

    if HAS_TORCH and torch is not None:
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(token_boxes, dtype=torch.long),
            torch.tensor(token_labels, dtype=torch.long),
        )
    return (
        np.array(input_ids, dtype=np.int64),
        np.array(attention_mask, dtype=np.int64),
        np.array(token_boxes, dtype=np.int64),
        np.array(token_labels, dtype=np.int64),
    )


def preprocess_image_tensor(image: Image.Image) -> Any:
    """
    Prepares a document image for LayoutLMv3:
    Resizes to 224x224 and normalizes RGB channels to [-1.0, 1.0].
    Returns shape: (3, 224, 224)
    """
    img_resized = image.convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32) / 255.0  # [0, 1]
    # Normalize with mean=0.5, std=0.5 -> [-1, 1]
    arr = (arr - 0.5) / 0.5
    # Channels first: (3, 224, 224)
    transposed = arr.transpose(2, 0, 1)
    if HAS_TORCH and torch is not None:
        return torch.tensor(transposed, dtype=torch.float32)
    return transposed


class SROIELayoutLMv3Dataset(Dataset):
    """
    PyTorch Dataset for fine-tuning LayoutLMv3 on SROIE.
    Converts AlignedReceipt records into multi-modal tensor features.
    """

    def __init__(
        self,
        receipts: List[AlignedReceipt],
        tokenizer: Any,
        max_seq_length: int = 512,
    ):
        self.receipts = receipts
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.receipts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.receipts[idx]

        # 1. Load or synthesize image
        if item.image_path and Path(item.image_path).exists():
            try:
                img = Image.open(item.image_path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (1000, 1000), color=(255, 255, 255))
        else:
            img = Image.new("RGB", (1000, 1000), color=(255, 255, 255))

        pixel_values = preprocess_image_tensor(img)

        # 2. Tokenize and align
        input_ids, attention_mask, bbox, labels = align_tokens_and_boxes(
            tokenizer=self.tokenizer,
            words=item.words,
            boxes=item.boxes,
            word_labels=item.labels,
            max_seq_length=self.max_seq_length,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "bbox": bbox,
            "labels": labels,
            "pixel_values": pixel_values,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collates a list of dataset sample dicts into a batched tensor dict."""
    if HAS_TORCH and torch is not None:
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            "bbox": torch.stack([x["bbox"] for x in batch]),
            "labels": torch.stack([x["labels"] for x in batch]),
            "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        }
    return {
        "input_ids": np.stack([x["input_ids"] for x in batch]),
        "attention_mask": np.stack([x["attention_mask"] for x in batch]),
        "bbox": np.stack([x["bbox"] for x in batch]),
        "labels": np.stack([x["labels"] for x in batch]),
        "pixel_values": np.stack([x["pixel_values"] for x in batch]),
    }
