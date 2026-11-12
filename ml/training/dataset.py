from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset

    HAS_TORCH = True
except (ImportError, OSError):
    torch = None  # type: ignore

    class Dataset:  # type: ignore
        pass

    HAS_TORCH = False

from ml.data.annotation_alignment import AlignedReceipt, LABEL2ID


def load_rgb_image(image_source: Union[str, Path, Image.Image]) -> Image.Image:
    """
    Loads and validates an image as RGB PIL Image.
    Fails loudly if the path is missing or corrupted rather than substituting a blank canvas.
    """
    if isinstance(image_source, Image.Image):
        rgb = image_source.convert("RGB")
        if rgb.size[0] <= 0 or rgb.size[1] <= 0:
            raise ValueError("Provided PIL image has empty dimensions.")
        return rgb

    if not image_source:
        raise FileNotFoundError("Image path is empty or None.")

    path = Path(image_source)
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Receipt image file missing or empty: {path}")

    with Image.open(path) as img:
        rgb = img.convert("RGB")
        rgb.load()
    if rgb.size[0] <= 0 or rgb.size[1] <= 0:
        raise ValueError(f"Decoded image has invalid dimensions: {path}")
    return rgb


def preprocess_image_array(
    image: Union[str, Path, Image.Image],
    image_processor: Optional[Any] = None,
    image_size: int = 224,
) -> np.ndarray:
    """
    Preprocesses an image into CHW float32 normalized pixels matching LayoutLMv3ImageProcessor
    (RGB resize 224x224, rescale 1/255, mean=0.5, std=0.5 -> range [-1.0, 1.0]).
    """
    rgb = load_rgb_image(image)

    if image_processor is not None:
        try:
            processed = image_processor(images=rgb, return_tensors="np")
            pixel_values = processed["pixel_values"][0]
            return np.asarray(pixel_values, dtype=np.float32)
        except Exception:
            if HAS_TORCH and torch is not None:
                processed = image_processor(images=rgb, return_tensors="pt")
                return processed["pixel_values"].squeeze(0).detach().cpu().numpy().astype(np.float32)

    resized = rgb.resize((image_size, image_size), resample=Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = np.transpose(arr, (2, 0, 1))
    return arr.astype(np.float32)


def preprocess_image_tensor(
    image: Union[str, Path, Image.Image],
    image_processor: Optional[Any] = None,
    image_size: int = 224,
) -> Any:
    """Returns preprocessed (3, H, W) torch.FloatTensor (or NumPy float32 array if torch is unavailable)."""
    arr = preprocess_image_array(image, image_processor=image_processor, image_size=image_size)
    if HAS_TORCH and torch is not None:
        return torch.from_numpy(arr)
    return arr


def align_tokens_and_boxes_arrays(
    tokenizer: Any,
    words: List[str],
    boxes: List[List[int]],
    word_labels: Optional[List[str]] = None,
    max_seq_length: int = 512,
    return_word_ids: bool = False,
) -> Dict[str, Any]:
    """
    Shared tokenization + subword bounding-box and BIO label alignment returning NumPy arrays.
    - Special tokens (word_id is None): bbox [0, 0, 0, 0], label -100.
    - First subword of word_i: bbox clamped[word_i], label LABEL2ID[word_labels[word_i]] (if provided).
    - Subsequent subwords of word_i: bbox clamped[word_i], label -100.
    """
    if not words:
        words = [""]
        boxes = [[0, 0, 0, 0]]
        if word_labels is not None:
            word_labels = ["O"]

    clamped_boxes: List[List[int]] = []
    for b in boxes:
        x0 = max(0, min(1000, int(b[0])))
        y0 = max(0, min(1000, int(b[1])))
        x1 = max(x0, min(1000, int(b[2])))
        y1 = max(y0, min(1000, int(b[3])))
        clamped_boxes.append([x0, y0, x1, y1])

    try:
        encoding = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
            return_attention_mask=True,
        )
    except TypeError:
        encoding = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
        )

    try:
        word_ids = encoding.word_ids(batch_index=0)
    except TypeError:
        word_ids = encoding.word_ids()
    aligned_boxes: List[List[int]] = []
    aligned_labels: List[int] = []

    previous_word_idx: Optional[int] = None
    for word_idx in word_ids:
        if word_idx is None:
            aligned_boxes.append([0, 0, 0, 0])
            if word_labels is not None:
                aligned_labels.append(-100)
        elif word_idx != previous_word_idx:
            aligned_boxes.append(clamped_boxes[word_idx])
            if word_labels is not None:
                lbl_str = word_labels[word_idx] if word_idx < len(word_labels) else "O"
                aligned_labels.append(LABEL2ID.get(lbl_str, 0))
        else:
            aligned_boxes.append(clamped_boxes[word_idx])
            if word_labels is not None:
                aligned_labels.append(-100)
        previous_word_idx = word_idx

    raw_ids = encoding["input_ids"]
    raw_mask = encoding["attention_mask"]
    if hasattr(raw_ids, "detach"):
        raw_ids = raw_ids.detach().cpu().numpy()
    if hasattr(raw_mask, "detach"):
        raw_mask = raw_mask.detach().cpu().numpy()

    input_ids_arr = np.asarray(raw_ids, dtype=np.int64)
    if input_ids_arr.ndim > 1:
        input_ids_arr = input_ids_arr[0]
    attn_arr = np.asarray(raw_mask, dtype=np.int64)
    if attn_arr.ndim > 1:
        attn_arr = attn_arr[0]

    result: Dict[str, Any] = {
        "input_ids": input_ids_arr,
        "attention_mask": attn_arr,
        "bbox": np.asarray(aligned_boxes, dtype=np.int64),
    }
    if word_labels is not None:
        result["labels"] = np.asarray(aligned_labels, dtype=np.int64)
    if return_word_ids:
        result["word_ids"] = word_ids
    return result


def align_tokens_and_boxes(
    tokenizer: Any,
    words: List[str],
    boxes: List[List[int]],
    word_labels: List[str],
    max_seq_length: int = 512,
) -> Tuple[Any, Any, Any, Any]:
    """Tensor wrapper (with NumPy fallback when PyTorch is unavailable) preserving the 4-tuple signature."""
    arrays = align_tokens_and_boxes_arrays(
        tokenizer=tokenizer,
        words=words,
        boxes=boxes,
        word_labels=word_labels,
        max_seq_length=max_seq_length,
        return_word_ids=False,
    )
    if HAS_TORCH and torch is not None:
        return (
            torch.from_numpy(arrays["input_ids"]),
            torch.from_numpy(arrays["attention_mask"]),
            torch.from_numpy(arrays["bbox"]),
            torch.from_numpy(arrays["labels"]),
        )
    return (
        arrays["input_ids"],
        arrays["attention_mask"],
        arrays["bbox"],
        arrays["labels"],
    )


def prepare_layoutlmv3_inputs(
    image: Union[str, Path, Image.Image],
    words: List[str],
    boxes: List[List[int]],
    tokenizer: Any,
    image_processor: Optional[Any] = None,
    word_labels: Optional[List[str]] = None,
    max_seq_length: int = 512,
    image_size: int = 224,
    return_word_ids: bool = False,
    as_numpy: bool = False,
) -> Dict[str, Any]:
    """
    Canonical feature preparation pipeline shared identically across
    LayoutLMv3 training, validation, scientific benchmarking, and FastAPI inference.
    """
    pixel_arr = preprocess_image_array(
        image=image, image_processor=image_processor, image_size=image_size
    )
    token_arrays = align_tokens_and_boxes_arrays(
        tokenizer=tokenizer,
        words=words,
        boxes=boxes,
        word_labels=word_labels,
        max_seq_length=max_seq_length,
        return_word_ids=return_word_ids,
    )

    if as_numpy or not HAS_TORCH or torch is None:
        out: Dict[str, Any] = {
            "input_ids": token_arrays["input_ids"],
            "attention_mask": token_arrays["attention_mask"],
            "bbox": token_arrays["bbox"],
            "pixel_values": pixel_arr,
        }
        if "labels" in token_arrays:
            out["labels"] = token_arrays["labels"]
        if return_word_ids:
            out["word_ids"] = token_arrays["word_ids"]
        return out

    out = {
        "input_ids": torch.from_numpy(token_arrays["input_ids"]),
        "attention_mask": torch.from_numpy(token_arrays["attention_mask"]),
        "bbox": torch.from_numpy(token_arrays["bbox"]),
        "pixel_values": torch.from_numpy(pixel_arr),
    }
    if "labels" in token_arrays:
        out["labels"] = torch.from_numpy(token_arrays["labels"])
    if return_word_ids:
        out["word_ids"] = token_arrays["word_ids"]
    return out


class SROIELayoutLMv3Dataset(Dataset):
    """PyTorch Dataset wrapping SROIE AlignedReceipt records for LayoutLMv3."""

    def __init__(
        self,
        receipts: List[AlignedReceipt],
        tokenizer: Any,
        max_seq_length: int = 512,
        image_processor: Optional[Any] = None,
        image_size: int = 224,
    ):
        self.receipts = receipts
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.image_processor = image_processor
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.receipts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        receipt = self.receipts[idx]
        if not receipt.image_path:
            raise FileNotFoundError(
                f"Receipt '{receipt.receipt_id}' has no image_path; cannot prepare LayoutLMv3 sample."
            )

        return prepare_layoutlmv3_inputs(
            image=receipt.image_path,
            words=receipt.words,
            boxes=receipt.boxes,
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            word_labels=receipt.labels,
            max_seq_length=self.max_seq_length,
            image_size=self.image_size,
            return_word_ids=False,
            as_numpy=False,
        )


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = batch[0].keys()
    if HAS_TORCH and torch is not None and isinstance(batch[0][next(iter(keys))], torch.Tensor):
        return {k: torch.stack([b[k] for b in batch]) for k in keys}
    return {k: np.stack([np.asarray(b[k]) for b in batch]) for k in keys}
