import json
import re
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
    from transformers import (
        AutoImageProcessor,
        AutoProcessor,
        AutoTokenizer,
        LayoutLMv3ForTokenClassification,
    )

    HAS_TORCH = True
except (ImportError, OSError):
    torch = None  # type: ignore
    AutoImageProcessor = None  # type: ignore
    AutoProcessor = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    LayoutLMv3ForTokenClassification = None  # type: ignore
    HAS_TORCH = False

from PIL import Image

from backend.app.config import settings
from backend.app.logger import logger
from ml.base import BaseExtractor, ExtractionOutput
from ml.data.annotation_alignment import ID2LABEL, LABEL_LIST
from ml.inference.decoder import decode_layoutlmv3_predictions
from ml.ocr.engine import execute_tesseract_ocr, get_tesseract_status
from ml.training.config import detect_default_device
from ml.training.dataset import prepare_layoutlmv3_inputs


def validate_checkpoint_directory(checkpoint_dir: Union[str, Path]) -> Tuple[bool, str]:
    """
    Strictly validates a LayoutLMv3 checkpoint directory before loading:
    - Directory exists
    - config.json exists, parses, has model_type == 'layoutlmv3'
    - id2label matches canonical 9 SROIE BIO labels
    - Tokenizer config/vocab exists
    - Non-empty weight file exists
    """
    p = Path(checkpoint_dir)
    if not p.exists() or not p.is_dir():
        return False, f"Checkpoint directory does not exist: {p}"

    config_path = p / "config.json"
    if not config_path.exists():
        return False, f"Missing config.json in checkpoint directory: {p}"

    try:
        with open(config_path, "r", encoding="utf-8") as cf:
            cfg = json.load(cf)
    except Exception as exc:
        return False, f"Invalid config.json in checkpoint directory {p}: {exc}"

    model_type = str(cfg.get("model_type", "")).lower()
    if model_type != "layoutlmv3":
        return False, f"Unsupported model_type '{model_type}' in {config_path}; expected 'layoutlmv3'."

    id2label = cfg.get("id2label")
    if not isinstance(id2label, dict) or len(id2label) != len(LABEL_LIST):
        return (
            False,
            f"Label count mismatch in {config_path}: expected {len(LABEL_LIST)} BIO labels, got {len(id2label) if isinstance(id2label, dict) else 'none'}.",
        )

    expected_map = {str(i): lbl for i, lbl in ID2LABEL.items()}
    normalized_id2label = {str(k): str(v) for k, v in id2label.items()}
    if normalized_id2label != expected_map:
        return (
            False,
            f"id2label mapping in {config_path} does not match canonical SROIE BIO labels.",
        )

    has_tokenizer = any(
        (p / fname).exists() and (p / fname).stat().st_size > 0
        for fname in ("tokenizer.json", "tokenizer_config.json", "vocab.json")
    )
    if not has_tokenizer:
        return False, f"Tokenizer files missing in checkpoint directory: {p}"

    weight_candidates = [
        p / "model.safetensors",
        p / "pytorch_model.bin",
    ]
    has_weights = any(w.exists() and w.stat().st_size > 0 for w in weight_candidates)
    if not has_weights:
        return False, f"Non-empty model weights (model.safetensors / pytorch_model.bin) missing in {p}"

    return True, "Valid LayoutLMv3 checkpoint"


def parse_extracted_date(date_str: Optional[str]) -> Optional[date]:
    """Attempts to parse common date formats into a datetime.date object."""
    if not date_str:
        return None
    cleaned = re.sub(r"^[^\w\d]+|[^\w\d]+$", "", date_str.strip())
    formats = [
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d.%m.%Y",
        "%d/%m/%y",
        "%m/%d/%y",
        "%d-%b-%Y",
        "%d-%b-%y",
        "%d %b %Y",
        "%d %B %Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


def parse_extracted_amount(amount_str: Optional[str]) -> Optional[Decimal]:
    """Attempts to parse monetary string into Decimal."""
    if not amount_str:
        return None
    cleaned = re.sub(r"[^\d.,]", "", amount_str.strip())
    if not cleaned:
        return None
    try:
        if "," in cleaned and "." in cleaned:
            if cleaned.rfind(",") > cleaned.rfind("."):
                cleaned = cleaned.replace(".", "").replace(",", ".")
            else:
                cleaned = cleaned.replace(",", "")
        elif "," in cleaned:
            parts = cleaned.split(",")
            if len(parts[-1]) == 2:
                cleaned = cleaned.replace(",", ".")
            else:
                cleaned = cleaned.replace(",", "")
        val = Decimal(cleaned)
        return val.quantize(Decimal("0.01"))
    except Exception:
        return None


class LayoutLMv3Extractor(BaseExtractor):
    """
    Multimodal LayoutLMv3 receipt extraction pipeline.
    Combines real document images, normalized 2D bounding boxes,
    and OCR text tokens using the shared training/inference preprocessing helper.
    """

    def __init__(self, checkpoint_dir: Optional[str] = None, device: Optional[str] = None):
        self.checkpoint_dir = checkpoint_dir or getattr(
            settings, "LAYOUTLMV3_CHECKPOINT_DIR", "checkpoints/layoutlmv3-sroie-best"
        )
        self._tokenizer: Optional[Any] = None
        self._image_processor: Optional[Any] = None
        self._model: Optional[Any] = None
        self._device: str = device or detect_default_device()

    @property
    def method_name(self) -> str:
        return "LAYOUTLMV3"

    def validate_checkpoint(self) -> Tuple[bool, str]:
        if not HAS_TORCH or torch is None:
            valid_files, reason = validate_checkpoint_directory(self.checkpoint_dir)
            if not valid_files:
                return False, reason
            return False, "PyTorch/Transformers runtime unavailable on this host."
        return validate_checkpoint_directory(self.checkpoint_dir)

    def is_checkpoint_available(self) -> bool:
        """Checks if runtime is ready and a structurally valid checkpoint directory exists."""
        ok, _ = self.validate_checkpoint()
        return ok

    def load_model(self) -> None:
        """Loads validated model, tokenizer, and image processor if not already loaded."""
        if not HAS_TORCH or torch is None:
            raise RuntimeError("PyTorch/Transformers runtime is unavailable.")

        valid, reason = validate_checkpoint_directory(self.checkpoint_dir)
        if not valid:
            raise RuntimeError(f"Invalid LayoutLMv3 checkpoint: {reason}")

        if self._model is None or self._tokenizer is None:
            logger.info("Loading LayoutLMv3 model from %s onto %s", self.checkpoint_dir, self._device)
            try:
                processor = AutoProcessor.from_pretrained(self.checkpoint_dir, apply_ocr=False)
                self._tokenizer = processor.tokenizer
                self._image_processor = getattr(processor, "image_processor", None)
            except Exception:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.checkpoint_dir, add_prefix_space=True
                )
                try:
                    self._image_processor = AutoImageProcessor.from_pretrained(
                        self.checkpoint_dir, apply_ocr=False
                    )
                except Exception:
                    self._image_processor = None

            self._model = LayoutLMv3ForTokenClassification.from_pretrained(self.checkpoint_dir)
            self._model.to(self._device)
            self._model.eval()

    def _load_document_image(self, file_path: Path) -> Image.Image:
        if file_path.suffix.lower() == ".pdf":
            from ml.ocr.pdf_converter import convert_pdf_to_images

            imgs = convert_pdf_to_images(file_path, dpi=150, max_pages=1)
            if not imgs:
                raise ValueError(f"Could not render any page image from PDF: {file_path}")
            return imgs[0].convert("RGB")
        with Image.open(file_path) as im:
            rgb = im.convert("RGB")
            rgb.load()
            return rgb

    async def extract(self, file_path: Path, **kwargs: Any) -> ExtractionOutput:
        valid_ckpt, ckpt_reason = self.validate_checkpoint()
        if not valid_ckpt:
            raise NotImplementedError(
                f"LayoutLMv3 multimodal extraction is unavailable in Phase 2A / without fine-tuned weights. "
                f"Model checkpoint is not available at '{self.checkpoint_dir}': {ckpt_reason}"
            )

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Document file does not exist: {file_path}")

        self.load_model()

        is_avail, _, msg = get_tesseract_status()
        if not is_avail:
            raise RuntimeError(f"OCR dependency missing: {msg}")

        ocr_result = execute_tesseract_ocr(file_path)
        if not ocr_result.tokens:
            return ExtractionOutput(
                extraction_method=self.method_name,
                is_placeholder=False,
                structured_data={
                    "vendor_name": None,
                    "vendor_address": None,
                    "document_date": None,
                    "total_amount": None,
                    "currency": None,
                    "tax_amount": None,
                    "line_items": [],
                },
                raw_text="",
                confidence=0.0,
                field_confidences={},
                validation_errors=["No text tokens detected in document"],
                metadata={"status": "NO_TEXT_DETECTED", "device": self._device},
            )

        words = [t.text for t in ocr_result.tokens]
        boxes = [[t.norm_bbox[0], t.norm_bbox[1], t.norm_bbox[2], t.norm_bbox[3]] for t in ocr_result.tokens]

        img = self._load_document_image(file_path)
        prepared = prepare_layoutlmv3_inputs(
            image=img,
            words=words,
            boxes=boxes,
            tokenizer=self._tokenizer,
            image_processor=self._image_processor,
            word_labels=None,
            max_seq_length=512,
            return_word_ids=True,
            as_numpy=False,
        )

        input_ids = prepared["input_ids"].unsqueeze(0).to(self._device)
        attention_mask = prepared["attention_mask"].unsqueeze(0).to(self._device)
        bbox = prepared["bbox"].unsqueeze(0).to(self._device)
        pixel_values = prepared["pixel_values"].unsqueeze(0).to(self._device)
        word_ids = prepared["word_ids"]

        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
            )

        logits = outputs.logits[0].detach().cpu()
        decoded = decode_layoutlmv3_predictions(
            words=words,
            boxes=boxes,
            word_ids=word_ids,
            logits=logits,
        )

        fields = decoded["fields"]
        confs = decoded["confidences"]

        parsed_date = parse_extracted_date(fields.get("date"))
        parsed_total = parse_extracted_amount(fields.get("total"))

        currency = "USD"
        if "RM" in ocr_result.full_text or "MYR" in ocr_result.full_text:
            currency = "MYR"
        elif "₹" in ocr_result.full_text or "INR" in ocr_result.full_text:
            currency = "INR"
        elif "€" in ocr_result.full_text or "EUR" in ocr_result.full_text:
            currency = "EUR"
        elif "£" in ocr_result.full_text or "GBP" in ocr_result.full_text:
            currency = "GBP"
        elif "$" in ocr_result.full_text:
            currency = "USD"

        structured_data = {
            "vendor_name": fields.get("company"),
            "vendor_address": fields.get("address"),
            "document_date": parsed_date.isoformat() if parsed_date else None,
            "total_amount": str(parsed_total) if parsed_total is not None else None,
            "currency": currency,
            "tax_amount": None,
            "line_items": [],
        }

        field_confs = {
            "vendor_name": confs.get("company", 0.0),
            "vendor_address": confs.get("address", 0.0),
            "document_date": confs.get("date", 0.0),
            "total_amount": confs.get("total", 0.0),
        }
        overall_conf = sum(field_confs.values()) / max(1, len(field_confs))

        validation_errors: List[str] = []
        if not structured_data["vendor_name"]:
            validation_errors.append("Missing vendor_name prediction")
        if not structured_data["total_amount"]:
            validation_errors.append("Missing total_amount prediction")
        if not structured_data["document_date"]:
            validation_errors.append("Unparsed or missing document_date")

        return ExtractionOutput(
            extraction_method=self.method_name,
            is_placeholder=False,
            structured_data=structured_data,
            raw_text=ocr_result.full_text,
            confidence=round(overall_conf, 3),
            field_confidences=field_confs,
            validation_errors=validation_errors,
            metadata={
                "model": "microsoft/layoutlmv3-base",
                "checkpoint": str(self.checkpoint_dir),
                "device": self._device,
                "num_tokens": len(words),
            },
        )
