import re
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import torch
    from transformers import AutoTokenizer, LayoutLMv3ForTokenClassification
    HAS_TORCH = True
except (ImportError, OSError):
    torch = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    LayoutLMv3ForTokenClassification = None  # type: ignore
    HAS_TORCH = False

from PIL import Image

from backend.app.config import settings
from backend.app.logger import logger
from ml.base import BaseExtractor, ExtractionOutput
from ml.inference.decoder import decode_layoutlmv3_predictions
from ml.ocr.engine import execute_tesseract_ocr, get_tesseract_status


def parse_extracted_date(date_str: Optional[str]) -> Optional[date]:
    """Attempts to parse common date formats into a datetime.date object."""
    if not date_str:
        return None
    cleaned = re.sub(r"^[^\w\d]+|[^\w\d]+$", "", date_str.strip())
    formats = [
        "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y",
        "%d.%m.%Y", "%d/%m/%y", "%m/%d/%y", "%d-%b-%Y",
        "%d-%b-%y", "%d %b %Y", "%d %B %Y",
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
    Combines visual features, 2D normalized bounding box positions,
    and text tokens to extract structured receipt entities.
    """

    def __init__(self, checkpoint_dir: Optional[str] = None):
        self.checkpoint_dir = checkpoint_dir or getattr(
            settings, "LAYOUTLMV3_CHECKPOINT_DIR", "checkpoints/layoutlmv3-sroie-best"
        )
        self._tokenizer: Optional[Any] = None
        self._model: Optional[Any] = None
        self._device: str = "cuda" if (HAS_TORCH and torch and torch.cuda.is_available()) else "cpu"

    @property
    def method_name(self) -> str:
        return "LAYOUTLMV3"

    def is_checkpoint_available(self) -> bool:
        """Checks if a valid trained checkpoint directory exists."""
        if not HAS_TORCH:
            return False
        p = Path(self.checkpoint_dir)
        if not p.exists() or not p.is_dir():
            return False
        has_config = (p / "config.json").exists()
        has_weights = (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()
        return has_config and has_weights

    def load_model(self):
        """Loads model and tokenizer if not already loaded."""
        if not HAS_TORCH:
            raise RuntimeError(
                "PyTorch is not available or blocked by system security policy."
            )
        if not self.is_checkpoint_available():
            raise RuntimeError(
                f"LayoutLMv3 checkpoint not found at '{self.checkpoint_dir}'. "
                "Please train the model using 'python -m ml.training.train_layoutlmv3' "
                "or specify a valid checkpoint path."
            )

        if self._model is None or self._tokenizer is None:
            logger.info("Loading LayoutLMv3 model from %s onto %s", self.checkpoint_dir, self._device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_dir, add_prefix_space=True)
            self._model = LayoutLMv3ForTokenClassification.from_pretrained(self.checkpoint_dir)
            self._model.to(self._device)
            self._model.eval()

    async def extract(self, file_path: Path, **kwargs: Any) -> ExtractionOutput:
        # 1. Verify model checkpoint exists
        if not HAS_TORCH or not self.is_checkpoint_available():
            raise NotImplementedError(
                f"LayoutLMv3 multimodal extraction is unavailable in Phase 2A / without fine-tuned weights. "
                f"Model checkpoint not found at '{self.checkpoint_dir}'. "
                "Please fine-tune the model using 'python -m ml.training.train_layoutlmv3' "
                "before selecting the LAYOUTLMV3 extraction method."
            )

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Document file does not exist: {file_path}")

        from ml.training.dataset import preprocess_image_tensor
        self.load_model()

        # 2. Check Tesseract OCR availability
        is_avail, _, msg = get_tesseract_status()
        if not is_avail:
            raise RuntimeError(f"OCR dependency missing: {msg}")

        # 3. Execute OCR to retrieve word tokens & bounding boxes
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
                metadata={"status": "NO_TEXT_DETECTED"},
            )

        words = [t.text for t in ocr_result.tokens]
        boxes = [[t.norm_bbox[0], t.norm_bbox[1], t.norm_bbox[2], t.norm_bbox[3]] for t in ocr_result.tokens]

        # 4. Prepare image pixel values
        try:
            if file_path.suffix.lower() == ".pdf":
                from ml.ocr.pdf_converter import convert_pdf_to_images
                imgs = convert_pdf_to_images(file_path, dpi=150, max_pages=1)
                img = imgs[0] if imgs else Image.new("RGB", (1000, 1000), (255, 255, 255))
            else:
                img = Image.open(file_path).convert("RGB")
        except Exception as e:
            logger.warning("Could not read image for pixel values: %s. Using blank canvas.", e)
            img = Image.new("RGB", (1000, 1000), (255, 255, 255))

        pixel_values = preprocess_image_tensor(img).unsqueeze(0).to(self._device)

        # 5. Tokenize for LayoutLMv3
        encoding = self._tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self._device)
        attention_mask = encoding["attention_mask"].to(self._device)

        # Map boxes to subwords
        clamped_boxes = [[max(0, min(1000, int(c))) for c in b] for b in boxes]
        word_ids = encoding.word_ids()
        t_boxes = []
        for w_idx in word_ids:
            if w_idx is None or w_idx >= len(clamped_boxes):
                t_boxes.append([0, 0, 0, 0])
            else:
                t_boxes.append(clamped_boxes[w_idx])
        bbox = torch.tensor([t_boxes], dtype=torch.long, device=self._device)

        # 6. Model forward pass
        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
            )

        logits = outputs.logits[0]

        # 7. Decode predictions
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

        validation_errors = []
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
                "num_tokens": len(words),
            },
        )
