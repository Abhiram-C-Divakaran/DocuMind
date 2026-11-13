import json
import re
import threading
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
from ml.evaluation.metrics import compute_checkpoint_disk_size_mb
from ml.inference.decoder import decode_layoutlmv3_predictions
from ml.ocr.engine import execute_tesseract_ocr, get_tesseract_status
from ml.training.config import detect_default_device
from ml.training.dataset import prepare_layoutlmv3_inputs

_CACHE_LOCK = threading.Lock()
# Cache keyed by (resolved_checkpoint_dir, device) -> (tokenizer, image_processor, model)
_WEIGHTS_CACHE: Dict[Tuple[str, str], Tuple[Any, Any, Any]] = {}
_EXTRACTOR_CACHE: Dict[Tuple[str, str], "LayoutLMv3Extractor"] = {}
_GPU_SEMAPHORE_LOCK = threading.Lock()
_GPU_SEMAPHORE: Optional[threading.BoundedSemaphore] = None
_GPU_SEMAPHORE_LIMIT: Optional[int] = None


def _get_gpu_inference_semaphore() -> threading.BoundedSemaphore:
    global _GPU_SEMAPHORE, _GPU_SEMAPHORE_LIMIT
    limit = max(1, int(getattr(settings, "MAX_CONCURRENT_GPU_INFERENCE", 1)))
    with _GPU_SEMAPHORE_LOCK:
        if _GPU_SEMAPHORE is None or _GPU_SEMAPHORE_LIMIT != limit:
            _GPU_SEMAPHORE = threading.BoundedSemaphore(limit)
            _GPU_SEMAPHORE_LIMIT = limit
        return _GPU_SEMAPHORE


def preload_worker_model_if_ready() -> bool:
    """Called on Celery worker process startup when PRELOAD_LAYOUTLMV3_ON_WORKER_START is enabled."""
    if not getattr(settings, "PRELOAD_LAYOUTLMV3_ON_WORKER_START", False):
        return False
    extractor = get_cached_layoutlmv3_extractor()
    valid, _ = extractor.validate_checkpoint()
    if not valid:
        logger.info("Skipping LayoutLMv3 preload: checkpoint/runtime not ready.")
        return False
    extractor.load_model()
    logger.info("Preloaded LayoutLMv3 checkpoint '%s' onto '%s'", sanitize_checkpoint_identity(extractor.checkpoint_dir), extractor._device)
    return True


def sanitize_checkpoint_identity(checkpoint_dir: Union[str, Path]) -> str:
    """Returns a non-sensitive checkpoint identifier without exposing absolute host directories."""
    raw = str(checkpoint_dir).strip()
    p = Path(raw)
    if p.is_absolute():
        return p.name or "layoutlmv3-checkpoint"
    return raw.replace("\\", "/") or "layoutlmv3-checkpoint"


def clear_layoutlmv3_cache() -> None:
    """Clears cached LayoutLMv3 weights and extractor instances (e.g. for tests/reloads)."""
    with _CACHE_LOCK:
        _WEIGHTS_CACHE.clear()
        _EXTRACTOR_CACHE.clear()


def _sanitize_error_reason(reason: str, checkpoint_dir: Union[str, Path]) -> str:
    """Strips full filesystem prefix from validation messages for safe API responses."""
    raw_str = str(checkpoint_dir)
    norm_str = str(Path(raw_str))
    safe_id = sanitize_checkpoint_identity(checkpoint_dir)
    cleaned = reason.replace(raw_str, safe_id).replace(norm_str, safe_id)
    try:
        resolved = str(Path(checkpoint_dir).resolve())
        cleaned = cleaned.replace(resolved, safe_id)
    except Exception:
        pass
    return cleaned


def read_sanitized_checkpoint_metadata(checkpoint_dir: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Reads safe scalar lineage fields from checkpoint_metadata.json if present."""
    meta_path = Path(checkpoint_dir) / "checkpoint_metadata.json"
    if not meta_path.exists() or not meta_path.is_file():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as mf:
            raw = json.load(mf)
        if not isinstance(raw, dict):
            return None
        safe_keys = (
            "experiment_id",
            "base_model",
            "seed",
            "epoch",
            "global_step",
            "best_val_f1",
            "is_synthetic",
            "smoke_test",
            "dataset_source",
            "saved_at_utc",
        )
        return {k: raw[k] for k in safe_keys if k in raw}
    except Exception:
        return None


def get_cached_layoutlmv3_extractor(
    checkpoint_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> "LayoutLMv3Extractor":
    """Returns a process-cached LayoutLMv3Extractor keyed by resolved checkpoint dir & device."""
    target_dir = checkpoint_dir or getattr(
        settings, "LAYOUTLMV3_CHECKPOINT_DIR", "checkpoints/layoutlmv3-sroie-best"
    )
    resolved_dir = str(Path(target_dir).resolve())
    resolved_dev = device or detect_default_device()
    key = (resolved_dir, resolved_dev)
    with _CACHE_LOCK:
        cached = _EXTRACTOR_CACHE.get(key)
        if cached is None or str(Path(cached.checkpoint_dir).resolve()) != resolved_dir:
            cached = LayoutLMv3Extractor(checkpoint_dir=target_dir, device=resolved_dev)
            _EXTRACTOR_CACHE[key] = cached
        return cached


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
        if not getattr(settings, "ENABLE_ML_INFERENCE", True):
            return False, "ML inference is disabled via configuration (ENABLE_ML_INFERENCE=false)."
        valid_files, reason = validate_checkpoint_directory(self.checkpoint_dir)
        if not valid_files:
            return False, reason
        if not HAS_TORCH or torch is None:
            return False, "PyTorch/Transformers runtime is unavailable on this host."
        return True, reason

    def get_readiness_info(self) -> Dict[str, Any]:
        """Returns sanitized readiness metadata suitable for API readiness checks."""
        ml_enabled = bool(getattr(settings, "ENABLE_ML_INFERENCE", True))
        valid_files, raw_reason = validate_checkpoint_directory(self.checkpoint_dir)
        ckpt_identity = sanitize_checkpoint_identity(self.checkpoint_dir)
        metadata = read_sanitized_checkpoint_metadata(self.checkpoint_dir)
        ckpt_size_mb = compute_checkpoint_disk_size_mb(self.checkpoint_dir)

        ready, full_reason = self.validate_checkpoint()
        safe_reason = None if ready else _sanitize_error_reason(full_reason, self.checkpoint_dir)
        is_synthetic = bool((metadata or {}).get("is_synthetic", False))

        return {
            "ready": bool(ready),
            "valid_files": bool(valid_files),
            "runtime_available": bool(HAS_TORCH and torch is not None),
            "inference_enabled": ml_enabled,
            "is_production_ready": bool(ready and not is_synthetic),
            "reason": safe_reason,
            "checkpoint_identity": ckpt_identity,
            "checkpoint_size_mb": ckpt_size_mb,
            "metadata": metadata,
        }

    def is_checkpoint_available(self) -> bool:
        """Checks if runtime is ready and a structurally valid checkpoint directory exists."""
        ok, _ = self.validate_checkpoint()
        return ok

    def load_model(self) -> None:
        """Loads validated model, tokenizer, and image processor, reusing process cache when available."""
        if not getattr(settings, "ENABLE_ML_INFERENCE", True):
            raise RuntimeError("ML inference is disabled via ENABLE_ML_INFERENCE=false.")
        if not HAS_TORCH or torch is None:
            raise RuntimeError("PyTorch/Transformers runtime is unavailable.")

        valid, reason = validate_checkpoint_directory(self.checkpoint_dir)
        if not valid:
            raise RuntimeError(f"Invalid LayoutLMv3 checkpoint: {reason}")

        resolved_dir = str(Path(self.checkpoint_dir).resolve())
        cache_key = (resolved_dir, self._device)

        with _CACHE_LOCK:
            if cache_key in _WEIGHTS_CACHE:
                tok, img_proc, mdl = _WEIGHTS_CACHE[cache_key]
                self._tokenizer = tok
                self._image_processor = img_proc
                self._model = mdl
                return

        if self._model is None or self._tokenizer is None:
            logger.info("Loading LayoutLMv3 model from %s onto %s", resolved_dir, self._device)
            try:
                processor = AutoProcessor.from_pretrained(self.checkpoint_dir, apply_ocr=False)
                tok = processor.tokenizer
                img_proc = getattr(processor, "image_processor", None)
            except Exception:
                tok = AutoTokenizer.from_pretrained(
                    self.checkpoint_dir, add_prefix_space=True
                )
                try:
                    img_proc = AutoImageProcessor.from_pretrained(
                        self.checkpoint_dir, apply_ocr=False
                    )
                except Exception:
                    img_proc = None

            mdl = LayoutLMv3ForTokenClassification.from_pretrained(self.checkpoint_dir)
            mdl.to(self._device)
            mdl.eval()

            self._tokenizer = tok
            self._image_processor = img_proc
            self._model = mdl
            with _CACHE_LOCK:
                _WEIGHTS_CACHE[cache_key] = (tok, img_proc, mdl)

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

        input_ids = prepared["input_ids"]
        attention_mask = prepared["attention_mask"]
        bbox = prepared["bbox"]
        pixel_values = prepared["pixel_values"]
        word_ids = prepared["word_ids"]

        device_used = self._device
        fallback_used = False

        if str(device_used).startswith("cuda"):
            sem = _get_gpu_inference_semaphore()
            with sem:
                try:
                    with torch.no_grad():
                        outputs = self._model(
                            input_ids=input_ids.unsqueeze(0).to(device_used),
                            attention_mask=attention_mask.unsqueeze(0).to(device_used),
                            bbox=bbox.unsqueeze(0).to(device_used),
                            pixel_values=pixel_values.unsqueeze(0).to(device_used),
                        )
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        logger.warning("GPU out-of-memory during LayoutLMv3 inference; falling back to CPU explicitly.")
                        if HAS_TORCH and torch is not None and hasattr(torch, "cuda"):
                            torch.cuda.empty_cache()
                        cpu_extractor = get_cached_layoutlmv3_extractor(self.checkpoint_dir, device="cpu")
                        cpu_extractor.load_model()
                        device_used = "cpu"
                        fallback_used = True
                        with torch.no_grad():
                            outputs = cpu_extractor._model(
                                input_ids=input_ids.unsqueeze(0).to("cpu"),
                                attention_mask=attention_mask.unsqueeze(0).to("cpu"),
                                bbox=bbox.unsqueeze(0).to("cpu"),
                                pixel_values=pixel_values.unsqueeze(0).to("cpu"),
                            )
                    else:
                        raise
        else:
            with torch.no_grad():
                outputs = self._model(
                    input_ids=input_ids.unsqueeze(0).to(device_used),
                    attention_mask=attention_mask.unsqueeze(0).to(device_used),
                    bbox=bbox.unsqueeze(0).to(device_used),
                    pixel_values=pixel_values.unsqueeze(0).to(device_used),
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
                "checkpoint": sanitize_checkpoint_identity(self.checkpoint_dir),
                "device": device_used,
                "device_fallback": fallback_used,
                "num_tokens": len(words),
                "ocr_tokens": [
                    {
                        "text": t.text,
                        "norm_bbox": list(t.norm_bbox),
                        "page_num": t.page_num,
                    }
                    for t in ocr_result.tokens[:1000]
                ],
            },
        )
