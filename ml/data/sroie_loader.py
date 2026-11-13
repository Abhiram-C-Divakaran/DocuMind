import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

from ml.data.annotation_alignment import (
    LABEL_LIST,
    split_segment_to_words_and_boxes,
)
from ml.ocr.engine import normalize_coordinates

logger = logging.getLogger("documind.ml.data.sroie_loader")

HF_DEFAULT_DATASET = "arvindrajan92/sroie_document_understanding"


@dataclass
class RawReceiptData:
    receipt_id: str
    image_path: Optional[str]
    width: int
    height: int
    words: List[str]
    boxes: List[List[int]]  # Normalized 0-1000 [x0, y0, x1, y1]
    ground_truth: Dict[str, str]
    split: Optional[str] = None
    image_sha256: Optional[str] = None
    is_synthetic: bool = False
    source: str = "unknown"
    pre_assigned_labels: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def normalize_xyxy(
    x0: float, y0: float, x1: float, y1: float, img_w: int, img_h: int
) -> List[int]:
    """Normalizes corner coordinates [x0, y0, x1, y1] directly to integer 0-1000 range."""
    w = max(1, int(img_w))
    h = max(1, int(img_h))
    nx0 = max(0, min(1000, int(round((float(x0) / w) * 1000))))
    ny0 = max(0, min(1000, int(round((float(y0) / h) * 1000))))
    nx1 = max(nx0, min(1000, int(round((float(x1) / w) * 1000))))
    ny1 = max(ny0, min(1000, int(round((float(y1) / h) * 1000))))
    return [nx0, ny0, nx1, ny1]


def parse_quad_bbox(
    coords: List[int],
    img_width: int = 1000,
    img_height: int = 1000,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> List[int]:
    """Converts 8-point quad [x1,y1,x2,y2,x3,y3,x4,y4] or 4-point box to normalized 0-1000 [x0,y0,x1,y1]."""
    w = width if width is not None else img_width
    h = height if height is not None else img_height
    if len(coords) == 8:
        xs = [coords[0], coords[2], coords[4], coords[6]]
        ys = [coords[1], coords[3], coords[5], coords[7]]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    elif len(coords) == 4:
        x0, y0, x1, y1 = coords[0], coords[1], coords[2], coords[3]
    else:
        raise ValueError(f"Expected 4 or 8 coordinates, got {len(coords)}")

    return normalize_xyxy(x0, y0, x1, y1, w, h)


def _compute_file_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_and_inspect_image(image_path: Path) -> Optional[Tuple[int, int, str]]:
    """
    Verifies an image file exists, is non-empty, decodes cleanly as RGB,
    and returns (width, height, sha256_hex). Returns None on failure.
    """
    if not image_path.exists() or not image_path.is_file() or image_path.stat().st_size == 0:
        return None
    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            width, height = rgb.size
        if width <= 0 or height <= 0:
            return None
        return width, height, _compute_file_sha256(image_path)
    except Exception:
        return None


class SROIELoader:
    """Loads SROIE receipts from local benchmarks or verified Hugging Face Hub repositories."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or "data/sroie_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.excluded_samples: List[Dict[str, str]] = []

    def _record_excluded(self, receipt_id: str, reason: str, detail: str = "") -> None:
        self.excluded_samples.append(
            {"receipt_id": receipt_id, "reason": reason, "detail": detail}
        )

    def _load_local_split_dir(
        self, base_dir: Path, split_label: Optional[str] = None
    ) -> List[RawReceiptData]:
        img_dir = base_dir / "img"
        if not img_dir.exists() and (base_dir / "images").exists():
            img_dir = base_dir / "images"
        box_dir = base_dir / "box"
        entities_dir = base_dir / "entities"

        if not img_dir.exists() or not box_dir.exists():
            raise FileNotFoundError(
                f"Expected 'img' (or 'images') and 'box' subdirectories in {base_dir}"
            )

        samples: List[RawReceiptData] = []
        image_files = sorted(
            [p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        )

        for img_path in image_files:
            stem = img_path.stem
            receipt_id = f"{split_label}_{stem}" if split_label else stem
            box_path = box_dir / f"{stem}.txt"
            ent_path = entities_dir / f"{stem}.txt" if entities_dir.exists() else None

            if not box_path.exists():
                self._record_excluded(receipt_id, "missing_box_annotation", str(box_path))
                continue

            img_info = validate_and_inspect_image(img_path)
            if img_info is None:
                self._record_excluded(receipt_id, "missing_or_corrupted_image", str(img_path))
                continue
            width, height, image_sha256 = img_info

            words: List[str] = []
            boxes: List[List[int]] = []
            try:
                with open(box_path, "r", encoding="utf-8", errors="replace") as bf:
                    for line in bf:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(",", 8)
                        if len(parts) < 9:
                            continue
                        coords = [int(float(c)) for c in parts[:8]]
                        text = parts[8].strip()
                        if not text:
                            continue
                        norm_box = parse_quad_bbox(coords, width, height)
                        seg_words, seg_boxes = split_segment_to_words_and_boxes(text, norm_box)
                        words.extend(seg_words)
                        boxes.extend(seg_boxes)
            except Exception as exc:
                self._record_excluded(receipt_id, "box_parse_error", str(exc))
                continue

            if not words:
                self._record_excluded(receipt_id, "empty_ocr_tokens", str(box_path))
                continue

            ground_truth: Dict[str, str] = {}
            if ent_path and ent_path.exists():
                try:
                    with open(ent_path, "r", encoding="utf-8", errors="replace") as ef:
                        raw_ent = json.load(ef)
                        if isinstance(raw_ent, dict):
                            ground_truth = {
                                k.lower(): str(v).strip()
                                for k, v in raw_ent.items()
                                if v is not None
                            }
                except Exception as exc:
                    logger.warning("Failed to parse entity file %s: %s", ent_path, exc)

            samples.append(
                RawReceiptData(
                    receipt_id=receipt_id,
                    image_path=str(img_path.resolve()),
                    width=width,
                    height=height,
                    words=words,
                    boxes=boxes,
                    ground_truth=ground_truth,
                    split=split_label,
                    image_sha256=image_sha256,
                    is_synthetic=False,
                    source="local_sroie",
                )
            )
        return samples

    def load_from_local_directory(self, root_dir: Path) -> List[RawReceiptData]:
        """
        Loads SROIE Task 3 data from local directory.
        Supports either flat (img/, box/, entities/) or official split subdirs (train/, test/).
        """
        root_dir = Path(root_dir)
        self.excluded_samples = []

        if (root_dir / "train").is_dir() or (root_dir / "test").is_dir():
            all_samples: List[RawReceiptData] = []
            for split_name in ("train", "val", "validation", "test"):
                sub = root_dir / split_name
                if sub.is_dir():
                    canon_split = "validation" if split_name in ("val", "validation") else split_name
                    all_samples.extend(self._load_local_split_dir(sub, split_label=canon_split))
            if not all_samples:
                raise RuntimeError(f"No valid receipts loaded from split directory {root_dir}")
            return all_samples

        samples = self._load_local_split_dir(root_dir, split_label=None)
        if not samples:
            raise RuntimeError(f"No valid receipts loaded from {root_dir}")
        return samples

    def _save_pil_image(self, image_obj: Any, dest_path: Path) -> Optional[Tuple[int, int, str]]:
        """Saves a PIL image or image bytes payload to dest_path and returns (width, height, sha256)."""
        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(image_obj, Image.Image):
                rgb = image_obj.convert("RGB")
                rgb.save(dest_path, format="JPEG", quality=95)
            elif isinstance(image_obj, dict) and image_obj.get("bytes"):
                from io import BytesIO

                with Image.open(BytesIO(image_obj["bytes"])) as im:
                    im.convert("RGB").save(dest_path, format="JPEG", quality=95)
            elif isinstance(image_obj, dict) and image_obj.get("path"):
                candidate = Path(image_obj["path"])
                if not candidate.exists():
                    return None
                with Image.open(candidate) as im:
                    im.convert("RGB").save(dest_path, format="JPEG", quality=95)
            else:
                return None

            return validate_and_inspect_image(dest_path)
        except Exception as exc:
            logger.warning("Failed to persist HF image to %s: %s", dest_path, exc)
            return None

    def _parse_arvindrajan_row(
        self, row: Dict[str, Any], receipt_id: str, split_name: str, images_dir: Path
    ) -> Optional[RawReceiptData]:
        """Parses a row from arvindrajan92/sroie_document_understanding."""
        dest_path = images_dir / f"{receipt_id}.jpg"
        img_info = validate_and_inspect_image(dest_path)
        if img_info is None:
            img_info = self._save_pil_image(row.get("image"), dest_path)
        if img_info is None:
            self._record_excluded(receipt_id, "missing_or_corrupted_image", "HF image decode failed")
            return None
        width, height, image_sha256 = img_info

        ocr_segments = row.get("ocr") or []
        words: List[str] = []
        boxes: List[List[int]] = []
        pre_labels: List[str] = []
        entity_chunks: Dict[str, List[str]] = {"company": [], "date": [], "address": [], "total": []}

        label_map = {
            "company": "COMPANY",
            "date": "DATE",
            "address": "ADDRESS",
            "total": "TOTAL",
        }
        prev_canon: Optional[str] = None

        for seg in ocr_segments:
            seg_text = str(seg.get("text") or "").strip()
            raw_box = seg.get("box")
            raw_label = str(seg.get("label") or "other").strip().lower()
            if not seg_text or not raw_box:
                continue

            try:
                if isinstance(raw_box[0], (list, tuple)):
                    xs = [float(pt[0]) for pt in raw_box]
                    ys = [float(pt[1]) for pt in raw_box]
                    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                elif len(raw_box) == 8:
                    xs = [float(raw_box[i]) for i in (0, 2, 4, 6)]
                    ys = [float(raw_box[i]) for i in (1, 3, 5, 7)]
                    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                elif len(raw_box) == 4:
                    x0, y0, x1, y1 = [float(c) for c in raw_box]
                else:
                    continue
            except Exception:
                continue

            norm_box = normalize_xyxy(x0, y0, x1, y1, width, height)
            seg_words, seg_boxes = split_segment_to_words_and_boxes(seg_text, norm_box)
            if not seg_words:
                continue

            canon = label_map.get(raw_label)
            if canon:
                entity_chunks[raw_label].append(seg_text)

            for idx_w, (w, b) in enumerate(zip(seg_words, seg_boxes)):
                words.append(w)
                boxes.append(b)
                if canon:
                    # Start a new BIO span on the first word whenever entity type changes
                    # (including after an intervening 'O' segment) to keep strict IOB2 transitions.
                    if idx_w == 0 and canon != prev_canon:
                        pre_labels.append(f"B-{canon}")
                    else:
                        pre_labels.append(f"I-{canon}")
                else:
                    pre_labels.append("O")

            prev_canon = canon

        if not words:
            self._record_excluded(receipt_id, "empty_ocr_tokens", "HF row yielded 0 words")
            return None

        ground_truth = {
            k: " ".join(v).strip() for k, v in entity_chunks.items() if v
        }

        return RawReceiptData(
            receipt_id=receipt_id,
            image_path=str(dest_path.resolve()),
            width=width,
            height=height,
            words=words,
            boxes=boxes,
            ground_truth=ground_truth,
            split=split_name,
            image_sha256=image_sha256,
            is_synthetic=False,
            source="hf_arvindrajan92_sroie",
            pre_assigned_labels=pre_labels,
        )

    def _parse_token_classified_row(
        self,
        row: Dict[str, Any],
        receipt_id: str,
        split_name: str,
        images_dir: Path,
        hf_label_names: Optional[List[str]],
    ) -> Optional[RawReceiptData]:
        """Parses datasets using words/bboxes/ner_tags + image or local image_path."""
        dest_path = images_dir / f"{receipt_id}.jpg"
        img_info = validate_and_inspect_image(dest_path)
        if img_info is None and row.get("image") is not None:
            img_info = self._save_pil_image(row.get("image"), dest_path)
        if img_info is None and row.get("image_path"):
            cand = Path(str(row["image_path"]))
            img_info = validate_and_inspect_image(cand)
            if img_info is not None:
                dest_path = cand

        if img_info is None:
            self._record_excluded(
                receipt_id,
                "missing_or_corrupted_image",
                f"No loadable image for HF row ({row.get('image_path', 'no path')})",
            )
            return None

        width, height, image_sha256 = img_info
        words = [str(w) for w in (row.get("words") or [])]
        raw_boxes = row.get("bboxes") or []
        raw_tags = row.get("ner_tags") or []
        if not words or len(words) != len(raw_boxes):
            self._record_excluded(receipt_id, "mismatched_words_boxes", "words/bboxes length mismatch")
            return None

        boxes: List[List[int]] = []
        for b in raw_boxes:
            if len(b) != 4:
                self._record_excluded(receipt_id, "invalid_bbox_format", str(b))
                return None
            # Clamped to 0-1000
            boxes.append([max(0, min(1000, int(c))) for c in b])

        pre_labels: Optional[List[str]] = None
        if raw_tags and len(raw_tags) == len(words):
            names = hf_label_names or LABEL_LIST
            pre_labels = [
                names[int(t)] if isinstance(t, int) and 0 <= int(t) < len(names) else "O"
                for t in raw_tags
            ]

        ground_truth: Dict[str, str] = {}
        if pre_labels:
            collected: Dict[str, List[str]] = {"company": [], "date": [], "address": [], "total": []}
            for w, l in zip(words, pre_labels):
                if "-" in l:
                    ent = l.split("-", 1)[1].lower()
                    if ent in collected:
                        collected[ent].append(w)
            ground_truth = {k: " ".join(v) for k, v in collected.items() if v}

        return RawReceiptData(
            receipt_id=str(row.get("id") or receipt_id),
            image_path=str(dest_path.resolve()),
            width=width,
            height=height,
            words=words,
            boxes=boxes,
            ground_truth=ground_truth,
            split=split_name,
            image_sha256=image_sha256,
            is_synthetic=False,
            source="hf_sroie_tokens",
            pre_assigned_labels=pre_labels,
        )

    def load_from_huggingface(
        self,
        dataset_name: str = HF_DEFAULT_DATASET,
        split: Optional[str] = None,
    ) -> List[RawReceiptData]:
        """
        Loads real SROIE receipt data from Hugging Face Hub.
        Requires a genuine downloadable dataset with real receipt images; raises RuntimeError on failure.
        """
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face 'datasets' package is required to download SROIE."
            ) from exc

        self.excluded_samples = []
        logger.info("Loading SROIE dataset '%s' from Hugging Face...", dataset_name)
        try:
            ds_dict = load_dataset(
                dataset_name,
                split=split,
                cache_dir=str(self.cache_dir / "hf_cache"),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Hugging Face dataset '{dataset_name}': {exc}"
            ) from exc

        splits_to_process = {split: ds_dict} if split else dict(ds_dict)
        images_dir = self.cache_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        samples: List[RawReceiptData] = []
        for split_name, ds in splits_to_process.items():
            norm_split = "validation" if split_name == "val" else split_name
            cols = set(getattr(ds, "column_names", []))
            hf_label_names = None
            features = getattr(ds, "features", {}) or {}
            if "ner_tags" in features and hasattr(features["ner_tags"], "feature"):
                hf_label_names = getattr(features["ner_tags"].feature, "names", None)

            for idx, row in enumerate(ds):
                rid = str(row.get("id") or f"hf_{norm_split}_{idx:04d}")
                if "ocr" in cols:
                    sample = self._parse_arvindrajan_row(row, rid, norm_split, images_dir)
                elif "words" in cols and "bboxes" in cols:
                    sample = self._parse_token_classified_row(
                        row, rid, norm_split, images_dir, hf_label_names
                    )
                else:
                    raise RuntimeError(
                        f"Unsupported dataset columns {sorted(cols)} in '{dataset_name}'."
                    )
                if sample is not None:
                    samples.append(sample)

        if not samples:
            raise RuntimeError(
                f"Dataset '{dataset_name}' yielded 0 valid receipts with verified images "
                f"(excluded={len(self.excluded_samples)}). Refusing to fall back to synthetic data."
            )

        logger.info(
            "Loaded %d verified HF receipts from '%s' (excluded=%d).",
            len(samples),
            dataset_name,
            len(self.excluded_samples),
        )
        return samples

    def generate_synthetic_samples(
        self, n_samples: int = 20, output_image_dir: Optional[Path] = None
    ) -> List[RawReceiptData]:
        """
        Generates synthetic SROIE-formatted receipts WITH rendered RGB PNG images
        strictly for unit tests and explicit CPU --smoke-test verification.
        Never used for real benchmarking or production training.
        """
        img_dir = Path(output_image_dir) if output_image_dir else (self.cache_dir / "synthetic_images")
        img_dir.mkdir(parents=True, exist_ok=True)

        templates = [
            ("99 SPEED MART S/B", "LOT P.T. 2811 JALAN ANGSA", "15/01/2018", "34.50"),
            ("MR. D.I.Y. (M) SDN BHD", "LOT 1851-A & 1851-B JALAN KPB 6", "22/04/2018", "12.90"),
            ("AEON CO. (M) BHD", "3RD FLR AEON TAMAN MALURI", "03/08/2018", "89.00"),
            ("STARBUCKS COFFEE", "GROUND FLOOR KLCC", "12/11/2018", "18.50"),
            ("WATSONS PERSONAL CARE", "NO 45 JALAN SULTAN ISMAIL", "19/02/2018", "42.30"),
            ("POPULAR BOOK CO", "LOT 4.01 LEVEL 4 PAVILION KL", "07/06/2018", "65.80"),
        ]

        width, height = 600, 900
        samples: List[RawReceiptData] = []
        for i in range(n_samples):
            comp, addr, dt, tot = templates[i % len(templates)]
            receipt_id = f"synthetic_{i:04d}"

            img = Image.new("RGB", (width, height), color=(250, 250, 248))
            draw = ImageDraw.Draw(img)

            # Vary layout/noise slightly by index so synthetic image hashes differ legitimately
            y_offset = 60 + (i % 5) * 8
            lines_with_y = [
                (comp, y_offset),
                (addr, y_offset + 65),
                (f"RECEIPT NO: {1000 + i}", y_offset + 140),
                (f"DATE: {dt}", y_offset + 215),
                ("ITEM A 10.00", y_offset + 300),
                (f"TOTAL RM {tot}", y_offset + 410),
                ("THANK YOU", y_offset + 520),
            ]

            words: List[str] = []
            boxes: List[List[int]] = []
            for line_text, y_px in lines_with_y:
                draw.text((40, y_px), line_text, fill=(20, 20, 20))
                line_words = line_text.split()
                x_px = 40
                for w in line_words:
                    w_px = max(24, len(w) * 14)
                    norm_box = normalize_xyxy(x_px, y_px, x_px + w_px, y_px + 28, width, height)
                    words.append(w)
                    boxes.append(norm_box)
                    x_px += w_px + 14

            img_path = img_dir / f"{receipt_id}.png"
            img.save(img_path, format="PNG")
            sha256 = _compute_file_sha256(img_path)

            samples.append(
                RawReceiptData(
                    receipt_id=receipt_id,
                    image_path=str(img_path.resolve()),
                    width=width,
                    height=height,
                    words=words,
                    boxes=boxes,
                    ground_truth={
                        "company": comp,
                        "address": addr,
                        "date": dt,
                        "total": tot,
                    },
                    image_sha256=sha256,
                    is_synthetic=True,
                    source="synthetic_smoke_test",
                )
            )
        return samples
