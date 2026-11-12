import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

from ml.ocr.engine import normalize_coordinates


@dataclass
class RawReceiptData:
    receipt_id: str
    words: List[str]
    boxes: List[List[int]]  # Normalized 0-1000 boxes [x0, y0, x1, y1]
    ground_truth: Dict[str, str]
    image_path: Optional[str] = None
    width: int = 1000
    height: int = 1000
    metadata: Dict[str, Any] = field(default_factory=dict)


def parse_quad_bbox(
    coords: Any,
    img_w: int = 1000,
    img_h: int = 1000,
    **kwargs: Any,
) -> Tuple[int, int, int, int]:
    """
    Parses SROIE 8-point polygon coordinate string 'x1,y1,x2,y2,x3,y3,x4,y4',
    4-point 'x0,y0,x1,y1', or list of ints, returning normalized [x0, y0, x1, y1] in 0-1000 space.
    """
    img_w = kwargs.get("img_width", img_w)
    img_h = kwargs.get("img_height", img_h)

    if isinstance(coords, str):
        parts = [int(float(c.strip())) for c in coords.split(",") if c.strip()]
    elif isinstance(coords, (list, tuple)):
        parts = [int(c) for c in coords]
    else:
        parts = [0, 0, 10, 10]

    if len(parts) >= 8:
        xs = [parts[0], parts[2], parts[4], parts[6]]
        ys = [parts[1], parts[3], parts[5], parts[7]]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
    elif len(parts) == 4:
        min_x, min_y, max_x, max_y = parts[0], parts[1], parts[2], parts[3]
    else:
        min_x, min_y, max_x, max_y = 0, 0, 10, 10

    w = max(1, max_x - min_x)
    h = max(1, max_y - min_y)

    norm_x0, norm_y0, norm_x1, norm_y1 = normalize_coordinates(
        min_x, min_y, w, h, img_w, img_h
    )
    return (
        min(norm_x0, norm_x1),
        min(norm_y0, norm_y1),
        max(norm_x0, norm_x1),
        max(norm_y0, norm_y1),
    )


class SROIELoader:
    """
    Loads and normalizes the SROIE benchmark dataset from local files,
    Hugging Face datasets hub, or generates deterministic mock samples for offline test suites.
    """

    def load_from_local_directory(self, base_dir: Path) -> List[RawReceiptData]:
        """
        Loads receipts from an SROIE directory containing:
        - img/ or images/
        - box/ or ocr/ (files containing bbox + text: 'x1,y1,x2,y2,x3,y3,x4,y4,text')
        - key/ (JSON files containing company, date, address, total)
        """
        base = Path(base_dir)
        img_dir = base / "images" if (base / "images").exists() else base / "img"
        box_dir = base / "box" if (base / "box").exists() else base / "ocr"
        key_dir = base / "key"

        if not (img_dir.exists() and key_dir.exists()):
            raise FileNotFoundError(f"SROIE directories not found in {base_dir}")

        receipts: List[RawReceiptData] = []

        key_files = sorted(list(key_dir.glob("*.json")))
        for key_file in key_files:
            receipt_id = key_file.stem
            try:
                with open(key_file, "r", encoding="utf-8") as f:
                    gt = json.load(f)
            except Exception:
                continue

            # Find matching image
            img_candidates = list(img_dir.glob(f"{receipt_id}.*"))
            if not img_candidates:
                continue
            img_path = img_candidates[0]

            try:
                with Image.open(img_path) as im:
                    img_w, img_h = im.size
            except Exception:
                img_w, img_h = 1000, 1000

            # Find matching OCR / Box file
            box_candidates = list(box_dir.glob(f"{receipt_id}.*")) if box_dir.exists() else []
            words: List[str] = []
            boxes: List[List[int]] = []

            if box_candidates:
                try:
                    with open(box_candidates[0], "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split(",")
                            if len(parts) >= 9:
                                coord_str = ",".join(parts[:8])
                                text = ",".join(parts[8:]).strip()
                                if text:
                                    norm_box = parse_quad_bbox(coord_str, img_w, img_h)
                                    words.append(text)
                                    boxes.append(list(norm_box))
                except Exception:
                    pass

            receipts.append(
                RawReceiptData(
                    receipt_id=receipt_id,
                    words=words,
                    boxes=boxes,
                    ground_truth=gt,
                    image_path=str(img_path),
                    width=img_w,
                    height=img_h,
                )
            )

        return receipts

    def load_from_huggingface(self, split: Optional[str] = None) -> List[RawReceiptData]:
        """Loads SROIE using Hugging Face datasets library."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("huggingface datasets library is required: pip install datasets")

        dataset = load_dataset("dpl/sroie", split=split)
        receipts: List[RawReceiptData] = []

        splits_to_iterate = [dataset] if split else dataset.values()

        for split_data in splits_to_iterate:
            for item in split_data:
                rec_id = str(item.get("id") or item.get("receipt_id") or "receipt")
                words = item.get("words", [])
                boxes = item.get("boxes", [])
                gt = {
                    "company": item.get("company", ""),
                    "date": item.get("date", ""),
                    "address": item.get("address", ""),
                    "total": str(item.get("total", "")),
                }

                norm_boxes = []
                for box in boxes:
                    norm_boxes.append(
                        [
                            max(0, min(1000, int(box[0]))),
                            max(0, min(1000, int(box[1]))),
                            max(0, min(1000, int(box[2]))),
                            max(0, min(1000, int(box[3]))),
                        ]
                    )

                receipts.append(
                    RawReceiptData(
                        receipt_id=rec_id,
                        words=words,
                        boxes=norm_boxes,
                        ground_truth=gt,
                    )
                )

        return receipts

    def generate_synthetic_samples(self, n_samples: int = 20) -> List[RawReceiptData]:
        """
        Generates realistic, deterministic synthetic SROIE receipts for offline testing,
        CI verification, and unit tests without external internet dependencies.
        """
        samples: List[RawReceiptData] = []

        merchants = [
            ("BOOK MART SDN BHD", "NO 12 JALAN BUKIT BINTANG, 55100 KUALA LUMPUR", "15/04/2018", "45.80"),
            ("AEON CO. (M) BHD", "LOT G25 GROUND FLOOR, 1 UTAMA, 47800 PJ", "22/07/2018", "128.50"),
            ("PETRONAS DAGANGAN", "KM 12 LEBUHRAYA PLUS, 43000 KAJANG", "05/09/2018", "70.00"),
            ("GUARDIAN HEALTH", "NO 5 GROUND FLOOR, JALAN MEDAN, IPOH", "18/11/2018", "34.20"),
        ]

        for i in range(n_samples):
            merchant, address, date_val, total_val = merchants[i % len(merchants)]
            rec_id = f"sroie_syn_{i:04d}"
            date_val = f"{(i % 28) + 1:02d}/05/2018"
            total_val = f"{float(total_val) + i:.2f}"

            # Create realistic word tokens and bounding boxes
            tokens = []
            boxes = []

            # Merchant header (top 150)
            for j, w in enumerate(merchant.split()):
                tokens.append(w)
                boxes.append([100 + j * 90, 80, 180 + j * 90, 120])

            # Address (130 - 240)
            addr_words = address.split()
            for j, w in enumerate(addr_words):
                row = j // 4
                col = j % 4
                tokens.append(w)
                boxes.append([80 + col * 120, 140 + row * 40, 190 + col * 120, 175 + row * 40])

            # Date (280)
            tokens.extend(["DATE:", date_val])
            boxes.extend([[100, 280, 180, 310], [200, 280, 350, 310]])

            # Items (350 - 500)
            tokens.extend(["ITEM", "QTY", "PRICE", "TOTAL"])
            boxes.extend([[80, 350, 140, 380], [200, 350, 250, 380], [300, 350, 370, 380], [420, 350, 490, 380]])

            tokens.extend(["BOOK", "1", "30.00", "30.00"])
            boxes.extend([[80, 400, 160, 430], [210, 400, 240, 430], [310, 400, 370, 430], [430, 400, 490, 430]])

            tokens.extend(["TAX", "GST", "6%", "2.60"])
            boxes.extend([[80, 450, 140, 480], [160, 450, 210, 480], [230, 450, 280, 480], [430, 450, 490, 480]])

            # Total (550)
            tokens.extend(["TOTAL", "AMOUNT", "RM", total_val])
            boxes.extend([[80, 540, 170, 580], [190, 540, 300, 580], [350, 540, 400, 580], [420, 540, 510, 580]])

            gt = {
                "company": merchant,
                "address": address,
                "date": date_val,
                "total": total_val,
            }

            samples.append(
                RawReceiptData(
                    receipt_id=rec_id,
                    words=tokens,
                    boxes=boxes,
                    ground_truth=gt,
                    width=1000,
                    height=1000,
                    metadata={"is_synthetic": True},
                )
            )

        return samples
