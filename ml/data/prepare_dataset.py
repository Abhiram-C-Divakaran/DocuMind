import argparse
import dataclasses
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ml.data.annotation_alignment import AlignedReceipt, align_receipt_annotations
from ml.data.dataset_validation import generate_dataset_validation_report
from ml.data.sroie_loader import HF_DEFAULT_DATASET, RawReceiptData, SROIELoader

logger = logging.getLogger("documind.ml.data.prepare_dataset")


def compute_receipt_hash(receipt: AlignedReceipt) -> str:
    """Computes a normalized text+entity hash to detect near/exact duplicate content across splits."""
    payload = (
        "|".join(w.lower() for w in receipt.words)
        + "||"
        + json.dumps(receipt.ground_truth, sort_keys=True)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_split_hash(receipts: List[AlignedReceipt]) -> str:
    """Computes a deterministic hash identifying the exact composition of a split."""
    entries = sorted(
        f"{r.receipt_id}:{r.image_sha256 or 'no_img_hash'}:{len(r.words)}"
        for r in receipts
    )
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


def check_data_leakage(
    train_receipts: List[AlignedReceipt],
    val_receipts: List[AlignedReceipt],
    test_receipts: List[AlignedReceipt],
) -> Dict[str, Any]:
    """
    Audits cross-split overlap using three independent signals:
    1. Receipt identifiers
    2. Image SHA-256 content hashes (exact duplicate scan detection)
    3. Normalized OCR text + ground-truth hashes
    """
    train_ids = {r.receipt_id for r in train_receipts}
    val_ids = {r.receipt_id for r in val_receipts}
    test_ids = {r.receipt_id for r in test_receipts}

    train_img = {r.image_sha256 for r in train_receipts if r.image_sha256}
    val_img = {r.image_sha256 for r in val_receipts if r.image_sha256}
    test_img = {r.image_sha256 for r in test_receipts if r.image_sha256}

    train_txt = {compute_receipt_hash(r) for r in train_receipts}
    val_txt = {compute_receipt_hash(r) for r in val_receipts}
    test_txt = {compute_receipt_hash(r) for r in test_receipts}

    report = {
        "train_val_id_overlaps": len(train_ids & val_ids),
        "train_test_id_overlaps": len(train_ids & test_ids),
        "val_test_id_overlaps": len(val_ids & test_ids),
        "train_val_image_hash_overlaps": len(train_img & val_img),
        "train_test_image_hash_overlaps": len(train_img & test_img),
        "val_test_image_hash_overlaps": len(val_img & test_img),
        "train_val_text_overlaps": len(train_txt & val_txt),
        "train_test_text_overlaps": len(train_txt & test_txt),
        "val_test_text_overlaps": len(val_txt & test_txt),
    }
    report["has_leakage"] = any(v > 0 for v in report.values())
    return report


def _save_jsonl(path: Path, receipts: List[AlignedReceipt]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in receipts:
            f.write(json.dumps(dataclasses.asdict(r), ensure_ascii=False) + "\n")


def load_aligned_receipts(jsonl_path: Path) -> List[AlignedReceipt]:
    receipts: List[AlignedReceipt] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            receipts.append(
                AlignedReceipt(
                    receipt_id=data["receipt_id"],
                    words=data["words"],
                    boxes=data["boxes"],
                    labels=data["labels"],
                    ner_tags=data["ner_tags"],
                    ground_truth=data.get("ground_truth", {}),
                    alignment_stats=data.get("alignment_stats", {}),
                    unmatched_fields=data.get("unmatched_fields", []),
                    image_path=data.get("image_path"),
                    width=data.get("width", 1000),
                    height=data.get("height", 1000),
                    image_sha256=data.get("image_sha256"),
                    is_synthetic=bool(data.get("is_synthetic", False)),
                    source=data.get("source", "unknown"),
                    alignment_audit=data.get("alignment_audit", {}),
                )
            )
    return receipts


def prepare_sroie_dataset(
    local_dir: Optional[str] = None,
    hf_dataset: Optional[str] = None,
    output_dir: str = "data/sroie_processed",
    seed: int = 42,
    allow_synthetic: bool = False,
    generate_synthetic_if_missing: Optional[bool] = None,
    cache_dir: Optional[str] = None,
    synthetic_count: int = 24,
) -> Dict[str, Any]:
    """
    End-to-end SROIE dataset ingestion, image validation, BIO alignment,
    reproducible split construction, leakage auditing, and report generation.

    By default (`allow_synthetic=False`), raises RuntimeError if genuine SROIE data
    cannot be loaded rather than silently substituting synthetic data.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    loader = SROIELoader(cache_dir=cache_dir or str(out_path.parent / "sroie_cache"))

    use_synthetic = allow_synthetic or bool(generate_synthetic_if_missing)

    raw_samples: List[RawReceiptData]
    if local_dir:
        raw_samples = loader.load_from_local_directory(Path(local_dir))
    elif use_synthetic and not hf_dataset:
        logger.warning(
            "Explicit synthetic dataset mode requested (%d samples) for smoke testing/tests.",
            synthetic_count,
        )
        raw_samples = loader.generate_synthetic_samples(
            n_samples=synthetic_count,
            output_image_dir=out_path / "synthetic_images",
        )
    else:
        target_ds = hf_dataset or HF_DEFAULT_DATASET
        raw_samples = loader.load_from_huggingface(target_ds)

    if not raw_samples:
        raise RuntimeError("No valid receipt samples loaded; aborting dataset preparation.")

    # Align entities and preserve raw split metadata
    aligned_pairs: List[Tuple[Optional[str], AlignedReceipt]] = []
    for raw in raw_samples:
        aligned = align_receipt_annotations(
            receipt_id=raw.receipt_id,
            words=raw.words,
            boxes=raw.boxes,
            ground_truth=raw.ground_truth,
            image_path=raw.image_path,
            width=raw.width,
            height=raw.height,
            image_sha256=raw.image_sha256,
            is_synthetic=raw.is_synthetic,
            source=raw.source,
            pre_assigned_labels=raw.pre_assigned_labels,
        )
        aligned_pairs.append((raw.split, aligned))

    # Deduplicate exact image hashes within single-split sources to avoid cross-split leakage
    has_official_test = any(sp == "test" for sp, _ in aligned_pairs)
    rng = random.Random(seed)

    if has_official_test:
        train_pool = [r for sp, r in aligned_pairs if sp in ("train", None)]
        val_receipts = [r for sp, r in aligned_pairs if sp == "validation"]
        test_receipts = [r for sp, r in aligned_pairs if sp == "test"]

        if not val_receipts and len(train_pool) >= 5:
            rng.shuffle(train_pool)
            n_val = max(1, int(0.15 * len(train_pool)))
            val_receipts = train_pool[:n_val]
            train_receipts = train_pool[n_val:]
        else:
            train_receipts = train_pool
    else:
        seen_hashes = set()
        deduped: List[AlignedReceipt] = []
        duplicate_count = 0
        for _, r in aligned_pairs:
            if r.image_sha256 and r.image_sha256 in seen_hashes:
                duplicate_count += 1
                continue
            if r.image_sha256:
                seen_hashes.add(r.image_sha256)
            deduped.append(r)
        if duplicate_count > 0:
            logger.info(
                "Removed %d duplicate-image receipts prior to train/val/test splitting.",
                duplicate_count,
            )

        rng.shuffle(deduped)
        n = len(deduped)
        n_train = max(1, int(0.80 * n)) if n >= 3 else max(1, n - 2)
        n_val = max(1, int(0.10 * n)) if n >= 5 else (1 if n >= 3 else 0)
        if n_train + n_val >= n and n >= 3:
            n_train = n - 2
            n_val = 1

        train_receipts = deduped[:n_train]
        val_receipts = deduped[n_train : n_train + n_val]
        test_receipts = deduped[n_train + n_val :]

    leakage = check_data_leakage(train_receipts, val_receipts, test_receipts)
    if leakage["has_leakage"]:
        logger.warning("Data leakage detected across splits: %s", leakage)

    _save_jsonl(out_path / "train.jsonl", train_receipts)
    _save_jsonl(out_path / "validation.jsonl", val_receipts)
    _save_jsonl(out_path / "test.jsonl", test_receipts)

    splits = {
        "train": train_receipts,
        "validation": val_receipts,
        "test": test_receipts,
    }
    report = generate_dataset_validation_report(
        splits=splits,
        output_dir=out_path,
        excluded_samples=loader.excluded_samples,
        leakage_report=leakage,
    )

    split_hashes = {name: compute_split_hash(recs) for name, recs in splits.items()}
    manifest = {
        "source": raw_samples[0].source if raw_samples else "unknown",
        "is_synthetic": any(r.is_synthetic for _, r in aligned_pairs),
        "has_official_test_split": has_official_test,
        "seed": seed,
        "counts": {k: len(v) for k, v in splits.items()},
        "split_hashes": split_hashes,
        "excluded_samples_count": len(loader.excluded_samples),
        "leakage": leakage,
    }
    with open(out_path / "dataset_manifest.json", "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)

    return {
        "output_dir": str(out_path),
        "counts": manifest["counts"],
        "split_hashes": split_hashes,
        "is_synthetic": manifest["is_synthetic"],
        "source": manifest["source"],
        "excluded_samples_count": len(loader.excluded_samples),
        "leakage": leakage,
        "validation_report": report,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(
        description="Prepare SROIE dataset splits for DocuMind LayoutLMv3"
    )
    parser.add_argument("--local-dir", type=str, default=None, help="Path to local SROIE directory")
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default=None,
        help=f"Hugging Face dataset ID (default: {HF_DEFAULT_DATASET})",
    )
    parser.add_argument("--output-dir", type=str, default="data/sroie_processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Explicitly generate synthetic smoke-test receipts with rendered PNGs",
    )
    args = parser.parse_args()

    res = prepare_sroie_dataset(
        local_dir=args.local_dir,
        hf_dataset=args.hf_dataset,
        output_dir=args.output_dir,
        seed=args.seed,
        allow_synthetic=args.allow_synthetic,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
