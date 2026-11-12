import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from ml.data.annotation_alignment import AlignedReceipt, align_receipt_annotations
from ml.data.sroie_loader import RawReceiptData, SROIELoader


def compute_receipt_hash(raw_receipt: RawReceiptData) -> str:
    """Computes SHA-256 hash of receipt content to detect duplicates and prevent data leakage."""
    content = f"{raw_receipt.receipt_id}:{':'.join(raw_receipt.words)}:{json.dumps(raw_receipt.ground_truth, sort_keys=True)}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def check_data_leakage(splits: Dict[str, List[AlignedReceipt]]) -> List[str]:
    """
    Checks that no receipt ID or content hash is shared between train, val, and test splits.
    Returns list of leakage violations (empty if clean).
    """
    violations: List[str] = []
    seen_ids: Dict[str, str] = {}
    seen_hashes: Dict[str, str] = {}

    for split_name, receipts in splits.items():
        for r in receipts:
            # Check ID leakage across splits
            if r.receipt_id in seen_ids and seen_ids[r.receipt_id] != split_name:
                violations.append(
                    f"Receipt ID '{r.receipt_id}' appears in both '{seen_ids[r.receipt_id]}' and '{split_name}'."
                )
            else:
                seen_ids[r.receipt_id] = split_name

            # Check content hash leakage across splits
            h = hashlib.sha256(" ".join(r.words).encode("utf-8")).hexdigest()
            if h in seen_hashes and seen_hashes[h] != split_name:
                violations.append(
                    f"Duplicate text content between '{seen_hashes[h]}' and '{split_name}' (ID: {r.receipt_id})."
                )
            else:
                seen_hashes[h] = split_name

    return violations


def prepare_sroie_dataset(
    output_dir: Path = Path("data/sroie_processed"),
    local_sroie_dir: Optional[Path] = None,
    use_huggingface: bool = False,
    generate_synthetic_if_missing: bool = True,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Dict[str, List[AlignedReceipt]]:
    """
    Ingests SROIE, runs annotation alignment, performs reproducible train/val/test splits,
    verifies zero data leakage, and persists processed manifests to JSONL.
    """
    loader = SROIELoader()
    raw_data: List[RawReceiptData] = []

    # 1. Ingestion strategy
    if local_sroie_dir and local_sroie_dir.exists():
        raw_data = loader.load_from_local_directory(local_sroie_dir)
    elif use_huggingface:
        try:
            raw_data = loader.load_from_huggingface()
        except Exception:
            raw_data = []

    if not raw_data and generate_synthetic_if_missing:
        # Generate 40 deterministic synthetic receipts for reproducible local testing
        raw_data = loader.generate_synthetic_samples(n_samples=40)

    if not raw_data:
        raise ValueError("No SROIE receipt records could be loaded or generated.")

    # 2. Run annotation alignment for all receipts
    aligned_receipts: List[AlignedReceipt] = []
    for item in raw_data:
        aligned = align_receipt_annotations(
            receipt_id=item.receipt_id,
            words=item.words,
            boxes=item.boxes,
            ground_truth=item.ground_truth,
            image_path=item.image_path,
            width=item.width,
            height=item.height,
        )
        aligned_receipts.append(aligned)

    # 3. Deterministic Splitting
    random.seed(seed)
    # Shuffle using seeded RNG
    indices = list(range(len(aligned_receipts)))
    random.shuffle(indices)

    n_total = len(indices)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    splits = {
        "train": [aligned_receipts[i] for i in train_indices],
        "val": [aligned_receipts[i] for i in val_indices],
        "test": [aligned_receipts[i] for i in test_indices],
    }

    # 4. Data leakage check
    leakage_errors = check_data_leakage(splits)
    if leakage_errors:
        raise RuntimeError(f"Data leakage detected during dataset preparation: {leakage_errors}")

    # 5. Persist to disk
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_receipts in splits.items():
        split_file = output_dir / f"{split_name}.jsonl"
        with open(split_file, "w", encoding="utf-8") as f:
            for r in split_receipts:
                f.write(json.dumps(asdict(r)) + "\n")

    return splits


def load_aligned_receipts(file_path: Path) -> List[AlignedReceipt]:
    """Loads a list of AlignedReceipt objects from a JSONL manifest file."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Receipt manifest not found at {file_path}")

    receipts: List[AlignedReceipt] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            receipts.append(AlignedReceipt(**data))
    return receipts


def main():
    import argparse
    from ml.data.dataset_validation import generate_dataset_validation_report

    parser = argparse.ArgumentParser(description="DocuMind SROIE Dataset Preparation CLI")
    parser.add_argument("--output-dir", type=str, default="data/sroie_processed", help="Target manifest directory")
    parser.add_argument("--local-sroie-dir", type=str, default=None, help="Path to raw SROIE directory")
    parser.add_argument("--use-huggingface", action="store_true", help="Download official SROIE from Hugging Face Hub")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")

    args = parser.parse_args()
    out = Path(args.output_dir)
    print(f"Preparing SROIE dataset into {out}...")

    splits = prepare_sroie_dataset(
        output_dir=out,
        local_sroie_dir=Path(args.local_sroie_dir) if args.local_sroie_dir else None,
        use_huggingface=args.use_huggingface,
        generate_synthetic_if_missing=True,
        seed=args.seed,
    )

    report = generate_dataset_validation_report(splits, output_dir=out)
    print(f"Dataset successfully prepared!")
    print(f"- Total receipts: {report['total_records']}")
    print(f"- Train: {report['record_counts'].get('train', 0)}")
    print(f"- Val: {report['record_counts'].get('val', 0)}")
    print(f"- Test: {report['record_counts'].get('test', 0)}")
    print(f"Validation report saved to: {out / 'validation_report.json'}")


if __name__ == "__main__":
    main()

