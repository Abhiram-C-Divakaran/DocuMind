"""
Checkpoint Import, Integrity Validation & Deployment Promotion Utility (Phase 4B)
--------------------------------------------------------------------------------
Validates a locally supplied LayoutLMv3 checkpoint directory or .zip bundle before
making it available to DocuMind:
1. Safely extracts .zip archives (with Zip Slip path-traversal protection) if supplied.
2. Verifies required config, tokenizer, weights, model_type == 'layoutlmv3', and exact
   9-class SROIE IOB2 id2label parity.
3. Reads checkpoint_metadata.json to detect synthetic/smoke-test checkpoints so they
   are NEVER silently promoted as production models.
4. Loads the model and runs a bounded 1-sample forward-pass + BIO decoder smoke test.
5. Updates deployment .env (LAYOUTLMV3_CHECKPOINT_DIR) ONLY after validation succeeds
   AND the checkpoint is confirmed non-synthetic (unless --allow-synthetic-smoke-check).

Usage:
    python -m ml.training.import_checkpoint --checkpoint path/to/layoutlmv3-sroie-best
    python -m ml.training.import_checkpoint --checkpoint bundle.zip --target-dir checkpoints/layoutlmv3-sroie-best --apply-env
"""

import argparse
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

from ml.data.annotation_alignment import ID2LABEL
from ml.evaluation.metrics import compute_checkpoint_disk_size_mb
from ml.extractors.layoutlmv3_extractor import (
    HAS_TORCH,
    LayoutLMv3Extractor,
    read_sanitized_checkpoint_metadata,
    validate_checkpoint_directory,
)
from ml.inference.decoder import decode_layoutlmv3_predictions
from ml.training.dataset import prepare_layoutlmv3_inputs


def _safe_extract_zip(archive_path: Path, dest_dir: Path) -> Path:
    """Safely unpacks a checkpoint zip archive while rejecting path traversal."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    resolved_dest = dest_dir.resolve()

    with zipfile.ZipFile(archive_path, "r") as zf:
        for member in zf.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe path detected inside zip archive: '{member.filename}'")
            target_path = (resolved_dest / member_path).resolve()
            if resolved_dest not in target_path.parents and target_path != resolved_dest:
                raise ValueError(f"Zip entry escapes target directory: '{member.filename}'")
        zf.extractall(resolved_dest)

    # Support archives wrapping a single top-level directory containing config.json
    if (resolved_dest / "config.json").exists():
        return resolved_dest
    subdirs = [p for p in resolved_dest.iterdir() if p.is_dir() and (p / "config.json").exists()]
    if len(subdirs) == 1:
        return subdirs[0]
    return resolved_dest


def validate_and_smoke_test_checkpoint(
    checkpoint_path: Path,
    *,
    target_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Validates checkpoint files, metadata provenance, model loadability, and bounded
    forward-pass compatibility. Does not download weights from external hubs.
    """
    report: Dict[str, Any] = {
        "source": str(checkpoint_path),
        "compatible": False,
        "production_ready": False,
        "is_synthetic": False,
        "runtime_available": bool(HAS_TORCH),
        "size_mb": None,
        "metadata": None,
        "checks": {},
        "smoke_test_passed": False,
        "errors": [],
    }

    if not checkpoint_path.exists():
        report["errors"].append(f"Checkpoint path does not exist: '{checkpoint_path}'")
        return report

    working_dir = checkpoint_path
    if checkpoint_path.is_file() and checkpoint_path.suffix.lower() == ".zip":
        unpack_dest = target_dir or Path("checkpoints") / checkpoint_path.stem
        try:
            working_dir = _safe_extract_zip(checkpoint_path, unpack_dest)
            report["checks"]["archive_extraction"] = "PASS"
        except Exception as exc:
            report["checks"]["archive_extraction"] = "FAIL"
            report["errors"].append(str(exc))
            return report
    elif target_dir is not None and checkpoint_path.resolve() != target_dir.resolve():
        # Copy directory only after structural inspection passes below
        pass

    if not working_dir.is_dir():
        report["errors"].append(f"Expected checkpoint directory, found '{working_dir}'")
        return report

    report["resolved_checkpoint_dir"] = str(working_dir)
    report["size_mb"] = compute_checkpoint_disk_size_mb(working_dir)
    meta = read_sanitized_checkpoint_metadata(working_dir)
    report["metadata"] = meta
    is_synthetic = bool((meta or {}).get("is_synthetic") or (meta or {}).get("smoke_test"))
    report["is_synthetic"] = is_synthetic

    structural_ok, structural_reason = validate_checkpoint_directory(working_dir)
    report["checks"]["structural_files_and_label_map"] = "PASS" if structural_ok else "FAIL"
    if not structural_ok:
        report["errors"].append(structural_reason)
        return report

    if not HAS_TORCH:
        report["checks"]["model_load"] = "BLOCKED"
        report["checks"]["bounded_inference_smoke_test"] = "BLOCKED"
        report["errors"].append(
            "PyTorch / Transformers runtime is unavailable on this host; structural checks passed only."
        )
        return report

    extractor = LayoutLMv3Extractor(checkpoint_dir=str(working_dir))
    try:
        extractor.load_model()
        tokenizer = extractor._tokenizer
        image_processor = extractor._image_processor
        model = extractor._model
        report["checks"]["model_load"] = "PASS"
    except Exception as exc:
        report["checks"]["model_load"] = "FAIL"
        report["errors"].append(f"Failed loading model weights: {exc}")
        return report

    try:
        import torch

        dummy_image = Image.new("RGB", (224, 224), color=(255, 255, 255))
        words = ["ACME", "MART", "TOTAL", "12.50"]
        boxes = [[50, 50, 220, 90], [230, 50, 410, 90], [50, 200, 180, 240], [220, 200, 360, 240]]
        encoded = prepare_layoutlmv3_inputs(
            image=dummy_image,
            words=words,
            boxes=boxes,
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_length=64,
            return_word_ids=True,
        )
        model_inputs = {
            k: v.unsqueeze(0).to(extractor._device)
            for k, v in encoded.items()
            if k in ("input_ids", "attention_mask", "bbox", "pixel_values")
        }
        with torch.no_grad():
            outputs = model(**model_inputs)
            logits = outputs.logits[0]
            if logits.shape[-1] != len(ID2LABEL):
                raise ValueError(f"Expected {len(ID2LABEL)} output classes, got {logits.shape[-1]}")
            probs = torch.softmax(logits, dim=-1)
            pred_ids = probs.argmax(dim=-1).tolist()
            confidences = probs.max(dim=-1).values.tolist()

        decode_layoutlmv3_predictions(
            words=words,
            boxes=boxes,
            word_ids=encoded["word_ids"],
            pred_label_ids=pred_ids,
            token_confidences=confidences,
        )
        report["checks"]["bounded_inference_smoke_test"] = "PASS"
        report["smoke_test_passed"] = True
        report["compatible"] = True
        report["production_ready"] = not is_synthetic
        if is_synthetic:
            report["errors"].append(
                "Checkpoint metadata marks is_synthetic/smoke_test=true; valid for integration smoke checks only, NOT production-ready."
            )
    except Exception as exc:
        report["checks"]["bounded_inference_smoke_test"] = "FAIL"
        report["errors"].append(f"Bounded inference smoke test failed: {exc}")
        return report

    if (
        target_dir is not None
        and checkpoint_path.is_dir()
        and checkpoint_path.resolve() != target_dir.resolve()
        and report["compatible"]
    ):
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(working_dir, target_dir)
        report["installed_dir"] = str(target_dir)

    return report


def update_env_checkpoint_dir(env_path: Path, checkpoint_dir: str) -> None:
    """Safely updates LAYOUTLMV3_CHECKPOINT_DIR in an existing .env file."""
    normalized = checkpoint_dir.replace("\\", "/")
    line_value = f"LAYOUTLMV3_CHECKPOINT_DIR={normalized}\n"
    if not env_path.exists():
        env_path.write_text(line_value, encoding="utf-8")
        return
    content = env_path.read_text(encoding="utf-8")
    pattern = re.compile(r"^LAYOUTLMV3_CHECKPOINT_DIR=.*$", flags=re.MULTILINE)
    if pattern.search(content):
        updated = pattern.sub(line_value.rstrip("\n"), content)
    else:
        updated = content.rstrip("\n") + "\n" + line_value
    env_path.write_text(updated if updated.endswith("\n") else updated + "\n", encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and optionally register a trained LayoutLMv3 checkpoint for DocuMind."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory or .zip bundle")
    parser.add_argument(
        "--target-dir",
        default=None,
        help="Optional target directory under checkpoints/ to install validated weights",
    )
    parser.add_argument(
        "--apply-env",
        action="store_true",
        help="Update LAYOUTLMV3_CHECKPOINT_DIR in .env only if validation AND non-synthetic checks pass",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to deployment .env file (default: .env)",
    )
    args = parser.parse_args(argv)

    src = Path(args.checkpoint)
    dest = Path(args.target_dir) if args.target_dir else None
    report = validate_and_smoke_test_checkpoint(src, target_dir=dest)

    if args.apply_env:
        if report["production_ready"]:
            chosen_dir = report.get("installed_dir") or report.get("resolved_checkpoint_dir") or str(src)
            update_env_checkpoint_dir(Path(args.env_file), chosen_dir)
            report["env_updated"] = str(args.env_file)
        else:
            report["env_updated"] = False
            report["errors"].append(
                "Refused updating .env because checkpoint is unverified, incompatible, or synthetic."
            )

    print(json.dumps(report, indent=2))
    return 0 if report["production_ready"] else (2 if report["compatible"] else 1)


if __name__ == "__main__":
    sys.exit(main())

