import argparse
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except (ImportError, OSError):
    torch = None  # type: ignore
    DataLoader = Any  # type: ignore
    HAS_TORCH = False

try:
    from transformers import (
        AutoImageProcessor,
        AutoProcessor,
        AutoTokenizer,
        LayoutLMv3ForTokenClassification,
        get_linear_schedule_with_warmup,
    )
    HAS_TRANSFORMERS = True
except (ImportError, OSError):
    AutoImageProcessor = None  # type: ignore
    AutoProcessor = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    LayoutLMv3ForTokenClassification = None  # type: ignore
    get_linear_schedule_with_warmup = None  # type: ignore
    HAS_TRANSFORMERS = False

from ml.data.annotation_alignment import ID2LABEL, LABEL2ID, LABEL_LIST, AlignedReceipt
from ml.data.prepare_dataset import (
    compute_split_hash,
    load_aligned_receipts,
    prepare_sroie_dataset,
)
from ml.evaluation.metrics import compute_seqeval_metrics
from ml.training.config import LayoutLMv3TrainingConfig
from ml.training.dataset import SROIELayoutLMv3Dataset, collate_fn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("DocuMind.TrainLayoutLMv3")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH and torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _resolve_split_files(data_dir: Path) -> Tuple[Path, Path]:
    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "validation.jsonl"
    if not val_file.exists() and (data_dir / "val.jsonl").exists():
        val_file = data_dir / "val.jsonl"
    return train_file, val_file


def load_training_splits(config: LayoutLMv3TrainingConfig) -> Tuple[List[AlignedReceipt], List[AlignedReceipt], Dict[str, Any]]:
    """
    Loads prepared SROIE splits for training.
    If processed manifests are missing:
    - In smoke_test mode: explicitly prepares synthetic smoke data with rendered PNGs.
    - In full training mode: fails loudly instead of silently generating synthetic data.
    """
    data_dir = Path(config.data_dir)
    train_file, val_file = _resolve_split_files(data_dir)

    if not train_file.exists() or not val_file.exists():
        if config.smoke_test:
            logger.warning(
                "Smoke test processed splits missing at %s; preparing explicit synthetic smoke dataset.",
                data_dir,
            )
            prepare_sroie_dataset(
                output_dir=str(data_dir),
                seed=config.seed,
                allow_synthetic=True,
                synthetic_count=max(8, config.max_smoke_samples * 3),
            )
            train_file, val_file = _resolve_split_files(data_dir)
        else:
            raise FileNotFoundError(
                f"Real processed SROIE training splits not found at '{data_dir}' "
                "(expected train.jsonl and validation.jsonl). "
                "Run `python -m ml.data.prepare_dataset --output-dir data/sroie_processed` first. "
                "Full training refuses to auto-generate synthetic data."
            )

    train_receipts = load_aligned_receipts(train_file)
    val_receipts = load_aligned_receipts(val_file)

    if not config.smoke_test:
        synthetic_count = sum(1 for r in (train_receipts + val_receipts) if r.is_synthetic)
        if synthetic_count > 0:
            raise RuntimeError(
                f"Refusing full LayoutLMv3 training: {synthetic_count} synthetic records found in '{data_dir}'. "
                "Prepare genuine SROIE data (`allow_synthetic=False`) or pass --smoke-test explicitly."
            )

    manifest: Dict[str, Any] = {}
    manifest_path = data_dir / "dataset_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as mf:
            manifest = json.load(mf)

    return train_receipts, val_receipts, manifest


def evaluate_model(
    model: Any,
    dataloader: Any,
    device: str,
) -> Tuple[float, Dict[str, Any]]:
    """Evaluates the model on a validation dataloader, computing loss and seqeval BIO metrics."""
    if not HAS_TORCH or torch is None:
        raise RuntimeError("PyTorch is required for model evaluation.")

    model.eval()
    total_loss = 0.0
    all_preds: List[List[str]] = []
    all_refs: List[List[str]] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bbox = batch["bbox"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
                labels=labels,
            )

            total_loss += float(outputs.loss.item())
            logits = outputs.logits.detach().cpu().numpy()
            label_ids = labels.detach().cpu().numpy()
            pred_ids = np.argmax(logits, axis=-1)

            for b_idx in range(len(label_ids)):
                seq_preds: List[str] = []
                seq_refs: List[str] = []
                for t_idx in range(len(label_ids[b_idx])):
                    if label_ids[b_idx, t_idx] != -100:
                        seq_refs.append(ID2LABEL.get(int(label_ids[b_idx, t_idx]), "O"))
                        seq_preds.append(ID2LABEL.get(int(pred_ids[b_idx, t_idx]), "O"))
                all_preds.append(seq_preds)
                all_refs.append(seq_refs)

    avg_loss = total_loss / max(1, len(dataloader))
    metrics = compute_seqeval_metrics(all_preds, all_refs)
    metrics["loss"] = round(avg_loss, 4)
    return avg_loss, metrics


def _save_checkpoint(
    target_dir: Path,
    model: Any,
    tokenizer: Any,
    image_processor: Any,
    metadata: Dict[str, Any],
    optimizer: Any = None,
    scheduler: Any = None,
    scaler: Any = None,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(target_dir))
    if tokenizer is not None:
        tokenizer.save_pretrained(str(target_dir))
    if image_processor is not None and hasattr(image_processor, "save_pretrained"):
        image_processor.save_pretrained(str(target_dir))

    with open(target_dir / "checkpoint_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if HAS_TORCH and torch is not None and optimizer is not None and scheduler is not None:
        state = {
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": metadata.get("epoch", 0),
            "global_step": metadata.get("global_step", 0),
            "best_val_f1": metadata.get("best_val_f1", 0.0),
            "epochs_without_improvement": metadata.get("epochs_without_improvement", 0),
        }
        if scaler is not None:
            state["scaler_state_dict"] = scaler.state_dict()
        torch.save(state, str(target_dir / "training_state.pt"))


def train_layoutlmv3(config: LayoutLMv3TrainingConfig) -> Dict[str, Any]:
    """
    Reproducible LayoutLMv3 fine-tuning with fixed seed, CUDA AMP fp16,
    gradient accumulation, early stopping, validation F1 best-checkpoint selection,
    resume support, and full checkpoint metadata auditing.
    """
    _seed_everything(config.seed)
    train_receipts, val_receipts, manifest = load_training_splits(config)

    if not HAS_TORCH or not HAS_TRANSFORMERS or torch is None:
        raise RuntimeError(
            "PyTorch and Hugging Face Transformers are required to run LayoutLMv3 training."
        )

    if config.smoke_test:
        logger.info("Smoke test active: capping splits to %d receipts.", config.max_smoke_samples)
        train_receipts = train_receipts[: config.max_smoke_samples]
        val_receipts = val_receipts[: max(1, min(len(val_receipts), config.max_smoke_samples))]

    model_source = config.resume_from_checkpoint or config.model_name_or_path
    logger.info(
        "Starting LayoutLMv3 training on device=%s | source=%s | train=%d | val=%d",
        config.device,
        model_source,
        len(train_receipts),
        len(val_receipts),
    )

    processor = None
    try:
        processor = AutoProcessor.from_pretrained(model_source, apply_ocr=False)
        tokenizer = processor.tokenizer
        image_processor = getattr(processor, "image_processor", None)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_source, add_prefix_space=True)
        try:
            image_processor = AutoImageProcessor.from_pretrained(model_source, apply_ocr=False)
        except Exception:
            image_processor = None

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        model_source,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(config.device)

    train_dataset = SROIELayoutLMv3Dataset(
        receipts=train_receipts,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        image_processor=image_processor,
        image_size=config.image_size,
    )
    val_dataset = SROIELayoutLMv3Dataset(
        receipts=val_receipts,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        image_processor=image_processor,
        image_size=config.image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
    )

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.learning_rate,
        eps=config.adam_epsilon,
    )

    num_update_steps_per_epoch = max(1, len(train_loader) // max(1, config.gradient_accumulation_steps))
    total_training_steps = (
        config.smoke_test_steps if config.smoke_test else max(1, num_update_steps_per_epoch * config.num_epochs)
    )
    warmup_steps = int(total_training_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps
    )

    use_amp = bool(config.fp16 and str(config.device).startswith("cuda") and torch.cuda.is_available())
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if hasattr(torch, "amp") else None

    start_epoch = 0
    global_step = 0
    best_val_f1 = -1.0
    epochs_without_improvement = 0

    if config.resume_from_checkpoint:
        state_file = Path(config.resume_from_checkpoint) / "training_state.pt"
        if state_file.exists():
            ckpt_state = torch.load(str(state_file), map_location=config.device)
            optimizer.load_state_dict(ckpt_state["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt_state["scheduler_state_dict"])
            if scaler is not None and "scaler_state_dict" in ckpt_state:
                scaler.load_state_dict(ckpt_state["scaler_state_dict"])
            start_epoch = int(ckpt_state.get("epoch", 0))
            global_step = int(ckpt_state.get("global_step", 0))
            best_val_f1 = float(ckpt_state.get("best_val_f1", -1.0))
            epochs_without_improvement = int(ckpt_state.get("epochs_without_improvement", 0))
            logger.info("Resumed training state from %s (epoch=%d, step=%d)", state_file, start_epoch, global_step)

    output_dir = Path(config.output_dir)
    best_dir = Path(config.best_model_dir)
    history: List[Dict[str, Any]] = []
    num_epochs = 1 if config.smoke_test else config.num_epochs

    split_hashes = manifest.get("split_hashes") or {
        "train": compute_split_hash(train_receipts),
        "validation": compute_split_hash(val_receipts),
    }

    last_val_metrics: Dict[str, Any] = {}
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            bbox = batch["bbox"].to(config.device)
            pixel_values = batch["pixel_values"].to(config.device)
            labels = batch["labels"].to(config.device)

            if use_amp and hasattr(torch, "amp"):
                with torch.amp.autocast("cuda", enabled=True):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        bbox=bbox,
                        pixel_values=pixel_values,
                        labels=labels,
                    )
                    loss = outputs.loss / max(1, config.gradient_accumulation_steps)
                scaler.scale(loss).backward()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=bbox,
                    pixel_values=pixel_values,
                    labels=labels,
                )
                loss = outputs.loss / max(1, config.gradient_accumulation_steps)
                loss.backward()

            running_loss += float(loss.item() * max(1, config.gradient_accumulation_steps))

            is_accum_boundary = (step + 1) % max(1, config.gradient_accumulation_steps) == 0
            is_last_batch = (step + 1) == len(train_loader)
            if is_accum_boundary or is_last_batch:
                if use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if config.smoke_test and global_step >= config.smoke_test_steps:
                    break

        avg_train_loss = running_loss / max(1, step + 1)
        val_loss, val_metrics = evaluate_model(model, val_loader, config.device)
        last_val_metrics = val_metrics
        val_f1 = float(val_metrics.get("overall_f1", 0.0))

        epoch_record = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_f1": round(val_f1, 4),
            "val_metrics": val_metrics,
        }
        history.append(epoch_record)
        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_f1=%.4f",
            epoch + 1,
            num_epochs,
            avg_train_loss,
            val_loss,
            val_f1,
        )

        common_meta = {
            "base_model": config.model_name_or_path,
            "label_list": LABEL_LIST,
            "seed": config.seed,
            "epoch": epoch + 1,
            "global_step": global_step,
            "device": str(config.device),
            "fp16_amp": use_amp,
            "is_synthetic": bool(config.smoke_test or manifest.get("is_synthetic", False)),
            "smoke_test": config.smoke_test,
            "dataset_source": manifest.get("source", "local_or_processed"),
            "split_hashes": split_hashes,
            "val_metrics": val_metrics,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            best_meta = {**common_meta, "best_val_f1": round(best_val_f1, 4), "epochs_without_improvement": 0}
            _save_checkpoint(
                best_dir, model, tokenizer, image_processor, best_meta, optimizer, scheduler, scaler
            )
            logger.info("Saved new best checkpoint to %s (val_f1=%.4f)", best_dir, best_val_f1)
        else:
            epochs_without_improvement += 1

        last_meta = {
            **common_meta,
            "best_val_f1": round(best_val_f1, 4),
            "epochs_without_improvement": epochs_without_improvement,
        }
        _save_checkpoint(
            output_dir / "last", model, tokenizer, image_processor, last_meta, optimizer, scheduler, scaler
        )

        if not config.smoke_test and epochs_without_improvement >= config.early_stopping_patience:
            logger.info(
                "Early stopping triggered after %d epochs without val F1 improvement.",
                epochs_without_improvement,
            )
            break

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as hf:
        json.dump(history, hf, indent=2)

    return {
        "status": "completed",
        "smoke_test": config.smoke_test,
        "best_model_dir": str(best_dir),
        "best_val_f1": round(max(0.0, best_val_f1), 4),
        "last_val_metrics": last_val_metrics,
        "epochs_completed": len(history),
        "global_step": global_step,
        "is_synthetic": bool(config.smoke_test or manifest.get("is_synthetic", False)),
        "split_hashes": split_hashes,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LayoutLMv3 on SROIE dataset")
    parser.add_argument("--data-dir", type=str, default="data/sroie_processed")
    parser.add_argument("--output-dir", type=str, default="checkpoints/layoutlmv3-sroie")
    parser.add_argument("--best-model-dir", type=str, default="checkpoints/layoutlmv3-sroie-best")
    parser.add_argument("--model-name", type=str, default="microsoft/layoutlmv3-base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = LayoutLMv3TrainingConfig(
        model_name_or_path=args.model_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        best_model_dir=args.best_model_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
        seed=args.seed,
        early_stopping_patience=args.patience,
        fp16=not args.no_fp16,
        resume_from_checkpoint=args.resume_from_checkpoint,
        smoke_test=args.smoke_test,
    )
    summary = train_layoutlmv3(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
