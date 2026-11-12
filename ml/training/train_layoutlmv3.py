import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    LayoutLMv3ForTokenClassification,
    get_linear_schedule_with_warmup,
)

from ml.data.annotation_alignment import ID2LABEL, LABEL2ID, LABEL_LIST, AlignedReceipt
from ml.data.prepare_dataset import load_aligned_receipts, prepare_sroie_dataset
from ml.evaluation.metrics import compute_seqeval_metrics
from ml.training.config import LayoutLMv3TrainingConfig
from ml.training.dataset import SROIELayoutLMv3Dataset, collate_fn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("DocuMind.TrainLayoutLMv3")


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
) -> Tuple[float, Dict[str, Any]]:
    """Evaluates the model on validation dataloader, computing loss and seqeval metrics."""
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

            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits.detach().cpu().numpy()
            label_ids = labels.detach().cpu().numpy()
            pred_ids = np.argmax(logits, axis=-1)

            # Convert to BIO tag strings, ignoring -100 subwords and padding
            for b_idx in range(len(label_ids)):
                seq_preds: List[str] = []
                seq_refs: List[str] = []
                for t_idx in range(len(label_ids[b_idx])):
                    if label_ids[b_idx, t_idx] != -100:
                        ref_label = ID2LABEL.get(int(label_ids[b_idx, t_idx]), "O")
                        pred_label = ID2LABEL.get(int(pred_ids[b_idx, t_idx]), "O")
                        seq_refs.append(ref_label)
                        seq_preds.append(pred_label)

                all_preds.append(seq_preds)
                all_refs.append(seq_refs)

    avg_loss = total_loss / max(1, len(dataloader))
    metrics = compute_seqeval_metrics(all_preds, all_refs)
    metrics["loss"] = round(avg_loss, 4)
    return avg_loss, metrics


def train_layoutlmv3(config: LayoutLMv3TrainingConfig) -> Dict[str, Any]:
    """Runs LayoutLMv3 training or smoke test with checkpointing and metric logging."""
    logger.info("Initializing LayoutLMv3 training with device: %s", config.device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    data_dir = Path(config.data_dir)
    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"

    if not train_file.exists() or not val_file.exists():
        logger.info("Processed dataset not found at %s. Running prepare_sroie_dataset...", data_dir)
        prepare_sroie_dataset(output_dir=data_dir, generate_synthetic_if_missing=True)

    train_receipts = load_aligned_receipts(train_file)
    val_receipts = load_aligned_receipts(val_file)

    if config.smoke_test:
        logger.info("Smoke test active: constraining dataset to %d samples", config.max_smoke_samples)
        train_receipts = train_receipts[: config.max_smoke_samples]
        val_receipts = val_receipts[: config.max_smoke_samples]

    logger.info("Loaded %d train receipts and %d val receipts", len(train_receipts), len(val_receipts))

    # 1. Load Tokenizer & Model
    logger.info("Loading tokenizer and model: %s", config.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, add_prefix_space=True)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        config.model_name_or_path,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(config.device)

    # 2. Datasets and Loaders
    train_dataset = SROIELayoutLMv3Dataset(
        receipts=train_receipts,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
    )
    val_dataset = SROIELayoutLMv3Dataset(
        receipts=val_receipts,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
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

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=config.adam_epsilon,
    )

    total_training_steps = (len(train_loader) // config.gradient_accumulation_steps) * config.num_epochs
    if config.smoke_test:
        total_training_steps = config.smoke_test_steps

    warmup_steps = int(total_training_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(1, total_training_steps),
    )

    output_dir = Path(config.output_dir)
    best_model_dir = Path(config.best_model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)

    training_logs: List[Dict[str, Any]] = []
    best_val_f1 = -1.0
    global_step = 0

    logger.info("Starting training loop (Total epochs: %d, Target steps: %d)", config.num_epochs, total_training_steps)

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        step_in_epoch = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            bbox = batch["bbox"].to(config.device)
            pixel_values = batch["pixel_values"].to(config.device)
            labels = batch["labels"].to(config.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
                labels=labels,
            )

            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()
            epoch_loss += loss.item() * config.gradient_accumulation_steps

            if (step + 1) % config.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                step_in_epoch += 1

            if config.smoke_test and global_step >= config.smoke_test_steps:
                logger.info("Smoke test target step reached (%d steps). Breaking training loop.", global_step)
                break

        # Validation step
        val_loss, val_metrics = evaluate_model(model, val_loader, config.device)
        logger.info(
            "Epoch %d finished | Train Loss: %.4f | Val Loss: %.4f | Val F1: %.4f (Prec: %.4f, Rec: %.4f)",
            epoch + 1,
            epoch_loss / max(1, step_in_epoch or 1),
            val_loss,
            val_metrics["overall_f1"],
            val_metrics["overall_precision"],
            val_metrics["overall_recall"],
        )

        log_entry = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss": round(epoch_loss / max(1, step_in_epoch or 1), 4),
            "val_loss": val_loss,
            "val_f1": val_metrics["overall_f1"],
            "val_precision": val_metrics["overall_precision"],
            "val_recall": val_metrics["overall_recall"],
            "entity_metrics": val_metrics["entities"],
        }
        training_logs.append(log_entry)

        # Save best model
        if val_metrics["overall_f1"] >= best_val_f1 or config.smoke_test:
            best_val_f1 = val_metrics["overall_f1"]
            logger.info("Saving new best model checkpoint to %s (Val F1: %.4f)", best_model_dir, best_val_f1)
            model.save_pretrained(str(best_model_dir))
            tokenizer.save_pretrained(str(best_model_dir))

        if config.smoke_test:
            break

    # Persist training logs
    log_file = output_dir / "training_log.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": config.to_dict(),
                "best_val_f1": best_val_f1,
                "history": training_logs,
            },
            f,
            indent=2,
        )
    logger.info("Training complete. Training logs saved to %s", log_file)

    return {
        "best_val_f1": best_val_f1,
        "best_model_dir": str(best_model_dir),
        "history": training_logs,
    }


def main():
    parser = argparse.ArgumentParser(description="DocuMind LayoutLMv3 SROIE Fine-Tuning CLI")
    parser.add_argument("--data-dir", type=str, default="data/sroie_processed", help="Path to processed SROIE dataset")
    parser.add_argument("--output-dir", type=str, default="checkpoints/layoutlmv3-sroie", help="Output directory")
    parser.add_argument("--best-model-dir", type=str, default="checkpoints/layoutlmv3-sroie-best", help="Best checkpoint directory")
    parser.add_argument("--model-name", type=str, default="microsoft/layoutlmv3-base", help="Hugging Face model ID")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast 2-step verification smoke test")

    args = parser.parse_args()

    config = LayoutLMv3TrainingConfig(
        model_name_or_path=args.model_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        best_model_dir=args.best_model_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        smoke_test=args.smoke_test,
    )

    results = train_layoutlmv3(config)
    print("\n=== Training Summary ===")
    print(f"Best Val F1: {results['best_val_f1']:.4f}")
    print(f"Model saved to: {results['best_model_dir']}")


if __name__ == "__main__":
    main()
