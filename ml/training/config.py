from dataclasses import dataclass, field
from typing import Optional

try:
    import torch

    HAS_TORCH = True
except (ImportError, OSError):
    torch = None  # type: ignore
    HAS_TORCH = False


def detect_default_device() -> str:
    """Selects best available compute backend: CUDA > Apple MPS > CPU."""
    if not HAS_TORCH or torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


@dataclass
class LayoutLMv3TrainingConfig:
    """Configuration hyperparameters for fine-tuning LayoutLMv3 on SROIE."""

    model_name_or_path: str = "microsoft/layoutlmv3-base"
    data_dir: str = "data/sroie_processed"
    output_dir: str = "checkpoints/layoutlmv3-sroie"
    best_model_dir: str = "checkpoints/layoutlmv3-sroie-best"

    max_seq_length: int = 512
    image_size: int = 224

    batch_size: int = 2
    eval_batch_size: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 4
    early_stopping_patience: int = 3

    seed: int = 42
    device: str = field(default_factory=detect_default_device)
    num_workers: int = 0
    fp16: bool = True

    logging_steps: int = 10
    eval_steps: int = 50
    save_total_limit: int = 2
    resume_from_checkpoint: Optional[str] = None

    smoke_test: bool = False
    max_smoke_samples: int = 4
    smoke_test_steps: int = 2
