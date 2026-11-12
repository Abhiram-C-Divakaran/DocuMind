from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
try:
    import torch
    HAS_TORCH = True
except (ImportError, OSError):
    torch = None  # type: ignore
    HAS_TORCH = False


@dataclass
class LayoutLMv3TrainingConfig:
    """Configuration hyperparameters for fine-tuning LayoutLMv3 on SROIE."""

    # Model & Tokenizer
    model_name_or_path: str = "microsoft/layoutlmv3-base"
    
    # Dataset paths
    data_dir: str = "data/sroie_processed"
    output_dir: str = "checkpoints/layoutlmv3-sroie"
    best_model_dir: str = "checkpoints/layoutlmv3-sroie-best"
    
    # Sequence & Input shapes
    max_seq_length: int = 512
    image_size: int = 224

    # Optimization
    batch_size: int = 2
    eval_batch_size: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 4
    seed: int = 42

    # Hardware & Runtime
    device: str = field(default_factory=lambda: "cuda" if (HAS_TORCH and torch and torch.cuda.is_available()) else "cpu")
    num_workers: int = 0
    fp16: bool = False  # Set to True on CUDA devices if desired

    # Logging & Checkpoints
    logging_steps: int = 10
    eval_steps: int = 50
    save_total_limit: int = 2

    # Smoke test mode for quick verification on CPU or CI
    smoke_test: bool = False
    max_smoke_samples: int = 4
    smoke_test_steps: int = 2

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
