import importlib.metadata
import json
import logging
import os
import platform
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import torch

    HAS_TORCH = True
except (ImportError, OSError):
    torch = None  # type: ignore
    HAS_TORCH = False

try:
    import mlflow  # type: ignore

    HAS_MLFLOW = True
except ImportError:
    mlflow = None  # type: ignore
    HAS_MLFLOW = False

logger = logging.getLogger("documind.ml.training.experiment_tracker")


def _safe_pkg_version(pkg_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(pkg_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_hardware_and_env(resolved_device: str) -> Dict[str, Any]:
    """Captures reproducible software versions and CPU/GPU hardware info."""
    gpu_info: Dict[str, Any] = {"cuda_available": False}
    if HAS_TORCH and torch is not None:
        try:
            cuda_ok = bool(torch.cuda.is_available())
            gpu_info["cuda_available"] = cuda_ok
            if cuda_ok:
                dev_idx = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(dev_idx)
                gpu_info["gpu_name"] = props.name
                gpu_info["gpu_vram_mb"] = round(props.total_memory / (1024 * 1024), 1)
                gpu_info["cuda_version"] = getattr(torch.version, "cuda", None)
        except Exception as exc:
            gpu_info["cuda_inspection_error"] = str(exc)

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "device": resolved_device,
        "hardware": gpu_info,
        "dependencies": {
            "torch": _safe_pkg_version("torch"),
            "transformers": _safe_pkg_version("transformers"),
            "datasets": _safe_pkg_version("datasets"),
            "seqeval": _safe_pkg_version("seqeval"),
            "pillow": _safe_pkg_version("pillow"),
            "accelerate": _safe_pkg_version("accelerate"),
            "mlflow": _safe_pkg_version("mlflow"),
        },
    }


class ExperimentTracker:
    """
    Lightweight, dependency-free JSON experiment tracker with optional MLflow mirroring.
    Persists complete training lineage to `<experiment_dir>/<experiment_id>/experiment_run.json`
    and updates `<experiment_dir>/latest_experiment.json`.
    """

    def __init__(
        self,
        experiment_dir: str = "ml/experiments",
        enable_mlflow: bool = False,
        experiment_name: str = "documind-layoutlmv3-sroie",
    ):
        self.base_dir = Path(experiment_dir)
        self.enable_mlflow = bool(enable_mlflow)
        self.experiment_name = experiment_name

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.experiment_id = f"layoutlmv3-sroie-{ts}-{uuid.uuid4().hex[:6]}"
        self.run_dir = self.base_dir / self.experiment_id
        self.run_file = self.run_dir / "experiment_run.json"
        self.latest_file = self.base_dir / "latest_experiment.json"

        self._start_perf: Optional[float] = None
        self._mlflow_active = False
        self.record: Dict[str, Any] = {}

    def start_run(
        self,
        model_architecture: str,
        label_list: List[str],
        hyperparameters: Dict[str, Any],
        dataset_info: Dict[str, Any],
        seed: int,
        resolved_device: str,
    ) -> str:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._start_perf = time.perf_counter()

        self.record = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "status": "RUNNING",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "finished_at_utc": None,
            "duration_seconds": None,
            "model_architecture": model_architecture,
            "num_labels": len(label_list),
            "label_list": label_list,
            "random_seed": seed,
            "hyperparameters": hyperparameters,
            "dataset": dataset_info,
            "environment": collect_hardware_and_env(resolved_device),
            "epochs": [],
            "best_checkpoint": None,
        }
        self._flush()

        if self.enable_mlflow:
            if HAS_MLFLOW and mlflow is not None:
                try:
                    mlflow.set_experiment(self.experiment_name)
                    mlflow.start_run(run_name=self.experiment_id)
                    self._mlflow_active = True
                    mlflow.log_params(
                        {
                            "model_architecture": model_architecture,
                            "seed": seed,
                            "device": resolved_device,
                            **{
                                k: v
                                for k, v in hyperparameters.items()
                                if isinstance(v, (str, int, float, bool))
                            },
                        }
                    )
                except Exception as exc:
                    logger.warning("MLflow tracking initialization failed (continuing with JSON tracker): %s", exc)
            else:
                logger.warning("enable_mlflow=True requested, but mlflow is not installed; using JSON tracker only.")

        return self.experiment_id

    def log_epoch(self, epoch_record: Dict[str, Any]) -> None:
        self.record.setdefault("epochs", []).append(epoch_record)
        self._flush()

        if self._mlflow_active and mlflow is not None:
            try:
                step = int(epoch_record.get("epoch", 0))
                mlflow.log_metric("train_loss", float(epoch_record.get("train_loss", 0.0)), step=step)
                mlflow.log_metric("val_loss", float(epoch_record.get("val_loss", 0.0)), step=step)
                mlflow.log_metric("val_f1", float(epoch_record.get("val_f1", 0.0)), step=step)
                val_metrics = epoch_record.get("val_metrics") or {}
                if "macro_f1" in val_metrics:
                    mlflow.log_metric("val_macro_f1", float(val_metrics["macro_f1"]), step=step)
            except Exception as exc:
                logger.warning("MLflow epoch logging failed cleanly: %s", exc)

    def finish_run(
        self,
        status: str,
        best_checkpoint: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        elapsed = (
            round(time.perf_counter() - self._start_perf, 2)
            if self._start_perf is not None
            else 0.0
        )
        self.record["status"] = status
        self.record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        self.record["duration_seconds"] = elapsed
        if best_checkpoint is not None:
            self.record["best_checkpoint"] = best_checkpoint
        if extra:
            self.record["summary"] = extra
        self._flush()

        if self._mlflow_active and mlflow is not None:
            try:
                mlflow.log_metric("duration_seconds", elapsed)
                if best_checkpoint and best_checkpoint.get("best_val_f1") is not None:
                    mlflow.log_metric("best_val_f1", float(best_checkpoint["best_val_f1"]))
                mlflow.log_artifact(str(self.run_file))
                mlflow.end_run(status="FINISHED" if status == "COMPLETED" else "FAILED")
            except Exception as exc:
                logger.warning("MLflow run finalization failed cleanly: %s", exc)
            finally:
                self._mlflow_active = False

        return self.run_file

    def _flush(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.run_file, "w", encoding="utf-8") as rf:
            json.dump(self.record, rf, indent=2)
        with open(self.latest_file, "w", encoding="utf-8") as lf:
            json.dump(self.record, lf, indent=2)

