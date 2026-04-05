"""Configuration for the self-loop-eval system."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for an LLM model (student or teacher)."""

    provider: str = "local"
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    api_key: str | None = None
    api_base: str | None = None
    temperature: float = 0.7
    max_tokens: int = 2048
    local_model_path: str | None = None
    quantization: str | None = "4bit"
    device_map: str = "auto"


@dataclass
class EvalLoopConfig:
    """Configuration for the self-evaluation loop."""

    max_rounds: int = 5
    convergence_threshold: float = 0.05
    min_score_improvement: float = 0.01
    stuck_plateau_rounds: int = 2
    enable_teacher: bool = True


@dataclass
class TrainingConfig:
    """Configuration for the training pipeline."""

    output_dir: str = "training_output"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    max_seq_length: int = 2048
    training_data_path: str = "training_data"


@dataclass
class RLConfig:
    """Configuration for the RL layer."""

    method: str = "dpo"  # "ppo" or "dpo"
    reward_self_improvement_weight: float = 0.6
    reward_teacher_alignment_weight: float = 0.4
    ppo_epochs: int = 4
    ppo_batch_size: int = 4
    dpo_beta: float = 0.1


@dataclass
class MetricsConfig:
    """Configuration for metrics tracking."""

    metrics_dir: str = "metrics"
    log_level: str = "INFO"
    track_teacher_dependency: bool = True
    track_self_eval_accuracy: bool = True


@dataclass
class SystemConfig:
    """Top-level system configuration."""

    student: ModelConfig = field(default_factory=ModelConfig)
    teacher: ModelConfig = field(
        default_factory=lambda: ModelConfig(model_name="Qwen/Qwen2.5-1.5B-Instruct")
    )
    eval_loop: EvalLoopConfig = field(default_factory=EvalLoopConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    data_dir: str = "data"

    @classmethod
    def from_dict(cls, data: dict) -> SystemConfig:
        """Create a SystemConfig from a dictionary."""
        return cls(
            student=ModelConfig(**data.get("student", {})),
            teacher=ModelConfig(**data.get("teacher", {})),
            eval_loop=EvalLoopConfig(**data.get("eval_loop", {})),
            training=TrainingConfig(**data.get("training", {})),
            rl=RLConfig(**data.get("rl", {})),
            metrics=MetricsConfig(**data.get("metrics", {})),
            data_dir=data.get("data_dir", "data"),
        )

    def ensure_dirs(self) -> None:
        """Create necessary directories."""
        for dir_path in [
            self.data_dir,
            self.training.output_dir,
            self.training.training_data_path,
            self.metrics.metrics_dir,
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
