"""
Configuration management for CLI.

Handles loading and validation of YAML configuration files.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MetricConfig:
    """Configuration for quality metrics."""

    consistency_weight: float = 0.25
    completeness_weight: float = 0.25
    expressiveness_weight: float = 0.25
    structural_coherence_weight: float = 0.25


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    checkpoint_dir: str = "checkpoints"
    alpha: float = 0.7  # Loss weight for rating loss
    beta: float = 0.3  # Loss weight for confidence loss


@dataclass
class ModelConfig:
    """Configuration for CTM model."""

    d_model: int = 256
    d_input: int = 64
    iterations: int = 20
    n_synch_out: int = 32
    vocab_size: int = 10000
    max_sequence_length: int = 512


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""

    k_folds: int = 5
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    calibration_bins: int = 10


@dataclass
class OutputConfig:
    """Configuration for output formats."""

    formats: list = field(default_factory=lambda: ["json", "csv"])
    output_dir: str = "output"
    include_traces: bool = True
    precision: int = 6


@dataclass
class Config:
    """Main configuration class."""

    metrics: MetricConfig = field(default_factory=MetricConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
        ValueError: If config validation fails
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in configuration file: {e}")

    if config_dict is None:
        config_dict = {}

    # Validate configuration
    validate_config(config_dict)

    return config_dict


def validate_config(config_dict: Dict[str, Any]) -> None:
    """
    Validate configuration dictionary.

    Args:
        config_dict: Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate metric weights sum to 1
    if "metrics" in config_dict:
        metrics = config_dict["metrics"]
        weights = [
            metrics.get("consistency_weight", 0.25),
            metrics.get("completeness_weight", 0.25),
            metrics.get("expressiveness_weight", 0.25),
            metrics.get("structural_coherence_weight", 0.25),
        ]

        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"Metric weights must sum to 1.0, got {sum(weights)}")

        if any(w < 0 for w in weights):
            raise ValueError("All metric weights must be non-negative")

    # Validate training parameters
    if "training" in config_dict:
        training = config_dict["training"]

        if "validation_split" in training:
            val_split = training["validation_split"]
            if not 0 < val_split < 1:
                raise ValueError(
                    f"validation_split must be between 0 and 1, got {val_split}"
                )

        if "alpha" in training and "beta" in training:
            alpha = training["alpha"]
            beta = training["beta"]
            if abs(alpha + beta - 1.0) > 1e-6:
                raise ValueError(
                    f"Training loss weights alpha + beta must sum to 1.0, got {alpha + beta}"
                )

    # Validate evaluation parameters
    if "evaluation" in config_dict:
        evaluation = config_dict["evaluation"]

        if "k_folds" in evaluation:
            k_folds = evaluation["k_folds"]
            if k_folds < 5:
                raise ValueError(f"k_folds must be at least 5, got {k_folds}")

        if "bootstrap_samples" in evaluation:
            bootstrap_samples = evaluation["bootstrap_samples"]
            if bootstrap_samples < 1000:
                raise ValueError(
                    f"bootstrap_samples must be at least 1000, got {bootstrap_samples}"
                )

        if "confidence_level" in evaluation:
            confidence_level = evaluation["confidence_level"]
            if not 0 < confidence_level < 1:
                raise ValueError(
                    f"confidence_level must be between 0 and 1, got {confidence_level}"
                )


def create_default_config() -> Config:
    """Create default configuration."""
    return Config()


def save_config(config: Config, config_path: Path) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration object to save
        config_path: Path where to save the configuration
    """
    config_dict = {
        "metrics": {
            "consistency_weight": config.metrics.consistency_weight,
            "completeness_weight": config.metrics.completeness_weight,
            "expressiveness_weight": config.metrics.expressiveness_weight,
            "structural_coherence_weight": config.metrics.structural_coherence_weight,
        },
        "training": {
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate,
            "epochs": config.training.epochs,
            "validation_split": config.training.validation_split,
            "early_stopping_patience": config.training.early_stopping_patience,
            "checkpoint_dir": config.training.checkpoint_dir,
            "alpha": config.training.alpha,
            "beta": config.training.beta,
        },
        "model": {
            "d_model": config.model.d_model,
            "d_input": config.model.d_input,
            "iterations": config.model.iterations,
            "n_synch_out": config.model.n_synch_out,
            "vocab_size": config.model.vocab_size,
            "max_sequence_length": config.model.max_sequence_length,
        },
        "evaluation": {
            "k_folds": config.evaluation.k_folds,
            "bootstrap_samples": config.evaluation.bootstrap_samples,
            "confidence_level": config.evaluation.confidence_level,
            "calibration_bins": config.evaluation.calibration_bins,
        },
        "output": {
            "formats": config.output.formats,
            "output_dir": config.output.output_dir,
            "include_traces": config.output.include_traces,
            "precision": config.output.precision,
        },
    }

    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
