"""
Train command for UDL Rating Framework CLI.

Provides functionality to train CTM models on UDL data.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import torch

# Import metrics to trigger registration
import udl_rating_framework.core.metrics
from udl_rating_framework.core.aggregation import MetricAggregator
from udl_rating_framework.core.metrics.base import MetricRegistry
from udl_rating_framework.core.representation import UDLRepresentation
from udl_rating_framework.io.file_discovery import FileDiscovery
from udl_rating_framework.models.ctm_adapter import UDLRatingCTM, UDLTokenVocabulary
from udl_rating_framework.training.training_pipeline import TrainingPipeline, UDLDataset

logger = logging.getLogger(__name__)


@click.command("train")
@click.argument("data_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default="checkpoints",
    help="Directory to save model checkpoints",
)
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Training configuration file",
)
@click.option("--batch-size", type=int, default=32, help="Training batch size")
@click.option("--learning-rate", type=float, default=0.001, help="Learning rate")
@click.option("--epochs", type=int, default=100, help="Number of training epochs")
@click.option(
    "--validation-split",
    type=float,
    default=0.2,
    help="Fraction of data to use for validation",
)
@click.option(
    "--early-stopping-patience", type=int, default=10, help="Early stopping patience"
)
@click.option(
    "--alpha", type=float, default=0.7, help="Weight for rating loss in combined loss"
)
@click.option(
    "--beta",
    type=float,
    default=0.3,
    help="Weight for confidence loss in combined loss",
)
@click.option("--d-model", type=int, default=256, help="CTM model dimension")
@click.option("--d-input", type=int, default=64, help="Input embedding dimension")
@click.option("--iterations", type=int, default=20, help="Number of CTM iterations")
@click.option("--vocab-size", type=int, default=10000, help="Vocabulary size")
@click.option("--max-length", type=int, default=512, help="Maximum sequence length")
@click.option(
    "--device", type=str, default="auto", help="Device to use (cpu, cuda, auto)"
)
@click.option(
    "--resume",
    type=click.Path(exists=True, path_type=Path),
    help="Resume training from checkpoint",
)
@click.pass_context
def train_command(
    ctx: click.Context,
    data_path: Path,
    output_dir: Path,
    config_file: Optional[Path],
    batch_size: int,
    learning_rate: float,
    epochs: int,
    validation_split: float,
    early_stopping_patience: int,
    alpha: float,
    beta: float,
    d_model: int,
    d_input: int,
    iterations: int,
    vocab_size: int,
    max_length: int,
    device: str,
    resume: Optional[Path],
):
    """
    Train a CTM model to approximate mathematical quality metrics.

    DATA_PATH should contain UDL files for training. The system will compute
    ground truth quality scores using mathematical metrics and train the CTM
    to approximate these scores.

    Examples:

        # Basic training
        udl-rating train ./training_data/

        # Training with custom parameters
        udl-rating train ./training_data/ --epochs 200 --batch-size 64

        # Resume from checkpoint
        udl-rating train ./training_data/ --resume ./checkpoints/model_epoch_50.pt

        # Use configuration file
        udl-rating train ./training_data/ --config-file train_config.yaml
    """
    try:
        # Validate parameters
        if not 0 < validation_split < 1:
            raise click.BadParameter(
                f"validation_split must be between 0 and 1, got {validation_split}"
            )

        if abs(alpha + beta - 1.0) > 1e-6:
            raise click.BadParameter(
                f"alpha + beta must sum to 1.0, got {alpha + beta}"
            )

        # Override with config if available
        config = ctx.obj.get("config", {})
        if "training" in config:
            training_config = config["training"]
            batch_size = training_config.get("batch_size", batch_size)
            learning_rate = training_config.get("learning_rate", learning_rate)
            epochs = training_config.get("epochs", epochs)
            validation_split = training_config.get(
                "validation_split", validation_split)
            early_stopping_patience = training_config.get(
                "early_stopping_patience", early_stopping_patience
            )
            alpha = training_config.get("alpha", alpha)
            beta = training_config.get("beta", beta)
            output_dir = Path(training_config.get(
                "checkpoint_dir", str(output_dir)))

        if "model" in config:
            model_config = config["model"]
            d_model = model_config.get("d_model", d_model)
            d_input = model_config.get("d_input", d_input)
            iterations = model_config.get("iterations", iterations)
            vocab_size = model_config.get("vocab_size", vocab_size)
            max_length = model_config.get("max_sequence_length", max_length)

        # Set up device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Training CTM model on data from: {data_path}")
        logger.info(f"Device: {device}")
        logger.info(
            f"Model parameters: d_model={d_model}, d_input={d_input}, iterations={iterations}"
        )
        logger.info(
            f"Training parameters: batch_size={batch_size}, lr={learning_rate}, epochs={epochs}"
        )
        logger.info(f"Loss weights: alpha={alpha}, beta={beta}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Discover UDL files
        file_discovery = FileDiscovery()
        discovery_result = file_discovery.discover_files(str(data_path))
        udl_files = discovery_result.discovered_files

        # Log any discovery errors
        for error in discovery_result.errors:
            logger.warning(f"File discovery error: {error}")

        if not udl_files:
            logger.error(f"No UDL files found in {data_path}")
            sys.exit(1)

        logger.info(f"Found {len(udl_files)} UDL files for training")

        # Load and parse UDL files
        logger.info("Loading and parsing UDL files...")
        udl_representations = []
        for udl_file in udl_files:
            try:
                with open(udl_file, "r", encoding="utf-8") as f:
                    content = f.read()
                udl_repr = UDLRepresentation(content, str(udl_file))
                udl_representations.append(udl_repr)
            except Exception as e:
                logger.warning(f"Failed to parse {udl_file}: {e}")
                continue

        if not udl_representations:
            logger.error("No UDL files could be parsed successfully")
            sys.exit(1)

        logger.info(
            f"Successfully parsed {len(udl_representations)} UDL files")

        # Build vocabulary
        logger.info("Building vocabulary...")
        vocab = UDLTokenVocabulary()
        for udl_repr in udl_representations:
            tokens = udl_repr.get_tokens()
            for token in tokens:
                vocab.add_token(token.text)

        vocab.build_vocab(max_size=vocab_size)
        logger.info(f"Built vocabulary with {len(vocab)} tokens")

        # Create dataset
        dataset = UDLDataset(udl_representations, vocab, max_length)

        # Set up metrics and aggregator
        metric_registry = MetricRegistry()
        available_metrics = metric_registry.list_metrics()

        # Map CLI weight names to registry names
        metric_name_mapping = {
            "ConsistencyMetric": "consistency",
            "CompletenessMetric": "completeness",
            "ExpressivenessMetric": "expressiveness",
            "StructuralCoherenceMetric": "structural_coherence",
        }

        # Use equal weights for training (can be overridden by config)
        weights_dict = {
            "consistency": 0.25,
            "completeness": 0.25,
            "expressiveness": 0.25,
            "structural_coherence": 0.25,
        }

        if "metrics" in config:
            metrics_config = config["metrics"]
            weights_dict.update(
                {
                    "consistency": metrics_config.get("consistency_weight", 0.25),
                    "completeness": metrics_config.get("completeness_weight", 0.25),
                    "expressiveness": metrics_config.get("expressiveness_weight", 0.25),
                    "structural_coherence": metrics_config.get(
                        "structural_coherence_weight", 0.25
                    ),
                }
            )

        aggregator = MetricAggregator(weights_dict)
        metric_names = list(weights_dict.keys())
        metrics = [metric_registry.get_metric(name)() for name in metric_names]

        # Create model
        model = UDLRatingCTM(
            vocab_size=len(vocab),
            d_model=d_model,
            d_input=d_input,
            iterations=iterations,
            n_synch_out=32,
        )

        # Resume from checkpoint if specified
        start_epoch = 0
        if resume:
            logger.info(f"Resuming training from: {resume}")
            checkpoint = torch.load(resume, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)

        # Create training pipeline
        pipeline = TrainingPipeline(
            model=model, metrics=metrics, aggregator=aggregator, alpha=alpha, beta=beta
        )

        # Train model
        logger.info("Starting training...")
        training_history = pipeline.train(
            dataset=dataset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            validation_split=validation_split,
            early_stopping_patience=early_stopping_patience,
            checkpoint_dir=output_dir,
            device=device,
            start_epoch=start_epoch,
        )

        # Save final model and training history
        final_model_path = output_dir / "final_model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "vocab": vocab,
                "config": {
                    "d_model": d_model,
                    "d_input": d_input,
                    "iterations": iterations,
                    "vocab_size": len(vocab),
                    "max_length": max_length,
                },
                "training_history": training_history,
            },
            final_model_path,
        )

        history_path = output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(training_history, f, indent=2, default=str)

        logger.info(f"Training completed!")
        logger.info(f"Final model saved to: {final_model_path}")
        logger.info(f"Training history saved to: {history_path}")

        # Print training summary
        if training_history:
            final_train_loss = training_history["train_loss"][-1]
            final_val_loss = training_history["val_loss"][-1]
            best_val_loss = min(training_history["val_loss"])

            logger.info(f"Final training loss: {final_train_loss:.6f}")
            logger.info(f"Final validation loss: {final_val_loss:.6f}")
            logger.info(f"Best validation loss: {best_val_loss:.6f}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        if logger.getEffectiveLevel() <= logging.DEBUG:
            import traceback

            traceback.print_exc()
        sys.exit(1)
