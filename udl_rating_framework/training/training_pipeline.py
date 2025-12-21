"""
Training Pipeline for UDL Rating Framework.

Trains CTM model to approximate mathematical metrics using ground truth computation.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from ..core.aggregation import MetricAggregator
from ..core.confidence import ConfidenceCalculator
from ..core.metrics.base import QualityMetric
from ..core.representation import UDLRepresentation
from ..models.ctm_adapter import UDLRatingCTM, UDLTokenVocabulary

logger = logging.getLogger(__name__)


class UDLDataset(Dataset):
    """
    Dataset for UDL training data.

    Handles UDL representations and their ground truth quality scores.
    """

    def __init__(
        self,
        udl_representations: List[UDLRepresentation],
        vocab: UDLTokenVocabulary,
        max_length: int = 512,
    ):
        """
        Initialize dataset.

        Args:
            udl_representations: List of UDL representations
            vocab: Token vocabulary
            max_length: Maximum sequence length
        """
        self.udl_representations = udl_representations
        self.vocab = vocab
        self.max_length = max_length

        # Pre-tokenize all UDLs
        self.tokenized_udls = []
        for udl in udl_representations:
            tokens = self._tokenize_udl(udl)
            self.tokenized_udls.append(tokens)

    def _tokenize_udl(self, udl: UDLRepresentation) -> torch.Tensor:
        """Convert UDL to token indices."""
        tokens = udl.get_tokens()

        # Convert to indices
        indices = [self.vocab.token_to_index("<BOS>")]
        for token in tokens:
            if len(indices) < self.max_length - 1:
                indices.append(self.vocab.token_to_index(token.text))
        indices.append(self.vocab.token_to_index("<EOS>"))

        # Pad or truncate to max_length
        if len(indices) > self.max_length:
            indices = indices[: self.max_length]
        else:
            indices.extend(
                [self.vocab.token_to_index("<PAD>")] *
                (self.max_length - len(indices))
            )

        return torch.tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.udl_representations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, UDLRepresentation]:
        return self.tokenized_udls[idx], self.udl_representations[idx]


class TrainingPipeline:
    """
    Trains CTM model to approximate mathematical metrics.

    Mathematical Definition:
    Loss Function: L = α · L_rating + β · L_confidence

    Where:
    - L_rating = MSE(predicted, ground_truth)
    - L_confidence = Calibration_Loss(confidence, accuracy)
    - α, β: Loss weighting parameters (α + β = 1)

    The pipeline computes ground truth using mathematical metrics and trains
    the CTM to approximate these values.
    """

    def __init__(
        self,
        model: UDLRatingCTM,
        metrics: List[QualityMetric],
        aggregator: MetricAggregator,
        alpha: float = 0.7,
        beta: float = 0.3,
        learning_rate: float = 1e-3,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize training pipeline.

        Args:
            model: UDL rating CTM model
            metrics: List of quality metrics for ground truth computation
            aggregator: Metric aggregator for combining scores
            alpha: Weight for rating loss (default: 0.7)
            beta: Weight for confidence loss (default: 0.3)
            learning_rate: Learning rate for optimizer
            device: Device for training (CPU/GPU)
        """
        # Validate loss weights
        if abs(alpha + beta - 1.0) > 1e-6:
            raise ValueError(
                f"Loss weights must sum to 1.0, got α={alpha}, β={beta}")
        if alpha < 0 or beta < 0:
            raise ValueError("Loss weights must be non-negative")

        self.model = model
        self.metrics = metrics
        self.aggregator = aggregator
        self.alpha = alpha
        self.beta = beta

        # Set device
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Initialize confidence calculator
        self.confidence_calculator = ConfidenceCalculator()

        # Training state
        self.current_epoch = 0
        self.training_history = {
            "train_loss": [],
            "train_rating_loss": [],
            "train_confidence_loss": [],
            "val_loss": [],
            "val_rating_loss": [],
            "val_confidence_loss": [],
            "val_mae": [],
            "val_rmse": [],
            "val_correlation": [],
        }

        logger.info(f"Initialized training pipeline with α={alpha}, β={beta}")
        logger.info(f"Using device: {self.device}")

    def compute_ground_truth(self, udl: UDLRepresentation) -> float:
        """
        Compute ground truth using mathematical metrics.

        This method computes the ground truth quality score by applying
        all registered metrics and aggregating them using the specified
        aggregation function.

        Args:
            udl: UDL representation

        Returns:
            Ground truth quality score in [0, 1]
        """
        try:
            # Compute individual metric values
            metric_values = {}

            # Get the metric names from the aggregator weights to ensure consistency
            aggregator_metric_names = list(self.aggregator.weights.keys())

            for i, metric in enumerate(self.metrics):
                # Use the metric name from aggregator if available, otherwise generate one
                if i < len(aggregator_metric_names):
                    metric_name = aggregator_metric_names[i]
                else:
                    metric_name = metric.__class__.__name__.replace(
                        "Metric", ""
                    ).lower()

                metric_values[metric_name] = metric.compute(udl)

            # Aggregate using the aggregation function
            ground_truth = self.aggregator.aggregate(metric_values)

            # Ensure bounded in [0, 1]
            return max(0.0, min(1.0, ground_truth))

        except Exception as e:
            logger.error(f"Error computing ground truth for UDL: {e}")
            # Return neutral score on error
            return 0.5

    def compute_loss(
        self,
        predictions: torch.Tensor,
        certainties: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute training loss: L = α·L_rating + β·L_confidence.

        Args:
            predictions: Model predictions [batch, 1]
            certainties: Model certainties [batch, 2]
            ground_truth: Ground truth scores [batch, 1]

        Returns:
            Tuple of (total_loss, rating_loss, confidence_loss)
        """
        # Rating loss: MSE between predictions and ground truth
        rating_loss = nn.MSELoss()(predictions, ground_truth)

        # Confidence loss: Calibration loss
        # Convert certainties to confidence scores
        confidence_scores = torch.softmax(certainties, dim=1)[
            :, 1
        ]  # Take "confident" class

        # Compute prediction accuracy (binary: correct if within threshold)
        threshold = 0.1  # Consider prediction correct if within 0.1 of ground truth
        prediction_accuracy = (
            torch.abs(predictions.squeeze() -
                      ground_truth.squeeze()) < threshold
        ).float()

        # Calibration loss: MSE between confidence and accuracy
        confidence_loss = nn.MSELoss()(confidence_scores, prediction_accuracy)

        # Combined loss
        total_loss = self.alpha * rating_loss + self.beta * confidence_loss

        return total_loss, rating_loss, confidence_loss

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader

        Returns:
            Dictionary with training metrics
        """
        self.model.train()

        total_loss = 0.0
        total_rating_loss = 0.0
        total_confidence_loss = 0.0
        num_batches = 0

        for batch_idx, (token_ids, udl_representations) in enumerate(dataloader):
            # Move to device
            token_ids = token_ids.to(self.device)

            # Compute ground truth for this batch
            ground_truth_scores = []
            for udl in udl_representations:
                gt_score = self.compute_ground_truth(udl)
                ground_truth_scores.append(gt_score)

            ground_truth = torch.tensor(
                ground_truth_scores, dtype=torch.float32, device=self.device
            ).unsqueeze(1)

            # Forward pass
            predictions, certainties, _ = self.model(token_ids)

            # Compute loss
            loss, rating_loss, confidence_loss = self.compute_loss(
                predictions, certainties, ground_truth
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            total_rating_loss += rating_loss.item()
            total_confidence_loss += confidence_loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                logger.debug(
                    f"Batch {batch_idx}: Loss={loss.item():.4f}, "
                    f"Rating={rating_loss.item():.4f}, "
                    f"Confidence={confidence_loss.item():.4f}"
                )

        # Compute average metrics
        avg_metrics = {
            "loss": total_loss / num_batches,
            "rating_loss": total_rating_loss / num_batches,
            "confidence_loss": total_confidence_loss / num_batches,
        }

        return avg_metrics

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model performance on validation set.

        Args:
            dataloader: Validation data loader

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_rating_loss = 0.0
        total_confidence_loss = 0.0
        predictions_list = []
        ground_truth_list = []
        num_batches = 0

        with torch.no_grad():
            for token_ids, udl_representations in dataloader:
                # Move to device
                token_ids = token_ids.to(self.device)

                # Compute ground truth
                ground_truth_scores = []
                for udl in udl_representations:
                    gt_score = self.compute_ground_truth(udl)
                    ground_truth_scores.append(gt_score)

                ground_truth = torch.tensor(
                    ground_truth_scores, dtype=torch.float32, device=self.device
                ).unsqueeze(1)

                # Forward pass
                predictions, certainties, _ = self.model(token_ids)

                # Compute loss
                loss, rating_loss, confidence_loss = self.compute_loss(
                    predictions, certainties, ground_truth
                )

                # Accumulate metrics
                total_loss += loss.item()
                total_rating_loss += rating_loss.item()
                total_confidence_loss += confidence_loss.item()
                num_batches += 1

                # Store predictions and ground truth for correlation computation
                predictions_list.extend(predictions.cpu().numpy().flatten())
                ground_truth_list.extend(ground_truth.cpu().numpy().flatten())

        # Compute evaluation metrics
        predictions_array = np.array(predictions_list)
        ground_truth_array = np.array(ground_truth_list)

        # Mean Absolute Error
        mae = np.mean(np.abs(predictions_array - ground_truth_array))

        # Root Mean Square Error
        rmse = np.sqrt(np.mean((predictions_array - ground_truth_array) ** 2))

        # Pearson correlation
        correlation = np.corrcoef(predictions_array, ground_truth_array)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0

        eval_metrics = {
            "loss": total_loss / num_batches,
            "rating_loss": total_rating_loss / num_batches,
            "confidence_loss": total_confidence_loss / num_batches,
            "mae": mae,
            "rmse": rmse,
            "correlation": correlation,
        }

        return eval_metrics

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        checkpoint_dir: Optional[str] = None,
        save_every: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.

        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            num_epochs: Number of training epochs
            checkpoint_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs

        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {num_epochs} epochs")

        # Create checkpoint directory if specified
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch(train_dataloader)

            # Update training history
            self.training_history["train_loss"].append(train_metrics["loss"])
            self.training_history["train_rating_loss"].append(
                train_metrics["rating_loss"]
            )
            self.training_history["train_confidence_loss"].append(
                train_metrics["confidence_loss"]
            )

            # Validation
            if val_dataloader is not None:
                val_metrics = self.evaluate(val_dataloader)
                self.training_history["val_loss"].append(val_metrics["loss"])
                self.training_history["val_rating_loss"].append(
                    val_metrics["rating_loss"]
                )
                self.training_history["val_confidence_loss"].append(
                    val_metrics["confidence_loss"]
                )
                self.training_history["val_mae"].append(val_metrics["mae"])
                self.training_history["val_rmse"].append(val_metrics["rmse"])
                self.training_history["val_correlation"].append(
                    val_metrics["correlation"]
                )

                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs}: "
                    f"Train Loss={train_metrics['loss']:.4f}, "
                    f"Val Loss={val_metrics['loss']:.4f}, "
                    f"Val MAE={val_metrics['mae']:.4f}, "
                    f"Val Corr={val_metrics['correlation']:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs}: "
                    f"Train Loss={train_metrics['loss']:.4f}"
                )

            # Save checkpoint
            if checkpoint_dir and (epoch + 1) % save_every == 0:
                self.save_checkpoint(checkpoint_dir, epoch + 1)

        logger.info("Training completed")
        return self.training_history

    def save_checkpoint(self, checkpoint_dir: str, epoch: int):
        """
        Save model checkpoint.

        Args:
            checkpoint_dir: Directory to save checkpoint
            epoch: Current epoch number
        """
        checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_history": self.training_history,
            "alpha": self.alpha,
            "beta": self.beta,
            "model_config": {
                "vocab_size": self.model.vocab_size,
                "d_model": self.model.d_model,
                "d_input": self.model.d_input,
                "iterations": self.model.iterations,
                "n_synch_out": self.model.n_synch_out,
            },
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Also save training history as JSON
        history_path = Path(checkpoint_dir) / \
            f"training_history_epoch_{epoch}.json"
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_history = checkpoint["training_history"]
        self.current_epoch = checkpoint["epoch"]
        self.alpha = checkpoint["alpha"]
        self.beta = checkpoint["beta"]

        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary of training progress.

        Returns:
            Dictionary with training summary
        """
        if not self.training_history["train_loss"]:
            return {"status": "not_started"}

        summary = {
            "status": "completed" if self.current_epoch > 0 else "in_progress",
            "epochs_completed": len(self.training_history["train_loss"]),
            "final_train_loss": self.training_history["train_loss"][-1],
            "loss_parameters": {"alpha": self.alpha, "beta": self.beta},
        }

        if self.training_history["val_loss"]:
            summary.update(
                {
                    "final_val_loss": self.training_history["val_loss"][-1],
                    "best_val_mae": min(self.training_history["val_mae"]),
                    "best_val_correlation": max(
                        self.training_history["val_correlation"]
                    ),
                }
            )

        return summary
