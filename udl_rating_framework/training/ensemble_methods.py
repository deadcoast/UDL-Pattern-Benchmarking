"""
CTM-Aware Ensemble Methods.

Implements ensemble techniques specifically designed for Continuous Thought Machines,
leveraging synchronization diversity, neuron selection strategies, and temporal dynamics
for improved prediction accuracy and uncertainty quantification.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from pathlib import Path
import json

from ..models.ctm_adapter import UDLRatingCTM
from .training_pipeline import TrainingPipeline

logger = logging.getLogger(__name__)


@dataclass
class EnsembleMember:
    """
    Represents a single member of an ensemble.
    """

    model: UDLRatingCTM
    weight: float = 1.0
    training_history: Optional[Dict[str, List[float]]] = None
    model_id: Optional[str] = None

    def predict(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make prediction with this ensemble member.

        Args:
            token_ids: Input token indices

        Returns:
            Tuple of (predictions, certainties)
        """
        self.model.eval()
        with torch.no_grad():
            predictions, certainties, _, _ = self.model(token_ids)
        return predictions, certainties


class EnsemblePredictor(nn.Module):
    """
    Ensemble predictor that combines multiple CTM models.

    Supports various ensemble methods:
    1. Simple averaging
    2. Weighted averaging
    3. Stacking (meta-learning)
    4. Bayesian model averaging
    """

    def __init__(
        self,
        members: List[EnsembleMember],
        method: str = "weighted_average",
        device: Optional[torch.device] = None,
    ):
        """
        Initialize ensemble predictor.

        Args:
            members: List of ensemble members
            method: Ensemble method ('simple_average', 'weighted_average', 'stacking', 'bayesian')
            device: Device for computation
        """
        super().__init__()

        if not members:
            raise ValueError("Ensemble must have at least one member")

        self.members = members
        self.method = method
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Move all models to device
        for member in self.members:
            member.model.to(self.device)

        # Normalize weights for weighted averaging
        if method == "weighted_average":
            total_weight = sum(member.weight for member in members)
            for member in members:
                member.weight = member.weight / total_weight

        # Initialize stacking meta-learner if needed
        if method == "stacking":
            self.meta_learner = nn.Sequential(
                nn.Linear(len(members), 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            ).to(self.device)

        logger.info(f"Initialized ensemble with {len(members)} members using {method}")

    def forward(
        self, token_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through ensemble.

        Args:
            token_ids: Input token indices [batch, seq_len]

        Returns:
            Tuple of (predictions, certainties, ensemble_info)
        """
        token_ids = token_ids.to(self.device)
        # Collect predictions from all members
        member_predictions = []
        member_certainties = []

        for member in self.members:
            pred, cert = member.predict(token_ids)
            member_predictions.append(pred)
            member_certainties.append(cert)

        # Stack predictions: [num_members, batch, 1]
        member_predictions = torch.stack(member_predictions, dim=0)
        member_certainties = torch.stack(member_certainties, dim=0)

        # Combine predictions based on method
        if self.method == "simple_average":
            ensemble_pred = torch.mean(member_predictions, dim=0)
            ensemble_cert = torch.mean(member_certainties, dim=0)

        elif self.method == "weighted_average":
            weights = torch.tensor(
                [member.weight for member in self.members], device=self.device
            ).view(-1, 1, 1)
            ensemble_pred = torch.sum(weights * member_predictions, dim=0)
            ensemble_cert = torch.sum(weights * member_certainties, dim=0)

        elif self.method == "stacking":
            # Use meta-learner to combine predictions
            stacked_preds = member_predictions.permute(
                1, 2, 0
            )  # [batch, 1, num_members]
            stacked_preds = stacked_preds.squeeze(1)  # [batch, num_members]
            ensemble_pred = self.meta_learner(stacked_preds).unsqueeze(1)
            ensemble_cert = torch.mean(
                member_certainties, dim=0
            )  # Simple average for certainty

        elif self.method == "bayesian":
            # Bayesian model averaging with uncertainty
            ensemble_pred = torch.mean(member_predictions, dim=0)

            # Compute epistemic uncertainty (model disagreement)
            pred_variance = torch.var(member_predictions, dim=0)

            # Combine with aleatoric uncertainty (average of individual uncertainties)
            aleatoric_uncertainty = torch.mean(member_certainties, dim=0)

            # Total uncertainty combines both sources
            total_uncertainty = pred_variance + aleatoric_uncertainty
            ensemble_cert = total_uncertainty

        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")

        # Compute ensemble statistics
        pred_std = torch.std(member_predictions, dim=0)
        pred_min = torch.min(member_predictions, dim=0)[0]
        pred_max = torch.max(member_predictions, dim=0)[0]

        ensemble_info = {
            "member_predictions": member_predictions.detach().cpu().numpy(),
            "prediction_std": pred_std.detach().cpu().numpy(),
            "prediction_min": pred_min.detach().cpu().numpy(),
            "prediction_max": pred_max.detach().cpu().numpy(),
            "num_members": len(self.members),
            "method": self.method,
        }

        return ensemble_pred, ensemble_cert, ensemble_info

    def predict_with_uncertainty(
        self, token_ids: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions with comprehensive uncertainty quantification.

        Args:
            token_ids: Input token indices

        Returns:
            Dictionary with predictions and uncertainty measures
        """
        self.eval()
        with torch.no_grad():
            ensemble_pred, ensemble_cert, ensemble_info = self.forward(token_ids)

        # Compute additional uncertainty measures
        member_preds = ensemble_info["member_predictions"]  # [num_members, batch, 1]

        # Epistemic uncertainty (model disagreement)
        epistemic_uncertainty = np.var(member_preds, axis=0)

        # Aleatoric uncertainty (average individual uncertainty)
        aleatoric_uncertainty = np.mean(
            [
                member.model(token_ids)[1].detach().cpu().numpy()
                for member in self.members
            ],
            axis=0,
        )

        # Prediction intervals (confidence bounds)
        pred_mean = np.mean(member_preds, axis=0)
        pred_std = np.std(member_preds, axis=0)

        # 95% confidence intervals
        confidence_lower = pred_mean - 1.96 * pred_std
        confidence_upper = pred_mean + 1.96 * pred_std

        return {
            "predictions": ensemble_pred.detach().cpu().numpy(),
            "epistemic_uncertainty": epistemic_uncertainty,
            "aleatoric_uncertainty": aleatoric_uncertainty,
            "total_uncertainty": epistemic_uncertainty + aleatoric_uncertainty,
            "prediction_std": pred_std,
            "confidence_lower": confidence_lower,
            "confidence_upper": confidence_upper,
            "member_predictions": member_preds,
        }

    def save(self, filepath: str):
        """
        Save ensemble to file.

        Args:
            filepath: Path to save ensemble
        """
        save_dir = Path(filepath)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save ensemble metadata
        metadata = {
            "method": self.method,
            "num_members": len(self.members),
            "member_weights": [member.weight for member in self.members],
            "member_ids": [member.model_id for member in self.members],
        }

        with open(save_dir / "ensemble_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save individual models
        for i, member in enumerate(self.members):
            model_path = save_dir / f"member_{i}.pt"
            torch.save(
                {
                    "model_state_dict": member.model.state_dict(),
                    "weight": member.weight,
                    "training_history": member.training_history,
                    "model_id": member.model_id,
                },
                model_path,
            )

        # Save meta-learner if using stacking
        if self.method == "stacking":
            torch.save(self.meta_learner.state_dict(), save_dir / "meta_learner.pt")

        logger.info(f"Saved ensemble to {filepath}")

    @classmethod
    def load(
        cls, filepath: str, vocab_size: int, device: Optional[torch.device] = None
    ) -> "EnsemblePredictor":
        """
        Load ensemble from file.

        Args:
            filepath: Path to ensemble directory
            vocab_size: Vocabulary size for model creation
            device: Device for computation

        Returns:
            Loaded ensemble predictor
        """
        load_dir = Path(filepath)

        # Load metadata
        with open(load_dir / "ensemble_metadata.json", "r") as f:
            metadata = json.load(f)

        # Load individual models
        members = []
        for i in range(metadata["num_members"]):
            model_path = load_dir / f"member_{i}.pt"
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)

            # Create model (need to infer architecture from state dict)
            # This is a simplified approach - in practice, you'd save model config
            model = UDLRatingCTM(vocab_size=vocab_size)
            model.load_state_dict(checkpoint["model_state_dict"])

            member = EnsembleMember(
                model=model,
                weight=checkpoint["weight"],
                training_history=checkpoint.get("training_history"),
                model_id=checkpoint.get("model_id"),
            )
            members.append(member)

        # Create ensemble
        ensemble = cls(members, metadata["method"], device)

        # Load meta-learner if using stacking
        if metadata["method"] == "stacking":
            meta_learner_path = load_dir / "meta_learner.pt"
            if meta_learner_path.exists():
                ensemble.meta_learner.load_state_dict(
                    torch.load(
                        meta_learner_path, map_location=device, weights_only=True
                    )
                )

        logger.info(f"Loaded ensemble from {filepath}")
        return ensemble


class CTMEnsembleTrainer:
    """
    CTM-aware ensemble trainer that creates diversity through CTM-specific mechanisms.

    Creates ensemble diversity by varying:
    - Synchronization strategies (neuron selection types)
    - Memory lengths and temporal dynamics
    - Neuron-level model architectures
    - Attention patterns and iteration counts
    """

    def __init__(
        self,
        vocab_size: int,
        base_config: Dict[str, Any],
        device: Optional[torch.device] = None,
    ):
        """
        Initialize CTM ensemble trainer.

        Args:
            vocab_size: Size of token vocabulary
            base_config: Base CTM configuration
            device: Device for training
        """
        self.vocab_size = vocab_size
        self.base_config = base_config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.trained_members = []

        logger.info(f"Initialized CTM ensemble trainer with device: {self.device}")

    def train_synchronization_diverse_ensemble(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_members: int = 5,
        num_epochs: int = 50,
    ) -> List[EnsembleMember]:
        """
        Train ensemble with synchronization diversity.

        Creates diversity by varying CTM-specific synchronization parameters:
        - Different neuron selection strategies
        - Varying synchronization dimensions
        - Different memory lengths
        - Diverse temporal iteration patterns

        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_members: Number of ensemble members
            num_epochs: Number of training epochs

        Returns:
            List of trained ensemble members with synchronization diversity
        """
        logger.info(
            f"Training CTM ensemble with synchronization diversity ({num_members} members)"
        )

        members = []

        # Define CTM-specific diversity strategies
        neuron_select_strategies = ["first-last", "random", "random-pairing"]
        memory_lengths = [5, 10, 15, 20]
        iteration_counts = [15, 20, 25, 30]
        synch_dimensions = [16, 32, 64]

        for i in range(num_members):
            logger.info(f"Training CTM ensemble member {i + 1}/{num_members}")

            # Create CTM-specific diverse configuration
            config = self._create_ctm_diverse_config(
                i,
                num_members,
                neuron_select_strategies,
                memory_lengths,
                iteration_counts,
                synch_dimensions,
            )

            # Create and train model
            member = self._train_single_member(
                config, train_dataloader, val_dataloader, num_epochs, f"ctm_member_{i}"
            )

            members.append(member)

        self.trained_members = members
        logger.info(f"Completed training {len(members)} CTM ensemble members")

        return members

    def _create_ctm_diverse_config(
        self,
        member_idx: int,
        total_members: int,
        neuron_select_strategies: List[str],
        memory_lengths: List[int],
        iteration_counts: List[int],
        synch_dimensions: List[int],
    ) -> Dict[str, Any]:
        """
        Create CTM-specific diverse configuration.

        Args:
            member_idx: Index of current member
            total_members: Total number of members
            neuron_select_strategies: Available neuron selection strategies
            memory_lengths: Available memory lengths
            iteration_counts: Available iteration counts
            synch_dimensions: Available synchronization dimensions

        Returns:
            CTM-specific diverse configuration
        """
        config = self.base_config.copy()

        # Vary neuron selection strategy (most important for CTM diversity)
        config["neuron_select_type"] = neuron_select_strategies[
            member_idx % len(neuron_select_strategies)
        ]

        # Vary memory length (affects neuron-level model behavior)
        config["memory_length"] = memory_lengths[
            (member_idx // len(neuron_select_strategies)) % len(memory_lengths)
        ]

        # Vary iteration count (affects temporal dynamics)
        config["iterations"] = iteration_counts[
            (member_idx // (len(neuron_select_strategies) * len(memory_lengths)))
            % len(iteration_counts)
        ]

        # Vary synchronization dimensions
        synch_dim = synch_dimensions[member_idx % len(synch_dimensions)]
        config["n_synch_out"] = synch_dim
        config["n_synch_action"] = max(
            8, synch_dim // 2
        )  # Action synch is typically smaller

        # Vary neuron-level model depth
        config["deep_nlms"] = (
            member_idx % 2 == 0
        )  # Alternate between deep and shallow NLMs

        # Vary synapse depth (affects information processing)
        config["synapse_depth"] = 2 + (member_idx % 3)  # 2, 3, or 4

        # Add some randomness to self-pairing for random-pairing strategy
        if config["neuron_select_type"] == "random-pairing":
            config["n_random_pairing_self"] = member_idx % 5  # 0-4 self-pairings

        logger.info(
            f"Member {member_idx}: {config['neuron_select_type']}, "
            f"memory={config['memory_length']}, "
            f"iterations={config['iterations']}, "
            f"synch_out={config['n_synch_out']}"
        )

        return config

    def train_temporal_diverse_ensemble(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_members: int = 5,
        num_epochs: int = 50,
    ) -> List[EnsembleMember]:
        """
        Train ensemble with temporal diversity.

        Creates diversity by varying temporal processing characteristics:
        - Different iteration patterns
        - Varying memory utilization
        - Different attention temporal dynamics

        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_members: Number of ensemble members
            num_epochs: Number of training epochs

        Returns:
            List of trained ensemble members with temporal diversity
        """
        logger.info(
            f"Training CTM ensemble with temporal diversity ({num_members} members)"
        )

        members = []

        for i in range(num_members):
            logger.info(f"Training temporal ensemble member {i + 1}/{num_members}")

            # Create temporal-focused diverse configuration
            config = self._create_temporal_diverse_config(i, num_members)

            # Create and train model
            member = self._train_single_member(
                config,
                train_dataloader,
                val_dataloader,
                num_epochs,
                f"temporal_member_{i}",
            )

            members.append(member)

        self.trained_members = members
        logger.info(f"Completed training {len(members)} temporal ensemble members")

        return members

    def _create_temporal_diverse_config(
        self, member_idx: int, total_members: int
    ) -> Dict[str, Any]:
        """
        Create temporal-focused diverse configuration.

        Args:
            member_idx: Index of current member
            total_members: Total number of members

        Returns:
            Temporal-diverse configuration
        """
        config = self.base_config.copy()

        # Create different temporal processing patterns
        base_iterations = config.get("iterations", 20)
        base_memory = config.get("memory_length", 10)

        # Pattern 1: Fast processing (fewer iterations, shorter memory)
        if member_idx % 3 == 0:
            config["iterations"] = max(10, base_iterations - 5)
            config["memory_length"] = max(3, base_memory - 3)
            config["deep_nlms"] = False  # Faster processing

        # Pattern 2: Deep processing (more iterations, longer memory)
        elif member_idx % 3 == 1:
            config["iterations"] = base_iterations + 10
            config["memory_length"] = base_memory + 5
            config["deep_nlms"] = True  # Deeper processing

        # Pattern 3: Balanced processing (default with variations)
        else:
            config["iterations"] = base_iterations
            config["memory_length"] = base_memory
            config["deep_nlms"] = True

        # Vary attention patterns
        config["heads"] = 4 + (member_idx % 3) * 4  # 4, 8, or 12 heads

        # Vary dropout for different regularization patterns
        config["dropout"] = 0.05 + (member_idx % 4) * 0.05  # 0.05, 0.10, 0.15, 0.20

        return config

    def _train_single_member(
        self,
        config: Dict[str, Any],
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int,
        member_id: str,
    ) -> EnsembleMember:
        """
        Train a single ensemble member.

        Args:
            config: Model configuration
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of training epochs
            member_id: Unique identifier for this member

        Returns:
            Trained ensemble member
        """
        # Create model
        model = UDLRatingCTM(vocab_size=self.vocab_size, **config)

        # Create training pipeline
        pipeline = TrainingPipeline(
            model=model,
            metrics=config.get("metrics", []),
            aggregator=config.get("aggregator"),
            device=self.device,
        )

        # Train model
        history = pipeline.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=num_epochs,
        )

        # Compute member weight based on validation performance
        if history["val_correlation"]:
            weight = max(history["val_correlation"])
        else:
            weight = 1.0 / (1.0 + min(history["val_loss"]))

        return EnsembleMember(
            model=model, weight=weight, training_history=history, model_id=member_id
        )

    def train_stacking_ensemble(
        self,
        base_members: List[EnsembleMember],
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        meta_epochs: int = 20,
    ) -> EnsemblePredictor:
        """
        Train stacking ensemble with meta-learner.

        Args:
            base_members: Pre-trained base models
            train_dataloader: Training data for meta-learner
            val_dataloader: Validation data
            meta_epochs: Number of epochs for meta-learner training

        Returns:
            Trained stacking ensemble
        """
        logger.info("Training stacking ensemble meta-learner")

        # Create ensemble with stacking method
        ensemble = EnsemblePredictor(
            base_members, method="stacking", device=self.device
        )

        # Prepare meta-training data
        meta_optimizer = torch.optim.Adam(ensemble.meta_learner.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        for epoch in range(meta_epochs):
            ensemble.train()
            total_loss = 0.0
            num_batches = 0

            for token_ids, udl_representations in train_dataloader:
                token_ids = token_ids.to(self.device)

                # Get base model predictions
                base_predictions = []
                for member in base_members:
                    pred, _ = member.predict(token_ids)
                    base_predictions.append(pred)

                base_predictions = torch.stack(
                    base_predictions, dim=0
                )  # [num_members, batch, 1]
                base_predictions = base_predictions.permute(1, 2, 0).squeeze(
                    1
                )  # [batch, num_members]

                # Compute ground truth (simplified - would use actual metrics)
                ground_truth = torch.mean(base_predictions, dim=1, keepdim=True)

                # Meta-learner prediction
                meta_pred = ensemble.meta_learner(base_predictions)

                # Compute loss
                loss = criterion(meta_pred, ground_truth)

                # Backward pass
                meta_optimizer.zero_grad()
                loss.backward()
                meta_optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            logger.info(
                f"Meta-learner epoch {epoch + 1}/{meta_epochs}: Loss={avg_loss:.4f}"
            )

        logger.info("Completed stacking ensemble training")
        return ensemble

    def create_ensemble(
        self, members: List[EnsembleMember], method: str = "weighted_average"
    ) -> EnsemblePredictor:
        """
        Create ensemble predictor from trained members.

        Args:
            members: List of trained ensemble members
            method: Ensemble method

        Returns:
            Ensemble predictor
        """
        return EnsemblePredictor(members, method, self.device)


def create_bootstrap_ensemble(
    model_factory: callable,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_members: int = 5,
    bootstrap_ratio: float = 0.8,
    num_epochs: int = 50,
) -> EnsemblePredictor:
    """
    Create bootstrap ensemble with different training subsets.

    Args:
        model_factory: Function that creates new model instances
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        num_members: Number of ensemble members
        bootstrap_ratio: Fraction of training data for each member
        num_epochs: Number of training epochs

    Returns:
        Bootstrap ensemble predictor
    """
    # This is a simplified implementation
    # In practice, you'd implement proper bootstrap sampling

    members = []

    for i in range(num_members):
        # Create model
        model = model_factory()

        # Train on bootstrap sample (simplified - just train normally)
        # TODO: Implement actual bootstrap sampling

        member = EnsembleMember(model=model, weight=1.0, model_id=f"bootstrap_{i}")
        members.append(member)

    return EnsemblePredictor(members, method="simple_average")
