"""
CTM-Aware Uncertainty Quantification for UDL Rating Framework.

Implements uncertainty quantification methods that leverage the unique
architecture of Continuous Thought Machines, including synchronization-based
uncertainty, neuron-level dynamics analysis, and temporal uncertainty evolution.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from ..core.representation import UDLRepresentation
from ..models.ctm_adapter import TrackingData, UDLRatingCTM

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyEstimate:
    """
    Container for uncertainty estimates.
    """

    # Point estimates
    prediction: float
    confidence: float

    # Uncertainty components
    epistemic_uncertainty: float  # Model uncertainty
    aleatoric_uncertainty: float  # Data uncertainty
    total_uncertainty: float  # Combined uncertainty

    # Confidence intervals
    confidence_interval_lower: float
    confidence_interval_upper: float
    confidence_level: float = 0.95

    # Calibration metrics
    calibration_error: Optional[float] = None
    reliability_score: Optional[float] = None

    # Additional metadata
    method: str = "unknown"
    n_samples: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "epistemic_uncertainty": self.epistemic_uncertainty,
            "aleatoric_uncertainty": self.aleatoric_uncertainty,
            "total_uncertainty": self.total_uncertainty,
            "confidence_interval_lower": self.confidence_interval_lower,
            "confidence_interval_upper": self.confidence_interval_upper,
            "confidence_level": self.confidence_level,
            "calibration_error": self.calibration_error,
            "reliability_score": self.reliability_score,
            "method": self.method,
            "n_samples": self.n_samples,
        }


class UncertaintyQuantifier(ABC):
    """
    Abstract base class for uncertainty quantification methods.
    """

    @abstractmethod
    def estimate_uncertainty(
        self, model: UDLRatingCTM, token_ids: torch.Tensor, **kwargs
    ) -> UncertaintyEstimate:
        """
        Estimate uncertainty for given input.

        Args:
            model: Trained CTM model
            token_ids: Input token IDs
            **kwargs: Additional method-specific arguments

        Returns:
            Uncertainty estimate
        """
        pass


class SynchronizationUncertainty(UncertaintyQuantifier):
    """
    CTM-specific uncertainty quantification using synchronization dynamics.

    Leverages the CTM's unique synchronization mechanism to estimate uncertainty:
    - High synchronization = high confidence
    - Low/unstable synchronization = high uncertainty
    - Temporal synchronization evolution provides additional uncertainty signals
    """

    def __init__(self, n_samples: int = 50, temporal_analysis: bool = True):
        """
        Initialize synchronization-based uncertainty estimation.

        Args:
            n_samples: Number of forward passes for temporal analysis
            temporal_analysis: Whether to analyze synchronization evolution over time
        """
        self.n_samples = n_samples
        self.temporal_analysis = temporal_analysis

    def _compute_synchronization_entropy(self, synch_matrix: torch.Tensor) -> float:
        """
        Compute entropy of synchronization matrix as uncertainty measure.

        Args:
            synch_matrix: Synchronization representation [batch, synch_dim]

        Returns:
            Entropy-based uncertainty score
        """
        # Normalize synchronization values to probabilities
        synch_probs = torch.softmax(synch_matrix.flatten(), dim=0)

        # Compute Shannon entropy
        entropy = -torch.sum(synch_probs * torch.log(synch_probs + 1e-10))

        # Normalize by maximum possible entropy
        max_entropy = torch.log(torch.tensor(
            len(synch_probs), dtype=torch.float32))
        normalized_entropy = entropy / max_entropy

        return normalized_entropy.item()

    def _analyze_synchronization_stability(
        self, tracking_data: TrackingData
    ) -> Dict[str, float]:
        """
        Analyze temporal stability of synchronization patterns.

        Args:
            tracking_data: CTM tracking data with synchronization evolution

        Returns:
            Dictionary with stability metrics
        """
        synch_out = tracking_data.synch_out  # [iterations, batch, synch_dim]

        # Compute temporal variance (instability)
        temporal_variance = np.var(synch_out, axis=0)  # [batch, synch_dim]
        mean_instability = np.mean(temporal_variance)

        # Compute synchronization convergence
        if synch_out.shape[0] > 1:
            initial_synch = synch_out[0]  # [batch, synch_dim]
            final_synch = synch_out[-1]  # [batch, synch_dim]
            convergence_distance = np.linalg.norm(
                final_synch - initial_synch, axis=-1)
            mean_convergence = np.mean(convergence_distance)
        else:
            mean_convergence = 0.0

        # Compute rate of synchronization change
        if synch_out.shape[0] > 2:
            synch_changes = np.diff(
                synch_out, axis=0
            )  # [iterations-1, batch, synch_dim]
            change_magnitudes = np.linalg.norm(
                synch_changes, axis=-1
            )  # [iterations-1, batch]
            mean_change_rate = np.mean(change_magnitudes)

            # Acceleration in synchronization changes
            change_acceleration = np.diff(
                change_magnitudes, axis=0
            )  # [iterations-2, batch]
            mean_acceleration = np.mean(np.abs(change_acceleration))
        else:
            mean_change_rate = 0.0
            mean_acceleration = 0.0

        return {
            "temporal_instability": mean_instability,
            "convergence_distance": mean_convergence,
            "change_rate": mean_change_rate,
            "change_acceleration": mean_acceleration,
        }

    def estimate_uncertainty(
        self, model: UDLRatingCTM, token_ids: torch.Tensor, **kwargs
    ) -> UncertaintyEstimate:
        """
        Estimate uncertainty using CTM synchronization dynamics.

        Args:
            model: Trained CTM model
            token_ids: Input token IDs [batch, seq_len]

        Returns:
            Uncertainty estimate based on synchronization patterns
        """
        device = token_ids.device
        batch_size = token_ids.shape[0]

        model.eval()

        predictions = []
        certainties = []
        synchronization_entropies = []
        tracking_data_list = []

        with torch.no_grad():
            for _ in range(self.n_samples):
                # Forward pass with tracking enabled
                pred, cert, synch_out, tracking_data = model(
                    token_ids, track=True)

                predictions.append(pred.cpu().numpy())
                certainties.append(cert.cpu().numpy())

                # Compute synchronization entropy
                if synch_out is not None:
                    synch_entropy = self._compute_synchronization_entropy(
                        synch_out)
                    synchronization_entropies.append(synch_entropy)

                if tracking_data is not None:
                    tracking_data_list.append(tracking_data)

        # Convert to numpy arrays
        predictions = np.array(predictions)  # [n_samples, batch, 1]
        certainties = np.array(certainties)  # [n_samples, batch, 2]

        # Compute statistics for first sample in batch
        pred_samples = predictions[:, 0, 0]  # [n_samples]
        cert_samples = certainties[:, 0, :]  # [n_samples, 2]

        # Point estimates
        mean_prediction = np.mean(pred_samples)
        mean_confidence = np.mean(cert_samples[:, 1])

        # Synchronization-based epistemic uncertainty
        if synchronization_entropies:
            mean_synch_entropy = np.mean(synchronization_entropies)
            synch_entropy_variance = np.var(synchronization_entropies)
        else:
            mean_synch_entropy = 0.5  # Neutral uncertainty
            synch_entropy_variance = 0.0

        # Traditional prediction variance (for comparison)
        prediction_variance = np.var(pred_samples)

        # Temporal analysis of synchronization stability
        stability_metrics = {}
        if self.temporal_analysis and tracking_data_list:
            # Analyze first tracking data sample
            stability_metrics = self._analyze_synchronization_stability(
                tracking_data_list[0]
            )

            # Use temporal instability as additional uncertainty signal
            temporal_uncertainty = stability_metrics.get(
                "temporal_instability", 0.0)
        else:
            temporal_uncertainty = 0.0

        # Combine different uncertainty sources
        # Epistemic: synchronization entropy + prediction variance
        epistemic_uncertainty = mean_synch_entropy + prediction_variance

        # Aleatoric: individual model uncertainties + temporal instability
        individual_uncertainties = 1.0 - cert_samples[:, 1]
        aleatoric_uncertainty = np.mean(
            individual_uncertainties) + temporal_uncertainty

        # Total uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty

        # Confidence intervals based on prediction samples
        confidence_level = kwargs.get("confidence_level", 0.95)
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(pred_samples, lower_percentile)
        ci_upper = np.percentile(pred_samples, upper_percentile)

        # Create uncertainty estimate with CTM-specific metadata
        estimate = UncertaintyEstimate(
            prediction=float(mean_prediction),
            confidence=float(mean_confidence),
            epistemic_uncertainty=float(epistemic_uncertainty),
            aleatoric_uncertainty=float(aleatoric_uncertainty),
            total_uncertainty=float(total_uncertainty),
            confidence_interval_lower=float(ci_lower),
            confidence_interval_upper=float(ci_upper),
            confidence_level=confidence_level,
            method="ctm_synchronization",
            n_samples=self.n_samples,
        )

        # Add CTM-specific uncertainty metadata
        estimate.synchronization_entropy = mean_synch_entropy
        estimate.synchronization_entropy_variance = synch_entropy_variance
        estimate.prediction_variance = prediction_variance
        estimate.stability_metrics = stability_metrics

        return estimate


class NeuronLevelUncertainty(UncertaintyQuantifier):
    """
    Uncertainty quantification based on neuron-level model dynamics.

    Analyzes the behavior of individual neuron-level models (NLMs) to estimate
    uncertainty based on:
    - Neuron activation diversity
    - Memory utilization patterns
    - Individual neuron confidence
    """

    def __init__(
        self, analyze_memory: bool = True, neuron_diversity_threshold: float = 0.1
    ):
        """
        Initialize neuron-level uncertainty analysis.

        Args:
            analyze_memory: Whether to analyze memory utilization patterns
            neuron_diversity_threshold: Threshold for considering neurons as diverse
        """
        self.analyze_memory = analyze_memory
        self.neuron_diversity_threshold = neuron_diversity_threshold

    def _analyze_neuron_diversity(
        self, tracking_data: TrackingData
    ) -> Dict[str, float]:
        """
        Analyze diversity in neuron activation patterns.

        Args:
            tracking_data: CTM tracking data

        Returns:
            Dictionary with neuron diversity metrics
        """
        post_activations = (
            tracking_data.post_activations
        )  # [iterations, batch, neurons]

        # Compute pairwise correlations between neurons
        final_activations = post_activations[
            -1, 0, :
        ]  # [neurons] - final iteration, first batch

        # Compute activation diversity (inverse of mean pairwise correlation)
        if len(final_activations) > 1:
            # [neurons, iterations]
            activation_matrix = post_activations[:, 0, :].T
            correlation_matrix = np.corrcoef(activation_matrix)

            # Remove diagonal (self-correlations)
            mask = ~np.eye(correlation_matrix.shape[0], dtype=bool)
            off_diagonal_correlations = correlation_matrix[mask]

            mean_correlation = np.mean(np.abs(off_diagonal_correlations))
            diversity_score = 1.0 - mean_correlation
        else:
            diversity_score = 0.0

        # Compute activation entropy across neurons
        activation_probs = torch.softmax(
            torch.tensor(final_activations), dim=0)
        neuron_entropy = -torch.sum(
            activation_probs * torch.log(activation_probs + 1e-10)
        ).item()

        # Normalize by maximum possible entropy
        max_entropy = np.log(len(final_activations))
        normalized_neuron_entropy = (
            neuron_entropy / max_entropy if max_entropy > 0 else 0.0
        )

        return {
            "neuron_diversity": diversity_score,
            "neuron_entropy": normalized_neuron_entropy,
            "mean_correlation": mean_correlation if len(final_activations) > 1 else 0.0,
        }

    def _analyze_memory_utilization(
        self, tracking_data: TrackingData
    ) -> Dict[str, float]:
        """
        Analyze how effectively the CTM utilizes its memory.

        Args:
            tracking_data: CTM tracking data

        Returns:
            Dictionary with memory utilization metrics
        """
        pre_activations = tracking_data.pre_activations  # [iterations, batch, neurons]
        post_activations = (
            tracking_data.post_activations
        )  # [iterations, batch, neurons]

        # Memory utilization: how much do activations change over time?
        if pre_activations.shape[0] > 1:
            pre_changes = np.diff(
                pre_activations, axis=0
            )  # [iterations-1, batch, neurons]
            post_changes = np.diff(
                post_activations, axis=0
            )  # [iterations-1, batch, neurons]

            pre_change_magnitude = np.mean(np.abs(pre_changes))
            post_change_magnitude = np.mean(np.abs(post_changes))

            # Memory effectiveness: ratio of output change to input change
            memory_effectiveness = post_change_magnitude / (
                pre_change_magnitude + 1e-10
            )
        else:
            memory_effectiveness = 1.0
            pre_change_magnitude = 0.0
            post_change_magnitude = 0.0

        # Memory consistency: how stable are the memory patterns?
        memory_variance = np.var(post_activations, axis=0)  # [batch, neurons]
        mean_memory_variance = np.mean(memory_variance)

        return {
            "memory_effectiveness": memory_effectiveness,
            "memory_variance": mean_memory_variance,
            "pre_change_magnitude": pre_change_magnitude,
            "post_change_magnitude": post_change_magnitude,
        }

    def estimate_uncertainty(
        self, model: UDLRatingCTM, token_ids: torch.Tensor, **kwargs
    ) -> UncertaintyEstimate:
        """
        Estimate uncertainty using neuron-level model analysis.

        Args:
            model: Trained CTM model
            token_ids: Input token IDs [batch, seq_len]

        Returns:
            Uncertainty estimate based on neuron-level dynamics
        """
        model.eval()

        with torch.no_grad():
            # Single forward pass with tracking
            pred, cert, synch_out, tracking_data = model(token_ids, track=True)

        # Extract basic predictions
        prediction = pred[0, 0].item()  # First batch, single output
        confidence = torch.softmax(cert[0], dim=0)[1].item()  # Confident class

        if tracking_data is None:
            # Fallback to basic uncertainty if no tracking data
            return UncertaintyEstimate(
                prediction=prediction,
                confidence=confidence,
                epistemic_uncertainty=0.5,
                aleatoric_uncertainty=1.0 - confidence,
                total_uncertainty=0.5 + (1.0 - confidence),
                confidence_interval_lower=max(0.0, prediction - 0.1),
                confidence_interval_upper=min(1.0, prediction + 0.1),
                method="neuron_level_fallback",
                n_samples=1,
            )

        # Analyze neuron diversity
        diversity_metrics = self._analyze_neuron_diversity(tracking_data)

        # Analyze memory utilization
        memory_metrics = {}
        if self.analyze_memory:
            memory_metrics = self._analyze_memory_utilization(tracking_data)

        # Compute uncertainty based on neuron-level analysis

        # Epistemic uncertainty: low neuron diversity = high uncertainty
        neuron_diversity = diversity_metrics["neuron_diversity"]
        diversity_uncertainty = 1.0 - neuron_diversity

        # Memory-based uncertainty
        if memory_metrics:
            memory_variance = memory_metrics["memory_variance"]
            memory_uncertainty = min(1.0, memory_variance)  # Cap at 1.0
        else:
            memory_uncertainty = 0.0

        # Combine uncertainty sources
        epistemic_uncertainty = diversity_uncertainty + memory_uncertainty
        aleatoric_uncertainty = 1.0 - confidence
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty

        # Simple confidence intervals based on uncertainty
        uncertainty_margin = total_uncertainty * 0.1  # 10% of uncertainty as margin
        ci_lower = max(0.0, prediction - uncertainty_margin)
        ci_upper = min(1.0, prediction + uncertainty_margin)

        # Create uncertainty estimate
        estimate = UncertaintyEstimate(
            prediction=prediction,
            confidence=confidence,
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=aleatoric_uncertainty,
            total_uncertainty=total_uncertainty,
            confidence_interval_lower=ci_lower,
            confidence_interval_upper=ci_upper,
            method="ctm_neuron_level",
            n_samples=1,
        )

        # Add neuron-level specific metadata
        estimate.diversity_metrics = diversity_metrics
        estimate.memory_metrics = memory_metrics

        return estimate


class DeepEnsembleUncertainty(UncertaintyQuantifier):
    """
    Deep ensemble uncertainty quantification.

    Uses multiple independently trained models to estimate
    both epistemic and aleatoric uncertainty.
    """

    def __init__(self, models: List[UDLRatingCTM]):
        """
        Initialize deep ensemble uncertainty.

        Args:
            models: List of independently trained models
        """
        if not models:
            raise ValueError("At least one model is required")

        self.models = models
        self.n_models = len(models)

    def estimate_uncertainty(
        self, model: UDLRatingCTM, token_ids: torch.Tensor, **kwargs
    ) -> UncertaintyEstimate:
        """
        Estimate uncertainty using deep ensemble.

        Args:
            model: Primary model (not used, ensemble models are used instead)
            token_ids: Input token IDs

        Returns:
            Uncertainty estimate
        """
        device = token_ids.device

        predictions = []
        certainties = []

        # Get predictions from all ensemble members
        for ensemble_model in self.models:
            ensemble_model.eval()
            with torch.no_grad():
                pred, cert, _ = ensemble_model(
                    token_ids.to(next(ensemble_model.parameters()).device)
                )
                predictions.append(pred.cpu().numpy())
                certainties.append(cert.cpu().numpy())

        # Convert to numpy arrays
        predictions = np.array(predictions)  # [n_models, batch, 1]
        certainties = np.array(certainties)  # [n_models, batch, 2]

        # Compute statistics for first sample in batch
        pred_samples = predictions[:, 0, 0]  # [n_models]
        cert_samples = certainties[:, 0, :]  # [n_models, 2]

        # Point estimates
        mean_prediction = np.mean(pred_samples)
        mean_confidence = np.mean(cert_samples[:, 1])

        # Epistemic uncertainty (model disagreement)
        epistemic_uncertainty = np.var(pred_samples)

        # Aleatoric uncertainty (average individual uncertainty)
        individual_uncertainties = 1.0 - cert_samples[:, 1]
        aleatoric_uncertainty = np.mean(individual_uncertainties)

        # Total uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty

        # Confidence intervals
        confidence_level = kwargs.get("confidence_level", 0.95)
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(pred_samples, lower_percentile)
        ci_upper = np.percentile(pred_samples, upper_percentile)

        return UncertaintyEstimate(
            prediction=float(mean_prediction),
            confidence=float(mean_confidence),
            epistemic_uncertainty=float(epistemic_uncertainty),
            aleatoric_uncertainty=float(aleatoric_uncertainty),
            total_uncertainty=float(total_uncertainty),
            confidence_interval_lower=float(ci_lower),
            confidence_interval_upper=float(ci_upper),
            confidence_level=confidence_level,
            method="deep_ensemble",
            n_samples=self.n_models,
        )


class VariationalInference(UncertaintyQuantifier):
    """
    Variational inference for uncertainty quantification.

    Uses variational Bayesian neural networks to estimate
    parameter uncertainty.
    """

    def __init__(self, n_samples: int = 100):
        """
        Initialize variational inference.

        Args:
            n_samples: Number of samples from variational posterior
        """
        self.n_samples = n_samples

    def estimate_uncertainty(
        self, model: UDLRatingCTM, token_ids: torch.Tensor, **kwargs
    ) -> UncertaintyEstimate:
        """
        Estimate uncertainty using variational inference.

        Note: This is a simplified implementation. Full variational inference
        would require modifying the model architecture to use variational layers.

        Args:
            model: Trained CTM model
            token_ids: Input token IDs

        Returns:
            Uncertainty estimate
        """
        # For now, fall back to Monte Carlo Dropout
        # In a full implementation, this would use proper variational layers
        mc_dropout = MonteCarloDropout(self.n_samples)
        estimate = mc_dropout.estimate_uncertainty(model, token_ids, **kwargs)
        estimate.method = "variational_inference"

        return estimate


class CalibrationAnalyzer:
    """
    Analyzes and improves model calibration.

    Provides methods to measure calibration error and apply
    post-hoc calibration techniques.
    """

    def __init__(self):
        """Initialize calibration analyzer."""
        self.calibration_map = None
        self.is_fitted = False

    def compute_calibration_error(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        ground_truth: np.ndarray,
        n_bins: int = 10,
    ) -> Dict[str, float]:
        """
        Compute Expected Calibration Error (ECE) and other calibration metrics.

        Args:
            predictions: Model predictions [n_samples]
            confidences: Model confidences [n_samples]
            ground_truth: Ground truth labels [n_samples]
            n_bins: Number of bins for calibration

        Returns:
            Dictionary with calibration metrics
        """
        # Convert to binary accuracy (within threshold)
        threshold = 0.1
        accuracies = (np.abs(predictions - ground_truth)
                      < threshold).astype(float)

        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        mce = 0.0  # Maximum Calibration Error
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()

                # Calibration error for this bin
                bin_error = abs(avg_confidence_in_bin - accuracy_in_bin)
                ece += bin_error * prop_in_bin
                mce = max(mce, bin_error)

                bin_accuracies.append(accuracy_in_bin)
                bin_confidences.append(avg_confidence_in_bin)
                bin_counts.append(in_bin.sum())
            else:
                bin_accuracies.append(0.0)
                bin_confidences.append(0.0)
                bin_counts.append(0)

        # Additional metrics
        avg_confidence = np.mean(confidences)
        avg_accuracy = np.mean(accuracies)

        return {
            "expected_calibration_error": ece,
            "maximum_calibration_error": mce,
            "average_confidence": avg_confidence,
            "average_accuracy": avg_accuracy,
            "bin_accuracies": bin_accuracies,
            "bin_confidences": bin_confidences,
            "bin_counts": bin_counts,
        }

    def fit_calibration_map(
        self, confidences: np.ndarray, accuracies: np.ndarray, method: str = "isotonic"
    ) -> None:
        """
        Fit calibration mapping.

        Args:
            confidences: Model confidences
            accuracies: Binary accuracies
            method: Calibration method ('isotonic', 'platt')
        """
        if method == "isotonic":
            self.calibration_map = IsotonicRegression(out_of_bounds="clip")
            self.calibration_map.fit(confidences, accuracies)
        elif method == "platt":
            # Platt scaling using logistic regression
            self.calibration_map = LogisticRegression()
            self.calibration_map.fit(confidences.reshape(-1, 1), accuracies)
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        self.is_fitted = True
        logger.info(f"Fitted {method} calibration mapping")

    def apply_calibration(self, confidences: np.ndarray) -> np.ndarray:
        """
        Apply calibration mapping to confidences.

        Args:
            confidences: Raw model confidences

        Returns:
            Calibrated confidences
        """
        if not self.is_fitted:
            raise ValueError("Calibration map must be fitted first")

        if hasattr(self.calibration_map, "predict"):
            # Platt scaling
            return self.calibration_map.predict_proba(confidences.reshape(-1, 1))[:, 1]
        else:
            # Isotonic regression
            return self.calibration_map.predict(confidences)


class UncertaintyAwarePredictor:
    """
    Uncertainty-aware predictor that combines multiple uncertainty quantification methods.
    """

    def __init__(
        self,
        model: UDLRatingCTM,
        uncertainty_methods: Optional[List[UncertaintyQuantifier]] = None,
        calibration_analyzer: Optional[CalibrationAnalyzer] = None,
    ):
        """
        Initialize uncertainty-aware predictor.

        Args:
            model: Trained CTM model
            uncertainty_methods: List of uncertainty quantification methods
            calibration_analyzer: Calibration analyzer (optional)
        """
        self.model = model
        self.calibration_analyzer = calibration_analyzer

        # Default uncertainty methods
        if uncertainty_methods is None:
            self.uncertainty_methods = [
                MonteCarloDropout(n_samples=50),
            ]
        else:
            self.uncertainty_methods = uncertainty_methods

        logger.info(
            f"Initialized uncertainty-aware predictor with {len(self.uncertainty_methods)} methods"
        )

    def predict_with_uncertainty(
        self, token_ids: torch.Tensor, return_all_methods: bool = False
    ) -> Union[UncertaintyEstimate, Dict[str, UncertaintyEstimate]]:
        """
        Make prediction with comprehensive uncertainty quantification.

        Args:
            token_ids: Input token IDs
            return_all_methods: Whether to return results from all methods

        Returns:
            Uncertainty estimate(s)
        """
        estimates = {}

        # Get estimates from all methods
        for method in self.uncertainty_methods:
            estimate = method.estimate_uncertainty(self.model, token_ids)
            estimates[estimate.method] = estimate

        if return_all_methods:
            return estimates

        # Return the first method's estimate (or combine multiple methods)
        primary_estimate = list(estimates.values())[0]

        # Apply calibration if available
        if self.calibration_analyzer and self.calibration_analyzer.is_fitted:
            calibrated_confidence = self.calibration_analyzer.apply_calibration(
                np.array([primary_estimate.confidence])
            )[0]
            primary_estimate.confidence = float(calibrated_confidence)

        return primary_estimate

    def evaluate_uncertainty_quality(
        self, dataloader: DataLoader, ground_truth_fn: Callable
    ) -> Dict[str, Any]:
        """
        Evaluate quality of uncertainty estimates.

        Args:
            dataloader: Data loader for evaluation
            ground_truth_fn: Function to get ground truth labels

        Returns:
            Evaluation metrics
        """
        all_predictions = []
        all_confidences = []
        all_uncertainties = []
        all_ground_truth = []

        self.model.eval()

        with torch.no_grad():
            for token_ids, udl_representations in dataloader:
                for i, udl_repr in enumerate(udl_representations):
                    # Get uncertainty estimate
                    estimate = self.predict_with_uncertainty(
                        token_ids[i: i + 1])

                    # Get ground truth
                    gt = ground_truth_fn(udl_repr)

                    all_predictions.append(estimate.prediction)
                    all_confidences.append(estimate.confidence)
                    all_uncertainties.append(estimate.total_uncertainty)
                    all_ground_truth.append(gt)

        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        confidences = np.array(all_confidences)
        uncertainties = np.array(all_uncertainties)
        ground_truth = np.array(all_ground_truth)

        # Compute evaluation metrics

        # 1. Calibration metrics
        accuracies = (np.abs(predictions - ground_truth) < 0.1).astype(float)
        calibration_metrics = (
            self.calibration_analyzer.compute_calibration_error(
                predictions, confidences, ground_truth
            )
            if self.calibration_analyzer
            else {}
        )

        # 2. Uncertainty quality metrics
        prediction_errors = np.abs(predictions - ground_truth)

        # Correlation between uncertainty and error
        uncertainty_error_correlation = np.corrcoef(uncertainties, prediction_errors)[
            0, 1
        ]

        # Area under the curve for uncertainty-based rejection
        sorted_indices = np.argsort(uncertainties)[
            ::-1
        ]  # Sort by uncertainty (descending)
        sorted_errors = prediction_errors[sorted_indices]

        # Compute rejection curve (fraction of data rejected vs. remaining error)
        rejection_fractions = np.linspace(0, 1, 100)
        remaining_errors = []

        for frac in rejection_fractions:
            n_reject = int(frac * len(sorted_errors))
            if n_reject < len(sorted_errors):
                remaining_error = np.mean(sorted_errors[n_reject:])
            else:
                remaining_error = 0.0
            remaining_errors.append(remaining_error)

        # Area under rejection curve (lower is better)
        auc_rejection = np.trapz(remaining_errors, rejection_fractions)

        # 3. Reliability metrics
        reliability_score = 1.0 - np.mean(np.abs(confidences - accuracies))

        evaluation_results = {
            "calibration_metrics": calibration_metrics,
            "uncertainty_error_correlation": uncertainty_error_correlation,
            "auc_rejection_curve": auc_rejection,
            "reliability_score": reliability_score,
            "mean_prediction_error": np.mean(prediction_errors),
            "mean_uncertainty": np.mean(uncertainties),
            "mean_confidence": np.mean(confidences),
        }

        logger.info(
            f"Uncertainty evaluation completed. "
            f"Correlation: {uncertainty_error_correlation:.3f}, "
            f"Reliability: {reliability_score:.3f}"
        )

        return evaluation_results

    def save_calibration_data(self, filepath: str):
        """
        Save calibration analyzer.

        Args:
            filepath: Path to save calibration data
        """
        if self.calibration_analyzer:
            torch.save(self.calibration_analyzer, filepath)
            logger.info(f"Saved calibration data to {filepath}")

    def load_calibration_data(self, filepath: str):
        """
        Load calibration analyzer.

        Args:
            filepath: Path to calibration data
        """
        self.calibration_analyzer = torch.load(
            filepath, map_location="cpu", weights_only=False
        )
        logger.info(f"Loaded calibration data from {filepath}")


def create_uncertainty_quantifier(
    method: str = "monte_carlo_dropout", **kwargs
) -> UncertaintyQuantifier:
    """
    Factory function to create uncertainty quantifier.

    Args:
        method: Uncertainty quantification method
        **kwargs: Method-specific arguments

    Returns:
        Uncertainty quantifier instance
    """
    if method == "monte_carlo_dropout":
        return MonteCarloDropout(**kwargs)
    elif method == "deep_ensemble":
        return DeepEnsembleUncertainty(**kwargs)
    elif method == "variational_inference":
        return VariationalInference(**kwargs)
    else:
        raise ValueError(
            f"Unknown uncertainty quantification method: {method}")


def bootstrap_confidence_intervals(
    predictions: np.ndarray, n_bootstrap: int = 1000, confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence intervals for predictions.

    Args:
        predictions: Array of predictions
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95%)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    bootstrap_means = []

    for _ in range(n_bootstrap):
        # Bootstrap sample
        bootstrap_sample = np.random.choice(
            predictions, size=len(predictions), replace=True
        )
        bootstrap_means.append(np.mean(bootstrap_sample))

    # Compute confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)

    return lower_bound, upper_bound
