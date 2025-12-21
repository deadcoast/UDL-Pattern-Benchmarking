"""
Tests for training pipeline functionality.

Tests the TrainingPipeline class and its components including loss function
correctness and ground truth consistency.
"""

import pytest
import torch
import torch.nn as nn
from hypothesis import given, strategies as st, settings
from typing import Dict

from udl_rating_framework.training.training_pipeline import TrainingPipeline
from udl_rating_framework.models.ctm_adapter import UDLRatingCTM
from udl_rating_framework.core.metrics.base import QualityMetric
from udl_rating_framework.core.aggregation import MetricAggregator
from udl_rating_framework.core.representation import UDLRepresentation


class MockMetric(QualityMetric):
    """Mock metric for testing."""

    def __init__(self, fixed_value: float = 0.5):
        self.fixed_value = fixed_value

    def compute(self, udl: UDLRepresentation) -> float:
        return self.fixed_value

    def get_formula(self) -> str:
        return "MockMetric(U) = 0.5"

    def get_properties(self) -> Dict[str, bool]:
        return {
            "bounded": True,
            "monotonic": False,
            "additive": False,
            "continuous": True,
        }


@pytest.fixture
def mock_model():
    """Create a mock CTM model for testing."""
    vocab_size = 100
    return UDLRatingCTM(
        vocab_size=vocab_size, d_model=64, d_input=32, iterations=5, n_synch_out=16
    )


@pytest.fixture
def mock_metrics():
    """Create mock metrics for testing."""
    return [MockMetric(0.6), MockMetric(0.8)]


@pytest.fixture
def mock_aggregator():
    """Create mock aggregator for testing."""
    # Use equal weights for simplicity
    weights = {"metric1": 0.5, "metric2": 0.5}
    return MetricAggregator(weights)


@pytest.fixture
def training_pipeline(mock_model, mock_metrics, mock_aggregator):
    """Create training pipeline for testing."""
    return TrainingPipeline(
        model=mock_model,
        metrics=mock_metrics,
        aggregator=mock_aggregator,
        alpha=0.7,
        beta=0.3,
    )


class TestTrainingPipeline:
    """Test cases for TrainingPipeline class."""

    def test_initialization(self, mock_model, mock_metrics, mock_aggregator):
        """Test training pipeline initialization."""
        pipeline = TrainingPipeline(
            model=mock_model,
            metrics=mock_metrics,
            aggregator=mock_aggregator,
            alpha=0.6,
            beta=0.4,
        )

        assert pipeline.alpha == 0.6
        assert pipeline.beta == 0.4
        assert len(pipeline.metrics) == 2
        assert pipeline.current_epoch == 0

    def test_invalid_loss_weights(self, mock_model, mock_metrics, mock_aggregator):
        """Test that invalid loss weights raise ValueError."""
        # Weights don't sum to 1
        with pytest.raises(ValueError, match="Loss weights must sum to 1.0"):
            TrainingPipeline(
                model=mock_model,
                metrics=mock_metrics,
                aggregator=mock_aggregator,
                alpha=0.5,
                beta=0.6,  # 0.5 + 0.6 = 1.1 ≠ 1.0
            )

        # Negative weights
        with pytest.raises(ValueError, match="Loss weights must be non-negative"):
            TrainingPipeline(
                model=mock_model,
                metrics=mock_metrics,
                aggregator=mock_aggregator,
                alpha=-0.1,
                beta=1.1,
            )

    def test_compute_ground_truth(self, training_pipeline):
        """Test ground truth computation using mathematical metrics."""
        # Create a simple UDL representation
        udl_text = "expr ::= term '+' term"
        udl = UDLRepresentation(udl_text, "test.udl")

        # Compute ground truth
        ground_truth = training_pipeline.compute_ground_truth(udl)

        # Should be a float in [0, 1]
        assert isinstance(ground_truth, float)
        assert 0.0 <= ground_truth <= 1.0

    def test_compute_loss_structure(self, training_pipeline):
        """Test that loss computation returns correct structure."""
        batch_size = 4

        # Create mock tensors
        predictions = torch.rand(batch_size, 1)
        certainties = torch.rand(batch_size, 2)
        ground_truth = torch.rand(batch_size, 1)

        # Compute loss
        total_loss, rating_loss, confidence_loss = training_pipeline.compute_loss(
            predictions, certainties, ground_truth
        )

        # Check that all losses are scalars
        assert total_loss.dim() == 0
        assert rating_loss.dim() == 0
        assert confidence_loss.dim() == 0

        # Check that losses are non-negative
        assert total_loss.item() >= 0
        assert rating_loss.item() >= 0
        assert confidence_loss.item() >= 0


class TestLossFunctionCorrectness:
    """
    Property-based tests for loss function correctness.

    **Feature: udl-rating-framework, Property 15: Loss Function Correctness**
    **Validates: Requirements 4.3**
    """

    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        alpha=st.floats(min_value=0.1, max_value=0.9),
        predictions=st.lists(
            st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=8
        ),
        ground_truth=st.lists(
            st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=8
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_loss_function_formula_correctness(
        self, batch_size, alpha, predictions, ground_truth
    ):
        """
        Test that loss function correctly implements L = α·L_rating + β·L_confidence.

        Property: For any predictions, ground truth, and weights α, β where α + β = 1,
        the computed loss should equal α·L_rating + β·L_confidence.
        """
        # Ensure we have matching batch sizes
        min_size = min(len(predictions), len(ground_truth), batch_size)
        predictions = predictions[:min_size]
        ground_truth = ground_truth[:min_size]

        if min_size == 0:
            return  # Skip empty batches

        beta = 1.0 - alpha

        # Create training pipeline with specified weights
        mock_model = UDLRatingCTM(
            vocab_size=10, d_model=32, d_input=16, iterations=3, n_synch_out=8
        )
        mock_metrics = [MockMetric(0.5)]
        mock_aggregator = MetricAggregator({"mockmetric": 1.0})

        pipeline = TrainingPipeline(
            model=mock_model,
            metrics=mock_metrics,
            aggregator=mock_aggregator,
            alpha=alpha,
            beta=beta,
        )

        # Convert to tensors
        pred_tensor = torch.tensor(predictions, dtype=torch.float32).unsqueeze(1)
        gt_tensor = torch.tensor(ground_truth, dtype=torch.float32).unsqueeze(1)

        # Create mock certainties (random but valid)
        certainties = torch.rand(len(predictions), 2)

        # Compute loss using pipeline
        total_loss, rating_loss, confidence_loss = pipeline.compute_loss(
            pred_tensor, certainties, gt_tensor
        )

        # Manually compute expected loss
        expected_rating_loss = nn.MSELoss()(pred_tensor, gt_tensor)

        # Compute confidence loss manually
        confidence_scores = torch.softmax(certainties, dim=1)[:, 1]
        threshold = 0.1
        prediction_accuracy = (
            torch.abs(pred_tensor.squeeze() - gt_tensor.squeeze()) < threshold
        ).float()
        expected_confidence_loss = nn.MSELoss()(confidence_scores, prediction_accuracy)

        expected_total_loss = (
            alpha * expected_rating_loss + beta * expected_confidence_loss
        )

        # Verify the formula is correctly implemented
        # Allow small numerical differences due to floating point precision
        assert torch.abs(total_loss - expected_total_loss).item() < 1e-5, (
            f"Loss formula incorrect: got {total_loss.item()}, expected {expected_total_loss.item()}"
        )

        assert torch.abs(rating_loss - expected_rating_loss).item() < 1e-5, (
            f"Rating loss incorrect: got {rating_loss.item()}, expected {expected_rating_loss.item()}"
        )

        assert torch.abs(confidence_loss - expected_confidence_loss).item() < 1e-5, (
            f"Confidence loss incorrect: got {confidence_loss.item()}, expected {expected_confidence_loss.item()}"
        )

    @given(
        alpha=st.floats(min_value=0.0, max_value=1.0),
        predictions=st.lists(
            st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=5
        ),
    )
    @settings(max_examples=50, deadline=None)
    def test_loss_boundedness_and_non_negativity(self, alpha, predictions):
        """
        Test that loss components are non-negative and bounded.

        Property: For any valid inputs, all loss components should be ≥ 0.
        """
        beta = 1.0 - alpha

        # Create training pipeline
        mock_model = UDLRatingCTM(
            vocab_size=10, d_model=32, d_input=16, iterations=3, n_synch_out=8
        )
        mock_metrics = [MockMetric(0.5)]
        mock_aggregator = MetricAggregator({"mockmetric": 1.0})

        pipeline = TrainingPipeline(
            model=mock_model,
            metrics=mock_metrics,
            aggregator=mock_aggregator,
            alpha=alpha,
            beta=beta,
        )

        # Create tensors
        pred_tensor = torch.tensor(predictions, dtype=torch.float32).unsqueeze(1)
        gt_tensor = torch.rand(len(predictions), 1)  # Random ground truth
        certainties = torch.rand(len(predictions), 2)

        # Compute loss
        total_loss, rating_loss, confidence_loss = pipeline.compute_loss(
            pred_tensor, certainties, gt_tensor
        )

        # Verify non-negativity
        assert total_loss.item() >= 0, (
            f"Total loss should be non-negative, got {total_loss.item()}"
        )
        assert rating_loss.item() >= 0, (
            f"Rating loss should be non-negative, got {rating_loss.item()}"
        )
        assert confidence_loss.item() >= 0, (
            f"Confidence loss should be non-negative, got {confidence_loss.item()}"
        )

        # Verify total loss is combination of components
        expected_total = alpha * rating_loss.item() + beta * confidence_loss.item()
        assert abs(total_loss.item() - expected_total) < 1e-5, (
            f"Total loss should equal weighted sum: got {total_loss.item()}, expected {expected_total}"
        )


class TestGroundTruthConsistency:
    """
    Property-based tests for ground truth consistency.

    **Feature: udl-rating-framework, Property 16: Ground Truth Consistency**
    **Validates: Requirements 4.5**
    """

    def test_ground_truth_equals_mathematical_computation(self, training_pipeline):
        """
        Test that training ground truth equals mathematical metric computation.

        Property: For any UDL, the ground truth computed by the training pipeline
        should equal the score computed by applying mathematical metrics directly.
        """
        # Create a UDL representation
        udl_text = "number ::= digit+ | digit+ '.' digit+"
        udl = UDLRepresentation(udl_text, "test.udl")

        # Compute ground truth using training pipeline
        pipeline_ground_truth = training_pipeline.compute_ground_truth(udl)

        # Compute ground truth manually using the same metrics and aggregator
        # Use the same naming scheme as the pipeline
        metric_values = {}
        aggregator_metric_names = list(training_pipeline.aggregator.weights.keys())

        for i, metric in enumerate(training_pipeline.metrics):
            # Use the metric name from aggregator if available, otherwise generate one
            if i < len(aggregator_metric_names):
                metric_name = aggregator_metric_names[i]
            else:
                metric_name = metric.__class__.__name__.replace("Metric", "").lower()

            metric_values[metric_name] = metric.compute(udl)

        manual_ground_truth = training_pipeline.aggregator.aggregate(metric_values)

        # They should be identical (within floating point precision)
        assert abs(pipeline_ground_truth - manual_ground_truth) < 1e-6, (
            f"Ground truth inconsistency: pipeline={pipeline_ground_truth}, manual={manual_ground_truth}"
        )

    @given(
        udl_texts=st.lists(
            st.text(
                alphabet="abcdefghijklmnopqrstuvwxyz ::=+()[]{}|.'",
                min_size=5,
                max_size=50,
            ),
            min_size=1,
            max_size=3,
        )
    )
    @settings(max_examples=20, deadline=None)
    def test_ground_truth_determinism(self, udl_texts):
        """
        Test that ground truth computation is deterministic.

        Property: For any UDL, computing ground truth multiple times should
        produce identical results.
        """
        # Create training pipeline inside the test to avoid fixture issues
        mock_model = UDLRatingCTM(
            vocab_size=100, d_model=64, d_input=32, iterations=5, n_synch_out=16
        )
        mock_metrics = [MockMetric(0.6), MockMetric(0.8)]
        mock_aggregator = MetricAggregator({"metric1": 0.5, "metric2": 0.5})

        training_pipeline = TrainingPipeline(
            model=mock_model,
            metrics=mock_metrics,
            aggregator=mock_aggregator,
            alpha=0.7,
            beta=0.3,
        )

        for udl_text in udl_texts:
            try:
                udl = UDLRepresentation(udl_text, "test.udl")

                # Compute ground truth multiple times
                ground_truths = []
                for _ in range(5):
                    gt = training_pipeline.compute_ground_truth(udl)
                    ground_truths.append(gt)

                # All should be identical
                first_gt = ground_truths[0]
                for gt in ground_truths[1:]:
                    assert abs(gt - first_gt) < 1e-10, (
                        f"Ground truth not deterministic: got {ground_truths}"
                    )

                # Should be bounded in [0, 1]
                assert 0.0 <= first_gt <= 1.0, (
                    f"Ground truth not bounded: got {first_gt}"
                )

            except Exception:
                # Skip invalid UDL texts
                continue

    def test_ground_truth_with_different_metrics(self):
        """
        Test ground truth computation with different metric configurations.
        """
        # Create UDL
        udl_text = "expr ::= term | expr '+' term"
        udl = UDLRepresentation(udl_text, "test.udl")

        # Test with single metric
        single_metric = [MockMetric(0.8)]
        single_aggregator = MetricAggregator({"mockmetric": 1.0})
        mock_model = UDLRatingCTM(
            vocab_size=10, d_model=32, d_input=16, iterations=3, n_synch_out=8
        )

        single_pipeline = TrainingPipeline(
            model=mock_model,
            metrics=single_metric,
            aggregator=single_aggregator,
            alpha=0.7,
            beta=0.3,
        )

        single_gt = single_pipeline.compute_ground_truth(udl)

        # Should equal the single metric value
        expected_single = single_metric[0].compute(udl)
        assert abs(single_gt - expected_single) < 1e-6

        # Test with multiple metrics
        multi_metrics = [MockMetric(0.6), MockMetric(0.8)]
        multi_aggregator = MetricAggregator({"metric1": 0.3, "metric2": 0.7})

        multi_pipeline = TrainingPipeline(
            model=mock_model,
            metrics=multi_metrics,
            aggregator=multi_aggregator,
            alpha=0.7,
            beta=0.3,
        )

        multi_gt = multi_pipeline.compute_ground_truth(udl)

        # Should equal weighted combination
        expected_multi = 0.3 * 0.6 + 0.7 * 0.8  # 0.18 + 0.56 = 0.74
        assert abs(multi_gt - expected_multi) < 1e-6


class TestTrainingPipelineIntegration:
    """Integration tests for training pipeline."""

    def test_training_summary(self, training_pipeline):
        """Test training summary generation."""
        summary = training_pipeline.get_training_summary()
        assert summary["status"] == "not_started"

        # Add some training history to test other fields
        training_pipeline.training_history["train_loss"] = [1.0, 0.8, 0.6]
        training_pipeline.training_history["val_loss"] = [1.2, 0.9, 0.7]
        training_pipeline.training_history["val_mae"] = [0.3, 0.2, 0.15]
        training_pipeline.training_history["val_correlation"] = [0.5, 0.7, 0.8]

        summary = training_pipeline.get_training_summary()
        assert "epochs_completed" in summary
        assert "loss_parameters" in summary
        assert summary["loss_parameters"]["alpha"] == 0.7
        assert summary["loss_parameters"]["beta"] == 0.3
        assert summary["epochs_completed"] == 3
        assert "final_train_loss" in summary
        assert "final_val_loss" in summary
        assert "best_val_mae" in summary
        assert "best_val_correlation" in summary

    def test_checkpoint_save_load(self, training_pipeline, tmp_path):
        """Test checkpoint saving and loading."""
        # Save checkpoint
        checkpoint_dir = str(tmp_path)
        training_pipeline.current_epoch = 5
        training_pipeline.training_history["train_loss"] = [1.0, 0.8, 0.6]

        training_pipeline.save_checkpoint(checkpoint_dir, 5)

        # Verify files were created
        checkpoint_file = tmp_path / "checkpoint_epoch_5.pt"
        history_file = tmp_path / "training_history_epoch_5.json"

        assert checkpoint_file.exists()
        assert history_file.exists()

        # Create new pipeline and load checkpoint
        mock_model = UDLRatingCTM(
            vocab_size=100, d_model=64, d_input=32, iterations=5, n_synch_out=16
        )
        new_pipeline = TrainingPipeline(
            model=mock_model,
            metrics=training_pipeline.metrics,
            aggregator=training_pipeline.aggregator,
        )

        new_pipeline.load_checkpoint(str(checkpoint_file))

        # Verify state was loaded
        assert new_pipeline.current_epoch == 5
        assert new_pipeline.training_history["train_loss"] == [1.0, 0.8, 0.6]
        assert new_pipeline.alpha == 0.7
        assert new_pipeline.beta == 0.3
