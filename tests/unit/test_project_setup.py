"""
Unit tests for project setup.

Tests package imports and dependency availability.
Validates: Requirements 11.7
"""

import pytest
import sys
import importlib


class TestPackageImports:
    """Test that all package modules can be imported."""

    def test_main_package_import(self):
        """Test that the main package can be imported."""
        import udl_rating_framework

        assert udl_rating_framework is not None
        assert hasattr(udl_rating_framework, "__version__")

    def test_core_module_import(self):
        """Test that core module can be imported."""
        import udl_rating_framework.core

        assert udl_rating_framework.core is not None

    def test_metrics_module_import(self):
        """Test that metrics module can be imported."""
        import udl_rating_framework.core.metrics

        assert udl_rating_framework.core.metrics is not None

    def test_models_module_import(self):
        """Test that models module can be imported."""
        import udl_rating_framework.models

        assert udl_rating_framework.models is not None

    def test_io_module_import(self):
        """Test that io module can be imported."""
        import udl_rating_framework.io

        assert udl_rating_framework.io is not None

    def test_evaluation_module_import(self):
        """Test that evaluation module can be imported."""
        import udl_rating_framework.evaluation

        assert udl_rating_framework.evaluation is not None

    def test_utils_module_import(self):
        """Test that utils module can be imported."""
        import udl_rating_framework.utils

        assert udl_rating_framework.utils is not None

    def test_cli_module_import(self):
        """Test that cli module can be imported."""
        import udl_rating_framework.cli

        assert udl_rating_framework.cli is not None


class TestCoreComponents:
    """Test that core components can be imported."""

    def test_udl_representation_import(self):
        """Test that UDLRepresentation can be imported."""
        from udl_rating_framework.core.representation import UDLRepresentation

        assert UDLRepresentation is not None

    def test_quality_metric_import(self):
        """Test that QualityMetric base class can be imported."""
        from udl_rating_framework.core.metrics.base import QualityMetric

        assert QualityMetric is not None

    def test_metric_aggregator_import(self):
        """Test that MetricAggregator can be imported."""
        from udl_rating_framework.core.aggregation import MetricAggregator

        assert MetricAggregator is not None

    def test_confidence_calculator_import(self):
        """Test that ConfidenceCalculator can be imported."""
        from udl_rating_framework.core.confidence import ConfidenceCalculator

        assert ConfidenceCalculator is not None


class TestDependencyAvailability:
    """Test that all required dependencies are available."""

    def test_torch_available(self):
        """Test that PyTorch is available."""
        try:
            import torch

            assert torch is not None
            # Check version is 2.0 or higher
            version = torch.__version__.split(".")
            major = int(version[0])
            assert (
                major >= 2
            ), f"PyTorch version {torch.__version__} is too old, need >= 2.0"
        except ImportError:
            pytest.fail("PyTorch is not installed")

    def test_networkx_available(self):
        """Test that NetworkX is available."""
        try:
            import networkx as nx

            assert nx is not None
            # Check version is 3.0 or higher
            version = nx.__version__.split(".")
            major = int(version[0])
            assert (
                major >= 3
            ), f"NetworkX version {nx.__version__} is too old, need >= 3.0"
        except ImportError:
            pytest.fail("NetworkX is not installed")

    def test_numpy_available(self):
        """Test that NumPy is available."""
        try:
            import numpy as np

            assert np is not None
            # Check version is 1.24 or higher
            version = np.__version__.split(".")
            major, minor = int(version[0]), int(version[1])
            assert major > 1 or (
                major == 1 and minor >= 24
            ), f"NumPy version {np.__version__} is too old, need >= 1.24"
        except ImportError:
            pytest.fail("NumPy is not installed")

    def test_scipy_available(self):
        """Test that SciPy is available."""
        try:
            import scipy

            assert scipy is not None
            # Check version is 1.10 or higher
            version = scipy.__version__.split(".")
            major, minor = int(version[0]), int(version[1])
            assert major > 1 or (
                major == 1 and minor >= 10
            ), f"SciPy version {scipy.__version__} is too old, need >= 1.10"
        except ImportError:
            pytest.fail("SciPy is not installed")

    def test_hypothesis_available(self):
        """Test that Hypothesis is available."""
        try:
            import hypothesis

            assert hypothesis is not None
            # Check version is 6.0 or higher
            version = hypothesis.__version__.split(".")
            major = int(version[0])
            assert (
                major >= 6
            ), f"Hypothesis version {hypothesis.__version__} is too old, need >= 6.0"
        except ImportError:
            pytest.fail("Hypothesis is not installed")


class TestBasicFunctionality:
    """Test basic functionality of core components."""

    def test_udl_representation_instantiation(self):
        """Test that UDLRepresentation can be instantiated."""
        from udl_rating_framework.core.representation import UDLRepresentation

        udl = UDLRepresentation("test source", "test.udl")
        assert udl is not None
        assert udl.source_text == "test source"
        assert udl.file_path == "test.udl"

    def test_metric_aggregator_instantiation(self):
        """Test that MetricAggregator can be instantiated with valid weights."""
        from udl_rating_framework.core.aggregation import MetricAggregator

        weights = {
            "consistency": 0.3,
            "completeness": 0.25,
            "expressiveness": 0.25,
            "structural_coherence": 0.2,
        }

        aggregator = MetricAggregator(weights)
        assert aggregator is not None
        assert aggregator.weights == weights

    def test_metric_aggregator_weight_validation(self):
        """Test that MetricAggregator validates weight constraints."""
        from udl_rating_framework.core.aggregation import MetricAggregator

        # Test weights that don't sum to 1
        with pytest.raises(ValueError, match="must sum to 1.0"):
            MetricAggregator({"metric1": 0.5, "metric2": 0.3})

        # Test negative weights
        with pytest.raises(ValueError, match="must be non-negative"):
            MetricAggregator({"metric1": 0.6, "metric2": -0.1, "metric3": 0.5})

    def test_confidence_calculator_instantiation(self):
        """Test that ConfidenceCalculator can be instantiated."""
        from udl_rating_framework.core.confidence import ConfidenceCalculator

        calculator = ConfidenceCalculator()
        assert calculator is not None

    def test_confidence_calculator_basic_computation(self):
        """Test basic confidence computation."""
        from udl_rating_framework.core.confidence import ConfidenceCalculator
        import numpy as np

        calculator = ConfidenceCalculator()

        # Test with uniform distribution (low confidence)
        uniform_probs = np.array([0.25, 0.25, 0.25, 0.25])
        confidence = calculator.compute_confidence(uniform_probs)
        assert 0.0 <= confidence <= 1.0
        assert confidence < 0.5  # Uniform should have low confidence

        # Test with delta distribution (high confidence)
        delta_probs = np.array([1.0, 0.0, 0.0, 0.0])
        confidence = calculator.compute_confidence(delta_probs)
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.9  # Delta should have high confidence


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
