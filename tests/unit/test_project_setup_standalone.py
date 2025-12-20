"""
Standalone unit tests for project setup (no pytest dependency).

Tests package imports and dependency availability.
Validates: Requirements 11.7
"""

import sys
import importlib
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_package_imports():
    """Test that all package modules can be imported."""
    print("Testing package imports...")

    # Test main package
    try:
        import udl_rating_framework

        assert udl_rating_framework is not None
        assert hasattr(udl_rating_framework, "__version__")
        print("✓ Main package import successful")
    except Exception as e:
        print(f"✗ Main package import failed: {e}")
        return False

    # Test core module
    try:
        import udl_rating_framework.core

        assert udl_rating_framework.core is not None
        print("✓ Core module import successful")
    except Exception as e:
        print(f"✗ Core module import failed: {e}")
        return False

    # Test metrics module
    try:
        import udl_rating_framework.core.metrics

        assert udl_rating_framework.core.metrics is not None
        print("✓ Metrics module import successful")
    except Exception as e:
        print(f"✗ Metrics module import failed: {e}")
        return False

    # Test models module
    try:
        import udl_rating_framework.models

        assert udl_rating_framework.models is not None
        print("✓ Models module import successful")
    except Exception as e:
        print(f"✗ Models module import failed: {e}")
        return False

    # Test io module
    try:
        import udl_rating_framework.io

        assert udl_rating_framework.io is not None
        print("✓ IO module import successful")
    except Exception as e:
        print(f"✗ IO module import failed: {e}")
        return False

    # Test evaluation module
    try:
        import udl_rating_framework.evaluation

        assert udl_rating_framework.evaluation is not None
        print("✓ Evaluation module import successful")
    except Exception as e:
        print(f"✗ Evaluation module import failed: {e}")
        return False

    # Test utils module
    try:
        import udl_rating_framework.utils

        assert udl_rating_framework.utils is not None
        print("✓ Utils module import successful")
    except Exception as e:
        print(f"✗ Utils module import failed: {e}")
        return False

    # Test cli module
    try:
        import udl_rating_framework.cli

        assert udl_rating_framework.cli is not None
        print("✓ CLI module import successful")
    except Exception as e:
        print(f"✗ CLI module import failed: {e}")
        return False

    return True


def test_core_components():
    """Test that core components can be imported."""
    print("\nTesting core components...")

    # Test UDLRepresentation
    try:
        from udl_rating_framework.core.representation import UDLRepresentation

        assert UDLRepresentation is not None
        print("✓ UDLRepresentation import successful")
    except Exception as e:
        print(f"✗ UDLRepresentation import failed: {e}")
        return False

    # Test QualityMetric
    try:
        from udl_rating_framework.core.metrics.base import QualityMetric

        assert QualityMetric is not None
        print("✓ QualityMetric import successful")
    except Exception as e:
        print(f"✗ QualityMetric import failed: {e}")
        return False

    # Test MetricAggregator
    try:
        from udl_rating_framework.core.aggregation import MetricAggregator

        assert MetricAggregator is not None
        print("✓ MetricAggregator import successful")
    except Exception as e:
        print(f"✗ MetricAggregator import failed: {e}")
        return False

    # Test ConfidenceCalculator
    try:
        from udl_rating_framework.core.confidence import ConfidenceCalculator

        assert ConfidenceCalculator is not None
        print("✓ ConfidenceCalculator import successful")
    except Exception as e:
        print(f"✗ ConfidenceCalculator import failed: {e}")
        return False

    return True


def test_dependency_availability():
    """Test that all required dependencies are available."""
    print("\nTesting dependency availability...")

    # Test NumPy (required for confidence calculator)
    try:
        import numpy as np

        assert np is not None
        version = np.__version__.split(".")
        major, minor = int(version[0]), int(version[1])
        assert major > 1 or (major == 1 and minor >= 24), (
            f"NumPy version {np.__version__} is too old, need >= 1.24"
        )
        print(f"✓ NumPy {np.__version__} available")
    except ImportError:
        print("✗ NumPy is not installed")
        return False
    except Exception as e:
        print(f"✗ NumPy check failed: {e}")
        return False

    # Note: We skip torch, networkx, scipy, hypothesis checks since they may not be
    # installed yet or may have architecture issues. These will be tested when needed.

    return True


def test_basic_functionality():
    """Test basic functionality of core components."""
    print("\nTesting basic functionality...")

    # Test UDLRepresentation instantiation
    try:
        from udl_rating_framework.core.representation import UDLRepresentation

        udl = UDLRepresentation("test source", "test.udl")
        assert udl is not None
        assert udl.source_text == "test source"
        assert udl.file_path == "test.udl"
        print("✓ UDLRepresentation instantiation successful")
    except Exception as e:
        print(f"✗ UDLRepresentation instantiation failed: {e}")
        return False

    # Test MetricAggregator instantiation
    try:
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
        print("✓ MetricAggregator instantiation successful")
    except Exception as e:
        print(f"✗ MetricAggregator instantiation failed: {e}")
        return False

    # Test MetricAggregator weight validation
    try:
        from udl_rating_framework.core.aggregation import MetricAggregator

        # Test weights that don't sum to 1
        try:
            MetricAggregator({"metric1": 0.5, "metric2": 0.3})
            print("✗ MetricAggregator should reject weights that don't sum to 1")
            return False
        except ValueError:
            print("✓ MetricAggregator correctly rejects invalid weight sum")

        # Test negative weights
        try:
            MetricAggregator({"metric1": 0.6, "metric2": -0.1, "metric3": 0.5})
            print("✗ MetricAggregator should reject negative weights")
            return False
        except ValueError:
            print("✓ MetricAggregator correctly rejects negative weights")
    except Exception as e:
        print(f"✗ MetricAggregator validation test failed: {e}")
        return False

    # Test ConfidenceCalculator
    try:
        from udl_rating_framework.core.confidence import ConfidenceCalculator
        import numpy as np

        calculator = ConfidenceCalculator()
        assert calculator is not None

        # Test with uniform distribution (low confidence)
        uniform_probs = np.array([0.25, 0.25, 0.25, 0.25])
        confidence = calculator.compute_confidence(uniform_probs)
        assert 0.0 <= confidence <= 1.0
        assert confidence < 0.5, (
            f"Uniform distribution should have low confidence, got {confidence}"
        )
        print(f"✓ ConfidenceCalculator uniform distribution: {confidence:.3f}")

        # Test with delta distribution (high confidence)
        delta_probs = np.array([1.0, 0.0, 0.0, 0.0])
        confidence = calculator.compute_confidence(delta_probs)
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.9, (
            f"Delta distribution should have high confidence, got {confidence}"
        )
        print(f"✓ ConfidenceCalculator delta distribution: {confidence:.3f}")
    except Exception as e:
        print(f"✗ ConfidenceCalculator test failed: {e}")
        return False

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("UDL Rating Framework - Project Setup Tests")
    print("=" * 60)

    all_passed = True

    if not test_package_imports():
        all_passed = False

    if not test_core_components():
        all_passed = False

    if not test_dependency_availability():
        all_passed = False

    if not test_basic_functionality():
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed!")
        print("=" * 60)
        return 0
    else:
        print("✗ Some tests failed")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
