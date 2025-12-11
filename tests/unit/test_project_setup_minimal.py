"""
Minimal unit tests for project setup (no external dependencies).

Tests package imports without requiring numpy/torch to be functional.
Validates: Requirements 11.7
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_package_structure():
    """Test that all package modules exist and can be imported."""
    print("Testing package structure...")

    # Test main package
    try:
        import udl_rating_framework

        assert udl_rating_framework is not None
        assert hasattr(udl_rating_framework, "__version__")
        print(f"✓ Main package (version {udl_rating_framework.__version__})")
    except Exception as e:
        print(f"✗ Main package import failed: {e}")
        return False

    # Test all submodules
    modules = [
        "udl_rating_framework.core",
        "udl_rating_framework.core.metrics",
        "udl_rating_framework.models",
        "udl_rating_framework.io",
        "udl_rating_framework.evaluation",
        "udl_rating_framework.utils",
        "udl_rating_framework.cli",
    ]

    for module_name in modules:
        try:
            module = __import__(module_name, fromlist=[""])
            assert module is not None
            print(f"✓ {module_name}")
        except Exception as e:
            print(f"✗ {module_name} import failed: {e}")
            return False

    return True


def test_core_classes():
    """Test that core classes can be imported."""
    print("\nTesting core classes...")

    classes_to_test = [
        ("udl_rating_framework.core.representation", "UDLRepresentation"),
        ("udl_rating_framework.core.representation", "Token"),
        ("udl_rating_framework.core.metrics.base", "QualityMetric"),
        ("udl_rating_framework.core.aggregation", "MetricAggregator"),
    ]

    for module_name, class_name in classes_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            assert cls is not None
            print(f"✓ {class_name}")
        except Exception as e:
            print(f"✗ {class_name} import failed: {e}")
            return False

    return True


def test_basic_instantiation():
    """Test basic instantiation without numpy."""
    print("\nTesting basic instantiation...")

    # Test UDLRepresentation
    try:
        from udl_rating_framework.core.representation import UDLRepresentation

        udl = UDLRepresentation("test source", "test.udl")
        assert udl.source_text == "test source"
        assert udl.file_path == "test.udl"
        print("✓ UDLRepresentation instantiation")
    except Exception as e:
        print(f"✗ UDLRepresentation instantiation failed: {e}")
        return False

    # Test Token
    try:
        from udl_rating_framework.core.representation import Token

        token = Token(text="keyword", type="KEYWORD",
                      position=0, line=1, column=1)
        assert token.text == "keyword"
        assert token.type == "KEYWORD"
        print("✓ Token instantiation")
    except Exception as e:
        print(f"✗ Token instantiation failed: {e}")
        return False

    # Test MetricAggregator
    try:
        from udl_rating_framework.core.aggregation import MetricAggregator

        weights = {
            "consistency": 0.3,
            "completeness": 0.25,
            "expressiveness": 0.25,
            "structural_coherence": 0.2,
        }
        aggregator = MetricAggregator(weights)
        assert aggregator.weights == weights
        print("✓ MetricAggregator instantiation")
    except Exception as e:
        print(f"✗ MetricAggregator instantiation failed: {e}")
        return False

    return True


def test_weight_validation():
    """Test MetricAggregator weight validation."""
    print("\nTesting weight validation...")

    from udl_rating_framework.core.aggregation import MetricAggregator

    # Test weights that don't sum to 1
    try:
        MetricAggregator({"metric1": 0.5, "metric2": 0.3})
        print("✗ Should reject weights that don't sum to 1")
        return False
    except ValueError as e:
        if "sum to 1.0" in str(e):
            print("✓ Correctly rejects invalid weight sum")
        else:
            print(f"✗ Wrong error message: {e}")
            return False

    # Test negative weights
    try:
        MetricAggregator({"metric1": 0.6, "metric2": -0.1, "metric3": 0.5})
        print("✗ Should reject negative weights")
        return False
    except ValueError as e:
        if "non-negative" in str(e):
            print("✓ Correctly rejects negative weights")
        else:
            print(f"✗ Wrong error message: {e}")
            return False

    # Test valid weights
    try:
        MetricAggregator({"m1": 0.5, "m2": 0.5})
        print("✓ Accepts valid weights")
    except Exception as e:
        print(f"✗ Should accept valid weights: {e}")
        return False

    return True


def test_aggregation_computation():
    """Test MetricAggregator computation."""
    print("\nTesting aggregation computation...")

    from udl_rating_framework.core.aggregation import MetricAggregator

    try:
        weights = {"m1": 0.3, "m2": 0.7}
        aggregator = MetricAggregator(weights)

        metric_values = {"m1": 0.8, "m2": 0.6}
        result = aggregator.aggregate(metric_values)

        expected = 0.3 * 0.8 + 0.7 * 0.6  # 0.24 + 0.42 = 0.66
        assert abs(
            result - expected) < 1e-6, f"Expected {expected}, got {result}"
        print(f"✓ Aggregation computation (result: {result:.3f})")
    except Exception as e:
        print(f"✗ Aggregation computation failed: {e}")
        return False

    return True


def test_file_structure():
    """Test that required files exist."""
    print("\nTesting file structure...")

    required_files = [
        "udl_rating_framework/__init__.py",
        "udl_rating_framework/core/__init__.py",
        "udl_rating_framework/core/metrics/__init__.py",
        "udl_rating_framework/core/representation.py",
        "udl_rating_framework/core/metrics/base.py",
        "udl_rating_framework/core/aggregation.py",
        "udl_rating_framework/core/confidence.py",
        "requirements-udl.txt",
        "setup.py",
        "docs/mathematical_framework.tex",
        "docs/README.md",
    ]

    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} not found")
            return False

    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("UDL Rating Framework - Project Setup Tests (Minimal)")
    print("=" * 70)

    all_passed = True

    tests = [
        test_package_structure,
        test_core_classes,
        test_basic_instantiation,
        test_weight_validation,
        test_aggregation_computation,
        test_file_structure,
    ]

    for test_func in tests:
        if not test_func():
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ All tests passed!")
        print("=" * 70)
        print("\nNote: NumPy/PyTorch dependency tests skipped due to architecture")
        print("mismatch in the current environment. These dependencies will be")
        print("tested when the framework is used in a properly configured environment.")
        return 0
    else:
        print("✗ Some tests failed")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
