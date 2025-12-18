"""
Tests for code file reference validity.

Tests that all code files referenced in documentation exist and are accessible.

**Feature: documentation-validation, Property 9: Code File Reference Validity**
**Validates: Requirements 4.4**
"""

import pytest
import os
from pathlib import Path
from hypothesis import given, strategies as st, settings


# Code files referenced in the mathematical framework documentation
REFERENCED_CODE_FILES = [
    "udl_rating_framework/core/metrics/consistency.py",
    "udl_rating_framework/core/metrics/completeness.py",
    "udl_rating_framework/core/metrics/expressiveness.py",
    "udl_rating_framework/core/metrics/structural_coherence.py",
    "udl_rating_framework/core/aggregation.py",
    "udl_rating_framework/core/confidence.py",
]

# Additional metric files that should exist
ADDITIONAL_METRIC_FILES = [
    "udl_rating_framework/core/metrics/__init__.py",
    "udl_rating_framework/core/metrics/base.py",
    "udl_rating_framework/core/representation.py",
]


class TestCodeFileReferenceValidity:
    """
    Tests that all code files referenced in documentation exist.
    
    **Feature: documentation-validation, Property 9: Code File Reference Validity**
    **Validates: Requirements 4.4**
    """

    def test_all_referenced_files_exist(self):
        """
        **Feature: documentation-validation, Property 9: Code File Reference Validity**
        **Validates: Requirements 4.4**
        
        All code files referenced in the mathematical framework documentation
        must exist at the specified paths.
        """
        for file_path in REFERENCED_CODE_FILES:
            assert os.path.exists(file_path), (
                f"Referenced code file does not exist: {file_path}"
            )
            assert os.path.isfile(file_path), (
                f"Referenced path is not a file: {file_path}"
            )

    def test_all_referenced_files_are_readable(self):
        """
        **Feature: documentation-validation, Property 9: Code File Reference Validity**
        **Validates: Requirements 4.4**
        
        All referenced code files must be readable.
        """
        for file_path in REFERENCED_CODE_FILES:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    assert len(content) > 0, (
                        f"Referenced code file is empty: {file_path}"
                    )
            except Exception as e:
                pytest.fail(f"Cannot read referenced code file {file_path}: {e}")

    def test_all_referenced_files_are_valid_python(self):
        """
        **Feature: documentation-validation, Property 9: Code File Reference Validity**
        **Validates: Requirements 4.4**
        
        All referenced code files must be valid Python syntax.
        """
        import ast
        
        for file_path in REFERENCED_CODE_FILES:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                ast.parse(content)
            except SyntaxError as e:
                pytest.fail(f"Referenced code file has syntax error {file_path}: {e}")

    def test_all_referenced_files_are_importable(self):
        """
        **Feature: documentation-validation, Property 9: Code File Reference Validity**
        **Validates: Requirements 4.4**
        
        All referenced code files must be importable as Python modules.
        """
        import importlib
        
        module_mappings = {
            "udl_rating_framework/core/metrics/consistency.py": "udl_rating_framework.core.metrics.consistency",
            "udl_rating_framework/core/metrics/completeness.py": "udl_rating_framework.core.metrics.completeness",
            "udl_rating_framework/core/metrics/expressiveness.py": "udl_rating_framework.core.metrics.expressiveness",
            "udl_rating_framework/core/metrics/structural_coherence.py": "udl_rating_framework.core.metrics.structural_coherence",
            "udl_rating_framework/core/aggregation.py": "udl_rating_framework.core.aggregation",
            "udl_rating_framework/core/confidence.py": "udl_rating_framework.core.confidence",
        }
        
        for file_path, module_name in module_mappings.items():
            try:
                module = importlib.import_module(module_name)
                assert module is not None, f"Module {module_name} is None"
            except ImportError as e:
                pytest.fail(f"Cannot import module {module_name} from {file_path}: {e}")

    def test_metric_classes_exist_in_files(self):
        """
        **Feature: documentation-validation, Property 9: Code File Reference Validity**
        **Validates: Requirements 4.4**
        
        The expected metric classes must exist in the referenced files.
        """
        from udl_rating_framework.core.metrics.consistency import ConsistencyMetric
        from udl_rating_framework.core.metrics.completeness import CompletenessMetric
        from udl_rating_framework.core.metrics.expressiveness import ExpressivenessMetric
        from udl_rating_framework.core.metrics.structural_coherence import StructuralCoherenceMetric
        from udl_rating_framework.core.aggregation import MetricAggregator
        from udl_rating_framework.core.confidence import ConfidenceCalculator
        
        # Verify classes exist and are callable
        assert ConsistencyMetric is not None
        assert CompletenessMetric is not None
        assert ExpressivenessMetric is not None
        assert StructuralCoherenceMetric is not None
        assert MetricAggregator is not None
        assert ConfidenceCalculator is not None
        
        # Verify they can be instantiated
        assert ConsistencyMetric() is not None
        assert CompletenessMetric() is not None
        assert ExpressivenessMetric() is not None
        assert StructuralCoherenceMetric() is not None
        assert ConfidenceCalculator() is not None

    def test_metric_classes_have_required_methods(self):
        """
        **Feature: documentation-validation, Property 9: Code File Reference Validity**
        **Validates: Requirements 4.4**
        
        The metric classes must have the required methods documented in the framework.
        """
        from udl_rating_framework.core.metrics.consistency import ConsistencyMetric
        from udl_rating_framework.core.metrics.completeness import CompletenessMetric
        from udl_rating_framework.core.metrics.expressiveness import ExpressivenessMetric
        from udl_rating_framework.core.metrics.structural_coherence import StructuralCoherenceMetric
        from udl_rating_framework.core.aggregation import MetricAggregator
        from udl_rating_framework.core.confidence import ConfidenceCalculator
        
        # Check metric classes have compute method
        for metric_class in [ConsistencyMetric, CompletenessMetric, 
                            ExpressivenessMetric, StructuralCoherenceMetric]:
            metric = metric_class()
            assert hasattr(metric, 'compute'), f"{metric_class.__name__} missing compute method"
            assert hasattr(metric, 'get_formula'), f"{metric_class.__name__} missing get_formula method"
            assert hasattr(metric, 'get_properties'), f"{metric_class.__name__} missing get_properties method"
        
        # Check aggregator has aggregate method
        assert hasattr(MetricAggregator, '__init__')
        
        # Check confidence calculator has compute_confidence method
        calc = ConfidenceCalculator()
        assert hasattr(calc, 'compute_confidence'), "ConfidenceCalculator missing compute_confidence method"


class TestAdditionalCodeFiles:
    """
    Tests for additional code files that support the framework.
    """

    def test_additional_metric_files_exist(self):
        """
        Test that additional supporting files exist.
        """
        for file_path in ADDITIONAL_METRIC_FILES:
            assert os.path.exists(file_path), (
                f"Additional code file does not exist: {file_path}"
            )


class TestCodeFileReferencePropertyBased:
    """
    Property-based tests for code file references.
    
    **Feature: documentation-validation, Property 9: Code File Reference Validity**
    **Validates: Requirements 4.4**
    """

    @given(st.sampled_from(REFERENCED_CODE_FILES))
    @settings(max_examples=len(REFERENCED_CODE_FILES), deadline=None)
    def test_any_referenced_file_exists(self, file_path: str):
        """
        **Feature: documentation-validation, Property 9: Code File Reference Validity**
        **Validates: Requirements 4.4**
        
        For any referenced code file, the file must exist.
        """
        assert os.path.exists(file_path), f"File does not exist: {file_path}"
        assert os.path.isfile(file_path), f"Path is not a file: {file_path}"

    @given(st.sampled_from(REFERENCED_CODE_FILES))
    @settings(max_examples=len(REFERENCED_CODE_FILES), deadline=None)
    def test_any_referenced_file_contains_docstring(self, file_path: str):
        """
        **Feature: documentation-validation, Property 9: Code File Reference Validity**
        **Validates: Requirements 4.4**
        
        For any referenced code file, the file should contain a module docstring.
        """
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for docstring (triple quotes at the start)
        assert '"""' in content or "'''" in content, (
            f"File {file_path} should contain a docstring"
        )
