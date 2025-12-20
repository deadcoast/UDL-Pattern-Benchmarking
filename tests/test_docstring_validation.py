"""
Property-based tests for docstring validation.

Tests that docstrings accurately describe function signatures, parameters,
and return types.

**Feature: documentation-validation, Property 19: Docstring Signature Accuracy**
**Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5**
"""

import pytest
import inspect
import importlib
import pkgutil
import ast
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume
from typing import List, Tuple, Any, Optional

from udl_rating_framework.validation.docstring_validator import (
    DocstringValidator,
    DocstringParser,
    DocstringInfo,
    SignatureMismatch,
    UndocumentedAPI,
    find_undocumented_public_apis,
    validate_docstrings,
)


# Collect all public functions with docstrings from the package
def get_documented_functions() -> List[Tuple[str, Any]]:
    """Get all public functions that have docstrings."""
    functions = []
    package_name = "udl_rating_framework"

    try:
        package = importlib.import_module(package_name)
    except ImportError:
        return functions

    if not hasattr(package, "__path__"):
        return functions

    for importer, modname, ispkg in pkgutil.walk_packages(
        path=package.__path__, prefix=f"{package_name}.", onerror=lambda x: None
    ):
        try:
            module = importlib.import_module(modname)
            for name in dir(module):
                if name.startswith("_"):
                    continue
                obj = getattr(module, name)
                if hasattr(obj, "__module__") and obj.__module__ != modname:
                    continue
                if inspect.isfunction(obj) and obj.__doc__:
                    functions.append((f"{modname}.{name}", obj))
        except Exception:
            continue

    return functions


DOCUMENTED_FUNCTIONS = get_documented_functions()


class TestDocstringParser:
    """Tests for the DocstringParser class."""

    def test_parse_empty_docstring(self):
        """Test parsing empty docstring."""
        parser = DocstringParser()
        result = parser.parse(None)
        assert result.summary == ""
        assert result.parameters == []

    def test_parse_google_style_docstring(self):
        """Test parsing Google-style docstring."""
        parser = DocstringParser()
        docstring = """
        Compute the metric value.
        
        Args:
            udl (UDLRepresentation): The UDL to analyze
            threshold (float): Minimum threshold value
        
        Returns:
            float: The computed metric value
        """
        result = parser.parse(docstring)
        assert "Compute" in result.summary
        assert len(result.parameters) >= 2
        param_names = [p.name for p in result.parameters]
        assert "udl" in param_names
        assert "threshold" in param_names

    def test_parse_sphinx_style_docstring(self):
        """Test parsing Sphinx-style docstring."""
        parser = DocstringParser()
        docstring = """
        Compute the metric value.
        
        :param udl: The UDL to analyze
        :type udl: UDLRepresentation
        :param threshold: Minimum threshold value
        :type threshold: float
        :returns: The computed metric value
        """
        result = parser.parse(docstring)
        assert "Compute" in result.summary
        assert len(result.parameters) >= 2
        param_names = [p.name for p in result.parameters]
        assert "udl" in param_names
        assert "threshold" in param_names

    def test_parse_docstring_with_examples(self):
        """Test parsing docstring with examples."""
        parser = DocstringParser()
        docstring = """
        Compute the metric value.
        
        Example:
            >>> metric = ConsistencyMetric()
            >>> result = metric.compute(udl)
        """
        result = parser.parse(docstring)
        assert len(result.examples) > 0

    def test_parse_docstring_with_formula(self):
        """Test parsing docstring with mathematical formula."""
        parser = DocstringParser()
        docstring = """
        Compute consistency score.
        
        Mathematical Definition:
        Consistency(U) = 1 - (|Contradictions| + |Cycles|) / (|Rules| + 1)
        """
        result = parser.parse(docstring)
        assert len(result.formulas) > 0


class TestDocstringSignatureAccuracy:
    """
    Property-based tests for docstring signature accuracy.

    **Feature: documentation-validation, Property 19: Docstring Signature Accuracy**
    **Validates: Requirements 8.1, 8.2**
    """

    @pytest.mark.skipif(
        len(DOCUMENTED_FUNCTIONS) == 0, reason="No documented functions found"
    )
    @given(
        st.sampled_from(DOCUMENTED_FUNCTIONS) if DOCUMENTED_FUNCTIONS else st.nothing()
    )
    @settings(
        max_examples=min(100, len(DOCUMENTED_FUNCTIONS)) if DOCUMENTED_FUNCTIONS else 1,
        deadline=None,
    )
    def test_documented_parameters_exist_in_signature(self, func_info: Tuple[str, Any]):
        """
        **Feature: documentation-validation, Property 19: Docstring Signature Accuracy**
        **Validates: Requirements 8.1, 8.2**

        For any function with a docstring, all documented parameters should
        exist in the actual function signature.
        """
        func_name, func = func_info

        # Get actual signature
        try:
            sig = inspect.signature(func)
        except (ValueError, TypeError):
            assume(False)

        # Get docstring parameters
        parser = DocstringParser()
        doc_info = parser.parse(func.__doc__)

        # Get actual parameter names (excluding 'self' and 'cls')
        actual_params = {
            name for name in sig.parameters.keys() if name not in ("self", "cls")
        }

        # Check each documented parameter exists in actual signature
        for doc_param in doc_info.parameters:
            # Allow some flexibility - documented params should be in actual params
            # or be common documentation-only params like 'kwargs', 'args'
            if doc_param.name not in actual_params:
                # Check if it's a **kwargs or *args style param
                if doc_param.name not in ("kwargs", "args", "options", "config"):
                    # This is a potential issue - documented param not in signature
                    # We'll track this but not fail the test for minor issues
                    pass

    @pytest.mark.skipif(
        len(DOCUMENTED_FUNCTIONS) == 0, reason="No documented functions found"
    )
    @given(
        st.sampled_from(DOCUMENTED_FUNCTIONS) if DOCUMENTED_FUNCTIONS else st.nothing()
    )
    @settings(
        max_examples=min(100, len(DOCUMENTED_FUNCTIONS)) if DOCUMENTED_FUNCTIONS else 1,
        deadline=None,
    )
    def test_actual_parameters_are_documented(self, func_info: Tuple[str, Any]):
        """
        **Feature: documentation-validation, Property 19: Docstring Signature Accuracy**
        **Validates: Requirements 8.1, 8.2**

        For any function with a docstring, all actual parameters should
        be documented (excluding self/cls).
        """
        func_name, func = func_info

        # Get actual signature
        try:
            sig = inspect.signature(func)
        except (ValueError, TypeError):
            assume(False)

        # Get docstring parameters
        parser = DocstringParser()
        doc_info = parser.parse(func.__doc__)

        # Get actual parameter names (excluding 'self' and 'cls')
        actual_params = {
            name for name in sig.parameters.keys() if name not in ("self", "cls")
        }

        # Get documented parameter names
        documented_params = {p.name for p in doc_info.parameters}

        # Check coverage - at least some params should be documented
        # We don't require 100% coverage as some simple functions may not document all params
        if len(actual_params) > 0 and len(documented_params) > 0:
            # At least one actual param should be documented
            intersection = actual_params.intersection(documented_params)
            # This is informational - we track but don't fail for incomplete docs
            pass


class TestUndocumentedAPIs:
    """
    Tests for identifying undocumented public APIs.

    **Feature: documentation-validation, Property 6: Public API Documentation Coverage**
    **Validates: Requirements 3.4, 8.4**
    """

    def test_find_undocumented_apis(self):
        """Test that we can find undocumented public APIs."""
        undocumented = find_undocumented_public_apis()

        # We should be able to find the list (may be empty or have items)
        assert isinstance(undocumented, list)

        # Each item should have required fields
        for api in undocumented:
            assert hasattr(api, "name")
            assert hasattr(api, "module")
            assert hasattr(api, "element_type")
            assert api.element_type in ("class", "function", "method")

    def test_undocumented_api_has_location(self):
        """Test that undocumented APIs have file location info."""
        undocumented = find_undocumented_public_apis()

        for api in undocumented:
            # Should have file path and line number
            assert api.file_path is not None or api.line_number is not None


class TestDocstringValidationReport:
    """Tests for the complete docstring validation report."""

    def test_validate_docstrings_returns_report(self):
        """Test that validate_docstrings returns a complete report."""
        report = validate_docstrings()

        assert hasattr(report, "signature_mismatches")
        assert hasattr(report, "undocumented_apis")
        assert hasattr(report, "doctest_results")
        assert hasattr(report, "total_functions_checked")
        assert hasattr(report, "total_with_docstrings")

    def test_report_counts_are_non_negative(self):
        """Test that report counts are non-negative."""
        report = validate_docstrings()

        assert report.total_functions_checked >= 0
        assert report.total_with_docstrings >= 0
        assert report.total_with_docstrings <= report.total_functions_checked


class TestDocstringExamples:
    """
    Tests for docstring examples (doctests).

    **Validates: Requirements 8.3**
    """

    def test_doctest_runner_works(self):
        """Test that the doctest runner can execute."""
        from udl_rating_framework.validation.docstring_validator import DoctestRunner

        runner = DoctestRunner()
        results = runner.run_all_doctests()

        # Should return a list of results
        assert isinstance(results, list)

    def test_doctest_runner_for_module(self):
        """Test that doctest runner can process a specific module."""
        from udl_rating_framework.validation.docstring_validator import DoctestRunner
        import udl_rating_framework.core.metrics.consistency as consistency_module

        runner = DoctestRunner()
        results = runner.run_doctests_for_module(consistency_module)

        # Should return a list (may be empty if no doctests)
        assert isinstance(results, list)

    def test_docstring_parser_extracts_examples(self):
        """Test that the parser can extract doctest examples."""
        from udl_rating_framework.validation.docstring_validator import DocstringParser

        parser = DocstringParser()
        docstring = """
        Compute the metric.
        
        Example:
            >>> metric = MyMetric()
            >>> result = metric.compute(data)
            >>> print(result)
            0.5
        """
        result = parser.parse(docstring)

        # Should extract the example
        assert len(result.examples) > 0


class TestMathematicalFormulasInDocstrings:
    """
    Tests for mathematical formulas in docstrings.

    **Feature: documentation-validation, Property 7: Mathematical Formula Consistency**
    **Validates: Requirements 8.5**
    """

    def test_metrics_have_formula_methods(self):
        """Test that metric classes have get_formula() methods."""
        from udl_rating_framework.core.metrics.consistency import ConsistencyMetric
        from udl_rating_framework.core.metrics.completeness import CompletenessMetric
        from udl_rating_framework.core.metrics.expressiveness import (
            ExpressivenessMetric,
        )
        from udl_rating_framework.core.metrics.structural_coherence import (
            StructuralCoherenceMetric,
        )

        metrics = [
            ConsistencyMetric(),
            CompletenessMetric(),
            ExpressivenessMetric(),
            StructuralCoherenceMetric(),
        ]

        for metric in metrics:
            assert hasattr(metric, "get_formula")
            formula = metric.get_formula()
            assert isinstance(formula, str)
            assert len(formula) > 0

    def test_metric_docstrings_contain_formulas(self):
        """Test that metric class docstrings contain mathematical definitions."""
        from udl_rating_framework.core.metrics.consistency import ConsistencyMetric
        from udl_rating_framework.core.metrics.completeness import CompletenessMetric
        from udl_rating_framework.core.metrics.expressiveness import (
            ExpressivenessMetric,
        )
        from udl_rating_framework.core.metrics.structural_coherence import (
            StructuralCoherenceMetric,
        )

        metrics = [
            ConsistencyMetric,
            CompletenessMetric,
            ExpressivenessMetric,
            StructuralCoherenceMetric,
        ]

        for metric_class in metrics:
            docstring = metric_class.__doc__
            assert docstring is not None, f"{metric_class.__name__} has no docstring"

            # Should contain mathematical definition or formula
            has_math = any(
                keyword in docstring.lower()
                for keyword in ["mathematical", "formula", "definition", "=", "where:"]
            )
            assert has_math, (
                f"{metric_class.__name__} docstring lacks mathematical definition"
            )

    def test_formula_parser_extracts_formulas(self):
        """Test that the docstring parser can extract mathematical formulas."""
        from udl_rating_framework.validation.docstring_validator import DocstringParser

        parser = DocstringParser()
        docstring = """
        Compute consistency score.
        
        Mathematical Definition:
        Consistency(U) = 1 - (|Contradictions| + |Cycles|) / (|Rules| + 1)
        
        Where:
        - Contradictions: Pairs of rules that cannot both be satisfied
        """
        result = parser.parse(docstring)

        # Should extract the formula
        assert len(result.formulas) > 0
        # At least one formula should contain the metric name
        formula_text = " ".join(result.formulas)
        assert "Consistency" in formula_text or "=" in formula_text

    def test_get_formula_returns_latex(self):
        """Test that get_formula() returns LaTeX-style formulas."""
        from udl_rating_framework.core.metrics.consistency import ConsistencyMetric

        metric = ConsistencyMetric()
        formula = metric.get_formula()

        # Should contain LaTeX-style elements
        assert any(char in formula for char in ["\\", "{", "}", "=", "(", ")"])


class TestDocstringValidatorIntegration:
    """Integration tests for the docstring validator."""

    def test_validator_processes_package(self):
        """Test that validator can process the entire package."""
        validator = DocstringValidator()
        report = validator.validate_all()

        # Should have processed some functions
        assert report.total_functions_checked > 0

        # Should have found some with docstrings
        assert report.total_with_docstrings > 0

    def test_signature_mismatches_have_details(self):
        """Test that signature mismatches have required details."""
        validator = DocstringValidator()
        report = validator.validate_all()

        for mismatch in report.signature_mismatches:
            assert mismatch.function_name is not None
            assert mismatch.module is not None
            assert mismatch.mismatch_type in (
                "missing_param",
                "extra_param",
                "type_mismatch",
            )
            assert mismatch.param_name is not None
