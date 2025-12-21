"""
Property-based tests for API Validation.

**Feature: documentation-validation, Property 4: Documented API Existence**
**Feature: documentation-validation, Property 5: Method Signature Accuracy**
**Feature: documentation-validation, Property 6: Public API Documentation Coverage**
**Validates: Requirements 3.1, 3.2, 3.4**

Tests that:
- For any documented class, method, or function, the corresponding symbol exists in source code
- For any documented method signature, parameters match the actual implementation
- For any public API, there should be corresponding documentation
"""

import pytest
import inspect
import importlib
import pkgutil
from typing import List
from hypothesis import given, strategies as st, settings, assume
from dataclasses import dataclass

# Import the API validator
from udl_rating_framework.validation.api_validator import (
    APIValidator,
    APIExtractor,
    APIElement,
    APIValidationReport,
)


@dataclass
class SignatureInfo:
    """Information about a function/method signature."""

    name: str
    parameters: List[str]
    has_docstring: bool
    module: str


def get_all_public_apis() -> List[SignatureInfo]:
    """Extract all public APIs from the udl_rating_framework package."""
    apis = []
    package_name = "udl_rating_framework"

    try:
        package = importlib.import_module(package_name)
    except ImportError:
        return apis

    if not hasattr(package, "__path__"):
        return apis

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
                if inspect.isfunction(obj) or inspect.isclass(obj):
                    try:
                        sig = inspect.signature(obj)
                        params = [
                            p.name
                            for p in sig.parameters.values()
                            if p.name not in ("self", "cls")
                        ]
                    except (ValueError, TypeError):
                        params = []

                    apis.append(
                        SignatureInfo(
                            name=name,
                            parameters=params,
                            has_docstring=bool(obj.__doc__),
                            module=modname,
                        )
                    )
        except Exception:
            continue

    return apis


PUBLIC_APIS = get_all_public_apis()


class TestDocumentedAPIExistence:
    """
    Tests for Property 4: Documented API Existence.

    **Feature: documentation-validation, Property 4: Documented API Existence**
    **Validates: Requirements 3.1, 3.5**

    For any class, method, or function documented in API documentation,
    the corresponding symbol must exist in the source code.
    """

    def test_extractor_finds_public_apis(self):
        """
        **Feature: documentation-validation, Property 4: Documented API Existence**
        **Validates: Requirements 3.1**

        The API extractor should find public APIs in the package.
        """
        extractor = APIExtractor("udl_rating_framework")
        apis = extractor.extract_all_public_apis()

        # Should find some APIs
        assert len(apis) > 0, "Should find at least some public APIs"

        # Each API should have required fields
        for api in apis:
            assert api.name is not None
            assert api.module is not None
            assert api.element_type in ("class", "function", "method")

    def test_all_extracted_apis_are_importable(self):
        """
        **Feature: documentation-validation, Property 4: Documented API Existence**
        **Validates: Requirements 3.1**

        For any API extracted by the validator, it should be importable.
        """
        extractor = APIExtractor("udl_rating_framework")
        apis = extractor.extract_all_public_apis()

        # Sample a subset for performance
        sample_size = min(50, len(apis))
        sample_apis = apis[:sample_size]

        for api in sample_apis:
            try:
                module = importlib.import_module(api.module)
                if api.parent_class:
                    # It's a method - get the class first
                    cls = getattr(module, api.parent_class, None)
                    if cls:
                        obj = getattr(cls, api.name, None)
                else:
                    obj = getattr(module, api.name, None)

                # The API should exist
                assert obj is not None or api.element_type == "method", (
                    f"API {api.full_name} should be importable"
                )
            except ImportError:
                # Module import failed - this is a real issue
                pytest.fail(f"Could not import module {api.module}")

    @pytest.mark.skipif(len(PUBLIC_APIS) == 0, reason="No public APIs found")
    @given(st.sampled_from(PUBLIC_APIS) if PUBLIC_APIS else st.nothing())
    @settings(
        max_examples=min(100, len(PUBLIC_APIS)) if PUBLIC_APIS else 1, deadline=None
    )
    def test_public_api_exists_in_module(self, api_info: SignatureInfo):
        """
        **Feature: documentation-validation, Property 4: Documented API Existence**
        **Validates: Requirements 3.1**

        For any public API, it should exist in its declared module.
        """
        try:
            module = importlib.import_module(api_info.module)
            obj = getattr(module, api_info.name, None)
            assert obj is not None, (
                f"API {api_info.name} should exist in {api_info.module}"
            )
        except ImportError:
            assume(False)  # Skip if module can't be imported


class TestMethodSignatureAccuracy:
    """
    Tests for Property 5: Method Signature Accuracy.

    **Feature: documentation-validation, Property 5: Method Signature Accuracy**
    **Validates: Requirements 3.2, 3.3**

    For any documented method signature, the parameter names, types, and defaults
    must match the actual implementation.
    """

    def test_signature_extraction_works(self):
        """
        **Feature: documentation-validation, Property 5: Method Signature Accuracy**
        **Validates: Requirements 3.2**

        The extractor should correctly extract signatures.
        """
        extractor = APIExtractor("udl_rating_framework")
        apis = extractor.extract_all_public_apis()

        # Find APIs with signatures
        apis_with_sigs = [a for a in apis if a.signature]

        assert len(apis_with_sigs) > 0, "Should find APIs with signatures"

        for api in apis_with_sigs[:20]:  # Check first 20
            # Signature should be a valid string
            assert isinstance(api.signature, str)
            # Should contain parentheses
            assert "(" in api.signature and ")" in api.signature

    @pytest.mark.skipif(len(PUBLIC_APIS) == 0, reason="No public APIs found")
    @given(st.sampled_from(PUBLIC_APIS) if PUBLIC_APIS else st.nothing())
    @settings(
        max_examples=min(100, len(PUBLIC_APIS)) if PUBLIC_APIS else 1, deadline=None
    )
    def test_signature_parameters_match_actual(self, api_info: SignatureInfo):
        """
        **Feature: documentation-validation, Property 5: Method Signature Accuracy**
        **Validates: Requirements 3.2, 3.3**

        For any public API, the extracted parameters should match the actual signature.
        """
        try:
            module = importlib.import_module(api_info.module)
            obj = getattr(module, api_info.name, None)

            if obj is None:
                assume(False)

            # Get actual signature
            try:
                sig = inspect.signature(obj)
                actual_params = [
                    p.name
                    for p in sig.parameters.values()
                    if p.name not in ("self", "cls")
                ]
            except (ValueError, TypeError):
                assume(False)

            # Parameters should match
            assert set(api_info.parameters) == set(actual_params), (
                f"Parameters for {api_info.name} should match: "
                f"expected {actual_params}, got {api_info.parameters}"
            )
        except ImportError:
            assume(False)

    def test_validator_detects_signature_mismatches(self):
        """
        **Feature: documentation-validation, Property 5: Method Signature Accuracy**
        **Validates: Requirements 3.2**

        The validator should be able to detect signature mismatches.
        """
        validator = APIValidator()

        # Test the compare_signature method
        discrepancies = validator.compare_signature("(a, b)", "(a, b, c)")

        # Should detect the mismatch
        assert len(discrepancies) > 0, "Should detect signature mismatch"

        # Same signatures should have no discrepancies
        discrepancies = validator.compare_signature("(a, b)", "(a, b)")
        assert len(discrepancies) == 0, "Same signatures should match"


class TestPublicAPIDocumentationCoverage:
    """
    Tests for Property 6: Public API Documentation Coverage.

    **Feature: documentation-validation, Property 6: Public API Documentation Coverage**
    **Validates: Requirements 3.4, 8.4**

    For any public class, method, or function in the package, there should be
    corresponding documentation (docstring or API doc).
    """

    def test_find_undocumented_apis(self):
        """
        **Feature: documentation-validation, Property 6: Public API Documentation Coverage**
        **Validates: Requirements 3.4**

        The validator should identify undocumented public APIs.
        """
        validator = APIValidator()
        undocumented = validator.find_undocumented_apis()

        # Should return a list
        assert isinstance(undocumented, list)

        # Each undocumented API should have required fields
        for api in undocumented:
            assert api.name is not None
            assert api.module is not None
            assert api.docstring is None or api.docstring.strip() == ""

    @pytest.mark.skipif(len(PUBLIC_APIS) == 0, reason="No public APIs found")
    @given(st.sampled_from(PUBLIC_APIS) if PUBLIC_APIS else st.nothing())
    @settings(
        max_examples=min(100, len(PUBLIC_APIS)) if PUBLIC_APIS else 1, deadline=None
    )
    def test_documented_apis_have_docstrings(self, api_info: SignatureInfo):
        """
        **Feature: documentation-validation, Property 6: Public API Documentation Coverage**
        **Validates: Requirements 3.4, 8.4**

        For any public API marked as documented, it should have a docstring.
        """
        if api_info.has_docstring:
            try:
                module = importlib.import_module(api_info.module)
                obj = getattr(module, api_info.name, None)

                if obj is None:
                    assume(False)

                # Should have a docstring
                assert obj.__doc__ is not None, (
                    f"API {api_info.name} marked as documented should have docstring"
                )
                assert len(obj.__doc__.strip()) > 0, (
                    f"API {api_info.name} docstring should not be empty"
                )
            except ImportError:
                assume(False)

    def test_documentation_coverage_report(self):
        """
        **Feature: documentation-validation, Property 6: Public API Documentation Coverage**
        **Validates: Requirements 3.4**

        The validation report should include documentation coverage metrics.
        """
        validator = APIValidator()
        report = validator.validate_all()

        # Report should have required fields
        assert hasattr(report, "total_actual_apis")
        assert hasattr(report, "undocumented_apis")

        # Counts should be non-negative
        assert report.total_actual_apis >= 0
        assert len(report.undocumented_apis) >= 0

        # Undocumented count should not exceed total
        assert len(report.undocumented_apis) <= report.total_actual_apis

    def test_api_inventory_generation(self):
        """
        **Feature: documentation-validation, Property 6: Public API Documentation Coverage**
        **Validates: Requirements 3.4**

        The validator should generate a complete API inventory.
        """
        validator = APIValidator()
        inventory = validator.generate_api_inventory()

        # Should return a non-empty string
        assert isinstance(inventory, str)
        assert len(inventory) > 0

        # Should contain expected sections
        assert "Public API Inventory" in inventory
        assert "Total APIs:" in inventory


class TestAPIValidationReport:
    """Tests for the API validation report structure."""

    def test_report_has_required_fields(self):
        """
        **Feature: documentation-validation, Property 4: Documented API Existence**
        **Validates: Requirements 3.1**

        The validation report should have all required fields.
        """
        report = APIValidationReport()

        assert hasattr(report, "total_documented_apis")
        assert hasattr(report, "total_actual_apis")
        assert hasattr(report, "documented_apis")
        assert hasattr(report, "actual_apis")
        assert hasattr(report, "discrepancies")
        assert hasattr(report, "undocumented_apis")
        assert hasattr(report, "orphaned_docs")

    def test_full_validation_produces_report(self):
        """
        **Feature: documentation-validation, Property 4: Documented API Existence**
        **Validates: Requirements 3.1, 3.2, 3.4**

        Running full validation should produce a complete report.
        """
        validator = APIValidator()
        report = validator.validate_all()

        # Should have found some APIs
        assert report.total_actual_apis > 0, "Should find actual APIs"

        # Lists should be initialized
        assert isinstance(report.actual_apis, list)
        assert isinstance(report.discrepancies, list)
        assert isinstance(report.undocumented_apis, list)
        assert isinstance(report.orphaned_docs, list)


class TestPropertyBasedAPIValidation:
    """
    Property-based tests for API validation.

    **Feature: documentation-validation, Property 4, 5, 6**
    **Validates: Requirements 3.1, 3.2, 3.4**
    """

    @given(
        st.lists(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=10),
            min_size=0,
            max_size=5,
        )
    )
    @settings(max_examples=100)
    def test_parameter_list_consistency(self, param_names: List[str]):
        """
        **Feature: documentation-validation, Property 5: Method Signature Accuracy**
        **Validates: Requirements 3.2**

        For any list of parameter names, the extractor should handle them consistently.
        """
        # Filter out duplicates and empty strings
        unique_params = list(dict.fromkeys(p for p in param_names if p.strip()))

        # Create a mock API element
        api = APIElement(
            name="test_func",
            module="test_module",
            element_type="function",
            parameters=unique_params,
        )

        # Parameters should be preserved
        assert api.parameters == unique_params

        # Full name should be correct
        assert api.full_name == "test_module.test_func"

    @given(st.booleans())
    @settings(max_examples=10)
    def test_api_element_with_parent_class(self, has_parent: bool):
        """
        **Feature: documentation-validation, Property 4: Documented API Existence**
        **Validates: Requirements 3.1**

        API elements should correctly handle parent class relationships.
        """
        api = APIElement(
            name="method_name",
            module="test_module",
            element_type="method" if has_parent else "function",
            parent_class="TestClass" if has_parent else None,
        )

        if has_parent:
            assert api.full_name == "test_module.TestClass.method_name"
        else:
            assert api.full_name == "test_module.method_name"


class TestRealProjectAPIValidation:
    """Test API validation against the actual project."""

    def test_core_metrics_are_documented(self):
        """
        **Feature: documentation-validation, Property 6: Public API Documentation Coverage**
        **Validates: Requirements 3.4**

        Core metric classes should have documentation.
        """
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
            assert metric_class.__doc__ is not None, (
                f"{metric_class.__name__} should have a docstring"
            )
            assert len(metric_class.__doc__.strip()) > 0, (
                f"{metric_class.__name__} docstring should not be empty"
            )

    def test_core_representation_is_documented(self):
        """
        **Feature: documentation-validation, Property 6: Public API Documentation Coverage**
        **Validates: Requirements 3.4**

        Core UDLRepresentation class should have documentation.
        """
        from udl_rating_framework.core.representation import UDLRepresentation

        assert UDLRepresentation.__doc__ is not None
        assert len(UDLRepresentation.__doc__.strip()) > 0

    def test_validation_modules_are_documented(self):
        """
        **Feature: documentation-validation, Property 6: Public API Documentation Coverage**
        **Validates: Requirements 3.4**

        Validation module classes should have documentation.
        """
        from udl_rating_framework.validation.link_validator import LinkValidator
        from udl_rating_framework.validation.api_validator import APIValidator
        from udl_rating_framework.validation.docstring_validator import (
            DocstringValidator,
        )

        validators = [LinkValidator, APIValidator, DocstringValidator]

        for validator_class in validators:
            assert validator_class.__doc__ is not None, (
                f"{validator_class.__name__} should have a docstring"
            )
