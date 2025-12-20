"""
Property-based tests for Test Requirement Reference Validity.

**Feature: documentation-validation, Property 23: Test Requirement Reference Validity**
**Validates: Requirements 11.2, 11.4**

Tests that for any requirement reference in a test file (e.g., "Validates: Requirements 1.2"),
the referenced requirement must exist in the requirements document.
"""

import pytest
import ast
import re
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume
from typing import List, Set, Tuple, Optional
from dataclasses import dataclass


# Valid requirement numbers from both specs
VALID_DOC_VALIDATION_REQUIREMENTS = {
    "1.1",
    "1.2",
    "1.3",
    "1.4",
    "1.5",
    "1.6",
    "2.1",
    "2.2",
    "2.3",
    "2.4",
    "2.5",
    "3.1",
    "3.2",
    "3.3",
    "3.4",
    "3.5",
    "4.1",
    "4.2",
    "4.3",
    "4.4",
    "4.5",
    "5.1",
    "5.2",
    "5.3",
    "5.4",
    "5.5",
    "6.1",
    "6.2",
    "6.3",
    "6.4",
    "6.5",
    "7.1",
    "7.2",
    "7.3",
    "7.4",
    "7.5",
    "8.1",
    "8.2",
    "8.3",
    "8.4",
    "8.5",
    "9.1",
    "9.2",
    "9.3",
    "9.4",
    "9.5",
    "10.1",
    "10.2",
    "10.3",
    "10.4",
    "10.5",
    "11.1",
    "11.2",
    "11.3",
    "11.4",
    "11.5",
    "12.1",
    "12.2",
    "12.3",
    "12.4",
    "12.5",
}

VALID_UDL_FRAMEWORK_REQUIREMENTS = {
    "1.1",
    "1.2",
    "1.3",
    "1.4",
    "1.5",
    "1.6",
    "1.7",
    "1.8",
    "1.9",
    "2.1",
    "2.2",
    "2.3",
    "2.4",
    "2.5",
    "3.1",
    "3.2",
    "3.3",
    "3.4",
    "3.5",
    "3.6",
    "3.7",
    "3.8",
    "3.9",
    "4.1",
    "4.2",
    "4.3",
    "4.4",
    "4.5",
    "4.6",
    "4.7",
    "4.8",
    "5.1",
    "5.2",
    "5.3",
    "5.4",
    "5.5",
    "5.6",
    "5.7",
    "5.8",
    "6.1",
    "6.2",
    "6.3",
    "6.4",
    "6.5",
    "7.1",
    "7.2",
    "7.3",
    "7.4",
    "7.5",
    "7.6",
    "8.1",
    "8.2",
    "8.3",
    "8.4",
    "8.5",
    "8.6",
    "9.1",
    "9.2",
    "9.3",
    "9.4",
    "9.5",
    "10.1",
    "10.2",
    "10.3",
    "10.4",
    "10.5",
    "10.6",
    "10.7",
    "11.1",
    "11.2",
    "11.3",
    "11.4",
    "11.5",
    "11.6",
    "11.7",
}

# Combined valid requirements
ALL_VALID_REQUIREMENTS = (
    VALID_DOC_VALIDATION_REQUIREMENTS | VALID_UDL_FRAMEWORK_REQUIREMENTS
)


@dataclass
class RequirementReference:
    """Represents a requirement reference found in a test file."""

    file_path: str
    line_number: int
    requirement_id: str
    context: str  # The surrounding text


def extract_requirement_refs_from_docstring(docstring: str) -> List[str]:
    """Extract requirement references from a docstring."""
    if not docstring:
        return []

    refs = []
    # Pattern: Validates: Requirements X.Y, X.Y, ...
    pattern = r"\*\*Validates:\s*Requirements?\s+([\d.,\s]+)\*\*"
    matches = re.findall(pattern, docstring, re.IGNORECASE)

    for match in matches:
        for ref in match.split(","):
            ref = ref.strip()
            if ref:
                refs.append(ref)

    # Also check for _Requirements: X.Y pattern
    pattern2 = r"_Requirements?:\s*([\d.,\s]+)_"
    matches2 = re.findall(pattern2, docstring, re.IGNORECASE)

    for match in matches2:
        for ref in match.split(","):
            ref = ref.strip()
            if ref:
                refs.append(ref)

    return refs


def validate_requirement_format(ref: str) -> bool:
    """Check if a requirement reference has valid format (X.Y)."""
    return bool(re.match(r"^\d+\.\d+$", ref.strip()))


def validate_requirement_exists(ref: str) -> bool:
    """Check if a requirement reference exists in the known requirements."""
    return ref.strip() in ALL_VALID_REQUIREMENTS


def get_all_test_files() -> List[Path]:
    """Get all test files in the tests directory."""
    tests_dir = Path(__file__).parent
    return list(tests_dir.glob("**/test_*.py"))


def extract_all_requirement_refs(file_path: Path) -> List[RequirementReference]:
    """Extract all requirement references from a test file."""
    refs = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)
    except (SyntaxError, UnicodeDecodeError):
        return refs

    for node in ast.walk(tree):
        docstring = None
        line_number = 0

        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
        ):
            docstring = ast.get_docstring(node)
            line_number = getattr(node, "lineno", 0)

        if docstring:
            req_refs = extract_requirement_refs_from_docstring(docstring)
            for req_id in req_refs:
                refs.append(
                    RequirementReference(
                        file_path=str(file_path),
                        line_number=line_number,
                        requirement_id=req_id,
                        context=docstring[:100] + "..."
                        if len(docstring) > 100
                        else docstring,
                    )
                )

    return refs


class TestRequirementReferenceFormat:
    """
    Tests for requirement reference format validation.

    **Feature: documentation-validation, Property 23: Test Requirement Reference Validity**
    **Validates: Requirements 11.2, 11.4**
    """

    def test_valid_format_accepted(self):
        """Valid requirement formats should be accepted."""
        valid_formats = ["1.1", "2.3", "10.5", "11.2", "12.1"]
        for ref in valid_formats:
            assert validate_requirement_format(ref), (
                f"Valid format {ref} should be accepted"
            )

    def test_invalid_format_rejected(self):
        """Invalid requirement formats should be rejected."""
        invalid_formats = ["1", "1.1.1", "a.b", "1.a", "", "  ", "1.", ".1"]
        for ref in invalid_formats:
            assert not validate_requirement_format(ref), (
                f"Invalid format {ref} should be rejected"
            )

    @given(
        st.integers(min_value=1, max_value=20), st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=100)
    def test_generated_valid_formats(self, major: int, minor: int):
        """
        Property: Any X.Y format where X and Y are positive integers should be valid format.

        **Feature: documentation-validation, Property 23: Test Requirement Reference Validity**
        **Validates: Requirements 11.2**
        """
        ref = f"{major}.{minor}"
        assert validate_requirement_format(ref), (
            f"Generated format {ref} should be valid"
        )


class TestRequirementReferenceExistence:
    """
    Tests for requirement reference existence validation.

    **Feature: documentation-validation, Property 23: Test Requirement Reference Validity**
    **Validates: Requirements 11.2, 11.4**
    """

    def test_known_requirements_exist(self):
        """Known valid requirements should be recognized."""
        known_valid = ["1.1", "3.3", "8.5", "11.2"]
        for ref in known_valid:
            assert validate_requirement_exists(ref), (
                f"Known requirement {ref} should exist"
            )

    def test_unknown_requirements_rejected(self):
        """Unknown requirements should be rejected."""
        unknown = ["99.99", "0.0", "100.1"]
        for ref in unknown:
            assert not validate_requirement_exists(ref), (
                f"Unknown requirement {ref} should not exist"
            )

    @given(st.sampled_from(list(ALL_VALID_REQUIREMENTS)))
    @settings(max_examples=100)
    def test_all_valid_requirements_recognized(self, ref: str):
        """
        Property: All requirements in the valid set should be recognized.

        **Feature: documentation-validation, Property 23: Test Requirement Reference Validity**
        **Validates: Requirements 11.4**
        """
        assert validate_requirement_exists(ref), (
            f"Valid requirement {ref} should be recognized"
        )


class TestDocstringExtraction:
    """
    Tests for extracting requirement references from docstrings.

    **Feature: documentation-validation, Property 23: Test Requirement Reference Validity**
    **Validates: Requirements 11.2**
    """

    def test_extract_single_reference(self):
        """Single requirement reference should be extracted."""
        docstring = "**Validates: Requirements 1.2**"
        refs = extract_requirement_refs_from_docstring(docstring)
        assert refs == ["1.2"]

    def test_extract_multiple_references(self):
        """Multiple requirement references should be extracted."""
        docstring = "**Validates: Requirements 1.2, 3.4, 5.6**"
        refs = extract_requirement_refs_from_docstring(docstring)
        assert set(refs) == {"1.2", "3.4", "5.6"}

    def test_extract_underscore_format(self):
        """Underscore format references should be extracted."""
        docstring = "_Requirements: 1.2, 3.4_"
        refs = extract_requirement_refs_from_docstring(docstring)
        assert set(refs) == {"1.2", "3.4"}

    def test_empty_docstring(self):
        """Empty docstring should return empty list."""
        refs = extract_requirement_refs_from_docstring("")
        assert refs == []

    def test_none_docstring(self):
        """None docstring should return empty list."""
        refs = extract_requirement_refs_from_docstring(None)
        assert refs == []

    def test_no_references(self):
        """Docstring without references should return empty list."""
        docstring = "This is a test docstring without any requirement references."
        refs = extract_requirement_refs_from_docstring(docstring)
        assert refs == []


class TestProjectRequirementReferences:
    """
    Integration tests for requirement references in the actual project.

    **Feature: documentation-validation, Property 23: Test Requirement Reference Validity**
    **Validates: Requirements 11.2, 11.4**
    """

    def test_all_project_references_have_valid_format(self):
        """
        All requirement references in test files should have valid format.

        **Feature: documentation-validation, Property 23: Test Requirement Reference Validity**
        **Validates: Requirements 11.2**
        """
        test_files = get_all_test_files()
        invalid_refs = []

        for test_file in test_files:
            refs = extract_all_requirement_refs(test_file)
            for ref in refs:
                if not validate_requirement_format(ref.requirement_id):
                    invalid_refs.append(ref)

        assert len(invalid_refs) == 0, (
            f"Found {len(invalid_refs)} invalid format references:\n"
            + "\n".join(
                f"  {Path(r.file_path).name}:{r.line_number} - '{r.requirement_id}'"
                for r in invalid_refs[:10]
            )
        )

    def test_all_project_references_exist(self):
        """
        All requirement references in test files should reference existing requirements.

        **Feature: documentation-validation, Property 23: Test Requirement Reference Validity**
        **Validates: Requirements 11.4**
        """
        test_files = get_all_test_files()
        nonexistent_refs = []

        for test_file in test_files:
            refs = extract_all_requirement_refs(test_file)
            for ref in refs:
                if validate_requirement_format(ref.requirement_id):
                    if not validate_requirement_exists(ref.requirement_id):
                        nonexistent_refs.append(ref)

        assert len(nonexistent_refs) == 0, (
            f"Found {len(nonexistent_refs)} references to non-existent requirements:\n"
            + "\n".join(
                f"  {Path(r.file_path).name}:{r.line_number} - '{r.requirement_id}'"
                for r in nonexistent_refs[:10]
            )
        )

    def test_requirement_reference_count(self):
        """
        Test files should have a reasonable number of requirement references.

        **Feature: documentation-validation, Property 23: Test Requirement Reference Validity**
        **Validates: Requirements 11.2**
        """
        test_files = get_all_test_files()
        total_refs = 0

        for test_file in test_files:
            refs = extract_all_requirement_refs(test_file)
            total_refs += len(refs)

        # Should have at least some requirement references
        assert total_refs > 0, (
            "Project should have at least some requirement references"
        )

        # Log the count for informational purposes
        print(f"\nTotal requirement references found: {total_refs}")


class TestPropertyBasedRequirementValidation:
    """
    Property-based tests for requirement reference validation.

    **Feature: documentation-validation, Property 23: Test Requirement Reference Validity**
    **Validates: Requirements 11.2, 11.4**
    """

    @given(
        st.lists(st.sampled_from(list(ALL_VALID_REQUIREMENTS)), min_size=1, max_size=5)
    )
    @settings(max_examples=100)
    def test_valid_refs_always_validate(self, refs: List[str]):
        """
        Property: Any combination of valid requirement references should all validate.

        **Feature: documentation-validation, Property 23: Test Requirement Reference Validity**
        **Validates: Requirements 11.2, 11.4**
        """
        for ref in refs:
            assert validate_requirement_format(ref), (
                f"Valid ref {ref} should have valid format"
            )
            assert validate_requirement_exists(ref), f"Valid ref {ref} should exist"

    @given(st.text(alphabet="0123456789.", min_size=1, max_size=10))
    @settings(max_examples=100)
    def test_format_validation_is_deterministic(self, ref: str):
        """
        Property: Format validation should be deterministic.

        **Feature: documentation-validation, Property 23: Test Requirement Reference Validity**
        **Validates: Requirements 11.2**
        """
        result1 = validate_requirement_format(ref)
        result2 = validate_requirement_format(ref)
        assert result1 == result2, (
            f"Format validation should be deterministic for '{ref}'"
        )

    @given(st.text(alphabet="0123456789.", min_size=1, max_size=10))
    @settings(max_examples=100)
    def test_existence_validation_is_deterministic(self, ref: str):
        """
        Property: Existence validation should be deterministic.

        **Feature: documentation-validation, Property 23: Test Requirement Reference Validity**
        **Validates: Requirements 11.4**
        """
        result1 = validate_requirement_exists(ref)
        result2 = validate_requirement_exists(ref)
        assert result1 == result2, (
            f"Existence validation should be deterministic for '{ref}'"
        )
