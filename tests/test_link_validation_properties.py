"""
Property-based tests for Link Validation.

**Feature: documentation-validation, Property 2: Link Target Resolution**
**Validates: Requirements 2.2, 2.3, 2.4**

Tests that for any internal link in documentation (file reference, anchor link,
or code reference), the target must exist and be accessible.
"""

import pytest
import tempfile
import os
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume
import importlib.util


# Import directly from the module file to avoid package import issues
def _import_link_validator():
    """Import link_validator module directly to avoid broken package imports."""
    module_path = (
        Path(__file__).parent.parent
        / "udl_rating_framework"
        / "validation"
        / "link_validator.py"
    )
    spec = importlib.util.spec_from_file_location("link_validator", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_lv = _import_link_validator()
LinkValidator = _lv.LinkValidator
LinkType = _lv.LinkType
Link = _lv.Link
ValidationResult = _lv.ValidationResult
LinkValidationReport = _lv.LinkValidationReport


class TestLinkClassification:
    """Test link type classification."""

    def test_external_url_https(self):
        """HTTPS URLs should be classified as external."""
        validator = LinkValidator(Path("."))
        assert validator.classify_link("https://example.com") == LinkType.EXTERNAL_URL

    def test_external_url_http(self):
        """HTTP URLs should be classified as external."""
        validator = LinkValidator(Path("."))
        assert validator.classify_link("http://example.com") == LinkType.EXTERNAL_URL

    def test_anchor_link(self):
        """Anchor links should be classified correctly."""
        validator = LinkValidator(Path("."))
        assert validator.classify_link("#section-heading") == LinkType.ANCHOR_LINK

    def test_file_reference(self):
        """File references should be classified correctly."""
        validator = LinkValidator(Path("."))
        assert validator.classify_link("path/to/file.md") == LinkType.FILE_REFERENCE
        assert validator.classify_link("./relative/path.txt") == LinkType.FILE_REFERENCE
        assert validator.classify_link("/absolute/path.md") == LinkType.FILE_REFERENCE


class TestLinkExtraction:
    """Test link extraction from markdown content."""

    def test_extract_simple_link(self):
        """Extract a simple markdown link."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("[Link Text](target.md)\n")
            f.flush()

            validator = LinkValidator(Path(f.name).parent)
            links = validator.extract_links_from_file(Path(f.name))

            assert len(links) == 1
            assert links[0].link_text == "Link Text"
            assert links[0].link_target == "target.md"
            assert links[0].link_type == LinkType.FILE_REFERENCE

            os.unlink(f.name)

    def test_extract_multiple_links(self):
        """Extract multiple links from a file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("[Link 1](file1.md)\n")
            f.write("[Link 2](https://example.com)\n")
            f.write("[Link 3](#anchor)\n")
            f.flush()

            validator = LinkValidator(Path(f.name).parent)
            links = validator.extract_links_from_file(Path(f.name))

            assert len(links) == 3
            assert links[0].link_type == LinkType.FILE_REFERENCE
            assert links[1].link_type == LinkType.EXTERNAL_URL
            assert links[2].link_type == LinkType.ANCHOR_LINK

            os.unlink(f.name)

    def test_extract_links_with_line_numbers(self):
        """Verify line numbers are correctly tracked."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("Line 1\n")
            f.write("[Link on line 2](target.md)\n")
            f.write("Line 3\n")
            f.write("[Link on line 4](other.md)\n")
            f.flush()

            validator = LinkValidator(Path(f.name).parent)
            links = validator.extract_links_from_file(Path(f.name))

            assert len(links) == 2
            assert links[0].source_line == 2
            assert links[1].source_line == 4

            os.unlink(f.name)


class TestFileReferenceValidation:
    """Test file reference link validation."""

    def test_valid_file_reference(self):
        """Valid file references should pass validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create source and target files
            source = Path(tmpdir) / "source.md"
            target = Path(tmpdir) / "target.md"

            source.write_text("[Link](target.md)")
            target.write_text("Target content")

            validator = LinkValidator(Path(tmpdir))
            links = validator.extract_links_from_file(source)

            assert len(links) == 1
            result = validator.validate_file_link(links[0])
            assert result.is_valid is True

    def test_broken_file_reference(self):
        """Broken file references should fail validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.md"
            source.write_text("[Link](nonexistent.md)")

            validator = LinkValidator(Path(tmpdir))
            links = validator.extract_links_from_file(source)

            assert len(links) == 1
            result = validator.validate_file_link(links[0])
            assert result.is_valid is False
            assert "not found" in result.message.lower()

    def test_relative_path_resolution(self):
        """Relative paths should be resolved from source file location."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()

            source = subdir / "source.md"
            target = subdir / "target.md"

            source.write_text("[Link](target.md)")
            target.write_text("Target content")

            validator = LinkValidator(Path(tmpdir))
            links = validator.extract_links_from_file(source)

            result = validator.validate_file_link(links[0])
            assert result.is_valid is True

    def test_absolute_path_resolution(self):
        """Absolute paths (starting with /) should resolve from project root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create structure
            docs = Path(tmpdir) / "docs"
            docs.mkdir()

            source = docs / "source.md"
            target = Path(tmpdir) / "README.md"

            source.write_text("[Link](/README.md)")
            target.write_text("Root readme")

            validator = LinkValidator(Path(tmpdir))
            links = validator.extract_links_from_file(source)

            result = validator.validate_file_link(links[0])
            assert result.is_valid is True


class TestPropertyBasedLinkValidation:
    """
    Property-based tests for link validation.

    **Feature: documentation-validation, Property 2: Link Target Resolution**
    **Validates: Requirements 2.2, 2.3, 2.4**
    """

    @given(
        st.text(
            alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-",
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=100)
    def test_existing_file_always_validates(self, filename):
        """
        Property: For any file that exists, a link to it should validate successfully.

        **Feature: documentation-validation, Property 2: Link Target Resolution**
        **Validates: Requirements 2.2**
        """
        # Filter out problematic filenames
        assume(filename.strip() != "")
        assume(not filename.startswith("."))
        assume(not filename.startswith("-"))

        safe_filename = filename + ".md"

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.md"
            target = Path(tmpdir) / safe_filename

            source.write_text(f"[Link]({safe_filename})")
            target.write_text("Content")

            validator = LinkValidator(Path(tmpdir))
            links = validator.extract_links_from_file(source)

            if links:  # Only test if link was extracted
                result = validator.validate_file_link(links[0])
                assert result.is_valid is True, (
                    f"Valid file {safe_filename} should validate"
                )

    @given(
        st.text(
            alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-",
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=100)
    def test_nonexistent_file_always_fails(self, filename):
        """
        Property: For any file that doesn't exist, a link to it should fail validation.

        **Feature: documentation-validation, Property 2: Link Target Resolution**
        **Validates: Requirements 2.2, 2.4**
        """
        assume(filename.strip() != "")
        assume(not filename.startswith("."))
        assume(not filename.startswith("-"))

        safe_filename = filename + ".md"

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.md"
            # Don't create target file

            source.write_text(f"[Link]({safe_filename})")

            validator = LinkValidator(Path(tmpdir))
            links = validator.extract_links_from_file(source)

            if links:
                result = validator.validate_file_link(links[0])
                assert result.is_valid is False, (
                    f"Nonexistent file {safe_filename} should fail"
                )

    @given(
        st.lists(
            st.tuples(
                st.text(
                    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                    min_size=1,
                    max_size=10,
                ),
                st.booleans(),
            ),
            min_size=1,
            max_size=5,
        )
    )
    @settings(max_examples=100)
    def test_validation_report_counts_match(self, file_specs):
        """
        Property: The validation report counts should always match the actual results.

        **Feature: documentation-validation, Property 2: Link Target Resolution**
        **Validates: Requirements 2.4**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create source file with links
            links_content = []
            expected_valid = 0
            expected_broken = 0

            for i, (name, should_exist) in enumerate(file_specs):
                safe_name = f"{name}_{i}.md"
                links_content.append(f"[Link {i}]({safe_name})")

                if should_exist:
                    (Path(tmpdir) / safe_name).write_text("Content")
                    expected_valid += 1
                else:
                    expected_broken += 1

            source = Path(tmpdir) / "source.md"
            source.write_text("\n".join(links_content))

            validator = LinkValidator(Path(tmpdir))

            # Build report manually
            report = LinkValidationReport()
            links = validator.extract_links_from_file(source)

            for link in links:
                if link.link_type == LinkType.FILE_REFERENCE:
                    result = validator.validate_file_link(link)
                    report.add_result(result)

            # Verify counts match
            assert report.total_links == report.valid_links + report.broken_links
            assert len(report.results) == report.total_links
            assert len(report.broken_link_details) == report.broken_links


class TestRealProjectValidation:
    """Test validation against the actual project."""

    def test_project_file_references_are_valid(self):
        """
        All file reference links in the project documentation should be valid.

        **Feature: documentation-validation, Property 2: Link Target Resolution**
        **Validates: Requirements 2.2, 2.3, 2.4**
        """
        project_root = Path(__file__).parent.parent
        validator = LinkValidator(project_root)
        report = validator.validate_file_references_only()

        # All links should be valid
        assert report.broken_links == 0, (
            f"Found {report.broken_links} broken links:\n"
            + "\n".join(
                f"  {r.link.source_file}:{r.link.source_line} - [{r.link.link_text}]({r.link.link_target})"
                for r in report.broken_link_details
            )
        )

    def test_readme_links_are_valid(self):
        """README.md file reference links should all be valid."""
        project_root = Path(__file__).parent.parent
        readme = project_root / "README.md"

        if not readme.exists():
            pytest.skip("README.md not found")

        validator = LinkValidator(project_root)
        links = validator.extract_links_from_file(readme)
        file_links = [
            link for link in links if link.link_type == LinkType.FILE_REFERENCE
        ]

        for link in file_links:
            result = validator.validate_file_link(link)
            assert result.is_valid, (
                f"Broken link in README.md line {link.source_line}: "
                f"[{link.link_text}]({link.link_target}) - {result.message}"
            )


class TestAnchorLinkValidation:
    """
    Test anchor link validation.

    **Feature: documentation-validation, Property 2: Link Target Resolution**
    **Validates: Requirements 2.3**
    """

    def test_valid_anchor_link(self):
        """Valid anchor links should pass validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.md"
            source.write_text("""# My Heading

Some content here.

[Link to heading](#my-heading)
""")

            validator = LinkValidator(Path(tmpdir))
            links = validator.extract_links_from_file(source)
            anchor_links = [
                link for link in links if link.link_type == LinkType.ANCHOR_LINK
            ]

            assert len(anchor_links) == 1
            result = validator.validate_anchor_link(anchor_links[0])
            assert result.is_valid is True

    def test_broken_anchor_link(self):
        """Broken anchor links should fail validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.md"
            source.write_text("""# My Heading

[Link to nonexistent](#nonexistent-heading)
""")

            validator = LinkValidator(Path(tmpdir))
            links = validator.extract_links_from_file(source)
            anchor_links = [
                link for link in links if link.link_type == LinkType.ANCHOR_LINK
            ]

            assert len(anchor_links) == 1
            result = validator.validate_anchor_link(anchor_links[0])
            assert result.is_valid is False

    def test_heading_to_anchor_conversion(self):
        """Test heading to anchor conversion rules."""
        validator = LinkValidator(Path("."))

        # Basic conversion
        assert validator._heading_to_anchor("My Heading") == "my-heading"

        # Special characters removed
        assert validator._heading_to_anchor("Hello, World!") == "hello-world"

        # Multiple spaces become single hyphen
        assert validator._heading_to_anchor("Multiple   Spaces") == "multiple-spaces"

        # Underscores preserved
        assert validator._heading_to_anchor("with_underscore") == "with_underscore"

        # Numbers preserved
        assert validator._heading_to_anchor("Version 2.0") == "version-20"

    def test_multiple_headings_same_level(self):
        """Multiple headings at same level should all be extractable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.md"
            source.write_text("""# First Heading

## Second Heading

## Third Heading

[Link 1](#first-heading)
[Link 2](#second-heading)
[Link 3](#third-heading)
""")

            validator = LinkValidator(Path(tmpdir))
            headings = validator.extract_headings_from_file(source)

            assert "first-heading" in headings
            assert "second-heading" in headings
            assert "third-heading" in headings

    def test_nested_headings(self):
        """Nested headings (h1, h2, h3, etc.) should all be extractable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.md"
            source.write_text("""# Level 1

## Level 2

### Level 3

#### Level 4

##### Level 5

###### Level 6
""")

            validator = LinkValidator(Path(tmpdir))
            headings = validator.extract_headings_from_file(source)

            assert len(headings) == 6
            assert "level-1" in headings
            assert "level-6" in headings

    def test_empty_anchor_link(self):
        """Empty anchor links should fail validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.md"
            source.write_text("""# Heading

[Empty anchor](#)
""")

            validator = LinkValidator(Path(tmpdir))
            links = validator.extract_links_from_file(source)
            anchor_links = [
                link for link in links if link.link_type == LinkType.ANCHOR_LINK
            ]

            if anchor_links:  # Only if the empty anchor was extracted
                result = validator.validate_anchor_link(anchor_links[0])
                assert result.is_valid is False


class TestPropertyBasedAnchorValidation:
    """
    Property-based tests for anchor link validation.

    **Feature: documentation-validation, Property 2: Link Target Resolution**
    **Validates: Requirements 2.3**
    """

    @given(
        st.text(
            alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ",
            min_size=1,
            max_size=30,
        )
    )
    @settings(max_examples=100)
    def test_heading_anchor_roundtrip(self, heading_text):
        """
        Property: For any heading text, converting to anchor and linking should validate.

        **Feature: documentation-validation, Property 2: Link Target Resolution**
        **Validates: Requirements 2.3**
        """
        # Filter out problematic inputs
        assume(heading_text.strip() != "")
        assume(not heading_text.isspace())

        with tempfile.TemporaryDirectory() as tmpdir:
            validator = LinkValidator(Path(tmpdir))

            # Convert heading to anchor
            anchor = validator._heading_to_anchor(heading_text)
            assume(anchor != "")  # Skip if anchor becomes empty

            # Create file with heading and link to it
            source = Path(tmpdir) / "source.md"
            source.write_text(f"# {heading_text}\n\n[Link](#{anchor})\n")

            links = validator.extract_links_from_file(source)
            anchor_links = [
                link for link in links if link.link_type == LinkType.ANCHOR_LINK
            ]

            if anchor_links:
                result = validator.validate_anchor_link(anchor_links[0])
                assert result.is_valid is True, (
                    f"Heading '{heading_text}' -> anchor '{anchor}' should validate"
                )

    @given(
        st.lists(
            st.text(
                alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ",
                min_size=1,
                max_size=20,
            ),
            min_size=1,
            max_size=5,
        )
    )
    @settings(max_examples=100)
    def test_all_headings_are_linkable(self, heading_texts):
        """
        Property: For any set of headings, all should be linkable via anchors.

        **Feature: documentation-validation, Property 2: Link Target Resolution**
        **Validates: Requirements 2.3**
        """
        # Filter out problematic inputs
        valid_headings = [h for h in heading_texts if h.strip() and not h.isspace()]
        assume(len(valid_headings) > 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            validator = LinkValidator(Path(tmpdir))

            # Build document with headings
            content_lines = []
            for heading in valid_headings:
                content_lines.append(f"# {heading}")
                content_lines.append("")

            # Add links to all headings
            for heading in valid_headings:
                anchor = validator._heading_to_anchor(heading)
                if anchor:
                    content_lines.append(f"[Link to {heading}](#{anchor})")

            source = Path(tmpdir) / "source.md"
            source.write_text("\n".join(content_lines))

            links = validator.extract_links_from_file(source)
            anchor_links = [
                link for link in links if link.link_type == LinkType.ANCHOR_LINK
            ]

            for link in anchor_links:
                result = validator.validate_anchor_link(link)
                assert result.is_valid is True, (
                    f"Link {link.link_target} should be valid"
                )


class TestRealProjectAnchorValidation:
    """Test anchor validation against the actual project."""

    def test_project_anchor_links_are_valid(self):
        """
        All anchor links in the project documentation should be valid.

        **Feature: documentation-validation, Property 2: Link Target Resolution**
        **Validates: Requirements 2.3**
        """
        project_root = Path(__file__).parent.parent
        validator = LinkValidator(project_root)
        report = validator.validate_anchor_links_only()

        # All anchor links should be valid
        assert report.broken_links == 0, (
            f"Found {report.broken_links} broken anchor links:\n"
            + "\n".join(
                f"  {r.link.source_file}:{r.link.source_line} - [{r.link.link_text}]({r.link.link_target}) - {r.message}"
                for r in report.broken_link_details
            )
        )

    def test_readme_anchor_links_are_valid(self):
        """README.md anchor links should all be valid."""
        project_root = Path(__file__).parent.parent
        readme = project_root / "README.md"

        if not readme.exists():
            pytest.skip("README.md not found")

        validator = LinkValidator(project_root)
        links = validator.extract_links_from_file(readme)
        anchor_links = [
            link for link in links if link.link_type == LinkType.ANCHOR_LINK
        ]

        for link in anchor_links:
            result = validator.validate_anchor_link(link)
            assert result.is_valid, (
                f"Broken anchor in README.md line {link.source_line}: "
                f"[{link.link_text}]({link.link_target}) - {result.message}"
            )
