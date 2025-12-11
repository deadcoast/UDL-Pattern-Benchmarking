"""
Property-based and unit tests for file discovery functionality.

Tests the file discovery system's ability to:
- Discover all UDL files in directory structures
- Handle errors gracefully
- Process various directory layouts
"""

import os
import tempfile
from pathlib import Path
from typing import Set

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from udl_rating_framework.io.file_discovery import FileDiscovery, FileDiscoveryError


# Strategies for generating test data
@st.composite
def directory_structure_strategy(draw):
    """Generate a directory structure with UDL files for testing."""
    # Generate directory depth (1-4 levels)
    max_depth = draw(st.integers(min_value=1, max_value=4))

    # Generate number of files per directory (0-10)
    files_per_dir = draw(st.integers(min_value=0, max_value=10))

    # Generate file extensions (mix of supported and unsupported)
    supported_extensions = [".udl", ".dsl", ".grammar", ".ebnf", ".txt"]
    unsupported_extensions = [".py", ".java", ".cpp", ".md", ".json"]

    structure = {}

    def generate_level(current_depth: int, path_prefix: str = ""):
        if current_depth > max_depth:
            return

        # Generate files at this level
        num_files = draw(st.integers(min_value=0, max_value=files_per_dir))
        for i in range(num_files):
            # Choose extension (bias toward supported ones)
            if draw(st.booleans()):  # 50% chance of supported extension
                ext = draw(st.sampled_from(supported_extensions))
            else:
                ext = draw(st.sampled_from(unsupported_extensions))

            filename = f"file_{i}{ext}"
            full_path = os.path.join(
                path_prefix, filename) if path_prefix else filename

            # Generate simple UDL content for supported files
            if ext in supported_extensions:
                content = draw(
                    st.one_of(
                        [
                            st.just("expr ::= term"),
                            st.just("stmt := assignment | expression"),
                            st.just("# Simple grammar\nrule := pattern"),
                            st.just(""),  # Empty file
                            st.text(
                                alphabet="abcdefghijklmnopqrstuvwxyz ::=|()'+*\n",
                                min_size=0,
                                max_size=100,
                            ),
                        ]
                    )
                )
            else:
                content = draw(st.text(min_size=0, max_size=50))

            structure[full_path] = content

        # Generate subdirectories
        if current_depth < max_depth:
            num_subdirs = draw(st.integers(min_value=0, max_value=3))
            for i in range(num_subdirs):
                subdir_name = f"subdir_{i}"
                subdir_path = (
                    os.path.join(path_prefix, subdir_name)
                    if path_prefix
                    else subdir_name
                )
                generate_level(current_depth + 1, subdir_path)

    generate_level(1)
    return structure


def create_test_directory(structure: dict, base_path: Path) -> Set[Path]:
    """
    Create a test directory structure and return set of expected UDL files.

    Args:
        structure: Dict mapping file paths to content
        base_path: Base directory to create structure in

    Returns:
        Set of paths to files with supported extensions
    """
    expected_files = set()
    supported_extensions = {".udl", ".dsl", ".grammar", ".ebnf", ".txt"}

    for file_path, content in structure.items():
        full_path = base_path / file_path

        # Create parent directories
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file content
        try:
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)

            # Track files with supported extensions
            if full_path.suffix in supported_extensions:
                expected_files.add(full_path)

        except Exception:
            # Skip files that can't be created
            pass

    return expected_files


class TestFileDiscoveryProperties:
    """Property-based tests for file discovery."""

    @given(directory_structure_strategy())
    @settings(max_examples=50, deadline=None)
    def test_file_discovery_completeness_property(self, structure: dict):
        """
        **Feature: udl-rating-framework, Property 6: File Discovery Completeness**
        **Validates: Requirements 2.1**

        For any directory structure containing UDL files with supported extensions,
        the system must discover all such files.
        """
        # Skip empty structures
        assume(len(structure) > 0)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create the test directory structure
            expected_files = create_test_directory(structure, temp_path)

            # Skip if no supported files were created
            assume(len(expected_files) > 0)

            # Run file discovery
            discovery = FileDiscovery()
            result = discovery.discover_files(str(temp_path))

            # Convert discovered files to set for comparison
            discovered_files = set(result.discovered_files)

            # Verify all expected files were discovered
            missing_files = expected_files - discovered_files
            assert len(missing_files) == 0, (
                f"File discovery missed {len(missing_files)} files: "
                f"{[str(f) for f in missing_files]}"
            )

            # Verify no extra files were discovered (only supported extensions)
            supported_extensions = discovery.get_supported_extensions()
            for discovered_file in discovered_files:
                assert discovered_file.suffix in supported_extensions, (
                    f"Discovered file with unsupported extension: {discovered_file}"
                )

            # Verify result metadata is reasonable
            assert result.total_files_examined >= len(discovered_files)
            assert result.total_directories_scanned >= 1

    @given(st.integers(min_value=1, max_value=5))
    @settings(max_examples=20, deadline=None)
    def test_graceful_error_handling_property(self, num_unreadable_files: int):
        """
        **Feature: udl-rating-framework, Property 7: Graceful Error Handling**
        **Validates: Requirements 2.3**

        For any directory containing at least one unreadable file,
        the system must continue processing remaining files and log the error.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create some readable UDL files
            readable_files = []
            for i in range(3):
                file_path = temp_path / f"readable_{i}.udl"
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"rule_{i} ::= term")
                readable_files.append(file_path)

            # Create unreadable files by creating them then removing read permissions
            unreadable_files = []
            for i in range(num_unreadable_files):
                file_path = temp_path / f"unreadable_{i}.udl"
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("content")

                # Make file unreadable (on Unix systems)
                try:
                    os.chmod(file_path, 0o000)
                    unreadable_files.append(file_path)
                except (OSError, PermissionError):
                    # Skip on systems where we can't change permissions
                    pass

            # Skip test if we couldn't create unreadable files
            assume(len(unreadable_files) > 0)

            try:
                # Run file discovery
                discovery = FileDiscovery()
                result = discovery.discover_files(str(temp_path))

                # Should have discovered the readable files
                discovered_paths = {str(f) for f in result.discovered_files}
                readable_paths = {str(f) for f in readable_files}

                # All readable files should be discovered
                assert readable_paths.issubset(discovered_paths), (
                    f"Some readable files were not discovered. "
                    f"Expected: {readable_paths}, Got: {discovered_paths}"
                )

                # Should have logged errors for unreadable files
                assert len(result.errors) > 0, (
                    "No errors were logged for unreadable files"
                )

                # Should have examined more files than discovered (due to unreadable ones)
                assert result.total_files_examined >= len(
                    result.discovered_files)

            finally:
                # Restore permissions for cleanup
                for file_path in unreadable_files:
                    try:
                        os.chmod(file_path, 0o644)
                    except (OSError, PermissionError):
                        pass


class TestFileDiscoveryUnit:
    """Unit tests for file discovery functionality."""

    def test_discover_files_in_simple_directory(self):
        """Test file discovery in a directory with multiple UDL files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            files_to_create = [
                ("test1.udl", "expr ::= term"),
                ("test2.dsl", "stmt := assignment"),
                ("test3.grammar", "rule pattern"),
                ("test4.ebnf", "expr = term { '+' term }"),
                ("test5.txt", "Simple text grammar"),
                ("ignored.py", "print('hello')"),  # Should be ignored
                ("ignored.json", '{"key": "value"}'),  # Should be ignored
            ]

            expected_udl_files = []
            for filename, content in files_to_create:
                file_path = temp_path / filename
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)

                # Track UDL files
                if file_path.suffix in {".udl", ".dsl", ".grammar", ".ebnf", ".txt"}:
                    expected_udl_files.append(file_path)

            # Run discovery
            discovery = FileDiscovery()
            result = discovery.discover_files(str(temp_path))

            # Verify results
            assert len(result.discovered_files) == len(expected_udl_files)
            assert len(result.errors) == 0
            assert result.total_files_examined == len(files_to_create)
            assert result.total_directories_scanned == 1

            # Verify all expected files were found
            discovered_paths = {str(f) for f in result.discovered_files}
            expected_paths = {str(f) for f in expected_udl_files}
            assert discovered_paths == expected_paths

    def test_discover_files_in_nested_directory(self):
        """Test file discovery in nested directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create nested structure
            structure = {
                "root.udl": "root_rule ::= term",
                "subdir1/file1.dsl": "rule1 := pattern",
                "subdir1/file2.grammar": "rule2 pattern",
                "subdir1/subdir2/file3.ebnf": "rule3 = term",
                "subdir1/subdir2/subdir3/file4.txt": "deep grammar",
                "other/ignored.py": "# Python file",
            }

            expected_files = create_test_directory(structure, temp_path)

            # Run discovery
            discovery = FileDiscovery()
            result = discovery.discover_files(str(temp_path))

            # Verify all UDL files were found
            discovered_files = set(result.discovered_files)
            assert discovered_files == expected_files

            # Should have scanned multiple directories
            assert result.total_directories_scanned >= 4  # root + 3 subdirs minimum
            assert len(result.errors) == 0

    def test_discover_files_in_empty_directory(self):
        """Test file discovery in empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run discovery on empty directory
            discovery = FileDiscovery()
            result = discovery.discover_files(str(temp_dir))

            # Should find no files but no errors
            assert len(result.discovered_files) == 0
            assert len(result.errors) == 0
            assert result.total_files_examined == 0
            assert result.total_directories_scanned == 1

    def test_discover_files_nonexistent_directory(self):
        """Test file discovery with nonexistent directory."""
        discovery = FileDiscovery()

        with pytest.raises(FileDiscoveryError) as exc_info:
            discovery.discover_files("/nonexistent/directory/path")

        assert "does not exist" in str(exc_info.value)

    def test_discover_files_file_instead_of_directory(self):
        """Test file discovery when given a file path instead of directory."""
        with tempfile.NamedTemporaryFile(suffix=".udl", delete=False) as temp_file:
            temp_file.write(b"expr ::= term")
            temp_file_path = temp_file.name

        try:
            discovery = FileDiscovery()

            with pytest.raises(FileDiscoveryError) as exc_info:
                discovery.discover_files(temp_file_path)

            assert "not a directory" in str(exc_info.value)
        finally:
            os.unlink(temp_file_path)

    def test_custom_extensions(self):
        """Test file discovery with custom extensions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create files with various extensions
            files = [
                ("test.udl", "standard UDL"),
                ("test.mydsl", "custom DSL"),
                ("test.lang", "custom language"),
                ("test.txt", "text file"),
            ]

            for filename, content in files:
                file_path = temp_path / filename
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)

            # Test with custom extensions
            custom_extensions = {".mydsl", ".lang"}
            discovery = FileDiscovery(supported_extensions=custom_extensions)
            result = discovery.discover_files(str(temp_path))

            # Should only find files with custom extensions
            assert len(result.discovered_files) == 2
            discovered_extensions = {f.suffix for f in result.discovered_files}
            assert discovered_extensions == custom_extensions

    def test_case_sensitivity(self):
        """Test case sensitivity in extension matching."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create files with mixed case extensions
            # Note: Use different base names to avoid filesystem case-insensitivity issues
            files = [
                ("test1.UDL", "uppercase UDL"),
                ("test2.Dsl", "mixed case DSL"),
                ("test3.udl", "lowercase udl"),
            ]

            for filename, content in files:
                file_path = temp_path / filename
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)

            # Verify all files were actually created
            created_files = list(temp_path.iterdir())

            # Test case-insensitive (default)
            discovery = FileDiscovery(case_sensitive=False)
            result = discovery.discover_files(str(temp_path))
            # Should find all files with supported extensions (case-insensitive)
            assert len(result.discovered_files) == len(created_files)

            # Test case-sensitive
            discovery_sensitive = FileDiscovery(case_sensitive=True)
            result_sensitive = discovery_sensitive.discover_files(
                str(temp_path))
            # Should only find files with exact case match (only .udl in lowercase)
            assert len(result_sensitive.discovered_files) == 1

            # Verify the case-sensitive result contains only the lowercase .udl file
            discovered_names = {
                f.name for f in result_sensitive.discovered_files}
            assert "test3.udl" in discovered_names

    def test_extension_management(self):
        """Test adding and removing extensions."""
        discovery = FileDiscovery()

        # Test initial extensions
        initial_extensions = discovery.get_supported_extensions()
        assert ".udl" in initial_extensions

        # Add new extension
        discovery.add_extension(".mylang")
        updated_extensions = discovery.get_supported_extensions()
        assert ".mylang" in updated_extensions

        # Remove extension
        discovery.remove_extension(".mylang")
        final_extensions = discovery.get_supported_extensions()
        assert ".mylang" not in final_extensions

        # Test adding extension without dot
        discovery.add_extension("newext")
        assert ".newext" in discovery.get_supported_extensions()
