"""
Property-based tests for environment variable documentation validation.

**Feature: documentation-validation, Property 25: Environment Variable Documentation**
**Validates: Requirements 12.4**

This module tests that all environment variables read by the application
are documented in deployment documentation.
"""

import ast
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest
from hypothesis import given, settings, strategies as st


class EnvVarExtractor(ast.NodeVisitor):
    """AST visitor to extract environment variable reads from Python code."""

    def __init__(self):
        self.env_vars: List[Tuple[str, int, str]] = []  # (var_name, line_no, file_path)
        self.current_file = ""

    def visit_Call(self, node: ast.Call):
        """Visit function calls to find os.environ.get() and os.getenv()."""
        # Check for os.environ.get('VAR')
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "environ"
        ):
            if node.args and isinstance(node.args[0], ast.Constant):
                self.env_vars.append(
                    (node.args[0].value, node.lineno, self.current_file)
                )

        # Check for os.getenv('VAR')
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "getenv":
            if node.args and isinstance(node.args[0], ast.Constant):
                self.env_vars.append(
                    (node.args[0].value, node.lineno, self.current_file)
                )

        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        """Visit subscript operations to find os.environ['VAR']."""
        if (
            isinstance(node.value, ast.Attribute)
            and node.value.attr == "environ"
            and isinstance(node.slice, ast.Constant)
        ):
            self.env_vars.append((node.slice.value, node.lineno, self.current_file))

        self.generic_visit(node)


def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path(__file__).parent.parent


def extract_env_vars_from_file(file_path: Path) -> List[Tuple[str, int, str]]:
    """Extract environment variable reads from a Python file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)
        extractor = EnvVarExtractor()
        extractor.current_file = str(file_path)
        extractor.visit(tree)
        return extractor.env_vars
    except (SyntaxError, UnicodeDecodeError):
        return []


def extract_all_env_vars(project_root: Path) -> Dict[str, List[Tuple[int, str]]]:
    """
    Extract all environment variable reads from the project.

    Returns:
        Dict mapping env var names to list of (line_no, file_path) tuples
    """
    env_vars: Dict[str, List[Tuple[int, str]]] = {}

    # Directories to scan
    scan_dirs = [
        project_root / "deployment",
        project_root / "udl_rating_framework",
        project_root / "scripts",
    ]

    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue

        for py_file in scan_dir.rglob("*.py"):
            # Skip test files and __pycache__
            if "__pycache__" in str(py_file) or "test_" in py_file.name:
                continue

            vars_in_file = extract_env_vars_from_file(py_file)
            for var_name, line_no, file_path in vars_in_file:
                if var_name not in env_vars:
                    env_vars[var_name] = []
                env_vars[var_name].append((line_no, file_path))

    return env_vars


def extract_documented_env_vars(project_root: Path) -> Set[str]:
    """
    Extract environment variables documented in deployment documentation.

    Looks for:
    - Tables with Variable/Description columns
    - Code blocks with environment variable assignments
    - Inline mentions of environment variables
    """
    documented_vars: Set[str] = set()

    # Documentation files to check
    doc_files = [
        project_root / "deployment" / "README.md",
        project_root / "deployment" / "DEPLOYMENT_SUMMARY.md",
        project_root / "README.md",
    ]

    for doc_file in doc_files:
        if not doc_file.exists():
            continue

        with open(doc_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Pattern 1: Table rows with | `VAR_NAME` | or | VAR_NAME |
        table_pattern = r"\|\s*`?([A-Z][A-Z0-9_]+)`?\s*\|"
        for match in re.finditer(table_pattern, content):
            documented_vars.add(match.group(1))

        # Pattern 2: Environment variable assignments in code blocks
        # e.g., ENVIRONMENT=production or export VAR_NAME=value
        assignment_pattern = r"(?:export\s+)?([A-Z][A-Z0-9_]+)="
        for match in re.finditer(assignment_pattern, content):
            documented_vars.add(match.group(1))

        # Pattern 3: ${VAR_NAME} or $VAR_NAME in shell/docker contexts
        shell_var_pattern = r"\$\{?([A-Z][A-Z0-9_]+)\}?"
        for match in re.finditer(shell_var_pattern, content):
            documented_vars.add(match.group(1))

        # Pattern 4: os.getenv("VAR") or os.environ.get("VAR") in code examples
        code_pattern = (
            r'(?:os\.getenv|os\.environ\.get)\s*\(\s*["\']([A-Z][A-Z0-9_]+)["\']'
        )
        for match in re.finditer(code_pattern, content):
            documented_vars.add(match.group(1))

    return documented_vars


def get_deployment_env_vars(project_root: Path) -> Dict[str, List[Tuple[int, str]]]:
    """
    Get environment variables specifically used in deployment-related code.

    This focuses on the deployment API and related scripts that users
    would need to configure when deploying the application.
    """
    all_vars = extract_all_env_vars(project_root)

    # Filter to deployment-relevant variables
    # Exclude DDP/distributed training vars as they're PyTorch-specific
    ddp_vars = {"RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"}

    deployment_vars = {}
    for var_name, locations in all_vars.items():
        # Include if used in deployment directory
        deployment_locations = [
            (line, path) for line, path in locations if "deployment" in path
        ]
        if deployment_locations:
            deployment_vars[var_name] = deployment_locations
        # Also include coverage monitoring vars (used in CI/CD)
        elif any("coverage_monitor" in path for _, path in locations):
            deployment_vars[var_name] = locations

    return deployment_vars


class TestEnvVarDocumentation:
    """Tests for environment variable documentation completeness."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.project_root = get_project_root()
        self.deployment_env_vars = get_deployment_env_vars(self.project_root)
        self.documented_vars = extract_documented_env_vars(self.project_root)

    def test_all_deployment_env_vars_documented(self):
        """
        **Feature: documentation-validation, Property 25: Environment Variable Documentation**
        **Validates: Requirements 12.4**

        *For any* environment variable read by the deployment application,
        it should be documented in deployment documentation.
        """
        undocumented = []

        for var_name, locations in self.deployment_env_vars.items():
            if var_name not in self.documented_vars:
                for line_no, file_path in locations:
                    undocumented.append(
                        {"variable": var_name, "file": file_path, "line": line_no}
                    )

        if undocumented:
            msg_lines = [
                "The following environment variables are used but not documented:"
            ]
            for item in undocumented:
                msg_lines.append(
                    f"  - {item['variable']} (used in {item['file']}:{item['line']})"
                )
            msg_lines.append(
                "\nPlease add documentation for these variables in deployment/README.md"
            )
            pytest.fail("\n".join(msg_lines))

    def test_documented_vars_are_used(self):
        """
        Test that documented environment variables are actually used in code.

        This helps identify stale documentation.
        """
        all_vars = extract_all_env_vars(self.project_root)
        all_var_names = set(all_vars.keys())

        # Variables that are documented but might not be directly read
        # (e.g., used by external tools like Grafana)
        external_vars = {"GRAFANA_PASSWORD", "GF_SECURITY_ADMIN_PASSWORD"}

        unused_documented = self.documented_vars - all_var_names - external_vars

        # Filter out common false positives (variables used in shell scripts, etc.)
        shell_only_vars = {"PUSH_TO_REGISTRY"}  # Used in build.sh
        unused_documented = unused_documented - shell_only_vars

        if unused_documented:
            # This is informational, not a failure
            print(
                f"\nNote: The following documented variables are not directly read in Python code:"
            )
            for var in sorted(unused_documented):
                print(f"  - {var}")

    def test_env_var_extraction_works(self):
        """Test that the environment variable extraction logic works correctly."""
        # Create a test Python snippet
        test_code = """
import os

# Various ways to read env vars
var1 = os.environ.get('TEST_VAR_1')
var2 = os.getenv('TEST_VAR_2')
var3 = os.environ['TEST_VAR_3']
var4 = os.environ.get('TEST_VAR_4', 'default')
"""

        tree = ast.parse(test_code)
        extractor = EnvVarExtractor()
        extractor.current_file = "test.py"
        extractor.visit(tree)

        var_names = [v[0] for v in extractor.env_vars]
        assert "TEST_VAR_1" in var_names
        assert "TEST_VAR_2" in var_names
        assert "TEST_VAR_3" in var_names
        assert "TEST_VAR_4" in var_names

    def test_documentation_extraction_works(self):
        """Test that documentation extraction finds documented variables."""
        # The deployment README should have at least these documented
        expected_vars = {
            "API_TOKEN",
            "CTM_MODEL_PATH",
            "ENVIRONMENT",
            "PORT",
            "LOG_LEVEL",
        }

        for var in expected_vars:
            assert var in self.documented_vars, (
                f"Expected {var} to be documented in deployment docs"
            )


@given(
    st.sampled_from(
        list(get_deployment_env_vars(get_project_root()).keys()) or ["PLACEHOLDER"]
    )
)
@settings(max_examples=50)
def test_property_env_var_has_documentation(env_var: str):
    """
    **Feature: documentation-validation, Property 25: Environment Variable Documentation**
    **Validates: Requirements 12.4**

    Property-based test: For any environment variable used in deployment code,
    verify it has corresponding documentation.
    """
    if env_var == "PLACEHOLDER":
        pytest.skip("No deployment environment variables found")

    project_root = get_project_root()
    documented_vars = extract_documented_env_vars(project_root)

    assert env_var in documented_vars, (
        f"Environment variable '{env_var}' is used in deployment code "
        f"but not documented in deployment/README.md"
    )
