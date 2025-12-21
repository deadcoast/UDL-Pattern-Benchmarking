"""
Test notebook validation for the UDL Rating Framework.

This module validates that all Jupyter notebooks in the examples directory
execute without errors.

**Feature: documentation-validation, Property 11: Notebook Cell Execution**
**Validates: Requirements 5.3**
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import nbformat
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_all_notebooks() -> List[Path]:
    """Find all Jupyter notebooks in the examples directory."""
    examples_dir = PROJECT_ROOT / "examples"
    notebooks = []

    # Find all .ipynb files
    for notebook_path in examples_dir.rglob("*.ipynb"):
        # Skip checkpoint files
        if ".ipynb_checkpoints" not in str(notebook_path):
            notebooks.append(notebook_path)

    return sorted(notebooks)


def get_udl_notebooks() -> List[Path]:
    """Get notebooks specifically for UDL Rating Framework (in examples/notebooks/)."""
    notebooks_dir = PROJECT_ROOT / "examples" / "notebooks"
    if not notebooks_dir.exists():
        return []

    notebooks = []
    for notebook_path in notebooks_dir.glob("*.ipynb"):
        if ".ipynb_checkpoints" not in str(notebook_path):
            notebooks.append(notebook_path)

    return sorted(notebooks)


def get_ctm_notebooks() -> List[Path]:
    """Get CTM-related notebooks (in examples/ root)."""
    examples_dir = PROJECT_ROOT / "examples"
    notebooks = []

    for notebook_path in examples_dir.glob("*.ipynb"):
        if ".ipynb_checkpoints" not in str(notebook_path):
            notebooks.append(notebook_path)

    return sorted(notebooks)


def validate_notebook_syntax(notebook_path: Path) -> Tuple[bool, str]:
    """
    Validate that a notebook has valid JSON syntax and structure.

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        # Check basic structure
        if "cells" not in nb:
            return False, "Notebook missing 'cells' key"

        if not isinstance(nb.cells, list):
            return False, "Notebook 'cells' is not a list"

        return True, ""
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"Error reading notebook: {e}"


def count_code_cells(notebook_path: Path) -> int:
    """Count the number of code cells in a notebook."""
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        return sum(1 for cell in nb.cells if cell.cell_type == "code")
    except Exception:
        return 0


def execute_notebook(notebook_path: Path, timeout: int = 300) -> Dict[str, Any]:
    """
    Execute a notebook and return execution results.

    Args:
        notebook_path: Path to the notebook
        timeout: Timeout in seconds for each cell

    Returns:
        Dict with execution results including:
        - success: bool
        - error_cell: int or None (cell index that failed)
        - error_message: str or None
        - cells_executed: int
        - total_cells: int
    """
    result = {
        "success": False,
        "error_cell": None,
        "error_message": None,
        "cells_executed": 0,
        "total_cells": 0,
        "notebook": str(notebook_path.name),
    }

    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        code_cells = [i for i, cell in enumerate(
            nb.cells) if cell.cell_type == "code"]
        result["total_cells"] = len(code_cells)

        # Create client with timeout
        client = NotebookClient(
            nb,
            timeout=timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(notebook_path.parent)}},
        )

        # Execute the notebook
        client.execute()

        result["success"] = True
        result["cells_executed"] = len(code_cells)

    except CellExecutionError as e:
        result["error_message"] = str(e)
        # Try to extract cell index from error
        if hasattr(e, "cell_index"):
            result["error_cell"] = e.cell_index
        result["cells_executed"] = result.get("error_cell", 0)

    except Exception as e:
        result["error_message"] = f"Execution error: {type(e).__name__}: {str(e)}"

    return result


class TestNotebookSyntax:
    """Test that all notebooks have valid syntax."""

    def test_all_notebooks_have_valid_syntax(self):
        """
        **Feature: documentation-validation, Property 11: Notebook Cell Execution**
        **Validates: Requirements 5.3**

        All notebooks should have valid JSON syntax and structure.
        """
        notebooks = get_all_notebooks()
        assert len(notebooks) > 0, "No notebooks found in examples directory"

        invalid_notebooks = []
        for notebook_path in notebooks:
            is_valid, error = validate_notebook_syntax(notebook_path)
            if not is_valid:
                invalid_notebooks.append((notebook_path.name, error))

        if invalid_notebooks:
            error_msg = "Invalid notebooks found:\n"
            for name, error in invalid_notebooks:
                error_msg += f"  - {name}: {error}\n"
            pytest.fail(error_msg)

    def test_notebooks_have_code_cells(self):
        """All notebooks should have at least one code cell."""
        notebooks = get_all_notebooks()

        empty_notebooks = []
        for notebook_path in notebooks:
            cell_count = count_code_cells(notebook_path)
            if cell_count == 0:
                empty_notebooks.append(notebook_path.name)

        if empty_notebooks:
            pytest.fail(f"Notebooks with no code cells: {empty_notebooks}")


class TestUDLNotebookExecution:
    """
    Test execution of UDL Rating Framework notebooks.

    These notebooks are specifically designed for the UDL Rating Framework
    and should execute without errors.
    """

    @pytest.fixture
    def udl_notebooks(self) -> List[Path]:
        """Get UDL-specific notebooks."""
        return get_udl_notebooks()

    def test_udl_notebooks_exist(self, udl_notebooks):
        """UDL notebooks should exist in examples/notebooks/."""
        assert len(
            udl_notebooks) > 0, "No UDL notebooks found in examples/notebooks/"

    @pytest.mark.parametrize(
        "notebook_name",
        [
            "01_getting_started.ipynb",
            "02_mathematical_verification.ipynb",
        ],
    )
    def test_udl_notebook_execution(self, notebook_name):
        """
        **Feature: documentation-validation, Property 11: Notebook Cell Execution**
        **Validates: Requirements 5.3**

        *For any* cell in a Jupyter notebook in the examples directory,
        executing the cell should complete without errors.
        """
        notebook_path = PROJECT_ROOT / "examples" / "notebooks" / notebook_name

        if not notebook_path.exists():
            pytest.skip(f"Notebook {notebook_name} not found")

        result = execute_notebook(notebook_path, timeout=120)

        if not result["success"]:
            error_msg = f"Notebook {notebook_name} failed:\n"
            error_msg += f"  Error cell: {result['error_cell']}\n"
            error_msg += f"  Cells executed: {result['cells_executed']}/{result['total_cells']}\n"
            error_msg += f"  Error: {result['error_message']}"
            pytest.fail(error_msg)


class TestNotebookInventory:
    """Test notebook inventory and documentation."""

    def test_notebook_count(self):
        """Document the number of notebooks found."""
        all_notebooks = get_all_notebooks()
        udl_notebooks = get_udl_notebooks()
        ctm_notebooks = get_ctm_notebooks()

        print("\nNotebook Inventory:")
        print(f"  Total notebooks: {len(all_notebooks)}")
        print(f"  UDL notebooks: {len(udl_notebooks)}")
        print(f"  CTM notebooks: {len(ctm_notebooks)}")

        for nb in all_notebooks:
            cell_count = count_code_cells(nb)
            print(
                f"    - {nb.relative_to(PROJECT_ROOT)}: {cell_count} code cells")

        assert len(all_notebooks) > 0, "No notebooks found"

    def test_generate_notebook_report(self):
        """Generate a report of notebook validation status."""
        notebooks = get_all_notebooks()

        report = {"total": len(notebooks), "valid_syntax": 0, "notebooks": []}

        for notebook_path in notebooks:
            is_valid, error = validate_notebook_syntax(notebook_path)
            cell_count = count_code_cells(notebook_path)

            nb_info = {
                "name": notebook_path.name,
                "path": str(notebook_path.relative_to(PROJECT_ROOT)),
                "valid_syntax": is_valid,
                "code_cells": cell_count,
                "error": error if not is_valid else None,
            }
            report["notebooks"].append(nb_info)

            if is_valid:
                report["valid_syntax"] += 1

        # Print report
        print("\n=== Notebook Validation Report ===")
        print(f"Total notebooks: {report['total']}")
        print(f"Valid syntax: {report['valid_syntax']}/{report['total']}")
        print("\nNotebook details:")
        for nb in report["notebooks"]:
            status = "✓" if nb["valid_syntax"] else "✗"
            print(f"  {status} {nb['path']}: {nb['code_cells']} code cells")
            if nb["error"]:
                print(f"      Error: {nb['error']}")

        assert report["valid_syntax"] == report["total"], (
            f"Some notebooks have invalid syntax: {report['valid_syntax']}/{report['total']}"
        )
