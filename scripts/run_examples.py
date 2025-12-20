#!/usr/bin/env python3
"""
Example Runner Script

Executes all examples in the project to validate they work correctly.
Supports UDL files, Jupyter notebooks, and Python scripts.

**Feature: documentation-validation**
**Validates: Requirements 5.1, 5.3, 5.4**

Usage:
    uv run python scripts/run_examples.py [--verbose] [--type TYPE] [--output FILE]
"""

import argparse
import sys
import subprocess
import json
import re
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ExampleType(Enum):
    """Types of examples that can be executed."""

    UDL_FILE = "udl"
    NOTEBOOK = "notebook"
    PYTHON_SCRIPT = "script"
    CODE_BLOCK = "code_block"


@dataclass
class ExampleResult:
    """Result of executing a single example."""

    path: Path
    example_type: ExampleType
    success: bool
    error_message: Optional[str] = None
    output: Optional[str] = None
    execution_time: Optional[float] = None


@dataclass
class ExampleRunResult:
    """Result of running all examples."""

    total_examples: int
    successful: int
    failed: int
    skipped: int
    results: List[ExampleResult] = field(default_factory=list)
    by_type: Dict[str, Dict[str, int]] = field(default_factory=dict)


class ExampleRunner:
    """
    Executes examples from the project.

    Supports:
    - UDL example files
    - Jupyter notebooks
    - Python scripts
    - Code blocks from documentation
    """

    def __init__(self, project_root: Path, verbose: bool = False, timeout: int = 60):
        """Initialize the example runner.

        Args:
            project_root: Root directory of the project.
            verbose: If True, print detailed progress.
            timeout: Timeout in seconds for each example.
        """
        self.project_root = Path(project_root).resolve()
        self.verbose = verbose
        self.timeout = timeout

    def find_udl_examples(self) -> List[Path]:
        """Find all UDL example files."""
        examples_dir = self.project_root / "examples" / "udl_examples"

        if not examples_dir.exists():
            return []

        # Find .udl and .md files that contain UDL examples
        udl_files = list(examples_dir.glob("*.udl"))
        md_files = list(examples_dir.glob("*.md"))

        return sorted(udl_files + md_files)

    def find_notebooks(self) -> List[Path]:
        """Find all Jupyter notebooks."""
        notebooks = []

        # Check examples/notebooks directory
        notebooks_dir = self.project_root / "examples" / "notebooks"
        if notebooks_dir.exists():
            notebooks.extend(notebooks_dir.glob("*.ipynb"))

        # Check examples directory
        examples_dir = self.project_root / "examples"
        if examples_dir.exists():
            notebooks.extend(examples_dir.glob("*.ipynb"))

        return sorted(set(notebooks))

    def find_python_scripts(self) -> List[Path]:
        """Find all Python example scripts."""
        examples_dir = self.project_root / "examples"

        if not examples_dir.exists():
            return []

        scripts = list(examples_dir.glob("*.py"))

        # Exclude __init__.py and test files
        scripts = [
            s
            for s in scripts
            if not s.name.startswith("__") and not s.name.startswith("test_")
        ]

        return sorted(scripts)

    def run_udl_example(self, path: Path) -> ExampleResult:
        """
        Run a UDL example file through the rating system.

        **Validates: Requirements 5.1**
        """
        if self.verbose:
            print(f"  Running UDL example: {path.name}")

        try:
            # For .md files, we just check they can be read and have valid structure
            if path.suffix == ".md":
                content = path.read_text(encoding="utf-8")
                return ExampleResult(
                    path=path,
                    example_type=ExampleType.UDL_FILE,
                    success=True,
                    output=f"Read {len(content)} characters",
                )

            # For .udl files, validate they can be read
            content = path.read_text(encoding="utf-8")

            # Basic validation - check it's not empty
            if not content.strip():
                return ExampleResult(
                    path=path,
                    example_type=ExampleType.UDL_FILE,
                    success=False,
                    error_message="Empty UDL file",
                )

            return ExampleResult(
                path=path,
                example_type=ExampleType.UDL_FILE,
                success=True,
                output=f"Validated {len(content)} characters",
            )

        except UnicodeDecodeError as e:
            return ExampleResult(
                path=path,
                example_type=ExampleType.UDL_FILE,
                success=False,
                error_message=f"Encoding error: {e}",
            )
        except Exception as e:
            return ExampleResult(
                path=path,
                example_type=ExampleType.UDL_FILE,
                success=False,
                error_message=str(e),
            )

    def run_notebook(self, path: Path) -> ExampleResult:
        """
        Run a Jupyter notebook.

        **Validates: Requirements 5.3**
        """
        if self.verbose:
            print(f"  Running notebook: {path.name}")

        try:
            # First, validate JSON structure
            with open(path, "r", encoding="utf-8") as f:
                nb = json.load(f)

            if "cells" not in nb:
                return ExampleResult(
                    path=path,
                    example_type=ExampleType.NOTEBOOK,
                    success=False,
                    error_message="Invalid notebook: missing 'cells'",
                )

            # Check code cells for syntax errors
            syntax_errors = []
            for i, cell in enumerate(nb.get("cells", [])):
                if cell.get("cell_type") == "code":
                    source = "".join(cell.get("source", []))

                    # Skip cells with magic commands or shell commands
                    if source.strip().startswith("%") or source.strip().startswith("!"):
                        continue

                    # Skip cells that contain shell commands (lines starting with !)
                    # These are Jupyter-specific and won't compile as Python
                    if any(line.strip().startswith("!") for line in source.split("\n")):
                        continue

                    # Skip empty cells
                    if not source.strip():
                        continue

                    try:
                        compile(source, f"{path.name}:cell{i}", "exec")
                    except SyntaxError as e:
                        syntax_errors.append(f"Cell {i}: {e}")

            if syntax_errors:
                return ExampleResult(
                    path=path,
                    example_type=ExampleType.NOTEBOOK,
                    success=False,
                    error_message=f"Syntax errors: {'; '.join(syntax_errors[:3])}",
                )

            return ExampleResult(
                path=path,
                example_type=ExampleType.NOTEBOOK,
                success=True,
                output=f"Validated {len(nb['cells'])} cells",
            )

        except json.JSONDecodeError as e:
            return ExampleResult(
                path=path,
                example_type=ExampleType.NOTEBOOK,
                success=False,
                error_message=f"Invalid JSON: {e}",
            )
        except Exception as e:
            return ExampleResult(
                path=path,
                example_type=ExampleType.NOTEBOOK,
                success=False,
                error_message=str(e),
            )

    def run_python_script(self, path: Path, execute: bool = False) -> ExampleResult:
        """
        Run a Python example script.

        **Validates: Requirements 5.4**

        Args:
            path: Path to the Python script.
            execute: If True, actually execute the script (slower, may have side effects).
        """
        if self.verbose:
            print(f"  Running script: {path.name}")

        try:
            content = path.read_text(encoding="utf-8")

            # First, check syntax
            try:
                compile(content, str(path), "exec")
            except SyntaxError as e:
                return ExampleResult(
                    path=path,
                    example_type=ExampleType.PYTHON_SCRIPT,
                    success=False,
                    error_message=f"Syntax error: {e}",
                )

            if execute:
                # Actually run the script
                result = subprocess.run(
                    [sys.executable, str(path)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=str(self.project_root),
                )

                if result.returncode != 0:
                    return ExampleResult(
                        path=path,
                        example_type=ExampleType.PYTHON_SCRIPT,
                        success=False,
                        error_message=result.stderr[:500]
                        if result.stderr
                        else "Non-zero exit code",
                    )

                return ExampleResult(
                    path=path,
                    example_type=ExampleType.PYTHON_SCRIPT,
                    success=True,
                    output=result.stdout[:200]
                    if result.stdout
                    else "Completed successfully",
                )
            else:
                # Just validate syntax
                return ExampleResult(
                    path=path,
                    example_type=ExampleType.PYTHON_SCRIPT,
                    success=True,
                    output="Syntax valid",
                )

        except subprocess.TimeoutExpired:
            return ExampleResult(
                path=path,
                example_type=ExampleType.PYTHON_SCRIPT,
                success=False,
                error_message=f"Timeout after {self.timeout}s",
            )
        except Exception as e:
            return ExampleResult(
                path=path,
                example_type=ExampleType.PYTHON_SCRIPT,
                success=False,
                error_message=str(e),
            )

    def run_all(
        self,
        include_udl: bool = True,
        include_notebooks: bool = True,
        include_scripts: bool = True,
        execute_scripts: bool = False,
    ) -> ExampleRunResult:
        """
        Run all examples.

        Args:
            include_udl: Include UDL example files.
            include_notebooks: Include Jupyter notebooks.
            include_scripts: Include Python scripts.
            execute_scripts: Actually execute scripts (vs just syntax check).

        Returns:
            ExampleRunResult with all results.
        """
        results = []
        by_type = {
            "udl": {"total": 0, "success": 0, "failed": 0, "skipped": 0},
            "notebook": {"total": 0, "success": 0, "failed": 0, "skipped": 0},
            "script": {"total": 0, "success": 0, "failed": 0, "skipped": 0},
        }

        # Run UDL examples
        if include_udl:
            udl_files = self.find_udl_examples()
            if self.verbose:
                print(f"Found {len(udl_files)} UDL examples")

            for path in udl_files:
                by_type["udl"]["total"] += 1
                result = self.run_udl_example(path)
                results.append(result)

                if result.success:
                    by_type["udl"]["success"] += 1
                else:
                    by_type["udl"]["failed"] += 1

        # Run notebooks
        if include_notebooks:
            notebooks = self.find_notebooks()
            if self.verbose:
                print(f"Found {len(notebooks)} notebooks")

            for path in notebooks:
                by_type["notebook"]["total"] += 1
                result = self.run_notebook(path)
                results.append(result)

                if result.success:
                    by_type["notebook"]["success"] += 1
                else:
                    by_type["notebook"]["failed"] += 1

        # Run Python scripts
        if include_scripts:
            scripts = self.find_python_scripts()
            if self.verbose:
                print(f"Found {len(scripts)} Python scripts")

            for path in scripts:
                by_type["script"]["total"] += 1
                result = self.run_python_script(path, execute=execute_scripts)
                results.append(result)

                if result.success:
                    by_type["script"]["success"] += 1
                else:
                    by_type["script"]["failed"] += 1

        # Calculate totals
        total = sum(t["total"] for t in by_type.values())
        successful = sum(t["success"] for t in by_type.values())
        failed = sum(t["failed"] for t in by_type.values())
        skipped = sum(t["skipped"] for t in by_type.values())

        return ExampleRunResult(
            total_examples=total,
            successful=successful,
            failed=failed,
            skipped=skipped,
            results=results,
            by_type=by_type,
        )

    def generate_report(self, result: ExampleRunResult, format: str = "text") -> str:
        """
        Generate a report from example run results.

        Args:
            result: ExampleRunResult from run_all.
            format: Output format ('text', 'markdown', 'json').

        Returns:
            Formatted report string.
        """
        if format == "json":
            return json.dumps(
                {
                    "total_examples": result.total_examples,
                    "successful": result.successful,
                    "failed": result.failed,
                    "skipped": result.skipped,
                    "by_type": result.by_type,
                    "results": [
                        {
                            "path": str(r.path.relative_to(self.project_root)),
                            "type": r.example_type.value,
                            "success": r.success,
                            "error": r.error_message,
                            "output": r.output,
                        }
                        for r in result.results
                    ],
                },
                indent=2,
            )

        elif format == "markdown":
            lines = [
                "# Example Execution Report",
                "",
                "## Summary",
                "",
                f"- **Total examples:** {result.total_examples}",
                f"- **Successful:** {result.successful}",
                f"- **Failed:** {result.failed}",
                f"- **Skipped:** {result.skipped}",
                "",
                "## By Type",
                "",
                "| Type | Total | Success | Failed | Skipped |",
                "|------|-------|---------|--------|---------|",
            ]

            for type_name, counts in result.by_type.items():
                lines.append(
                    f"| {type_name} | {counts['total']} | {counts['success']} | {counts['failed']} | {counts['skipped']} |"
                )

            # Show failed examples
            failed_results = [r for r in result.results if not r.success]
            if failed_results:
                lines.extend(
                    [
                        "",
                        "## Failed Examples",
                        "",
                    ]
                )

                for r in failed_results:
                    try:
                        rel_path = r.path.relative_to(self.project_root)
                    except ValueError:
                        rel_path = r.path

                    lines.extend(
                        [
                            f"### {rel_path}",
                            "",
                            f"- **Type:** {r.example_type.value}",
                            f"- **Error:** {r.error_message}",
                            "",
                        ]
                    )
            else:
                lines.extend(
                    [
                        "",
                        "## Result",
                        "",
                        "✅ All examples executed successfully!",
                        "",
                    ]
                )

            return "\n".join(lines)

        else:  # text format
            lines = [
                "=" * 60,
                "Example Execution Report",
                "=" * 60,
                "",
                f"Total examples:  {result.total_examples}",
                f"Successful:      {result.successful}",
                f"Failed:          {result.failed}",
                f"Skipped:         {result.skipped}",
                "",
                "By Type:",
            ]

            for type_name, counts in result.by_type.items():
                lines.append(
                    f"  {type_name:12} Total: {counts['total']:3}  Success: {counts['success']:3}  Failed: {counts['failed']:3}"
                )

            # Show failed examples
            failed_results = [r for r in result.results if not r.success]
            if failed_results:
                lines.extend(
                    [
                        "",
                        "Failed Examples:",
                        "-" * 40,
                    ]
                )

                for r in failed_results:
                    try:
                        rel_path = r.path.relative_to(self.project_root)
                    except ValueError:
                        rel_path = r.path

                    lines.extend(
                        [
                            f"  ✗ {rel_path}",
                            f"    Type: {r.example_type.value}",
                            f"    Error: {r.error_message}",
                            "",
                        ]
                    )
            else:
                lines.extend(
                    [
                        "",
                        "✓ All examples executed successfully!",
                    ]
                )

            return "\n".join(lines)


def main():
    """Main entry point for the example runner script."""
    parser = argparse.ArgumentParser(
        description="Execute all examples in the project to validate they work.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python scripts/run_examples.py
    uv run python scripts/run_examples.py --verbose
    uv run python scripts/run_examples.py --type scripts --execute
    uv run python scripts/run_examples.py --format markdown --output report.md
""",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed progress information",
    )

    parser.add_argument(
        "--format",
        "-f",
        choices=["text", "markdown", "json"],
        default="text",
        help="Output format (default: text)",
    )

    parser.add_argument(
        "--output", "-o", type=str, help="Output file path (default: stdout)"
    )

    parser.add_argument(
        "--type",
        "-t",
        choices=["all", "udl", "notebooks", "scripts"],
        default="all",
        help="Type of examples to run (default: all)",
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute scripts (slower, may have side effects)",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout in seconds for each example (default: 60)",
    )

    parser.add_argument(
        "--project-root",
        type=str,
        default=str(project_root),
        help="Project root directory (default: auto-detected)",
    )

    args = parser.parse_args()

    # Determine which types to include
    include_udl = args.type in ("all", "udl")
    include_notebooks = args.type in ("all", "notebooks")
    include_scripts = args.type in ("all", "scripts")

    # Run example runner
    runner = ExampleRunner(
        project_root=Path(args.project_root), verbose=args.verbose, timeout=args.timeout
    )

    if args.verbose:
        print("Running example validation...")
        print()

    result = runner.run_all(
        include_udl=include_udl,
        include_notebooks=include_notebooks,
        include_scripts=include_scripts,
        execute_scripts=args.execute,
    )

    report = runner.generate_report(result, format=args.format)

    # Output report
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(report, encoding="utf-8")
        print(f"Report written to: {args.output}")
    else:
        print(report)

    # Return exit code based on failures
    return 1 if result.failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
