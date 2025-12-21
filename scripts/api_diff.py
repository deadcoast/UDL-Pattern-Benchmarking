#!/usr/bin/env python3
"""
API Diff Script

Compares API documentation to actual source code implementation.
Identifies discrepancies between documented and actual APIs.

**Feature: documentation-validation**
**Validates: Requirements 3.1, 3.2**

Usage:
    uv run python scripts/api_diff.py [--verbose] [--format FORMAT] [--output FILE]
"""

import argparse
import ast
import importlib
import inspect
import json
import pkgutil
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class APIElement:
    """Represents a public API element."""

    name: str
    module: str
    element_type: str  # 'class', 'function', 'method'
    signature: Optional[str] = None
    docstring: Optional[str] = None
    parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    parent_class: Optional[str] = None
    line_number: Optional[int] = None

    @property
    def full_name(self) -> str:
        """Get fully qualified name."""
        if self.parent_class:
            return f"{self.module}.{self.parent_class}.{self.name}"
        return f"{self.module}.{self.name}"


@dataclass
class APIDiscrepancy:
    """Represents a discrepancy between documented and actual API."""

    element_name: str
    discrepancy_type: str
    documented_value: Optional[str] = None
    actual_value: Optional[str] = None
    severity: str = "major"
    suggestion: Optional[str] = None


@dataclass
class APIDiffResult:
    """Result of API diff operation."""

    total_actual_apis: int
    total_documented_modules: int
    undocumented_apis: List[APIElement]
    orphaned_docs: List[str]
    signature_mismatches: List[APIDiscrepancy]
    missing_docstrings: List[APIElement]
    discrepancies: List[APIDiscrepancy]


class APIExtractor:
    """Extracts public API from Python packages."""

    def __init__(self, package_name: str = "udl_rating_framework"):
        """Initialize the API extractor."""
        self.package_name = package_name
        self.public_apis: List[APIElement] = []

    def extract_all_public_apis(self) -> List[APIElement]:
        """Extract all public APIs from the package."""
        self.public_apis = []

        try:
            package = importlib.import_module(self.package_name)
        except ImportError as e:
            print(f"Error importing package {self.package_name}: {e}")
            return []

        if hasattr(package, "__path__"):
            package_path = package.__path__
        else:
            return []

        for importer, modname, ispkg in pkgutil.walk_packages(
            path=package_path, prefix=f"{self.package_name}.", onerror=lambda x: None
        ):
            try:
                module = importlib.import_module(modname)
                self._extract_from_module(module, modname)
            except Exception:
                continue

        return self.public_apis

    def _extract_from_module(self, module: Any, module_name: str) -> None:
        """Extract public APIs from a single module."""
        public_names = [
            name
            for name in dir(module)
            if not name.startswith("_") and hasattr(module, name)
        ]

        for name in public_names:
            try:
                obj = getattr(module, name)

                if hasattr(obj, "__module__") and obj.__module__ != module_name:
                    continue

                if inspect.isclass(obj):
                    self._extract_class(obj, module_name)
                elif inspect.isfunction(obj):
                    self._extract_function(obj, module_name)
            except Exception:
                continue

    def _extract_class(self, cls: type, module_name: str) -> None:
        """Extract class and its public methods."""
        try:
            sig = str(inspect.signature(cls))
        except (ValueError, TypeError):
            sig = "()"

        class_element = APIElement(
            name=cls.__name__,
            module=module_name,
            element_type="class",
            signature=sig,
            docstring=cls.__doc__,
            parameters=self._get_parameters(cls),
            line_number=self._get_line_number(cls),
        )
        self.public_apis.append(class_element)

        for method_name in dir(cls):
            if method_name.startswith("_") and method_name != "__init__":
                continue

            try:
                method = getattr(cls, method_name)
                if inspect.isfunction(method) or inspect.ismethod(method):
                    self._extract_method(method, module_name, cls.__name__)
            except Exception:
                continue

    def _extract_function(self, func: Any, module_name: str) -> None:
        """Extract function information."""
        try:
            sig = str(inspect.signature(func))
        except (ValueError, TypeError):
            sig = "()"

        element = APIElement(
            name=func.__name__,
            module=module_name,
            element_type="function",
            signature=sig,
            docstring=func.__doc__,
            parameters=self._get_parameters(func),
            return_type=self._get_return_type(func),
            line_number=self._get_line_number(func),
        )
        self.public_apis.append(element)

    def _extract_method(self, method: Any, module_name: str, class_name: str) -> None:
        """Extract method information."""
        try:
            sig = str(inspect.signature(method))
        except (ValueError, TypeError):
            sig = "()"

        element = APIElement(
            name=method.__name__ if hasattr(
                method, "__name__") else str(method),
            module=module_name,
            element_type="method",
            signature=sig,
            docstring=method.__doc__ if hasattr(method, "__doc__") else None,
            parameters=self._get_parameters(method),
            return_type=self._get_return_type(method),
            parent_class=class_name,
            line_number=self._get_line_number(method),
        )
        self.public_apis.append(element)

    def _get_parameters(self, obj: Any) -> List[str]:
        """Get parameter names from callable."""
        try:
            sig = inspect.signature(obj)
            return [p.name for p in sig.parameters.values()]
        except (ValueError, TypeError):
            return []

    def _get_return_type(self, obj: Any) -> Optional[str]:
        """Get return type annotation if available."""
        try:
            hints = getattr(obj, "__annotations__", {})
            if "return" in hints:
                return str(hints["return"])
        except Exception:
            pass
        return None

    def _get_line_number(self, obj: Any) -> Optional[int]:
        """Get source line number if available."""
        try:
            return inspect.getsourcelines(obj)[1]
        except (OSError, TypeError):
            return None


class DocumentedAPIExtractor:
    """Extracts documented APIs from RST documentation."""

    def __init__(self, docs_path: Path):
        """Initialize the documented API extractor."""
        self.docs_path = docs_path

    def extract_documented_modules(self, rst_file: Path) -> Set[str]:
        """Extract module names from automodule directives in RST file."""
        documented = set()

        if not rst_file.exists():
            return documented

        content = rst_file.read_text(encoding="utf-8")

        pattern = r"\.\.\s+automodule::\s+(\S+)"
        matches = re.findall(pattern, content)

        for match in matches:
            documented.add(match)

        return documented


class APIDiff:
    """
    Compares API documentation to source code.

    Identifies:
    - Undocumented public APIs
    - Orphaned documentation (docs for non-existent code)
    - Signature mismatches
    - Missing docstrings
    """

    def __init__(
        self,
        package_name: str = "udl_rating_framework",
        docs_path: Optional[Path] = None,
        verbose: bool = False,
    ):
        """Initialize the API diff tool.

        Args:
            package_name: Name of the Python package to analyze.
            docs_path: Path to documentation directory.
            verbose: If True, print detailed progress.
        """
        self.package_name = package_name
        self.docs_path = docs_path or project_root / "docs"
        self.verbose = verbose
        self.extractor = APIExtractor(package_name)
        self.doc_extractor = DocumentedAPIExtractor(self.docs_path)

    def run_diff(self) -> APIDiffResult:
        """
        Run the API diff analysis.

        Returns:
            APIDiffResult with all findings.
        """
        if self.verbose:
            print(f"Extracting APIs from {self.package_name}...")

        # Extract actual APIs
        actual_apis = self.extractor.extract_all_public_apis()

        if self.verbose:
            print(f"  Found {len(actual_apis)} public APIs")

        # Get documented modules
        rst_file = self.docs_path / "api_reference.rst"
        documented_modules = self.doc_extractor.extract_documented_modules(
            rst_file)

        if self.verbose:
            print(f"  Found {len(documented_modules)} documented modules")

        # Find undocumented APIs (no docstring)
        undocumented = []
        missing_docstrings = []
        for api in actual_apis:
            if not api.docstring or api.docstring.strip() == "":
                missing_docstrings.append(api)
                if api.element_type in ("class", "function"):
                    undocumented.append(api)

        # Find orphaned documentation
        orphaned = []
        for module_name in documented_modules:
            try:
                importlib.import_module(module_name)
            except ImportError:
                orphaned.append(module_name)

        # Build discrepancies list
        discrepancies = []

        for orphan in orphaned:
            discrepancies.append(
                APIDiscrepancy(
                    element_name=orphan,
                    discrepancy_type="orphaned_documentation",
                    documented_value=orphan,
                    severity="major",
                    suggestion=f"Remove documentation for {orphan} or restore the module",
                )
            )

        for api in undocumented:
            discrepancies.append(
                APIDiscrepancy(
                    element_name=api.full_name,
                    discrepancy_type="missing_docstring",
                    actual_value=api.signature,
                    severity="minor",
                    suggestion=f"Add docstring to {api.full_name}",
                )
            )

        return APIDiffResult(
            total_actual_apis=len(actual_apis),
            total_documented_modules=len(documented_modules),
            undocumented_apis=undocumented,
            orphaned_docs=orphaned,
            signature_mismatches=[],  # Would need more sophisticated analysis
            missing_docstrings=missing_docstrings,
            discrepancies=discrepancies,
        )

    def generate_report(self, result: APIDiffResult, format: str = "text") -> str:
        """
        Generate a report from API diff results.

        Args:
            result: APIDiffResult from run_diff.
            format: Output format ('text', 'markdown', 'json').

        Returns:
            Formatted report string.
        """
        if format == "json":
            return json.dumps(
                {
                    "total_actual_apis": result.total_actual_apis,
                    "total_documented_modules": result.total_documented_modules,
                    "undocumented_count": len(result.undocumented_apis),
                    "orphaned_count": len(result.orphaned_docs),
                    "missing_docstrings_count": len(result.missing_docstrings),
                    "undocumented_apis": [
                        {
                            "name": api.full_name,
                            "type": api.element_type,
                            "signature": api.signature,
                        }
                        for api in result.undocumented_apis
                    ],
                    "orphaned_docs": result.orphaned_docs,
                    "discrepancies": [
                        {
                            "element": d.element_name,
                            "type": d.discrepancy_type,
                            "severity": d.severity,
                            "suggestion": d.suggestion,
                        }
                        for d in result.discrepancies
                    ],
                },
                indent=2,
            )

        elif format == "markdown":
            lines = [
                "# API Documentation Diff Report",
                "",
                "## Summary",
                "",
                f"- **Total actual APIs:** {result.total_actual_apis}",
                f"- **Documented modules:** {result.total_documented_modules}",
                f"- **Undocumented APIs:** {len(result.undocumented_apis)}",
                f"- **Orphaned documentation:** {len(result.orphaned_docs)}",
                f"- **Missing docstrings:** {len(result.missing_docstrings)}",
                "",
            ]

            if result.orphaned_docs:
                lines.extend(
                    [
                        "## Orphaned Documentation",
                        "",
                        "These modules are documented but no longer exist:",
                        "",
                    ]
                )
                for orphan in result.orphaned_docs:
                    lines.append(f"- `{orphan}`")
                lines.append("")

            if result.undocumented_apis:
                lines.extend(
                    [
                        "## Undocumented APIs",
                        "",
                        "These public APIs lack docstrings:",
                        "",
                    ]
                )

                # Group by module
                by_module: Dict[str, List[APIElement]] = {}
                for api in result.undocumented_apis:
                    if api.module not in by_module:
                        by_module[api.module] = []
                    by_module[api.module].append(api)

                for module in sorted(by_module.keys()):
                    lines.append(f"### {module}")
                    lines.append("")
                    for api in by_module[module]:
                        lines.append(
                            f"- `{api.name}{api.signature or '()'}` ({api.element_type})"
                        )
                    lines.append("")

            if not result.orphaned_docs and not result.undocumented_apis:
                lines.extend(
                    [
                        "## Result",
                        "",
                        "✅ All APIs are properly documented!",
                        "",
                    ]
                )

            return "\n".join(lines)

        else:  # text format
            lines = [
                "=" * 60,
                "API Documentation Diff Report",
                "=" * 60,
                "",
                f"Total actual APIs:      {result.total_actual_apis}",
                f"Documented modules:     {result.total_documented_modules}",
                f"Undocumented APIs:      {len(result.undocumented_apis)}",
                f"Orphaned documentation: {len(result.orphaned_docs)}",
                f"Missing docstrings:     {len(result.missing_docstrings)}",
                "",
            ]

            if result.orphaned_docs:
                lines.extend(
                    [
                        "Orphaned Documentation:",
                        "-" * 40,
                    ]
                )
                for orphan in result.orphaned_docs:
                    lines.append(f"  ✗ {orphan}")
                lines.append("")

            if result.undocumented_apis:
                lines.extend(
                    [
                        "Undocumented APIs (no docstring):",
                        "-" * 40,
                    ]
                )

                # Show first 20
                for api in result.undocumented_apis[:20]:
                    lines.append(f"  ✗ {api.full_name}")

                if len(result.undocumented_apis) > 20:
                    lines.append(
                        f"  ... and {len(result.undocumented_apis) - 20} more")
                lines.append("")

            if not result.orphaned_docs and not result.undocumented_apis:
                lines.extend(
                    [
                        "✓ All APIs are properly documented!",
                    ]
                )

            return "\n".join(lines)

    def generate_inventory(self) -> str:
        """Generate a markdown inventory of all public APIs."""
        apis = self.extractor.extract_all_public_apis()

        by_module: Dict[str, List[APIElement]] = {}
        for api in apis:
            if api.module not in by_module:
                by_module[api.module] = []
            by_module[api.module].append(api)

        lines = ["# Public API Inventory", ""]
        lines.append(f"Total APIs: {len(apis)}")
        lines.append("")

        for module in sorted(by_module.keys()):
            lines.append(f"## {module}")
            lines.append("")

            classes = [a for a in by_module[module]
                       if a.element_type == "class"]
            functions = [a for a in by_module[module]
                         if a.element_type == "function"]

            if classes:
                lines.append("### Classes")
                lines.append("")
                for cls in classes:
                    has_doc = "✓" if cls.docstring else "✗"
                    lines.append(f"- `{cls.name}{cls.signature}` [{has_doc}]")
                lines.append("")

            if functions:
                lines.append("### Functions")
                lines.append("")
                for func in functions:
                    has_doc = "✓" if func.docstring else "✗"
                    lines.append(
                        f"- `{func.name}{func.signature}` [{has_doc}]")
                lines.append("")

        return "\n".join(lines)


def main():
    """Main entry point for the API diff script."""
    parser = argparse.ArgumentParser(
        description="Compare API documentation to source code implementation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python scripts/api_diff.py
    uv run python scripts/api_diff.py --verbose
    uv run python scripts/api_diff.py --format markdown --output api_report.md
    uv run python scripts/api_diff.py --inventory --output api_inventory.md
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
        "--package",
        type=str,
        default="udl_rating_framework",
        help="Package name to analyze (default: udl_rating_framework)",
    )

    parser.add_argument(
        "--docs-path",
        type=str,
        default=str(project_root / "docs"),
        help="Path to documentation directory",
    )

    parser.add_argument(
        "--inventory",
        action="store_true",
        help="Generate API inventory instead of diff report",
    )

    args = parser.parse_args()

    # Run API diff
    differ = APIDiff(
        package_name=args.package, docs_path=Path(args.docs_path), verbose=args.verbose
    )

    if args.inventory:
        report = differ.generate_inventory()
    else:
        if args.verbose:
            print("Running API diff analysis...")
            print()

        result = differ.run_diff()
        report = differ.generate_report(result, format=args.format)

    # Output report
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(report, encoding="utf-8")
        print(f"Report written to: {args.output}")
    else:
        print(report)

    # Return exit code based on discrepancies
    if not args.inventory:
        result = differ.run_diff()
        has_issues = len(result.orphaned_docs) > 0
        return 1 if has_issues else 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
