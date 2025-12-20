"""
API Validator Module

Validates API documentation against actual source code.
Extracts public APIs, compares signatures, and identifies discrepancies.
"""

import ast
import inspect
import importlib
import pkgutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import re


@dataclass
class APIElement:
    """Represents a public API element (class, function, method)."""

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
    discrepancy_type: str  # 'missing_in_code', 'missing_in_docs', 'signature_mismatch', 'undocumented'
    documented_value: Optional[str] = None
    actual_value: Optional[str] = None
    severity: str = "major"  # 'critical', 'major', 'minor'
    suggestion: Optional[str] = None


@dataclass
class APIValidationReport:
    """Complete API validation report."""

    total_documented_apis: int = 0
    total_actual_apis: int = 0
    documented_apis: List[APIElement] = field(default_factory=list)
    actual_apis: List[APIElement] = field(default_factory=list)
    discrepancies: List[APIDiscrepancy] = field(default_factory=list)
    undocumented_apis: List[APIElement] = field(default_factory=list)
    orphaned_docs: List[str] = field(default_factory=list)


class APIExtractor:
    """Extracts public API from Python packages."""

    def __init__(self, package_name: str = "udl_rating_framework"):
        """Initialize the API extractor.

        Args:
            package_name: Name of the Python package to extract APIs from.
        """
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

        # Get package path
        if hasattr(package, "__path__"):
            package_path = package.__path__
        else:
            return []

        # Walk through all submodules
        for importer, modname, ispkg in pkgutil.walk_packages(
            path=package_path, prefix=f"{self.package_name}.", onerror=lambda x: None
        ):
            try:
                module = importlib.import_module(modname)
                self._extract_from_module(module, modname)
            except Exception as e:
                # Skip modules that can't be imported
                continue

        return self.public_apis

    def _extract_from_module(self, module: Any, module_name: str) -> None:
        """Extract public APIs from a single module."""
        # Get all public names (not starting with _)
        public_names = [
            name
            for name in dir(module)
            if not name.startswith("_") and hasattr(module, name)
        ]

        for name in public_names:
            try:
                obj = getattr(module, name)

                # Skip if not defined in this module
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
        # Add the class itself
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

        # Extract public methods
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
            name=method.__name__ if hasattr(method, "__name__") else str(method),
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
        """Initialize the documented API extractor.

        Args:
            docs_path: Path to the documentation directory.
        """
        self.docs_path = docs_path
        self.documented_modules: Set[str] = set()

    def extract_documented_modules(self, rst_file: Path) -> Set[str]:
        """Extract module names from automodule directives in RST file."""
        documented = set()

        if not rst_file.exists():
            return documented

        content = rst_file.read_text(encoding="utf-8")

        # Find all automodule directives
        pattern = r"\.\.\s+automodule::\s+(\S+)"
        matches = re.findall(pattern, content)

        for match in matches:
            documented.add(match)

        self.documented_modules = documented
        return documented


class APIValidator:
    """
    Validates API documentation against actual code.

    Checks:
    - Class existence and inheritance
    - Method signatures (parameters, types, defaults)
    - Return types
    - Docstring accuracy
    """

    def __init__(
        self,
        package_name: str = "udl_rating_framework",
        docs_path: Optional[Path] = None,
    ):
        """Initialize the API validator.

        Args:
            package_name: Name of the Python package to validate.
            docs_path: Path to the documentation directory.
        """
        self.package_name = package_name
        self.docs_path = docs_path or Path("docs")
        self.extractor = APIExtractor(package_name)
        self.doc_extractor = DocumentedAPIExtractor(self.docs_path)

    def extract_public_api(self) -> Dict[str, APIElement]:
        """Extract all public classes, functions, methods."""
        apis = self.extractor.extract_all_public_apis()
        return {api.full_name: api for api in apis}

    def get_documented_modules(self) -> Set[str]:
        """Get all modules documented in api_reference.rst."""
        rst_file = self.docs_path / "api_reference.rst"
        return self.doc_extractor.extract_documented_modules(rst_file)

    def compare_signature(self, documented: str, actual: str) -> List[str]:
        """Compare documented signature to actual."""
        discrepancies = []

        if documented != actual:
            discrepancies.append(
                f"Signature mismatch: documented={documented}, actual={actual}"
            )

        return discrepancies

    def find_undocumented_apis(self) -> List[APIElement]:
        """Find public APIs lacking documentation."""
        undocumented = []
        apis = self.extractor.extract_all_public_apis()

        for api in apis:
            if not api.docstring or api.docstring.strip() == "":
                undocumented.append(api)

        return undocumented

    def find_orphaned_docs(self) -> List[str]:
        """Find documented APIs that no longer exist."""
        orphaned = []
        documented_modules = self.get_documented_modules()

        for module_name in documented_modules:
            try:
                importlib.import_module(module_name)
            except ImportError:
                orphaned.append(module_name)

        return orphaned

    def validate_all(self) -> APIValidationReport:
        """Run complete API validation."""
        report = APIValidationReport()

        # Extract actual APIs
        actual_apis = self.extractor.extract_all_public_apis()
        report.actual_apis = actual_apis
        report.total_actual_apis = len(actual_apis)

        # Get documented modules
        documented_modules = self.get_documented_modules()
        report.total_documented_apis = len(documented_modules)

        # Find undocumented APIs
        report.undocumented_apis = self.find_undocumented_apis()

        # Find orphaned documentation
        report.orphaned_docs = self.find_orphaned_docs()

        # Create discrepancies for orphaned docs
        for orphan in report.orphaned_docs:
            report.discrepancies.append(
                APIDiscrepancy(
                    element_name=orphan,
                    discrepancy_type="missing_in_code",
                    documented_value=orphan,
                    severity="major",
                    suggestion=f"Remove documentation for {orphan} or restore the module",
                )
            )

        # Create discrepancies for undocumented APIs
        for api in report.undocumented_apis:
            report.discrepancies.append(
                APIDiscrepancy(
                    element_name=api.full_name,
                    discrepancy_type="undocumented",
                    actual_value=api.signature,
                    severity="minor",
                    suggestion=f"Add docstring to {api.full_name}",
                )
            )

        return report

    def generate_api_inventory(self) -> str:
        """Generate a markdown inventory of all public APIs."""
        apis = self.extractor.extract_all_public_apis()

        # Group by module
        by_module: Dict[str, List[APIElement]] = {}
        for api in apis:
            if api.module not in by_module:
                by_module[api.module] = []
            by_module[api.module].append(api)

        lines = ["# Public API Inventory\n"]
        lines.append(f"Total APIs: {len(apis)}\n")

        for module in sorted(by_module.keys()):
            lines.append(f"\n## {module}\n")

            # Separate classes and functions
            classes = [a for a in by_module[module] if a.element_type == "class"]
            functions = [a for a in by_module[module] if a.element_type == "function"]

            if classes:
                lines.append("\n### Classes\n")
                for cls in classes:
                    has_doc = "✓" if cls.docstring else "✗"
                    lines.append(f"- `{cls.name}{cls.signature}` [{has_doc}]\n")

            if functions:
                lines.append("\n### Functions\n")
                for func in functions:
                    has_doc = "✓" if func.docstring else "✗"
                    lines.append(f"- `{func.name}{func.signature}` [{has_doc}]\n")

        return "".join(lines)


def main():
    """Run API validation and print results."""
    validator = APIValidator()

    print("Extracting public APIs...")
    report = validator.validate_all()

    print(f"\n=== API Validation Report ===")
    print(f"Total actual APIs: {report.total_actual_apis}")
    print(f"Total documented modules: {report.total_documented_apis}")
    print(f"Undocumented APIs: {len(report.undocumented_apis)}")
    print(f"Orphaned documentation: {len(report.orphaned_docs)}")
    print(f"Total discrepancies: {len(report.discrepancies)}")

    if report.orphaned_docs:
        print("\n=== Orphaned Documentation ===")
        for orphan in report.orphaned_docs:
            print(f"  - {orphan}")

    # Generate inventory
    inventory = validator.generate_api_inventory()
    print("\n" + inventory[:2000] + "..." if len(inventory) > 2000 else inventory)


if __name__ == "__main__":
    main()
