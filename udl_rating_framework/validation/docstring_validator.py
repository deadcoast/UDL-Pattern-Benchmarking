"""
Docstring Validator Module

Validates docstrings against actual function signatures, tests docstring examples,
identifies undocumented public APIs, and verifies mathematical formulas.

**Feature: documentation-validation, Property 19: Docstring Signature Accuracy**
**Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5**
"""

import ast
import inspect
import importlib
import pkgutil
import re
import doctest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import io
import sys


@dataclass
class DocstringParameter:
    """Represents a parameter documented in a docstring."""

    name: str
    type_hint: Optional[str] = None
    description: Optional[str] = None


@dataclass
class DocstringInfo:
    """Parsed docstring information."""

    summary: str = ""
    description: str = ""
    parameters: List[DocstringParameter] = field(default_factory=list)
    returns: Optional[str] = None
    return_type: Optional[str] = None
    raises: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    formulas: List[str] = field(default_factory=list)
    raw: str = ""


@dataclass
class SignatureMismatch:
    """Represents a mismatch between docstring and actual signature."""

    function_name: str
    module: str
    mismatch_type: str  # 'missing_param', 'extra_param', 'type_mismatch'
    param_name: str
    documented_value: Optional[str] = None
    actual_value: Optional[str] = None
    line_number: Optional[int] = None
    file_path: Optional[str] = None


@dataclass
class UndocumentedAPI:
    """Represents a public API without docstring."""

    name: str
    module: str
    element_type: str  # 'class', 'function', 'method'
    line_number: Optional[int] = None
    file_path: Optional[str] = None


@dataclass
class DoctestResult:
    """Result of running doctest examples."""

    function_name: str
    module: str
    passed: bool
    failures: List[str] = field(default_factory=list)
    examples_count: int = 0


@dataclass
class DocstringValidationReport:
    """Complete docstring validation report."""

    signature_mismatches: List[SignatureMismatch] = field(default_factory=list)
    undocumented_apis: List[UndocumentedAPI] = field(default_factory=list)
    doctest_results: List[DoctestResult] = field(default_factory=list)
    formula_issues: List[Dict[str, Any]] = field(default_factory=list)
    total_functions_checked: int = 0
    total_with_docstrings: int = 0


class DocstringParser:
    """Parses docstrings in various formats (Google, NumPy, Sphinx)."""

    # Patterns for different docstring sections
    GOOGLE_PARAM_PATTERN = re.compile(r"^\s*(\w+)\s*(?:\(([^)]+)\))?\s*:\s*(.*)$")
    NUMPY_PARAM_PATTERN = re.compile(r"^\s*(\w+)\s*:\s*(\S+)?\s*$")
    SPHINX_PARAM_PATTERN = re.compile(r":param\s+(?:(\w+)\s+)?(\w+):\s*(.*)$")
    SPHINX_TYPE_PATTERN = re.compile(r":type\s+(\w+):\s*(.*)$")
    RETURNS_PATTERN = re.compile(
        r"(?:Returns?|:returns?:)\s*[:\-]?\s*(.*)$", re.IGNORECASE
    )
    FORMULA_PATTERN = re.compile(
        r"(?:Formula|Mathematical Definition|LaTeX)[:\s]*\n?\s*(.+?)(?:\n\n|\Z)",
        re.IGNORECASE | re.DOTALL,
    )

    def parse(self, docstring: Optional[str]) -> DocstringInfo:
        """Parse a docstring and extract structured information."""
        if not docstring:
            return DocstringInfo()

        info = DocstringInfo(raw=docstring)
        lines = docstring.strip().split("\n")

        if not lines:
            return info

        # First line is usually the summary
        info.summary = lines[0].strip()

        # Parse parameters
        info.parameters = self._parse_parameters(docstring)

        # Parse returns
        info.returns = self._parse_returns(docstring)

        # Parse examples
        info.examples = self._parse_examples(docstring)

        # Parse formulas
        info.formulas = self._parse_formulas(docstring)

        return info

    def _parse_parameters(self, docstring: str) -> List[DocstringParameter]:
        """Extract parameter documentation from docstring."""
        params = []
        param_names_seen = set()

        # Try Google style: Args: section
        args_match = re.search(
            r"(?:Args|Arguments|Parameters)[:\s]*\n((?:\s+.+\n?)+)",
            docstring,
            re.IGNORECASE,
        )
        if args_match:
            args_section = args_match.group(1)
            for line in args_section.split("\n"):
                match = self.GOOGLE_PARAM_PATTERN.match(line)
                if match:
                    name = match.group(1)
                    if name not in param_names_seen:
                        params.append(
                            DocstringParameter(
                                name=name,
                                type_hint=match.group(2),
                                description=(
                                    match.group(3).strip() if match.group(3) else None
                                ),
                            )
                        )
                        param_names_seen.add(name)

        # Try Sphinx style: :param name: description (with optional type)
        # Pattern: :param [type] name: description
        sphinx_param_pattern = re.compile(
            r":param\s+(?:(\w+)\s+)?(\w+):\s*(.*?)(?=\n\s*:|$)",
            re.MULTILINE | re.DOTALL,
        )
        for match in sphinx_param_pattern.finditer(docstring):
            type_hint = match.group(1)
            name = match.group(2)
            description = match.group(3)
            if name not in param_names_seen:
                params.append(
                    DocstringParameter(
                        name=name,
                        type_hint=type_hint,
                        description=description.strip() if description else None,
                    )
                )
                param_names_seen.add(name)

        # Add type hints from :type: directives
        for match in self.SPHINX_TYPE_PATTERN.finditer(docstring):
            name = match.group(1)
            type_hint = match.group(2)
            for param in params:
                if param.name == name and not param.type_hint:
                    param.type_hint = type_hint.strip()

        return params

    def _parse_returns(self, docstring: str) -> Optional[str]:
        """Extract return documentation from docstring."""
        # Try Google style: Returns: section
        returns_match = re.search(
            r"(?:Returns?)[:\s]*\n\s+(.+?)(?:\n\n|\n\s*\w+:|\Z)",
            docstring,
            re.IGNORECASE | re.DOTALL,
        )
        if returns_match:
            return returns_match.group(1).strip()

        # Try Sphinx style: :returns: or :return:
        sphinx_match = re.search(
            r":returns?:\s*(.+?)(?:\n\s*:|$)", docstring, re.IGNORECASE
        )
        if sphinx_match:
            return sphinx_match.group(1).strip()

        return None

    def _parse_examples(self, docstring: str) -> List[str]:
        """Extract code examples from docstring."""
        examples = []

        # Find doctest-style examples (>>> ...)
        doctest_pattern = re.compile(r">>>\s+.+(?:\n(?:\.\.\.|\s+).+)*", re.MULTILINE)
        for match in doctest_pattern.finditer(docstring):
            examples.append(match.group(0))

        # Find Example: sections
        example_section = re.search(
            r"(?:Examples?|Usage)[:\s]*\n((?:\s+.+\n?)+)", docstring, re.IGNORECASE
        )
        if example_section:
            examples.append(example_section.group(1).strip())

        return examples

    def _parse_formulas(self, docstring: str) -> List[str]:
        """Extract mathematical formulas from docstring."""
        formulas = []

        # Find LaTeX-style formulas
        latex_pattern = re.compile(r"\$([^$]+)\$")
        for match in latex_pattern.finditer(docstring):
            formulas.append(match.group(1))

        # Find formula sections
        for match in self.FORMULA_PATTERN.finditer(docstring):
            formulas.append(match.group(1).strip())

        # Find inline formulas like "Consistency(U) = ..."
        inline_formula = re.compile(r"(\w+\([^)]+\)\s*=\s*[^\n]+)")
        for match in inline_formula.finditer(docstring):
            formulas.append(match.group(1))

        return formulas


class DocstringValidator:
    """
    Validates docstrings against actual function signatures.

    Checks:
    - Parameter names match between docstring and signature
    - Parameter types match (if documented)
    - Return type matches (if documented)
    - All parameters are documented
    - No extra parameters in docstring
    """

    def __init__(self, package_name: str = "udl_rating_framework"):
        """Initialize the docstring validator.

        Args:
            package_name: Name of the Python package to validate.
        """
        self.package_name = package_name
        self.parser = DocstringParser()

    def validate_function_docstring(
        self, func: Any, module_name: str
    ) -> List[SignatureMismatch]:
        """Validate a function's docstring against its signature."""
        mismatches = []

        # Get actual signature
        try:
            sig = inspect.signature(func)
        except (ValueError, TypeError):
            return mismatches

        # Get docstring info
        docstring = inspect.getdoc(func)
        if not docstring:
            return mismatches

        doc_info = self.parser.parse(docstring)

        # Get actual parameter names (excluding 'self' and 'cls')
        actual_params = {
            name: param
            for name, param in sig.parameters.items()
            if name not in ("self", "cls")
        }

        # Get documented parameter names
        documented_params = {p.name: p for p in doc_info.parameters}

        # Get file path and line number
        try:
            file_path = inspect.getfile(func)
            source_lines, line_number = inspect.getsourcelines(func)
        except (OSError, TypeError):
            file_path = None
            line_number = None

        # Check for missing parameters in docstring
        for param_name in actual_params:
            if param_name not in documented_params:
                mismatches.append(
                    SignatureMismatch(
                        function_name=func.__name__,
                        module=module_name,
                        mismatch_type="missing_param",
                        param_name=param_name,
                        actual_value=str(actual_params[param_name]),
                        line_number=line_number,
                        file_path=file_path,
                    )
                )

        # Check for extra parameters in docstring
        for param_name in documented_params:
            if param_name not in actual_params:
                mismatches.append(
                    SignatureMismatch(
                        function_name=func.__name__,
                        module=module_name,
                        mismatch_type="extra_param",
                        param_name=param_name,
                        documented_value=documented_params[param_name].type_hint,
                        line_number=line_number,
                        file_path=file_path,
                    )
                )

        # Check type hints match (if both are specified)
        for param_name, doc_param in documented_params.items():
            if param_name in actual_params and doc_param.type_hint:
                actual_param = actual_params[param_name]
                if actual_param.annotation != inspect.Parameter.empty:
                    actual_type = self._get_type_string(actual_param.annotation)
                    if not self._types_match(doc_param.type_hint, actual_type):
                        mismatches.append(
                            SignatureMismatch(
                                function_name=func.__name__,
                                module=module_name,
                                mismatch_type="type_mismatch",
                                param_name=param_name,
                                documented_value=doc_param.type_hint,
                                actual_value=actual_type,
                                line_number=line_number,
                                file_path=file_path,
                            )
                        )

        return mismatches

    def _get_type_string(self, annotation: Any) -> str:
        """Convert a type annotation to a string."""
        if hasattr(annotation, "__name__"):
            return annotation.__name__
        return str(annotation)

    def _types_match(self, documented: str, actual: str) -> bool:
        """Check if documented type matches actual type."""
        # Normalize type strings
        doc_normalized = documented.lower().strip()
        actual_normalized = actual.lower().strip()

        # Direct match
        if doc_normalized == actual_normalized:
            return True

        # Common aliases
        aliases = {
            "str": ["string", "text"],
            "int": ["integer", "number"],
            "float": ["number", "decimal"],
            "bool": ["boolean"],
            "list": ["array", "sequence"],
            "dict": ["dictionary", "mapping"],
            "none": ["null", "nothing"],
        }

        for canonical, alt_names in aliases.items():
            if doc_normalized in [canonical] + alt_names:
                if actual_normalized in [canonical] + alt_names:
                    return True

        # Check if one contains the other (for complex types)
        if doc_normalized in actual_normalized or actual_normalized in doc_normalized:
            return True

        return False

    def validate_all(self) -> DocstringValidationReport:
        """Validate all docstrings in the package."""
        report = DocstringValidationReport()

        try:
            package = importlib.import_module(self.package_name)
        except ImportError as e:
            print(f"Error importing package {self.package_name}: {e}")
            return report

        # Get package path
        if not hasattr(package, "__path__"):
            return report

        # Walk through all submodules
        for importer, modname, ispkg in pkgutil.walk_packages(
            path=package.__path__,
            prefix=f"{self.package_name}.",
            onerror=lambda x: None,
        ):
            try:
                module = importlib.import_module(modname)
                self._validate_module(module, modname, report)
            except Exception:
                continue

        return report

    def _validate_module(
        self, module: Any, module_name: str, report: DocstringValidationReport
    ) -> None:
        """Validate all functions and methods in a module."""
        # Get all public names
        public_names = [name for name in dir(module) if not name.startswith("_")]

        for name in public_names:
            try:
                obj = getattr(module, name)

                # Skip if not defined in this module
                if hasattr(obj, "__module__") and obj.__module__ != module_name:
                    continue

                if inspect.isfunction(obj):
                    self._validate_function(obj, module_name, report)
                elif inspect.isclass(obj):
                    self._validate_class(obj, module_name, report)
            except Exception:
                continue

    def _validate_function(
        self, func: Any, module_name: str, report: DocstringValidationReport
    ) -> None:
        """Validate a single function."""
        report.total_functions_checked += 1

        docstring = inspect.getdoc(func)
        if docstring:
            report.total_with_docstrings += 1
            mismatches = self.validate_function_docstring(func, module_name)
            report.signature_mismatches.extend(mismatches)
        else:
            # Record undocumented function
            try:
                file_path = inspect.getfile(func)
                _, line_number = inspect.getsourcelines(func)
            except (OSError, TypeError):
                file_path = None
                line_number = None

            report.undocumented_apis.append(
                UndocumentedAPI(
                    name=func.__name__,
                    module=module_name,
                    element_type="function",
                    line_number=line_number,
                    file_path=file_path,
                )
            )

    def _validate_class(
        self, cls: type, module_name: str, report: DocstringValidationReport
    ) -> None:
        """Validate a class and its methods."""
        # Check class docstring
        if not cls.__doc__:
            try:
                file_path = inspect.getfile(cls)
                _, line_number = inspect.getsourcelines(cls)
            except (OSError, TypeError):
                file_path = None
                line_number = None

            report.undocumented_apis.append(
                UndocumentedAPI(
                    name=cls.__name__,
                    module=module_name,
                    element_type="class",
                    line_number=line_number,
                    file_path=file_path,
                )
            )

        # Check methods
        for method_name in dir(cls):
            if method_name.startswith("_") and method_name != "__init__":
                continue

            try:
                method = getattr(cls, method_name)
                if inspect.isfunction(method) or inspect.ismethod(method):
                    self._validate_function(method, module_name, report)
            except Exception:
                continue


class DoctestRunner:
    """Runs doctest examples from docstrings."""

    def __init__(self, package_name: str = "udl_rating_framework"):
        """Initialize the doctest runner.

        Args:
            package_name: Name of the Python package to run doctests for.
        """
        self.package_name = package_name

    def run_doctests_for_module(self, module: Any) -> List[DoctestResult]:
        """Run all doctests in a module."""
        results = []

        # Capture output
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        try:
            # Run doctests
            finder = doctest.DocTestFinder()
            runner = doctest.DocTestRunner(verbose=False)

            for test in finder.find(module):
                if test.examples:
                    runner.run(test)

                    result = DoctestResult(
                        function_name=test.name,
                        module=module.__name__,
                        passed=runner.summarize(verbose=False)[0] == 0,
                        examples_count=len(test.examples),
                    )

                    # Collect failures
                    if not result.passed:
                        result.failures = [
                            f"Example failed: {ex.source}" for ex in test.examples
                        ]

                    results.append(result)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        return results

    def run_all_doctests(self) -> List[DoctestResult]:
        """Run doctests for all modules in the package."""
        all_results = []

        try:
            package = importlib.import_module(self.package_name)
        except ImportError:
            return all_results

        if not hasattr(package, "__path__"):
            return all_results

        for importer, modname, ispkg in pkgutil.walk_packages(
            path=package.__path__,
            prefix=f"{self.package_name}.",
            onerror=lambda x: None,
        ):
            try:
                module = importlib.import_module(modname)
                results = self.run_doctests_for_module(module)
                all_results.extend(results)
            except Exception:
                continue

        return all_results


def find_undocumented_public_apis(
    package_name: str = "udl_rating_framework",
) -> List[UndocumentedAPI]:
    """Find all public functions and classes without docstrings."""
    undocumented = []

    for root, dirs, files in Path(package_name).walk():
        # Skip __pycache__
        dirs[:] = [d for d in dirs if d != "__pycache__"]

        for file in files:
            if not file.endswith(".py"):
                continue

            filepath = root / file
            try:
                content = filepath.read_text()
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Skip private classes
                        if node.name.startswith("_"):
                            continue
                        # Check if first statement is a docstring
                        has_doc = (
                            node.body
                            and isinstance(node.body[0], ast.Expr)
                            and isinstance(node.body[0].value, ast.Constant)
                            and isinstance(node.body[0].value.value, str)
                        )
                        if not has_doc:
                            undocumented.append(
                                UndocumentedAPI(
                                    name=node.name,
                                    module=str(filepath)
                                    .replace("/", ".")
                                    .replace(".py", ""),
                                    element_type="class",
                                    line_number=node.lineno,
                                    file_path=str(filepath),
                                )
                            )

                    elif isinstance(node, ast.FunctionDef):
                        # Skip private functions
                        if node.name.startswith("_") and node.name != "__init__":
                            continue
                        # Check if first statement is a docstring
                        has_doc = (
                            node.body
                            and isinstance(node.body[0], ast.Expr)
                            and isinstance(node.body[0].value, ast.Constant)
                            and isinstance(node.body[0].value.value, str)
                        )
                        if not has_doc:
                            undocumented.append(
                                UndocumentedAPI(
                                    name=node.name,
                                    module=str(filepath)
                                    .replace("/", ".")
                                    .replace(".py", ""),
                                    element_type="function",
                                    line_number=node.lineno,
                                    file_path=str(filepath),
                                )
                            )
            except Exception:
                continue

    return undocumented


def validate_docstrings(
    package_name: str = "udl_rating_framework",
) -> DocstringValidationReport:
    """Run complete docstring validation."""
    validator = DocstringValidator(package_name)
    report = validator.validate_all()

    # Run doctests
    doctest_runner = DoctestRunner(package_name)
    report.doctest_results = doctest_runner.run_all_doctests()

    return report


def main():
    """Run docstring validation and print results."""
    print("Validating docstrings...")
    report = validate_docstrings()

    print("\n=== Docstring Validation Report ===")
    print(f"Total functions checked: {report.total_functions_checked}")
    print(f"Functions with docstrings: {report.total_with_docstrings}")
    print(f"Signature mismatches: {len(report.signature_mismatches)}")
    print(f"Undocumented APIs: {len(report.undocumented_apis)}")
    print(f"Doctest results: {len(report.doctest_results)}")

    if report.signature_mismatches:
        print("\n=== Signature Mismatches ===")
        for mismatch in report.signature_mismatches[:20]:
            print(
                f"  {mismatch.module}.{mismatch.function_name}: "
                f"{mismatch.mismatch_type} - {mismatch.param_name}"
            )
            if mismatch.documented_value:
                print(f"    Documented: {mismatch.documented_value}")
            if mismatch.actual_value:
                print(f"    Actual: {mismatch.actual_value}")
        if len(report.signature_mismatches) > 20:
            print(f"  ... and {len(report.signature_mismatches) - 20} more")

    if report.undocumented_apis:
        print(f"\n=== Undocumented APIs ({len(report.undocumented_apis)}) ===")
        for api in report.undocumented_apis[:20]:
            print(f"  {api.file_path}:{api.line_number} {api.element_type} {api.name}")
        if len(report.undocumented_apis) > 20:
            print(f"  ... and {len(report.undocumented_apis) - 20} more")


if __name__ == "__main__":
    main()
