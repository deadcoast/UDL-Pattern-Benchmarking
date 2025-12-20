#!/usr/bin/env python3
"""
Verify test file correspondence for UDL Rating Framework.

Checks that each source module has a corresponding test file.

Property 15: Test Module Correspondence
Validates: Requirements 6.4
"""

import sys
from pathlib import Path
from typing import Dict, List, Set

project_root = Path(__file__).parent.parent


def get_source_modules(framework_dir: Path) -> Set[str]:
    """Get all source module names (excluding __init__.py)."""
    modules = set()

    for py_file in framework_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        if py_file.name == "__init__.py":
            continue

        # Get relative path from framework dir
        rel_path = py_file.relative_to(framework_dir)
        # Convert to module name
        module_name = str(rel_path).replace("/", "_").replace(".py", "")
        modules.add(module_name)

    return modules


def get_test_coverage(tests_dir: Path) -> Dict[str, List[str]]:
    """
    Analyze test files to determine what they cover.

    Returns dict mapping source module patterns to test files that cover them.
    """
    coverage = {}

    if not tests_dir.exists():
        return coverage

    for test_file in tests_dir.rglob("test_*.py"):
        if "__pycache__" in str(test_file):
            continue

        # Extract what the test covers from its name
        test_name = test_file.stem  # e.g., "test_consistency_metric"

        # Remove "test_" prefix
        covered = test_name[5:] if test_name.startswith("test_") else test_name

        if covered not in coverage:
            coverage[covered] = []
        coverage[covered].append(str(test_file.relative_to(project_root)))

    return coverage


def find_module_test_mapping(
    source_modules: Set[str], test_coverage: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """Map source modules to their test files."""
    mapping = {}

    for module in source_modules:
        tests = []

        # Direct match
        if module in test_coverage:
            tests.extend(test_coverage[module])

        # Partial match (e.g., "consistency" matches "metrics_consistency")
        module_parts = module.split("_")
        for test_name, test_files in test_coverage.items():
            test_parts = test_name.split("_")
            # Check if any significant part of module name is in test name
            for part in module_parts:
                if len(part) > 3 and part in test_parts:
                    tests.extend(test_files)
                    break

        mapping[module] = list(set(tests))  # Remove duplicates

    return mapping


def main():
    """Verify test file correspondence."""
    print("=" * 60)
    print("UDL Rating Framework - Test Correspondence Verification")
    print("=" * 60)
    print()

    framework_dir = project_root / "udl_rating_framework"
    tests_dir = project_root / "tests"

    # Get source modules
    source_modules = get_source_modules(framework_dir)
    print(f"Found {len(source_modules)} source modules")

    # Get test coverage
    test_coverage = get_test_coverage(tests_dir)
    print(f"Found {len(test_coverage)} test file patterns")
    print()

    # Map modules to tests
    mapping = find_module_test_mapping(source_modules, test_coverage)

    # Categorize results
    covered = []
    uncovered = []

    for module, tests in sorted(mapping.items()):
        if tests:
            covered.append((module, tests))
        else:
            uncovered.append(module)

    print("=" * 60)
    print("Modules WITH test coverage:")
    print("=" * 60)
    for module, tests in covered:
        print(f"  ✓ {module}")
        for test in tests[:2]:  # Show max 2 test files
            print(f"      → {test}")
        if len(tests) > 2:
            print(f"      ... and {len(tests) - 2} more")

    print()
    print("=" * 60)
    print("Modules WITHOUT direct test coverage:")
    print("=" * 60)
    if uncovered:
        for module in uncovered:
            print(f"  ✗ {module}")
    else:
        print("  All modules have test coverage!")

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Total source modules: {len(source_modules)}")
    print(f"  Modules with tests: {len(covered)}")
    print(f"  Modules without tests: {len(uncovered)}")
    coverage_pct = (len(covered) / len(source_modules)
                    * 100) if source_modules else 0
    print(f"  Test coverage: {coverage_pct:.1f}%")

    print()
    print("Note: Some modules may be tested indirectly through integration tests")
    print("or property-based tests that cover multiple modules.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
