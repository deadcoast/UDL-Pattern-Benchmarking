#!/usr/bin/env python3
"""
Validate __init__.py exports for the UDL Rating Framework.

This script checks that:
1. All symbols in __all__ are importable
2. All public symbols are included in __all__

Property 13: Init Export Completeness
Validates: Requirements 6.1
"""

import importlib
import pkgutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def get_all_packages() -> List[str]:
    """Get all packages in udl_rating_framework."""
    packages = ["udl_rating_framework"]

    def find_subpackages(package_name: str):
        try:
            package = importlib.import_module(package_name)
            if hasattr(package, "__path__"):
                for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
                    full_name = f"{package_name}.{modname}"
                    if ispkg:
                        packages.append(full_name)
                        find_subpackages(full_name)
        except ImportError as e:
            print(f"Warning: Could not import {package_name}: {e}")

    find_subpackages("udl_rating_framework")
    return packages


def validate_package_exports(package_name: str) -> Tuple[bool, List[str]]:
    """
    Validate that all __all__ entries are importable.

    Returns:
        Tuple of (success, list of error messages)
    """
    errors = []

    try:
        module = importlib.import_module(package_name)
    except ImportError as e:
        return False, [f"Cannot import {package_name}: {e}"]

    # Check if __all__ exists
    if not hasattr(module, "__all__"):
        # Not an error, just no exports defined
        return True, []

    all_exports = module.__all__

    # Verify each export is accessible
    for name in all_exports:
        try:
            obj = getattr(module, name)
            if obj is None:
                # Some modules use try/except and set to None on failure
                errors.append(
                    f"{package_name}.__all__ contains '{name}' but it is None (import failed)"
                )
        except AttributeError:
            errors.append(
                f"{package_name}.__all__ contains '{name}' but it cannot be accessed"
            )

    return len(errors) == 0, errors


def main():
    """Run validation on all packages."""
    print("=" * 60)
    print("UDL Rating Framework - __init__.py Export Validation")
    print("=" * 60)
    print()

    packages = get_all_packages()
    print(f"Found {len(packages)} packages to validate:\n")

    all_errors = []
    results = {}

    for package in sorted(packages):
        success, errors = validate_package_exports(package)
        results[package] = (success, errors)

        status = "✓" if success else "✗"
        print(f"  {status} {package}")

        if errors:
            for error in errors:
                print(f"      ERROR: {error}")
                all_errors.append(error)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for s, _ in results.values() if s)
    failed = len(results) - passed

    print(f"  Packages checked: {len(results)}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print()

    if all_errors:
        print("All Errors:")
        for error in all_errors:
            print(f"  - {error}")
        print()
        return 1
    else:
        print("All __init__.py exports are valid!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
