#!/usr/bin/env python3
"""
Verify module organization matches documented architecture.

Compares actual project structure to documented structure in README.md.

Requirements: 6.3
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent

# Expected structure from README.md
EXPECTED_STRUCTURE = {
    "udl_rating_framework": {
        "core": {
            "representation.py": True,
            "metrics": True,  # Directory
            "aggregation.py": True,
            "confidence.py": True,
            "caching.py": True,
            "pipeline.py": True,
            "performance.py": True,
        },
        "models": {
            "ctm_adapter.py": True,
        },
        "io": {
            "file_discovery.py": True,
            "input_validation.py": True,
            "report_generator.py": True,
        },
        "evaluation": {
            "comparison.py": True,
            "evaluation_suite.py": True,
        },
        "analytics": {
            "portfolio_analyzer.py": True,
            "trend_predictor.py": True,
            "bi_exporter.py": True,
        },
        "training": {
            "training_pipeline.py": True,
            "active_learning.py": True,
            "ensemble_methods.py": True,
        },
        "visualization": {
            "web_visualizer.py": True,
            "realtime_metrics.py": True,
            "webgl_visualizer.py": True,
        },
        "integration": {
            "cicd.py": True,
            "git_hooks.py": True,
            "lsp_server.py": True,
        },
        "validation": True,  # Directory
        "benchmarks": True,  # Directory
        "utils": True,  # Directory
        "cli": {
            "commands": True,  # Directory
        },
    },
    "tests": {
        "unit": True,  # Directory
        "conftest.py": True,
    },
    "docs": {
        "mathematical_framework.tex": True,
        "api_reference.rst": True,
        "examples.rst": True,
    },
}


def check_structure(
    expected: dict, base_path: Path, prefix: str = ""
) -> tuple[list, list]:
    """
    Check if expected structure exists.

    Returns:
        Tuple of (missing items, extra items not in docs)
    """
    missing = []

    for name, value in expected.items():
        path = base_path / name
        full_name = f"{prefix}/{name}" if prefix else name

        if isinstance(value, dict):
            # It's a directory with contents
            if not path.exists():
                missing.append(f"Directory: {full_name}")
            elif not path.is_dir():
                missing.append(f"Expected directory but found file: {full_name}")
            else:
                # Recursively check contents
                sub_missing, _ = check_structure(value, path, full_name)
                missing.extend(sub_missing)
        elif value is True:
            # It's a file or directory that should exist
            if not path.exists():
                missing.append(
                    f"{'Directory' if '.' not in name else 'File'}: {full_name}"
                )

    return missing, []


def find_undocumented(base_path: Path, expected: dict, prefix: str = "") -> list:
    """Find files/directories that exist but aren't documented."""
    undocumented = []

    if not base_path.exists():
        return undocumented

    for item in base_path.iterdir():
        # Skip hidden files and __pycache__
        if item.name.startswith(".") or item.name == "__pycache__":
            continue

        full_name = f"{prefix}/{item.name}" if prefix else item.name

        if item.name not in expected:
            # Not documented
            if item.is_dir():
                undocumented.append(f"Directory: {full_name}")
            else:
                undocumented.append(f"File: {full_name}")
        elif isinstance(expected.get(item.name), dict) and item.is_dir():
            # Recursively check subdirectories
            sub_undoc = find_undocumented(item, expected[item.name], full_name)
            undocumented.extend(sub_undoc)

    return undocumented


def main():
    """Verify project structure matches documentation."""
    print("=" * 60)
    print("UDL Rating Framework - Structure Verification")
    print("=" * 60)
    print()

    # Check for missing items
    missing, _ = check_structure(EXPECTED_STRUCTURE, project_root)

    # Check for undocumented items in key directories
    undocumented = []
    for key in ["udl_rating_framework"]:
        if key in EXPECTED_STRUCTURE:
            undoc = find_undocumented(project_root / key, EXPECTED_STRUCTURE[key], key)
            undocumented.extend(undoc)

    print("Missing (documented but not found):")
    if missing:
        for item in sorted(missing):
            print(f"  ✗ {item}")
    else:
        print("  None - all documented items exist!")

    print()
    print("Undocumented (exists but not in README):")
    if undocumented:
        for item in sorted(undocumented):
            print(f"  ? {item}")
    else:
        print("  None - all items are documented!")

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Missing items: {len(missing)}")
    print(f"  Undocumented items: {len(undocumented)}")

    if missing:
        print()
        print("Action needed: Update code or documentation to resolve missing items")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
