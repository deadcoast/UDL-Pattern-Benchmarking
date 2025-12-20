#!/usr/bin/env python3
"""
Find orphaned Python files in the UDL Rating Framework.

Orphaned files are Python files that are not imported anywhere in the codebase.

Requirements: 6.2
"""

import ast
import sys
from pathlib import Path
from typing import Set, Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent


def get_all_python_files(directory: Path) -> Set[Path]:
    """Get all Python files in a directory recursively."""
    files = set()
    for py_file in directory.rglob("*.py"):
        # Skip __pycache__ and test files
        if "__pycache__" in str(py_file):
            continue
        files.add(py_file.relative_to(project_root))
    return files


def extract_imports(file_path: Path) -> Set[str]:
    """Extract all import statements from a Python file."""
    imports = set()
    
    try:
        with open(project_root / file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
                    # Also add the full path for relative imports
                    for alias in node.names:
                        imports.add(f"{node.module}.{alias.name}")
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Warning: Could not parse {file_path}: {e}")
    
    return imports


def module_path_to_import(file_path: Path) -> str:
    """Convert a file path to its import path."""
    # Remove .py extension and convert to module path
    parts = list(file_path.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].replace(".py", "")
    return ".".join(parts)


def main():
    """Find orphaned files in the codebase."""
    print("=" * 60)
    print("UDL Rating Framework - Orphaned File Detection")
    print("=" * 60)
    print()
    
    # Get all Python files in the framework
    framework_dir = project_root / "udl_rating_framework"
    framework_files = get_all_python_files(framework_dir)
    
    # Get all Python files in tests
    tests_dir = project_root / "tests"
    test_files = get_all_python_files(tests_dir) if tests_dir.exists() else set()
    
    # Get all Python files in scripts
    scripts_dir = project_root / "scripts"
    script_files = get_all_python_files(scripts_dir) if scripts_dir.exists() else set()
    
    # Get all Python files in examples
    examples_dir = project_root / "examples"
    example_files = get_all_python_files(examples_dir) if examples_dir.exists() else set()
    
    all_files = framework_files | test_files | script_files | example_files
    
    print(f"Found {len(framework_files)} framework files")
    print(f"Found {len(test_files)} test files")
    print(f"Found {len(script_files)} script files")
    print(f"Found {len(example_files)} example files")
    print()
    
    # Collect all imports from all files
    all_imports: Set[str] = set()
    for file_path in all_files:
        imports = extract_imports(file_path)
        all_imports.update(imports)
    
    # Convert framework files to import paths
    file_to_import: Dict[Path, str] = {}
    for file_path in framework_files:
        import_path = module_path_to_import(file_path)
        file_to_import[file_path] = import_path
    
    # Find orphaned files
    orphaned: List[Path] = []
    imported: List[Path] = []
    
    for file_path, import_path in file_to_import.items():
        # Skip __init__.py files - they're always needed
        if file_path.name == "__init__.py":
            continue
        
        # Check if this module is imported anywhere
        is_imported = False
        for imp in all_imports:
            if imp == import_path or imp.startswith(import_path + "."):
                is_imported = True
                break
            # Check if any part of the import matches
            if import_path in imp or imp in import_path:
                is_imported = True
                break
        
        if is_imported:
            imported.append(file_path)
        else:
            orphaned.append(file_path)
    
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print()
    
    if orphaned:
        print(f"Found {len(orphaned)} potentially orphaned files:")
        print()
        for f in sorted(orphaned):
            print(f"  - {f}")
        print()
        print("Note: These files may be:")
        print("  - Entry points (CLI, scripts)")
        print("  - Dynamically imported")
        print("  - Unused and should be removed")
    else:
        print("No orphaned files found!")
    
    print()
    print(f"Summary: {len(imported)} imported, {len(orphaned)} potentially orphaned")
    
    return 0 if not orphaned else 1


if __name__ == "__main__":
    sys.exit(main())
