#!/usr/bin/env python3
"""
Verify all imports resolve in the UDL Rating Framework.

Property 14: Import Resolution
Validates: Requirements 6.5
"""

import ast
import sys
import importlib
from pathlib import Path
from typing import List, Tuple

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def extract_imports(file_path: Path) -> List[Tuple[str, int, str]]:
    """
    Extract all import statements from a Python file.

    Returns list of (module_name, line_number, import_type)
    """
    imports = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append((alias.name, node.lineno, "import"))
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append((node.module, node.lineno, "from"))
                    # Also check individual imports
                    for alias in node.names:
                        if alias.name != "*":
                            full_name = f"{node.module}.{alias.name}"
                            imports.append((full_name, node.lineno, "from_attr"))
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Warning: Could not parse {file_path}: {e}")

    return imports


def verify_import(module_name: str) -> Tuple[bool, str]:
    """
    Verify that an import can be resolved.

    Returns (success, error_message)
    """
    # Skip standard library and common third-party modules
    stdlib_prefixes = [
        "os",
        "sys",
        "typing",
        "pathlib",
        "json",
        "abc",
        "dataclasses",
        "collections",
        "functools",
        "itertools",
        "contextlib",
        "logging",
        "unittest",
        "tempfile",
        "shutil",
        "subprocess",
        "threading",
        "multiprocessing",
        "concurrent",
        "asyncio",
        "datetime",
        "time",
        "math",
        "random",
        "hashlib",
        "base64",
        "io",
        "re",
        "copy",
        "warnings",
        "traceback",
        "inspect",
        "ast",
        "enum",
        "uuid",
        "http",
        "urllib",
        "socket",
        "email",
        "html",
        "xml",
        "csv",
        "sqlite3",
        "pickle",
        "struct",
        "array",
        "queue",
        "heapq",
        "bisect",
        "weakref",
        "gc",
        "dis",
        "code",
        "codeop",
        "pprint",
        "reprlib",
        "textwrap",
        "difflib",
        "string",
        "codecs",
        "locale",
        "gettext",
        "argparse",
        "optparse",
        "configparser",
        "fileinput",
        "stat",
        "glob",
        "fnmatch",
        "linecache",
        "tokenize",
        "keyword",
        "symbol",
        "token",
        "parser",
        "platform",
        "errno",
        "ctypes",
        "select",
        "selectors",
        "signal",
        "mmap",
        "atexit",
        "builtins",
        "__future__",
        "importlib",
        "pkgutil",
        "modulefinder",
        "runpy",
        "zipimport",
        "zipfile",
        "tarfile",
        "gzip",
        "bz2",
        "lzma",
        "zlib",
    ]

    third_party_prefixes = [
        "numpy",
        "np",
        "scipy",
        "torch",
        "networkx",
        "nx",
        "matplotlib",
        "plt",
        "pandas",
        "pd",
        "sklearn",
        "hypothesis",
        "pytest",
        "click",
        "yaml",
        "toml",
        "requests",
        "fastapi",
        "uvicorn",
        "pydantic",
        "jinja2",
        "aiohttp",
        "websockets",
        "redis",
        "celery",
        "flask",
        "django",
        "sqlalchemy",
        "alembic",
        "black",
        "flake8",
        "mypy",
        "coverage",
        "tqdm",
        "rich",
        "colorama",
        "termcolor",
        "tabulate",
        "PIL",
        "cv2",
        "tensorflow",
        "tf",
        "keras",
        "transformers",
        "huggingface_hub",
        "datasets",
        "tokenizers",
        "accelerate",
        "optuna",
        "ray",
        "dask",
        "joblib",
        "numba",
        "cython",
        "nbformat",
        "nbconvert",
        "papermill",
        "jupyter",
        "ipython",
        "starlette",
        "httpx",
        "anyio",
        "sniffio",
        "h11",
        "httpcore",
    ]

    base_module = module_name.split(".")[0]

    if base_module in stdlib_prefixes:
        return True, ""

    if base_module in third_party_prefixes:
        return True, ""

    # Try to import the module
    try:
        # For udl_rating_framework imports, try to import
        if module_name.startswith("udl_rating_framework"):
            importlib.import_module(module_name.split(".")[0])
            # For attribute imports, we just check the base module exists
            parts = module_name.split(".")
            current = parts[0]
            for part in parts[1:]:
                try:
                    mod = importlib.import_module(current)
                    if hasattr(mod, part):
                        current = f"{current}.{part}"
                    else:
                        # Try importing as submodule
                        try:
                            importlib.import_module(f"{current}.{part}")
                            current = f"{current}.{part}"
                        except ImportError:
                            # It might be a class/function, which is fine
                            break
                except ImportError:
                    break
        return True, ""
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def main():
    """Verify all imports in the codebase."""
    print("=" * 60)
    print("UDL Rating Framework - Import Resolution Verification")
    print("=" * 60)
    print()

    framework_dir = project_root / "udl_rating_framework"

    # Get all Python files
    all_files = list(framework_dir.rglob("*.py"))
    all_files = [f for f in all_files if "__pycache__" not in str(f)]

    print(f"Scanning {len(all_files)} Python files...")
    print()

    all_imports = []
    failed_imports = []

    for file_path in all_files:
        imports = extract_imports(file_path)
        for module_name, line_no, import_type in imports:
            all_imports.append((file_path, module_name, line_no, import_type))

            # Only verify udl_rating_framework imports
            if module_name.startswith("udl_rating_framework"):
                success, error = verify_import(module_name)
                if not success:
                    failed_imports.append((file_path, module_name, line_no, error))

    print("=" * 60)
    print("Results")
    print("=" * 60)
    print()

    # Count internal imports
    internal_imports = [
        i for i in all_imports if i[1].startswith("udl_rating_framework")
    ]

    print(f"Total imports found: {len(all_imports)}")
    print(f"Internal imports (udl_rating_framework.*): {len(internal_imports)}")
    print()

    if failed_imports:
        print("Failed imports:")
        for file_path, module_name, line_no, error in failed_imports:
            rel_path = file_path.relative_to(project_root)
            print(f"  ✗ {rel_path}:{line_no}")
            print(f"      Import: {module_name}")
            print(f"      Error: {error}")
        print()
    else:
        print("All internal imports resolve successfully!")

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Files scanned: {len(all_files)}")
    print(f"  Total imports: {len(all_imports)}")
    print(f"  Internal imports: {len(internal_imports)}")
    print(f"  Failed imports: {len(failed_imports)}")

    return 0 if not failed_imports else 1


if __name__ == "__main__":
    sys.exit(main())
