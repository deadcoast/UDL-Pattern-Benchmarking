#!/usr/bin/env python3
"""
Verify that all dependencies in pyproject.toml are used in the codebase,
and that all third-party imports are listed in pyproject.toml.

Property 16: Dependency Usage - For any dependency listed in pyproject.toml,
the package should be imported somewhere in the codebase.

Property 17: Import Coverage - For any third-party package imported in the codebase,
it should be listed in pyproject.toml dependencies.

Validates: Requirements 7.1, 7.2
"""

import ast
import os
import re
import sys
from pathlib import Path
from typing import Set, Dict, List, Tuple

# Standard library modules (Python 3.10+)
STDLIB_MODULES = {
    'abc', 'aifc', 'argparse', 'array', 'ast', 'asynchat', 'asyncio', 'asyncore',
    'atexit', 'audioop', 'base64', 'bdb', 'binascii', 'binhex', 'bisect',
    'builtins', 'bz2', 'calendar', 'cgi', 'cgitb', 'chunk', 'cmath', 'cmd',
    'code', 'codecs', 'codeop', 'collections', 'colorsys', 'compileall',
    'concurrent', 'configparser', 'contextlib', 'contextvars', 'copy', 'copyreg',
    'cProfile', 'crypt', 'csv', 'ctypes', 'curses', 'dataclasses', 'datetime',
    'dbm', 'decimal', 'difflib', 'dis', 'distutils', 'doctest', 'email',
    'encodings', 'enum', 'errno', 'faulthandler', 'fcntl', 'filecmp', 'fileinput',
    'fnmatch', 'fractions', 'ftplib', 'functools', 'gc', 'getopt', 'getpass',
    'gettext', 'glob', 'graphlib', 'grp', 'gzip', 'hashlib', 'heapq', 'hmac',
    'html', 'http', 'idlelib', 'imaplib', 'imghdr', 'imp', 'importlib', 'inspect',
    'io', 'ipaddress', 'itertools', 'json', 'keyword', 'lib2to3', 'linecache',
    'locale', 'logging', 'lzma', 'mailbox', 'mailcap', 'marshal', 'math',
    'mimetypes', 'mmap', 'modulefinder', 'multiprocessing', 'netrc', 'nis',
    'nntplib', 'numbers', 'operator', 'optparse', 'os', 'ossaudiodev', 'pathlib',
    'pdb', 'pickle', 'pickletools', 'pipes', 'pkgutil', 'platform', 'plistlib',
    'poplib', 'posix', 'posixpath', 'pprint', 'profile', 'pstats', 'pty', 'pwd',
    'py_compile', 'pyclbr', 'pydoc', 'queue', 'quopri', 'random', 're', 'readline',
    'reprlib', 'resource', 'rlcompleter', 'runpy', 'sched', 'secrets', 'select',
    'selectors', 'shelve', 'shlex', 'shutil', 'signal', 'site', 'smtpd', 'smtplib',
    'sndhdr', 'socket', 'socketserver', 'spwd', 'sqlite3', 'ssl', 'stat',
    'statistics', 'string', 'stringprep', 'struct', 'subprocess', 'sunau',
    'symtable', 'sys', 'sysconfig', 'syslog', 'tabnanny', 'tarfile', 'telnetlib',
    'tempfile', 'termios', 'test', 'textwrap', 'threading', 'time', 'timeit',
    'tkinter', 'token', 'tokenize', 'trace', 'traceback', 'tracemalloc', 'tty',
    'turtle', 'turtledemo', 'types', 'typing', 'typing_extensions', 'unicodedata',
    'unittest', 'urllib', 'uu', 'uuid', 'venv', 'warnings', 'wave', 'weakref',
    'webbrowser', 'winreg', 'winsound', 'wsgiref', 'xdrlib', 'xml', 'xmlrpc',
    'zipapp', 'zipfile', 'zipimport', 'zlib', '_thread', '__future__',
}

# Mapping from import names to pyproject.toml package names
IMPORT_TO_PACKAGE = {
    'cv2': 'opencv-python',
    'PIL': 'pillow',
    'sklearn': 'scikit-learn',
    'yaml': 'pyyaml',
    'dotenv': 'python-dotenv',
    'umap': 'umap-learn',
    'jose': 'python-jose',
    'memory_profiler': 'memory-profiler',
    'gym': 'gymnasium',
}

# Optional dependencies that are imported with try/except
OPTIONAL_DEPENDENCIES = {
    'cupy',      # GPU acceleration (optional)
    'dask',      # Distributed computing (optional)
    'ray',       # Distributed computing (optional)
    'optuna',    # Hyperparameter optimization (optional)
    'manim',     # Animation (optional, for blog gifs)
    'transformers',  # Transfer learning (optional)
}

# Test-only dependencies (should be in dev dependencies)
TEST_DEPENDENCIES = {
    'pytest',
}

# Build-time dependencies (not needed at runtime)
BUILD_DEPENDENCIES = {
    'setuptools',
}

# Transitive dependencies (provided by other packages)
TRANSITIVE_DEPENDENCIES = {
    'urllib3',  # Provided by requests
}

# Dependencies used indirectly (via other packages' APIs)
INDIRECT_DEPENDENCIES = {
    'python-multipart',  # Required by FastAPI for file uploads
    'tensorboard',       # Used via torch.utils.tensorboard
}

# Local modules that might be imported dynamically
LOCAL_DYNAMIC_IMPORTS = {
    'ctm',           # Local models/ctm.py
    'python_client', # Local deployment/client/python_client.py
}

# Mapping from pyproject.toml package names to import names
PACKAGE_TO_IMPORT = {
    'opencv-python': ['cv2'],
    'pillow': ['PIL'],
    'scikit-learn': ['sklearn'],
    'pyyaml': ['yaml'],
    'python-dotenv': ['dotenv'],
    'umap-learn': ['umap'],
    'python-jose': ['jose'],
    'python-jose[cryptography]': ['jose'],
    'memory-profiler': ['memory_profiler'],
    'gymnasium': ['gymnasium', 'gym'],
    'uvicorn[standard]': ['uvicorn'],
    'python-multipart': ['multipart'],
}


def parse_pyproject_dependencies(pyproject_path: Path) -> Set[str]:
    """Extract dependency names from pyproject.toml."""
    dependencies = set()
    
    with open(pyproject_path, 'r') as f:
        content = f.read()
    
    # Find the dependencies section
    in_deps = False
    for line in content.split('\n'):
        if line.strip() == 'dependencies = [':
            in_deps = True
            continue
        if in_deps:
            if line.strip() == ']':
                break
            # Extract package name from line like '    "numpy>=1.24.0",'
            match = re.search(r'"([a-zA-Z0-9_-]+)', line)
            if match:
                pkg_name = match.group(1).lower()
                dependencies.add(pkg_name)
    
    return dependencies


def get_import_names_for_package(package: str) -> List[str]:
    """Get possible import names for a package."""
    package_lower = package.lower()
    
    # Check explicit mapping
    if package_lower in PACKAGE_TO_IMPORT:
        return PACKAGE_TO_IMPORT[package_lower]
    
    # Handle packages with extras like 'uvicorn[standard]'
    base_package = re.sub(r'\[.*\]', '', package_lower)
    if base_package in PACKAGE_TO_IMPORT:
        return PACKAGE_TO_IMPORT[base_package]
    
    # Default: use package name with hyphens replaced by underscores
    import_name = base_package.replace('-', '_')
    return [import_name]


def extract_imports_from_file(filepath: Path) -> Set[str]:
    """Extract all import statements from a Python file."""
    imports = set()
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Get the top-level module
                    module = alias.name.split('.')[0]
                    imports.add(module)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # Get the top-level module
                    module = node.module.split('.')[0]
                    imports.add(module)
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Warning: Could not parse {filepath}: {e}")
    
    return imports


def get_local_packages(root_dir: Path) -> Set[str]:
    """Get all local package names in the project."""
    local_packages = set()
    
    # Add top-level directories that are Python packages
    for item in root_dir.iterdir():
        if item.is_dir() and (item / '__init__.py').exists():
            local_packages.add(item.name)
        # Also add directories that contain Python files (even without __init__.py)
        elif item.is_dir() and any(f.suffix == '.py' for f in item.iterdir() if f.is_file()):
            local_packages.add(item.name)
    
    # Add known local module names from udl_rating_framework subpackages
    udl_framework = root_dir / 'udl_rating_framework'
    if udl_framework.exists():
        for item in udl_framework.iterdir():
            if item.is_dir() and (item / '__init__.py').exists():
                local_packages.add(item.name)
            # Also add Python files as module names
            elif item.is_file() and item.suffix == '.py' and item.stem != '__init__':
                local_packages.add(item.stem)
        
        # Recursively add submodule names
        for subdir in ['core', 'cli', 'models', 'training', 'validation', 'visualization', 
                       'integration', 'analytics', 'benchmarks', 'evaluation', 'io', 'utils']:
            subpath = udl_framework / subdir
            if subpath.exists():
                for item in subpath.iterdir():
                    if item.is_file() and item.suffix == '.py' and item.stem != '__init__':
                        local_packages.add(item.stem)
    
    return local_packages


def scan_codebase_imports(root_dir: Path, exclude_dirs: Set[str] = None) -> Set[str]:
    """Scan all Python files in the codebase for imports."""
    if exclude_dirs is None:
        exclude_dirs = {'.venv', 'venv', '.git', '__pycache__', 'node_modules', '.hypothesis', 'htmlcov', 'dist'}
    
    all_imports = set()
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip excluded directories
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        
        for filename in filenames:
            if filename.endswith('.py'):
                filepath = Path(dirpath) / filename
                imports = extract_imports_from_file(filepath)
                all_imports.update(imports)
    
    return all_imports


def normalize_import_to_package(import_name: str) -> str:
    """Convert an import name to its package name."""
    if import_name in IMPORT_TO_PACKAGE:
        return IMPORT_TO_PACKAGE[import_name]
    return import_name.replace('_', '-')


def verify_dependencies(project_root: Path) -> Tuple[Set[str], Set[str], Dict[str, List[str]]]:
    """
    Verify dependencies match imports.
    
    Returns:
        - unused_deps: Dependencies in pyproject.toml not imported anywhere
        - missing_deps: Third-party imports not in pyproject.toml
        - usage_map: Map of dependency to files that import it
    """
    pyproject_path = project_root / 'pyproject.toml'
    
    # Get declared dependencies
    declared_deps = parse_pyproject_dependencies(pyproject_path)
    print(f"Found {len(declared_deps)} declared dependencies in pyproject.toml")
    
    # Scan codebase for imports
    all_imports = scan_codebase_imports(project_root)
    print(f"Found {len(all_imports)} unique imports in codebase")
    
    # Get local packages
    local_packages = get_local_packages(project_root)
    print(f"Found {len(local_packages)} local packages/modules")
    
    # Filter out stdlib and local imports
    third_party_imports = set()
    for imp in all_imports:
        if imp in STDLIB_MODULES:
            continue
        # Skip local packages
        if imp in local_packages:
            continue
        # Skip common local module patterns
        if imp in {'udl_rating_framework', 'models', 'utils', 'data', 'tasks', 'tests', 
                   'scripts', 'deployment', 'examples', 'conftest'}:
            continue
        third_party_imports.add(imp)
    
    print(f"Found {len(third_party_imports)} third-party imports")
    
    # Check which declared deps are used
    unused_deps = set()
    usage_map = {}
    
    for dep in declared_deps:
        import_names = get_import_names_for_package(dep)
        found = False
        for import_name in import_names:
            if import_name in third_party_imports:
                found = True
                if dep not in usage_map:
                    usage_map[dep] = []
                usage_map[dep].append(import_name)
        if not found:
            # Check if it's an indirect dependency
            if dep.lower() in INDIRECT_DEPENDENCIES:
                usage_map[dep] = ['(indirect)']
            else:
                unused_deps.add(dep)
    
    # Check which imports are missing from deps
    missing_deps = set()
    optional_missing = set()
    test_missing = set()
    declared_import_names = set()
    for dep in declared_deps:
        declared_import_names.update(get_import_names_for_package(dep))
    
    for imp in third_party_imports:
        if imp in LOCAL_DYNAMIC_IMPORTS:
            continue
        if imp in BUILD_DEPENDENCIES:
            continue
        if imp in TRANSITIVE_DEPENDENCIES:
            continue
        if imp not in declared_import_names:
            # Check if it's a submodule of a declared package
            is_submodule = False
            for declared_imp in declared_import_names:
                if imp.startswith(declared_imp + '.') or declared_imp.startswith(imp + '.'):
                    is_submodule = True
                    break
            if not is_submodule:
                if imp in OPTIONAL_DEPENDENCIES:
                    optional_missing.add(imp)
                elif imp in TEST_DEPENDENCIES:
                    test_missing.add(imp)
                else:
                    missing_deps.add(imp)
    
    return unused_deps, missing_deps, optional_missing, test_missing, usage_map


def main():
    project_root = Path(__file__).parent.parent
    
    print("=" * 60)
    print("Dependency Verification Report")
    print("=" * 60)
    print()
    
    unused_deps, missing_deps, optional_missing, test_missing, usage_map = verify_dependencies(project_root)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    if unused_deps:
        print(f"\n⚠️  POTENTIALLY UNUSED DEPENDENCIES ({len(unused_deps)}):")
        print("   These are listed in pyproject.toml but not directly imported:")
        for dep in sorted(unused_deps):
            print(f"   - {dep}")
    else:
        print("\n✅ All declared dependencies are used")
    
    if missing_deps:
        print(f"\n❌ MISSING DEPENDENCIES ({len(missing_deps)}):")
        print("   These are imported but not in pyproject.toml:")
        for dep in sorted(missing_deps):
            pkg_name = normalize_import_to_package(dep)
            print(f"   - {dep} (add as: {pkg_name})")
    else:
        print("\n✅ All required third-party imports are declared")
    
    if optional_missing:
        print(f"\n📦 OPTIONAL DEPENDENCIES NOT IN pyproject.toml ({len(optional_missing)}):")
        print("   These are optional (imported with try/except):")
        for dep in sorted(optional_missing):
            pkg_name = normalize_import_to_package(dep)
            print(f"   - {dep} (consider adding to [project.optional-dependencies])")
    
    if test_missing:
        print(f"\n🧪 TEST DEPENDENCIES ({len(test_missing)}):")
        print("   These should be in [project.optional-dependencies] dev:")
        for dep in sorted(test_missing):
            print(f"   - {dep}")
    
    print("\n" + "=" * 60)
    print("DEPENDENCY USAGE MAP")
    print("=" * 60)
    for dep in sorted(usage_map.keys()):
        imports = usage_map[dep]
        print(f"  {dep}: imported as {', '.join(imports)}")
    
    # Return exit code based on findings (only critical missing deps)
    if missing_deps:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
