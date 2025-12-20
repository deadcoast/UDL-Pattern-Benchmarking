"""
Validation module for UDL Rating Framework.

This module provides comprehensive quality assurance and validation capabilities including:
- Formal verification of metric properties
- Benchmarking against academic datasets
- Link validation for documentation
- API validation for documentation accuracy
- Docstring validation for signature accuracy
"""

# Import only modules that exist
try:
    from .formal_verification import FormalVerifier
except ImportError:
    FormalVerifier = None

try:
    from .dataset_benchmark import DatasetBenchmark
except ImportError:
    DatasetBenchmark = None

try:
    from .link_validator import LinkValidator
except ImportError:
    LinkValidator = None

try:
    from .api_validator import APIExtractor, APIValidationReport, APIValidator
except ImportError:
    APIValidator = None
    APIExtractor = None
    APIValidationReport = None

try:
    from .docstring_validator import (
        DocstringParser,
        DocstringValidationReport,
        DocstringValidator,
        find_undocumented_public_apis,
        validate_docstrings,
    )
except ImportError:
    DocstringValidator = None
    DocstringParser = None
    DocstringValidationReport = None
    validate_docstrings = None
    find_undocumented_public_apis = None

__all__ = [
    "FormalVerifier",
    "DatasetBenchmark",
    "LinkValidator",
    "APIValidator",
    "APIExtractor",
    "APIValidationReport",
    "DocstringValidator",
    "DocstringParser",
    "DocstringValidationReport",
    "validate_docstrings",
    "find_undocumented_public_apis",
]
