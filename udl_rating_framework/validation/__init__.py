"""
Validation module for UDL Rating Framework.

This module provides comprehensive quality assurance and validation capabilities including:
- Formal verification of metric properties
- Benchmarking against academic datasets
- Link validation for documentation
- API validation for documentation accuracy
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

__all__ = [
    "FormalVerifier",
    "DatasetBenchmark",
    "LinkValidator",
    "APIValidator",
    "APIExtractor",
    "APIValidationReport",
]
