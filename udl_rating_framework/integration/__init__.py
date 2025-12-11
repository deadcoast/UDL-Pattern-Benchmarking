"""
Integration and workflow features for UDL Rating Framework.

This module provides integration capabilities including:
- Git hooks for automatic UDL quality checking
- CI/CD pipeline integration
- IDE plugins and LSP support
- Batch processing workflows
"""

from .batch_processor import BatchProcessor
from .cicd import CICDIntegration
from .git_hooks import GitHookManager
from .ide_plugin import IDEPluginManager
from .lsp_server import UDLLanguageServer

__all__ = [
    "GitHookManager",
    "CICDIntegration",
    "UDLLanguageServer",
    "BatchProcessor",
    "IDEPluginManager",
]
