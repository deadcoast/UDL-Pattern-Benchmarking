"""
Integration and workflow features for UDL Rating Framework.

This module provides integration capabilities including:
- Git hooks for automatic UDL quality checking
- CI/CD pipeline integration
- IDE plugins and LSP support
- Batch processing workflows
"""

from .git_hooks import GitHookManager
from .cicd import CICDIntegration
from .lsp_server import UDLLanguageServer
from .batch_processor import BatchProcessor
from .ide_plugin import IDEPluginManager

__all__ = [
    "GitHookManager",
    "CICDIntegration",
    "UDLLanguageServer",
    "BatchProcessor",
    "IDEPluginManager",
]
