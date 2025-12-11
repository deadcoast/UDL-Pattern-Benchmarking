"""
File discovery module for UDL Rating Framework.

This module provides functionality for:
- Recursive directory traversal
- File extension filtering
- Error handling for unreadable files
"""

import os
import logging
from pathlib import Path
from typing import List, Set, Generator, Optional
from dataclasses import dataclass


# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class FileDiscoveryResult:
    """Result of file discovery operation."""

    discovered_files: List[Path]
    errors: List[str]
    total_directories_scanned: int
    total_files_examined: int


class FileDiscoveryError(Exception):
    """Exception raised during file discovery operations."""

    pass


class FileDiscovery:
    """
    File discovery engine for UDL files.

    Supports recursive directory traversal with configurable file extension filtering.
    Implements graceful error handling for unreadable files and directories.
    """

    # Supported UDL file extensions as specified in requirements
    DEFAULT_EXTENSIONS = {".udl", ".dsl", ".grammar", ".ebnf", ".txt"}

    def __init__(
        self,
        supported_extensions: Optional[Set[str]] = None,
        case_sensitive: bool = False,
    ):
        """
        Initialize file discovery engine.

        Args:
            supported_extensions: Set of file extensions to discover (default: UDL extensions)
            case_sensitive: Whether extension matching is case-sensitive
        """
        self.supported_extensions = supported_extensions or self.DEFAULT_EXTENSIONS
        self.case_sensitive = case_sensitive

        # Normalize extensions for consistent matching
        if not case_sensitive:
            self.supported_extensions = {
                ext.lower() for ext in self.supported_extensions
            }

    def discover_files(self, directory_path: str) -> FileDiscoveryResult:
        """
        Recursively discover UDL files in directory.

        Args:
            directory_path: Path to directory to scan

        Returns:
            FileDiscoveryResult with discovered files and any errors

        Raises:
            FileDiscoveryError: If directory path is invalid or inaccessible
        """
        path = Path(directory_path)

        # Validate directory path
        if not path.exists():
            raise FileDiscoveryError(f"Directory does not exist: {directory_path}")

        if not path.is_dir():
            raise FileDiscoveryError(f"Path is not a directory: {directory_path}")

        discovered_files = []
        errors = []
        directories_scanned = 0
        files_examined = 0

        try:
            for file_path in self._walk_directory(path):
                files_examined += 1

                try:
                    if self._is_supported_file(file_path):
                        # Verify file is readable
                        if self._is_readable(file_path):
                            discovered_files.append(file_path)
                        else:
                            error_msg = f"File is not readable: {file_path}"
                            errors.append(error_msg)
                            logger.warning(error_msg)

                except Exception as e:
                    error_msg = f"Error processing file {file_path}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                    # Continue processing other files
                    continue

            # Count directories scanned
            for _ in self._walk_directories(path):
                directories_scanned += 1

        except Exception as e:
            error_msg = f"Error during directory traversal: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)

        result = FileDiscoveryResult(
            discovered_files=discovered_files,
            errors=errors,
            total_directories_scanned=directories_scanned,
            total_files_examined=files_examined,
        )

        # Log summary
        logger.info(
            f"File discovery completed: {len(discovered_files)} files found, "
            f"{len(errors)} errors, {directories_scanned} directories scanned"
        )

        return result

    def _walk_directory(self, directory: Path) -> Generator[Path, None, None]:
        """
        Recursively walk directory and yield all files.

        Args:
            directory: Directory to walk

        Yields:
            Path objects for all files found
        """
        try:
            for item in directory.iterdir():
                try:
                    if item.is_file():
                        yield item
                    elif item.is_dir():
                        # Recursively process subdirectories
                        yield from self._walk_directory(item)
                except (PermissionError, OSError) as e:
                    error_msg = f"Cannot access {item}: {str(e)}"
                    logger.warning(error_msg)
                    # Continue with other items
                    continue

        except (PermissionError, OSError) as e:
            error_msg = f"Cannot access directory {directory}: {str(e)}"
            logger.warning(error_msg)
            # Don't re-raise, just log and continue

    def _walk_directories(self, directory: Path) -> Generator[Path, None, None]:
        """
        Walk directory structure and yield directory paths for counting.

        Args:
            directory: Directory to walk

        Yields:
            Path objects for all directories found
        """
        yield directory

        try:
            for item in directory.iterdir():
                try:
                    if item.is_dir():
                        yield from self._walk_directories(item)
                except (PermissionError, OSError):
                    # Skip inaccessible directories
                    continue
        except (PermissionError, OSError):
            # Skip if can't read directory
            pass

    def _is_supported_file(self, file_path: Path) -> bool:
        """
        Check if file has supported extension.

        Args:
            file_path: Path to file

        Returns:
            True if file extension is supported
        """
        extension = file_path.suffix

        if not self.case_sensitive:
            extension = extension.lower()

        return extension in self.supported_extensions

    def _is_readable(self, file_path: Path) -> bool:
        """
        Check if file is readable.

        Args:
            file_path: Path to file

        Returns:
            True if file can be read
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                # Try to read first few bytes to verify readability
                f.read(1)
            return True
        except (PermissionError, OSError, UnicodeDecodeError):
            return False

    def get_supported_extensions(self) -> Set[str]:
        """
        Get set of supported file extensions.

        Returns:
            Set of supported extensions
        """
        return self.supported_extensions.copy()

    def add_extension(self, extension: str) -> None:
        """
        Add a new supported file extension.

        Args:
            extension: File extension to add (e.g., '.mydsl')
        """
        if not extension.startswith("."):
            extension = "." + extension

        if not self.case_sensitive:
            extension = extension.lower()

        self.supported_extensions.add(extension)

    def remove_extension(self, extension: str) -> None:
        """
        Remove a supported file extension.

        Args:
            extension: File extension to remove
        """
        if not extension.startswith("."):
            extension = "." + extension

        if not self.case_sensitive:
            extension = extension.lower()

        self.supported_extensions.discard(extension)
