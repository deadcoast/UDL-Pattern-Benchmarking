"""
Input validation module for UDL Rating Framework.

This module provides functionality for:
- UDL file format validation
- Content structure validation
- Grammar specification validation
"""

import re
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum


# Configure logging
logger = logging.getLogger(__name__)


class UDLFormat(Enum):
    """Supported UDL format types."""

    EBNF = "ebnf"
    BNF = "bnf"
    GRAMMAR = "grammar"
    DSL = "dsl"
    UDL = "udl"
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass
class ValidationResult:
    """Result of input validation."""

    is_valid: bool
    format_type: UDLFormat
    errors: List[str]
    warnings: List[str]
    line_count: int
    character_count: int
    detected_constructs: Set[str]


class ValidationError(Exception):
    """Exception raised during input validation."""

    pass


class InputValidator:
    """
    Input validation engine for UDL files.

    Validates file format, content structure, and grammar specifications
    according to formal language theory principles.
    """

    # Common grammar patterns for format detection
    EBNF_PATTERNS = [
        r"::=",  # EBNF production rule
        r"\[.*\]",  # Optional elements
        r"\{.*\}",  # Repetition
        r"\|",  # Alternation
    ]

    BNF_PATTERNS = [
        r"<[^>]+>",  # Non-terminals in angle brackets
        r"::=",  # Production rule
        r"\|",  # Alternation
    ]

    GRAMMAR_PATTERNS = [
        r"grammar\s+\w+",  # Grammar declaration
        r"rule\s+\w+",  # Rule declaration
        r"token\s+\w+",  # Token declaration
    ]

    # Common language constructs
    LANGUAGE_CONSTRUCTS = {
        "keywords",
        "operators",
        "literals",
        "identifiers",
        "expressions",
        "statements",
        "declarations",
        "types",
        "functions",
        "variables",
        "comments",
        "whitespace",
    }

    def __init__(
        self,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB default
        encoding: str = "utf-8",
    ):
        """
        Initialize input validator.

        Args:
            max_file_size: Maximum allowed file size in bytes
            encoding: Text encoding to use for file reading
        """
        self.max_file_size = max_file_size
        self.encoding = encoding

    def validate_file(self, file_path: Path) -> ValidationResult:
        """
        Validate a UDL file.

        Args:
            file_path: Path to UDL file

        Returns:
            ValidationResult with validation details

        Raises:
            ValidationError: If file cannot be processed
        """
        errors = []
        warnings = []

        # Check file existence and accessibility
        if not file_path.exists():
            raise ValidationError(f"File does not exist: {file_path}")

        if not file_path.is_file():
            raise ValidationError(f"Path is not a file: {file_path}")

        # Check file size
        try:
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                errors.append(
                    f"File size ({file_size} bytes) exceeds maximum ({self.max_file_size} bytes)"
                )
        except OSError as e:
            raise ValidationError(f"Cannot access file stats: {e}")

        # Read and validate content
        try:
            content = self._read_file_content(file_path)
        except Exception as e:
            raise ValidationError(f"Cannot read file content: {e}")

        # Perform validation
        format_type = self._detect_format(content, file_path)
        content_errors, content_warnings = self._validate_content(content, format_type)
        detected_constructs = self._detect_constructs(content)

        errors.extend(content_errors)
        warnings.extend(content_warnings)

        # Calculate metrics
        line_count = len(content.splitlines())
        character_count = len(content)

        # Log validation summary
        logger.info(
            f"Validated {file_path}: {format_type.value} format, "
            f"{line_count} lines, {len(errors)} errors, {len(warnings)} warnings"
        )

        return ValidationResult(
            is_valid=len(errors) == 0,
            format_type=format_type,
            errors=errors,
            warnings=warnings,
            line_count=line_count,
            character_count=character_count,
            detected_constructs=detected_constructs,
        )

    def validate_content(
        self, content: str, format_hint: Optional[UDLFormat] = None
    ) -> ValidationResult:
        """
        Validate UDL content string.

        Args:
            content: UDL content to validate
            format_hint: Optional format type hint

        Returns:
            ValidationResult with validation details
        """
        errors = []
        warnings = []

        # Detect format if not provided
        format_type = format_hint or self._detect_format(content)

        # Validate content structure
        content_errors, content_warnings = self._validate_content(content, format_type)
        detected_constructs = self._detect_constructs(content)

        errors.extend(content_errors)
        warnings.extend(content_warnings)

        # Calculate metrics
        line_count = len(content.splitlines())
        character_count = len(content)

        return ValidationResult(
            is_valid=len(errors) == 0,
            format_type=format_type,
            errors=errors,
            warnings=warnings,
            line_count=line_count,
            character_count=character_count,
            detected_constructs=detected_constructs,
        )

    def _read_file_content(self, file_path: Path) -> str:
        """
        Read file content with proper encoding handling.

        Args:
            file_path: Path to file

        Returns:
            File content as string

        Raises:
            ValidationError: If file cannot be read
        """
        try:
            with open(file_path, "r", encoding=self.encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encodings
            for encoding in ["latin-1", "cp1252", "utf-16"]:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        content = f.read()
                        logger.warning(
                            f"File {file_path} read with {encoding} encoding"
                        )
                        return content
                except UnicodeDecodeError:
                    continue
            raise ValidationError("Cannot decode file with any supported encoding")
        except Exception as e:
            raise ValidationError(f"Error reading file: {e}")

    def _detect_format(
        self, content: str, file_path: Optional[Path] = None
    ) -> UDLFormat:
        """
        Detect UDL format from content and file extension.

        Args:
            content: File content
            file_path: Optional file path for extension hint

        Returns:
            Detected UDL format
        """
        # Check file extension first
        if file_path:
            extension = file_path.suffix.lower()
            if extension == ".ebnf":
                return UDLFormat.EBNF
            elif extension == ".grammar":
                return UDLFormat.GRAMMAR
            elif extension == ".dsl":
                return UDLFormat.DSL
            elif extension == ".udl":
                return UDLFormat.UDL
            elif extension == ".txt":
                # Need to analyze content for .txt files
                pass

        # Analyze content patterns
        content_lower = content.lower()

        # Check for EBNF patterns
        ebnf_score = sum(
            1 for pattern in self.EBNF_PATTERNS if re.search(pattern, content)
        )

        # Check for BNF patterns
        bnf_score = sum(
            1 for pattern in self.BNF_PATTERNS if re.search(pattern, content)
        )

        # Check for grammar patterns
        grammar_score = sum(
            1 for pattern in self.GRAMMAR_PATTERNS if re.search(pattern, content_lower)
        )

        # Determine format based on scores
        if ebnf_score > 0 and ebnf_score >= bnf_score:
            return UDLFormat.EBNF
        elif bnf_score > 0:
            return UDLFormat.BNF
        elif grammar_score > 0:
            return UDLFormat.GRAMMAR
        elif file_path and file_path.suffix.lower() in [".dsl", ".udl"]:
            return (
                UDLFormat.DSL if file_path.suffix.lower() == ".dsl" else UDLFormat.UDL
            )
        else:
            return UDLFormat.TEXT

    def _validate_content(
        self, content: str, format_type: UDLFormat
    ) -> Tuple[List[str], List[str]]:
        """
        Validate content structure based on format type.

        Args:
            content: Content to validate
            format_type: Detected format type

        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []

        # Basic content checks
        if not content.strip():
            errors.append("File is empty or contains only whitespace")
            return errors, warnings

        lines = content.splitlines()

        # Check for extremely long lines
        for i, line in enumerate(lines, 1):
            if len(line) > 1000:
                warnings.append(f"Line {i} is very long ({len(line)} characters)")

        # Format-specific validation
        if format_type in [UDLFormat.EBNF, UDLFormat.BNF]:
            self._validate_grammar_syntax(content, format_type, errors, warnings)
        elif format_type == UDLFormat.GRAMMAR:
            self._validate_grammar_declaration(content, errors, warnings)

        # Check for common issues
        if "\t" in content and "    " in content:
            warnings.append("Mixed tab and space indentation detected")

        return errors, warnings

    def _validate_grammar_syntax(
        self,
        content: str,
        format_type: UDLFormat,
        errors: List[str],
        warnings: List[str],
    ) -> None:
        """
        Validate grammar syntax for EBNF/BNF formats.

        Args:
            content: Content to validate
            format_type: Grammar format type
            errors: List to append errors to
            warnings: List to append warnings to
        """
        lines = content.splitlines()

        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue  # Skip empty lines and comments

            # Check for production rules
            if "::=" in line:
                # Validate production rule structure
                parts = line.split("::=")
                if len(parts) != 2:
                    errors.append(f"Line {i}: Invalid production rule syntax")
                    continue

                lhs = parts[0].strip()
                rhs = parts[1].strip()

                if not lhs:
                    errors.append(f"Line {i}: Empty left-hand side in production rule")

                if not rhs:
                    errors.append(f"Line {i}: Empty right-hand side in production rule")

                # Check for balanced brackets in EBNF
                if format_type == UDLFormat.EBNF:
                    if not self._check_balanced_brackets(rhs):
                        errors.append(
                            f"Line {i}: Unbalanced brackets in EBNF expression"
                        )

    def _validate_grammar_declaration(
        self, content: str, errors: List[str], warnings: List[str]
    ) -> None:
        """
        Validate grammar declaration format.

        Args:
            content: Content to validate
            errors: List to append errors to
            warnings: List to append warnings to
        """
        # Check for grammar declaration
        if not re.search(r"grammar\s+\w+", content, re.IGNORECASE):
            warnings.append("No grammar declaration found")

        # Check for at least one rule
        if not re.search(r"rule\s+\w+", content, re.IGNORECASE):
            warnings.append("No rules found in grammar")

    def _check_balanced_brackets(self, text: str) -> bool:
        """
        Check if brackets are balanced in EBNF expression.

        Args:
            text: Text to check

        Returns:
            True if brackets are balanced
        """
        stack = []
        pairs = {"(": ")", "[": "]", "{": "}"}

        for char in text:
            if char in pairs:
                stack.append(char)
            elif char in pairs.values():
                if not stack:
                    return False
                if pairs[stack.pop()] != char:
                    return False

        return len(stack) == 0

    def _detect_constructs(self, content: str) -> Set[str]:
        """
        Detect language constructs in content.

        Args:
            content: Content to analyze

        Returns:
            Set of detected construct types
        """
        detected = set()
        content_lower = content.lower()

        # Simple keyword-based detection
        construct_keywords = {
            "keywords": ["keyword", "reserved", "if", "else", "while", "for"],
            "operators": ["operator", "+", "-", "*", "/", "=", "==", "!="],
            "literals": ["literal", "string", "number", "boolean", "true", "false"],
            "identifiers": ["identifier", "name", "variable"],
            "expressions": ["expression", "expr", "term", "factor"],
            "statements": ["statement", "stmt", "block"],
            "declarations": ["declaration", "decl", "define"],
            "types": ["type", "int", "string", "bool", "float"],
            "functions": ["function", "func", "method", "procedure"],
            "comments": ["comment", "//", "/*", "#"],
        }

        for construct, keywords in construct_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                detected.add(construct)

        return detected
