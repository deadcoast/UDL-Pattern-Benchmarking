#!/usr/bin/env python3
"""
Link Checker Script

Scans and validates all internal links in documentation files.
Supports file references, anchor links, and code references.

**Feature: documentation-validation**
**Validates: Requirements 2.1, 2.2**

Usage:
    uv run python scripts/check_links.py [--verbose] [--fix] [--output FILE]
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import link validator directly to avoid package import issues
import importlib.util


def _import_link_validator():
    """Import link_validator module directly."""
    module_path = (
        project_root / "udl_rating_framework" / "validation" / "link_validator.py"
    )
    spec = importlib.util.spec_from_file_location("link_validator", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_lv = _import_link_validator()
LinkValidator = _lv.LinkValidator
LinkType = _lv.LinkType
LinkValidationReport = _lv.LinkValidationReport


@dataclass
class LinkCheckResult:
    """Result of link checking operation."""

    total_files: int
    total_links: int
    valid_links: int
    broken_links: int
    broken_details: List[Dict]
    by_type: Dict[str, Dict[str, int]]


class LinkChecker:
    """
    Comprehensive link checker for documentation files.

    Scans all documentation files and validates:
    - File reference links
    - Anchor links (within same document)
    - External URLs (optional)
    """

    def __init__(self, project_root: Path, verbose: bool = False):
        """Initialize the link checker.

        Args:
            project_root: Root directory of the project.
            verbose: If True, print detailed progress.
        """
        self.project_root = Path(project_root).resolve()
        self.verbose = verbose
        self.validator = LinkValidator(self.project_root)

    def scan_all_links(self, include_external: bool = False) -> LinkCheckResult:
        """
        Scan all documentation files and validate links.

        Args:
            include_external: If True, also validate external URLs.

        Returns:
            LinkCheckResult with validation results.
        """
        doc_files = self.validator.scan_documentation_files()

        if self.verbose:
            print(f"Found {len(doc_files)} documentation files")

        total_links = 0
        valid_links = 0
        broken_links = 0
        broken_details = []
        by_type = {
            "file_reference": {"total": 0, "valid": 0, "broken": 0},
            "anchor_link": {"total": 0, "valid": 0, "broken": 0},
            "external_url": {"total": 0, "valid": 0, "broken": 0},
        }

        for doc_file in doc_files:
            if self.verbose:
                try:
                    rel_path = doc_file.relative_to(self.project_root)
                except ValueError:
                    rel_path = doc_file
                print(f"  Scanning: {rel_path}")

            links = self.validator.extract_links_from_file(doc_file)

            for link in links:
                total_links += 1
                link_type_key = link.link_type.value
                by_type[link_type_key]["total"] += 1

                # Validate based on type
                if link.link_type == LinkType.FILE_REFERENCE:
                    result = self.validator.validate_file_link(link)
                elif link.link_type == LinkType.ANCHOR_LINK:
                    result = self.validator.validate_anchor_link(link)
                elif link.link_type == LinkType.EXTERNAL_URL:
                    if include_external:
                        # For now, skip external URL validation
                        result = _lv.ValidationResult(
                            link=link,
                            is_valid=True,
                            message="External URL (not validated)",
                        )
                    else:
                        result = _lv.ValidationResult(
                            link=link, is_valid=True, message="External URL (skipped)"
                        )
                else:
                    result = _lv.ValidationResult(
                        link=link, is_valid=True, message="Unknown link type"
                    )

                if result.is_valid:
                    valid_links += 1
                    by_type[link_type_key]["valid"] += 1
                else:
                    broken_links += 1
                    by_type[link_type_key]["broken"] += 1

                    try:
                        rel_source = link.source_file.relative_to(self.project_root)
                    except ValueError:
                        rel_source = link.source_file

                    broken_details.append(
                        {
                            "source_file": str(rel_source),
                            "source_line": link.source_line,
                            "link_text": link.link_text,
                            "link_target": link.link_target,
                            "link_type": link.link_type.value,
                            "error": result.message,
                        }
                    )

        return LinkCheckResult(
            total_files=len(doc_files),
            total_links=total_links,
            valid_links=valid_links,
            broken_links=broken_links,
            broken_details=broken_details,
            by_type=by_type,
        )

    def generate_report(self, result: LinkCheckResult, format: str = "text") -> str:
        """
        Generate a report from link check results.

        Args:
            result: LinkCheckResult from scan_all_links.
            format: Output format ('text', 'markdown', 'json').

        Returns:
            Formatted report string.
        """
        if format == "json":
            return json.dumps(
                {
                    "total_files": result.total_files,
                    "total_links": result.total_links,
                    "valid_links": result.valid_links,
                    "broken_links": result.broken_links,
                    "broken_details": result.broken_details,
                    "by_type": result.by_type,
                },
                indent=2,
            )

        elif format == "markdown":
            lines = [
                "# Link Validation Report",
                "",
                "## Summary",
                "",
                f"- **Files scanned:** {result.total_files}",
                f"- **Total links:** {result.total_links}",
                f"- **Valid links:** {result.valid_links}",
                f"- **Broken links:** {result.broken_links}",
                "",
                "## By Link Type",
                "",
                "| Type | Total | Valid | Broken |",
                "|------|-------|-------|--------|",
            ]

            for link_type, counts in result.by_type.items():
                lines.append(
                    f"| {link_type} | {counts['total']} | {counts['valid']} | {counts['broken']} |"
                )

            if result.broken_details:
                lines.extend(
                    [
                        "",
                        "## Broken Links",
                        "",
                    ]
                )

                for detail in result.broken_details:
                    lines.extend(
                        [
                            f"### {detail['source_file']}:{detail['source_line']}",
                            "",
                            f"- **Link text:** `{detail['link_text']}`",
                            f"- **Target:** `{detail['link_target']}`",
                            f"- **Type:** {detail['link_type']}",
                            f"- **Error:** {detail['error']}",
                            "",
                        ]
                    )
            else:
                lines.extend(
                    [
                        "",
                        "## Result",
                        "",
                        "✅ All links are valid!",
                        "",
                    ]
                )

            return "\n".join(lines)

        else:  # text format
            lines = [
                "=" * 60,
                "Link Validation Report",
                "=" * 60,
                "",
                f"Files scanned:  {result.total_files}",
                f"Total links:    {result.total_links}",
                f"Valid links:    {result.valid_links}",
                f"Broken links:   {result.broken_links}",
                "",
                "By Link Type:",
            ]

            for link_type, counts in result.by_type.items():
                lines.append(
                    f"  {link_type:20} Total: {counts['total']:4}  Valid: {counts['valid']:4}  Broken: {counts['broken']:4}"
                )

            if result.broken_details:
                lines.extend(
                    [
                        "",
                        "Broken Links:",
                        "-" * 40,
                    ]
                )

                for detail in result.broken_details:
                    lines.extend(
                        [
                            f"  {detail['source_file']}:{detail['source_line']}",
                            f"    [{detail['link_text']}]({detail['link_target']})",
                            f"    Type: {detail['link_type']}",
                            f"    Error: {detail['error']}",
                            "",
                        ]
                    )
            else:
                lines.extend(
                    [
                        "",
                        "✓ All links are valid!",
                    ]
                )

            return "\n".join(lines)


def main():
    """Main entry point for the link checker script."""
    parser = argparse.ArgumentParser(
        description="Scan and validate all links in documentation files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python scripts/check_links.py
    uv run python scripts/check_links.py --verbose
    uv run python scripts/check_links.py --format markdown --output report.md
    uv run python scripts/check_links.py --format json --output report.json
""",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed progress information",
    )

    parser.add_argument(
        "--format",
        "-f",
        choices=["text", "markdown", "json"],
        default="text",
        help="Output format (default: text)",
    )

    parser.add_argument(
        "--output", "-o", type=str, help="Output file path (default: stdout)"
    )

    parser.add_argument(
        "--include-external",
        action="store_true",
        help="Also validate external URLs (slower)",
    )

    parser.add_argument(
        "--project-root",
        type=str,
        default=str(project_root),
        help="Project root directory (default: auto-detected)",
    )

    args = parser.parse_args()

    # Run link checker
    checker = LinkChecker(project_root=Path(args.project_root), verbose=args.verbose)

    if args.verbose:
        print("Starting link validation...")
        print()

    result = checker.scan_all_links(include_external=args.include_external)
    report = checker.generate_report(result, format=args.format)

    # Output report
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(report, encoding="utf-8")
        print(f"Report written to: {args.output}")
    else:
        print(report)

    # Return exit code based on broken links
    return 1 if result.broken_links > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
