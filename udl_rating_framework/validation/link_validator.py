"""
Link Validator Module

Validates internal file reference links in documentation files.
Supports markdown link syntax: [text](path/to/file)

**Feature: documentation-validation, Property 2: Link Target Resolution**
**Validates: Requirements 2.2, 2.3, 2.4**
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class LinkType(Enum):
    """Types of links found in documentation."""
    FILE_REFERENCE = "file_reference"
    ANCHOR_LINK = "anchor_link"
    EXTERNAL_URL = "external_url"
    CODE_REFERENCE = "code_reference"


@dataclass
class Link:
    """Represents a link found in documentation."""
    source_file: Path
    source_line: int
    link_text: str
    link_target: str
    link_type: LinkType
    
    def __hash__(self):
        return hash((str(self.source_file), self.source_line, self.link_target))


@dataclass
class ValidationResult:
    """Result of validating a single link."""
    link: Link
    is_valid: bool
    message: str
    resolved_path: Optional[Path] = None


@dataclass
class LinkValidationReport:
    """Complete report of link validation."""
    total_links: int = 0
    valid_links: int = 0
    broken_links: int = 0
    results: List[ValidationResult] = field(default_factory=list)
    broken_link_details: List[ValidationResult] = field(default_factory=list)
    
    def add_result(self, result: ValidationResult):
        """Add a validation result to the report."""
        self.results.append(result)
        self.total_links += 1
        if result.is_valid:
            self.valid_links += 1
        else:
            self.broken_links += 1
            self.broken_link_details.append(result)


class LinkValidator:
    """
    Validates internal and external links in documentation.
    
    Link types:
    - File references: [text](path/to/file.md)
    - Anchor links: [text](#section-heading)
    - External URLs: [text](https://example.com)
    - Code references: `module.Class.method`
    """
    
    # Regex pattern for markdown links: [text](target)
    MARKDOWN_LINK_PATTERN = re.compile(r'\[([^\]]*)\]\(([^)]+)\)')
    
    # Pattern for external URLs
    EXTERNAL_URL_PATTERN = re.compile(r'^https?://')
    
    # Pattern for anchor links
    ANCHOR_PATTERN = re.compile(r'^#')
    
    def __init__(self, project_root: Path):
        """Initialize validator with project root."""
        self.project_root = Path(project_root).resolve()
    
    def classify_link(self, target: str) -> LinkType:
        """Classify a link target by type."""
        if self.EXTERNAL_URL_PATTERN.match(target):
            return LinkType.EXTERNAL_URL
        elif self.ANCHOR_PATTERN.match(target):
            return LinkType.ANCHOR_LINK
        else:
            return LinkType.FILE_REFERENCE
    
    def extract_links_from_file(self, file_path: Path) -> List[Link]:
        """Extract all links from a documentation file."""
        links = []
        try:
            content = file_path.read_text(encoding='utf-8')
        except (IOError, UnicodeDecodeError) as e:
            return links
        
        for line_num, line in enumerate(content.split('\n'), start=1):
            for match in self.MARKDOWN_LINK_PATTERN.finditer(line):
                link_text = match.group(1)
                link_target = match.group(2)
                link_type = self.classify_link(link_target)
                
                links.append(Link(
                    source_file=file_path,
                    source_line=line_num,
                    link_text=link_text,
                    link_target=link_target,
                    link_type=link_type
                ))
        
        return links
    
    def resolve_file_path(self, source_file: Path, target: str) -> Tuple[bool, Optional[Path], str]:
        """
        Resolve a file reference link to an actual file path.
        
        Returns:
            Tuple of (is_valid, resolved_path, message)
        """
        # Remove anchor part if present (e.g., "file.md#section" -> "file.md")
        target_without_anchor = target.split('#')[0]
        
        if not target_without_anchor:
            # Pure anchor link, not a file reference
            return True, None, "Anchor-only link"
        
        # Handle absolute paths (starting with /)
        if target_without_anchor.startswith('/'):
            resolved = self.project_root / target_without_anchor.lstrip('/')
        else:
            # Relative path from source file's directory
            resolved = (source_file.parent / target_without_anchor).resolve()
        
        # Check if file exists
        if resolved.exists():
            return True, resolved, "File exists"
        
        # Try common variations
        variations = [
            resolved,
            resolved.with_suffix('.md'),
            self.project_root / target_without_anchor,
            self.project_root / target_without_anchor.lstrip('./'),
        ]
        
        for variation in variations:
            if variation.exists():
                return True, variation, f"File exists at {variation}"
        
        return False, resolved, f"File not found: {resolved}"
    
    def validate_file_link(self, link: Link) -> ValidationResult:
        """Verify file reference resolves to existing file."""
        is_valid, resolved_path, message = self.resolve_file_path(
            link.source_file, link.link_target
        )
        
        return ValidationResult(
            link=link,
            is_valid=is_valid,
            message=message,
            resolved_path=resolved_path
        )
    
    def validate_anchor_link(self, link: Link) -> ValidationResult:
        """Verify anchor points to valid heading in the same document."""
        # For now, we'll mark anchor links as valid (task 4.3 handles these)
        return ValidationResult(
            link=link,
            is_valid=True,
            message="Anchor link (validation deferred to task 4.3)"
        )
    
    def validate_link(self, link: Link) -> ValidationResult:
        """Validate a single link based on its type."""
        if link.link_type == LinkType.EXTERNAL_URL:
            # Skip external URL validation for now
            return ValidationResult(
                link=link,
                is_valid=True,
                message="External URL (not validated)"
            )
        elif link.link_type == LinkType.ANCHOR_LINK:
            return self.validate_anchor_link(link)
        elif link.link_type == LinkType.FILE_REFERENCE:
            return self.validate_file_link(link)
        else:
            return ValidationResult(
                link=link,
                is_valid=True,
                message="Unknown link type"
            )
    
    def scan_documentation_files(self, include_specs: bool = False) -> List[Path]:
        """Find all documentation files in the project.
        
        Args:
            include_specs: If True, include .kiro/specs files (default False)
        """
        doc_files = []
        
        # Scan for markdown files
        for pattern in ['**/*.md', '**/*.rst']:
            doc_files.extend(self.project_root.glob(pattern))
        
        # Filter out common non-documentation directories
        excluded_dirs = {'.venv', 'node_modules', '.git', '__pycache__', 'dist', 'build', '.hypothesis'}
        
        # Optionally exclude spec files (they contain example syntax, not real links)
        if not include_specs:
            excluded_dirs.add('.kiro')
        
        filtered_files = []
        for f in doc_files:
            # Check if any parent directory is in excluded list
            if not any(excluded in f.parts for excluded in excluded_dirs):
                filtered_files.append(f)
        
        return filtered_files
    
    def find_broken_links(self) -> LinkValidationReport:
        """Scan all docs and return validation report."""
        report = LinkValidationReport()
        
        doc_files = self.scan_documentation_files()
        
        for doc_file in doc_files:
            links = self.extract_links_from_file(doc_file)
            for link in links:
                result = self.validate_link(link)
                report.add_result(result)
        
        return report
    
    def validate_file_references_only(self) -> LinkValidationReport:
        """Validate only file reference links (not anchors or external URLs)."""
        report = LinkValidationReport()
        
        doc_files = self.scan_documentation_files()
        
        for doc_file in doc_files:
            links = self.extract_links_from_file(doc_file)
            for link in links:
                if link.link_type == LinkType.FILE_REFERENCE:
                    result = self.validate_file_link(link)
                    report.add_result(result)
        
        return report
    
    def generate_report_markdown(self, report: LinkValidationReport) -> str:
        """Generate a markdown report of validation results."""
        lines = [
            "# Link Validation Report",
            "",
            "## Summary",
            "",
            f"- Total links validated: {report.total_links}",
            f"- Valid links: {report.valid_links}",
            f"- Broken links: {report.broken_links}",
            "",
        ]
        
        if report.broken_link_details:
            lines.extend([
                "## Broken Links",
                "",
            ])
            
            for result in report.broken_link_details:
                lines.extend([
                    f"### {result.link.source_file}:{result.link.source_line}",
                    "",
                    f"- Link text: `{result.link.link_text}`",
                    f"- Target: `{result.link.link_target}`",
                    f"- Error: {result.message}",
                    "",
                ])
        else:
            lines.extend([
                "## Result",
                "",
                "All file reference links are valid! ✓",
                "",
            ])
        
        return '\n'.join(lines)


def validate_project_links(project_root: str = ".") -> LinkValidationReport:
    """
    Convenience function to validate all links in a project.
    
    Args:
        project_root: Path to project root directory
        
    Returns:
        LinkValidationReport with validation results
    """
    validator = LinkValidator(Path(project_root))
    return validator.validate_file_references_only()


if __name__ == "__main__":
    from pathlib import Path
    
    # Run validation on current project
    validator = LinkValidator(Path('.'))
    doc_files = validator.scan_documentation_files()
    
    print("Documentation files scanned:")
    for f in sorted(doc_files):
        try:
            rel_path = f.relative_to(Path('.').resolve())
            print(f"  {rel_path}")
        except ValueError:
            print(f"  {f}")
    
    print()
    print("File reference links found and validated:")
    
    all_results = []
    for doc_file in doc_files:
        links = validator.extract_links_from_file(doc_file)
        file_links = [l for l in links if l.link_type == LinkType.FILE_REFERENCE]
        if file_links:
            try:
                rel_path = doc_file.relative_to(Path('.').resolve())
            except ValueError:
                rel_path = doc_file
            print(f"\n{rel_path}:")
            for link in file_links:
                result = validator.validate_file_link(link)
                all_results.append(result)
                status = '✓' if result.is_valid else '✗'
                print(f"  Line {link.source_line}: [{link.link_text}]({link.link_target}) {status}")
    
    # Summary
    valid_count = sum(1 for r in all_results if r.is_valid)
    broken_count = sum(1 for r in all_results if not r.is_valid)
    
    print(f"\n--- Summary ---")
    print(f"Total file reference links: {len(all_results)}")
    print(f"Valid: {valid_count}")
    print(f"Broken: {broken_count}")
    
    if broken_count > 0:
        print("\nBroken links details:")
        for result in all_results:
            if not result.is_valid:
                print(f"  {result.link.source_file}:{result.link.source_line}")
                print(f"    [{result.link.link_text}]({result.link.link_target})")
                print(f"    Error: {result.message}")
