"""
Audit Reporter Module

Generates comprehensive audit reports from documentation validation results.
Compiles findings from all validation phases, categorizes by severity,
and generates fix suggestions.

**Feature: documentation-validation, Property 21: Finding Completeness**
**Feature: documentation-validation, Property 22: Fix Suggestion Generation**
**Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5**
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional
import json


class Severity(Enum):
    """Severity levels for audit findings."""
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"


class FindingCategory(Enum):
    """Categories of audit findings."""
    LINK = "link"
    API = "api"
    EXAMPLE = "example"
    STRUCTURE = "structure"
    CONFIG = "config"
    DOCSTRING = "docstring"
    FORMULA = "formula"
    TEST = "test"
    DEPLOYMENT = "deployment"


class ResolutionStatus(Enum):
    """Resolution status for findings."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    WONT_FIX = "wont_fix"


@dataclass
class SourceLocation:
    """Location of a finding in source code or documentation."""
    file_path: str
    line_number: Optional[int] = None
    column: Optional[int] = None
    
    def __str__(self) -> str:
        if self.line_number:
            return f"{self.file_path}:{self.line_number}"
        return self.file_path


@dataclass
class Finding:
    """
    Represents a validation finding.
    
    **Feature: documentation-validation, Property 21: Finding Completeness**
    **Validates: Requirements 10.2, 10.3**
    
    Each finding must include:
    - Severity category
    - File location
    - Line number (when applicable)
    """
    id: str
    category: FindingCategory
    severity: Severity
    source_location: SourceLocation
    description: str
    expected: Optional[str] = None
    actual: Optional[str] = None
    suggestion: Optional[str] = None
    requirement_ref: Optional[str] = None
    status: ResolutionStatus = ResolutionStatus.OPEN
    resolution_notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert finding to dictionary for serialization."""
        return {
            "id": self.id,
            "category": self.category.value,
            "severity": self.severity.value,
            "source_file": self.source_location.file_path,
            "source_line": self.source_location.line_number,
            "description": self.description,
            "expected": self.expected,
            "actual": self.actual,
            "suggestion": self.suggestion,
            "requirement_ref": self.requirement_ref,
            "status": self.status.value,
            "resolution_notes": self.resolution_notes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Finding":
        """Create finding from dictionary."""
        return cls(
            id=data["id"],
            category=FindingCategory(data["category"]),
            severity=Severity(data["severity"]),
            source_location=SourceLocation(
                file_path=data["source_file"],
                line_number=data.get("source_line")
            ),
            description=data["description"],
            expected=data.get("expected"),
            actual=data.get("actual"),
            suggestion=data.get("suggestion"),
            requirement_ref=data.get("requirement_ref"),
            status=ResolutionStatus(data.get("status", "open")),
            resolution_notes=data.get("resolution_notes"),
        )


@dataclass
class AuditReport:
    """
    Complete audit report containing all findings.
    
    **Feature: documentation-validation, Property 21: Finding Completeness**
    **Validates: Requirements 10.1, 10.2, 10.3**
    """
    timestamp: datetime = field(default_factory=datetime.now)
    project_version: str = "1.0.0"
    total_files_scanned: int = 0
    findings: List[Finding] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    coverage_metrics: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def add_finding(self, finding: Finding):
        """Add a finding to the report."""
        self.findings.append(finding)
        self._update_summary()
    
    def _update_summary(self):
        """Update summary counts."""
        self.summary = {
            "total": len(self.findings),
            "critical": sum(1 for f in self.findings if f.severity == Severity.CRITICAL),
            "major": sum(1 for f in self.findings if f.severity == Severity.MAJOR),
            "minor": sum(1 for f in self.findings if f.severity == Severity.MINOR),
            "info": sum(1 for f in self.findings if f.severity == Severity.INFO),
            "open": sum(1 for f in self.findings if f.status == ResolutionStatus.OPEN),
            "resolved": sum(1 for f in self.findings if f.status == ResolutionStatus.RESOLVED),
        }
        
        # Category counts
        for category in FindingCategory:
            self.summary[f"category_{category.value}"] = sum(
                1 for f in self.findings if f.category == category
            )
    
    def get_findings_by_severity(self, severity: Severity) -> List[Finding]:
        """Get all findings of a specific severity."""
        return [f for f in self.findings if f.severity == severity]
    
    def get_findings_by_category(self, category: FindingCategory) -> List[Finding]:
        """Get all findings of a specific category."""
        return [f for f in self.findings if f.category == category]
    
    def get_open_findings(self) -> List[Finding]:
        """Get all unresolved findings."""
        return [f for f in self.findings if f.status == ResolutionStatus.OPEN]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "project_version": self.project_version,
            "total_files_scanned": self.total_files_scanned,
            "summary": self.summary,
            "coverage_metrics": self.coverage_metrics,
            "recommendations": self.recommendations,
            "findings": [f.to_dict() for f in self.findings],
        }


class FixSuggestionGenerator:
    """
    Generates fix suggestions for common issue types.
    
    **Feature: documentation-validation, Property 22: Fix Suggestion Generation**
    **Validates: Requirements 10.4**
    """
    
    # Common fix patterns by category and issue type
    FIX_PATTERNS = {
        FindingCategory.LINK: {
            "broken_file_link": "Update the link target to point to the correct file path: {expected}",
            "broken_anchor": "Update the anchor to match an existing heading, or add the missing heading",
            "orphaned_link": "Remove the link or update it to point to existing content",
        },
        FindingCategory.API: {
            "missing_docstring": "Add a docstring to the function/class describing its purpose, parameters, and return value",
            "signature_mismatch": "Update the documentation to match the actual signature: {actual}",
            "orphaned_doc": "Remove documentation for non-existent API, or restore the API if it was accidentally deleted",
            "undocumented_api": "Add documentation for the public API in the appropriate documentation file",
        },
        FindingCategory.DOCSTRING: {
            "param_mismatch": "Update the docstring parameter '{expected}' to match the actual parameter '{actual}'",
            "missing_param": "Add documentation for parameter '{expected}' in the docstring",
            "extra_param": "Remove documentation for non-existent parameter '{actual}' from the docstring",
            "type_mismatch": "Update the type annotation in the docstring to match the actual type",
        },
        FindingCategory.EXAMPLE: {
            "execution_error": "Fix the code example to execute without errors",
            "outdated_api": "Update the example to use the current API",
            "missing_import": "Add the required import statement to the example",
        },
        FindingCategory.STRUCTURE: {
            "orphaned_file": "Either import the file from an appropriate module or remove it if unused",
            "missing_export": "Add the symbol to __all__ in the package's __init__.py",
            "import_error": "Fix the import statement to reference the correct module path",
        },
        FindingCategory.CONFIG: {
            "unused_dependency": "Remove the unused dependency from pyproject.toml",
            "missing_dependency": "Add the missing dependency to pyproject.toml: {expected}",
            "version_mismatch": "Update version numbers to be consistent across all configuration files",
        },
        FindingCategory.FORMULA: {
            "formula_mismatch": "Update the documentation formula to match the implementation, or fix the implementation",
            "worked_example_error": "Recalculate the worked example with the correct formula",
        },
        FindingCategory.TEST: {
            "invalid_requirement_ref": "Update the requirement reference to point to an existing requirement",
            "missing_test": "Add a test file for the module",
        },
        FindingCategory.DEPLOYMENT: {
            "missing_env_var_doc": "Document the environment variable in the deployment documentation",
            "endpoint_mismatch": "Update the API documentation to match the actual endpoint",
        },
    }
    
    def generate_suggestion(self, finding: Finding) -> str:
        """
        Generate a fix suggestion for a finding.
        
        **Feature: documentation-validation, Property 22: Fix Suggestion Generation**
        **Validates: Requirements 10.4**
        """
        # If finding already has a suggestion, return it
        if finding.suggestion:
            return finding.suggestion
        
        # Try to find a pattern-based suggestion
        category_patterns = self.FIX_PATTERNS.get(finding.category, {})
        
        # Try to match based on description keywords
        description_lower = finding.description.lower()
        
        for issue_type, pattern in category_patterns.items():
            if issue_type.replace("_", " ") in description_lower:
                suggestion = pattern.format(
                    expected=finding.expected or "N/A",
                    actual=finding.actual or "N/A"
                )
                return suggestion
        
        # Default suggestions by category
        default_suggestions = {
            FindingCategory.LINK: "Verify the link target exists and update the path if necessary",
            FindingCategory.API: "Review the API documentation and update to match the implementation",
            FindingCategory.DOCSTRING: "Update the docstring to accurately describe the function",
            FindingCategory.EXAMPLE: "Test the example code and fix any errors",
            FindingCategory.STRUCTURE: "Review the project structure and fix any inconsistencies",
            FindingCategory.CONFIG: "Review configuration files for consistency",
            FindingCategory.FORMULA: "Verify the mathematical formula matches the implementation",
            FindingCategory.TEST: "Review test documentation and fix references",
            FindingCategory.DEPLOYMENT: "Update deployment documentation to match actual configuration",
        }
        
        return default_suggestions.get(
            finding.category,
            "Review the finding and apply appropriate fix"
        )
    
    def generate_all_suggestions(self, findings: List[Finding]) -> List[Finding]:
        """Generate suggestions for all findings that don't have one."""
        for finding in findings:
            if not finding.suggestion:
                finding.suggestion = self.generate_suggestion(finding)
        return findings


class AuditReporter:
    """
    Generates comprehensive audit reports from validation results.
    
    **Feature: documentation-validation, Property 21: Finding Completeness**
    **Feature: documentation-validation, Property 22: Fix Suggestion Generation**
    **Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5**
    """
    
    def __init__(self):
        """Initialize the audit reporter."""
        self.report = AuditReport()
        self.suggestion_generator = FixSuggestionGenerator()
        self._finding_counter = 0
    
    def _generate_finding_id(self, category: FindingCategory) -> str:
        """Generate a unique finding ID."""
        self._finding_counter += 1
        return f"{category.value.upper()}-{self._finding_counter:04d}"
    
    def add_finding(
        self,
        category: FindingCategory,
        severity: Severity,
        file_path: str,
        description: str,
        line_number: Optional[int] = None,
        expected: Optional[str] = None,
        actual: Optional[str] = None,
        requirement_ref: Optional[str] = None,
        status: ResolutionStatus = ResolutionStatus.OPEN,
        resolution_notes: Optional[str] = None,
    ) -> Finding:
        """
        Add a finding to the report.
        
        **Feature: documentation-validation, Property 21: Finding Completeness**
        **Validates: Requirements 10.2, 10.3**
        """
        finding = Finding(
            id=self._generate_finding_id(category),
            category=category,
            severity=severity,
            source_location=SourceLocation(
                file_path=file_path,
                line_number=line_number
            ),
            description=description,
            expected=expected,
            actual=actual,
            requirement_ref=requirement_ref,
            status=status,
            resolution_notes=resolution_notes,
        )
        
        # Generate suggestion
        finding.suggestion = self.suggestion_generator.generate_suggestion(finding)
        
        self.report.add_finding(finding)
        return finding
    
    def categorize_findings(self) -> Dict[Severity, List[Finding]]:
        """Group findings by severity."""
        return {
            Severity.CRITICAL: self.report.get_findings_by_severity(Severity.CRITICAL),
            Severity.MAJOR: self.report.get_findings_by_severity(Severity.MAJOR),
            Severity.MINOR: self.report.get_findings_by_severity(Severity.MINOR),
            Severity.INFO: self.report.get_findings_by_severity(Severity.INFO),
        }
    
    def generate_recommendations(self) -> List[str]:
        """Generate high-level recommendations based on findings."""
        recommendations = []
        
        # Check for critical issues
        critical_count = self.report.summary.get("critical", 0)
        if critical_count > 0:
            recommendations.append(
                f"URGENT: Address {critical_count} critical issue(s) immediately - "
                "these may cause system failures or security vulnerabilities"
            )
        
        # Check for major issues
        major_count = self.report.summary.get("major", 0)
        if major_count > 0:
            recommendations.append(
                f"HIGH PRIORITY: Fix {major_count} major issue(s) - "
                "these affect documentation accuracy and user experience"
            )
        
        # Category-specific recommendations
        link_issues = self.report.summary.get("category_link", 0)
        if link_issues > 0:
            recommendations.append(
                f"Run link validation regularly to catch broken links early ({link_issues} found)"
            )
        
        api_issues = self.report.summary.get("category_api", 0)
        if api_issues > 0:
            recommendations.append(
                f"Consider adding API documentation generation to CI/CD ({api_issues} API issues found)"
            )
        
        docstring_issues = self.report.summary.get("category_docstring", 0)
        if docstring_issues > 0:
            recommendations.append(
                f"Enable docstring linting in pre-commit hooks ({docstring_issues} docstring issues found)"
            )
        
        test_issues = self.report.summary.get("category_test", 0)
        if test_issues > 0:
            recommendations.append(
                f"Review test documentation for accuracy ({test_issues} test documentation issues found)"
            )
        
        # General recommendations
        if len(self.report.findings) > 50:
            recommendations.append(
                "Consider prioritizing fixes by severity and addressing issues incrementally"
            )
        
        if not recommendations:
            recommendations.append(
                "Documentation is in good shape! Continue regular validation to maintain quality."
            )
        
        self.report.recommendations = recommendations
        return recommendations
    
    def generate_markdown_report(self) -> str:
        """
        Generate Markdown summary report.
        
        **Feature: documentation-validation, Property 21: Finding Completeness**
        **Validates: Requirements 10.1, 10.5**
        """
        self.generate_recommendations()
        
        lines = [
            "# Documentation Audit Report",
            "",
            f"**Generated:** {self.report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Project Version:** {self.report.project_version}",
            f"**Files Scanned:** {self.report.total_files_scanned}",
            "",
            "## Executive Summary",
            "",
            f"Total findings: **{self.report.summary.get('total', 0)}**",
            "",
            "| Severity | Count |",
            "|----------|-------|",
            f"| 🔴 Critical | {self.report.summary.get('critical', 0)} |",
            f"| 🟠 Major | {self.report.summary.get('major', 0)} |",
            f"| 🟡 Minor | {self.report.summary.get('minor', 0)} |",
            f"| 🔵 Info | {self.report.summary.get('info', 0)} |",
            "",
            "### Resolution Status",
            "",
            f"- Open: {self.report.summary.get('open', 0)}",
            f"- Resolved: {self.report.summary.get('resolved', 0)}",
            "",
            "## Recommendations",
            "",
        ]
        
        for i, rec in enumerate(self.report.recommendations, 1):
            lines.append(f"{i}. {rec}")
        
        # Add coverage metrics if available
        if self.report.coverage_metrics:
            lines.extend([
                "",
                "## Coverage Metrics",
                "",
                "| Metric | Coverage |",
                "|--------|----------|",
            ])
            for metric, value in self.report.coverage_metrics.items():
                metric_name = metric.replace("_", " ").title()
                lines.append(f"| {metric_name} | {value:.0%} |")
        
        lines.extend(["", "## Findings by Category", ""])
        
        # Group findings by category
        for category in FindingCategory:
            category_findings = self.report.get_findings_by_category(category)
            if category_findings:
                lines.extend([
                    f"### {category.value.title()} ({len(category_findings)})",
                    "",
                ])
                
                for finding in category_findings:
                    severity_icon = {
                        Severity.CRITICAL: "🔴",
                        Severity.MAJOR: "🟠",
                        Severity.MINOR: "🟡",
                        Severity.INFO: "🔵",
                    }.get(finding.severity, "⚪")
                    
                    status_icon = {
                        ResolutionStatus.OPEN: "❌",
                        ResolutionStatus.IN_PROGRESS: "🔄",
                        ResolutionStatus.RESOLVED: "✅",
                        ResolutionStatus.WONT_FIX: "⏭️",
                    }.get(finding.status, "❓")
                    
                    lines.extend([
                        f"#### {severity_icon} {finding.id}: {finding.description}",
                        "",
                        f"- **Status:** {status_icon} {finding.status.value}",
                        f"- **Location:** `{finding.source_location}`",
                    ])
                    
                    if finding.requirement_ref:
                        lines.append(f"- **Requirement:** {finding.requirement_ref}")
                    
                    if finding.expected:
                        lines.append(f"- **Expected:** {finding.expected}")
                    
                    if finding.actual:
                        lines.append(f"- **Actual:** {finding.actual}")
                    
                    if finding.suggestion:
                        lines.append(f"- **Suggestion:** {finding.suggestion}")
                    
                    if finding.resolution_notes:
                        lines.append(f"- **Resolution Notes:** {finding.resolution_notes}")
                    
                    lines.append("")
        
        # Add detailed findings table
        lines.extend([
            "## All Findings Summary",
            "",
            "| ID | Severity | Category | Location | Status | Description |",
            "|----|----------|----------|----------|--------|-------------|",
        ])
        
        for finding in self.report.findings:
            desc_short = finding.description[:50] + "..." if len(finding.description) > 50 else finding.description
            lines.append(
                f"| {finding.id} | {finding.severity.value} | {finding.category.value} | "
                f"`{finding.source_location.file_path}` | {finding.status.value} | {desc_short} |"
            )
        
        lines.extend([
            "",
            "---",
            "",
            "*This report was generated by the UDL Rating Framework Documentation Validation System.*",
            "",
            "**Validates:** Requirements 10.1, 10.2, 10.3, 10.4, 10.5",
        ])
        
        return "\n".join(lines)
    
    def generate_json_report(self) -> str:
        """Generate detailed JSON report."""
        self.generate_recommendations()
        return json.dumps(self.report.to_dict(), indent=2)
    
    def save_report(self, output_path: Path, format: str = "markdown"):
        """Save report to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "markdown":
            content = self.generate_markdown_report()
        elif format == "json":
            content = self.generate_json_report()
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
