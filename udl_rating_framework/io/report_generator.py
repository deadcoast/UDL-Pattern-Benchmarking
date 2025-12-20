"""
Report generation and output formatting module.

Provides comprehensive report generation in multiple formats (JSON, CSV, HTML)
with visualizations and mathematical traces.
"""

import json
import csv
import html
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import asdict
import base64
from io import StringIO, BytesIO

from udl_rating_framework.core.pipeline import QualityReport, ComputationStep


class ReportGenerator:
    """
    Generates comprehensive quality assessment reports in multiple formats.

    Supports:
    - JSON: Structured data format for programmatic access
    - CSV: Tabular format for spreadsheet analysis
    - HTML: Rich format with visualizations and mathematical formulas
    """

    def __init__(self, include_visualizations: bool = True):
        """
        Initialize report generator.

        Args:
            include_visualizations: Whether to include charts and graphs in HTML reports
        """
        self.include_visualizations = include_visualizations

    def generate_json_report(
        self,
        reports: Union[QualityReport, List[QualityReport]],
        output_path: Optional[Path] = None,
        indent: int = 2,
    ) -> str:
        """
        Generate JSON format report.

        Args:
            reports: Single report or list of reports
            output_path: Optional file path to save report
            indent: JSON indentation level

        Returns:
            JSON string representation of the report(s)
        """
        if isinstance(reports, QualityReport):
            reports = [reports]

        # Convert reports to serializable format
        json_data = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "generator_version": "1.0.0",
                "total_reports": len(reports),
                "format": "JSON",
            },
            "reports": [],
        }

        for report in reports:
            report_dict = self._report_to_dict(report)
            json_data["reports"].append(report_dict)

        # Add summary statistics for multiple reports
        if len(reports) > 1:
            json_data["summary"] = self._generate_summary_stats(reports)

        json_str = json.dumps(json_data, indent=indent, ensure_ascii=False)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(json_str)

        return json_str

    def generate_csv_report(
        self,
        reports: Union[QualityReport, List[QualityReport]],
        output_path: Optional[Path] = None,
        include_trace: bool = False,
    ) -> str:
        """
        Generate CSV format report.

        Args:
            reports: Single report or list of reports
            output_path: Optional file path to save report
            include_trace: Whether to include computation trace in separate sheet

        Returns:
            CSV string representation of the report(s)
        """
        if isinstance(reports, QualityReport):
            reports = [reports]

        output = StringIO()

        # Main report data
        fieldnames = [
            "udl_file",
            "timestamp",
            "overall_score",
            "confidence",
            "has_errors",
            "has_warnings",
            "error_count",
            "warning_count",
        ]

        # Add metric columns dynamically
        all_metrics = set()
        for report in reports:
            all_metrics.update(report.metric_scores.keys())

        metric_columns = sorted(all_metrics)
        fieldnames.extend([f"metric_{metric}" for metric in metric_columns])

        # Add error bound columns
        fieldnames.extend([f"error_bound_{metric}_lower" for metric in metric_columns])
        fieldnames.extend([f"error_bound_{metric}_upper" for metric in metric_columns])

        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for report in reports:
            row = {
                "udl_file": report.udl_file,
                "timestamp": report.timestamp.isoformat(),
                "overall_score": report.overall_score,
                "confidence": report.confidence,
                "has_errors": len(report.errors) > 0,
                "has_warnings": len(report.warnings) > 0,
                "error_count": len(report.errors),
                "warning_count": len(report.warnings),
            }

            # Add metric scores
            for metric in metric_columns:
                row[f"metric_{metric}"] = report.metric_scores.get(metric, "")

            # Add error bounds
            for metric in metric_columns:
                bounds = report.error_bounds.get(metric, ("", ""))
                row[f"error_bound_{metric}_lower"] = (
                    bounds[0] if bounds[0] != "" else ""
                )
                row[f"error_bound_{metric}_upper"] = (
                    bounds[1] if bounds[1] != "" else ""
                )

            writer.writerow(row)

        csv_content = output.getvalue()
        output.close()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                f.write(csv_content)

        return csv_content

    def generate_html_report(
        self,
        reports: Union[QualityReport, List[QualityReport]],
        output_path: Optional[Path] = None,
        title: str = "UDL Quality Assessment Report",
    ) -> str:
        """
        Generate HTML format report with visualizations.

        Args:
            reports: Single report or list of reports
            output_path: Optional file path to save report
            title: Report title

        Returns:
            HTML string representation of the report(s)
        """
        if isinstance(reports, QualityReport):
            reports = [reports]

        html_content = self._generate_html_template(reports, title)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)

        return html_content

    def _report_to_dict(self, report: QualityReport) -> Dict[str, Any]:
        """Convert QualityReport to dictionary for JSON serialization."""
        report_dict = {
            "udl_file": report.udl_file,
            "timestamp": report.timestamp.isoformat(),
            "overall_score": report.overall_score,
            "confidence": report.confidence,
            "metric_scores": report.metric_scores,
            "metric_formulas": report.metric_formulas,
            "error_bounds": {
                k: {"lower": v[0], "upper": v[1]}
                for k, v in report.error_bounds.items()
            },
            "errors": report.errors,
            "warnings": report.warnings,
            "computation_trace": [],
        }

        # Convert computation trace
        for step in report.computation_trace:
            step_dict = {
                "step_number": step.step_number,
                "operation": step.operation,
                "formula": step.formula,
                "inputs": step.inputs,
                "output": str(step.output),  # Convert to string for JSON serialization
                "intermediate_values": step.intermediate_values,
            }
            report_dict["computation_trace"].append(step_dict)

        return report_dict

    def _generate_summary_stats(self, reports: List[QualityReport]) -> Dict[str, Any]:
        """Generate summary statistics for multiple reports."""
        if not reports:
            return {}

        # Overall score statistics
        overall_scores = [r.overall_score for r in reports]
        confidence_scores = [r.confidence for r in reports]

        # Metric statistics
        all_metrics = set()
        for report in reports:
            all_metrics.update(report.metric_scores.keys())

        metric_stats = {}
        for metric in all_metrics:
            values = [
                r.metric_scores.get(metric, 0.0)
                for r in reports
                if metric in r.metric_scores
            ]
            if values:
                metric_stats[metric] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }

        # Error statistics
        total_errors = sum(len(r.errors) for r in reports)
        total_warnings = sum(len(r.warnings) for r in reports)

        return {
            "overall_score": {
                "mean": (
                    sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
                ),
                "min": min(overall_scores) if overall_scores else 0.0,
                "max": max(overall_scores) if overall_scores else 0.0,
                "count": len(overall_scores),
            },
            "confidence": {
                "mean": (
                    sum(confidence_scores) / len(confidence_scores)
                    if confidence_scores
                    else 0.0
                ),
                "min": min(confidence_scores) if confidence_scores else 0.0,
                "max": max(confidence_scores) if confidence_scores else 0.0,
                "count": len(confidence_scores),
            },
            "metrics": metric_stats,
            "errors": {
                "total_errors": total_errors,
                "total_warnings": total_warnings,
                "reports_with_errors": sum(1 for r in reports if r.errors),
                "reports_with_warnings": sum(1 for r in reports if r.warnings),
            },
        }

    def _generate_html_template(self, reports: List[QualityReport], title: str) -> str:
        """Generate complete HTML report template."""

        # Generate CSS styles
        css_styles = self._generate_css_styles()

        # Generate JavaScript for interactivity
        javascript = self._generate_javascript()

        # Generate report content
        content_sections = []

        # Header section
        content_sections.append(self._generate_html_header(reports, title))

        # Summary section (for multiple reports)
        if len(reports) > 1:
            content_sections.append(self._generate_html_summary(reports))

        # Individual report sections
        for i, report in enumerate(reports):
            content_sections.append(self._generate_html_report_section(report, i))

        # Footer section
        content_sections.append(self._generate_html_footer())

        # Combine all sections
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <style>{css_styles}</style>
</head>
<body>
    <div class="container">
        {"".join(content_sections)}
    </div>
    <script>{javascript}</script>
</body>
</html>"""

        return html_content

    def _generate_css_styles(self) -> str:
        """Generate CSS styles for HTML report."""
        return """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        
        .header {
            text-align: center;
            border-bottom: 2px solid #333;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        
        .header h1 {
            color: #333;
            margin: 0;
        }
        
        .header .subtitle {
            color: #666;
            font-size: 1.1em;
            margin-top: 10px;
        }
        
        .summary-section {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        
        .report-section {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            background-color: white;
        }
        
        .report-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #eee;
            padding-bottom: 15px;
            margin-bottom: 20px;
        }
        
        .score-display {
            display: flex;
            gap: 30px;
            margin: 20px 0;
        }
        
        .score-item {
            text-align: center;
            padding: 15px;
            border-radius: 8px;
            background-color: #f8f9fa;
            min-width: 120px;
        }
        
        .score-value {
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }
        
        .score-label {
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
        }
        
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        .metrics-table th,
        .metrics-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        .metrics-table th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        
        .formula {
            font-family: 'Courier New', monospace;
            background-color: #f8f9fa;
            padding: 2px 4px;
            border-radius: 3px;
            font-size: 0.9em;
        }
        
        .computation-trace {
            margin-top: 30px;
        }
        
        .trace-step {
            background-color: #f8f9fa;
            border-left: 4px solid #007bff;
            padding: 15px;
            margin: 10px 0;
            border-radius: 0 8px 8px 0;
        }
        
        .trace-step-header {
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }
        
        .trace-formula {
            font-family: 'Courier New', monospace;
            background-color: white;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
            border: 1px solid #ddd;
        }
        
        .error-section {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }
        
        .warning-section {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }
        
        .collapsible {
            cursor: pointer;
            padding: 10px;
            background-color: #f1f1f1;
            border: none;
            text-align: left;
            outline: none;
            font-size: 1em;
            width: 100%;
            border-radius: 4px;
        }
        
        .collapsible:hover {
            background-color: #ddd;
        }
        
        .collapsible-content {
            padding: 0 18px;
            display: none;
            overflow: hidden;
            background-color: #f9f9f9;
            border-radius: 0 0 4px 4px;
        }
        
        .footer {
            text-align: center;
            color: #666;
            font-size: 0.9em;
            border-top: 1px solid #eee;
            padding-top: 20px;
            margin-top: 40px;
        }
        
        .progress-bar {
            width: 100%;
            height: 20px;
            background-color: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #dc3545 0%, #ffc107 50%, #28a745 100%);
            transition: width 0.3s ease;
        }
        """

    def _generate_javascript(self) -> str:
        """Generate JavaScript for HTML report interactivity."""
        return """
        // Toggle collapsible sections
        document.addEventListener('DOMContentLoaded', function() {
            var collapsibles = document.getElementsByClassName('collapsible');
            for (var i = 0; i < collapsibles.length; i++) {
                collapsibles[i].addEventListener('click', function() {
                    this.classList.toggle('active');
                    var content = this.nextElementSibling;
                    if (content.style.display === 'block') {
                        content.style.display = 'none';
                    } else {
                        content.style.display = 'block';
                    }
                });
            }
        });
        
        // Format numbers for display
        function formatScore(score) {
            return parseFloat(score).toFixed(3);
        }
        
        // Generate progress bar color based on score
        function getScoreColor(score) {
            if (score < 0.3) return '#dc3545';  // Red
            if (score < 0.7) return '#ffc107';  // Yellow
            return '#28a745';  // Green
        }
        """

    def _generate_html_header(self, reports: List[QualityReport], title: str) -> str:
        """Generate HTML header section."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return f"""
        <div class="header">
            <h1>{html.escape(title)}</h1>
            <div class="subtitle">
                Generated on {timestamp} | {len(reports)} report(s)
            </div>
        </div>
        """

    def _generate_html_summary(self, reports: List[QualityReport]) -> str:
        """Generate HTML summary section for multiple reports."""
        summary_stats = self._generate_summary_stats(reports)

        overall_stats = summary_stats.get("overall_score", {})
        confidence_stats = summary_stats.get("confidence", {})
        error_stats = summary_stats.get("errors", {})

        return f"""
        <div class="summary-section">
            <h2>Summary Statistics</h2>
            <div class="score-display">
                <div class="score-item">
                    <div class="score-value">{overall_stats.get("mean", 0.0):.3f}</div>
                    <div class="score-label">Average Quality</div>
                </div>
                <div class="score-item">
                    <div class="score-value">{confidence_stats.get("mean", 0.0):.3f}</div>
                    <div class="score-label">Average Confidence</div>
                </div>
                <div class="score-item">
                    <div class="score-value">{error_stats.get("total_errors", 0)}</div>
                    <div class="score-label">Total Errors</div>
                </div>
                <div class="score-item">
                    <div class="score-value">{error_stats.get("total_warnings", 0)}</div>
                    <div class="score-label">Total Warnings</div>
                </div>
            </div>
        </div>
        """

    def _generate_html_report_section(self, report: QualityReport, index: int) -> str:
        """Generate HTML section for individual report."""
        file_name = Path(report.udl_file).name

        # Generate metrics table
        metrics_table = self._generate_metrics_table_html(report)

        # Generate computation trace
        trace_html = self._generate_computation_trace_html(report)

        # Generate error/warning sections
        error_html = self._generate_error_warning_html(report)

        return f"""
        <div class="report-section">
            <div class="report-header">
                <h2>Report {index + 1}: {html.escape(file_name)}</h2>
                <div class="timestamp">{report.timestamp.strftime("%Y-%m-%d %H:%M:%S")}</div>
            </div>
            
            <div class="score-display">
                <div class="score-item">
                    <div class="score-value">{report.overall_score:.3f}</div>
                    <div class="score-label">Overall Quality Score</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {report.overall_score * 100}%; background-color: {self._get_score_color(report.overall_score)};"></div>
                    </div>
                </div>
                <div class="score-item">
                    <div class="score-value">{report.confidence:.3f}</div>
                    <div class="score-label">Confidence</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {report.confidence * 100}%; background-color: {self._get_score_color(report.confidence)};"></div>
                    </div>
                </div>
            </div>
            
            {metrics_table}
            {error_html}
            {trace_html}
        </div>
        """

    def _generate_metrics_table_html(self, report: QualityReport) -> str:
        """Generate HTML table for metrics."""
        if not report.metric_scores:
            return "<p>No metrics computed.</p>"

        rows = []
        for metric_name, score in report.metric_scores.items():
            formula = report.metric_formulas.get(metric_name, "N/A")
            bounds = report.error_bounds.get(metric_name, ("N/A", "N/A"))

            # Format bounds safely
            bounds_str = "N/A"
            if (
                bounds != ("N/A", "N/A")
                and isinstance(bounds[0], (int, float))
                and isinstance(bounds[1], (int, float))
            ):
                bounds_str = f"[{bounds[0]:.6f}, {bounds[1]:.6f}]"

            rows.append(
                f"""
            <tr>
                <td><strong>{html.escape(metric_name)}</strong></td>
                <td>{score:.6f}</td>
                <td><span class="formula">{html.escape(formula)}</span></td>
                <td>{bounds_str}</td>
            </tr>
            """
            )

        return f"""
        <h3>Metric Scores</h3>
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Score</th>
                    <th>Formula</th>
                    <th>Error Bounds</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
        """

    def _generate_computation_trace_html(self, report: QualityReport) -> str:
        """Generate HTML for computation trace."""
        if not report.computation_trace:
            return ""

        trace_steps = []
        for step in report.computation_trace:
            inputs_str = ", ".join(f"{k}: {v}" for k, v in step.inputs.items())
            intermediate_str = ", ".join(
                f"{k}: {v}" for k, v in step.intermediate_values.items()
            )

            trace_steps.append(
                f"""
            <div class="trace-step">
                <div class="trace-step-header">
                    Step {step.step_number}: {html.escape(step.operation)}
                </div>
                {f'<div class="trace-formula">{html.escape(step.formula)}</div>' if step.formula else ""}
                <div><strong>Inputs:</strong> {html.escape(inputs_str)}</div>
                <div><strong>Output:</strong> {html.escape(str(step.output))}</div>
                {f"<div><strong>Intermediate Values:</strong> {html.escape(intermediate_str)}</div>" if step.intermediate_values else ""}
            </div>
            """
            )

        return f"""
        <button class="collapsible">Computation Trace ({len(report.computation_trace)} steps)</button>
        <div class="collapsible-content">
            <div class="computation-trace">
                {"".join(trace_steps)}
            </div>
        </div>
        """

    def _generate_error_warning_html(self, report: QualityReport) -> str:
        """Generate HTML for errors and warnings."""
        html_sections = []

        if report.errors:
            error_items = [f"<li>{html.escape(error)}</li>" for error in report.errors]
            html_sections.append(
                f"""
            <div class="error-section">
                <h4>Errors ({len(report.errors)})</h4>
                <ul>{"".join(error_items)}</ul>
            </div>
            """
            )

        if report.warnings:
            warning_items = [
                f"<li>{html.escape(warning)}</li>" for warning in report.warnings
            ]
            html_sections.append(
                f"""
            <div class="warning-section">
                <h4>Warnings ({len(report.warnings)})</h4>
                <ul>{"".join(warning_items)}</ul>
            </div>
            """
            )

        return "".join(html_sections)

    def _generate_html_footer(self) -> str:
        """Generate HTML footer section."""
        return f"""
        <div class="footer">
            <p>Generated by UDL Rating Framework v1.0.0</p>
            <p>Mathematical formulas and computation traces ensure full traceability of all quality assessments.</p>
        </div>
        """

    def _get_score_color(self, score: float) -> str:
        """Get color for score visualization."""
        if score < 0.3:
            return "#dc3545"  # Red
        elif score < 0.7:
            return "#ffc107"  # Yellow
        else:
            return "#28a745"  # Green
