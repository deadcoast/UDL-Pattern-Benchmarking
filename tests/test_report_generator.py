"""
Unit tests for report generation functionality.

Tests the report generation system's ability to:
- Generate JSON format reports correctly
- Generate CSV format reports correctly
- Generate HTML reports with visualizations
- Verify all required fields are present
"""

import pytest
import json
import csv
import tempfile
from datetime import datetime
from pathlib import Path
from io import StringIO
from typing import List

from udl_rating_framework.io.report_generator import ReportGenerator
from udl_rating_framework.core.pipeline import QualityReport, ComputationStep


class TestReportGenerator:
    """Test suite for ReportGenerator class."""

    @pytest.fixture
    def sample_computation_steps(self) -> List[ComputationStep]:
        """Create sample computation steps for testing."""
        return [
            ComputationStep(
                step_number=1,
                operation="Initialize metric computation",
                formula="",
                inputs={
                    "udl_file": "test.udl",
                    "metrics": ["consistency", "completeness"],
                },
                output="Starting computation",
                intermediate_values={},
            ),
            ComputationStep(
                step_number=2,
                operation="Compute consistency metric",
                formula=r"C = 1 - \frac{|Contradictions| + |Cycles|}{|Rules| + 1}",
                inputs={"udl": "test.udl"},
                output=0.85,
                intermediate_values={"consistency": 0.85},
            ),
            ComputationStep(
                step_number=3,
                operation="Compute completeness metric",
                formula=r"Completeness = \frac{|Defined|}{|Required|}",
                inputs={"udl": "test.udl"},
                output=0.75,
                intermediate_values={"completeness": 0.75},
            ),
            ComputationStep(
                step_number=4,
                operation="Aggregate metrics",
                formula=r"Q = \sum_{i} w_i \cdot m_i",
                inputs={
                    "weights": {"consistency": 0.5, "completeness": 0.5},
                    "metric_values": {"consistency": 0.85, "completeness": 0.75},
                },
                output=0.80,
                intermediate_values={"overall_score": 0.80},
            ),
        ]

    @pytest.fixture
    def sample_quality_report(self, sample_computation_steps) -> QualityReport:
        """Create a sample quality report for testing."""
        return QualityReport(
            overall_score=0.80,
            confidence=0.92,
            metric_scores={
                "consistency": 0.85,
                "completeness": 0.75,
                "expressiveness": 0.78,
                "structural_coherence": 0.82,
            },
            metric_formulas={
                "consistency": r"C = 1 - \frac{|Contradictions| + |Cycles|}{|Rules| + 1}",
                "completeness": r"Completeness = \frac{|Defined|}{|Required|}",
                "expressiveness": r"E = \frac{Chomsky\_Level + Complexity\_Score}{2}",
                "structural_coherence": r"SC = 1 - \frac{H(G)}{H_{max}}",
            },
            computation_trace=sample_computation_steps,
            error_bounds={
                "consistency": (0.80, 0.90),
                "completeness": (0.70, 0.80),
                "expressiveness": (0.73, 0.83),
                "structural_coherence": (0.77, 0.87),
                "overall": (0.75, 0.85),
            },
            timestamp=datetime(2024, 1, 15, 10, 30, 45),
            udl_file="test_sample.udl",
            errors=[],
            warnings=["Minor parsing ambiguity resolved"],
        )

    @pytest.fixture
    def sample_quality_report_with_errors(
        self, sample_computation_steps
    ) -> QualityReport:
        """Create a sample quality report with errors for testing."""
        return QualityReport(
            overall_score=0.45,
            confidence=0.60,
            metric_scores={"consistency": 0.50, "completeness": 0.40},
            metric_formulas={
                "consistency": r"C = 1 - \frac{|Contradictions| + |Cycles|}{|Rules| + 1}",
                "completeness": r"Completeness = \frac{|Defined|}{|Required|}",
            },
            computation_trace=sample_computation_steps[:2],  # Fewer steps due to errors
            error_bounds={
                "consistency": (0.45, 0.55),
                "completeness": (0.35, 0.45),
                "overall": (0.40, 0.50),
            },
            timestamp=datetime(2024, 1, 15, 11, 15, 30),
            udl_file="error_sample.udl",
            errors=[
                "Failed to compute expressiveness metric: Division by zero",
                "Failed to compute structural coherence: Graph is empty",
            ],
            warnings=[
                "Input file has unusual structure",
                "Some tokens could not be classified",
            ],
        )

    @pytest.fixture
    def report_generator(self) -> ReportGenerator:
        """Create a ReportGenerator instance for testing."""
        return ReportGenerator(include_visualizations=True)

    def test_json_report_single_report(
        self, report_generator: ReportGenerator, sample_quality_report: QualityReport
    ):
        """Test JSON report generation for a single report."""
        json_str = report_generator.generate_json_report(sample_quality_report)

        # Parse JSON to verify structure
        data = json.loads(json_str)

        # Verify top-level structure
        assert "report_metadata" in data
        assert "reports" in data
        assert len(data["reports"]) == 1

        # Verify metadata
        metadata = data["report_metadata"]
        assert metadata["total_reports"] == 1
        assert metadata["format"] == "JSON"
        assert "generated_at" in metadata
        assert "generator_version" in metadata

        # Verify report content
        report = data["reports"][0]
        assert report["udl_file"] == "test_sample.udl"
        assert report["overall_score"] == 0.80
        assert report["confidence"] == 0.92
        assert len(report["metric_scores"]) == 4
        assert len(report["metric_formulas"]) == 4
        assert len(report["computation_trace"]) == 4
        assert len(report["error_bounds"]) == 5
        assert len(report["errors"]) == 0
        assert len(report["warnings"]) == 1

        # Verify metric scores
        assert report["metric_scores"]["consistency"] == 0.85
        assert report["metric_scores"]["completeness"] == 0.75

        # Verify error bounds structure
        assert "lower" in report["error_bounds"]["consistency"]
        assert "upper" in report["error_bounds"]["consistency"]

        # Verify computation trace structure
        trace_step = report["computation_trace"][0]
        assert "step_number" in trace_step
        assert "operation" in trace_step
        assert "formula" in trace_step
        assert "inputs" in trace_step
        assert "output" in trace_step
        assert "intermediate_values" in trace_step

    def test_json_report_multiple_reports(
        self,
        report_generator: ReportGenerator,
        sample_quality_report: QualityReport,
        sample_quality_report_with_errors: QualityReport,
    ):
        """Test JSON report generation for multiple reports."""
        reports = [sample_quality_report, sample_quality_report_with_errors]
        json_str = report_generator.generate_json_report(reports)

        data = json.loads(json_str)

        # Verify multiple reports
        assert len(data["reports"]) == 2
        assert data["report_metadata"]["total_reports"] == 2

        # Verify summary statistics are included
        assert "summary" in data
        summary = data["summary"]
        assert "overall_score" in summary
        assert "confidence" in summary
        assert "metrics" in summary
        assert "errors" in summary

        # Verify summary statistics values
        assert summary["overall_score"]["count"] == 2
        assert summary["overall_score"]["mean"] == pytest.approx(
            (0.80 + 0.45) / 2, rel=1e-3
        )
        assert summary["errors"]["total_errors"] == 2
        assert summary["errors"]["total_warnings"] == 3

    def test_json_report_file_output(
        self, report_generator: ReportGenerator, sample_quality_report: QualityReport
    ):
        """Test JSON report generation with file output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_report.json"

            json_str = report_generator.generate_json_report(
                sample_quality_report, output_path=output_path
            )

            # Verify file was created
            assert output_path.exists()

            # Verify file content matches returned string
            with open(output_path, "r", encoding="utf-8") as f:
                file_content = f.read()

            assert file_content == json_str

            # Verify JSON is valid
            data = json.loads(file_content)
            assert data["reports"][0]["udl_file"] == "test_sample.udl"

    def test_csv_report_single_report(
        self, report_generator: ReportGenerator, sample_quality_report: QualityReport
    ):
        """Test CSV report generation for a single report."""
        csv_str = report_generator.generate_csv_report(sample_quality_report)

        # Parse CSV to verify structure
        reader = csv.DictReader(StringIO(csv_str))
        rows = list(reader)

        # Verify single row
        assert len(rows) == 1
        row = rows[0]

        # Verify basic fields
        assert row["udl_file"] == "test_sample.udl"
        assert float(row["overall_score"]) == pytest.approx(0.80, rel=1e-3)
        assert float(row["confidence"]) == pytest.approx(0.92, rel=1e-3)
        assert row["has_errors"] == "False"
        assert row["has_warnings"] == "True"
        assert int(row["error_count"]) == 0
        assert int(row["warning_count"]) == 1

        # Verify metric columns exist
        assert "metric_consistency" in row
        assert "metric_completeness" in row
        assert "metric_expressiveness" in row
        assert "metric_structural_coherence" in row

        # Verify metric values
        assert float(row["metric_consistency"]) == pytest.approx(0.85, rel=1e-3)
        assert float(row["metric_completeness"]) == pytest.approx(0.75, rel=1e-3)

        # Verify error bound columns exist
        assert "error_bound_consistency_lower" in row
        assert "error_bound_consistency_upper" in row

        # Verify error bound values
        assert float(row["error_bound_consistency_lower"]) == pytest.approx(
            0.80, rel=1e-3
        )
        assert float(row["error_bound_consistency_upper"]) == pytest.approx(
            0.90, rel=1e-3
        )

    def test_csv_report_multiple_reports(
        self,
        report_generator: ReportGenerator,
        sample_quality_report: QualityReport,
        sample_quality_report_with_errors: QualityReport,
    ):
        """Test CSV report generation for multiple reports."""
        reports = [sample_quality_report, sample_quality_report_with_errors]
        csv_str = report_generator.generate_csv_report(reports)

        reader = csv.DictReader(StringIO(csv_str))
        rows = list(reader)

        # Verify multiple rows
        assert len(rows) == 2

        # Verify first report
        row1 = rows[0]
        assert row1["udl_file"] == "test_sample.udl"
        assert row1["has_errors"] == "False"

        # Verify second report
        row2 = rows[1]
        assert row2["udl_file"] == "error_sample.udl"
        assert row2["has_errors"] == "True"
        assert int(row2["error_count"]) == 2

    def test_csv_report_file_output(
        self, report_generator: ReportGenerator, sample_quality_report: QualityReport
    ):
        """Test CSV report generation with file output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_report.csv"

            csv_str = report_generator.generate_csv_report(
                sample_quality_report, output_path=output_path
            )

            # Verify file was created
            assert output_path.exists()

            # Verify file content matches returned string
            with open(output_path, "r", encoding="utf-8", newline="") as f:
                file_content = f.read()

            assert file_content == csv_str

    def test_html_report_single_report(
        self, report_generator: ReportGenerator, sample_quality_report: QualityReport
    ):
        """Test HTML report generation for a single report."""
        html_str = report_generator.generate_html_report(sample_quality_report)

        # Verify HTML structure
        assert "<!DOCTYPE html>" in html_str
        assert '<html lang="en">' in html_str
        assert "<head>" in html_str
        assert "<body>" in html_str
        assert "</html>" in html_str

        # Verify title and metadata
        assert "UDL Quality Assessment Report" in html_str
        assert "1 report(s)" in html_str

        # Verify report content
        assert "test_sample.udl" in html_str
        assert "0.800" in html_str  # Overall score
        assert "0.920" in html_str  # Confidence

        # Verify metrics table
        assert "consistency" in html_str
        assert "completeness" in html_str
        assert "0.850" in html_str  # Consistency score
        assert "0.750" in html_str  # Completeness score

        # Verify formulas are included
        assert "Contradictions" in html_str
        assert "Defined" in html_str

        # Verify computation trace
        assert "Computation Trace" in html_str
        assert "Step 1:" in html_str
        assert "Initialize metric computation" in html_str

        # Verify warnings section
        assert "Warnings" in html_str
        assert "Minor parsing ambiguity resolved" in html_str

        # Verify CSS and JavaScript
        assert "<style>" in html_str
        assert "<script>" in html_str
        assert "collapsible" in html_str

    def test_html_report_multiple_reports(
        self,
        report_generator: ReportGenerator,
        sample_quality_report: QualityReport,
        sample_quality_report_with_errors: QualityReport,
    ):
        """Test HTML report generation for multiple reports."""
        reports = [sample_quality_report, sample_quality_report_with_errors]
        html_str = report_generator.generate_html_report(reports)

        # Verify multiple reports
        assert "2 report(s)" in html_str
        assert "Report 1:" in html_str
        assert "Report 2:" in html_str

        # Verify both file names
        assert "test_sample.udl" in html_str
        assert "error_sample.udl" in html_str

        # Verify summary section
        assert "Summary Statistics" in html_str
        assert "Average Quality" in html_str
        assert "Average Confidence" in html_str
        assert "Total Errors" in html_str
        assert "Total Warnings" in html_str

        # Verify error section for second report
        assert "Errors (2)" in html_str
        assert "Failed to compute expressiveness metric" in html_str

    def test_html_report_file_output(
        self, report_generator: ReportGenerator, sample_quality_report: QualityReport
    ):
        """Test HTML report generation with file output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_report.html"

            html_str = report_generator.generate_html_report(
                sample_quality_report, output_path=output_path
            )

            # Verify file was created
            assert output_path.exists()

            # Verify file content matches returned string
            with open(output_path, "r", encoding="utf-8") as f:
                file_content = f.read()

            assert file_content == html_str

    def test_html_report_custom_title(
        self, report_generator: ReportGenerator, sample_quality_report: QualityReport
    ):
        """Test HTML report generation with custom title."""
        custom_title = "Custom UDL Analysis Report"
        html_str = report_generator.generate_html_report(
            sample_quality_report, title=custom_title
        )

        assert custom_title in html_str
        assert f"<title>{custom_title}</title>" in html_str

    def test_report_generator_without_visualizations(
        self, sample_quality_report: QualityReport
    ):
        """Test report generator without visualizations."""
        generator = ReportGenerator(include_visualizations=False)

        # Should still generate reports successfully
        json_str = generator.generate_json_report(sample_quality_report)
        csv_str = generator.generate_csv_report(sample_quality_report)
        html_str = generator.generate_html_report(sample_quality_report)

        assert len(json_str) > 0
        assert len(csv_str) > 0
        assert len(html_str) > 0

    def test_empty_reports_list(self, report_generator: ReportGenerator):
        """Test report generation with empty reports list."""
        empty_reports = []

        # JSON should handle empty list gracefully
        json_str = report_generator.generate_json_report(empty_reports)
        data = json.loads(json_str)
        assert data["report_metadata"]["total_reports"] == 0
        assert len(data["reports"]) == 0

        # CSV should handle empty list gracefully
        csv_str = report_generator.generate_csv_report(empty_reports)
        reader = csv.DictReader(StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 0

        # HTML should handle empty list gracefully
        html_str = report_generator.generate_html_report(empty_reports)
        assert "0 report(s)" in html_str

    def test_report_with_missing_fields(self, report_generator: ReportGenerator):
        """Test report generation with reports missing some fields."""
        minimal_report = QualityReport(
            overall_score=0.5,
            confidence=0.7,
            metric_scores={},  # Empty metrics
            metric_formulas={},
            computation_trace=[],  # Empty trace
            error_bounds={},
            timestamp=datetime.now(),
            udl_file="minimal.udl",
            errors=[],
            warnings=[],
        )

        # Should handle missing fields gracefully
        json_str = report_generator.generate_json_report(minimal_report)
        csv_str = report_generator.generate_csv_report(minimal_report)
        html_str = report_generator.generate_html_report(minimal_report)

        # Verify basic structure is maintained
        json_data = json.loads(json_str)
        assert json_data["reports"][0]["overall_score"] == 0.5

        csv_reader = csv.DictReader(StringIO(csv_str))
        csv_rows = list(csv_reader)
        assert len(csv_rows) == 1
        assert float(csv_rows[0]["overall_score"]) == pytest.approx(0.5, rel=1e-3)

        assert "minimal.udl" in html_str
        assert "0.500" in html_str

    def test_special_characters_in_content(self, report_generator: ReportGenerator):
        """Test report generation with special characters in content."""
        special_report = QualityReport(
            overall_score=0.6,
            confidence=0.8,
            metric_scores={"test_metric": 0.7},
            metric_formulas={"test_metric": "Special chars: <>&\"'"},
            computation_trace=[
                ComputationStep(
                    step_number=1,
                    operation="Test with special chars: <>&\"'",
                    formula="x = y & z",
                    inputs={"special": "<test>"},
                    output="Result with & and <",
                    intermediate_values={},
                )
            ],
            error_bounds={},
            timestamp=datetime.now(),
            udl_file="special_chars_<>&\"'.udl",
            errors=["Error with <special> chars"],
            warnings=["Warning with & chars"],
        )

        # JSON should handle special characters
        json_str = report_generator.generate_json_report(special_report)
        json_data = json.loads(json_str)
        assert json_data["reports"][0]["udl_file"] == "special_chars_<>&\"'.udl"

        # CSV should handle special characters
        csv_str = report_generator.generate_csv_report(special_report)
        csv_reader = csv.DictReader(StringIO(csv_str))
        csv_rows = list(csv_reader)
        assert csv_rows[0]["udl_file"] == "special_chars_<>&\"'.udl"

        # HTML should escape special characters
        html_str = report_generator.generate_html_report(special_report)
        assert "&lt;" in html_str  # < should be escaped
        assert "&gt;" in html_str  # > should be escaped
        assert "&amp;" in html_str  # & should be escaped

    def test_large_numbers_precision(self, report_generator: ReportGenerator):
        """Test report generation maintains numerical precision."""
        precision_report = QualityReport(
            overall_score=0.123456789,
            confidence=0.987654321,
            metric_scores={
                "precise_metric": 0.111111111,
                "another_metric": 0.999999999,
            },
            metric_formulas={
                "precise_metric": "P = 1/9",
                "another_metric": "A = 1 - ε",
            },
            computation_trace=[],
            error_bounds={
                "precise_metric": (0.111111110, 0.111111112),
                "another_metric": (0.999999998, 1.000000000),
            },
            timestamp=datetime.now(),
            udl_file="precision_test.udl",
            errors=[],
            warnings=[],
        )

        # JSON should maintain full precision
        json_str = report_generator.generate_json_report(precision_report)
        json_data = json.loads(json_str)
        assert json_data["reports"][0]["overall_score"] == pytest.approx(
            0.123456789, rel=1e-9
        )
        assert json_data["reports"][0]["metric_scores"][
            "precise_metric"
        ] == pytest.approx(0.111111111, rel=1e-9)

        # CSV should maintain precision
        csv_str = report_generator.generate_csv_report(precision_report)
        csv_reader = csv.DictReader(StringIO(csv_str))
        csv_rows = list(csv_reader)
        assert float(csv_rows[0]["overall_score"]) == pytest.approx(
            0.123456789, rel=1e-9
        )

        # HTML should display appropriate precision
        html_str = report_generator.generate_html_report(precision_report)
        assert "0.123" in html_str  # Should show reasonable precision in display
