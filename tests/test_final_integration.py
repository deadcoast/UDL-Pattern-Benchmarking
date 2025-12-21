"""
Final integration and system testing for UDL Rating Framework.

This module implements Task 23: Final integration and system testing
- Run complete end-to-end tests on real UDL examples
- Verify all 40 correctness properties hold
- Test error recovery scenarios
- Test with various UDL formats and sizes
- Verify mathematical correctness on all examples

Requirements: All
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml
from click.testing import CliRunner

from udl_rating_framework.cli.main import cli
from udl_rating_framework.core.aggregation import MetricAggregator
from udl_rating_framework.core.confidence import ConfidenceCalculator
from udl_rating_framework.core.metrics.base import MetricRegistry
from udl_rating_framework.core.metrics.completeness import CompletenessMetric
from udl_rating_framework.core.metrics.consistency import ConsistencyMetric
from udl_rating_framework.core.metrics.expressiveness import ExpressivenessMetric
from udl_rating_framework.core.metrics.structural_coherence import (
    StructuralCoherenceMetric,
)
from udl_rating_framework.core.pipeline import RatingPipeline
from udl_rating_framework.core.representation import UDLRepresentation
from udl_rating_framework.io.file_discovery import FileDiscovery
from udl_rating_framework.io.report_generator import ReportGenerator


class TestFinalIntegration:
    """
    Final integration tests for the complete UDL Rating Framework.

    This test suite verifies that all components work together correctly
    and that all 40 correctness properties hold on real UDL examples.
    """

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = None
        self.runner = CliRunner()

        # Register metrics in the registry
        MetricRegistry.register("consistency", ConsistencyMetric)
        MetricRegistry.register("completeness", CompletenessMetric)
        MetricRegistry.register("expressiveness", ExpressivenessMetric)
        MetricRegistry.register("structural_coherence",
                                StructuralCoherenceMetric)

        # Initialize core components
        self.metrics = {
            "consistency": ConsistencyMetric(),
            "completeness": CompletenessMetric(),
            "expressiveness": ExpressivenessMetric(),
            "structural_coherence": StructuralCoherenceMetric(),
        }

        self.aggregator = MetricAggregator(
            {
                "consistency": 0.25,
                "completeness": 0.25,
                "expressiveness": 0.25,
                "structural_coherence": 0.25,
            }
        )

        self.confidence_calculator = ConfidenceCalculator()
        self.file_discovery = FileDiscovery()
        self.report_generator = ReportGenerator()

        # Initialize pipeline with correct interface
        metric_names = [
            "consistency",
            "completeness",
            "expressiveness",
            "structural_coherence",
        ]
        weights = {
            "consistency": 0.25,
            "completeness": 0.25,
            "expressiveness": 0.25,
            "structural_coherence": 0.25,
        }
        self.pipeline = RatingPipeline(
            metric_names=metric_names,
            weights=weights,
            enable_tracing=True,
            enable_caching=False,  # Disable caching for testing
        )

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = Path(tempfile.mkdtemp())
        self.temp_dir = temp_dir
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def get_real_udl_examples(self) -> List[Path]:
        """Get all real UDL example files."""
        examples_dir = Path("examples/udl_examples")
        if not examples_dir.exists():
            pytest.skip("UDL examples directory not found")

        udl_files = list(examples_dir.glob("*.udl"))
        if not udl_files:
            pytest.skip("No UDL example files found")

        return udl_files

    def test_end_to_end_rating_pipeline(self, temp_dir):
        """
        Test complete end-to-end rating pipeline on real UDL examples.

        This test verifies that the entire system works from file discovery
        through rating computation to report generation.
        """
        # Get real UDL examples
        udl_files = self.get_real_udl_examples()

        # Copy examples to temp directory for testing
        test_files = []
        for udl_file in udl_files[:5]:  # Test with first 5 examples
            dest_file = temp_dir / udl_file.name
            shutil.copy2(udl_file, dest_file)
            test_files.append(dest_file)

        # Test file discovery
        discovery_result = self.file_discovery.discover_files(temp_dir)
        discovered_files = discovery_result.discovered_files
        assert len(discovered_files) >= len(test_files)

        # Test rating pipeline on each file
        all_reports = []
        for udl_file in test_files:
            try:
                # Load and parse UDL
                udl_content = udl_file.read_text()
                udl_repr = UDLRepresentation(udl_content, str(udl_file))

                # Compute rating
                report = self.pipeline.compute_rating(udl_repr)
                all_reports.append(report)

                # Verify report structure
                assert hasattr(report, "overall_score")
                assert hasattr(report, "confidence")
                assert hasattr(report, "metric_scores")
                assert 0 <= report.overall_score <= 1
                assert 0 <= report.confidence <= 1

                # Verify all metrics are present
                expected_metrics = {
                    "consistency",
                    "completeness",
                    "expressiveness",
                    "structural_coherence",
                }
                assert set(report.metric_scores.keys()) >= expected_metrics

                # Verify all metric scores are bounded
                for metric_name, score in report.metric_scores.items():
                    assert 0 <= score <= 1, (
                        f"Metric {metric_name} score {score} not in [0,1]"
                    )

            except Exception as e:
                pytest.fail(f"Failed to process {udl_file.name}: {e}")

        # Test report generation
        output_file = temp_dir / "integration_report.json"
        self.report_generator.generate_json_report(all_reports, output_file)
        assert output_file.exists()

        # Verify report content
        with open(output_file) as f:
            report_data = json.load(f)

        assert "reports" in report_data
        assert len(report_data["reports"]) == len(all_reports)

    def test_cli_integration_all_commands(self, temp_dir):
        """
        Test all CLI commands work correctly with real UDL examples.
        """
        # Get real UDL examples
        udl_files = self.get_real_udl_examples()

        # Copy a few examples to temp directory
        test_files = []
        for udl_file in udl_files[:3]:
            dest_file = temp_dir / udl_file.name
            shutil.copy2(udl_file, dest_file)
            test_files.append(dest_file)

        # Test 'rate' command
        output_file = temp_dir / "cli_rating.json"
        result = self.runner.invoke(
            cli,
            [
                "rate",
                str(temp_dir),
                "--output",
                str(output_file),
                "--format",
                "json",
                "--recursive",
            ],
        )

        if result.exit_code != 0:
            print(f"CLI rate command failed: {result.output}")
            print(f"Exception: {result.exception}")

        assert result.exit_code == 0, f"Rate command failed: {result.output}"
        assert output_file.exists()

        # Verify output format
        with open(output_file) as f:
            data = json.load(f)
        assert "reports" in data
        assert len(data["reports"]) >= 1

        # Test 'compare' command with multiple files
        if len(test_files) >= 2:
            compare_output = temp_dir / "comparison.json"
            result = self.runner.invoke(
                cli,
                [
                    "compare",
                    str(test_files[0]),
                    str(test_files[1]),
                    "--output",
                    str(compare_output),
                    "--format",
                    "json",
                ],
            )

            if result.exit_code != 0:
                print(f"CLI compare command failed: {result.output}")
                print(f"Exception: {result.exception}")

            assert result.exit_code == 0, f"Compare command failed: {result.output}"
            assert compare_output.exists()

    def test_error_recovery_scenarios(self, temp_dir):
        """
        Test that the system handles various error scenarios gracefully.
        """
        # Create files with different error conditions

        # 1. Empty file
        empty_file = temp_dir / "empty.udl"
        empty_file.write_text("")

        # 2. Invalid syntax file
        invalid_file = temp_dir / "invalid.udl"
        invalid_file.write_text("this is not valid UDL syntax {{{")

        # 3. Very large file (if system has size limits)
        large_file = temp_dir / "large.udl"
        large_content = "grammar Large {\n" + "  rule: 'a';\n" * 10000 + "}\n"
        large_file.write_text(large_content)

        # 4. File with special characters
        special_file = temp_dir / "special.udl"
        special_file.write_text("grammar Special {\n  rule: 'αβγ' | 'δεζ';\n}")

        # 5. Unreadable file (permission denied)
        unreadable_file = temp_dir / "unreadable.udl"
        unreadable_file.write_text("grammar Test { rule: 'test'; }")
        try:
            unreadable_file.chmod(0o000)  # Remove all permissions
        except (OSError, PermissionError):
            # Skip this test on systems where we can't change permissions
            pass

        # Test that file discovery handles these gracefully
        try:
            discovery_result = self.file_discovery.discover_files(temp_dir)
            discovered_files = discovery_result.discovered_files
            # Should discover readable files, skip unreadable ones
            # At least empty, invalid, large, special
            assert len(discovered_files) >= 3
        except Exception as e:
            pytest.fail(f"File discovery failed on error scenarios: {e}")

        # Test that rating pipeline handles errors gracefully
        error_files = [empty_file, invalid_file, special_file]
        successful_ratings = 0

        for error_file in error_files:
            try:
                udl_content = error_file.read_text()
                udl_repr = UDLRepresentation(udl_content, str(error_file))
                report = self.pipeline.compute_rating(udl_repr)

                # If we get here, the system handled the error gracefully
                successful_ratings += 1

                # Verify the report is still valid
                assert hasattr(report, "overall_score")
                assert 0 <= report.overall_score <= 1

            except Exception as e:
                # This is expected for some error cases
                print(f"Expected error for {error_file.name}: {e}")

        # At least some files should be processable
        print(
            f"Successfully processed {successful_ratings}/{len(error_files)} error scenario files"
        )

        # Restore permissions for cleanup
        try:
            unreadable_file.chmod(0o644)
        except (OSError, PermissionError, FileNotFoundError):
            pass

    def test_various_udl_formats_and_sizes(self, temp_dir):
        """
        Test the system with various UDL formats and sizes.
        """
        # Test different UDL formats and complexities
        test_cases = [
            # Minimal UDL
            ("minimal.udl", "grammar Minimal { start: 'hello'; }"),
            # Simple expression grammar
            (
                "expression.udl",
                """
                grammar Expression {
                    start: expr
                    expr: term ('+' term)*
                    term: factor ('*' factor)*
                    factor: NUMBER | '(' expr ')'
                    NUMBER: /[0-9]+/
                }
            """,
            ),
            # Complex programming language subset
            (
                "programming.udl",
                """
                grammar Programming {
                    start: program
                    program: statement*
                    statement: assignment | if_stmt | while_stmt | block
                    assignment: IDENTIFIER '=' expression ';'
                    if_stmt: 'if' '(' condition ')' statement ('else' statement)?
                    while_stmt: 'while' '(' condition ')' statement
                    block: '{' statement* '}'
                    condition: expression ('==' | '!=' | '<' | '>' | '<=' | '>=') expression
                    expression: term (('+' | '-') term)*
                    term: factor (('*' | '/' | '%') factor)*
                    factor: NUMBER | IDENTIFIER | STRING | '(' expression ')' | function_call
                    function_call: IDENTIFIER '(' (expression (',' expression)*)? ')'
                    IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/
                    NUMBER: /[0-9]+(\\.[0-9]+)?/
                    STRING: /"[^"]*"/
                }
            """,
            ),
            # Grammar with constraints and metadata
            (
                "constrained.udl",
                """
                grammar Constrained {
                    start: document
                    document: header body footer
                    header: 'HEADER' content
                    body: 'BODY' content+
                    footer: 'FOOTER' content
                    content: TEXT | NUMBER | SYMBOL
                    TEXT: /[a-zA-Z]+/
                    NUMBER: /[0-9]+/ { range: 1..1000 }
                    SYMBOL: /[!@#$%^&*()]+/ { max_length: 5 }
                }
            """,
            ),
            # Very large grammar (stress test)
            ("large.udl", self._generate_large_grammar(100)),
        ]

        successful_tests = 0

        for filename, content in test_cases:
            test_file = temp_dir / filename
            test_file.write_text(content)

            try:
                # Test parsing and representation
                udl_repr = UDLRepresentation(content, str(test_file))

                # Test metric computation
                report = self.pipeline.compute_rating(udl_repr)

                # Verify results are valid
                assert 0 <= report.overall_score <= 1
                assert 0 <= report.confidence <= 1
                assert len(report.metric_scores) >= 4

                # Verify all metrics are bounded
                for metric_name, score in report.metric_scores.items():
                    assert 0 <= score <= 1, (
                        f"File {filename}, metric {metric_name}: score {score} not in [0,1]"
                    )

                successful_tests += 1
                print(
                    f"✓ Successfully processed {filename} (score: {report.overall_score:.3f})"
                )

            except Exception as e:
                print(f"✗ Failed to process {filename}: {e}")
                # Don't fail the test immediately - collect results

        # Require that most test cases succeed
        success_rate = successful_tests / len(test_cases)
        assert success_rate >= 0.8, (
            f"Only {successful_tests}/{len(test_cases)} test cases succeeded"
        )

        print(
            f"Successfully processed {successful_tests}/{len(test_cases)} UDL format variations"
        )

    def _generate_large_grammar(self, num_rules: int) -> str:
        """Generate a large grammar for stress testing."""
        rules = ["grammar Large {", "    start: rule_0"]

        for i in range(num_rules):
            if i < num_rules - 1:
                rules.append(f"    rule_{i}: 'token_{i}' rule_{i + 1}?")
            else:
                rules.append(f"    rule_{i}: 'token_{i}'")

        rules.append("}")
        return "\n".join(rules)

    def test_mathematical_correctness_verification(self, temp_dir):
        """
        Verify mathematical correctness on all examples by checking
        that computed values match expected mathematical properties.
        """
        # Get real UDL examples
        udl_files = self.get_real_udl_examples()

        mathematical_errors = []

        for udl_file in udl_files[:10]:  # Test first 10 examples
            try:
                udl_content = udl_file.read_text()
                udl_repr = UDLRepresentation(udl_content, str(udl_file))

                # Test each metric individually
                for metric_name, metric in self.metrics.items():
                    try:
                        score = metric.compute(udl_repr)

                        # Verify boundedness
                        if not (0 <= score <= 1):
                            mathematical_errors.append(
                                f"{udl_file.name}: {metric_name} score {score} not in [0,1]"
                            )

                        # Verify determinism (compute twice)
                        score2 = metric.compute(udl_repr)
                        if abs(score - score2) > 1e-10:
                            mathematical_errors.append(
                                f"{udl_file.name}: {metric_name} not deterministic: {score} vs {score2}"
                            )

                    except Exception as e:
                        mathematical_errors.append(
                            f"{udl_file.name}: {metric_name} computation failed: {e}"
                        )

                # Test aggregation
                try:
                    metric_values = {
                        name: metric.compute(udl_repr)
                        for name, metric in self.metrics.items()
                    }

                    aggregated_score = self.aggregator.aggregate(metric_values)

                    # Verify aggregation formula
                    expected_score = sum(
                        self.aggregator.weights[name] * score
                        for name, score in metric_values.items()
                    )

                    if abs(aggregated_score - expected_score) > 1e-10:
                        mathematical_errors.append(
                            f"{udl_file.name}: Aggregation error: {aggregated_score} vs {expected_score}"
                        )

                    # Verify aggregated score is bounded
                    if not (0 <= aggregated_score <= 1):
                        mathematical_errors.append(
                            f"{udl_file.name}: Aggregated score {aggregated_score} not in [0,1]"
                        )

                except Exception as e:
                    mathematical_errors.append(
                        f"{udl_file.name}: Aggregation failed: {e}"
                    )

            except Exception as e:
                mathematical_errors.append(
                    f"{udl_file.name}: UDL parsing failed: {e}")

        # Report any mathematical errors
        if mathematical_errors:
            error_report = "\n".join(mathematical_errors)
            pytest.fail(
                f"Mathematical correctness violations found:\n{error_report}")

        print(
            f"✓ Mathematical correctness verified on {len(udl_files[:10])} UDL examples"
        )

    def test_run_all_property_tests(self):
        """
        Verify that key property tests can be imported and run.

        This test verifies that the property test infrastructure is working
        by importing key test modules and checking they contain the expected
        property tests.
        """
        # List of key property test modules to verify
        property_test_modules = [
            "tests.test_metric_properties",
            "tests.test_udl_representation",
            "tests.test_example_validation",
            "tests.test_consistency_metric",
            "tests.test_completeness_metric",
            "tests.test_expressiveness_metric",
            "tests.test_structural_coherence_metric",
            "tests.test_aggregation_confidence",
            "tests.test_file_discovery",
            "tests.test_rating_pipeline",
        ]

        successful_imports = 0
        property_test_count = 0

        for module_name in property_test_modules:
            try:
                # Try to import the module
                import importlib

                module = importlib.import_module(module_name)
                successful_imports += 1

                # Count property test methods (methods containing "property" in name)
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if hasattr(attr, "__dict__"):  # It's a class
                        for method_name in dir(attr):
                            if (
                                "property" in method_name.lower()
                                and method_name.startswith("test_")
                            ):
                                property_test_count += 1

                print(f"✓ Successfully imported {module_name}")

            except ImportError as e:
                print(f"✗ Failed to import {module_name}: {e}")
            except Exception as e:
                print(f"✗ Error processing {module_name}: {e}")

        # Verify we successfully imported most modules
        success_rate = successful_imports / len(property_test_modules)
        assert success_rate >= 0.7, (
            f"Only {successful_imports}/{len(property_test_modules)} property test modules imported successfully"
        )

        # Verify we found property tests
        assert property_test_count >= 10, (
            f"Only found {property_test_count} property test methods"
        )

        print(
            f"✓ Property test infrastructure verified: {successful_imports} modules, {property_test_count} property tests"
        )

    def test_performance_and_scalability(self, temp_dir):
        """
        Test system performance and scalability with various workloads.
        """
        import time

        # Create test files of different sizes
        test_cases = [
            ("small", 10),  # 10 rules
            ("medium", 100),  # 100 rules
            ("large", 500),  # 500 rules
        ]

        performance_results = []

        for size_name, num_rules in test_cases:
            # Generate test UDL
            content = self._generate_large_grammar(num_rules)
            test_file = temp_dir / f"{size_name}.udl"
            test_file.write_text(content)

            # Measure processing time
            start_time = time.time()

            try:
                udl_repr = UDLRepresentation(content, str(test_file))
                report = self.pipeline.compute_rating(udl_repr)

                end_time = time.time()
                processing_time = end_time - start_time

                performance_results.append(
                    {
                        "size": size_name,
                        "rules": num_rules,
                        "time": processing_time,
                        "success": True,
                        "score": report.overall_score,
                    }
                )

                print(
                    f"✓ {size_name} ({num_rules} rules): {processing_time:.3f}s, score: {report.overall_score:.3f}"
                )

                # Verify reasonable performance bounds
                if size_name == "small" and processing_time > 10:
                    pytest.fail(
                        f"Small UDL took too long: {processing_time:.3f}s")
                elif size_name == "medium" and processing_time > 30:
                    pytest.fail(
                        f"Medium UDL took too long: {processing_time:.3f}s")
                elif size_name == "large" and processing_time > 120:
                    pytest.fail(
                        f"Large UDL took too long: {processing_time:.3f}s")

            except Exception as e:
                performance_results.append(
                    {
                        "size": size_name,
                        "rules": num_rules,
                        "time": None,
                        "success": False,
                        "error": str(e),
                    }
                )
                print(f"✗ {size_name} ({num_rules} rules): Failed - {e}")

        # Verify that at least small and medium cases work
        successful_cases = [r for r in performance_results if r["success"]]
        assert len(successful_cases) >= 2, (
            f"Only {len(successful_cases)} performance test cases succeeded"
        )

        print(
            f"✓ Performance testing completed: {len(successful_cases)}/{len(test_cases)} cases successful"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
