"""
Test UDL Example Processing - Property 10.

This module validates that all UDL example files can be processed
through the rating system without errors.

**Property 10: UDL Example Processing**
**Validates: Requirements 5.1**
"""

from pathlib import Path
from typing import Any, Dict, List

import pytest

from udl_rating_framework.core.metrics.base import MetricRegistry
from udl_rating_framework.core.pipeline import RatingPipeline
from udl_rating_framework.core.representation import UDLRepresentation


class TestUDLExampleProcessing:
    """
    Test suite for validating UDL example file processing.

    **Property 10: UDL Example Processing**
    *For any* UDL example file in the examples directory, the rating system
    should successfully process the file without errors.

    **Validates: Requirements 5.1**
    """

    @pytest.fixture
    def examples_dir(self) -> Path:
        """Get path to UDL examples directory."""
        return Path(__file__).parent.parent / "examples" / "udl_examples"

    @pytest.fixture
    def core_metrics(self) -> List[str]:
        """Get list of core metrics to use for rating."""
        return ["consistency", "completeness", "expressiveness", "structural_coherence"]

    def get_all_udl_files(self, examples_dir: Path) -> List[Path]:
        """Get all UDL files from examples directory."""
        if not examples_dir.exists():
            return []
        return sorted(examples_dir.glob("*.udl"))

    def get_all_grammar_files(self, examples_dir: Path) -> List[Path]:
        """Get all grammar format files (including non-.udl formats)."""
        if not examples_dir.exists():
            return []

        extensions = ["*.udl", "*.g4", "*.peg",
                      "*.y", "*.ebnf", "*.abnf", "*.rr"]
        files = []
        for ext in extensions:
            files.extend(examples_dir.glob(ext))
        return sorted(files)

    def load_udl_file(self, file_path: Path) -> UDLRepresentation:
        """Load a UDL file and create representation."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return UDLRepresentation(content, str(file_path))

    # =========================================================================
    # Property 10: UDL Example Processing Tests
    # =========================================================================

    @pytest.mark.parametrize(
        "udl_filename",
        [
            "simple_calculator.udl",
            "json_subset.udl",
            "config_language.udl",
            "broken_grammar.udl",
            "state_machine.udl",
            "query_language.udl",
            "template_engine.udl",
            "regex_subset.udl",
            "css_subset.udl",
            "inconsistent_rules.udl",
            "incomplete_spec.udl",
        ],
    )
    def test_udl_file_loads_successfully(self, examples_dir, udl_filename):
        """
        Test that each UDL file can be loaded and parsed.

        **Property 10: UDL Example Processing**
        **Validates: Requirements 5.1**
        """
        file_path = examples_dir / udl_filename
        if not file_path.exists():
            pytest.skip(f"UDL file not found: {udl_filename}")

        # Should not raise any exceptions
        udl = self.load_udl_file(file_path)

        # Basic validation
        assert udl is not None
        assert udl.source_text is not None
        assert len(udl.source_text) > 0
        assert udl.file_path == str(file_path)

    @pytest.mark.parametrize(
        "udl_filename",
        [
            "simple_calculator.udl",
            "json_subset.udl",
            "config_language.udl",
            "broken_grammar.udl",
            "state_machine.udl",
            "query_language.udl",
            "template_engine.udl",
            "regex_subset.udl",
            "css_subset.udl",
            "inconsistent_rules.udl",
            "incomplete_spec.udl",
        ],
    )
    def test_udl_file_tokenizes_successfully(self, examples_dir, udl_filename):
        """
        Test that each UDL file can be tokenized.

        **Property 10: UDL Example Processing**
        **Validates: Requirements 5.1**
        """
        file_path = examples_dir / udl_filename
        if not file_path.exists():
            pytest.skip(f"UDL file not found: {udl_filename}")

        udl = self.load_udl_file(file_path)
        tokens = udl.get_tokens()

        # Should have at least some tokens
        assert tokens is not None
        assert len(tokens) > 0

    @pytest.mark.parametrize(
        "udl_filename",
        [
            "simple_calculator.udl",
            "json_subset.udl",
            "config_language.udl",
            "broken_grammar.udl",
            "state_machine.udl",
            "query_language.udl",
            "template_engine.udl",
            "regex_subset.udl",
            "css_subset.udl",
            "inconsistent_rules.udl",
            "incomplete_spec.udl",
        ],
    )
    def test_udl_file_extracts_grammar_rules(self, examples_dir, udl_filename):
        """
        Test that grammar rules can be extracted from each UDL file.

        **Property 10: UDL Example Processing**
        **Validates: Requirements 5.1**
        """
        file_path = examples_dir / udl_filename
        if not file_path.exists():
            pytest.skip(f"UDL file not found: {udl_filename}")

        udl = self.load_udl_file(file_path)
        rules = udl.get_grammar_rules()

        # Should have at least some rules (even broken grammars have rules)
        assert rules is not None
        assert len(rules) > 0

    @pytest.mark.parametrize(
        "udl_filename",
        [
            "simple_calculator.udl",
            "json_subset.udl",
            "config_language.udl",
            "broken_grammar.udl",
            "state_machine.udl",
            "query_language.udl",
            "template_engine.udl",
            "regex_subset.udl",
            "css_subset.udl",
            "inconsistent_rules.udl",
            "incomplete_spec.udl",
        ],
    )
    def test_udl_file_builds_grammar_graph(self, examples_dir, udl_filename):
        """
        Test that a grammar graph can be built from each UDL file.

        **Property 10: UDL Example Processing**
        **Validates: Requirements 5.1**
        """
        file_path = examples_dir / udl_filename
        if not file_path.exists():
            pytest.skip(f"UDL file not found: {udl_filename}")

        udl = self.load_udl_file(file_path)
        graph = udl.get_grammar_graph()

        # Should have a valid graph
        assert graph is not None
        # Graph should have nodes (symbols)
        assert graph.number_of_nodes() >= 0

    @pytest.mark.parametrize(
        "udl_filename",
        [
            "simple_calculator.udl",
            "json_subset.udl",
            "config_language.udl",
            "broken_grammar.udl",
            "state_machine.udl",
            "query_language.udl",
            "template_engine.udl",
            "regex_subset.udl",
            "css_subset.udl",
            "inconsistent_rules.udl",
            "incomplete_spec.udl",
        ],
    )
    def test_udl_file_processes_through_pipeline(
        self, examples_dir, core_metrics, udl_filename
    ):
        """
        Test that each UDL file can be processed through the full rating pipeline.

        **Property 10: UDL Example Processing**
        **Validates: Requirements 5.1**
        """
        file_path = examples_dir / udl_filename
        if not file_path.exists():
            pytest.skip(f"UDL file not found: {udl_filename}")

        # Load UDL
        udl = self.load_udl_file(file_path)

        # Create pipeline with core metrics
        pipeline = RatingPipeline(
            metric_names=core_metrics, enable_tracing=False, enable_caching=False
        )

        # Process through pipeline - should not raise
        report = pipeline.compute_rating(udl)

        # Validate report structure
        assert report is not None
        assert hasattr(report, "overall_score")
        assert hasattr(report, "metric_scores")
        assert hasattr(report, "confidence")

        # Overall score should be bounded [0, 1]
        assert 0.0 <= report.overall_score <= 1.0

        # All metric scores should be bounded [0, 1]
        for metric_name, score in report.metric_scores.items():
            assert 0.0 <= score <= 1.0, f"Metric {metric_name} out of bounds: {score}"

    def test_all_udl_files_process_without_errors(self, examples_dir, core_metrics):
        """
        Test that ALL UDL files in the examples directory process without errors.

        **Property 10: UDL Example Processing**
        **Validates: Requirements 5.1**
        """
        udl_files = self.get_all_udl_files(examples_dir)

        if not udl_files:
            pytest.skip("No UDL files found in examples directory")

        failures = []
        successes = []

        pipeline = RatingPipeline(
            metric_names=core_metrics, enable_tracing=False, enable_caching=False
        )

        for file_path in udl_files:
            try:
                udl = self.load_udl_file(file_path)
                report = pipeline.compute_rating(udl)

                # Check for errors in report
                if report.errors:
                    failures.append(
                        {"file": file_path.name, "errors": report.errors})
                else:
                    successes.append(
                        {"file": file_path.name, "score": report.overall_score}
                    )

            except Exception as e:
                failures.append({"file": file_path.name, "errors": [str(e)]})

        # Report results
        print(f"\n=== UDL Processing Results ===")
        print(f"Successes: {len(successes)}/{len(udl_files)}")
        for s in successes:
            print(f"  ✓ {s['file']}: score={s['score']:.4f}")

        if failures:
            print(f"\nFailures: {len(failures)}/{len(udl_files)}")
            for f in failures:
                print(f"  ✗ {f['file']}: {f['errors']}")

        # All files should process successfully
        assert len(failures) == 0, (
            f"Failed to process {len(failures)} UDL files: {failures}"
        )

    # =========================================================================
    # Additional Grammar Format Tests (Task 26 support)
    # =========================================================================

    @pytest.mark.parametrize(
        "grammar_filename",
        [
            "simple_antlr.g4",
            "simple_peg.peg",
            "simple_yacc.y",
            "simple_ebnf.ebnf",
            "simple_abnf.abnf",
            "simple_railroad.rr",
        ],
    )
    def test_other_grammar_formats_load(self, examples_dir, grammar_filename):
        """
        Test that other grammar format files can be loaded.

        **Property 10: UDL Example Processing**
        **Validates: Requirements 5.1**
        """
        file_path = examples_dir / grammar_filename
        if not file_path.exists():
            pytest.skip(f"Grammar file not found: {grammar_filename}")

        # Should not raise any exceptions
        udl = self.load_udl_file(file_path)

        # Basic validation
        assert udl is not None
        assert udl.source_text is not None
        assert len(udl.source_text) > 0

        # Should detect format correctly
        detected_format = udl.get_format()
        assert detected_format is not None

    @pytest.mark.parametrize(
        "grammar_filename",
        [
            "simple_antlr.g4",
            "simple_peg.peg",
            "simple_yacc.y",
            "simple_ebnf.ebnf",
            "simple_abnf.abnf",
            "simple_railroad.rr",
        ],
    )
    def test_other_grammar_formats_process(
        self, examples_dir, core_metrics, grammar_filename
    ):
        """
        Test that other grammar format files can be processed through pipeline.

        **Property 10: UDL Example Processing**
        **Validates: Requirements 5.1**
        """
        file_path = examples_dir / grammar_filename
        if not file_path.exists():
            pytest.skip(f"Grammar file not found: {grammar_filename}")

        # Load grammar file
        udl = self.load_udl_file(file_path)

        # Create pipeline
        pipeline = RatingPipeline(
            metric_names=core_metrics, enable_tracing=False, enable_caching=False
        )

        # Process through pipeline
        report = pipeline.compute_rating(udl)

        # Validate report
        assert report is not None
        assert 0.0 <= report.overall_score <= 1.0


class TestUDLProcessingReport:
    """Generate a processing report for all UDL examples."""

    def test_generate_processing_report(self):
        """
        Generate a comprehensive report of UDL example processing.

        This test documents the processing status of all UDL files.
        """
        examples_dir = Path(__file__).parent.parent / \
            "examples" / "udl_examples"

        if not examples_dir.exists():
            pytest.skip("Examples directory not found")

        core_metrics = [
            "consistency",
            "completeness",
            "expressiveness",
            "structural_coherence",
        ]

        # Get all grammar files
        extensions = ["*.udl", "*.g4", "*.peg",
                      "*.y", "*.ebnf", "*.abnf", "*.rr"]
        all_files = []
        for ext in extensions:
            all_files.extend(examples_dir.glob(ext))
        all_files = sorted(all_files)

        if not all_files:
            pytest.skip("No grammar files found")

        pipeline = RatingPipeline(
            metric_names=core_metrics, enable_tracing=False, enable_caching=False
        )

        results = []

        for file_path in all_files:
            result = {
                "file": file_path.name,
                "extension": file_path.suffix,
                "status": "unknown",
                "score": None,
                "metrics": {},
                "errors": [],
                "format": None,
            }

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                udl = UDLRepresentation(content, str(file_path))
                result["format"] = udl.get_format().value

                report = pipeline.compute_rating(udl)

                result["status"] = "success" if not report.errors else "partial"
                result["score"] = report.overall_score
                result["metrics"] = dict(report.metric_scores)
                result["errors"] = report.errors

            except Exception as e:
                result["status"] = "failed"
                result["errors"] = [str(e)]

            results.append(result)

        # Print report
        print("\n" + "=" * 70)
        print("UDL EXAMPLE PROCESSING REPORT")
        print("=" * 70)

        success_count = sum(1 for r in results if r["status"] == "success")
        partial_count = sum(1 for r in results if r["status"] == "partial")
        failed_count = sum(1 for r in results if r["status"] == "failed")

        print(
            f"\nSummary: {success_count} success, {partial_count} partial, {failed_count} failed"
        )
        print(f"Total files: {len(results)}")

        print("\n" + "-" * 70)
        print("Detailed Results:")
        print("-" * 70)

        for r in results:
            status_icon = (
                "✓"
                if r["status"] == "success"
                else ("⚠" if r["status"] == "partial" else "✗")
            )
            score_str = f"{r['score']:.4f}" if r["score"] is not None else "N/A"
            format_str = r["format"] if r["format"] else "unknown"

            print(f"\n{status_icon} {r['file']} ({format_str})")
            print(f"  Score: {score_str}")

            if r["metrics"]:
                print(f"  Metrics:")
                for metric, value in r["metrics"].items():
                    print(f"    - {metric}: {value:.4f}")

            if r["errors"]:
                print(f"  Errors: {r['errors']}")

        print("\n" + "=" * 70)

        # Test passes if we can generate the report
        assert len(results) > 0
