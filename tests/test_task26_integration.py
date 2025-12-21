"""
Integration tests for Task 26: Add more UDL format support.

Tests end-to-end functionality for new grammar formats:
- ANTLR grammar files (.g4)
- PEG grammar files (.peg)
- Yacc/Bison files (.y, .yacc)
- EBNF variants (ISO/IEC 14977)
- Railroad diagram formats
"""

import tempfile
from pathlib import Path
from udl_rating_framework.io.file_discovery import FileDiscovery
from udl_rating_framework.core.representation import UDLRepresentation, GrammarFormat
from udl_rating_framework.core.pipeline import RatingPipeline
from udl_rating_framework.core.metrics.consistency import ConsistencyMetric
from udl_rating_framework.core.metrics.completeness import CompletenessMetric


class TestTask26Integration:
    """Integration tests for new UDL format support."""

    def test_all_new_formats_discoverable(self):
        """Test that all new formats are discoverable by file discovery."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create files with all new format extensions
            new_format_files = {
                "test.g4": "grammar Test; expr : term ;",
                "test.peg": "expr <- term",
                "test.y": "%% expr : term ;",
                "test.yacc": "%% stmt : expr ;",
                "test.bnf": "expr ::= term",
                "test.abnf": "expr = term",
                "test.xbnf": "expr ::= term { '+' term }",
                "test.wsn": "expr = term.",
                "test.wirth": "expr = term.",
                "test.rr": "expr: term",
                "test.railroad": "stmt: expression",
            }

            for filename, content in new_format_files.items():
                file_path = temp_path / filename
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)

            # Test file discovery
            discovery = FileDiscovery()
            result = discovery.discover_files(str(temp_path))

            # Should discover all files
            assert len(result.discovered_files) == len(new_format_files)
            assert len(result.errors) == 0

            # Verify all expected extensions are found
            discovered_extensions = {f.suffix for f in result.discovered_files}
            expected_extensions = {Path(f).suffix for f in new_format_files.keys()}
            assert discovered_extensions == expected_extensions

    def test_format_detection_accuracy(self):
        """Test that format detection works correctly for all new formats."""
        test_cases = [
            ("test.g4", "grammar Test; expr : term ;", GrammarFormat.ANTLR),
            ("test.peg", "expr <- term", GrammarFormat.PEG),
            ("test.y", "%% expr : term ;", GrammarFormat.YACC_BISON),
            ("test.yacc", "%% stmt : expr ;", GrammarFormat.YACC_BISON),
            ("test.ebnf", "expr ::= term", GrammarFormat.EBNF),
            ("test.bnf", "expr ::= term", GrammarFormat.BNF),
            ("test.abnf", "expr = term", GrammarFormat.ABNF),
            ("test.xbnf", "expr ::= term", GrammarFormat.EBNF),
            ("test.wsn", "expr = term.", GrammarFormat.EBNF),
            ("test.rr", "expr: term", GrammarFormat.RAILROAD),
            ("test.railroad", "stmt: expression", GrammarFormat.RAILROAD),
        ]

        for filename, content, expected_format in test_cases:
            with tempfile.NamedTemporaryFile(
                suffix=Path(filename).suffix, mode="w", delete=False
            ) as f:
                f.write(content)
                f.flush()

                udl = UDLRepresentation(content, f.name)
                assert udl.get_format() == expected_format, (
                    f"Format detection failed for {filename}"
                )

            Path(f.name).unlink()

    def test_rule_extraction_for_all_formats(self):
        """Test that grammar rules can be extracted from all new formats."""
        test_cases = [
            (
                "test.g4",
                "grammar Test;\nexpr : term '+' factor ;\nterm : NUMBER ;",
                ["expr", "term"],
            ),
            ("test.peg", "Expr <- Term '+' Factor\nTerm <- NUMBER", ["Expr", "Term"]),
            (
                "test.y",
                "%% expr : term '+' factor ;\nterm : NUMBER ;",
                ["expr", "term"],
            ),
            (
                "test.ebnf",
                "expr ::= term '+' factor\nterm ::= NUMBER",
                ["expr", "term"],
            ),
            ("test.abnf", 'expr = term "+" factor\nterm = NUMBER', ["expr", "term"]),
            ("test.rr", "expr: term plus factor\nterm: number", ["expr", "term"]),
        ]

        for filename, content, expected_rules in test_cases:
            with tempfile.NamedTemporaryFile(
                suffix=Path(filename).suffix, mode="w", delete=False
            ) as f:
                f.write(content)
                f.flush()

                udl = UDLRepresentation(content, f.name)
                rules = udl.get_grammar_rules()

                # Should extract expected rules
                rule_names = {rule.lhs for rule in rules}
                for expected_rule in expected_rules:
                    assert expected_rule in rule_names, (
                        f"Rule {expected_rule} not found in {filename}"
                    )

            Path(f.name).unlink()

    def test_metrics_computation_for_new_formats(self):
        """Test that quality metrics can be computed for all new formats."""
        test_cases = [
            ("test.g4", "grammar Test;\nexpr : term '+' factor ;\nterm : NUMBER ;"),
            ("test.peg", "Expr <- Term '+' Factor\nTerm <- NUMBER"),
            ("test.y", "%% expr : term '+' factor ;\nterm : NUMBER ;"),
            ("test.ebnf", "expr ::= term '+' factor\nterm ::= NUMBER"),
            ("test.abnf", 'expr = term "+" factor\nterm = NUMBER'),
            ("test.rr", "expr: term plus factor\nterm: number"),
        ]

        consistency_metric = ConsistencyMetric()
        completeness_metric = CompletenessMetric()

        for filename, content in test_cases:
            with tempfile.NamedTemporaryFile(
                suffix=Path(filename).suffix, mode="w", delete=False
            ) as f:
                f.write(content)
                f.flush()

                udl = UDLRepresentation(content, f.name)

                # Should be able to compute metrics without errors
                consistency_score = consistency_metric.compute(udl)
                completeness_score = completeness_metric.compute(udl)

                # Scores should be valid (between 0 and 1)
                assert 0.0 <= consistency_score <= 1.0, (
                    f"Invalid consistency score for {filename}: {consistency_score}"
                )
                assert 0.0 <= completeness_score <= 1.0, (
                    f"Invalid completeness score for {filename}: {completeness_score}"
                )

            Path(f.name).unlink()

    def test_end_to_end_rating_pipeline(self):
        """Test complete rating pipeline with new formats."""
        test_cases = [
            (
                "antlr_test.g4",
                "grammar Test;\nexpr : term '+' factor ;\nterm : NUMBER ;",
            ),
            ("peg_test.peg", "Expr <- Term '+' Factor\nTerm <- NUMBER"),
            ("yacc_test.y", "%% expr : term '+' factor ;\nterm : NUMBER ;"),
            ("ebnf_test.ebnf", "expr ::= term '+' factor\nterm ::= NUMBER"),
        ]

        # Initialize rating pipeline
        metric_names = [
            "consistency",
            "completeness",
            "expressiveness",
            "structural_coherence",
        ]
        pipeline = RatingPipeline(metric_names)

        for filename, content in test_cases:
            with tempfile.NamedTemporaryFile(
                suffix=Path(filename).suffix, mode="w", delete=False
            ) as f:
                f.write(content)
                f.flush()

                # Run complete rating pipeline
                udl = UDLRepresentation(content, f.name)
                report = pipeline.compute_rating(udl)

                # Should produce valid report
                assert report is not None
                assert 0.0 <= report.overall_score <= 1.0
                assert 0.0 <= report.confidence <= 1.0
                assert len(report.metric_scores) > 0
                assert (
                    len(report.errors) == 0
                )  # Should not have errors for valid grammars

                # Should have computation trace
                assert len(report.computation_trace) > 0

            Path(f.name).unlink()

    def test_backward_compatibility(self):
        """Test that existing formats still work correctly."""
        existing_formats = [
            ("test.udl", "expr ::= term"),
            ("test.dsl", "stmt := assignment"),
            ("test.grammar", "rule pattern"),
            ("test.txt", "Simple grammar"),
        ]

        discovery = FileDiscovery()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            for filename, content in existing_formats:
                file_path = temp_path / filename
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)

            # Should still discover existing formats
            result = discovery.discover_files(str(temp_path))
            assert len(result.discovered_files) == len(existing_formats)
            assert len(result.errors) == 0

            # Should still be able to process them
            for discovered_file in result.discovered_files:
                with open(discovered_file, "r", encoding="utf-8") as f:
                    content = f.read()

                udl = UDLRepresentation(content, str(discovered_file))

                # Should be able to extract tokens and rules
                tokens = udl.get_tokens()
                rules = udl.get_grammar_rules()

                assert len(tokens) > 0
                assert rules is not None
                # Rules may be empty for simple formats, but should not error

    def test_mixed_format_directory(self):
        """Test processing a directory with mixed old and new formats."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create mix of old and new format files
            mixed_files = {
                "old1.udl": "expr ::= term",
                "old2.dsl": "stmt := assignment",
                "new1.g4": "grammar Test; expr : term ;",
                "new2.peg": "Expr <- Term",
                "new3.y": "%% expr : term ;",
                "new4.ebnf": "expr ::= term { '+' term }",
                "new5.abnf": 'expr = term "+" factor',
                "new6.rr": "expr: term plus factor",
                "ignored.py": "print('hello')",  # Should be ignored
            }

            for filename, content in mixed_files.items():
                file_path = temp_path / filename
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)

            # Test file discovery
            discovery = FileDiscovery()
            result = discovery.discover_files(str(temp_path))

            # Should discover all UDL files but not Python file
            expected_udl_files = [
                f for f in mixed_files.keys() if not f.endswith(".py")
            ]
            assert len(result.discovered_files) == len(expected_udl_files)
            assert len(result.errors) == 0

            # Should be able to process all discovered files
            metric_names = [
                "consistency",
                "completeness",
                "expressiveness",
                "structural_coherence",
            ]
            pipeline = RatingPipeline(metric_names)

            for discovered_file in result.discovered_files:
                with open(discovered_file, "r", encoding="utf-8") as f:
                    content = f.read()
                udl = UDLRepresentation(content, str(discovered_file))
                report = pipeline.compute_rating(udl)

                # Should produce valid report for each file
                assert report is not None
                assert 0.0 <= report.overall_score <= 1.0
                assert 0.0 <= report.confidence <= 1.0
                assert len(report.errors) == 0
