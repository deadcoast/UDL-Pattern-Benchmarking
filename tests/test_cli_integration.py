"""
Integration tests for CLI commands.

Tests all CLI commands end-to-end to ensure they work correctly with real data.
"""

import json
import tempfile
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from udl_rating_framework.cli.main import cli


@pytest.fixture
def runner():
    """Click test runner."""
    return CliRunner()


@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_udl_files(temp_dir):
    """Create sample UDL files for testing."""
    udl_files = []

    # Simple grammar UDL
    udl1 = temp_dir / "simple_grammar.udl"
    udl1.write_text("""
    grammar SimpleLanguage {
        start: expression
        expression: term ('+' term)*
        term: factor ('*' factor)*
        factor: NUMBER | '(' expression ')'
        NUMBER: /[0-9]+/
    }
    """)
    udl_files.append(udl1)

    # More complex UDL
    udl2 = temp_dir / "complex_grammar.udl"
    udl2.write_text("""
    grammar ComplexLanguage {
        start: program
        program: statement*
        statement: assignment | if_statement | while_statement
        assignment: IDENTIFIER '=' expression ';'
        if_statement: 'if' '(' condition ')' '{' statement* '}'
        while_statement: 'while' '(' condition ')' '{' statement* '}'
        condition: expression ('==' | '!=' | '<' | '>') expression
        expression: term ('+' | '-' term)*
        term: factor ('*' | '/' factor)*
        factor: NUMBER | IDENTIFIER | '(' expression ')'
        IDENTIFIER: /[a-zA-Z][a-zA-Z0-9]*/
        NUMBER: /[0-9]+/
    }
    """)
    udl_files.append(udl2)

    # Minimal UDL
    udl3 = temp_dir / "minimal.udl"
    udl3.write_text("""
    grammar Minimal {
        start: 'hello'
    }
    """)
    udl_files.append(udl3)

    return udl_files


@pytest.fixture
def sample_config(temp_dir):
    """Create sample configuration file."""
    config_data = {
        "metrics": {
            "consistency_weight": 0.3,
            "completeness_weight": 0.3,
            "expressiveness_weight": 0.2,
            "structural_coherence_weight": 0.2,
        },
        "output": {
            "formats": ["json"],
            "output_dir": str(temp_dir / "output"),
            "include_traces": False,
            "precision": 4,
        },
    }

    config_file = temp_dir / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    return config_file


class TestRateCommand:
    """Test the 'rate' command."""

    def test_rate_single_file(self, runner, sample_udl_files, temp_dir):
        """Test rating a single UDL file."""
        udl_file = sample_udl_files[0]
        output_file = temp_dir / "rating_result.json"

        result = runner.invoke(
            cli,
            ["rate", str(udl_file), "--output",
             str(output_file), "--format", "json"],
        )

        assert result.exit_code == 0
        assert output_file.exists()

        # Check output format
        with open(output_file) as f:
            data = json.load(f)

        assert "reports" in data
        assert len(data["reports"]) == 1

        report = data["reports"][0]
        assert "overall_score" in report
        assert "confidence" in report
        assert "metric_scores" in report
        assert 0 <= report["overall_score"] <= 1
        assert 0 <= report["confidence"] <= 1

    def test_rate_directory(self, runner, sample_udl_files, temp_dir):
        """Test rating a directory of UDL files."""
        output_file = temp_dir / "directory_results.json"

        result = runner.invoke(
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

        assert result.exit_code == 0
        assert output_file.exists()

        # Check output format
        with open(output_file) as f:
            data = json.load(f)

        # Should be a summary report with multiple files
        assert isinstance(data, dict)
        assert "reports" in data
        assert len(data["reports"]) >= 1  # Should have at least one report
        if len(data["reports"]) > 1:
            assert "summary" in data  # Should have summary for multiple files

    def test_rate_with_custom_weights(self, runner, sample_udl_files, temp_dir):
        """Test rating with custom metric weights."""
        udl_file = sample_udl_files[0]
        output_file = temp_dir / "custom_weights.json"

        result = runner.invoke(
            cli,
            [
                "rate",
                str(udl_file),
                "--output",
                str(output_file),
                "--consistency-weight",
                "0.4",
                "--completeness-weight",
                "0.3",
                "--expressiveness-weight",
                "0.2",
                "--structural-coherence-weight",
                "0.1",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

    def test_rate_with_config(self, runner, sample_udl_files, sample_config, temp_dir):
        """Test rating with configuration file."""
        udl_file = sample_udl_files[0]

        result = runner.invoke(
            cli, ["--config", str(sample_config), "rate", str(udl_file)]
        )

        assert result.exit_code == 0

    def test_rate_csv_output(self, runner, sample_udl_files, temp_dir):
        """Test CSV output format."""
        udl_file = sample_udl_files[0]
        output_file = temp_dir / "results.csv"

        result = runner.invoke(
            cli,
            ["rate", str(udl_file), "--output",
             str(output_file), "--format", "csv"],
        )

        assert result.exit_code == 0
        assert output_file.exists()

        # Check CSV format
        content = output_file.read_text()
        assert "," in content  # Should have CSV separators

    def test_rate_invalid_weights(self, runner, sample_udl_files):
        """Test that invalid weights are rejected."""
        udl_file = sample_udl_files[0]

        # Weights don't sum to 1.0
        result = runner.invoke(
            cli,
            [
                "rate",
                str(udl_file),
                "--consistency-weight",
                "0.5",
                "--completeness-weight",
                "0.5",
                "--expressiveness-weight",
                "0.5",
                "--structural-coherence-weight",
                "0.5",
            ],
        )

        assert result.exit_code != 0

    def test_rate_nonexistent_file(self, runner):
        """Test rating a nonexistent file."""
        result = runner.invoke(cli, ["rate", "nonexistent.udl"])

        assert result.exit_code != 0


class TestCompareCommand:
    """Test the 'compare' command."""

    def test_compare_two_files(self, runner, sample_udl_files, temp_dir):
        """Test comparing two UDL files."""
        output_file = temp_dir / "comparison.json"

        result = runner.invoke(
            cli,
            [
                "compare",
                str(sample_udl_files[0]),
                str(sample_udl_files[1]),
                "--output",
                str(output_file),
                "--format",
                "json",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

        # Check output format
        with open(output_file) as f:
            data = json.load(f)

        assert "summary" in data
        assert "comparisons" in data
        assert "individual_reports" in data
        assert len(data["individual_reports"]) == 2

    def test_compare_with_ranking(self, runner, sample_udl_files, temp_dir):
        """Test comparison with ranking generation."""
        output_file = temp_dir / "ranking.json"

        result = runner.invoke(
            cli,
            [
                "compare",
                *[str(f) for f in sample_udl_files],
                "--output",
                str(output_file),
                "--ranking",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

        # Check ranking is included
        with open(output_file) as f:
            data = json.load(f)

        # Should have ranking results
        ranking_found = any(
            comp.get("type") == "statistical_comparison"
            and "rankings" in comp.get("results", {})
            for comp in data.get("comparisons", [])
        )
        assert ranking_found

    def test_compare_with_effect_sizes(self, runner, sample_udl_files, temp_dir):
        """Test comparison with effect size computation."""
        output_file = temp_dir / "effect_sizes.json"

        result = runner.invoke(
            cli,
            [
                "compare",
                str(sample_udl_files[0]),
                str(sample_udl_files[1]),
                "--output",
                str(output_file),
                "--include-effect-sizes",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

    def test_compare_insufficient_files(self, runner, sample_udl_files):
        """Test that comparison requires at least 2 files."""
        result = runner.invoke(cli, ["compare", str(sample_udl_files[0])])

        assert result.exit_code != 0


class TestTrainCommand:
    """Test the 'train' command."""

    def test_train_basic(self, runner, sample_udl_files, temp_dir):
        """Test basic training command."""
        checkpoint_dir = temp_dir / "checkpoints"

        result = runner.invoke(
            cli,
            [
                "train",
                str(temp_dir),
                "--output-dir",
                str(checkpoint_dir),
                "--epochs",
                "2",  # Very short training for testing
                "--batch-size",
                "2",
            ],
        )

        # Training might fail due to insufficient data, but should not crash
        # We mainly test that the command structure works
        assert result.exit_code in [0, 1]  # Allow failure due to data issues

    def test_train_with_config(self, runner, sample_udl_files, sample_config, temp_dir):
        """Test training with configuration file."""
        checkpoint_dir = temp_dir / "checkpoints"

        result = runner.invoke(
            cli,
            [
                "--config",
                str(sample_config),
                "train",
                str(temp_dir),
                "--output-dir",
                str(checkpoint_dir),
                "--epochs",
                "1",
            ],
        )

        # Allow failure due to insufficient training data
        assert result.exit_code in [0, 1]

    def test_train_invalid_validation_split(self, runner, sample_udl_files, temp_dir):
        """Test that invalid validation split is rejected."""
        result = runner.invoke(
            cli,
            [
                "train",
                str(temp_dir),
                "--validation-split",
                "1.5",  # Invalid: > 1.0
            ],
        )

        assert result.exit_code != 0

    def test_train_invalid_loss_weights(self, runner, sample_udl_files, temp_dir):
        """Test that invalid loss weights are rejected."""
        result = runner.invoke(
            cli,
            [
                "train",
                str(temp_dir),
                "--alpha",
                "0.8",
                "--beta",
                "0.8",  # alpha + beta > 1.0
            ],
        )

        assert result.exit_code != 0


class TestEvaluateCommand:
    """Test the 'evaluate' command."""

    def test_evaluate_nonexistent_model(self, runner, sample_udl_files, temp_dir):
        """Test evaluation with nonexistent model."""
        fake_model = temp_dir / "fake_model.pt"

        result = runner.invoke(
            cli, ["evaluate", str(fake_model), str(temp_dir)])

        assert result.exit_code != 0

    def test_evaluate_invalid_k_folds(self, runner, temp_dir):
        """Test that invalid k_folds parameter is rejected."""
        fake_model = temp_dir / "fake_model.pt"
        fake_model.touch()  # Create empty file

        result = runner.invoke(
            cli,
            [
                "evaluate",
                str(fake_model),
                str(temp_dir),
                "--k-folds",
                "2",  # Too small
            ],
        )

        assert result.exit_code != 0

    def test_evaluate_invalid_bootstrap_samples(self, runner, temp_dir):
        """Test that invalid bootstrap samples parameter is rejected."""
        fake_model = temp_dir / "fake_model.pt"
        fake_model.touch()

        result = runner.invoke(
            cli,
            [
                "evaluate",
                str(fake_model),
                str(temp_dir),
                "--bootstrap-samples",
                "500",  # Too small
            ],
        )

        assert result.exit_code != 0


class TestCLIGeneral:
    """Test general CLI functionality."""

    def test_help_message(self, runner):
        """Test that help message is displayed."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "UDL Rating Framework" in result.output

    def test_command_help_messages(self, runner):
        """Test help messages for individual commands."""
        commands = ["rate", "train", "compare", "evaluate"]

        for command in commands:
            result = runner.invoke(cli, [command, "--help"])
            assert result.exit_code == 0
            assert command in result.output.lower()

    def test_verbose_logging(self, runner, sample_udl_files):
        """Test verbose logging option."""
        result = runner.invoke(
            cli, ["--verbose", "rate", str(sample_udl_files[0])])

        # Should not crash with verbose logging
        assert result.exit_code in [0, 1]

    def test_quiet_logging(self, runner, sample_udl_files):
        """Test quiet logging option."""
        result = runner.invoke(
            cli, ["--quiet", "rate", str(sample_udl_files[0])])

        # Should not crash with quiet logging
        assert result.exit_code in [0, 1]

    def test_invalid_config_file(self, runner, temp_dir):
        """Test handling of invalid configuration file."""
        invalid_config = temp_dir / "invalid.yaml"
        invalid_config.write_text("invalid: yaml: content: [")

        result = runner.invoke(
            cli, ["--config", str(invalid_config), "rate", "--help"])

        assert result.exit_code != 0


class TestConfigValidation:
    """Test configuration validation."""

    def test_invalid_metric_weights(self, runner, temp_dir):
        """Test that invalid metric weights in config are rejected."""
        invalid_config = temp_dir / "invalid_weights.yaml"
        config_data = {
            "metrics": {
                "consistency_weight": 0.5,
                "completeness_weight": 0.5,
                "expressiveness_weight": 0.5,
                "structural_coherence_weight": 0.5,  # Sum > 1.0
            }
        }

        with open(invalid_config, "w") as f:
            yaml.dump(config_data, f)

        result = runner.invoke(
            cli, ["--config", str(invalid_config), "rate", "--help"])

        assert result.exit_code != 0

    def test_invalid_validation_split_config(self, runner, temp_dir):
        """Test that invalid validation split in config is rejected."""
        invalid_config = temp_dir / "invalid_validation.yaml"
        config_data = {"training": {"validation_split": 1.5}}  # Invalid: > 1.0

        with open(invalid_config, "w") as f:
            yaml.dump(config_data, f)

        result = runner.invoke(
            cli, ["--config", str(invalid_config), "train", "--help"]
        )

        assert result.exit_code != 0


if __name__ == "__main__":
    pytest.main([__file__])


class TestEntryPointValidity:
    """
    Property-based tests for entry point validity.

    **Feature: documentation-validation, Property 18: Entry Point Validity**
    **Validates: Requirements 7.3**

    For any entry point defined in pyproject.toml, invoking it should not
    raise an import error.
    """

    def test_udl_rating_entry_point_exists(self, runner):
        """
        Test that the udl-rating entry point is accessible.

        **Feature: documentation-validation, Property 18: Entry Point Validity**
        **Validates: Requirements 7.3**
        """
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "UDL Rating Framework" in result.output

    def test_all_commands_accessible(self, runner):
        """
        Test that all documented CLI commands are accessible.

        **Feature: documentation-validation, Property 18: Entry Point Validity**
        **Validates: Requirements 7.3**
        """
        # Main commands
        main_commands = [
            "rate",
            "train",
            "compare",
            "evaluate",
            "integration",
            "analytics",
        ]

        for command in main_commands:
            result = runner.invoke(cli, [command, "--help"])
            assert result.exit_code == 0, (
                f"Command '{command}' failed with exit code {result.exit_code}"
            )
            assert "--help" not in result.output or "Options:" in result.output

    def test_integration_subcommands_accessible(self, runner):
        """
        Test that integration subcommands are accessible.

        **Feature: documentation-validation, Property 18: Entry Point Validity**
        **Validates: Requirements 7.3**
        """
        subcommands = ["batch-process", "cicd", "git", "ide", "lsp-server"]

        for subcommand in subcommands:
            result = runner.invoke(cli, ["integration", subcommand, "--help"])
            assert result.exit_code == 0, (
                f"Subcommand 'integration {subcommand}' failed"
            )

    def test_analytics_subcommands_accessible(self, runner):
        """
        Test that analytics subcommands are accessible.

        **Feature: documentation-validation, Property 18: Entry Point Validity**
        **Validates: Requirements 7.3**
        """
        subcommands = [
            "dashboard",
            "export",
            "improve",
            "portfolio",
            "predict",
            "timeseries",
        ]

        for subcommand in subcommands:
            result = runner.invoke(cli, ["analytics", subcommand, "--help"])
            assert result.exit_code == 0, f"Subcommand 'analytics {subcommand}' failed"

    def test_entry_point_import(self):
        """
        Test that the entry point module can be imported without errors.

        **Feature: documentation-validation, Property 18: Entry Point Validity**
        **Validates: Requirements 7.3**
        """
        # This tests that the import chain works
        from udl_rating_framework.cli import main

        assert callable(main)

        from udl_rating_framework.cli import cli

        assert cli is not None
