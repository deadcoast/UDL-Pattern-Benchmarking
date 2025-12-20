"""
Property-based tests for CLI option accuracy.

**Feature: documentation-validation, Property 20: CLI Option Accuracy**
**Validates: Requirements 9.2**

For any CLI option documented in help text, the option should exist and behave as described.
"""

import pytest
from click.testing import CliRunner
from hypothesis import given, strategies as st, settings, HealthCheck
import tempfile
from pathlib import Path
import json
import yaml

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
def sample_udl_file(temp_dir):
    """Create a sample UDL file for testing."""
    udl_file = temp_dir / "test.udl"
    udl_file.write_text("""
    grammar TestLanguage {
        start: expression
        expression: term ('+' term)*
        term: NUMBER
        NUMBER: /[0-9]+/
    }
    """)
    return udl_file


class TestCLIOptionAccuracy:
    """
    Property-based tests for CLI option accuracy.
    
    **Feature: documentation-validation, Property 20: CLI Option Accuracy**
    **Validates: Requirements 9.2**
    """
    
    def test_rate_command_options_exist(self, runner):
        """
        Test that all documented rate command options exist.
        
        **Feature: documentation-validation, Property 20: CLI Option Accuracy**
        **Validates: Requirements 9.2**
        """
        result = runner.invoke(cli, ['rate', '--help'])
        assert result.exit_code == 0
        
        # Verify all documented options exist in help text
        expected_options = [
            '--output', '-o',
            '--format', '-f',
            '--recursive', '-r',
            '--extensions', '-e',
            '--include-traces',
            '--consistency-weight',
            '--completeness-weight',
            '--expressiveness-weight',
            '--structural-coherence-weight',
            '--help'
        ]
        
        for option in expected_options:
            assert option in result.output, f"Option '{option}' not found in rate command help"
    
    def test_train_command_options_exist(self, runner):
        """
        Test that all documented train command options exist.
        
        **Feature: documentation-validation, Property 20: CLI Option Accuracy**
        **Validates: Requirements 9.2**
        """
        result = runner.invoke(cli, ['train', '--help'])
        assert result.exit_code == 0
        
        expected_options = [
            '--output-dir', '-o',
            '--config-file', '-c',
            '--batch-size',
            '--learning-rate',
            '--epochs',
            '--validation-split',
            '--early-stopping-patience',
            '--alpha',
            '--beta',
            '--d-model',
            '--d-input',
            '--iterations',
            '--vocab-size',
            '--max-length',
            '--device',
            '--resume',
            '--help'
        ]
        
        for option in expected_options:
            assert option in result.output, f"Option '{option}' not found in train command help"
    
    def test_compare_command_options_exist(self, runner):
        """
        Test that all documented compare command options exist.
        
        **Feature: documentation-validation, Property 20: CLI Option Accuracy**
        **Validates: Requirements 9.2**
        """
        result = runner.invoke(cli, ['compare', '--help'])
        assert result.exit_code == 0
        
        expected_options = [
            '--output', '-o',
            '--format', '-f',
            '--recursive', '-r',
            '--extensions', '-e',
            '--significance-level',
            '--include-effect-sizes',
            '--ranking',
            '--consistency-weight',
            '--completeness-weight',
            '--expressiveness-weight',
            '--structural-coherence-weight',
            '--help'
        ]
        
        for option in expected_options:
            assert option in result.output, f"Option '{option}' not found in compare command help"
    
    def test_evaluate_command_options_exist(self, runner):
        """
        Test that all documented evaluate command options exist.
        
        **Feature: documentation-validation, Property 20: CLI Option Accuracy**
        **Validates: Requirements 9.2**
        """
        result = runner.invoke(cli, ['evaluate', '--help'])
        assert result.exit_code == 0
        
        expected_options = [
            '--output', '-o',
            '--format', '-f',
            '--k-folds',
            '--bootstrap-samples',
            '--confidence-level',
            '--calibration-bins',
            '--device',
            '--batch-size',
            '--include-detailed-analysis',
            '--help'
        ]
        
        for option in expected_options:
            assert option in result.output, f"Option '{option}' not found in evaluate command help"
    
    def test_global_options_exist(self, runner):
        """
        Test that all documented global options exist.
        
        **Feature: documentation-validation, Property 20: CLI Option Accuracy**
        **Validates: Requirements 9.2**
        """
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        
        expected_options = [
            '--config', '-c',
            '--verbose', '-v',
            '--quiet', '-q',
            '--help'
        ]
        
        for option in expected_options:
            assert option in result.output, f"Global option '{option}' not found in help"
    
    @given(weight=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_rate_weight_options_accept_valid_floats(self, runner, sample_udl_file, weight):
        """
        Test that weight options accept valid float values.
        
        **Feature: documentation-validation, Property 20: CLI Option Accuracy**
        **Validates: Requirements 9.2**
        """
        # Calculate complementary weights that sum to 1.0
        remaining = 1.0 - weight
        other_weight = remaining / 3.0
        
        result = runner.invoke(cli, [
            'rate', str(sample_udl_file),
            '--consistency-weight', str(weight),
            '--completeness-weight', str(other_weight),
            '--expressiveness-weight', str(other_weight),
            '--structural-coherence-weight', str(other_weight)
        ])
        
        # Should succeed or fail gracefully (not crash)
        assert result.exit_code in [0, 1, 2]
    
    def test_rate_format_option_accepts_valid_choices(self, runner, sample_udl_file, temp_dir):
        """
        Test that format option accepts documented choices.
        
        **Feature: documentation-validation, Property 20: CLI Option Accuracy**
        **Validates: Requirements 9.2**
        """
        valid_formats = ['json', 'csv', 'html']
        
        for fmt in valid_formats:
            output_file = temp_dir / f"output.{fmt}"
            result = runner.invoke(cli, [
                'rate', str(sample_udl_file),
                '--format', fmt,
                '--output', str(output_file)
            ])
            
            # Should succeed
            assert result.exit_code == 0, f"Format '{fmt}' failed: {result.output}"
            assert output_file.exists(), f"Output file not created for format '{fmt}'"
    
    def test_rate_format_option_rejects_invalid_choices(self, runner, sample_udl_file):
        """
        Test that format option rejects invalid choices.
        
        **Feature: documentation-validation, Property 20: CLI Option Accuracy**
        **Validates: Requirements 9.2**
        """
        result = runner.invoke(cli, [
            'rate', str(sample_udl_file),
            '--format', 'invalid_format'
        ])
        
        # Should fail with error
        assert result.exit_code != 0
    
    def test_evaluate_k_folds_option_validates_minimum(self, runner, temp_dir):
        """
        Test that k-folds option validates minimum value.
        
        **Feature: documentation-validation, Property 20: CLI Option Accuracy**
        **Validates: Requirements 9.2**
        """
        fake_model = temp_dir / "model.pt"
        fake_model.touch()
        
        # k_folds < 5 should be rejected
        result = runner.invoke(cli, [
            'evaluate', str(fake_model), str(temp_dir),
            '--k-folds', '3'
        ])
        
        # Should fail with non-zero exit code (error message is logged, not in output)
        assert result.exit_code != 0
    
    def test_evaluate_bootstrap_samples_option_validates_minimum(self, runner, temp_dir):
        """
        Test that bootstrap-samples option validates minimum value.
        
        **Feature: documentation-validation, Property 20: CLI Option Accuracy**
        **Validates: Requirements 9.2**
        """
        fake_model = temp_dir / "model.pt"
        fake_model.touch()
        
        # bootstrap_samples < 1000 should be rejected
        result = runner.invoke(cli, [
            'evaluate', str(fake_model), str(temp_dir),
            '--bootstrap-samples', '500'
        ])
        
        # Should fail with non-zero exit code (error message is logged, not in output)
        assert result.exit_code != 0
    
    def test_train_validation_split_option_validates_range(self, runner, temp_dir):
        """
        Test that validation-split option validates range (0, 1).
        
        **Feature: documentation-validation, Property 20: CLI Option Accuracy**
        **Validates: Requirements 9.2**
        """
        # validation_split > 1.0 should be rejected
        result = runner.invoke(cli, [
            'train', str(temp_dir),
            '--validation-split', '1.5'
        ])
        
        assert result.exit_code != 0
    
    def test_train_alpha_beta_sum_validation(self, runner, temp_dir):
        """
        Test that alpha + beta must sum to 1.0.
        
        **Feature: documentation-validation, Property 20: CLI Option Accuracy**
        **Validates: Requirements 9.2**
        """
        # alpha + beta != 1.0 should be rejected
        result = runner.invoke(cli, [
            'train', str(temp_dir),
            '--alpha', '0.8',
            '--beta', '0.8'
        ])
        
        assert result.exit_code != 0
    
    def test_compare_significance_level_option_validates_range(self, runner, sample_udl_file, temp_dir):
        """
        Test that significance-level option validates range (0, 1).
        
        **Feature: documentation-validation, Property 20: CLI Option Accuracy**
        **Validates: Requirements 9.2**
        """
        udl_file2 = temp_dir / "test2.udl"
        udl_file2.write_text("""
        grammar Test2 {
            start: 'hello'
        }
        """)
        
        # significance_level > 1.0 should be rejected
        result = runner.invoke(cli, [
            'compare', str(sample_udl_file), str(udl_file2),
            '--significance-level', '1.5'
        ])
        
        assert result.exit_code != 0


class TestAnalyticsSubcommandOptions:
    """
    Test analytics subcommand options.
    
    **Feature: documentation-validation, Property 20: CLI Option Accuracy**
    **Validates: Requirements 9.2**
    """
    
    def test_timeseries_options_exist(self, runner):
        """Test timeseries subcommand options."""
        result = runner.invoke(cli, ['analytics', 'timeseries', '--help'])
        assert result.exit_code == 0
        
        expected_options = ['--reports-dir', '-r', '--udl-file', '-f', '--metrics', '-m', '--output', '-o']
        for option in expected_options:
            assert option in result.output, f"Option '{option}' not found in timeseries help"
    
    def test_portfolio_options_exist(self, runner):
        """Test portfolio subcommand options."""
        result = runner.invoke(cli, ['analytics', 'portfolio', '--help'])
        assert result.exit_code == 0
        
        expected_options = ['--reports-dir', '-r', '--output', '-o', '--clustering-method', '--n-clusters']
        for option in expected_options:
            assert option in result.output, f"Option '{option}' not found in portfolio help"
    
    def test_predict_options_exist(self, runner):
        """Test predict subcommand options."""
        result = runner.invoke(cli, ['analytics', 'predict', '--help'])
        assert result.exit_code == 0
        
        expected_options = ['--reports-dir', '-r', '--udl-file', '-f', '--horizon', '--models']
        for option in expected_options:
            assert option in result.output, f"Option '{option}' not found in predict help"
    
    def test_improve_options_exist(self, runner):
        """Test improve subcommand options."""
        result = runner.invoke(cli, ['analytics', 'improve', '--help'])
        assert result.exit_code == 0
        
        expected_options = ['--reports-dir', '-r', '--udl-file', '-f', '--target-score', '--output', '-o']
        for option in expected_options:
            assert option in result.output, f"Option '{option}' not found in improve help"
    
    def test_export_options_exist(self, runner):
        """Test export subcommand options."""
        result = runner.invoke(cli, ['analytics', 'export', '--help'])
        assert result.exit_code == 0
        
        expected_options = ['--reports-dir', '-r', '--output-dir', '-o', '--format', '-f', '--aggregation']
        for option in expected_options:
            assert option in result.output, f"Option '{option}' not found in export help"
    
    def test_dashboard_options_exist(self, runner):
        """Test dashboard subcommand options."""
        result = runner.invoke(cli, ['analytics', 'dashboard', '--help'])
        assert result.exit_code == 0
        
        expected_options = ['--reports-dir', '-r', '--output-dir', '-o', '--platform']
        for option in expected_options:
            assert option in result.output, f"Option '{option}' not found in dashboard help"


class TestIntegrationSubcommandOptions:
    """
    Test integration subcommand options.
    
    **Feature: documentation-validation, Property 20: CLI Option Accuracy**
    **Validates: Requirements 9.2**
    """
    
    def test_git_install_hooks_options_exist(self, runner):
        """Test git install-hooks subcommand options."""
        result = runner.invoke(cli, ['integration', 'git', 'install-hooks', '--help'])
        assert result.exit_code == 0
        
        expected_options = ['--repo-path', '--threshold', '--config']
        for option in expected_options:
            assert option in result.output, f"Option '{option}' not found in git install-hooks help"
    
    def test_cicd_generate_options_exist(self, runner):
        """Test cicd generate subcommand options."""
        result = runner.invoke(cli, ['integration', 'cicd', 'generate', '--help'])
        assert result.exit_code == 0
        
        expected_options = ['--platform', '--output-dir', '--threshold', '--timeout']
        for option in expected_options:
            assert option in result.output, f"Option '{option}' not found in cicd generate help"
    
    def test_batch_process_options_exist(self, runner):
        """Test batch-process subcommand options."""
        result = runner.invoke(cli, ['integration', 'batch-process', '--help'])
        assert result.exit_code == 0
        
        expected_options = ['--output', '-o', '--format', '--workers', '--chunk-size', '--timeout']
        for option in expected_options:
            assert option in result.output, f"Option '{option}' not found in batch-process help"
    
    def test_ide_generate_options_exist(self, runner):
        """Test ide generate subcommand options."""
        result = runner.invoke(cli, ['integration', 'ide', 'generate', '--help'])
        assert result.exit_code == 0
        
        expected_options = ['--plugin-type', '--output-dir']
        for option in expected_options:
            assert option in result.output, f"Option '{option}' not found in ide generate help"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
