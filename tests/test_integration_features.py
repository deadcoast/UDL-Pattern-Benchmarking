"""
Tests for integration and workflow features.

Tests Git hooks, CI/CD integration, LSP server, batch processing, and IDE plugins.
"""

import pytest
import tempfile
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch

from udl_rating_framework.integration.git_hooks import GitHookManager
from udl_rating_framework.integration.cicd import CICDIntegration, CICDConfig
from udl_rating_framework.integration.lsp_server import UDLLanguageServer, LSPServer
from udl_rating_framework.integration.batch_processor import BatchProcessor, BatchConfig
from udl_rating_framework.integration.ide_plugin import IDEPluginManager


class TestGitHooks:
    """Test Git hooks integration."""

    def test_git_hook_manager_initialization(self):
        """Test GitHookManager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)

            manager = GitHookManager(
                repo_path=repo_path, min_quality_threshold=0.8, config_file=None
            )

            assert manager.repo_path == repo_path
            assert manager.min_quality_threshold == 0.8
            assert manager.git_hooks_path == repo_path / ".git" / "hooks"

    def test_hook_template_generation(self):
        """Test hook template generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            manager = GitHookManager(repo_path=repo_path)

            pre_commit_template = manager._get_pre_commit_template()
            pre_push_template = manager._get_pre_push_template()

            assert "UDL_RATING_HOOK" in pre_commit_template
            assert "UDL_RATING_HOOK" in pre_push_template
            assert "check_staged_files" in pre_commit_template
            assert "check_all_udl_files" in pre_push_template

    @patch("subprocess.run")
    def test_get_staged_udl_files(self, mock_run):
        """Test getting staged UDL files."""
        mock_run.return_value = Mock(
            stdout="file1.udl\nfile2.dsl\nfile3.txt\n", returncode=0
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            manager = GitHookManager(repo_path=repo_path)

            staged_files = manager._get_staged_udl_files()

            assert len(staged_files) == 2  # Only .udl and .dsl files
            assert any(f.name == "file1.udl" for f in staged_files)
            assert any(f.name == "file2.dsl" for f in staged_files)

    def test_install_uninstall_hooks(self):
        """Test hook installation and uninstallation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            git_dir = repo_path / ".git"
            git_dir.mkdir()

            manager = GitHookManager(repo_path=repo_path)

            # Install hooks
            success = manager.install_hooks()
            assert success

            # Check hooks exist
            pre_commit_hook = manager.git_hooks_path / "pre-commit"
            pre_push_hook = manager.git_hooks_path / "pre-push"

            assert pre_commit_hook.exists()
            assert pre_push_hook.exists()
            assert pre_commit_hook.stat().st_mode & 0o111  # Executable
            assert pre_push_hook.stat().st_mode & 0o111  # Executable

            # Check hook content
            pre_commit_content = pre_commit_hook.read_text()
            assert "UDL_RATING_HOOK" in pre_commit_content

            # Uninstall hooks
            success = manager.uninstall_hooks()
            assert success

            # Check hooks removed
            assert not pre_commit_hook.exists()
            assert not pre_push_hook.exists()


class TestCICDIntegration:
    """Test CI/CD integration."""

    def test_cicd_config(self):
        """Test CI/CD configuration."""
        config = CICDConfig(
            min_quality_threshold=0.8,
            fail_on_quality_drop=True,
            generate_reports=True,
            report_format="json",
            artifact_retention_days=14,
            parallel_jobs=2,
            timeout_minutes=15,
        )

        assert config.min_quality_threshold == 0.8
        assert config.fail_on_quality_drop is True
        assert config.generate_reports is True
        assert config.report_format == "json"
        assert config.artifact_retention_days == 14
        assert config.parallel_jobs == 2
        assert config.timeout_minutes == 15

    def test_github_actions_workflow_generation(self):
        """Test GitHub Actions workflow generation."""
        config = CICDConfig(min_quality_threshold=0.7)
        integration = CICDIntegration(config)

        workflow = integration.generate_github_actions_workflow(
            workflow_name="test-workflow", triggers=["push", "pull_request"]
        )

        # Parse YAML
        workflow_data = yaml.safe_load(workflow)

        assert workflow_data["name"] == "UDL Quality Check"
        assert "push" in workflow_data["on"]
        assert "pull_request" in workflow_data["on"]
        assert "udl-quality" in workflow_data["jobs"]

        # Check job steps
        steps = workflow_data["jobs"]["udl-quality"]["steps"]
        step_names = [step["name"] for step in steps]

        assert "Checkout code" in step_names
        assert "Set up Python" in step_names
        assert "Run UDL quality check" in step_names

    def test_jenkins_pipeline_generation(self):
        """Test Jenkins pipeline generation."""
        config = CICDConfig(min_quality_threshold=0.7, timeout_minutes=20)
        integration = CICDIntegration(config)

        pipeline = integration.generate_jenkins_pipeline("Test Pipeline")

        assert "pipeline {" in pipeline
        assert "timeout(time: 20, unit: 'MINUTES')" in pipeline
        assert "UDL_QUALITY_THRESHOLD = '0.7'" in pipeline
        assert "udl-rating rate" in pipeline

    def test_gitlab_ci_generation(self):
        """Test GitLab CI configuration generation."""
        config = CICDConfig(min_quality_threshold=0.8)
        integration = CICDIntegration(config)

        gitlab_ci = integration.generate_gitlab_ci()

        # Parse YAML
        ci_data = yaml.safe_load(gitlab_ci)

        assert "stages" in ci_data
        assert "quality" in ci_data["stages"]
        assert "UDL_QUALITY_THRESHOLD" in ci_data["variables"]
        assert ci_data["variables"]["UDL_QUALITY_THRESHOLD"] == "0.8"
        assert "udl_quality_check" in ci_data

    def test_azure_devops_pipeline_generation(self):
        """Test Azure DevOps pipeline generation."""
        config = CICDConfig(min_quality_threshold=0.6)
        integration = CICDIntegration(config)

        pipeline = integration.generate_azure_devops_pipeline()

        # Parse YAML
        pipeline_data = yaml.safe_load(pipeline)

        assert "trigger" in pipeline_data
        assert "jobs" in pipeline_data
        assert pipeline_data["variables"]["UDL_QUALITY_THRESHOLD"] == 0.6

        # Check job steps
        job = pipeline_data["jobs"][0]
        assert job["job"] == "UDLQualityCheck"
        assert any("udl-rating rate" in step.get("script", "") for step in job["steps"])

    def test_workflow_file_creation(self):
        """Test workflow file creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            config = CICDConfig()
            integration = CICDIntegration(config)

            created_files = integration.create_workflow_files(
                output_dir, platforms=["github", "jenkins"]
            )

            assert "github" in created_files
            assert "jenkins" in created_files

            # Check GitHub workflow
            github_file = created_files["github"]
            assert github_file.exists()
            assert github_file.name == "udl-quality-check.yml"

            # Check Jenkins file
            jenkins_file = created_files["jenkins"]
            assert jenkins_file.exists()
            assert jenkins_file.name == "Jenkinsfile"

    def test_workflow_syntax_validation(self):
        """Test workflow syntax validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            integration = CICDIntegration()

            # Create valid YAML file
            valid_yaml = output_dir / "valid.yml"
            valid_yaml.write_text(
                "name: test\non: push\njobs:\n  test:\n    runs-on: ubuntu-latest"
            )

            # Create invalid YAML file
            invalid_yaml = output_dir / "invalid.yml"
            invalid_yaml.write_text(
                "name: test\non: push\njobs:\n  test:\n    runs-on: ubuntu-latest\n  invalid: ["
            )

            assert integration.validate_workflow_syntax(valid_yaml) is True
            assert integration.validate_workflow_syntax(invalid_yaml) is False


class TestLSPServer:
    """Test Language Server Protocol implementation."""

    def test_udl_language_server_initialization(self):
        """Test UDL Language Server initialization."""
        server = UDLLanguageServer(
            min_quality_threshold=0.8, enable_real_time=True, debounce_delay=1.0
        )

        assert server.min_quality_threshold == 0.8
        assert server.enable_real_time is True
        assert server.debounce_delay == 1.0
        assert server.initialized is False
        assert server.shutdown_requested is False

    def test_lsp_initialize(self):
        """Test LSP initialize request."""
        import asyncio

        server = UDLLanguageServer()

        params = {
            "capabilities": {
                "textDocument": {"publishDiagnostics": {"versionSupport": True}}
            }
        }

        response = asyncio.get_event_loop().run_until_complete(
            server.initialize(params)
        )

        assert server.initialized is True
        assert "capabilities" in response
        assert "serverInfo" in response
        assert response["serverInfo"]["name"] == "UDL Rating Language Server"

    def test_text_document_did_open(self):
        """Test textDocument/didOpen notification."""
        import asyncio

        server = UDLLanguageServer()

        params = {
            "textDocument": {
                "uri": "file:///test.udl",
                "languageId": "udl",
                "version": 1,
                "text": "grammar Test { rule test = 'hello' }",
            }
        }

        asyncio.get_event_loop().run_until_complete(
            server.text_document_did_open(params)
        )

        uri = params["textDocument"]["uri"]
        assert uri in server.documents
        assert server.documents[uri].text == params["textDocument"]["text"]
        assert server.document_versions[uri] == 1

    def test_text_document_hover(self):
        """Test textDocument/hover request."""
        import asyncio

        server = UDLLanguageServer()

        # First open a document
        asyncio.get_event_loop().run_until_complete(
            server.text_document_did_open(
                {
                    "textDocument": {
                        "uri": "file:///test.udl",
                        "languageId": "udl",
                        "version": 1,
                        "text": "grammar Test { rule test = 'hello' }",
                    }
                }
            )
        )

        # Add some quality info to cache
        server.quality_cache["file:///test.udl"] = {
            "overall_score": 0.85,
            "confidence": 0.92,
            "metric_scores": {"consistency": 0.9, "completeness": 0.8},
        }

        # Test hover
        params = {
            "textDocument": {"uri": "file:///test.udl"},
            "position": {"line": 0, "character": 5},
        }

        response = asyncio.get_event_loop().run_until_complete(
            server.text_document_hover(params)
        )

        assert response is not None
        assert "contents" in response
        assert response["contents"]["kind"] == "markdown"
        assert "Overall Score" in response["contents"]["value"]
        assert "0.850" in response["contents"]["value"]


class TestBatchProcessor:
    """Test batch processing functionality."""

    def test_batch_config(self):
        """Test batch configuration."""
        config = BatchConfig(
            max_workers=8,
            chunk_size=20,
            timeout_per_file=60.0,
            memory_limit_mb=2048,
            enable_caching=True,
            error_handling="retry",
            max_retries=5,
            output_format="csv",
        )

        assert config.max_workers == 8
        assert config.chunk_size == 20
        assert config.timeout_per_file == 60.0
        assert config.memory_limit_mb == 2048
        assert config.enable_caching is True
        assert config.error_handling == "retry"
        assert config.max_retries == 5
        assert config.output_format == "csv"

    def test_batch_processor_initialization(self):
        """Test batch processor initialization."""
        config = BatchConfig(max_workers=4, enable_caching=False)
        processor = BatchProcessor(config)

        assert processor.config.max_workers == 4
        assert processor.config.enable_caching is False
        assert processor.processed_count == 0
        assert processor.total_count == 0

    def test_processing_task_creation(self):
        """Test processing task creation."""
        from udl_rating_framework.integration.batch_processor import ProcessingTask

        file_path = Path("test.udl")
        task = ProcessingTask(
            file_path=file_path, content="grammar Test {}", priority=1, retry_count=0
        )

        assert task.file_path == file_path
        assert task.content == "grammar Test {}"
        assert task.priority == 1
        assert task.retry_count == 0

    def test_create_sample_udl_files(self):
        """Test creating sample UDL files for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_dir = Path(temp_dir) / "samples"

            # Create sample files
            sample_dir.mkdir()
            for i in range(3):
                file_path = sample_dir / f"test_{i}.udl"
                file_path.write_text(f'grammar Test{i} {{ rule test = "value{i}" }}')

            # Test file discovery
            config = BatchConfig(max_workers=1)
            BatchProcessor(config)

            udl_files = list(sample_dir.glob("*.udl"))
            assert len(udl_files) == 3

    @patch("udl_rating_framework.integration.batch_processor.RatingPipeline")
    def test_batch_result_creation(self, mock_pipeline_class):
        """Test batch result creation."""
        # Mock the pipeline
        mock_pipeline = Mock()
        mock_report = Mock()
        mock_report.overall_score = 0.75
        mock_report.confidence = 0.85
        mock_report.metric_scores = {"consistency": 0.8, "completeness": 0.7}
        mock_report.computation_trace = []
        mock_pipeline.rate_udl.return_value = mock_report
        mock_pipeline_class.return_value = mock_pipeline

        with tempfile.TemporaryDirectory() as temp_dir:
            sample_dir = Path(temp_dir)

            # Create sample file
            test_file = sample_dir / "test.udl"
            test_file.write_text('grammar Test { rule test = "hello" }')

            config = BatchConfig(max_workers=1, enable_caching=False)
            processor = BatchProcessor(config)

            # Process files
            result = processor.process_files([test_file])

            assert result.total_files == 1
            assert result.processed_files == 1
            assert result.failed_files == 0
            assert result.average_quality == 0.75
            assert len(result.file_results) == 1

    def test_save_results_json(self):
        """Test saving results in JSON format."""
        from udl_rating_framework.integration.batch_processor import BatchResult

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "results.json"

            # Create sample result
            result = BatchResult(
                total_files=2,
                processed_files=2,
                failed_files=0,
                processing_time=1.5,
                average_quality=0.8,
                quality_distribution={"good": 2},
                file_results={
                    "file1.udl": {"overall_score": 0.75, "confidence": 0.9},
                    "file2.udl": {"overall_score": 0.85, "confidence": 0.8},
                },
                errors={},
                summary_stats={"processing_rate": 1.33},
            )

            config = BatchConfig()
            processor = BatchProcessor(config)
            processor.save_results(result, output_file, "json")

            assert output_file.exists()

            # Verify JSON content
            with open(output_file) as f:
                data = json.load(f)

            assert data["total_files"] == 2
            assert data["average_quality"] == 0.8
            assert "file1.udl" in data["file_results"]


class TestIDEPluginManager:
    """Test IDE plugin management."""

    def test_ide_plugin_manager_initialization(self):
        """Test IDE plugin manager initialization."""
        from udl_rating_framework.integration.ide_plugin import PluginConfig

        config = PluginConfig(
            enable_real_time_checking=True,
            quality_threshold=0.8,
            show_detailed_metrics=True,
            auto_save_reports=False,
            update_interval=2.0,
            max_file_size_mb=5,
        )

        manager = IDEPluginManager(config)

        assert manager.config.enable_real_time_checking is True
        assert manager.config.quality_threshold == 0.8
        assert manager.config.show_detailed_metrics is True
        assert manager.config.auto_save_reports is False
        assert manager.config.update_interval == 2.0
        assert manager.config.max_file_size_mb == 5

    def test_vscode_extension_generation(self):
        """Test VS Code extension generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            manager = IDEPluginManager()
            extension_dir = manager.generate_vscode_extension(output_dir)

            assert extension_dir.exists()
            assert extension_dir.name == "udl-rating-vscode"

            # Check required files
            assert (extension_dir / "package.json").exists()
            assert (extension_dir / "src" / "extension.ts").exists()
            assert (extension_dir / "tsconfig.json").exists()
            assert (extension_dir / "language-configuration.json").exists()
            assert (extension_dir / "syntaxes" / "udl.tmGrammar.json").exists()

            # Verify package.json content
            with open(extension_dir / "package.json") as f:
                package_data = json.load(f)

            assert package_data["name"] == "udl-rating"
            assert package_data["displayName"] == "UDL Rating Framework"
            assert "udl-rating.checkQuality" in [
                cmd["command"] for cmd in package_data["contributes"]["commands"]
            ]

    def test_intellij_plugin_generation(self):
        """Test IntelliJ plugin generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            manager = IDEPluginManager()
            plugin_dir = manager.generate_intellij_plugin(output_dir)

            assert plugin_dir.exists()
            assert plugin_dir.name == "udl-rating-intellij"

            # Check required files
            assert (plugin_dir / "plugin.xml").exists()
            assert (plugin_dir / "build.gradle").exists()
            assert (
                plugin_dir
                / "src"
                / "main"
                / "java"
                / "com"
                / "udlrating"
                / "intellij"
                / "UDLFileType.java"
            ).exists()

            # Verify plugin.xml content
            plugin_xml_content = (plugin_dir / "plugin.xml").read_text()
            assert "com.udlrating.intellij" in plugin_xml_content
            assert "UDL Rating Framework" in plugin_xml_content

    def test_vim_plugin_generation(self):
        """Test Vim plugin generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            manager = IDEPluginManager()
            plugin_dir = manager.generate_vim_plugin(output_dir)

            assert plugin_dir.exists()
            assert plugin_dir.name == "udl-rating-vim"

            # Check required directories and files
            assert (plugin_dir / "plugin").exists()
            assert (plugin_dir / "autoload").exists()
            assert (plugin_dir / "syntax").exists()
            assert (plugin_dir / "ftdetect").exists()

            assert (plugin_dir / "plugin" / "udl_rating.vim").exists()
            assert (plugin_dir / "autoload" / "udl_rating.vim").exists()
            assert (plugin_dir / "syntax" / "udl.vim").exists()
            assert (plugin_dir / "ftdetect" / "udl.vim").exists()

            # Verify plugin content
            plugin_content = (plugin_dir / "plugin" / "udl_rating.vim").read_text()
            assert "UDLCheckQuality" in plugin_content
            assert "udl_rating#check_quality" in plugin_content


class TestIntegrationCLI:
    """Test integration CLI commands."""

    @patch("udl_rating_framework.cli.commands.integration.GitHookManager")
    def test_git_hooks_cli_integration(self, mock_manager_class):
        """Test Git hooks CLI integration."""
        from udl_rating_framework.cli.commands.integration import install_git_hooks
        from click.testing import CliRunner

        # Mock the manager
        mock_manager = Mock()
        mock_manager.install_hooks.return_value = True
        mock_manager_class.return_value = mock_manager

        runner = CliRunner()

        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(
                install_git_hooks, ["--repo-path", temp_dir, "--threshold", "0.8"]
            )

            assert result.exit_code == 0
            assert "✅ Git hooks installed successfully" in result.output
            mock_manager.install_hooks.assert_called_once()

    @patch("udl_rating_framework.cli.commands.integration.CICDIntegration")
    def test_cicd_cli_integration(self, mock_integration_class):
        """Test CI/CD CLI integration."""
        from udl_rating_framework.cli.commands.integration import (
            generate_cicd_workflows,
        )
        from click.testing import CliRunner

        # Mock the integration
        mock_integration = Mock()
        mock_integration.create_workflow_files.return_value = {
            "github": Path("test/.github/workflows/udl-quality-check.yml")
        }
        mock_integration_class.return_value = mock_integration

        runner = CliRunner()

        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(
                generate_cicd_workflows,
                [
                    "--platform",
                    "github",
                    "--output-dir",
                    temp_dir,
                    "--threshold",
                    "0.7",
                ],
            )

            assert result.exit_code == 0
            assert "Generated CI/CD workflow files" in result.output
            mock_integration.create_workflow_files.assert_called_once()

    @patch("udl_rating_framework.cli.commands.integration.GitHookManager")
    def test_git_hooks_uninstall_cli(self, mock_manager_class):
        """Test Git hooks uninstall CLI command."""
        from udl_rating_framework.cli.commands.integration import uninstall_git_hooks
        from click.testing import CliRunner

        # Mock the manager
        mock_manager = Mock()
        mock_manager.uninstall_hooks.return_value = True
        mock_manager_class.return_value = mock_manager

        runner = CliRunner()

        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(uninstall_git_hooks, ["--repo-path", temp_dir])

            assert result.exit_code == 0
            assert "✅ Git hooks uninstalled successfully" in result.output
            mock_manager.uninstall_hooks.assert_called_once()

    @patch("udl_rating_framework.cli.commands.integration.GitHookManager")
    def test_git_hooks_check_staged_cli(self, mock_manager_class):
        """Test Git hooks check-staged CLI command."""
        from udl_rating_framework.cli.commands.integration import check_staged_files
        from click.testing import CliRunner

        # Mock the manager
        mock_manager = Mock()
        mock_manager.check_staged_files.return_value = (
            True,
            {"test.udl": {"score": 0.85, "passed": True}},
        )
        mock_manager_class.return_value = mock_manager

        runner = CliRunner()

        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(
                check_staged_files, ["--repo-path", temp_dir, "--threshold", "0.7"]
            )

            assert result.exit_code == 0
            mock_manager.check_staged_files.assert_called_once()

    @patch("udl_rating_framework.cli.commands.integration.CICDIntegration")
    def test_cicd_validate_cli(self, mock_integration_class):
        """Test CI/CD validate CLI command."""
        from udl_rating_framework.cli.commands.integration import (
            validate_cicd_workflows,
        )
        from click.testing import CliRunner

        # Mock the integration
        mock_integration = Mock()
        mock_integration.validate_workflow_syntax.return_value = True
        mock_integration_class.return_value = mock_integration

        runner = CliRunner()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock workflow file
            github_dir = Path(temp_dir) / ".github" / "workflows"
            github_dir.mkdir(parents=True)
            workflow_file = github_dir / "test.yml"
            workflow_file.write_text("name: test\non: push")

            result = runner.invoke(
                validate_cicd_workflows, ["--workflow-dir", temp_dir]
            )

            assert result.exit_code == 0

    def test_cicd_generate_multiple_platforms(self):
        """Test CI/CD workflow generation for multiple platforms."""
        from udl_rating_framework.cli.commands.integration import (
            generate_cicd_workflows,
        )
        from click.testing import CliRunner

        runner = CliRunner()

        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(
                generate_cicd_workflows,
                [
                    "--platform",
                    "github",
                    "--platform",
                    "jenkins",
                    "--output-dir",
                    temp_dir,
                    "--threshold",
                    "0.8",
                ],
            )

            assert result.exit_code == 0
            assert "Generated CI/CD workflow files" in result.output

            # Check files were created
            github_workflow = (
                Path(temp_dir) / ".github" / "workflows" / "udl-quality-check.yml"
            )
            jenkinsfile = Path(temp_dir) / "Jenkinsfile"

            assert github_workflow.exists()
            assert jenkinsfile.exists()


class TestGitHooksEndToEnd:
    """End-to-end tests for Git hooks functionality."""

    def test_git_hooks_install_and_uninstall_real(self):
        """Test real Git hooks installation and uninstallation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            git_dir = repo_path / ".git"
            git_dir.mkdir()

            manager = GitHookManager(repo_path=repo_path, min_quality_threshold=0.7)

            # Install hooks
            success = manager.install_hooks()
            assert success

            # Verify hooks exist
            pre_commit = manager.git_hooks_path / "pre-commit"
            pre_push = manager.git_hooks_path / "pre-push"

            assert pre_commit.exists()
            assert pre_push.exists()

            # Verify hooks are executable
            assert pre_commit.stat().st_mode & 0o111
            assert pre_push.stat().st_mode & 0o111

            # Verify hook content
            content = pre_commit.read_text()
            assert "UDL_RATING_HOOK" in content
            assert "check_staged_files" in content

            # Uninstall hooks
            success = manager.uninstall_hooks()
            assert success

            # Verify hooks are removed
            assert not pre_commit.exists()
            assert not pre_push.exists()

    def test_git_hooks_template_escaping(self):
        """Test that hook templates properly escape Python f-string braces."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            git_dir = repo_path / ".git"
            git_dir.mkdir()

            manager = GitHookManager(repo_path=repo_path, min_quality_threshold=0.7)

            # Install hooks
            success = manager.install_hooks()
            assert success

            # Read the generated hook
            pre_commit = manager.git_hooks_path / "pre-commit"
            content = pre_commit.read_text()

            # Verify that Python f-string braces are properly escaped
            # The template should have {fp} and {score:.3f} etc. in the generated script
            assert "{fp}" in content or "fp" in content

            # Verify no unsubstituted template variables remain
            assert "{python_path}" not in content
            assert "{repo_path}" not in content
            assert "{min_quality_threshold}" not in content


class TestCICDEndToEnd:
    """End-to-end tests for CI/CD integration."""

    def test_github_workflow_generation_real(self):
        """Test real GitHub Actions workflow generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            config = CICDConfig(min_quality_threshold=0.75, timeout_minutes=25)
            integration = CICDIntegration(config)

            created_files = integration.create_workflow_files(output_dir, ["github"])

            assert "github" in created_files
            workflow_file = created_files["github"]
            assert workflow_file.exists()

            # Parse and validate workflow
            content = workflow_file.read_text()
            workflow_data = yaml.safe_load(content)

            assert workflow_data["name"] == "UDL Quality Check"
            assert "jobs" in workflow_data
            assert "udl-quality" in workflow_data["jobs"]

    def test_jenkins_pipeline_generation_real(self):
        """Test real Jenkins pipeline generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            config = CICDConfig(min_quality_threshold=0.8, timeout_minutes=30)
            integration = CICDIntegration(config)

            created_files = integration.create_workflow_files(output_dir, ["jenkins"])

            assert "jenkins" in created_files
            jenkinsfile = created_files["jenkins"]
            assert jenkinsfile.exists()

            # Validate Jenkinsfile content
            content = jenkinsfile.read_text()
            assert "pipeline {" in content
            assert "UDL_QUALITY_THRESHOLD" in content
            assert "0.8" in content

    def test_all_platforms_generation(self):
        """Test workflow generation for all supported platforms."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            integration = CICDIntegration()
            platforms = ["github", "jenkins", "gitlab", "azure"]

            created_files = integration.create_workflow_files(output_dir, platforms)

            assert len(created_files) == 4
            for platform in platforms:
                assert platform in created_files
                assert created_files[platform].exists()


class TestLSPServerEndToEnd:
    """End-to-end tests for LSP server functionality."""

    def test_lsp_server_full_workflow(self):
        """Test complete LSP server workflow."""
        import asyncio

        server = UDLLanguageServer(min_quality_threshold=0.7)
        lsp_server = LSPServer(server)

        # Initialize
        init_response = asyncio.get_event_loop().run_until_complete(
            lsp_server.handle_request(
                {"id": 1, "method": "initialize", "params": {"capabilities": {}}}
            )
        )

        assert init_response is not None
        assert "result" in init_response
        assert "capabilities" in init_response["result"]

        # Open document
        asyncio.get_event_loop().run_until_complete(
            lsp_server.handle_request(
                {
                    "method": "textDocument/didOpen",
                    "params": {
                        "textDocument": {
                            "uri": "file:///test.udl",
                            "languageId": "udl",
                            "version": 1,
                            "text": "grammar Test { rule test = 'hello' }",
                        }
                    },
                }
            )
        )

        assert "file:///test.udl" in server.documents

        # Close document
        asyncio.get_event_loop().run_until_complete(
            lsp_server.handle_request(
                {
                    "method": "textDocument/didClose",
                    "params": {"textDocument": {"uri": "file:///test.udl"}},
                }
            )
        )

        assert "file:///test.udl" not in server.documents

    def test_lsp_server_code_actions(self):
        """Test LSP server code actions."""
        import asyncio

        server = UDLLanguageServer(min_quality_threshold=0.7)

        # Open document
        asyncio.get_event_loop().run_until_complete(
            server.text_document_did_open(
                {
                    "textDocument": {
                        "uri": "file:///test.udl",
                        "languageId": "udl",
                        "version": 1,
                        "text": "grammar Test { rule test = 'hello' }",
                    }
                }
            )
        )

        # Add quality info to cache
        server.quality_cache["file:///test.udl"] = {
            "overall_score": 0.5,  # Below threshold
            "confidence": 0.8,
            "metric_scores": {"consistency": 0.5},
        }

        # Get code actions
        actions = asyncio.get_event_loop().run_until_complete(
            server.text_document_code_action(
                {
                    "textDocument": {"uri": "file:///test.udl"},
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 0},
                    },
                    "context": {"diagnostics": []},
                }
            )
        )

        assert len(actions) >= 1
        action_titles = [a["title"] for a in actions]
        assert "Show UDL Quality Report" in action_titles


if __name__ == "__main__":
    pytest.main([__file__])
