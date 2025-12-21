"""
CLI commands for integration and workflow features.
"""

import logging
from pathlib import Path
from typing import List, Optional

import click

from udl_rating_framework.integration.batch_processor import BatchConfig, BatchProcessor
from udl_rating_framework.integration.cicd import CICDConfig, CICDIntegration
from udl_rating_framework.integration.git_hooks import GitHookManager
from udl_rating_framework.integration.ide_plugin import IDEPluginManager
from udl_rating_framework.integration.lsp_server import LSPServer, UDLLanguageServer

logger = logging.getLogger(__name__)


@click.group()
def integration():
    """Integration and workflow commands."""
    pass


@integration.group()
def git():
    """Git integration commands."""
    pass


@git.command("install-hooks")
@click.option(
    "--repo-path",
    type=click.Path(exists=True, path_type=Path),
    default=Path.cwd(),
    help="Git repository path",
)
@click.option("--threshold", type=float, default=0.7, help="Minimum quality threshold")
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Configuration file path",
)
def install_git_hooks(repo_path: Path, threshold: float, config: Optional[Path]):
    """Install Git hooks for UDL quality checking."""
    try:
        manager = GitHookManager(
            repo_path=repo_path, min_quality_threshold=threshold, config_file=config
        )

        if manager.install_hooks():
            click.echo(f"✅ Git hooks installed successfully in {repo_path}")
        else:
            click.echo("❌ Failed to install Git hooks", err=True)
            exit(1)

    except Exception as e:
        click.echo(f"❌ Error installing Git hooks: {e}", err=True)
        exit(1)


@git.command("uninstall-hooks")
@click.option(
    "--repo-path",
    type=click.Path(exists=True, path_type=Path),
    default=Path.cwd(),
    help="Git repository path",
)
def uninstall_git_hooks(repo_path: Path):
    """Uninstall Git hooks for UDL quality checking."""
    try:
        manager = GitHookManager(repo_path=repo_path)

        if manager.uninstall_hooks():
            click.echo(
                f"✅ Git hooks uninstalled successfully from {repo_path}")
        else:
            click.echo("❌ Failed to uninstall Git hooks", err=True)
            exit(1)

    except Exception as e:
        click.echo(f"❌ Error uninstalling Git hooks: {e}", err=True)
        exit(1)


@git.command("check-staged")
@click.option(
    "--repo-path",
    type=click.Path(exists=True, path_type=Path),
    default=Path.cwd(),
    help="Git repository path",
)
@click.option("--threshold", type=float, default=0.7, help="Minimum quality threshold")
def check_staged_files(repo_path: Path, threshold: float):
    """Check quality of staged UDL files."""
    try:
        manager = GitHookManager(
            repo_path=repo_path, min_quality_threshold=threshold)

        passed, results = manager.check_staged_files()

        if "message" in results:
            click.echo(results["message"])
            return

        click.echo("Staged UDL files quality check:")
        for file_path, result in results.items():
            if result.get("passed", False):
                click.echo(f"✅ {file_path}: {result['score']:.3f}")
            else:
                error = result.get("error", "Quality below threshold")
                score = result.get("score", "N/A")
                click.echo(f"❌ {file_path}: {score} ({error})")

        if not passed:
            exit(1)

    except Exception as e:
        click.echo(f"❌ Error checking staged files: {e}", err=True)
        exit(1)


@integration.group()
def cicd():
    """CI/CD integration commands."""
    pass


@cicd.command("generate")
@click.option(
    "--platform",
    type=click.Choice(["github", "jenkins", "gitlab", "azure"]),
    multiple=True,
    default=["github"],
    help="CI/CD platform(s)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path.cwd(),
    help="Output directory for workflow files",
)
@click.option("--threshold", type=float, default=0.7, help="Minimum quality threshold")
@click.option("--timeout", type=int, default=30, help="Timeout in minutes")
@click.option("--retention-days", type=int, default=30, help="Artifact retention days")
def generate_cicd_workflows(
    platform: List[str],
    output_dir: Path,
    threshold: float,
    timeout: int,
    retention_days: int,
):
    """Generate CI/CD workflow files."""
    try:
        config = CICDConfig(
            min_quality_threshold=threshold,
            timeout_minutes=timeout,
            artifact_retention_days=retention_days,
        )

        integration = CICDIntegration(config)
        created_files = integration.create_workflow_files(output_dir, platform)

        click.echo("Generated CI/CD workflow files:")
        for platform_name, file_path in created_files.items():
            click.echo(f"  {platform_name}: {file_path}")

    except Exception as e:
        click.echo(f"❌ Error generating CI/CD workflows: {e}", err=True)
        exit(1)


@cicd.command("validate")
@click.option(
    "--workflow-dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path.cwd(),
    help="Directory containing workflow files",
)
def validate_cicd_workflows(workflow_dir: Path):
    """Validate CI/CD workflow files."""
    try:
        integration = CICDIntegration()

        # Find workflow files
        workflow_files = []

        github_workflows = workflow_dir / ".github" / "workflows"
        if github_workflows.exists():
            workflow_files.extend(github_workflows.glob("*.yml"))
            workflow_files.extend(github_workflows.glob("*.yaml"))

        for filename in ["Jenkinsfile", ".gitlab-ci.yml", "azure-pipelines.yml"]:
            file_path = workflow_dir / filename
            if file_path.exists():
                workflow_files.append(file_path)

        if not workflow_files:
            click.echo("No workflow files found")
            return

        all_valid = True
        for workflow_file in workflow_files:
            is_valid = integration.validate_workflow_syntax(workflow_file)
            status = "✅ Valid" if is_valid else "❌ Invalid"
            click.echo(f"{status}: {workflow_file}")
            all_valid = all_valid and is_valid

        if not all_valid:
            exit(1)

    except Exception as e:
        click.echo(f"❌ Error validating workflows: {e}", err=True)
        exit(1)


@integration.command("lsp-server")
@click.option("--threshold", type=float, default=0.7, help="Minimum quality threshold")
@click.option(
    "--real-time/--no-real-time", default=True, help="Enable real-time quality checking"
)
@click.option("--debounce", type=float, default=0.5, help="Debounce delay in seconds")
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Log level",
)
def start_lsp_server(
    threshold: float, real_time: bool, debounce: float, log_level: str
):
    """Start UDL Language Server Protocol server."""
    import asyncio
    import json
    import sys

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Create language server
        language_server = UDLLanguageServer(
            min_quality_threshold=threshold,
            enable_real_time=real_time,
            debounce_delay=debounce,
        )

        lsp_server = LSPServer(language_server)

        click.echo("Starting UDL Language Server...")

        # Simple stdio-based LSP server
        async def stdio_server():
            while not language_server.shutdown_requested:
                try:
                    # Read request from stdin
                    line = sys.stdin.readline()
                    if not line:
                        break

                    request = json.loads(line.strip())
                    response = await lsp_server.handle_request(request)

                    if response:
                        print(json.dumps(response))
                        sys.stdout.flush()

                except Exception as e:
                    logger.error(f"Error in stdio server: {e}")
                    break

        # Run server
        asyncio.run(stdio_server())

    except KeyboardInterrupt:
        click.echo("Server interrupted by user")
    except Exception as e:
        click.echo(f"❌ Server error: {e}", err=True)
        exit(1)


@integration.command("batch-process")
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output file path",
)
@click.option(
    "--format",
    type=click.Choice(["json", "csv", "html"]),
    default="json",
    help="Output format",
)
@click.option("--workers", type=int, default=None, help="Number of worker processes")
@click.option(
    "--chunk-size", type=int, default=10, help="Chunk size for parallel processing"
)
@click.option("--timeout", type=float, default=30.0, help="Timeout per file in seconds")
@click.option(
    "--cache-dir", type=click.Path(path_type=Path), help="Cache directory path"
)
@click.option("--no-cache", is_flag=True, help="Disable caching")
@click.option(
    "--include-patterns",
    multiple=True,
    default=["*.udl", "*.dsl", "*.grammar", "*.ebnf"],
    help="File patterns to include",
)
@click.option("--exclude-patterns", multiple=True, help="File patterns to exclude")
@click.option(
    "--streaming", is_flag=True, help="Use streaming processing for large datasets"
)
def batch_process_files(
    input_path: Path,
    output: Path,
    format: str,
    workers: Optional[int],
    chunk_size: int,
    timeout: float,
    cache_dir: Optional[Path],
    no_cache: bool,
    include_patterns: List[str],
    exclude_patterns: List[str],
    streaming: bool,
):
    """Batch process UDL files for quality analysis."""
    import multiprocessing as mp

    try:
        # Create configuration
        config = BatchConfig(
            max_workers=workers or mp.cpu_count(),
            chunk_size=chunk_size,
            timeout_per_file=timeout,
            enable_caching=not no_cache,
            cache_dir=cache_dir,
            output_format=format,
        )

        # Progress callback
        def progress_callback(processed: int, total: int):
            percentage = processed / total * 100 if total > 0 else 0
            click.echo(f"Progress: {processed}/{total} ({percentage:.1f}%)")

        config.progress_callback = progress_callback

        # Create processor
        processor = BatchProcessor(config)

        if input_path.is_dir():
            # Process directory
            if streaming:
                click.echo(
                    "Streaming processing not supported for directory input")
                return

            result = processor.process_directory(
                input_path,
                patterns=list(include_patterns),
                exclude_patterns=list(
                    exclude_patterns) if exclude_patterns else None,
            )
        else:
            # Process file list
            with open(input_path) as f:
                file_paths = [Path(line.strip()) for line in f if line.strip()]

            if streaming:
                click.echo("Starting streaming processing...")
                for file_result in processor.process_files_streaming(
                    file_paths, output
                ):
                    click.echo(f"Processed: {file_result['file_path']}")
                return
            else:
                result = processor.process_files(file_paths)

        # Save results
        processor.save_results(result, output, format)

        # Print summary
        click.echo("\nBatch processing completed:")
        click.echo(f"  Total files: {result.total_files}")
        click.echo(f"  Processed: {result.processed_files}")
        click.echo(f"  Failed: {result.failed_files}")
        click.echo(f"  Average quality: {result.average_quality:.3f}")
        click.echo(f"  Processing time: {result.processing_time:.1f}s")
        click.echo(f"  Results saved to: {output}")

    except Exception as e:
        click.echo(f"❌ Batch processing failed: {e}", err=True)
        exit(1)


@integration.group()
def ide():
    """IDE plugin commands."""
    pass


@ide.command("generate")
@click.option(
    "--plugin-type",
    type=click.Choice(["vscode", "intellij", "vim", "emacs", "sublime"]),
    required=True,
    help="Type of plugin to generate",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path.cwd(),
    help="Output directory for generated plugins",
)
def generate_ide_plugin(plugin_type: str, output_dir: Path):
    """Generate IDE plugin for UDL quality checking."""
    try:
        manager = IDEPluginManager()

        if plugin_type == "vscode":
            plugin_dir = manager.generate_vscode_extension(output_dir)
        elif plugin_type == "intellij":
            plugin_dir = manager.generate_intellij_plugin(output_dir)
        elif plugin_type == "vim":
            plugin_dir = manager.generate_vim_plugin(output_dir)
        else:
            click.echo(
                f"Plugin generation for {plugin_type} not yet implemented")
            return

        click.echo(f"✅ {plugin_type} plugin generated at: {plugin_dir}")

        # Provide installation instructions
        if plugin_type == "vscode":
            click.echo("\nTo install:")
            click.echo("1. cd " + str(plugin_dir))
            click.echo("2. npm install")
            click.echo("3. npm run compile")
            click.echo("4. vsce package")
            click.echo("5. code --install-extension *.vsix")
        elif plugin_type == "intellij":
            click.echo("\nTo install:")
            click.echo("1. cd " + str(plugin_dir))
            click.echo("2. ./gradlew buildPlugin")
            click.echo("3. Install through IntelliJ IDEA plugin manager")
        elif plugin_type == "vim":
            click.echo("\nTo install:")
            click.echo("Copy plugin files to your Vim configuration directory")

    except Exception as e:
        click.echo(f"❌ Error generating {plugin_type} plugin: {e}", err=True)
        exit(1)


@ide.command("install")
@click.option(
    "--plugin-type",
    type=click.Choice(["vscode", "intellij", "vim"]),
    required=True,
    help="Type of plugin to install",
)
@click.option(
    "--plugin-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing plugin files",
)
def install_ide_plugin(plugin_type: str, plugin_dir: Path):
    """Install IDE plugin."""
    try:
        manager = IDEPluginManager()

        if manager.install_plugin(plugin_type, plugin_dir):
            click.echo(f"✅ {plugin_type} plugin installed successfully")
        else:
            click.echo(f"❌ Failed to install {plugin_type} plugin", err=True)
            exit(1)

    except Exception as e:
        click.echo(f"❌ Error installing {plugin_type} plugin: {e}", err=True)
        exit(1)


# Add integration command to main CLI
def add_integration_commands(cli):
    """Add integration commands to main CLI."""
    cli.add_command(integration)
