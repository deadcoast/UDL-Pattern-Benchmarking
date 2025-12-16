"""
Git hooks for automatic UDL quality checking.

Provides pre-commit and pre-push hooks that automatically check UDL quality
and prevent commits/pushes if quality thresholds are not met.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from udl_rating_framework.core.pipeline import RatingPipeline
from udl_rating_framework.core.representation import UDLRepresentation
from udl_rating_framework.io.file_discovery import FileDiscovery

logger = logging.getLogger(__name__)


class GitHookManager:
    """
    Manages Git hooks for automatic UDL quality checking.

    Features:
    - Pre-commit hook to check staged UDL files
    - Pre-push hook to validate all UDL files
    - Configurable quality thresholds
    - Integration with existing Git workflows
    """

    def __init__(
        self,
        repo_path: Path,
        min_quality_threshold: float = 0.7,
        config_file: Optional[Path] = None,
    ):
        """
        Initialize Git hook manager.

        Args:
            repo_path: Path to Git repository root
            min_quality_threshold: Minimum quality score required
            config_file: Optional configuration file path
        """
        self.repo_path = Path(repo_path)
        self.git_hooks_path = self.repo_path / ".git" / "hooks"
        self.min_quality_threshold = min_quality_threshold
        self.config_file = config_file

        # Initialize rating pipeline with default metrics
        default_metrics = [
            "consistency",
            "completeness",
            "expressiveness",
            "structural_coherence",
        ]
        self.pipeline = RatingPipeline(metric_names=default_metrics)
        self.file_discovery = FileDiscovery()

        # Hook templates
        self.pre_commit_template = self._get_pre_commit_template()
        self.pre_push_template = self._get_pre_push_template()

    def install_hooks(self) -> bool:
        """
        Install Git hooks for UDL quality checking.

        Returns:
            True if hooks installed successfully
        """
        try:
            # Ensure hooks directory exists
            self.git_hooks_path.mkdir(parents=True, exist_ok=True)

            # Install pre-commit hook
            pre_commit_path = self.git_hooks_path / "pre-commit"
            self._write_hook_script(pre_commit_path, self.pre_commit_template)

            # Install pre-push hook
            pre_push_path = self.git_hooks_path / "pre-push"
            self._write_hook_script(pre_push_path, self.pre_push_template)

            logger.info("Git hooks installed successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to install Git hooks: {e}")
            return False

    def uninstall_hooks(self) -> bool:
        """
        Remove UDL quality checking hooks.

        Returns:
            True if hooks removed successfully
        """
        try:
            hooks_to_remove = ["pre-commit", "pre-push"]

            for hook_name in hooks_to_remove:
                hook_path = self.git_hooks_path / hook_name
                if hook_path.exists():
                    # Check if it's our hook before removing
                    content = hook_path.read_text()
                    if "UDL_RATING_HOOK" in content:
                        hook_path.unlink()
                        logger.info(f"Removed {hook_name} hook")

            return True

        except Exception as e:
            logger.error(f"Failed to uninstall Git hooks: {e}")
            return False

    def check_staged_files(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Check quality of staged UDL files.

        Returns:
            Tuple of (passed, results)
        """
        try:
            # Get staged files
            staged_files = self._get_staged_udl_files()

            if not staged_files:
                return True, {"message": "No UDL files staged"}

            # Check quality of staged files
            results = {}
            all_passed = True

            for file_path in staged_files:
                try:
                    # Get staged content
                    content = self._get_staged_file_content(file_path)

                    # Create temporary file for analysis
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".udl", delete=False
                    ) as tmp:
                        tmp.write(content)
                        tmp_path = Path(tmp.name)

                    try:
                        # Analyze UDL
                        udl_repr = UDLRepresentation(content, str(file_path))
                        report = self.pipeline.rate_udl(udl_repr)

                        file_passed = report.overall_score >= self.min_quality_threshold
                        all_passed = all_passed and file_passed

                        results[str(file_path)] = {
                            "score": report.overall_score,
                            "confidence": report.confidence,
                            "passed": file_passed,
                            "metrics": report.metric_scores,
                        }

                    finally:
                        tmp_path.unlink()

                except Exception as e:
                    logger.error(f"Error checking {file_path}: {e}")
                    results[str(file_path)] = {
                        "error": str(e), "passed": False}
                    all_passed = False

            return all_passed, results

        except Exception as e:
            logger.error(f"Error checking staged files: {e}")
            return False, {"error": str(e)}

    def check_all_udl_files(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Check quality of all UDL files in repository.

        Returns:
            Tuple of (passed, results)
        """
        try:
            # Discover all UDL files
            udl_files = self.file_discovery.discover_files(
                self.repo_path, extensions=[
                    ".udl", ".dsl", ".grammar", ".ebnf"]
            )

            if not udl_files:
                return True, {"message": "No UDL files found"}

            # Check quality of all files
            results = {}
            all_passed = True

            for file_path in udl_files:
                try:
                    content = file_path.read_text(encoding="utf-8")
                    udl_repr = UDLRepresentation(content, str(file_path))
                    report = self.pipeline.rate_udl(udl_repr)

                    file_passed = report.overall_score >= self.min_quality_threshold
                    all_passed = all_passed and file_passed

                    results[str(file_path)] = {
                        "score": report.overall_score,
                        "confidence": report.confidence,
                        "passed": file_passed,
                        "metrics": report.metric_scores,
                    }

                except Exception as e:
                    logger.error(f"Error checking {file_path}: {e}")
                    results[str(file_path)] = {
                        "error": str(e), "passed": False}
                    all_passed = False

            return all_passed, results

        except Exception as e:
            logger.error(f"Error checking all UDL files: {e}")
            return False, {"error": str(e)}

    def _get_staged_udl_files(self) -> List[Path]:
        """Get list of staged UDL files."""
        try:
            result = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )

            staged_files = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    file_path = self.repo_path / line
                    if file_path.suffix.lower() in [
                        ".udl",
                        ".dsl",
                        ".grammar",
                        ".ebnf",
                    ]:
                        staged_files.append(file_path)

            return staged_files

        except subprocess.CalledProcessError as e:
            logger.error(f"Error getting staged files: {e}")
            return []

    def _get_staged_file_content(self, file_path: Path) -> str:
        """Get content of staged file."""
        try:
            result = subprocess.run(
                ["git", "show", f":{file_path.relative_to(self.repo_path)}"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout

        except subprocess.CalledProcessError as e:
            logger.error(f"Error getting staged content for {file_path}: {e}")
            # Fallback to current file content
            return file_path.read_text(encoding="utf-8")

    def _write_hook_script(self, hook_path: Path, template: str) -> None:
        """Write hook script to file."""
        # Get Python executable path
        python_path = sys.executable

        # Replace template variables
        script_content = template.format(
            python_path=python_path,
            repo_path=self.repo_path,
            min_quality_threshold=self.min_quality_threshold,
            config_file=self.config_file or "",
        )

        hook_path.write_text(script_content)
        hook_path.chmod(0o755)  # Make executable

    def _get_pre_commit_template(self) -> str:
        """Get pre-commit hook template."""
        return """#!/bin/bash
# UDL_RATING_HOOK - Pre-commit hook for UDL quality checking
# Generated by UDL Rating Framework

set -e

echo "Checking UDL quality..."

# Run UDL quality check
{python_path} -c "
import sys
sys.path.insert(0, '{repo_path}')
from udl_rating_framework.integration.git_hooks import GitHookManager
from pathlib import Path

manager = GitHookManager(
    repo_path=Path('{repo_path}'),
    min_quality_threshold={min_quality_threshold},
    config_file=Path('{config_file}') if '{config_file}' else None
)

passed, results = manager.check_staged_files()

if not passed:
    print('UDL quality check failed!')
    print('Files that failed quality threshold:')
    for fp, result in results.items():
        if not result.get('passed', False):
            score = result.get('score', 'N/A')
            print(f'  {{fp}}: {{score:.3f}} (threshold: {min_quality_threshold})')
    print()
    print('Please improve UDL quality or use --no-verify to bypass.')
    sys.exit(1)
else:
    print('All UDL files passed quality check!')
    for fp, result in results.items():
        if 'score' in result:
            print(f'  {{fp}}: {{result[\"score\"]:.3f}}')
"

echo "UDL quality check completed successfully."
"""

    def _get_pre_push_template(self) -> str:
        """Get pre-push hook template."""
        return """#!/bin/bash
# UDL_RATING_HOOK - Pre-push hook for UDL quality checking
# Generated by UDL Rating Framework

set -e

echo "Running comprehensive UDL quality check..."

# Run UDL quality check on all files
{python_path} -c "
import sys
sys.path.insert(0, '{repo_path}')
from udl_rating_framework.integration.git_hooks import GitHookManager
from pathlib import Path

manager = GitHookManager(
    repo_path=Path('{repo_path}'),
    min_quality_threshold={min_quality_threshold},
    config_file=Path('{config_file}') if '{config_file}' else None
)

passed, results = manager.check_all_udl_files()

if not passed:
    print('UDL quality check failed!')
    print('Files that failed quality threshold:')
    for fp, result in results.items():
        if not result.get('passed', False):
            score = result.get('score', 'N/A')
            print(f'  {{fp}}: {{score:.3f}} (threshold: {min_quality_threshold})')
    print()
    print('Please improve UDL quality or use --no-verify to bypass.')
    sys.exit(1)
else:
    print('All UDL files passed quality check!')
    total_files = len([r for r in results.values() if 'score' in r])
    avg_score = sum(r['score'] for r in results.values() if 'score' in r) / max(total_files, 1)
    print(f'Average quality score: {{avg_score:.3f}}')
"

echo "Comprehensive UDL quality check completed successfully."
"""


def main():
    """CLI entry point for Git hook management."""
    import argparse

    parser = argparse.ArgumentParser(description="UDL Git Hook Manager")
    parser.add_argument(
        "action", choices=["install", "uninstall", "check-staged", "check-all"]
    )
    parser.add_argument(
        "--repo-path", type=Path, default=Path.cwd(), help="Path to Git repository"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.7, help="Minimum quality threshold"
    )
    parser.add_argument("--config", type=Path, help="Configuration file path")

    args = parser.parse_args()

    manager = GitHookManager(
        repo_path=args.repo_path,
        min_quality_threshold=args.threshold,
        config_file=args.config,
    )

    if args.action == "install":
        success = manager.install_hooks()
        sys.exit(0 if success else 1)
    elif args.action == "uninstall":
        success = manager.uninstall_hooks()
        sys.exit(0 if success else 1)
    elif args.action == "check-staged":
        passed, results = manager.check_staged_files()
        print(json.dumps(results, indent=2))
        sys.exit(0 if passed else 1)
    elif args.action == "check-all":
        passed, results = manager.check_all_udl_files()
        print(json.dumps(results, indent=2))
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
