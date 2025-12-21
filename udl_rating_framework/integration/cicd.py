"""
CI/CD pipeline integration for UDL Rating Framework.

Provides integration with popular CI/CD systems including GitHub Actions,
Jenkins, GitLab CI, and others for automated UDL quality checking.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CICDConfig:
    """Configuration for CI/CD integration."""

    min_quality_threshold: float = 0.7
    fail_on_quality_drop: bool = True
    generate_reports: bool = True
    report_format: str = "json"  # json, xml, html
    artifact_retention_days: int = 30
    parallel_jobs: int = 4
    timeout_minutes: int = 30


class CICDIntegration:
    """
    CI/CD integration manager for UDL quality checking.

    Supports:
    - GitHub Actions workflows
    - Jenkins pipelines
    - GitLab CI/CD
    - Azure DevOps
    - Generic CI/CD systems
    """

    def __init__(self, config: Optional[CICDConfig] = None):
        """Initialize CI/CD integration."""
        self.config = config or CICDConfig()

    def generate_github_actions_workflow(
        self,
        workflow_name: str = "udl-quality-check",
        triggers: Optional[List[str]] = None,
    ) -> str:
        """
        Generate GitHub Actions workflow for UDL quality checking.

        Args:
            workflow_name: Name of the workflow
            triggers: List of triggers (push, pull_request, etc.)

        Returns:
            YAML workflow content
        """
        if triggers is None:
            triggers = ["push", "pull_request"]

        workflow = {
            "name": "UDL Quality Check",
            "on": triggers,
            "jobs": {
                "udl-quality": {
                    "runs-on": "ubuntu-latest",
                    "timeout-minutes": self.config.timeout_minutes,
                    "steps": [
                        {"name": "Checkout code", "uses": "actions/checkout@v4"},
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v4",
                            "with": {"python-version": "3.10"},
                        },
                        {"name": "Install dependencies", "run": "pip install -e ."},
                        {
                            "name": "Run UDL quality check",
                            "run": f"udl-rating rate . --threshold {self.config.min_quality_threshold} --format {self.config.report_format} --output udl-quality-report.{self.config.report_format}",
                        },
                        {
                            "name": "Upload quality report",
                            "uses": "actions/upload-artifact@v3",
                            "if": "always()",
                            "with": {
                                "name": f"udl-quality-report-{self.config.report_format}",
                                "path": f"udl-quality-report.{self.config.report_format}",
                                "retention-days": self.config.artifact_retention_days,
                            },
                        },
                    ],
                }
            },
        }

        # Add quality comparison for pull requests
        if "pull_request" in triggers:
            workflow["jobs"]["udl-quality-comparison"] = {
                "runs-on": "ubuntu-latest",
                "if": "github.event_name == 'pull_request'",
                "steps": [
                    {"name": "Checkout PR branch", "uses": "actions/checkout@v4"},
                    {
                        "name": "Set up Python",
                        "uses": "actions/setup-python@v4",
                        "with": {"python-version": "3.10"},
                    },
                    {"name": "Install dependencies", "run": "pip install -e ."},
                    {
                        "name": "Check PR UDL quality",
                        "run": f"udl-rating rate . --threshold {self.config.min_quality_threshold} --format json --output pr-quality.json",
                    },
                    {
                        "name": "Checkout base branch",
                        "run": "git fetch origin ${{ github.base_ref }} && git checkout origin/${{ github.base_ref }}",
                    },
                    {
                        "name": "Check base UDL quality",
                        "run": f"udl-rating rate . --threshold {self.config.min_quality_threshold} --format json --output base-quality.json",
                    },
                    {
                        "name": "Compare quality",
                        "run": "udl-rating compare base-quality.json pr-quality.json --output comparison-report.json",
                    },
                    {
                        "name": "Comment PR with results",
                        "uses": "actions/github-script@v6",
                        "with": {
                            "script": """
                            const fs = require('fs');
                            const comparison = JSON.parse(fs.readFileSync('comparison-report.json', 'utf8'));
                            
                            let comment = '## UDL Quality Check Results\\n\\n';
                            comment += `**Overall Quality Change**: ${comparison.overall_change > 0 ? '📈' : comparison.overall_change < 0 ? '📉' : '➡️'} ${comparison.overall_change.toFixed(3)}\\n\\n`;
                            
                            comment += '### File Changes\\n';
                            for (const [file, change] of Object.entries(comparison.file_changes)) {
                                const emoji = change.score_change > 0 ? '✅' : change.score_change < 0 ? '❌' : '➡️';
                                comment += `- ${emoji} \\`${file}\\`: ${change.score_change.toFixed(3)}\\n`;
                            }
                            
                            github.rest.issues.createComment({
                                issue_number: context.issue.number,
                                owner: context.repo.owner,
                                repo: context.repo.repo,
                                body: comment
                            });
                            """
                        },
                    },
                ],
            }

        return yaml.dump(workflow, default_flow_style=False, sort_keys=False)

    def generate_jenkins_pipeline(
        self, pipeline_name: str = "UDL Quality Check"
    ) -> str:
        """
        Generate Jenkins pipeline for UDL quality checking.

        Args:
            pipeline_name: Name of the pipeline

        Returns:
            Jenkinsfile content
        """
        jenkinsfile = f"""
pipeline {{
    agent any
    
    options {{
        timeout(time: {self.config.timeout_minutes}, unit: 'MINUTES')
        buildDiscarder(logRotator(daysToKeepStr: '{self.config.artifact_retention_days}'))
    }}
    
    environment {{
        UDL_QUALITY_THRESHOLD = '{self.config.min_quality_threshold}'
        REPORT_FORMAT = '{self.config.report_format}'
    }}
    
    stages {{
        stage('Checkout') {{
            steps {{
                checkout scm
            }}
        }}
        
        stage('Setup Python Environment') {{
            steps {{
                sh \"\"\"
                python3 -m venv venv
                . venv/bin/activate
                pip install --upgrade pip
                pip install -e .
                \"\"\"
            }}
        }}
        
        stage('UDL Quality Check') {{
            parallel {{
                stage('Current Branch Quality') {{
                    steps {{
                        sh \"\"\"
                        . venv/bin/activate
                        udl-rating rate . --threshold $$UDL_QUALITY_THRESHOLD --format $$REPORT_FORMAT --output current-quality.$$REPORT_FORMAT
                        \"\"\"
                    }}
                }}
                
                stage('Baseline Quality') {{
                    when {{
                        not {{ branch 'main' }}
                    }}
                    steps {{
                        sh \"\"\"
                        git fetch origin main
                        git checkout origin/main
                        . venv/bin/activate
                        udl-rating rate . --threshold $$UDL_QUALITY_THRESHOLD --format json --output baseline-quality.json
                        git checkout -
                        \"\"\"
                    }}
                }}
            }}
        }}
        
        stage('Quality Comparison') {{
            when {{
                not {{ branch 'main' }}
            }}
            steps {{
                sh \"\"\"
                . venv/bin/activate
                if [ -f baseline-quality.json ] && [ -f current-quality.json ]; then
                    udl-rating compare baseline-quality.json current-quality.json --output comparison-report.json
                fi
                \"\"\"
            }}
        }}
        
        stage('Generate Reports') {{
            steps {{
                sh \"\"\"
                . venv/bin/activate
                # Generate HTML report if not already in HTML format
                if [ "$$REPORT_FORMAT" != "html" ]; then
                    udl-rating rate . --format html --output udl-quality-report.html
                fi
                \"\"\"
            }}
        }}
    }}
    
    post {{
        always {{
            archiveArtifacts artifacts: '*.json,*.html,*.xml', allowEmptyArchive: true
            
            script {{
                if (fileExists('current-quality.json')) {{
                    def qualityData = readJSON file: 'current-quality.json'
                    def overallScore = qualityData.overall_score
                    
                    if (overallScore < {self.config.min_quality_threshold}) {{
                        currentBuild.result = 'FAILURE'
                        error("UDL quality score ${{overallScore}} is below threshold {self.config.min_quality_threshold}")
                    }}
                }}
            }}
        }}
        
        success {{
            echo 'UDL quality check passed!'
        }}
        
        failure {{
            echo 'UDL quality check failed!'
        }}
    }}
}}
"""
        return jenkinsfile.strip()

    def generate_gitlab_ci(self) -> str:
        """
        Generate GitLab CI/CD configuration for UDL quality checking.

        Returns:
            .gitlab-ci.yml content
        """
        gitlab_ci = {
            "stages": ["test", "quality", "report"],
            "variables": {
                "UDL_QUALITY_THRESHOLD": str(self.config.min_quality_threshold),
                "REPORT_FORMAT": self.config.report_format,
                "PIP_CACHE_DIR": "$CI_PROJECT_DIR/.cache/pip",
            },
            "cache": {"paths": [".cache/pip/", "venv/"]},
            "before_script": [
                "python3 -m venv venv",
                "source venv/bin/activate",
                "pip install --upgrade pip",
                "pip install -e .",
            ],
            "udl_quality_check": {
                "stage": "quality",
                "script": [
                    "source venv/bin/activate",
                    "udl-rating rate . --threshold $UDL_QUALITY_THRESHOLD --format $REPORT_FORMAT --output udl-quality-report.$REPORT_FORMAT",
                ],
                "artifacts": {
                    "reports": {
                        "junit": (
                            "udl-quality-report.xml"
                            if self.config.report_format == "xml"
                            else None
                        )
                    },
                    "paths": [f"udl-quality-report.{self.config.report_format}"],
                    "expire_in": f"{self.config.artifact_retention_days} days",
                },
                "only": ["merge_requests", "main", "develop"],
            },
        }

        # Add quality comparison for merge requests
        gitlab_ci["udl_quality_comparison"] = {
            "stage": "quality",
            "script": [
                "source venv/bin/activate",
                "udl-rating rate . --format json --output mr-quality.json",
                "git fetch origin $CI_MERGE_REQUEST_TARGET_BRANCH_NAME",
                "git checkout origin/$CI_MERGE_REQUEST_TARGET_BRANCH_NAME",
                "udl-rating rate . --format json --output target-quality.json",
                "git checkout $CI_COMMIT_SHA",
                "udl-rating compare target-quality.json mr-quality.json --output comparison-report.json",
            ],
            "artifacts": {
                "paths": ["comparison-report.json"],
                "expire_in": f"{self.config.artifact_retention_days} days",
            },
            "only": ["merge_requests"],
        }

        return yaml.dump(gitlab_ci, default_flow_style=False, sort_keys=False)

    def generate_azure_devops_pipeline(self) -> str:
        """
        Generate Azure DevOps pipeline for UDL quality checking.

        Returns:
            azure-pipelines.yml content
        """
        azure_pipeline = {
            "trigger": ["main", "develop"],
            "pr": ["main", "develop"],
            "pool": {"vmImage": "ubuntu-latest"},
            "variables": {
                "UDL_QUALITY_THRESHOLD": self.config.min_quality_threshold,
                "REPORT_FORMAT": self.config.report_format,
                "python.version": "3.10",
            },
            "jobs": [
                {
                    "job": "UDLQualityCheck",
                    "displayName": "UDL Quality Check",
                    "timeoutInMinutes": self.config.timeout_minutes,
                    "steps": [
                        {
                            "task": "UsePythonVersion@0",
                            "inputs": {"versionSpec": "$(python.version)"},
                            "displayName": "Use Python $(python.version)",
                        },
                        {
                            "script": """
                            python -m pip install --upgrade pip
                            pip install -e .
                            """,
                            "displayName": "Install dependencies",
                        },
                        {
                            "script": "udl-rating rate . --threshold $(UDL_QUALITY_THRESHOLD) --format $(REPORT_FORMAT) --output udl-quality-report.$(REPORT_FORMAT)",
                            "displayName": "Run UDL quality check",
                        },
                        {
                            "task": "PublishTestResults@2",
                            "condition": "always()",
                            "inputs": {
                                "testResultsFiles": "udl-quality-report.xml",
                                "testRunTitle": "UDL Quality Results",
                            },
                            "displayName": "Publish test results",
                        },
                        {
                            "task": "PublishBuildArtifacts@1",
                            "condition": "always()",
                            "inputs": {
                                "pathToPublish": f"udl-quality-report.{self.config.report_format}",
                                "artifactName": "udl-quality-report",
                            },
                            "displayName": "Publish quality report",
                        },
                    ],
                }
            ],
        }

        return yaml.dump(azure_pipeline, default_flow_style=False, sort_keys=False)

    def create_workflow_files(
        self, target_dir: Path, platforms: Optional[List[str]] = None
    ) -> Dict[str, Path]:
        """
        Create CI/CD workflow files for specified platforms.

        Args:
            target_dir: Directory to create workflow files in
            platforms: List of platforms ('github', 'jenkins', 'gitlab', 'azure')

        Returns:
            Dictionary mapping platform names to created file paths
        """
        if platforms is None:
            platforms = ["github"]

        created_files = {}

        for platform in platforms:
            if platform == "github":
                workflow_dir = target_dir / ".github" / "workflows"
                workflow_dir.mkdir(parents=True, exist_ok=True)

                workflow_file = workflow_dir / "udl-quality-check.yml"
                workflow_content = self.generate_github_actions_workflow()
                workflow_file.write_text(workflow_content)
                created_files["github"] = workflow_file

            elif platform == "jenkins":
                jenkinsfile = target_dir / "Jenkinsfile"
                jenkins_content = self.generate_jenkins_pipeline()
                jenkinsfile.write_text(jenkins_content)
                created_files["jenkins"] = jenkinsfile

            elif platform == "gitlab":
                gitlab_file = target_dir / ".gitlab-ci.yml"
                gitlab_content = self.generate_gitlab_ci()
                gitlab_file.write_text(gitlab_content)
                created_files["gitlab"] = gitlab_file

            elif platform == "azure":
                azure_file = target_dir / "azure-pipelines.yml"
                azure_content = self.generate_azure_devops_pipeline()
                azure_file.write_text(azure_content)
                created_files["azure"] = azure_file

        return created_files

    def validate_workflow_syntax(self, workflow_file: Path) -> bool:
        """
        Validate workflow file syntax.

        Args:
            workflow_file: Path to workflow file

        Returns:
            True if syntax is valid
        """
        try:
            content = workflow_file.read_text()

            if workflow_file.suffix in [".yml", ".yaml"]:
                yaml.safe_load(content)
            elif workflow_file.name == "Jenkinsfile":
                # Basic Jenkinsfile validation (could be enhanced)
                required_sections = ["pipeline", "stages", "steps"]
                for section in required_sections:
                    if section not in content:
                        logger.warning(
                            f"Missing required section '{section}' in Jenkinsfile"
                        )
                        return False

            return True

        except Exception as e:
            logger.error(f"Workflow syntax validation failed: {e}")
            return False


def main():
    """CLI entry point for CI/CD integration."""
    import argparse

    parser = argparse.ArgumentParser(description="UDL CI/CD Integration")
    parser.add_argument("action", choices=["generate", "validate"])
    parser.add_argument(
        "--platform",
        choices=["github", "jenkins", "gitlab", "azure"],
        action="append",
        help="CI/CD platform(s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Output directory for workflow files",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.7, help="Minimum quality threshold"
    )
    parser.add_argument("--config", type=Path, help="Configuration file path")

    args = parser.parse_args()

    # Load configuration
    config = CICDConfig()
    if args.config and args.config.exists():
        with open(args.config) as f:
            config_data = yaml.safe_load(f)
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)

    config.min_quality_threshold = args.threshold

    integration = CICDIntegration(config)

    if args.action == "generate":
        platforms = args.platform or ["github"]
        created_files = integration.create_workflow_files(args.output_dir, platforms)

        print("Created CI/CD workflow files:")
        for platform, file_path in created_files.items():
            print(f"  {platform}: {file_path}")

    elif args.action == "validate":
        # Validate existing workflow files
        workflow_files = []

        # Find workflow files
        github_workflows = args.output_dir / ".github" / "workflows"
        if github_workflows.exists():
            workflow_files.extend(github_workflows.glob("*.yml"))
            workflow_files.extend(github_workflows.glob("*.yaml"))

        for filename in ["Jenkinsfile", ".gitlab-ci.yml", "azure-pipelines.yml"]:
            file_path = args.output_dir / filename
            if file_path.exists():
                workflow_files.append(file_path)

        all_valid = True
        for workflow_file in workflow_files:
            is_valid = integration.validate_workflow_syntax(workflow_file)
            status = "✅ Valid" if is_valid else "❌ Invalid"
            print(f"{status}: {workflow_file}")
            all_valid = all_valid and is_valid

        if not all_valid:
            exit(1)


if __name__ == "__main__":
    main()
