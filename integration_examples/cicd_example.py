#!/usr/bin/env python3
"""
Example script demonstrating CI/CD integration for UDL quality checking.

This script shows how to generate CI/CD workflow files for different platforms
and configure automated UDL quality checking in your CI/CD pipelines.
"""

from pathlib import Path
from udl_rating_framework.integration.cicd import CICDIntegration, CICDConfig


def main():
    """Demonstrate CI/CD integration."""
    print("UDL Rating Framework - CI/CD Integration Example")
    print("=" * 60)
    
    # Create configuration
    config = CICDConfig(
        min_quality_threshold=0.7,
        fail_on_quality_drop=True,
        generate_reports=True,
        report_format='json',
        artifact_retention_days=30,
        parallel_jobs=4,
        timeout_minutes=30
    )
    
    print("CI/CD Configuration:")
    print(f"  Quality threshold: {config.min_quality_threshold}")
    print(f"  Fail on quality drop: {config.fail_on_quality_drop}")
    print(f"  Report format: {config.report_format}")
    print(f"  Artifact retention: {config.artifact_retention_days} days")
    print(f"  Parallel jobs: {config.parallel_jobs}")
    print(f"  Timeout: {config.timeout_minutes} minutes")
    print()
    
    # Initialize CI/CD integration
    integration = CICDIntegration(config)
    
    # Generate workflow files for different platforms
    output_dir = Path('generated_workflows')
    output_dir.mkdir(exist_ok=True)
    
    platforms = ['github', 'jenkins', 'gitlab', 'azure']
    
    print("Generating CI/CD workflow files...")
    created_files = integration.create_workflow_files(output_dir, platforms)
    
    print("\nGenerated workflow files:")
    for platform, file_path in created_files.items():
        print(f"  {platform}: {file_path}")
    
    # Demonstrate individual workflow generation
    print("\nGenerating individual workflows...")
    
    # GitHub Actions
    print("\n1. GitHub Actions Workflow:")
    github_workflow = integration.generate_github_actions_workflow(
        workflow_name='udl-quality-check',
        triggers=['push', 'pull_request']
    )
    print("Generated GitHub Actions workflow with:")
    print("- Quality checking on push and pull requests")
    print("- Artifact upload for reports")
    print("- PR comment with quality comparison")
    
    # Jenkins Pipeline
    print("\n2. Jenkins Pipeline:")
    jenkins_pipeline = integration.generate_jenkins_pipeline('UDL Quality Check')
    print("Generated Jenkins pipeline with:")
    print("- Parallel quality checking")
    print("- Baseline comparison for feature branches")
    print("- HTML report generation")
    
    # GitLab CI
    print("\n3. GitLab CI/CD:")
    gitlab_ci = integration.generate_gitlab_ci()
    print("Generated GitLab CI configuration with:")
    print("- Quality checking stage")
    print("- Merge request comparison")
    print("- Artifact management")
    
    # Azure DevOps
    print("\n4. Azure DevOps Pipeline:")
    azure_pipeline = integration.generate_azure_devops_pipeline()
    print("Generated Azure DevOps pipeline with:")
    print("- Multi-stage pipeline")
    print("- Test result publishing")
    print("- Build artifact management")
    
    # Validate generated workflows
    print("\nValidating generated workflows...")
    for platform, file_path in created_files.items():
        is_valid = integration.validate_workflow_syntax(file_path)
        status = "✅ Valid" if is_valid else "❌ Invalid"
        print(f"  {status}: {platform} ({file_path.name})")
    
    print("\nCI/CD integration example completed!")
    print("\nNext steps:")
    print("1. Copy the appropriate workflow file to your repository")
    print("2. Customize the configuration as needed")
    print("3. Commit and push to trigger the workflow")
    print("4. Monitor quality reports in your CI/CD system")


if __name__ == '__main__':
    main()