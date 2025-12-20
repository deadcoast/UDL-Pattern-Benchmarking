#!/usr/bin/env python3
"""
Example script demonstrating Git hooks integration for UDL quality checking.

This script shows how to set up and use Git hooks to automatically check
UDL quality before commits and pushes.
"""

from pathlib import Path
from udl_rating_framework.integration.git_hooks import GitHookManager


def main():
    """Demonstrate Git hooks integration."""
    print("UDL Rating Framework - Git Hooks Integration Example")
    print("=" * 60)

    # Initialize Git hook manager
    repo_path = Path.cwd()
    min_quality_threshold = 0.7

    manager = GitHookManager(
        repo_path=repo_path, min_quality_threshold=min_quality_threshold
    )

    print(f"Repository path: {repo_path}")
    print(f"Quality threshold: {min_quality_threshold}")
    print()

    # Install hooks
    print("Installing Git hooks...")
    if manager.install_hooks():
        print("✅ Git hooks installed successfully!")
        print()
        print("The following hooks have been installed:")
        print("- pre-commit: Checks staged UDL files before commit")
        print("- pre-push: Checks all UDL files before push")
        print()
        print("To bypass hooks temporarily, use:")
        print("  git commit --no-verify")
        print("  git push --no-verify")
    else:
        print("❌ Failed to install Git hooks")
        return

    # Demonstrate checking staged files
    print("\nChecking staged files...")
    passed, results = manager.check_staged_files()

    if "message" in results:
        print(results["message"])
    else:
        print("Staged file results:")
        for file_path, result in results.items():
            if result.get("passed", False):
                print(f"✅ {file_path}: {result['score']:.3f}")
            else:
                error = result.get("error", "Quality below threshold")
                score = result.get("score", "N/A")
                print(f"❌ {file_path}: {score} ({error})")

    # Demonstrate checking all files
    print("\nChecking all UDL files...")
    passed, results = manager.check_all_udl_files()

    if "message" in results:
        print(results["message"])
    else:
        successful_files = [r for r in results.values() if "score" in r]
        if successful_files:
            avg_score = sum(r["score"] for r in successful_files) / len(
                successful_files
            )
            print(f"Average quality score: {avg_score:.3f}")
            print(f"Files processed: {len(successful_files)}")
            print(f"Files failed: {len(results) - len(successful_files)}")
        else:
            print("No files processed successfully")

    print("\nGit hooks integration example completed!")


if __name__ == "__main__":
    main()
