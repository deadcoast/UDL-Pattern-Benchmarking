# UDL Rating Framework - Integration Guide

This guide covers the integration and workflow features of the UDL Rating Framework, including Git hooks, CI/CD pipelines, IDE plugins, Language Server Protocol (LSP) support, and batch processing capabilities.

## Table of Contents

1. [Git Hooks Integration](#git-hooks-integration)
2. [CI/CD Pipeline Integration](#cicd-pipeline-integration)
3. [Language Server Protocol (LSP)](#language-server-protocol-lsp)
4. [IDE Plugins](#ide-plugins)
5. [Batch Processing](#batch-processing)
6. [Configuration](#configuration)
7. [Troubleshooting](#troubleshooting)

## Git Hooks Integration

Git hooks provide automatic UDL quality checking before commits and pushes, ensuring code quality standards are maintained.

### Installation

Install Git hooks for automatic quality checking:

```bash
# Install hooks with default settings
udl-rating integration git install-hooks

# Install with custom threshold and repository path
udl-rating integration git install-hooks --repo-path /path/to/repo --threshold 0.8

# Install with configuration file
udl-rating integration git install-hooks --config config.yaml
```

### Features

- **Pre-commit Hook**: Checks staged UDL files before allowing commits
- **Pre-push Hook**: Validates all UDL files before pushing to remote
- **Configurable Thresholds**: Set minimum quality scores required
- **Bypass Options**: Use `--no-verify` to bypass hooks when needed

### Usage Examples

```bash
# Check staged files manually
udl-rating integration git check-staged --threshold 0.7

# Uninstall hooks
udl-rating integration git uninstall-hooks

# Bypass hooks for emergency commits
git commit --no-verify -m "Emergency fix"
```

### Hook Behavior

1. **Pre-commit**: Analyzes only staged UDL files
2. **Pre-push**: Analyzes all UDL files in repository
3. **Quality Threshold**: Prevents commits/pushes if quality is below threshold
4. **Error Reporting**: Provides detailed feedback on quality issues

## CI/CD Pipeline Integration

Automated UDL quality checking in your CI/CD pipelines with support for multiple platforms.

### Supported Platforms

- **GitHub Actions**
- **Jenkins**
- **GitLab CI/CD**
- **Azure DevOps**

### Generate Workflow Files

```bash
# Generate GitHub Actions workflow
udl-rating integration cicd generate --platform github --output-dir .

# Generate multiple platform workflows
udl-rating integration cicd generate --platform github --platform jenkins --output-dir workflows/

# Customize settings
udl-rating integration cicd generate \
  --platform github \
  --threshold 0.8 \
  --timeout 45 \
  --retention-days 14
```

### GitHub Actions Example

Generated workflow includes:

- Quality checking on push and pull requests
- Artifact upload for reports
- PR comments with quality comparison
- Parallel job execution

```yaml
name: UDL Quality Check
on: [push, pull_request]
jobs:
  udl-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -e .
      - run: udl-rating rate . --threshold 0.7 --format json --output udl-quality-report.json
      - uses: actions/upload-artifact@v3
        with:
          name: udl-quality-report
          path: udl-quality-report.json
```

### Jenkins Pipeline Example

Generated Jenkinsfile includes:

- Parallel quality checking stages
- Baseline comparison for feature branches
- HTML report generation
- Quality gate enforcement

### Validation

Validate existing workflow files:

```bash
# Validate all workflow files in directory
udl-rating integration cicd validate --workflow-dir .

# Check specific workflow syntax
udl-rating integration cicd validate --workflow-dir .github/workflows/
```

## Language Server Protocol (LSP)

Real-time UDL quality feedback in IDEs and editors that support LSP.

### Starting the LSP Server

```bash
# Start with default settings
udl-rating integration lsp-server

# Customize settings
udl-rating integration lsp-server \
  --threshold 0.8 \
  --real-time \
  --debounce 1.0 \
  --log-level DEBUG
```

### Features

- **Real-time Diagnostics**: Quality issues highlighted as you type
- **Hover Information**: Quality metrics shown on hover
- **Code Actions**: Quick fixes and quality improvements
- **Document Symbols**: UDL structure navigation
- **Configurable Thresholds**: Customize warning levels

### IDE Integration

The LSP server works with any editor that supports the Language Server Protocol:

- **VS Code**: Use with generic LSP extension
- **Vim/Neovim**: Use with LSP plugins like `nvim-lspconfig`
- **Emacs**: Use with `lsp-mode`
- **Sublime Text**: Use with LSP package

### VS Code Configuration

```json
{
  "languageServerExample.trace.server": "verbose",
  "languageServerExample.udlRating.enable": true,
  "languageServerExample.udlRating.threshold": 0.7
}
```

## IDE Plugins

Native IDE plugins for popular development environments.

### Generate Plugins

```bash
# Generate VS Code extension
udl-rating integration ide generate --plugin-type vscode --output-dir plugins/

# Generate IntelliJ IDEA plugin
udl-rating integration ide generate --plugin-type intellij --output-dir plugins/

# Generate Vim plugin
udl-rating integration ide generate --plugin-type vim --output-dir plugins/
```

### VS Code Extension

Features:
- Syntax highlighting for UDL files
- Real-time quality checking
- Status bar quality indicator
- Command palette integration
- Configurable settings

Installation:
```bash
cd plugins/udl-rating-vscode
npm install
npm run compile
vsce package
code --install-extension *.vsix
```

### IntelliJ IDEA Plugin

Features:
- File type recognition
- Syntax highlighting
- Code inspections
- Tool window integration
- Quick fixes

Installation:
```bash
cd plugins/udl-rating-intellij
./gradlew buildPlugin
# Install through IntelliJ IDEA plugin manager
```

### Vim Plugin

Features:
- Syntax highlighting
- Real-time quality checking
- Status line integration
- Key mappings
- Command integration

Installation:
```bash
# Copy plugin files to Vim configuration directory
cp -r plugins/udl-rating-vim/* ~/.vim/
```

## Batch Processing

Efficient processing of large numbers of UDL files with parallel execution and progress tracking.

### Basic Usage

```bash
# Process directory
udl-rating integration batch-process /path/to/udl/files --output results.json

# Process with custom settings
udl-rating integration batch-process /path/to/udl/files \
  --output results.json \
  --format html \
  --workers 8 \
  --chunk-size 20 \
  --timeout 60
```

### Advanced Options

```bash
# Enable caching for better performance
udl-rating integration batch-process /path/to/udl/files \
  --output results.json \
  --cache-dir /tmp/udl-cache

# Use streaming for very large datasets
udl-rating integration batch-process /path/to/udl/files \
  --output results.json \
  --streaming

# Filter files with patterns
udl-rating integration batch-process /path/to/udl/files \
  --output results.json \
  --include-patterns "*.udl" "*.dsl" \
  --exclude-patterns "*test*" "*temp*"
```

### Output Formats

- **JSON**: Detailed results with full metric data
- **CSV**: Tabular format for spreadsheet analysis
- **HTML**: Interactive report with visualizations

### Performance Features

- **Parallel Processing**: Configurable worker processes
- **Caching**: Avoid reprocessing unchanged files
- **Streaming**: Handle very large datasets efficiently
- **Progress Tracking**: Real-time progress updates
- **Error Handling**: Continue processing despite individual file errors

## Configuration

### Configuration File Format

Create a YAML configuration file for consistent settings:

```yaml
# udl-rating-config.yaml
quality_threshold: 0.7
enable_caching: true
cache_directory: "/tmp/udl-rating-cache"

git_hooks:
  pre_commit_enabled: true
  pre_push_enabled: true
  fail_on_quality_drop: true

cicd:
  timeout_minutes: 30
  artifact_retention_days: 30
  parallel_jobs: 4
  report_format: "json"

lsp:
  enable_real_time: true
  debounce_delay: 0.5
  show_detailed_metrics: true

batch_processing:
  max_workers: 8
  chunk_size: 10
  memory_limit_mb: 1024
  error_handling: "continue"
  max_retries: 3
```

### Environment Variables

Set environment variables for global configuration:

```bash
export UDL_RATING_THRESHOLD=0.8
export UDL_RATING_CACHE_DIR=/tmp/udl-cache
export UDL_RATING_LOG_LEVEL=INFO
```

## Troubleshooting

### Common Issues

#### Git Hooks Not Working

1. **Check Hook Installation**:
   ```bash
   ls -la .git/hooks/
   # Should show pre-commit and pre-push with execute permissions
   ```

2. **Verify Hook Content**:
   ```bash
   head -5 .git/hooks/pre-commit
   # Should contain UDL_RATING_HOOK marker
   ```

3. **Test Hook Manually**:
   ```bash
   .git/hooks/pre-commit
   ```

#### CI/CD Pipeline Failures

1. **Check Dependencies**:
   - Ensure Python 3.10+ is available
   - Verify UDL Rating Framework is installed
   - Check for required system packages

2. **Validate Workflow Syntax**:
   ```bash
   udl-rating integration cicd validate
   ```

3. **Review Logs**:
   - Check CI/CD system logs for detailed error messages
   - Look for timeout issues or resource constraints

#### LSP Server Issues

1. **Check Server Status**:
   ```bash
   udl-rating integration lsp-server --log-level DEBUG
   ```

2. **Verify IDE Configuration**:
   - Ensure LSP client is properly configured
   - Check server connection settings
   - Verify file type associations

#### Performance Issues

1. **Enable Caching**:
   ```bash
   udl-rating integration batch-process --cache-dir /tmp/cache
   ```

2. **Adjust Worker Count**:
   ```bash
   udl-rating integration batch-process --workers 4
   ```

3. **Use Streaming for Large Datasets**:
   ```bash
   udl-rating integration batch-process --streaming
   ```

### Getting Help

1. **Check Logs**: Enable debug logging for detailed information
2. **Validate Configuration**: Use validation commands to check setup
3. **Test Components**: Test individual components in isolation
4. **Review Documentation**: Check specific integration guides
5. **Report Issues**: Submit bug reports with detailed error information

### Best Practices

1. **Start Simple**: Begin with basic configurations and gradually add complexity
2. **Test Locally**: Validate integrations locally before deploying
3. **Monitor Performance**: Track processing times and resource usage
4. **Use Caching**: Enable caching for repeated processing
5. **Configure Thresholds**: Set appropriate quality thresholds for your project
6. **Regular Updates**: Keep the framework updated for latest features and fixes

## Examples

See the `integration_examples/` directory for complete working examples:

- `git_hooks_example.py`: Git hooks integration demonstration
- `cicd_example.py`: CI/CD workflow generation examples
- `batch_processing_example.py`: Large-scale processing examples

Each example includes detailed comments and can be run independently to demonstrate the integration features.