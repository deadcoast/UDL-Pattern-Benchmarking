# UDL Pattern Benchmarking

A mathematically-grounded system for evaluating the quality of User Defined Languages (UDLs), built on the Continuous Thought Machine (CTM) architecture.

<!-- TOC -->

- [UDL Pattern Benchmarking](#udl-pattern-benchmarking)
    - [Overview](#overview)
    - [Features](#features)
    - [Installation](#installation)
        - [Prerequisites](#prerequisites)
        - [From Source](#from-source)
        - [Development Installation](#development-installation)
        - [Using pip when published](#using-pip-when-published)
    - [Quick Start](#quick-start)
        - [Rating a UDL File](#rating-a-udl-file)
        - [Command-Line Interface](#command-line-interface)
    - [Project Structure](#project-structure)
    - [Mathematical Foundation](#mathematical-foundation)
        - [Consistency Metric](#consistency-metric)
        - [Completeness Metric](#completeness-metric)
        - [Overall Quality Score](#overall-quality-score)
    - [Development](#development)
        - [Running Tests](#running-tests)
        - [Code Quality](#code-quality)
    - [Requirements](#requirements)
    - [Documentation](#documentation)
    - [Contributing](#contributing)
    - [License](#license)
    - [Acknowledgments](#acknowledgments)
    - [Status](#status)

<!-- /TOC -->

## Overview

The UDL Rating Framework provides objective, reproducible quality assessments of domain-specific languages through formal mathematical metrics. Every rating is traceable to rigorous mathematical foundations, eliminating subjective judgments.

## Features

- **Mathematical Rigor**: All metrics are formally defined with proven properties
- **Core Quality Dimensions**:
  - **Consistency**: Internal coherence using graph-theoretic analysis
  - **Completeness**: Construct coverage using set theory
  - **Expressiveness**: Language power via Chomsky hierarchy classification
  - **Structural Coherence**: Organizational quality using information theory
- **Advanced Metrics**:
  - **Semantic Similarity**: Cross-language semantic comparison
  - **Readability**: Code readability assessment
  - **Maintainability**: Long-term maintenance cost estimation
  - **Cross-Language Compatibility**: Multi-language support analysis
  - **Evolution Tracking**: Version-to-version change analysis
- **Confidence Scores**: Entropy-based certainty measures
- **CTM Integration**: Neural approximation for fast inference via CTM adapter
- **Comprehensive Testing**: Property-based and unit testing (652+ tests)
- **Analytics & Visualization**: Real-time metrics, web dashboards, and WebGL visualizations
- **Training Pipeline**: Active learning, ensemble methods, hyperparameter optimization
- **Integration Tools**: CI/CD, Git hooks, IDE plugins, LSP server

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable Python package management.

### Prerequisites

First, install uv if you haven't already:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# macOS homebrew
brew install uv

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/udl-rating-framework.git
cd udl-rating-framework

# Create virtual environment and install dependencies
uv sync

# The virtual environment is automatically activated when using uv run
# Or manually activate with:
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Development Installation

```bash
# Install with development dependencies
uv sync --extra dev --extra docs

# Or install specific groups
uv sync --extra dev  # Just development tools
uv sync --extra docs # Just documentation tools
```

### Using pip (when published)

```bash
pip install continuous-thought-machines
```

## Quick Start

### Rating a UDL File

```python
from udl_rating_framework import UDLRepresentation
from udl_rating_framework.core.metrics import ConsistencyMetric, CompletenessMetric

# Define a simple UDL grammar
udl_content = """
grammar MyLanguage;
program: statement+;
statement: assignment | expression;
assignment: ID '=' expression;
expression: term (('+' | '-') term)*;
term: factor (('*' | '/') factor)*;
factor: ID | NUMBER | '(' expression ')';
ID: [a-zA-Z_][a-zA-Z0-9_]*;
NUMBER: [0-9]+;
"""

# Create UDL representation
udl = UDLRepresentation(udl_content, "my_language.udl")

# Compute metrics
consistency = ConsistencyMetric()
completeness = CompletenessMetric()

print(f"Consistency: {consistency.compute(udl):.3f}")
print(f"Completeness: {completeness.compute(udl):.3f}")
```

### Command-Line Interface

```bash
# Show available commands
udl-rating --help

# Rate a single UDL file
udl-rating rate my_language.udl

# Rate a directory of UDL files (recursively)
udl-rating rate ./my_udls --recursive --output report.json

# Rate with custom metric weights
udl-rating rate ./my_udls --consistency-weight 0.4 --output results.json

# Train a CTM model
udl-rating train ./training_data --epochs 50 --output-dir ./checkpoints

# Compare multiple UDLs
udl-rating compare udl_v1.dsl udl_v2.dsl --output comparison.json

# Compare directories of UDLs
udl-rating compare ./lang1_versions/ ./lang2_versions/ --recursive

# Evaluate model performance
udl-rating evaluate ./checkpoints/model.pt ./test_data --output evaluation.json

# Analytics commands
udl-rating analytics --help

# Integration commands (CI/CD, Git hooks)
udl-rating integration --help
```

## Project Structure

```
udl_rating_framework/
├── __init__.py           # Package initialization and exports
├── core/                 # Core components
│   ├── __init__.py
│   ├── representation.py # UDL representation
│   ├── pipeline.py       # Processing pipeline
│   ├── aggregation.py    # Metric aggregation
│   ├── confidence.py     # Confidence calculation
│   ├── caching.py        # Result caching
│   ├── performance.py    # Performance utilities
│   ├── multiprocessing.py # Parallel processing
│   ├── distributed.py    # Distributed computing
│   ├── streaming.py      # Streaming processing
│   ├── incremental.py    # Incremental updates
│   ├── memory_mapping.py # Memory-mapped files
│   ├── gpu_acceleration.py # GPU acceleration
│   └── metrics/          # Quality metrics
│       ├── __init__.py
│       ├── base.py       # Base metric class and registry
│       ├── consistency.py
│       ├── completeness.py
│       ├── expressiveness.py
│       ├── structural_coherence.py
│       ├── semantic_similarity.py
│       ├── readability.py
│       ├── maintainability.py
│       ├── cross_language_compatibility.py
│       └── evolution_tracking.py
├── models/               # CTM model components
│   ├── __init__.py
│   └── ctm_adapter.py    # CTM integration adapter
├── io/                   # Input/output handling
│   ├── __init__.py
│   ├── file_discovery.py # File discovery utilities
│   ├── input_validation.py # Input validation
│   └── report_generator.py # Report generation
├── evaluation/           # Evaluation utilities
│   ├── __init__.py
│   ├── comparison.py     # UDL comparison
│   └── evaluation_suite.py # Evaluation suite
├── analytics/            # Analytics and reporting
│   ├── __init__.py
│   ├── portfolio_analyzer.py # Portfolio analysis
│   ├── trend_predictor.py # Trend prediction
│   ├── time_series_analyzer.py # Time series analysis
│   ├── improvement_advisor.py # Improvement suggestions
│   └── bi_exporter.py    # BI tool export
├── training/             # ML training components
│   ├── __init__.py
│   ├── training_pipeline.py # Training pipeline
│   ├── active_learning.py # Active learning
│   ├── ensemble_methods.py # Ensemble methods
│   ├── hyperparameter_optimization.py # Hyperparameter tuning
│   ├── transfer_learning.py # Transfer learning
│   └── uncertainty_quantification.py # Uncertainty estimation
├── visualization/        # Visualization tools
│   ├── __init__.py
│   ├── web_visualizer.py # Web dashboards
│   ├── realtime_metrics.py # Real-time metrics
│   ├── webgl_visualizer.py # WebGL 3D visualizations
│   ├── activation_visualizer.py # Activation visualization
│   └── synchronization_visualizer.py # Sync visualization
├── integration/          # Integration tools
│   ├── __init__.py
│   ├── cicd.py           # CI/CD integration
│   ├── git_hooks.py      # Git hooks
│   ├── lsp_server.py     # LSP server
│   ├── ide_plugin.py     # IDE plugin support
│   └── batch_processor.py # Batch processing
├── validation/           # Validation utilities
│   ├── __init__.py
│   ├── formal_verification.py # Formal verification
│   ├── dataset_benchmark.py # Dataset benchmarks
│   ├── api_validator.py  # API validation
│   ├── link_validator.py # Link validation
│   ├── docstring_validator.py # Docstring validation
│   └── audit_reporter.py # Audit reporting
├── benchmarks/           # Performance benchmarks
│   ├── __init__.py
│   └── performance_benchmarks.py
├── utils/                # Utility functions
│   └── __init__.py
└── cli/                  # Command-line interface
    ├── __init__.py
    ├── main.py           # CLI entry point
    ├── config.py         # CLI configuration
    └── commands/         # CLI commands
        ├── __init__.py
        ├── rate.py       # Rate command
        ├── train.py      # Train command
        ├── compare.py    # Compare command
        ├── evaluate.py   # Evaluate command
        ├── analytics.py  # Analytics command
        └── integration.py # Integration command

tests/
├── __init__.py
├── conftest.py           # Test configuration
├── unit/                 # Unit tests
├── test_*.py             # Test modules (property-based, integration, etc.)
└── README.md             # Test documentation

docs/
├── math_framework/       # Mathematical specification
│   └── mathematical_framework.pdf
├── api/                  # API documentation
│   └── api_reference.rst
├── udl/                  # UDL documentation
│   └── examples.rst
├── AUDIT_REPORT.md       # Documentation audit report
├── STRUCTURE_VALIDATION_REPORT.md # Structure validation
└── CONTRIBUTING.md       # Contribution guidelines

examples/
├── notebooks/            # Jupyter notebooks
├── udl_examples/         # Example UDL files
└── *.py                  # Example scripts
```

## Mathematical Foundation

All metrics are formally defined in [Mathematical Framework](docs/math_framework/mathematical_framework.pdf). Key definitions:

### Consistency Metric

```math
Consistency(U) = 1 - (|Contradictions| + |Cycles|) / (|Rules| + 1)
```

### Completeness Metric

```math
Completeness(U) = |Defined_Constructs| / |Required_Constructs|
```

### Overall Quality Score

```math
Q(U) = Σᵢ wᵢ · mᵢ(U)
```

where wᵢ are weights (Σwᵢ = 1) and mᵢ are individual metrics.

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run unit tests only
uv run pytest tests/unit/

# Run with coverage
uv run pytest --cov=udl_rating_framework --cov-report=html
```

### Code Quality

```bash
# Format code
uv run black udl_rating_framework/

# Lint
uv run flake8 udl_rating_framework/

# Type checking
uv run mypy udl_rating_framework/
```

## Requirements

- Python 3.10+
- uv package manager
- PyTorch 2.0+
- NetworkX 3.0+
- NumPy 1.24+
- SciPy 1.10+
- Hypothesis 6.0+

All dependencies are managed through [pyproject.toml](pyproject.toml) and installed via uv. The [uv.lock](uv.lock) file ensures reproducible builds across environments.

## Documentation

- [Mathematical Framework](docs/math_framework/mathematical_framework.pdf) - Complete mathematical specification
- [API Documentation](docs/api/api_reference.rst) - API reference
- [Tutorial Notebooks](examples/notebooks/) - Jupyter notebooks with examples
- [UDL Examples](docs/udl/examples.rst) - Example UDL files and documentation

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](/docs/CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Built on the Continuous Thought Machine (CTM) architecture for temporal sequence processing.

## Status

This project is under active development. Current status:

- [x] Project structure and foundation
- [x] Core metrics implementation (Consistency, Completeness, Expressiveness, Structural Coherence)
- [x] Advanced metrics (Semantic Similarity, Readability, Maintainability, Cross-Language Compatibility, Evolution Tracking)
- [x] CTM integration (CTM adapter)
- [x] Evaluation utilities (Comparison, Evaluation Suite)
- [x] Training pipeline (Active Learning, Ensemble Methods, Hyperparameter Optimization)
- [x] Analytics (Portfolio Analyzer, Trend Predictor, BI Exporter)
- [x] Visualization (Web, WebGL, Real-time Metrics)
- [x] Integration tools (CI/CD, Git Hooks, LSP Server)
- [x] Comprehensive testing (650+ tests, 59-66% coverage, 40 correctness properties)
- [ ] Documentation polish and validation (in progress)
