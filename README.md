# UDL Pattern Benchmarking

A mathematically-grounded system for evaluating the quality of User Defined Languages (UDLs), built on the Continuous Thought Machine (CTM) architecture.

<!-- TOC -->
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [From Source](#from-source)
  - [Development Installation](#development-installation)
  - [Using pip (when published)](#using-pip-when-published)
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
- **Four Quality Dimensions**:
  - **Consistency**: Internal coherence using graph-theoretic analysis
  - **Completeness**: Construct coverage using set theory
  - **Expressiveness**: Language power via Chomsky hierarchy classification
  - **Structural Coherence**: Organizational quality using information theory
- **Confidence Scores**: Entropy-based certainty measures
- **CTM Integration**: Neural approximation for fast inference
- **Comprehensive Testing**: Property-based and unit testing

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
from udl_rating_framework.core.metrics import ConsistencyMetric

# Load a UDL
with open("my_language.udl", "r") as f:
    udl = UDLRepresentation(f.read(), "my_language.udl")

# Compute consistency
metric = ConsistencyMetric()
score = metric.compute(udl)
print(f"Consistency: {score:.3f}")
```

### Command-Line Interface

```bash
# Rate a directory of UDL files
udl-rate --directory ./my_udls --output report.json

# Train a CTM model
udl-train --dataset ./training_data --epochs 50 --output model.pt

# Compare multiple UDLs
udl-compare --files udl_v1.dsl udl_v2.dsl --output comparison.html

# Evaluate model performance
udl-evaluate --model model.pt --test-set ./test_data
```

## Project Structure

```
udl_rating_framework/
├── core/                  # Core components
│   ├── representation.py  # UDL representation
│   ├── metrics/          # Quality metrics
│   ├── aggregation.py    # Metric aggregation
│   └── confidence.py     # Confidence calculation
├── models/               # CTM model components
├── io/                   # Input/output handling
├── evaluation/           # Evaluation utilities
├── utils/                # Utility functions
└── cli/                  # Command-line interface

tests/
├── unit/                 # Unit tests
├── property/             # Property-based tests
├── integration/          # Integration tests
└── performance/          # Performance benchmarks

docs/
└── mathematical_framework.tex  # Mathematical specification
```

## Mathematical Foundation

All metrics are formally defined in [Mathematical Framework](docs/mathematical_framework.pdf). Key definitions:

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

All dependencies are managed through [UV via pyproject.toml](pyproject.toml) [#LINKTODO]  and installed via uv. The [UV lock file](uv.lock) [#LINKTODO] ensures reproducible builds across environments.

## Documentation

- [Mathematical Framework](https://github.com/deadcoast/UDL-Pattern-Benchmarking/blob/main/docs/mathematical_framework.pdf) - [#LINKTODO] Complete mathematical specification
- [API Documentation](docs/api_reference.rst) - API reference
- [Tutorial Notebooks](docs/examples.rst) - Jupyter notebooks with examples

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](/docs/CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Built on the Continuous Thought Machine (CTM) architecture for temporal sequence processing.

## Status

This project is under active development. Current status:

- [x] Project structure and foundation
- [x] Core metrics implementation
- [ ] CTM integration
- [ ] Evaluation utilities
- [ ] Documentation and examples
