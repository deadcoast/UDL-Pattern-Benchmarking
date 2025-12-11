# UDL Rating Framework

A mathematically-grounded system for evaluating the quality of User Defined Languages (UDLs), built on the Continuous Thought Machine (CTM) architecture.

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

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/udl-rating-framework.git
cd udl-rating-framework

# Install dependencies
pip install -r requirements-udl.txt

# Install the package
pip install -e .
```

### Using pip (when published)

```bash
pip install udl-rating-framework
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

All metrics are formally defined in `docs/mathematical_framework.pdf`. Key definitions:

### Consistency Metric

```
Consistency(U) = 1 - (|Contradictions| + |Cycles|) / (|Rules| + 1)
```

### Completeness Metric

```
Completeness(U) = |Defined_Constructs| / |Required_Constructs|
```

### Overall Quality Score

```
Q(U) = Σᵢ wᵢ · mᵢ(U)
```

where wᵢ are weights (Σwᵢ = 1) and mᵢ are individual metrics.

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run with coverage
pytest --cov=udl_rating_framework --cov-report=html
```

### Code Quality

```bash
# Format code
black udl_rating_framework/

# Lint
flake8 udl_rating_framework/

# Type checking
mypy udl_rating_framework/
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NetworkX 3.0+
- NumPy 1.24+
- SciPy 1.10+
- Hypothesis 6.0+

See `requirements-udl.txt` for complete list.

## Documentation

- [Mathematical Framework](docs/mathematical_framework.pdf) - Complete mathematical specification
- [API Documentation](https://udl-rating-framework.readthedocs.io/) - API reference
- [Tutorial Notebooks](examples/) - Jupyter notebooks with examples

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{udl_rating_framework,
  title = {UDL Rating Framework: Mathematically-Grounded Language Quality Assessment},
  author = {UDL Rating Framework Team},
  year = {2024},
  url = {https://github.com/yourusername/udl-rating-framework}
}
```

## Acknowledgments

Built on the Continuous Thought Machine (CTM) architecture for temporal sequence processing.

## Status

This project is under active development. Current status:

- [x] Project structure and foundation
- [ ] Core metrics implementation
- [ ] CTM integration
- [ ] Evaluation utilities
- [ ] Documentation and examples

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.
