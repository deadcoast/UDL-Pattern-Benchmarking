# Documentation Style Guide

This guide defines the documentation conventions for the UDL Rating Framework project. Following these conventions ensures consistency across all documentation and enables automated validation tools to work correctly.

## Table of Contents

- [Docstring Conventions](#docstring-conventions)
- [Link Conventions](#link-conventions)
- [Code Examples](#code-examples)
- [Mathematical Formulas](#mathematical-formulas)
- [Property Test Documentation](#property-test-documentation)

---

## Docstring Conventions

The project uses **Google-style docstrings** for all Python code. This format is readable, well-supported by documentation generators, and compatible with our automated validation tools.

### Basic Structure

```python
def function_name(param1: str, param2: int = 10) -> bool:
    """Short one-line summary of the function.
    
    Longer description if needed. This can span multiple lines and should
    explain what the function does, any important behavior, and context
    that helps users understand when to use it.
    
    Args:
        param1: Description of the first parameter. No type annotation
            needed here since it's in the signature.
        param2: Description of the second parameter. Mention the default
            value behavior if relevant.
    
    Returns:
        Description of what is returned. For complex return types,
        describe the structure.
    
    Raises:
        ValueError: When param1 is empty.
        TypeError: When param2 is not an integer.
    
    Example:
        >>> result = function_name("test", 5)
        >>> print(result)
        True
    """
```

### Required Sections

| Section | Required | Description |
|---------|----------|-------------|
| Summary | Yes | One-line description (first line) |
| Description | No | Extended explanation if needed |
| Args | Yes* | Document all parameters (*if function has parameters) |
| Returns | Yes* | Document return value (*if function returns something) |
| Raises | No | Document exceptions that may be raised |
| Example | No | Executable doctest examples |

### Parameter Documentation Format

Each parameter should be documented with:
- **Name**: Must match the actual parameter name exactly
- **Description**: Clear explanation of what the parameter does

```python
def process_udl(
    file_path: Path,
    metrics: List[str] = None,
    verbose: bool = False
) -> Dict[str, float]:
    """Process a UDL file and compute quality metrics.
    
    Args:
        file_path: Path to the UDL file to process. Must be a valid
            file path that exists on disk.
        metrics: List of metric names to compute. If None, computes
            all available metrics.
        verbose: If True, prints progress information during processing.
    
    Returns:
        Dictionary mapping metric names to their computed scores.
        All scores are normalized to the range [0.0, 1.0].
    """
```

### Class Documentation

Classes should have a docstring immediately after the class definition:

```python
class MetricCalculator:
    """Calculates quality metrics for UDL files.
    
    This class provides methods for computing various quality metrics
    including consistency, completeness, and expressiveness scores.
    
    Attributes:
        config: Configuration dictionary for metric computation.
        cache: Optional cache for storing intermediate results.
    
    Example:
        >>> calc = MetricCalculator(config={'normalize': True})
        >>> score = calc.compute('consistency', udl_content)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the metric calculator.
        
        Args:
            config: Configuration options for metric computation.
                Supported keys: 'normalize', 'weights', 'threshold'.
        """
```

### Method Documentation

Methods follow the same format as functions. For `__init__`, document parameters but not the return value:

```python
def __init__(self, package_name: str = "udl_rating_framework"):
    """Initialize the validator.
    
    Args:
        package_name: Name of the Python package to validate.
    """
```

### Documenting Mathematical Formulas

When a function implements a mathematical formula, include it in the docstring:

```python
def compute_consistency(self, udl_content: str) -> float:
    """Compute the consistency metric for UDL content.
    
    The consistency score measures how well the UDL follows
    established patterns and conventions.
    
    Formula:
        Consistency(U) = Σ(w_i × c_i) / Σ(w_i)
        
        where:
        - w_i = weight for criterion i
        - c_i = score for criterion i (0 or 1)
    
    Args:
        udl_content: The UDL content to analyze.
    
    Returns:
        Consistency score in range [0.0, 1.0].
    """
```

### Common Mistakes to Avoid

❌ **Wrong**: Parameter name mismatch
```python
def process(file_path: str) -> None:
    """Process a file.
    
    Args:
        path: The file path.  # Wrong! Should be 'file_path'
    """
```

✅ **Correct**: Parameter names match
```python
def process(file_path: str) -> None:
    """Process a file.
    
    Args:
        file_path: The file path.
    """
```

❌ **Wrong**: Missing parameter documentation
```python
def calculate(x: int, y: int, z: int) -> int:
    """Calculate a value.
    
    Args:
        x: First value.
        y: Second value.
        # Missing z!
    """
```

✅ **Correct**: All parameters documented
```python
def calculate(x: int, y: int, z: int) -> int:
    """Calculate a value.
    
    Args:
        x: First value.
        y: Second value.
        z: Third value.
    """
```

---

## Link Conventions

Documentation links must follow specific conventions to ensure they resolve correctly and can be validated automatically.

### File Reference Links

Use relative paths from the current file's location:

```markdown
<!-- From docs/api/overview.md linking to docs/CONTRIBUTING.md -->
See the [Contributing Guide](../CONTRIBUTING.md) for details.

<!-- From README.md linking to docs/api/overview.md -->
See the [API Overview](docs/api/overview.md) for details.
```

### Anchor Links

Anchor links reference headings within the same document:

```markdown
## Installation

See [Configuration](#configuration) below for setup options.

## Configuration

Configuration options are...
```

**Anchor Generation Rules:**
- Convert heading text to lowercase
- Replace spaces with hyphens
- Remove special characters (except hyphens and underscores)
- Remove leading/trailing hyphens

| Heading | Anchor |
|---------|--------|
| `## Getting Started` | `#getting-started` |
| `## API Reference (v2)` | `#api-reference-v2` |
| `## What's New?` | `#whats-new` |

### Cross-Document Anchor Links

Link to a specific section in another document:

```markdown
See [Installation Steps](docs/SETUP_COMPLETE.md#installation) for details.
```

### Code Reference Links

When referencing code files or modules:

```markdown
<!-- Link to a Python file -->
See [`udl_rating_framework/core/metrics/consistency.py`](udl_rating_framework/core/metrics/consistency.py)

<!-- Link to a specific class (use anchor if documented) -->
See the [MetricCalculator](docs/api/metrics.md#metriccalculator) class.
```

### External Links

External URLs should use the full URL:

```markdown
See the [Python documentation](https://docs.python.org/3/) for more information.
```

### Link Validation

All internal links are validated automatically. To ensure your links pass validation:

1. **File links**: Target file must exist at the specified path
2. **Anchor links**: Target heading must exist in the document
3. **Relative paths**: Must be relative to the source file's directory

### Common Link Mistakes to Avoid

❌ **Wrong**: Broken relative path
```markdown
<!-- From docs/api/overview.md -->
See [README](README.md)  # Wrong! Should be ../../README.md
```

✅ **Correct**: Proper relative path
```markdown
<!-- From docs/api/overview.md -->
See [README](../../README.md)
```

❌ **Wrong**: Invalid anchor
```markdown
See [Setup](#setup-guide)  # Wrong if heading is "## Setup Guide"
```

✅ **Correct**: Proper anchor format
```markdown
See [Setup](#setup-guide)  # Correct for heading "## Setup Guide"
```

---

## Code Examples

### Inline Code

Use single backticks for inline code:

```markdown
Use the `compute_metrics()` function to calculate scores.
The `--verbose` flag enables detailed output.
```

### Code Blocks

Use triple backticks with language specification:

````markdown
```python
from udl_rating_framework import rate_udl

result = rate_udl("path/to/file.udl")
print(f"Score: {result.score}")
```
````

### Executable Examples

Code examples in documentation should be executable. Test them before committing:

```python
# This example should work when copy-pasted
from udl_rating_framework.core.metrics import ConsistencyMetric

metric = ConsistencyMetric()
score = metric.compute("sample UDL content")
assert 0.0 <= score <= 1.0
```

### Shell Commands

Use `bash` or `shell` for command-line examples:

````markdown
```bash
# Install the package
uv add udl-rating-framework

# Run the CLI
uv run udl-rating rate examples/sample.udl
```
````

---

## Mathematical Formulas

### LaTeX in Documentation

Use LaTeX syntax for mathematical formulas in Markdown:

```markdown
The consistency metric is computed as:

$$
\text{Consistency}(U) = \frac{\sum_{i=1}^{n} w_i \cdot c_i}{\sum_{i=1}^{n} w_i}
$$

where $w_i$ is the weight and $c_i$ is the criterion score.
```

### Formulas in Docstrings

For docstrings, use plain text or simple ASCII notation:

```python
"""Compute weighted average.

Formula:
    result = Σ(w_i × v_i) / Σ(w_i)
    
    where:
    - w_i = weight for item i
    - v_i = value for item i
"""
```

---

## Property Test Documentation

Property-based tests must include specific annotations linking them to requirements.

### Required Annotations

```python
def test_consistency_metric_bounds():
    """Test that consistency scores are always in valid range.
    
    **Feature: udl-rating-framework, Property 1: Score Bounds**
    **Validates: Requirements 2.1, 2.3**
    """
```

### Annotation Format

- **Feature line**: `**Feature: {feature_name}, Property {number}: {property_name}**`
- **Validates line**: `**Validates: Requirements {requirement_refs}**`

### Example Property Test

```python
from hypothesis import given, strategies as st

@given(st.text(min_size=1))
def test_round_trip_serialization(content: str):
    """Test that serialization round-trips correctly.
    
    For any valid content, serializing and deserializing should
    produce the original value.
    
    **Feature: documentation-validation, Property 1: Round Trip**
    **Validates: Requirements 3.1, 3.2**
    """
    serialized = serialize(content)
    deserialized = deserialize(serialized)
    assert deserialized == content
```

---

## Quick Reference

### Docstring Template

```python
def function_name(param1: Type1, param2: Type2 = default) -> ReturnType:
    """One-line summary.
    
    Extended description if needed.
    
    Args:
        param1: Description of param1.
        param2: Description of param2.
    
    Returns:
        Description of return value.
    
    Raises:
        ExceptionType: When this happens.
    
    Example:
        >>> function_name("value", 42)
        expected_result
    """
```

### Link Syntax Quick Reference

| Type | Syntax | Example |
|------|--------|---------|
| File link | `[text](path/to/file.md)` | `[Guide](docs/guide.md)` |
| Anchor link | `[text](#heading)` | `[Setup](#setup)` |
| Cross-doc anchor | `[text](file.md#heading)` | `[Install](docs/setup.md#install)` |
| External | `[text](https://...)` | `[Python](https://python.org)` |
| Code file | `[`path`](path)` | `[\`main.py\`](src/main.py)` |

---

## Validation

Documentation is automatically validated by CI/CD. Run validation locally:

```bash
# Validate docstrings
uv run python -m udl_rating_framework.validation.docstring_validator

# Validate links
uv run python scripts/check_links.py

# Run all documentation tests
uv run pytest tests/test_docstring_validation.py tests/test_link_validation_properties.py
```
