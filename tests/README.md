# Tests

This directory contains the test suite for the UDL Rating Framework and CTM integration.

## Running Tests

To execute all tests:

```bash
uv run pytest tests/
```

To run with coverage:

```bash
uv run pytest tests/ --cov=udl_rating_framework --cov-report=html
```

To run specific test categories:

```bash
# Property-based tests only
uv run pytest tests/ -k "property"

# Integration tests only
uv run pytest tests/ -k "integration"

# Metric tests only
uv run pytest tests/ -k "metric"
```

## Test Organization

The test suite uses a flat structure with descriptive naming conventions. As of December 2024, the test suite contains 56 test files with 193 test classes and 1535 test functions, achieving 100% docstring coverage.

### Core Tests
- `tests.py` - Golden tests for CTM model behavior
- `conftest.py` - Shared pytest fixtures

### Unit Tests (`unit/`)
- `test_project_setup.py` - Project setup validation
- `test_project_setup_minimal.py` - Minimal setup tests

### Property-Based Tests
- `test_metric_properties.py` - Metric boundedness and determinism (Properties 2, 3)
- `test_link_validation_properties.py` - Link validation (Property 2)
- `test_property_based_performance_modules.py` - Performance module properties
- `test_property_based_uncovered_paths.py` - Uncovered path properties

### Documentation Validation Tests
- `test_docstring_validation.py` - Docstring accuracy (Property 19)
- `test_mathematical_formula_validation.py` - Formula consistency (Property 7)
- `test_worked_examples_validation.py` - Worked example correctness (Property 8)
- `test_code_file_references.py` - Code file references (Property 9)
- `test_requirement_reference_validity.py` - Requirement references (Property 23)
- `test_api_endpoint_documentation.py` - API endpoint documentation (Property 24)
- `test_env_var_documentation.py` - Environment variable documentation (Property 25)
- `test_audit_reporter.py` - Audit report generation (Properties 21, 22)

### Metric Tests
- `test_consistency_metric.py` - Consistency metric
- `test_completeness_metric.py` - Completeness metric
- `test_expressiveness_metric.py` - Expressiveness metric
- `test_structural_coherence_metric.py` - Structural coherence metric
- `test_advanced_metrics.py` - Advanced metric tests
- `test_aggregation_confidence.py` - Aggregation and confidence tests

### Integration Tests
- `test_cli_integration.py` - CLI command integration
- `test_cli_option_accuracy.py` - CLI option validation (Property 20)
- `test_api_integration.py` - API integration tests
- `test_final_integration.py` - End-to-end integration
- `test_integration_features.py` - Feature integration tests
- `test_task26_integration.py` - Task 26 integration tests
- `test_deployment.py` - Deployment tests

### Example Validation Tests
- `test_example_validation.py` - Example file validation
- `test_notebook_validation.py` - Jupyter notebook validation (Property 11)
- `test_udl_example_processing.py` - UDL example processing (Property 10)

### Performance Tests
- `test_performance.py` - Performance benchmarks
- `test_performance_improvements.py` - Performance improvement tests
- `test_performance_modules_fixed.py` - Fixed performance module tests

### Other Tests
- `test_analytics.py` - Analytics module tests
- `test_caching_comprehensive.py` - Caching system tests
- `test_comparison_engine.py` - Comparison engine tests
- `test_ctm_adapter.py` - CTM adapter tests
- `test_data.py` - Data handling tests
- `test_enhanced_visualizations.py` - Visualization tests
- `test_error_injection_fault_tolerance.py` - Fault tolerance tests
- `test_evaluation_suite.py` - Evaluation suite tests
- `test_file_discovery.py` - File discovery tests
- `test_ml_enhancements.py` - ML enhancement tests
- `test_new_format_support.py` - New format support tests
- `test_rating_pipeline.py` - Rating pipeline tests
- `test_report_generator.py` - Report generator tests
- `test_training_pipeline.py` - Training pipeline tests
- `test_udl_representation.py` - UDL representation tests
- `test_visualization_edge_cases.py` - Visualization edge case tests

## Golden Tests

The golden tests verify that changes in the code have not resulted in differences in model behavior:

```bash
uv run pytest tests/tests.py::test_golden_parity
uv run pytest tests/tests.py::test_golden_qamnist
uv run pytest tests/tests.py::test_golden_rl
```

## Test Conventions

### Docstrings
All test functions should have docstrings that describe:
1. What behavior is being tested
2. Property references (if applicable): `**Property X: Name**`
3. Requirement references: `**Validates: Requirements X.Y**`

Example:
```python
def test_metric_boundedness(self):
    """
    **Feature: udl-rating-framework, Property 2: Metric Boundedness**
    **Validates: Requirements 1.4, 3.7**
    
    For any UDL and any quality metric m, the computed value must satisfy 0 ≤ m(UDL) ≤ 1.
    """
```

### Property-Based Testing
- Uses Hypothesis library
- Minimum 100 iterations per property test
- Tests should be deterministic and reproducible

### Coverage
Coverage varies based on which tests are run. The full test suite achieves approximately 59-66% coverage depending on test configuration.

To run coverage:
```bash
uv run pytest tests/ --cov=udl_rating_framework --cov-report=html
```

Coverage reports are generated in `htmlcov/`

See `tests/final_coverage_report.md` for detailed coverage analysis.

## Test Documentation Validation

The test suite maintains 100% docstring coverage. All test functions have docstrings that describe:
1. What behavior is being tested
2. Property references (if applicable)
3. Requirement references

To validate test documentation:
```bash
uv run python scripts/validate_test_documentation.py
```

This script checks:
- All test functions have docstrings
- All requirement references are valid
- Test organization matches documented strategy
