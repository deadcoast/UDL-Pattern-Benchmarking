# Design Document: Documentation Validation and Project Polish

## Overview

This design document specifies a comprehensive documentation validation system for the UDL Rating Framework. The system performs automated and semi-automated audits to ensure documentation accurately reflects source code, all links resolve correctly, examples work as documented, and the project structure is coherent.

The validation system operates in three modes:

1. **Automated Validation**: Scripts that programmatically verify documentation claims against source code
2. **Semi-Automated Validation**: Tools that assist manual review by highlighting discrepancies
3. **Manual Audit**: Structured checklists for human verification of subjective quality

The core principle is establishing a validated baseline: every documentation claim should be traceable to working code, and every public API should have corresponding documentation.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Documentation Validation System                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐   │
│  │   Scanner    │      │  Validator   │      │   Reporter   │   │
│  │   Module     │─────▶│    Engine    │─────▶│    Module    │   │
│  └──────────────┘      └──────────────┘      └──────────────┘   │
│         │                      │                      │         │
│         ▼                      ▼                      ▼         │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐   │
│  │  File        │      │  Validation  │      │  Audit       │   │
│  │  Discovery   │      │  Rules       │      │  Report      │   │
│  └──────────────┘      └──────────────┘      └──────────────┘   │
│         │                      │                                │
│         │              ┌───────┴───────┐                        │
│         │              │               │                        │
│         ▼              ▼               ▼                        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │
│  │  Markdown    │ │  Link        │ │  Code        │             │
│  │  Parser      │ │  Validator   │ │  Analyzer    │             │
│  └──────────────┘ └──────────────┘ └──────────────┘             │
│         │              │               │                        │
│         ▼              ▼               ▼                        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │
│  │  RST         │ │  API         │ │  Example     │             │
│  │  Parser      │ │  Validator   │ │  Executor    │             │
│  └──────────────┘ └──────────────┘ └──────────────┘             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Validation Categories

1. **Content Validation**: Verify documentation content matches implementation
2. **Link Validation**: Verify all references resolve correctly
3. **Structure Validation**: Verify project organization is consistent
4. **Example Validation**: Verify all examples execute correctly
5. **Configuration Validation**: Verify all config files are consistent

## Components and Interfaces

### 1. Documentation Scanner

**Purpose**: Discover and parse all documentation files in the project

**Interface**:
```python
class DocumentationScanner:
    """
    Scans project for documentation files and extracts structured content.
    
    Supported formats:
    - Markdown (.md)
    - reStructuredText (.rst)
    - LaTeX (.tex)
    - Python docstrings
    - YAML/JSON configs
    """
    
    def __init__(self, project_root: Path):
        """Initialize scanner with project root directory."""
        pass
    
    def scan_markdown_files(self) -> List[MarkdownDocument]:
        """Find and parse all Markdown files."""
        pass
    
    def scan_rst_files(self) -> List[RSTDocument]:
        """Find and parse all RST files."""
        pass
    
    def scan_docstrings(self) -> Dict[str, Docstring]:
        """Extract docstrings from all Python modules."""
        pass
    
    def extract_code_blocks(self, doc: Document) -> List[CodeBlock]:
        """Extract code blocks from documentation."""
        pass
    
    def extract_links(self, doc: Document) -> List[Link]:
        """Extract all links from documentation."""
        pass
```

### 2. Link Validator

**Purpose**: Verify all documentation links resolve correctly

**Interface**:
```python
class LinkValidator:
    """
    Validates internal and external links in documentation.
    
    Link types:
    - File references: [text](path/to/file.md)
    - Anchor links: [text](#section-heading)
    - External URLs: [text](https://example.com)
    - Code references: `module.Class.method`
    """
    
    def __init__(self, project_root: Path):
        """Initialize validator with project root."""
        pass
    
    def validate_file_link(self, source: Path, target: str) -> ValidationResult:
        """Verify file reference resolves to existing file."""
        pass
    
    def validate_anchor_link(self, doc: Document, anchor: str) -> ValidationResult:
        """Verify anchor points to valid heading."""
        pass
    
    def validate_code_reference(self, reference: str) -> ValidationResult:
        """Verify code reference points to existing symbol."""
        pass
    
    def find_broken_links(self) -> List[BrokenLink]:
        """Scan all docs and return list of broken links."""
        pass
```

### 3. API Validator

**Purpose**: Verify API documentation matches source code

**Interface**:
```python
class APIValidator:
    """
    Validates API documentation against actual code.
    
    Checks:
    - Class existence and inheritance
    - Method signatures (parameters, types, defaults)
    - Return types
    - Docstring accuracy
    """
    
    def __init__(self, package_path: Path):
        """Initialize with package to validate."""
        pass
    
    def extract_public_api(self) -> Dict[str, APIElement]:
        """Extract all public classes, functions, methods."""
        pass
    
    def compare_signature(self, documented: Signature, actual: Signature) -> List[Discrepancy]:
        """Compare documented signature to actual."""
        pass
    
    def find_undocumented_apis(self) -> List[APIElement]:
        """Find public APIs lacking documentation."""
        pass
    
    def find_orphaned_docs(self) -> List[DocumentedAPI]:
        """Find documented APIs that no longer exist."""
        pass
```

### 4. Example Executor

**Purpose**: Execute code examples and verify they work

**Interface**:
```python
class ExampleExecutor:
    """
    Executes code examples from documentation and verifies results.
    
    Supports:
    - Python code blocks in Markdown
    - Jupyter notebook cells
    - Shell commands
    - UDL file processing
    """
    
    def __init__(self, project_root: Path):
        """Initialize executor with project context."""
        pass
    
    def execute_python_block(self, code: str) -> ExecutionResult:
        """Execute Python code block and capture output."""
        pass
    
    def execute_shell_command(self, command: str) -> ExecutionResult:
        """Execute shell command and capture output."""
        pass
    
    def validate_notebook(self, notebook_path: Path) -> NotebookResult:
        """Execute all cells in Jupyter notebook."""
        pass
    
    def validate_udl_example(self, udl_path: Path) -> UDLResult:
        """Process UDL file and verify expected values."""
        pass
```

### 5. Structure Validator

**Purpose**: Verify project structure consistency

**Interface**:
```python
class StructureValidator:
    """
    Validates project structure and organization.
    
    Checks:
    - __init__.py exports match public API
    - Import graph is acyclic and complete
    - Test files correspond to source modules
    - No orphaned files
    """
    
    def __init__(self, project_root: Path):
        """Initialize with project root."""
        pass
    
    def validate_init_exports(self, package: Path) -> List[ExportIssue]:
        """Verify __init__.py properly exports public API."""
        pass
    
    def find_orphaned_files(self) -> List[Path]:
        """Find files not imported or referenced anywhere."""
        pass
    
    def validate_import_graph(self) -> ImportGraphResult:
        """Analyze import graph for issues."""
        pass
    
    def validate_test_coverage(self) -> TestCoverageResult:
        """Verify test files exist for source modules."""
        pass
```

### 6. Configuration Validator

**Purpose**: Verify configuration file consistency

**Interface**:
```python
class ConfigValidator:
    """
    Validates configuration files for consistency.
    
    Files checked:
    - pyproject.toml
    - setup.py (if exists)
    - pytest.ini
    - requirements*.txt
    """
    
    def __init__(self, project_root: Path):
        """Initialize with project root."""
        pass
    
    def validate_dependencies(self) -> DependencyResult:
        """Check declared vs actual dependencies."""
        pass
    
    def validate_entry_points(self) -> EntryPointResult:
        """Verify CLI entry points work."""
        pass
    
    def validate_version_consistency(self) -> VersionResult:
        """Check version numbers are consistent."""
        pass
```

### 7. Audit Reporter

**Purpose**: Generate comprehensive audit reports

**Interface**:
```python
class AuditReporter:
    """
    Generates audit reports from validation results.
    
    Output formats:
    - Markdown summary
    - JSON detailed report
    - HTML dashboard
    """
    
    def __init__(self):
        """Initialize reporter."""
        pass
    
    def add_finding(self, finding: Finding):
        """Add a validation finding."""
        pass
    
    def categorize_findings(self) -> Dict[Severity, List[Finding]]:
        """Group findings by severity."""
        pass
    
    def generate_markdown_report(self) -> str:
        """Generate Markdown summary report."""
        pass
    
    def generate_json_report(self) -> Dict:
        """Generate detailed JSON report."""
        pass
    
    def generate_fix_suggestions(self) -> List[FixSuggestion]:
        """Generate suggested fixes for findings."""
        pass
```

## Data Models

### Finding

```python
@dataclass
class Finding:
    """Represents a validation finding."""
    id: str
    category: str  # link, api, example, structure, config
    severity: Severity  # critical, major, minor, info
    source_file: Path
    source_line: Optional[int]
    description: str
    expected: Optional[str]
    actual: Optional[str]
    suggestion: Optional[str]
    requirement_ref: str  # e.g., "1.2"
```

### ValidationResult

```python
@dataclass
class ValidationResult:
    """Result of a single validation check."""
    passed: bool
    message: str
    details: Optional[Dict[str, Any]]
    source_location: Optional[SourceLocation]
```

### AuditReport

```python
@dataclass
class AuditReport:
    """Complete audit report."""
    timestamp: datetime
    project_version: str
    total_files_scanned: int
    findings: List[Finding]
    summary: Dict[str, int]  # counts by category and severity
    coverage_metrics: Dict[str, float]
    recommendations: List[str]
```



## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Code Example Execution

*For any* code example extracted from documentation (README, docstrings, tutorials), executing the code should complete without raising exceptions.

**Validates: Requirements 1.2, 8.3**

**Rationale**: Documentation code examples must work. We can extract all code blocks and verify they execute successfully.

### Property 2: Link Target Resolution

*For any* internal link in documentation (file reference, anchor link, or code reference), the target must exist and be accessible.

**Validates: Requirements 2.2, 2.3, 2.4**

**Rationale**: Broken links degrade documentation quality. We can extract all links and verify each target exists.

### Property 3: CLI Command Validity

*For any* CLI command documented in README or help text, executing the command with `--help` should succeed and the command should be registered.

**Validates: Requirements 1.4, 9.1**

**Rationale**: Documented CLI commands must exist. We can extract command examples and verify they're valid.

### Property 4: Documented API Existence

*For any* class, method, or function documented in API documentation, the corresponding symbol must exist in the source code.

**Validates: Requirements 3.1, 3.5**

**Rationale**: API documentation must reference real code. We can parse API docs and verify each symbol exists.

### Property 5: Method Signature Accuracy

*For any* documented method signature, the parameter names, types, and defaults must match the actual implementation.

**Validates: Requirements 3.2, 3.3**

**Rationale**: Signature mismatches cause user confusion. We can compare documented signatures to actual signatures.

### Property 6: Public API Documentation Coverage

*For any* public class, method, or function in the package, there should be corresponding documentation (docstring or API doc).

**Validates: Requirements 3.4, 8.4**

**Rationale**: All public APIs should be documented. We can enumerate public APIs and check for documentation.

### Property 7: Mathematical Formula Consistency

*For any* mathematical formula documented in mathematical_framework.tex or docstrings, the formula should match the implementation in code.

**Validates: Requirements 4.1, 8.5**

**Rationale**: Mathematical claims must be accurate. We can extract formulas and compare to implementations.

### Property 8: Worked Example Correctness

*For any* worked example in the mathematical framework document, executing the computation should produce the documented result (within ε = 1e-6).

**Validates: Requirements 4.2, 5.2**

**Rationale**: Examples must be verifiable. We can execute worked examples and compare to documented values.

### Property 9: Code File Reference Validity

*For any* code file referenced in documentation, the file must exist at the specified path.

**Validates: Requirements 4.4**

**Rationale**: File references must be valid. We can extract file paths and verify existence.

### Property 10: UDL Example Processing

*For any* UDL example file in the examples directory, the rating system should successfully process the file without errors.

**Validates: Requirements 5.1**

**Rationale**: Example files must work with the system. We can process each UDL file and verify success.

### Property 11: Notebook Cell Execution

*For any* cell in a Jupyter notebook in the examples directory, executing the cell should complete without errors.

**Validates: Requirements 5.3**

**Rationale**: Tutorial notebooks must work. We can execute all cells and verify no exceptions.

### Property 12: Example Script Execution

*For any* Python script in the examples directory, running the script should complete without errors.

**Validates: Requirements 5.4**

**Rationale**: Example scripts must be runnable. We can execute each script and verify success.

### Property 13: Init Export Completeness

*For any* `__init__.py` file in the package, all symbols listed in `__all__` must be importable from that module.

**Validates: Requirements 6.1**

**Rationale**: Package exports must work. We can check each `__all__` entry is importable.

### Property 14: Import Resolution

*For any* import statement in the codebase, the imported module or symbol must exist and be resolvable.

**Validates: Requirements 6.5**

**Rationale**: All imports must work. We can analyze imports and verify resolution.

### Property 15: Test Module Correspondence

*For any* source module in udl_rating_framework, there should be a corresponding test file in the tests directory.

**Validates: Requirements 6.4**

**Rationale**: All modules should have tests. We can enumerate source modules and check for test files.

### Property 16: Dependency Usage

*For any* dependency listed in pyproject.toml, the package should be imported somewhere in the codebase.

**Validates: Requirements 7.1**

**Rationale**: Listed dependencies should be used. We can check each dependency is imported.

### Property 17: Import Coverage

*For any* third-party package imported in the codebase, it should be listed in pyproject.toml dependencies.

**Validates: Requirements 7.2**

**Rationale**: Used packages must be declared. We can check each import has a corresponding dependency.

### Property 18: Entry Point Validity

*For any* entry point defined in pyproject.toml, invoking it should not raise an import error.

**Validates: Requirements 7.3**

**Rationale**: Entry points must work. We can test each entry point is callable.

### Property 19: Docstring Signature Accuracy

*For any* function or method with a docstring, the documented parameters must match the actual function signature.

**Validates: Requirements 8.1, 8.2**

**Rationale**: Docstrings must be accurate. We can compare docstring params to actual params.

### Property 20: CLI Option Accuracy

*For any* CLI option documented in help text, the option should exist and behave as described.

**Validates: Requirements 9.2**

**Rationale**: CLI help must be accurate. We can verify each option exists.

### Property 21: Finding Completeness

*For any* validation finding generated by the audit system, it must include severity category, file location, and line number.

**Validates: Requirements 10.2, 10.3**

**Rationale**: Findings must be actionable. We can verify each finding has required metadata.

### Property 22: Fix Suggestion Generation

*For any* common issue type (broken link, missing docstring, signature mismatch), the system should generate a fix suggestion.

**Validates: Requirements 10.4**

**Rationale**: Common issues should have suggestions. We can verify suggestions exist for known issue types.

### Property 23: Test Requirement Reference Validity

*For any* requirement reference in a test file (e.g., "Validates: Requirements 1.2"), the referenced requirement must exist in the requirements document.

**Validates: Requirements 11.2, 11.4**

**Rationale**: Test references must be valid. We can extract references and verify they exist.

### Property 24: API Endpoint Documentation

*For any* API endpoint defined in the FastAPI application, there should be corresponding documentation.

**Validates: Requirements 12.3**

**Rationale**: API endpoints must be documented. We can enumerate endpoints and check for docs.

### Property 25: Environment Variable Documentation

*For any* environment variable read by the application, it should be documented in deployment documentation.

**Validates: Requirements 12.4**

**Rationale**: Env vars must be documented. We can find env var reads and check for docs.

## Error Handling

### Error Categories

1. **Scan Errors**
   - File not found
   - Permission denied
   - Encoding errors
   - Parse failures

2. **Validation Errors**
   - Link resolution failure
   - Symbol not found
   - Signature mismatch
   - Execution failure

3. **Report Errors**
   - Output directory not writable
   - Template rendering failure

### Error Handling Strategy

```python
class ValidationError(Exception):
    """Base exception for validation errors."""
    pass

class ScanError(ValidationError):
    """Error during file scanning."""
    pass

class LinkError(ValidationError):
    """Error validating a link."""
    pass

class ExecutionError(ValidationError):
    """Error executing code example."""
    pass

def safe_validate(validator: Callable, item: Any) -> ValidationResult:
    """
    Safely run validation with error handling.
    
    Returns ValidationResult with error details if validation fails.
    """
    try:
        return validator(item)
    except Exception as e:
        return ValidationResult(
            passed=False,
            message=f"Validation error: {str(e)}",
            details={"exception": type(e).__name__, "traceback": traceback.format_exc()}
        )
```

### Graceful Degradation

When errors occur, the system should:

1. **Log the error** with full context
2. **Continue validation** of other items
3. **Include error in report** as a finding
4. **Provide partial results** rather than failing completely

## Testing Strategy

### Unit Testing

**Scope**: Individual validation functions

**Key Test Cases**:

1. **Link Validator Tests**
   - Test file link resolution
   - Test anchor link resolution
   - Test code reference resolution
   - Test broken link detection

2. **API Validator Tests**
   - Test class existence checking
   - Test signature comparison
   - Test undocumented API detection

3. **Example Executor Tests**
   - Test Python code execution
   - Test shell command execution
   - Test notebook validation

### Property-Based Testing

**Library**: Hypothesis

**Configuration**: Minimum 100 iterations per property test

**Key Properties**:

1. **Link Validation Completeness**
   - Generate documents with various link types
   - Verify all links are found and validated

2. **API Coverage**
   - Generate mock packages with public APIs
   - Verify all public APIs are detected

3. **Finding Completeness**
   - Generate validation results
   - Verify all findings have required fields

### Integration Testing

**Scope**: End-to-end validation workflows

**Key Scenarios**:

1. **Full Project Audit**
   - Run complete audit on test project
   - Verify all categories are checked
   - Verify report is generated

2. **Incremental Validation**
   - Modify documentation
   - Re-run validation
   - Verify changes are detected

### Test Organization

The test suite uses a flat structure with descriptive naming conventions:

```
tests/
├── unit/                                    # Unit tests for project setup
│   ├── test_project_setup.py
│   └── test_project_setup_minimal.py
├── test_link_validation_properties.py       # Link validation (Property 2)
├── test_docstring_validation.py             # Docstring validation (Property 19)
├── test_api_integration.py                  # API validation (Properties 4, 5, 6)
├── test_example_validation.py               # Example validation (Properties 10, 11, 12)
├── test_notebook_validation.py              # Notebook validation (Property 11)
├── test_udl_example_processing.py           # UDL example processing (Property 10)
├── test_file_discovery.py                   # File discovery (Property 6, 7)
├── test_metric_properties.py                # Metric properties (Properties 2, 3)
├── test_mathematical_formula_validation.py  # Formula validation (Property 7)
├── test_worked_examples_validation.py       # Worked examples (Property 8)
├── test_code_file_references.py             # Code file references (Property 9)
├── test_cli_option_accuracy.py              # CLI validation (Property 20)
├── test_requirement_reference_validity.py   # Requirement refs (Property 23)
├── test_final_integration.py                # Integration tests
├── test_cli_integration.py                  # CLI integration tests
└── ... (additional test files)
```

**Naming Convention:**
- `test_*_properties.py` - Property-based tests
- `test_*_validation.py` - Validation tests
- `test_*_integration.py` - Integration tests
- `test_*_metric.py` - Metric-specific tests

## Implementation Notes

### Technology Stack

- **Language**: Python 3.10+
- **Markdown Parsing**: markdown-it-py or mistune
- **RST Parsing**: docutils
- **AST Analysis**: ast module, inspect module
- **Code Execution**: subprocess, exec()
- **Notebook Execution**: nbconvert, papermill
- **Report Generation**: Jinja2 templates

### Validation Approach

The validation will be primarily manual with scripted assistance:

1. **Scripted Discovery**: Automated tools find potential issues
2. **Manual Review**: Human verifies and categorizes findings
3. **Scripted Fixes**: Automated tools apply common fixes
4. **Manual Verification**: Human confirms fixes are correct

This hybrid approach balances thoroughness with practicality.

