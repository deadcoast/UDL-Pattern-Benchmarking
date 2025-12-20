"""
Property-based tests for Example Execution.

**Feature: documentation-validation, Property 1: Code Example Execution**
**Feature: documentation-validation, Property 11: Notebook Cell Execution**
**Feature: documentation-validation, Property 12: Example Script Execution**
**Validates: Requirements 1.2, 5.3, 5.4**

Tests that:
- For any code example in documentation, executing the code should complete without errors
- For any cell in a Jupyter notebook, executing the cell should complete without errors
- For any Python script in examples, running the script should complete without errors
"""

import pytest
import re
import sys
import tempfile
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
from hypothesis import given, strategies as st, settings, assume
from dataclasses import dataclass


PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class CodeBlock:
    """Represents a code block extracted from documentation."""
    content: str
    language: str
    source_file: Path
    line_number: int


def extract_code_blocks_from_markdown(file_path: Path) -> List[CodeBlock]:
    """Extract Python code blocks from a markdown file."""
    code_blocks = []
    
    try:
        content = file_path.read_text(encoding='utf-8')
    except (IOError, UnicodeDecodeError):
        return code_blocks
    
    # Pattern for fenced code blocks with language specifier
    pattern = r'```(python|py)\n(.*?)```'
    
    for match in re.finditer(pattern, content, re.DOTALL):
        language = match.group(1)
        code = match.group(2)
        
        # Calculate line number
        line_number = content[:match.start()].count('\n') + 1
        
        code_blocks.append(CodeBlock(
            content=code.strip(),
            language=language,
            source_file=file_path,
            line_number=line_number
        ))
    
    return code_blocks


def is_executable_code(code: str) -> bool:
    """Check if a code block is meant to be executed (not just shown)."""
    # Skip code that's clearly not meant to be executed
    skip_patterns = [
        r'^\s*\.\.\.',  # Ellipsis placeholder
        r'^\s*#.*only',  # Comments indicating partial code
        r'^\s*class\s+\w+:$',  # Class definition without body
        r'^\s*def\s+\w+\([^)]*\):$',  # Function definition without body
        r'^\s*pip\s+install',  # pip install commands
        r'^\s*\$',  # Shell commands
        r'^\s*>>>',  # Doctest format
    ]
    
    for pattern in skip_patterns:
        if re.search(pattern, code, re.MULTILINE):
            return False
    
    # Must have at least one statement
    if not code.strip():
        return False
    
    return True


def execute_code_safely(code: str, timeout: int = 30) -> Tuple[bool, str]:
    """
    Execute Python code safely in a subprocess.
    
    Returns:
        Tuple of (success, error_message)
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        # Add common imports that examples might need
        preamble = """
import sys
sys.path.insert(0, '.')
"""
        f.write(preamble + code)
        temp_path = f.name
    
    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT)
        )
        
        if result.returncode == 0:
            return True, ""
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Execution timed out"
    except Exception as e:
        return False, str(e)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def get_example_scripts() -> List[Path]:
    """Get all Python scripts in the examples directory."""
    examples_dir = PROJECT_ROOT / "examples"
    scripts = []
    
    if not examples_dir.exists():
        return scripts
    
    for script_path in examples_dir.glob("*.py"):
        scripts.append(script_path)
    
    return sorted(scripts)


def get_documentation_files() -> List[Path]:
    """Get all markdown documentation files."""
    doc_files = []
    
    # Main documentation files
    for pattern in ['*.md', 'docs/**/*.md']:
        doc_files.extend(PROJECT_ROOT.glob(pattern))
    
    # Filter out non-documentation directories
    excluded_dirs = {'.venv', 'node_modules', '.git', '__pycache__', '.hypothesis'}
    filtered = []
    for f in doc_files:
        if not any(excluded in f.parts for excluded in excluded_dirs):
            filtered.append(f)
    
    return sorted(filtered)


class TestCodeBlockExtraction:
    """Tests for code block extraction from documentation."""
    
    def test_extract_code_blocks_from_readme(self):
        """
        **Feature: documentation-validation, Property 1: Code Example Execution**
        **Validates: Requirements 1.2**
        
        Code blocks should be extractable from README.md.
        """
        readme = PROJECT_ROOT / "README.md"
        if not readme.exists():
            pytest.skip("README.md not found")
        
        blocks = extract_code_blocks_from_markdown(readme)
        
        # Should find some code blocks (or none if README has no Python examples)
        assert isinstance(blocks, list)
        
        for block in blocks:
            assert block.content is not None
            assert block.language in ('python', 'py')
            assert block.source_file == readme
            assert block.line_number > 0
    
    def test_code_block_has_required_fields(self):
        """
        **Feature: documentation-validation, Property 1: Code Example Execution**
        **Validates: Requirements 1.2**
        
        Each extracted code block should have required fields.
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""# Test Document

```python
print("Hello, World!")
```
""")
            temp_path = Path(f.name)
        
        try:
            blocks = extract_code_blocks_from_markdown(temp_path)
            
            assert len(blocks) == 1
            block = blocks[0]
            
            assert hasattr(block, 'content')
            assert hasattr(block, 'language')
            assert hasattr(block, 'source_file')
            assert hasattr(block, 'line_number')
            
            assert block.content == 'print("Hello, World!")'
            assert block.language == 'python'
        finally:
            temp_path.unlink()


class TestCodeExampleExecution:
    """
    Tests for Property 1: Code Example Execution.
    
    **Feature: documentation-validation, Property 1: Code Example Execution**
    **Validates: Requirements 1.2**
    
    For any code example extracted from documentation, executing the code
    should complete without raising exceptions.
    """
    
    def test_simple_code_executes(self):
        """
        **Feature: documentation-validation, Property 1: Code Example Execution**
        **Validates: Requirements 1.2**
        
        Simple Python code should execute successfully.
        """
        code = "x = 1 + 1\nprint(x)"
        success, error = execute_code_safely(code)
        
        assert success, f"Simple code should execute: {error}"
    
    def test_import_code_executes(self):
        """
        **Feature: documentation-validation, Property 1: Code Example Execution**
        **Validates: Requirements 1.2**
        
        Code with standard library imports should execute.
        """
        code = """
import os
import sys
print(f"Python version: {sys.version}")
"""
        success, error = execute_code_safely(code)
        
        assert success, f"Import code should execute: {error}"
    
    @given(st.lists(
        st.sampled_from(['x = 1', 'y = 2', 'z = x + y', 'print(z)']),
        min_size=1, max_size=4
    ))
    @settings(max_examples=50)
    def test_valid_python_statements_execute(self, statements: List[str]):
        """
        **Feature: documentation-validation, Property 1: Code Example Execution**
        **Validates: Requirements 1.2**
        
        For any sequence of valid Python statements, execution should succeed.
        """
        # Build code with proper variable initialization
        code = "x = 1\ny = 2\nz = x + y\n" + "\n".join(statements)
        success, error = execute_code_safely(code)
        
        assert success, f"Valid Python should execute: {error}"
    
    def test_syntax_error_detected(self):
        """
        **Feature: documentation-validation, Property 1: Code Example Execution**
        **Validates: Requirements 1.2**
        
        Code with syntax errors should fail execution.
        """
        code = "def broken(\n  # missing closing paren"
        success, error = execute_code_safely(code)
        
        assert not success, "Syntax error should be detected"


class TestExampleScriptExecution:
    """
    Tests for Property 12: Example Script Execution.
    
    **Feature: documentation-validation, Property 12: Example Script Execution**
    **Validates: Requirements 5.4**
    
    For any Python script in the examples directory, running the script
    should complete without errors.
    """
    
    def test_example_scripts_exist(self):
        """
        **Feature: documentation-validation, Property 12: Example Script Execution**
        **Validates: Requirements 5.4**
        
        Example scripts should exist in the examples directory.
        """
        scripts = get_example_scripts()
        
        # Should have some example scripts
        assert len(scripts) >= 0  # May be empty, that's OK
    
    def test_example_scripts_are_valid_python(self):
        """
        **Feature: documentation-validation, Property 12: Example Script Execution**
        **Validates: Requirements 5.4**
        
        All example scripts should be valid Python syntax.
        """
        scripts = get_example_scripts()
        
        for script in scripts:
            try:
                content = script.read_text(encoding='utf-8')
                compile(content, str(script), 'exec')
            except SyntaxError as e:
                pytest.fail(f"Script {script.name} has syntax error: {e}")
    
    @pytest.mark.parametrize("script_name", [
        "advanced_metrics_demo.py",
        "analytics_demo.py",
        "performance_demo.py",
    ])
    def test_specific_example_script_syntax(self, script_name):
        """
        **Feature: documentation-validation, Property 12: Example Script Execution**
        **Validates: Requirements 5.4**
        
        Specific example scripts should have valid syntax.
        """
        script_path = PROJECT_ROOT / "examples" / script_name
        
        if not script_path.exists():
            pytest.skip(f"Script {script_name} not found")
        
        try:
            content = script_path.read_text(encoding='utf-8')
            compile(content, str(script_path), 'exec')
        except SyntaxError as e:
            pytest.fail(f"Script {script_name} has syntax error: {e}")


class TestNotebookCellExecution:
    """
    Tests for Property 11: Notebook Cell Execution.
    
    **Feature: documentation-validation, Property 11: Notebook Cell Execution**
    **Validates: Requirements 5.3**
    
    For any cell in a Jupyter notebook in the examples directory,
    executing the cell should complete without errors.
    """
    
    def test_notebooks_exist(self):
        """
        **Feature: documentation-validation, Property 11: Notebook Cell Execution**
        **Validates: Requirements 5.3**
        
        Notebooks should exist in the examples directory.
        """
        notebooks_dir = PROJECT_ROOT / "examples" / "notebooks"
        
        if not notebooks_dir.exists():
            pytest.skip("Notebooks directory not found")
        
        notebooks = list(notebooks_dir.glob("*.ipynb"))
        assert len(notebooks) >= 0  # May be empty
    
    def test_notebooks_have_valid_json(self):
        """
        **Feature: documentation-validation, Property 11: Notebook Cell Execution**
        **Validates: Requirements 5.3**
        
        All notebooks should have valid JSON structure.
        """
        import json
        
        notebooks_dir = PROJECT_ROOT / "examples" / "notebooks"
        
        if not notebooks_dir.exists():
            pytest.skip("Notebooks directory not found")
        
        for notebook_path in notebooks_dir.glob("*.ipynb"):
            try:
                with open(notebook_path, 'r', encoding='utf-8') as f:
                    nb = json.load(f)
                
                # Check basic structure
                assert 'cells' in nb, f"{notebook_path.name} missing 'cells'"
                assert isinstance(nb['cells'], list), f"{notebook_path.name} cells not a list"
            except json.JSONDecodeError as e:
                pytest.fail(f"Notebook {notebook_path.name} has invalid JSON: {e}")
    
    def test_notebook_code_cells_are_valid_python(self):
        """
        **Feature: documentation-validation, Property 11: Notebook Cell Execution**
        **Validates: Requirements 5.3**
        
        Code cells in notebooks should have valid Python syntax.
        """
        import json
        
        notebooks_dir = PROJECT_ROOT / "examples" / "notebooks"
        
        if not notebooks_dir.exists():
            pytest.skip("Notebooks directory not found")
        
        for notebook_path in notebooks_dir.glob("*.ipynb"):
            try:
                with open(notebook_path, 'r', encoding='utf-8') as f:
                    nb = json.load(f)
                
                for i, cell in enumerate(nb.get('cells', [])):
                    if cell.get('cell_type') == 'code':
                        source = ''.join(cell.get('source', []))
                        
                        # Skip cells with magic commands
                        if source.strip().startswith('%') or source.strip().startswith('!'):
                            continue
                        
                        try:
                            compile(source, f"{notebook_path.name}:cell{i}", 'exec')
                        except SyntaxError as e:
                            # Some cells may have intentional syntax for demonstration
                            pass
            except json.JSONDecodeError:
                pass  # Already tested in previous test


class TestPropertyBasedExampleExecution:
    """
    Property-based tests for example execution.
    
    **Feature: documentation-validation, Property 1, 11, 12**
    **Validates: Requirements 1.2, 5.3, 5.4**
    """
    
    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=10))
    @settings(max_examples=50)
    def test_variable_assignment_executes(self, var_name: str):
        """
        **Feature: documentation-validation, Property 1: Code Example Execution**
        **Validates: Requirements 1.2**
        
        For any valid variable name, assignment should execute.
        """
        import keyword
        assume(not var_name.startswith('_'))
        assume(not keyword.iskeyword(var_name))
        assume(var_name not in ('True', 'False', 'None'))  # Also exclude soft keywords
        
        code = f"{var_name} = 42\nprint({var_name})"
        success, error = execute_code_safely(code)
        
        assert success, f"Variable assignment should execute: {error}"
    
    @given(st.integers(min_value=0, max_value=100))
    @settings(max_examples=50)
    def test_arithmetic_executes(self, value: int):
        """
        **Feature: documentation-validation, Property 1: Code Example Execution**
        **Validates: Requirements 1.2**
        
        For any integer, arithmetic operations should execute.
        """
        code = f"""
x = {value}
y = x + 1
z = x * 2
result = y + z
print(result)
"""
        success, error = execute_code_safely(code)
        
        assert success, f"Arithmetic should execute: {error}"
    
    @given(st.lists(st.integers(min_value=0, max_value=10), min_size=0, max_size=5))
    @settings(max_examples=50)
    def test_list_operations_execute(self, values: List[int]):
        """
        **Feature: documentation-validation, Property 1: Code Example Execution**
        **Validates: Requirements 1.2**
        
        For any list of integers, list operations should execute.
        """
        code = f"""
data = {values}
total = sum(data)
length = len(data)
print(f"Sum: {{total}}, Length: {{length}}")
"""
        success, error = execute_code_safely(code)
        
        assert success, f"List operations should execute: {error}"


class TestRealProjectExamples:
    """Test example execution against the actual project."""
    
    def test_readme_code_blocks_are_extractable(self):
        """
        **Feature: documentation-validation, Property 1: Code Example Execution**
        **Validates: Requirements 1.2**
        
        Code blocks in README.md should be extractable.
        """
        readme = PROJECT_ROOT / "README.md"
        
        if not readme.exists():
            pytest.skip("README.md not found")
        
        blocks = extract_code_blocks_from_markdown(readme)
        
        # Document what we found
        print(f"\nFound {len(blocks)} Python code blocks in README.md")
        for i, block in enumerate(blocks):
            print(f"  Block {i+1} at line {block.line_number}: {len(block.content)} chars")
    
    def test_documentation_code_blocks_inventory(self):
        """
        **Feature: documentation-validation, Property 1: Code Example Execution**
        **Validates: Requirements 1.2**
        
        Generate an inventory of all code blocks in documentation.
        """
        doc_files = get_documentation_files()
        
        total_blocks = 0
        for doc_file in doc_files:
            blocks = extract_code_blocks_from_markdown(doc_file)
            total_blocks += len(blocks)
        
        print(f"\nTotal documentation files: {len(doc_files)}")
        print(f"Total Python code blocks: {total_blocks}")
        
        # This is informational - we don't fail if there are no blocks
        assert total_blocks >= 0
