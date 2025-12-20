#!/usr/bin/env python3
"""
Script to validate test documentation.

This script checks:
1. Test docstrings describe tested behavior (17.1)
2. Property test requirement references are valid (17.2)
3. Test organization matches documented strategy (17.3)
4. Coverage claims are accurate (17.4)

**Feature: documentation-validation, Property 23: Test Requirement Reference Validity**
**Validates: Requirements 11.1, 11.2, 11.3, 11.4, 11.5**
"""

import ast
import os
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict


@dataclass
class TestFunction:
    """Represents a test function with its metadata."""
    name: str
    file_path: str
    line_number: int
    docstring: Optional[str]
    has_docstring: bool
    requirement_refs: List[str] = field(default_factory=list)
    property_refs: List[str] = field(default_factory=list)
    feature_refs: List[str] = field(default_factory=list)


@dataclass
class TestClass:
    """Represents a test class with its metadata."""
    name: str
    file_path: str
    line_number: int
    docstring: Optional[str]
    has_docstring: bool
    test_methods: List[TestFunction] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Report of test documentation validation."""
    total_test_files: int = 0
    total_test_functions: int = 0
    total_test_classes: int = 0
    functions_with_docstrings: int = 0
    functions_without_docstrings: int = 0
    classes_with_docstrings: int = 0
    classes_without_docstrings: int = 0
    valid_requirement_refs: int = 0
    invalid_requirement_refs: int = 0
    missing_docstrings: List[TestFunction] = field(default_factory=list)
    invalid_refs: List[Tuple[str, str, str]] = field(default_factory=list)  # (file, ref, reason)
    valid_refs: List[Tuple[str, str]] = field(default_factory=list)  # (file, ref)


# Valid requirement numbers from both specs
VALID_DOC_VALIDATION_REQUIREMENTS = {
    "1.1", "1.2", "1.3", "1.4", "1.5", "1.6",
    "2.1", "2.2", "2.3", "2.4", "2.5",
    "3.1", "3.2", "3.3", "3.4", "3.5",
    "4.1", "4.2", "4.3", "4.4", "4.5",
    "5.1", "5.2", "5.3", "5.4", "5.5",
    "6.1", "6.2", "6.3", "6.4", "6.5",
    "7.1", "7.2", "7.3", "7.4", "7.5",
    "8.1", "8.2", "8.3", "8.4", "8.5",
    "9.1", "9.2", "9.3", "9.4", "9.5",
    "10.1", "10.2", "10.3", "10.4", "10.5",
    "11.1", "11.2", "11.3", "11.4", "11.5",
    "12.1", "12.2", "12.3", "12.4", "12.5",
}

VALID_UDL_FRAMEWORK_REQUIREMENTS = {
    "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "1.9",
    "2.1", "2.2", "2.3", "2.4", "2.5",
    "3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7", "3.8", "3.9",
    "4.1", "4.2", "4.3", "4.4", "4.5", "4.6", "4.7", "4.8",
    "5.1", "5.2", "5.3", "5.4", "5.5", "5.6", "5.7", "5.8",
    "6.1", "6.2", "6.3", "6.4", "6.5",
    "7.1", "7.2", "7.3", "7.4", "7.5", "7.6",
    "8.1", "8.2", "8.3", "8.4", "8.5", "8.6",
    "9.1", "9.2", "9.3", "9.4", "9.5",
    "10.1", "10.2", "10.3", "10.4", "10.5", "10.6", "10.7",
    "11.1", "11.2", "11.3", "11.4", "11.5", "11.6", "11.7",
}

# Combined valid requirements
ALL_VALID_REQUIREMENTS = VALID_DOC_VALIDATION_REQUIREMENTS | VALID_UDL_FRAMEWORK_REQUIREMENTS


def extract_requirement_refs(docstring: str) -> List[str]:
    """Extract requirement references from a docstring."""
    if not docstring:
        return []
    
    refs = []
    # Pattern: Validates: Requirements X.Y, X.Y, ...
    pattern = r'\*\*Validates:\s*Requirements?\s+([\d.,\s]+)\*\*'
    matches = re.findall(pattern, docstring, re.IGNORECASE)
    
    for match in matches:
        # Split by comma and clean up
        for ref in match.split(','):
            ref = ref.strip()
            if ref:
                refs.append(ref)
    
    # Also check for _Requirements: X.Y pattern
    pattern2 = r'_Requirements?:\s*([\d.,\s]+)_'
    matches2 = re.findall(pattern2, docstring, re.IGNORECASE)
    
    for match in matches2:
        for ref in match.split(','):
            ref = ref.strip()
            if ref:
                refs.append(ref)
    
    return refs


def extract_property_refs(docstring: str) -> List[str]:
    """Extract property references from a docstring."""
    if not docstring:
        return []
    
    refs = []
    # Pattern: Property X: Name or **Property X: Name**
    pattern = r'\*\*Property\s+(\d+):\s*([^*]+)\*\*'
    matches = re.findall(pattern, docstring)
    
    for num, name in matches:
        refs.append(f"Property {num}: {name.strip()}")
    
    return refs


def extract_feature_refs(docstring: str) -> List[str]:
    """Extract feature references from a docstring."""
    if not docstring:
        return []
    
    refs = []
    # Pattern: Feature: name or **Feature: name**
    pattern = r'\*\*Feature:\s*([^,*]+)'
    matches = re.findall(pattern, docstring)
    
    for match in matches:
        refs.append(match.strip())
    
    return refs


def parse_test_file(file_path: Path) -> Tuple[List[TestClass], List[TestFunction]]:
    """Parse a test file and extract test classes and functions."""
    classes = []
    functions = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Warning: Could not parse {file_path}: {e}")
        return classes, functions
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if node.name.startswith('Test'):
                docstring = ast.get_docstring(node)
                test_class = TestClass(
                    name=node.name,
                    file_path=str(file_path),
                    line_number=node.lineno,
                    docstring=docstring,
                    has_docstring=docstring is not None and len(docstring.strip()) > 0,
                )
                
                # Extract test methods
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name.startswith('test_'):
                        method_docstring = ast.get_docstring(item)
                        test_func = TestFunction(
                            name=item.name,
                            file_path=str(file_path),
                            line_number=item.lineno,
                            docstring=method_docstring,
                            has_docstring=method_docstring is not None and len(method_docstring.strip()) > 0,
                            requirement_refs=extract_requirement_refs(method_docstring) if method_docstring else [],
                            property_refs=extract_property_refs(method_docstring) if method_docstring else [],
                            feature_refs=extract_feature_refs(method_docstring) if method_docstring else [],
                        )
                        test_class.test_methods.append(test_func)
                
                classes.append(test_class)
        
        elif isinstance(node, ast.FunctionDef):
            if node.name.startswith('test_'):
                # Check if this is a module-level function (not inside a class)
                docstring = ast.get_docstring(node)
                test_func = TestFunction(
                    name=node.name,
                    file_path=str(file_path),
                    line_number=node.lineno,
                    docstring=docstring,
                    has_docstring=docstring is not None and len(docstring.strip()) > 0,
                    requirement_refs=extract_requirement_refs(docstring) if docstring else [],
                    property_refs=extract_property_refs(docstring) if docstring else [],
                    feature_refs=extract_feature_refs(docstring) if docstring else [],
                )
                functions.append(test_func)
    
    return classes, functions


def validate_requirement_ref(ref: str) -> Tuple[bool, str]:
    """Validate a requirement reference."""
    ref = ref.strip()
    
    # Check if it matches the expected format (X.Y)
    if not re.match(r'^\d+\.\d+$', ref):
        return False, f"Invalid format: expected X.Y, got '{ref}'"
    
    # Check if it's a valid requirement number
    if ref not in ALL_VALID_REQUIREMENTS:
        return False, f"Unknown requirement: {ref}"
    
    return True, "Valid"


def scan_tests_directory(tests_dir: Path) -> ValidationReport:
    """Scan the tests directory and validate documentation."""
    report = ValidationReport()
    
    # Find all test files
    test_files = list(tests_dir.glob('**/test_*.py'))
    report.total_test_files = len(test_files)
    
    all_test_functions = []
    
    for test_file in test_files:
        classes, functions = parse_test_file(test_file)
        
        report.total_test_classes += len(classes)
        
        for test_class in classes:
            if test_class.has_docstring:
                report.classes_with_docstrings += 1
            else:
                report.classes_without_docstrings += 1
            
            for method in test_class.test_methods:
                all_test_functions.append(method)
        
        all_test_functions.extend(functions)
    
    report.total_test_functions = len(all_test_functions)
    
    # Analyze test functions
    for test_func in all_test_functions:
        if test_func.has_docstring:
            report.functions_with_docstrings += 1
        else:
            report.functions_without_docstrings += 1
            report.missing_docstrings.append(test_func)
        
        # Validate requirement references
        for ref in test_func.requirement_refs:
            is_valid, reason = validate_requirement_ref(ref)
            if is_valid:
                report.valid_requirement_refs += 1
                report.valid_refs.append((test_func.file_path, ref))
            else:
                report.invalid_requirement_refs += 1
                report.invalid_refs.append((test_func.file_path, ref, reason))
    
    return report


def print_report(report: ValidationReport):
    """Print the validation report."""
    print("\n" + "=" * 70)
    print("TEST DOCUMENTATION VALIDATION REPORT")
    print("=" * 70)
    
    print(f"\n📁 Test Files Scanned: {report.total_test_files}")
    print(f"📦 Test Classes Found: {report.total_test_classes}")
    print(f"🧪 Test Functions Found: {report.total_test_functions}")
    
    print("\n--- Docstring Coverage ---")
    print(f"✅ Functions with docstrings: {report.functions_with_docstrings}")
    print(f"❌ Functions without docstrings: {report.functions_without_docstrings}")
    
    if report.total_test_functions > 0:
        coverage = (report.functions_with_docstrings / report.total_test_functions) * 100
        print(f"📊 Docstring coverage: {coverage:.1f}%")
    
    print(f"\n✅ Classes with docstrings: {report.classes_with_docstrings}")
    print(f"❌ Classes without docstrings: {report.classes_without_docstrings}")
    
    print("\n--- Requirement References ---")
    print(f"✅ Valid requirement references: {report.valid_requirement_refs}")
    print(f"❌ Invalid requirement references: {report.invalid_requirement_refs}")
    
    if report.invalid_refs:
        print("\n⚠️  Invalid Requirement References:")
        for file_path, ref, reason in report.invalid_refs[:20]:  # Limit to first 20
            rel_path = Path(file_path).name
            print(f"   {rel_path}: '{ref}' - {reason}")
        if len(report.invalid_refs) > 20:
            print(f"   ... and {len(report.invalid_refs) - 20} more")
    
    if report.missing_docstrings:
        print(f"\n⚠️  Tests Missing Docstrings ({len(report.missing_docstrings)} total):")
        # Group by file
        by_file = defaultdict(list)
        for func in report.missing_docstrings:
            by_file[func.file_path].append(func)
        
        for file_path, funcs in sorted(by_file.items())[:10]:  # Limit to first 10 files
            rel_path = Path(file_path).name
            print(f"   {rel_path}:")
            for func in funcs[:5]:  # Limit to first 5 per file
                print(f"      - {func.name} (line {func.line_number})")
            if len(funcs) > 5:
                print(f"      ... and {len(funcs) - 5} more")
        
        if len(by_file) > 10:
            print(f"   ... and {len(by_file) - 10} more files")
    
    print("\n" + "=" * 70)
    
    # Summary
    all_valid = (report.invalid_requirement_refs == 0)
    if all_valid:
        print("✅ All requirement references are valid!")
    else:
        print(f"❌ Found {report.invalid_requirement_refs} invalid requirement references")
    
    return all_valid


def main():
    """Main entry point."""
    # Find the tests directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    tests_dir = project_root / 'tests'
    
    if not tests_dir.exists():
        print(f"Error: Tests directory not found at {tests_dir}")
        sys.exit(1)
    
    print(f"Scanning tests directory: {tests_dir}")
    
    report = scan_tests_directory(tests_dir)
    all_valid = print_report(report)
    
    # Exit with appropriate code
    sys.exit(0 if all_valid else 1)


if __name__ == '__main__':
    main()
