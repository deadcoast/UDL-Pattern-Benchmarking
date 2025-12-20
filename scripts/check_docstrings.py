#!/usr/bin/env python3
"""Check for undocumented public APIs in the udl_rating_framework package."""

import ast
import os
from pathlib import Path


def check_docstrings():
    """Find all public classes and functions without docstrings."""
    undocumented_classes = []
    undocumented_functions = []
    
    for root, dirs, files in os.walk('udl_rating_framework'):
        # Skip __pycache__
        dirs[:] = [d for d in dirs if d != '__pycache__']
        
        for file in files:
            if not file.endswith('.py'):
                continue
            
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Skip private classes
                        if node.name.startswith('_'):
                            continue
                        # Check if first statement is a docstring
                        has_doc = (node.body and 
                                  isinstance(node.body[0], ast.Expr) and 
                                  isinstance(node.body[0].value, ast.Constant) and
                                  isinstance(node.body[0].value.value, str))
                        if not has_doc:
                            undocumented_classes.append((filepath, node.lineno, node.name))
                    
                    elif isinstance(node, ast.FunctionDef):
                        # Skip private functions
                        if node.name.startswith('_') and node.name != '__init__':
                            continue
                        # Check if first statement is a docstring
                        has_doc = (node.body and 
                                  isinstance(node.body[0], ast.Expr) and 
                                  isinstance(node.body[0].value, ast.Constant) and
                                  isinstance(node.body[0].value.value, str))
                        if not has_doc:
                            undocumented_functions.append((filepath, node.lineno, node.name))
            except Exception as e:
                print(f"Error parsing {filepath}: {e}")
                continue
    
    return undocumented_classes, undocumented_functions


def main():
    classes, functions = check_docstrings()
    
    print(f"=== Undocumented Public Classes ({len(classes)}) ===\n")
    for filepath, line, name in sorted(classes)[:20]:
        print(f"  {filepath}:{line} class {name}")
    if len(classes) > 20:
        print(f"  ... and {len(classes) - 20} more")
    
    print(f"\n=== Undocumented Public Functions ({len(functions)}) ===\n")
    for filepath, line, name in sorted(functions)[:20]:
        print(f"  {filepath}:{line} def {name}")
    if len(functions) > 20:
        print(f"  ... and {len(functions) - 20} more")
    
    print(f"\n=== Summary ===")
    print(f"Total undocumented classes: {len(classes)}")
    print(f"Total undocumented functions: {len(functions)}")


if __name__ == '__main__':
    main()
