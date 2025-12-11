#!/usr/bin/env python3
"""
Calculate actual metric values for all example UDLs to update test expectations.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.abspath('.'))

from udl_rating_framework.core.representation import UDLRepresentation
from udl_rating_framework.core.metrics.consistency import ConsistencyMetric
from udl_rating_framework.core.metrics.completeness import CompletenessMetric
from udl_rating_framework.core.metrics.expressiveness import ExpressivenessMetric
from udl_rating_framework.core.metrics.structural_coherence import StructuralCoherenceMetric
from udl_rating_framework.core.aggregation import MetricAggregator

def calculate_all_examples():
    """Calculate actual values for all example UDLs."""
    
    examples_dir = Path("examples/udl_examples")
    if not examples_dir.exists():
        print(f"Examples directory not found: {examples_dir}")
        return
    
    # Initialize metrics
    metrics = {
        "consistency": ConsistencyMetric(),
        "completeness": CompletenessMetric(),
        "expressiveness": ExpressivenessMetric(),
        "structural_coherence": StructuralCoherenceMetric()
    }
    
    # Default weights
    weights = {
        "consistency": 0.3,
        "completeness": 0.3,
        "expressiveness": 0.2,
        "structural_coherence": 0.2
    }
    aggregator = MetricAggregator(weights)
    
    example_files = [
        "simple_calculator.udl",
        "json_subset.udl",
        "config_language.udl",
        "broken_grammar.udl",
        "state_machine.udl",
        "query_language.udl",
        "template_engine.udl",
        "regex_subset.udl",
        "css_subset.udl",
        "inconsistent_rules.udl",
        "incomplete_spec.udl"
    ]
    
    results = {}
    
    for example_file in example_files:
        file_path = examples_dir / example_file
        if not file_path.exists():
            print(f"Skipping {example_file} - file not found")
            continue
            
        print(f"\n=== {example_file} ===")
        
        try:
            # Load UDL
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            udl = UDLRepresentation(content, str(file_path))
            
            # Calculate metrics
            metric_values = {}
            for name, metric in metrics.items():
                try:
                    value = metric.compute(udl)
                    metric_values[name] = value
                    print(f"  {name}: {value:.6f}")
                except Exception as e:
                    print(f"  {name}: ERROR - {e}")
                    metric_values[name] = 0.0
            
            # Calculate overall score
            try:
                overall = aggregator.aggregate(metric_values)
                print(f"  overall: {overall:.6f}")
                metric_values["overall"] = overall
            except Exception as e:
                print(f"  overall: ERROR - {e}")
                metric_values["overall"] = 0.0
            
            results[example_file] = metric_values
            
        except Exception as e:
            print(f"  ERROR loading {example_file}: {e}")
    
    # Print results in test format
    print("\n" + "="*60)
    print("EXPECTED VALUES FOR TEST FILE:")
    print("="*60)
    
    print("        expected_values = {")
    for example_file, values in results.items():
        print(f'            "{example_file}": {{')
        for metric, value in values.items():
            print(f'                "{metric}": {value:.3f},')
        print("            },")
    print("        }")

if __name__ == "__main__":
    calculate_all_examples()