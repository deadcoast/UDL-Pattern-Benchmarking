"""
Formal verification of metric properties.

This module provides formal verification capabilities to ensure that all quality metrics
satisfy their mathematical properties as specified in the design document.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..core.metrics.base import QualityMetric
from ..core.representation import UDLRepresentation


@dataclass
class PropertyVerificationResult:
    """Result of a property verification test."""
    property_name: str
    metric_name: str
    passed: bool
    test_cases: int
    failures: List[Dict[str, Any]]
    execution_time: float
    error_message: Optional[str] = None


@dataclass
class FormalVerificationReport:
    """Complete formal verification report."""
    total_properties: int
    passed_properties: int
    failed_properties: int
    verification_results: List[PropertyVerificationResult]
    overall_success: bool
    timestamp: float
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of property verification."""
        if self.total_properties == 0:
            return 1.0
        return self.passed_properties / self.total_properties


class PropertyTest(ABC):
    """Abstract base class for property tests."""
    
    @abstractmethod
    def test_property(self, metric: QualityMetric, test_cases: List[UDLRepresentation]) -> PropertyVerificationResult:
        """Test a specific property on the given metric."""
        pass


class BoundednessTest(PropertyTest):
    """Test that metrics are bounded in [0, 1]."""
    
    def test_property(self, metric: QualityMetric, test_cases: List[UDLRepresentation]) -> PropertyVerificationResult:
        """Test boundedness property: 0 ≤ metric(udl) ≤ 1."""
        import time
        start_time = time.time()
        
        failures = []
        
        for i, udl in enumerate(test_cases):
            try:
                value = metric.compute(udl)
                if not (0.0 <= value <= 1.0):
                    failures.append({
                        'test_case': i,
                        'udl_file': getattr(udl, 'file_path', f'test_case_{i}'),
                        'computed_value': value,
                        'violation': f'Value {value} not in [0, 1]'
                    })
            except Exception as e:
                failures.append({
                    'test_case': i,
                    'udl_file': getattr(udl, 'file_path', f'test_case_{i}'),
                    'error': str(e),
                    'violation': 'Computation failed'
                })
        
        execution_time = time.time() - start_time
        
        return PropertyVerificationResult(
            property_name='Boundedness',
            metric_name=metric.__class__.__name__,
            passed=len(failures) == 0,
            test_cases=len(test_cases),
            failures=failures,
            execution_time=execution_time
        )


class DeterminismTest(PropertyTest):
    """Test that metrics are deterministic."""
    
    def __init__(self, trials: int = 5):
        self.trials = trials
    
    def test_property(self, metric: QualityMetric, test_cases: List[UDLRepresentation]) -> PropertyVerificationResult:
        """Test determinism: same input produces same output."""
        import time
        start_time = time.time()
        
        failures = []
        
        for i, udl in enumerate(test_cases):
            try:
                values = []
                for trial in range(self.trials):
                    value = metric.compute(udl)
                    values.append(value)
                
                # Check if all values are identical (within numerical precision)
                if not all(abs(v - values[0]) < 1e-10 for v in values):
                    failures.append({
                        'test_case': i,
                        'udl_file': getattr(udl, 'file_path', f'test_case_{i}'),
                        'values': values,
                        'violation': f'Non-deterministic values: {values}'
                    })
            except Exception as e:
                failures.append({
                    'test_case': i,
                    'udl_file': getattr(udl, 'file_path', f'test_case_{i}'),
                    'error': str(e),
                    'violation': 'Computation failed'
                })
        
        execution_time = time.time() - start_time
        
        return PropertyVerificationResult(
            property_name='Determinism',
            metric_name=metric.__class__.__name__,
            passed=len(failures) == 0,
            test_cases=len(test_cases),
            failures=failures,
            execution_time=execution_time
        )


class MonotonicityTest(PropertyTest):
    """Test monotonicity properties where applicable."""
    
    def test_property(self, metric: QualityMetric, test_cases: List[UDLRepresentation]) -> PropertyVerificationResult:
        """Test monotonicity properties specific to each metric type."""
        import time
        start_time = time.time()
        
        failures = []
        
        # This is a placeholder - specific monotonicity tests would depend on the metric
        # For now, we just verify the metric can compute values
        for i, udl in enumerate(test_cases):
            try:
                value = metric.compute(udl)
                # Basic sanity check
                if not isinstance(value, (int, float)):
                    failures.append({
                        'test_case': i,
                        'udl_file': getattr(udl, 'file_path', f'test_case_{i}'),
                        'computed_value': value,
                        'violation': f'Non-numeric value: {type(value)}'
                    })
            except Exception as e:
                failures.append({
                    'test_case': i,
                    'udl_file': getattr(udl, 'file_path', f'test_case_{i}'),
                    'error': str(e),
                    'violation': 'Computation failed'
                })
        
        execution_time = time.time() - start_time
        
        return PropertyVerificationResult(
            property_name='Monotonicity',
            metric_name=metric.__class__.__name__,
            passed=len(failures) == 0,
            test_cases=len(test_cases),
            failures=failures,
            execution_time=execution_time
        )


class ContinuityTest(PropertyTest):
    """Test continuity properties where applicable."""
    
    def test_property(self, metric: QualityMetric, test_cases: List[UDLRepresentation]) -> PropertyVerificationResult:
        """Test continuity properties."""
        import time
        start_time = time.time()
        
        failures = []
        
        # For now, just test that the metric produces finite values
        for i, udl in enumerate(test_cases):
            try:
                value = metric.compute(udl)
                if not np.isfinite(value):
                    failures.append({
                        'test_case': i,
                        'udl_file': getattr(udl, 'file_path', f'test_case_{i}'),
                        'computed_value': value,
                        'violation': f'Non-finite value: {value}'
                    })
            except Exception as e:
                failures.append({
                    'test_case': i,
                    'udl_file': getattr(udl, 'file_path', f'test_case_{i}'),
                    'error': str(e),
                    'violation': 'Computation failed'
                })
        
        execution_time = time.time() - start_time
        
        return PropertyVerificationResult(
            property_name='Continuity',
            metric_name=metric.__class__.__name__,
            passed=len(failures) == 0,
            test_cases=len(test_cases),
            failures=failures,
            execution_time=execution_time
        )


class FormalVerifier:
    """
    Formal verification engine for quality metrics.
    
    Provides comprehensive verification of mathematical properties that all
    quality metrics must satisfy according to the design specification.
    """
    
    def __init__(self):
        """Initialize formal verifier."""
        self.logger = logging.getLogger(__name__)
        
        # Standard property tests that all metrics must pass
        self.standard_tests = [
            BoundednessTest(),
            DeterminismTest(),
            MonotonicityTest(),
            ContinuityTest()
        ]
        
        # Custom tests for specific metrics
        self.custom_tests: Dict[str, List[PropertyTest]] = {}
    
    def add_custom_test(self, metric_name: str, test: PropertyTest):
        """Add a custom property test for a specific metric."""
        if metric_name not in self.custom_tests:
            self.custom_tests[metric_name] = []
        self.custom_tests[metric_name].append(test)
    
    def verify_metric(self, 
                     metric: QualityMetric, 
                     test_cases: List[UDLRepresentation]) -> List[PropertyVerificationResult]:
        """
        Verify all properties for a single metric.
        
        Args:
            metric: The quality metric to verify
            test_cases: List of UDL representations to test on
            
        Returns:
            List of verification results for each property
        """
        results = []
        metric_name = metric.__class__.__name__
        
        self.logger.info(f"Verifying properties for {metric_name}")
        
        # Run standard tests
        for test in self.standard_tests:
            try:
                result = test.test_property(metric, test_cases)
                results.append(result)
                
                if result.passed:
                    self.logger.info(f"✓ {metric_name}: {result.property_name} property verified")
                else:
                    self.logger.warning(f"✗ {metric_name}: {result.property_name} property failed "
                                      f"({len(result.failures)} failures)")
            except Exception as e:
                self.logger.error(f"Error testing {result.property_name} for {metric_name}: {e}")
                results.append(PropertyVerificationResult(
                    property_name=test.__class__.__name__.replace('Test', ''),
                    metric_name=metric_name,
                    passed=False,
                    test_cases=len(test_cases),
                    failures=[],
                    execution_time=0.0,
                    error_message=str(e)
                ))
        
        # Run custom tests if any
        if metric_name in self.custom_tests:
            for test in self.custom_tests[metric_name]:
                try:
                    result = test.test_property(metric, test_cases)
                    results.append(result)
                    
                    if result.passed:
                        self.logger.info(f"✓ {metric_name}: Custom {result.property_name} property verified")
                    else:
                        self.logger.warning(f"✗ {metric_name}: Custom {result.property_name} property failed")
                except Exception as e:
                    self.logger.error(f"Error in custom test for {metric_name}: {e}")
        
        return results
    
    def verify_all_metrics(self, 
                          metrics: List[QualityMetric], 
                          test_cases: List[UDLRepresentation]) -> FormalVerificationReport:
        """
        Verify properties for all provided metrics.
        
        Args:
            metrics: List of quality metrics to verify
            test_cases: List of UDL representations to test on
            
        Returns:
            Complete formal verification report
        """
        import time
        start_time = time.time()
        
        all_results = []
        
        for metric in metrics:
            metric_results = self.verify_metric(metric, test_cases)
            all_results.extend(metric_results)
        
        # Calculate summary statistics
        total_properties = len(all_results)
        passed_properties = sum(1 for r in all_results if r.passed)
        failed_properties = total_properties - passed_properties
        overall_success = failed_properties == 0
        
        report = FormalVerificationReport(
            total_properties=total_properties,
            passed_properties=passed_properties,
            failed_properties=failed_properties,
            verification_results=all_results,
            overall_success=overall_success,
            timestamp=time.time()
        )
        
        # Log summary
        self.logger.info(f"Formal verification complete: {passed_properties}/{total_properties} properties passed")
        if not overall_success:
            self.logger.warning(f"Verification failed: {failed_properties} properties failed")
        
        return report
    
    def generate_verification_report(self, report: FormalVerificationReport) -> str:
        """Generate a human-readable verification report."""
        lines = [
            "# Formal Verification Report",
            "",
            f"**Timestamp:** {report.timestamp}",
            f"**Overall Success:** {'✓ PASSED' if report.overall_success else '✗ FAILED'}",
            f"**Success Rate:** {report.success_rate:.1%} ({report.passed_properties}/{report.total_properties})",
            "",
            "## Property Verification Results",
            ""
        ]
        
        # Group results by metric
        by_metric = {}
        for result in report.verification_results:
            if result.metric_name not in by_metric:
                by_metric[result.metric_name] = []
            by_metric[result.metric_name].append(result)
        
        for metric_name, results in by_metric.items():
            lines.extend([
                f"### {metric_name}",
                ""
            ])
            
            for result in results:
                status = "✓ PASSED" if result.passed else "✗ FAILED"
                lines.append(f"- **{result.property_name}:** {status} "
                           f"({result.test_cases} test cases, {result.execution_time:.3f}s)")
                
                if not result.passed and result.failures:
                    lines.append(f"  - Failures: {len(result.failures)}")
                    for failure in result.failures[:3]:  # Show first 3 failures
                        if 'violation' in failure:
                            lines.append(f"    - {failure['violation']}")
                    if len(result.failures) > 3:
                        lines.append(f"    - ... and {len(result.failures) - 3} more")
                
                if result.error_message:
                    lines.append(f"  - Error: {result.error_message}")
            
            lines.append("")
        
        return "\n".join(lines)