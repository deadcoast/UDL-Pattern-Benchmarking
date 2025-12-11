"""
Tests for enhanced visualization capabilities.

Tests the new visualization components including web-based, WebGL,
and real-time visualizations.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from udl_rating_framework.visualization import (
    WebVisualizer,
    WebGLVisualizer, 
    RealTimeMetricsVisualizer,
    MetricUpdate
)
from udl_rating_framework.core.representation import UDLRepresentation
from udl_rating_framework.models.ctm_adapter import TrackingData


@pytest.fixture
def sample_udl():
    """Create a sample UDL for testing."""
    udl_text = """
    grammar TestGrammar {
        expr := term ('+' term)*
        term := factor ('*' factor)*
        factor := number | '(' expr ')'
        number := digit+
        digit := '0' | '1' | '2'
    }
    """
    return UDLRepresentation(udl_text, "test.udl")


@pytest.fixture
def sample_tracking_data():
    """Create sample tracking data for testing."""
    iterations = 5
    batch_size = 1
    n_neurons = 8
    n_synch_out = 4
    n_synch_action = 2
    seq_len = 6
    n_heads = 2
    
    pre_activations = np.random.randn(iterations, batch_size, n_neurons) * 0.5
    post_activations = np.tanh(pre_activations)
    synch_out = np.random.rand(iterations, batch_size, n_synch_out)
    synch_action = np.random.rand(iterations, batch_size, n_synch_action)
    attention_weights = np.random.rand(iterations, batch_size, n_heads, seq_len)
    attention_weights = attention_weights / attention_weights.sum(axis=-1, keepdims=True)
    
    return TrackingData(
        pre_activations=pre_activations,
        post_activations=post_activations,
        synch_out=synch_out,
        synch_action=synch_action,
        attention_weights=attention_weights,
        iterations=iterations,
        batch_size=batch_size,
        seq_len=seq_len,
        n_neurons=n_neurons,
        n_synch_out=n_synch_out,
        n_synch_action=n_synch_action,
        n_heads=n_heads
    )


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestWebVisualizer:
    """Test web-based interactive visualizations."""
    
    def test_web_visualizer_initialization(self, temp_output_dir):
        """Test WebVisualizer initialization."""
        viz = WebVisualizer(temp_output_dir)
        assert viz.output_dir.exists()
        assert (viz.output_dir / "static" / "style.css").exists()
    
    def test_interactive_grammar_graph(self, sample_udl, temp_output_dir):
        """Test interactive grammar graph creation."""
        viz = WebVisualizer(temp_output_dir)
        html_path = viz.create_interactive_grammar_graph(sample_udl)
        
        assert Path(html_path).exists()
        
        # Check HTML content
        with open(html_path, 'r') as f:
            content = f.read()
            assert "Interactive Grammar Graph" in content
            assert "d3js.org" in content
            assert "three.min.js" in content
    
    def test_ctm_animation(self, sample_tracking_data, temp_output_dir):
        """Test CTM animation creation."""
        viz = WebVisualizer(temp_output_dir)
        html_path = viz.create_ctm_animation(sample_tracking_data)
        
        assert Path(html_path).exists()
        
        # Check HTML content
        with open(html_path, 'r') as f:
            content = f.read()
            assert "CTM Processing Animation" in content
            assert "playAnimation" in content
            assert "pauseAnimation" in content
    
    def test_metrics_dashboard(self, temp_output_dir):
        """Test real-time metrics dashboard creation."""
        viz = WebVisualizer(temp_output_dir)
        metric_history = {
            "consistency": [0.5, 0.6, 0.7],
            "completeness": [0.4, 0.5, 0.6]
        }
        html_path = viz.create_real_time_metrics_dashboard(metric_history)
        
        assert Path(html_path).exists()
        
        # Check HTML content
        with open(html_path, 'r') as f:
            content = f.read()
            assert "Real-time Metrics Dashboard" in content
            assert "consistency" in content
            assert "completeness" in content
    
    def test_dependency_flow_diagram(self, sample_udl, temp_output_dir):
        """Test dependency flow diagram creation."""
        viz = WebVisualizer(temp_output_dir)
        html_path = viz.create_dependency_flow_diagram(sample_udl)
        
        assert Path(html_path).exists()
        
        # Check HTML content
        with open(html_path, 'r') as f:
            content = f.read()
            assert "Grammar Dependency Flow" in content
            assert "d3-sankey" in content


class TestWebGLVisualizer:
    """Test WebGL-based 3D visualizations."""
    
    def test_webgl_visualizer_initialization(self, temp_output_dir):
        """Test WebGLVisualizer initialization."""
        viz = WebGLVisualizer(temp_output_dir)
        assert viz.output_dir.exists()
        assert (viz.output_dir / "shaders" / "vertex.glsl").exists()
        assert (viz.output_dir / "shaders" / "fragment.glsl").exists()
    
    def test_3d_grammar_graph(self, sample_udl, temp_output_dir):
        """Test 3D grammar graph creation."""
        viz = WebGLVisualizer(temp_output_dir)
        html_path = viz.create_3d_grammar_graph(sample_udl)
        
        assert Path(html_path).exists()
        
        # Check HTML content
        with open(html_path, 'r') as f:
            content = f.read()
            assert "3D Grammar Graph" in content
            assert "three.min.js" in content
            assert "OrbitControls" in content
    
    def test_ctm_3d_animation(self, sample_tracking_data, temp_output_dir):
        """Test 3D CTM animation creation."""
        viz = WebGLVisualizer(temp_output_dir)
        html_path = viz.create_ctm_3d_animation(sample_tracking_data)
        
        assert Path(html_path).exists()
        
        # Check HTML content
        with open(html_path, 'r') as f:
            content = f.read()
            assert "3D CTM Animation" in content
            assert "WebGLRenderer" in content
            assert "PerspectiveCamera" in content


class TestRealTimeMetricsVisualizer:
    """Test real-time metrics visualization."""
    
    def test_realtime_visualizer_initialization(self, temp_output_dir):
        """Test RealTimeMetricsVisualizer initialization."""
        viz = RealTimeMetricsVisualizer(temp_output_dir)
        assert viz.output_dir.exists()
        assert len(viz.metric_history) == 0
        assert len(viz.active_computations) == 0
    
    def test_add_metric_update(self, temp_output_dir):
        """Test adding metric updates."""
        viz = RealTimeMetricsVisualizer(temp_output_dir)
        
        viz.add_metric_update("consistency", 0.75, "test.udl", 1)
        
        assert "consistency" in viz.metric_history
        assert len(viz.metric_history["consistency"]) == 1
        
        update = viz.metric_history["consistency"][0]
        assert update.metric_name == "consistency"
        assert update.value == 0.75
        assert update.udl_file == "test.udl"
        assert update.iteration == 1
    
    def test_computation_tracking(self, temp_output_dir):
        """Test computation tracking."""
        viz = RealTimeMetricsVisualizer(temp_output_dir)
        
        viz.start_computation_tracking("comp1", "test.udl")
        assert "comp1" in viz.active_computations
        assert viz.active_computations["comp1"]["status"] == "running"
        
        viz.finish_computation_tracking("comp1")
        assert viz.active_computations["comp1"]["status"] == "completed"
        assert "duration" in viz.active_computations["comp1"]
    
    def test_realtime_dashboard_creation(self, temp_output_dir):
        """Test real-time dashboard creation."""
        viz = RealTimeMetricsVisualizer(temp_output_dir)
        
        # Add some sample data
        viz.add_metric_update("consistency", 0.75, "test.udl", 1)
        viz.add_metric_update("completeness", 0.65, "test.udl", 1)
        
        html_path = viz.create_realtime_dashboard()
        
        assert Path(html_path).exists()
        
        # Check HTML content
        with open(html_path, 'r') as f:
            content = f.read()
            assert "Real-time Metrics Dashboard" in content
            assert "WebSocket" in content
            assert "plotly" in content
    
    def test_performance_monitor_creation(self, temp_output_dir):
        """Test performance monitor creation."""
        viz = RealTimeMetricsVisualizer(temp_output_dir)
        
        # Add performance data
        viz.performance_metrics["computation_time"] = [1.2, 1.5, 1.1]
        viz.performance_metrics["memory_usage"] = [100000, 120000, 110000]
        
        html_path = viz.create_performance_monitor()
        
        assert Path(html_path).exists()
        
        # Check HTML content
        with open(html_path, 'r') as f:
            content = f.read()
            assert "Performance Monitor" in content
            assert "computation_time" in content
    
    def test_metric_comparison_view(self, temp_output_dir):
        """Test metric comparison view creation."""
        viz = RealTimeMetricsVisualizer(temp_output_dir)
        
        # Add sample data
        for i in range(5):
            viz.add_metric_update("consistency", 0.7 + i * 0.05, "test.udl", i)
            viz.add_metric_update("completeness", 0.6 + i * 0.04, "test.udl", i)
        
        html_path = viz.create_metric_comparison_view(["consistency", "completeness"])
        
        assert Path(html_path).exists()
        
        # Check HTML content
        with open(html_path, 'r') as f:
            content = f.read()
            assert "Metric Comparison" in content
            assert "consistency" in content
            assert "completeness" in content
    
    def test_data_export(self, temp_output_dir):
        """Test data export functionality."""
        viz = RealTimeMetricsVisualizer(temp_output_dir)
        
        # Add sample data
        viz.add_metric_update("consistency", 0.75, "test.udl", 1)
        viz.performance_metrics["computation_time"] = [1.2, 1.5]
        
        # Test JSON export
        export_path = viz.export_data("json")
        assert Path(export_path).exists()
        
        # Check exported data
        import json
        with open(export_path, 'r') as f:
            data = json.load(f)
            assert "metric_history" in data
            assert "performance_metrics" in data
            assert "consistency" in data["metric_history"]


class TestMetricUpdate:
    """Test MetricUpdate dataclass."""
    
    def test_metric_update_creation(self):
        """Test MetricUpdate creation."""
        update = MetricUpdate(
            timestamp=1234567890.0,
            metric_name="consistency",
            value=0.75,
            udl_file="test.udl",
            iteration=1
        )
        
        assert update.timestamp == 1234567890.0
        assert update.metric_name == "consistency"
        assert update.value == 0.75
        assert update.udl_file == "test.udl"
        assert update.iteration == 1


def test_visualization_integration(sample_udl, sample_tracking_data, temp_output_dir):
    """Test integration between different visualization components."""
    # Test that all visualizers can work with the same data
    web_viz = WebVisualizer(temp_output_dir + "/web")
    webgl_viz = WebGLVisualizer(temp_output_dir + "/webgl")
    rt_viz = RealTimeMetricsVisualizer(temp_output_dir + "/realtime")
    
    # Create visualizations
    web_grammar = web_viz.create_interactive_grammar_graph(sample_udl)
    web_ctm = web_viz.create_ctm_animation(sample_tracking_data)
    
    webgl_grammar = webgl_viz.create_3d_grammar_graph(sample_udl)
    webgl_ctm = webgl_viz.create_ctm_3d_animation(sample_tracking_data)
    
    rt_viz.add_metric_update("test_metric", 0.8, "test.udl", 1)
    rt_dashboard = rt_viz.create_realtime_dashboard()
    
    # Verify all files exist
    assert Path(web_grammar).exists()
    assert Path(web_ctm).exists()
    assert Path(webgl_grammar).exists()
    assert Path(webgl_ctm).exists()
    assert Path(rt_dashboard).exists()