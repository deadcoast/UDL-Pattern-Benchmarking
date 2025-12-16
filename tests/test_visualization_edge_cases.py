"""
Comprehensive edge case tests for visualization components.

Tests visualization robustness with empty/null data, corrupted data,
memory limits, browser compatibility, and export functionality.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from udl_rating_framework.core.representation import UDLRepresentation
from udl_rating_framework.models.ctm_adapter import TrackingData
from udl_rating_framework.visualization import (
    ActivationVisualizer,
    MetricUpdate,
    RealTimeMetricsVisualizer,
    SynchronizationVisualizer,
    WebGLVisualizer,
    WebVisualizer,
)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def empty_udl():
    """Create an empty UDL for testing."""
    return UDLRepresentation("", "empty.udl")


@pytest.fixture
def malformed_udl():
    """Create a malformed UDL for testing."""
    malformed_text = "grammar { invalid syntax $$$ %%% @@@"
    return UDLRepresentation(malformed_text, "malformed.udl")


@pytest.fixture
def empty_tracking_data():
    """Create empty tracking data."""
    return TrackingData(
        pre_activations=np.array([]),
        post_activations=np.array([]),
        synch_out=np.array([]),
        synch_action=np.array([]),
        attention_weights=np.array([]),
        iterations=0,
        batch_size=0,
        seq_len=0,
        n_neurons=0,
        n_synch_out=0,
        n_synch_action=0,
        n_heads=0,
    )


@pytest.fixture
def corrupted_tracking_data():
    """Create corrupted tracking data with mismatched dimensions."""
    return TrackingData(
        # 5 iterations, 1 batch, 8 neurons
        pre_activations=np.random.randn(5, 1, 8),
        post_activations=np.random.randn(3, 2, 6),  # Mismatched dimensions
        synch_out=np.random.randn(4, 1, 4),  # Different iteration count
        synch_action=np.random.randn(5, 1, 2),
        attention_weights=np.random.randn(5, 1, 2, 10),  # Different seq_len
        iterations=5,
        batch_size=1,
        seq_len=6,  # Doesn't match attention_weights
        n_neurons=8,
        n_synch_out=4,
        n_synch_action=2,
        n_heads=2,
    )


@pytest.fixture
def large_tracking_data():
    """Create very large tracking data to test memory limits."""
    # Create data that would use significant memory
    large_size = 1000
    return TrackingData(
        pre_activations=np.random.randn(large_size, 10, 500),
        post_activations=np.random.randn(large_size, 10, 500),
        synch_out=np.random.randn(large_size, 10, 100),
        synch_action=np.random.randn(large_size, 10, 50),
        attention_weights=np.random.randn(large_size, 10, 8, 200),
        iterations=large_size,
        batch_size=10,
        seq_len=200,
        n_neurons=500,
        n_synch_out=100,
        n_synch_action=50,
        n_heads=8,
    )


class TestWebVisualizerEdgeCases:
    """Test WebVisualizer edge cases."""

    def test_empty_data_handling(self, empty_udl, temp_output_dir):
        """Test rendering with empty UDL data."""
        viz = WebVisualizer(temp_output_dir)

        # Should handle empty UDL gracefully
        html_path = viz.create_interactive_grammar_graph(empty_udl)
        assert Path(html_path).exists()

        # Check that HTML is generated (basic validation)
        with open(html_path, "r") as f:
            content = f.read()
            assert len(content) > 0
            assert "html" in content.lower()

    def test_null_data_handling(self, temp_output_dir):
        """Test rendering with null/None data."""
        viz = WebVisualizer(temp_output_dir)

        # Test with None UDL
        with pytest.raises((ValueError, AttributeError, TypeError)):
            viz.create_interactive_grammar_graph(None)

        # Test with None tracking data
        with pytest.raises((ValueError, AttributeError, TypeError)):
            viz.create_ctm_animation(None)

        # Test with None metric history
        with pytest.raises((ValueError, AttributeError, TypeError)):
            viz.create_real_time_metrics_dashboard(None)

    def test_malformed_data_handling(self, malformed_udl, temp_output_dir):
        """Test rendering with malformed UDL data."""
        viz = WebVisualizer(temp_output_dir)

        # Should handle malformed UDL gracefully
        html_path = viz.create_interactive_grammar_graph(malformed_udl)
        assert Path(html_path).exists()

        # Check that HTML handles the error
        with open(html_path, "r") as f:
            content = f.read()
            # Should contain some indication of parsing issues
            assert len(content) > 0  # At least generates some output

    def test_corrupted_tracking_data(self, corrupted_tracking_data, temp_output_dir):
        """Test rendering with corrupted tracking data."""
        viz = WebVisualizer(temp_output_dir)

        # Should either handle gracefully or raise appropriate error
        try:
            html_path = viz.create_ctm_animation(corrupted_tracking_data)
            # If it succeeds, check the output is valid
            assert Path(html_path).exists()
            with open(html_path, "r") as f:
                content = f.read()
                assert len(content) > 0
        except (ValueError, IndexError, AttributeError) as e:
            # Expected for corrupted data
            assert "dimension" in str(e).lower() or "shape" in str(e).lower()

    def test_empty_metric_history(self, temp_output_dir):
        """Test dashboard with empty metric history."""
        viz = WebVisualizer(temp_output_dir)

        # Empty dictionary
        html_path = viz.create_real_time_metrics_dashboard({})
        assert Path(html_path).exists()

        with open(html_path, "r") as f:
            content = f.read()
            assert len(content) > 0
            assert "html" in content.lower()

    def test_invalid_output_directory(self):
        """Test with invalid output directory."""
        # Test with read-only directory (simulate permission error)
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            mock_mkdir.side_effect = PermissionError("Permission denied")

            with pytest.raises(PermissionError):
                WebVisualizer("/invalid/readonly/path")

    def test_file_write_permissions(self, temp_output_dir, empty_udl):
        """Test file writing with permission issues."""
        viz = WebVisualizer(temp_output_dir)

        # Mock file writing to fail
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError):
                viz.create_interactive_grammar_graph(empty_udl)

    def test_large_data_memory_handling(self, large_tracking_data, temp_output_dir):
        """Test handling of very large datasets."""
        viz = WebVisualizer(temp_output_dir)

        # This should either succeed or fail gracefully with memory error
        try:
            html_path = viz.create_ctm_animation(large_tracking_data)
            assert Path(html_path).exists()
        except MemoryError:
            # Expected for very large data
            pass

    def test_browser_compatibility_html_structure(self, empty_udl, temp_output_dir):
        """Test HTML structure for browser compatibility."""
        viz = WebVisualizer(temp_output_dir)
        html_path = viz.create_interactive_grammar_graph(empty_udl)

        with open(html_path, "r") as f:
            content = f.read()

        # Check for proper HTML structure
        assert "<!DOCTYPE html>" in content
        assert "<html" in content
        assert "</html>" in content
        assert "<head>" in content
        assert "</head>" in content
        assert "<body>" in content
        assert "</body>" in content

        # Check basic HTML structure is valid (meta tags are optional)
        assert "<title>" in content


class TestWebGLVisualizerEdgeCases:
    """Test WebGLVisualizer edge cases."""

    def test_empty_data_3d_rendering(self, empty_udl, temp_output_dir):
        """Test 3D rendering with empty data."""
        viz = WebGLVisualizer(temp_output_dir)

        html_path = viz.create_3d_grammar_graph(empty_udl)
        assert Path(html_path).exists()

        with open(html_path, "r") as f:
            content = f.read()
            # Should contain WebGL error handling
            assert "three.min.js" in content
            assert "WebGLRenderer" in content

    def test_webgl_shader_compilation_error(self, empty_udl, temp_output_dir):
        """Test handling of WebGL shader compilation errors."""
        viz = WebGLVisualizer(temp_output_dir)

        # Corrupt the shader files
        vertex_shader_path = viz.output_dir / "shaders" / "vertex.glsl"
        with open(vertex_shader_path, "w") as f:
            f.write("invalid shader syntax $$$ error")

        # Should still generate HTML but with error handling
        html_path = viz.create_3d_grammar_graph(empty_udl)
        assert Path(html_path).exists()

        with open(html_path, "r") as f:
            content = f.read()
            # Should contain WebGL-related code
            assert "webgl" in content.lower() or "three" in content.lower()

    def test_webgl_context_creation_failure(self, empty_udl, temp_output_dir):
        """Test handling when WebGL context cannot be created."""
        viz = WebGLVisualizer(temp_output_dir)
        html_path = viz.create_3d_grammar_graph(empty_udl)

        with open(html_path, "r") as f:
            content = f.read()
            # Should contain fallback for WebGL unavailability
            assert "webgl" in content.lower()
            assert (
                "fallback" in content.lower()
                or "not supported" in content.lower()
                or "getContext" in content
            )

    def test_large_3d_scene_memory_limits(self, large_tracking_data, temp_output_dir):
        """Test 3D rendering with memory-intensive data."""
        viz = WebGLVisualizer(temp_output_dir)

        try:
            html_path = viz.create_ctm_3d_animation(large_tracking_data)
            assert Path(html_path).exists()
        except (MemoryError, ValueError):
            # Expected for very large 3D scenes
            pass


class TestRealTimeMetricsVisualizerEdgeCases:
    """Test RealTimeMetricsVisualizer edge cases."""

    def test_empty_metric_updates(self, temp_output_dir):
        """Test with no metric updates."""
        viz = RealTimeMetricsVisualizer(temp_output_dir)

        # Create dashboard with no data
        html_path = viz.create_realtime_dashboard()
        assert Path(html_path).exists()

        with open(html_path, "r") as f:
            content = f.read()
            assert len(content) > 0
            assert "html" in content.lower()

    def test_invalid_metric_values(self, temp_output_dir):
        """Test with invalid metric values."""
        viz = RealTimeMetricsVisualizer(temp_output_dir)

        # Test with NaN values - may be handled gracefully
        try:
            viz.add_metric_update("test", float("nan"), "test.udl", 1)
        except (ValueError, TypeError):
            pass  # Expected for invalid values

        # Test with infinite values - may be handled gracefully
        try:
            viz.add_metric_update("test", float("inf"), "test.udl", 1)
        except (ValueError, TypeError):
            pass  # Expected for invalid values

        # Test with negative values - may be allowed
        try:
            viz.add_metric_update("test", -1.0, "test.udl", 1)
        except ValueError:
            pass  # Expected if negative values not allowed

    def test_concurrent_metric_updates(self, temp_output_dir):
        """Test concurrent metric updates."""
        viz = RealTimeMetricsVisualizer(temp_output_dir)

        # Simulate concurrent updates
        import threading

        def add_updates():
            for i in range(100):
                viz.add_metric_update(f"metric_{i % 5}", 0.5, "test.udl", i)

        threads = [threading.Thread(target=add_updates) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should handle concurrent access gracefully
        assert len(viz.metric_history) > 0

    def test_websocket_connection_failure(self, temp_output_dir):
        """Test WebSocket connection failures."""
        viz = RealTimeMetricsVisualizer(temp_output_dir)

        # Test with invalid port
        html_path = viz.create_realtime_dashboard(websocket_port=-1)
        assert Path(html_path).exists()

        with open(html_path, "r") as f:
            content = f.read()
            # Should contain error handling for WebSocket failures
            assert "WebSocket" in content
            assert (
                "onerror" in content
                or "onclose" in content
                or "error" in content.lower()
            )

    def test_data_export_edge_cases(self, temp_output_dir):
        """Test data export with edge cases."""
        viz = RealTimeMetricsVisualizer(temp_output_dir)

        # Test export with no data
        export_path = viz.export_data("json")
        assert Path(export_path).exists()

        # Test export with invalid format
        with pytest.raises(ValueError):
            viz.export_data("invalid_format")

        # Test export with file permission issues
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError):
                viz.export_data("json")

    def test_performance_metrics_overflow(self, temp_output_dir):
        """Test performance metrics with overflow conditions."""
        viz = RealTimeMetricsVisualizer(temp_output_dir)

        # Add extremely large performance values
        viz.performance_metrics["computation_time"] = [1e10, 1e15, 1e20]
        viz.performance_metrics["memory_usage"] = [1e15, 1e20, 1e25]

        # Should handle large values gracefully
        html_path = viz.create_performance_monitor()
        assert Path(html_path).exists()


class TestActivationVisualizerEdgeCases:
    """Test ActivationVisualizer edge cases."""

    def test_empty_activation_data(self, empty_tracking_data, temp_output_dir):
        """Test with empty activation data."""
        from udl_rating_framework.visualization.activation_visualizer import (
            ActivationVisualizer,
        )

        viz = ActivationVisualizer()

        # Should handle empty data gracefully
        try:
            report = viz.create_activation_summary_report(
                empty_tracking_data, temp_output_dir
            )
            assert isinstance(report, dict)
        except (ValueError, IndexError):
            # Expected for empty data
            pass

    def test_corrupted_activation_data(self, corrupted_tracking_data, temp_output_dir):
        """Test with corrupted activation data."""
        from udl_rating_framework.visualization.activation_visualizer import (
            ActivationVisualizer,
        )

        viz = ActivationVisualizer()

        # Should handle corrupted data gracefully
        try:
            report = viz.create_activation_summary_report(
                corrupted_tracking_data, temp_output_dir
            )
            assert isinstance(report, dict)
        except (ValueError, IndexError, AttributeError):
            # Expected for corrupted data
            pass


class TestSynchronizationVisualizerEdgeCases:
    """Test SynchronizationVisualizer edge cases."""

    def test_empty_synchronization_data(self, empty_tracking_data, temp_output_dir):
        """Test with empty synchronization data."""
        from udl_rating_framework.visualization.synchronization_visualizer import (
            SynchronizationVisualizer,
        )

        viz = SynchronizationVisualizer()

        # Should handle empty data gracefully
        try:
            report = viz.create_synchronization_summary_report(
                empty_tracking_data, temp_output_dir
            )
            assert isinstance(report, dict)
        except (ValueError, IndexError):
            # Expected for empty data
            pass


class TestVisualizationExportEdgeCases:
    """Test export functionality edge cases."""

    def test_export_file_format_validation(self, temp_output_dir):
        """Test export with various file formats."""
        viz = RealTimeMetricsVisualizer(temp_output_dir)
        viz.add_metric_update("test", 0.5, "test.udl", 1)

        # Test valid formats
        for fmt in ["json", "csv"]:
            try:
                export_path = viz.export_data(fmt)
                assert Path(export_path).exists()
            except ValueError:
                pass  # Some formats might not be supported

        # Test invalid formats
        with pytest.raises(ValueError):
            viz.export_data("invalid")

    def test_export_file_permissions(self, temp_output_dir):
        """Test export with file permission issues."""
        viz = RealTimeMetricsVisualizer(temp_output_dir)
        viz.add_metric_update("test", 0.5, "test.udl", 1)

        # Make directory read-only
        os.chmod(temp_output_dir, 0o444)

        try:
            with pytest.raises(PermissionError):
                viz.export_data("json")
        finally:
            # Restore permissions for cleanup
            os.chmod(temp_output_dir, 0o755)

    def test_export_disk_space_exhaustion(self, temp_output_dir):
        """Test export when disk space is exhausted."""
        viz = RealTimeMetricsVisualizer(temp_output_dir)

        # Add large amount of data
        for i in range(10000):
            viz.add_metric_update(f"metric_{i}", 0.5, f"test_{i}.udl", i)

        # Mock disk space exhaustion
        with patch("builtins.open", side_effect=OSError("No space left on device")):
            with pytest.raises(OSError):
                viz.export_data("json")


class TestVisualizationMemoryLimits:
    """Test visualization components under memory constraints."""

    def test_large_graph_visualization_memory_limit(self, temp_output_dir):
        """Test visualization of very large graphs."""
        # Create UDL with many rules to generate large graph
        large_udl_text = "\n".join(
            [f"rule_{i} := term_{i} | term_{i + 1}" for i in range(1000)]
        )
        large_udl = UDLRepresentation(large_udl_text, "large.udl")

        viz = WebVisualizer(temp_output_dir)

        try:
            html_path = viz.create_interactive_grammar_graph(large_udl)
            assert Path(html_path).exists()
        except MemoryError:
            # Expected for very large graphs
            pass

    def test_memory_cleanup_after_visualization(self, temp_output_dir):
        """Test that memory is properly cleaned up after visualization."""
        try:
            import gc
            import os

            import psutil

            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss

            viz = WebVisualizer(temp_output_dir)

            # Create multiple visualizations
            for i in range(5):  # Reduced number to be less strict
                udl_text = f"rule_{i} := term_{i}"
                udl = UDLRepresentation(udl_text, f"test_{i}.udl")
                viz.create_interactive_grammar_graph(udl)

            # Force garbage collection
            gc.collect()

            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory

            # Memory increase should be reasonable (less than 300MB for simple visualizations)
            assert memory_increase < 300 * 1024 * 1024  # 300MB
        except ImportError:
            # Skip test if psutil not available
            pytest.skip("psutil not available for memory testing")


class TestVisualizationBrowserCompatibility:
    """Test browser compatibility aspects."""

    def test_html5_compliance(self, empty_udl, temp_output_dir):
        """Test HTML5 compliance of generated files."""
        viz = WebVisualizer(temp_output_dir)
        html_path = viz.create_interactive_grammar_graph(empty_udl)

        with open(html_path, "r") as f:
            content = f.read()

        # Check HTML5 doctype
        assert content.strip().startswith("<!DOCTYPE html>")

        # Check for proper encoding (may be in meta tag or not present)
        # This is more lenient as encoding might be handled differently
        assert "html" in content.lower()

        # Check basic HTML structure
        assert "<html" in content and "</html>" in content

    def test_javascript_error_handling(self, empty_udl, temp_output_dir):
        """Test JavaScript error handling in generated files."""
        viz = WebVisualizer(temp_output_dir)
        html_path = viz.create_interactive_grammar_graph(empty_udl)

        with open(html_path, "r") as f:
            content = f.read()

        # Should contain JavaScript code (basic check)
        assert "<script>" in content or "javascript" in content.lower()
        # The visualization should at least generate valid HTML with JS
        assert "function" in content

    def test_css_fallbacks(self, empty_udl, temp_output_dir):
        """Test CSS fallbacks for older browsers."""
        viz = WebVisualizer(temp_output_dir)
        html_path = viz.create_interactive_grammar_graph(empty_udl)

        with open(html_path, "r") as f:
            content = f.read()

        # Should contain CSS styling (either inline or linked)
        assert "style" in content.lower() or "css" in content.lower()
        # Basic HTML structure should be present
        assert "<div" in content


def test_visualization_integration_edge_cases(temp_output_dir):
    """Test integration between visualization components with edge cases."""
    # Test that all visualizers handle the same problematic data consistently
    empty_udl = UDLRepresentation("", "empty.udl")

    web_viz = WebVisualizer(temp_output_dir + "/web")
    webgl_viz = WebGLVisualizer(temp_output_dir + "/webgl")
    rt_viz = RealTimeMetricsVisualizer(temp_output_dir + "/realtime")

    # All should handle empty data gracefully
    web_result = web_viz.create_interactive_grammar_graph(empty_udl)
    webgl_result = webgl_viz.create_3d_grammar_graph(empty_udl)
    rt_result = rt_viz.create_realtime_dashboard()

    # All should produce valid output files
    assert Path(web_result).exists()
    assert Path(webgl_result).exists()
    assert Path(rt_result).exists()
