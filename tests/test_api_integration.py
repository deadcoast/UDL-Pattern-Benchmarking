"""
Integration tests for UDL Rating Framework API endpoints.

This module provides comprehensive integration tests for the FastAPI
endpoints, including authentication, error handling, and end-to-end functionality.
"""

import asyncio
import json
import os

# Import the FastAPI app
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock, patch

import pytest
import requests
from fastapi.testclient import TestClient

deployment_api_path = str(Path(__file__).parent.parent / "deployment" / "api")
if deployment_api_path not in sys.path:
    sys.path.insert(0, deployment_api_path)

try:
    from main import app
except ImportError:
    # Create a mock app for testing if the real app can't be imported
    from fastapi import FastAPI

    app = FastAPI()


class TestAPIAuthentication:
    """Test API authentication mechanisms."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    @patch.dict(os.environ, {"API_TOKEN": "test-secret-token"})
    def test_authentication_required(self):
        """Test that authentication is required when token is set."""
        # Request without token should fail
        response = self.client.post(
            "/rate",
            json={"content": "grammar Test { rule = 'hello' }",
                  "filename": "test.udl"},
        )
        assert response.status_code == 401
        assert "Authentication token required" in response.json()["detail"]

    @patch.dict(os.environ, {"API_TOKEN": "test-secret-token"})
    def test_invalid_token(self):
        """Test that invalid tokens are rejected."""
        headers = {"Authorization": "Bearer invalid-token"}
        response = self.client.post(
            "/rate",
            json={"content": "grammar Test { rule = 'hello' }",
                  "filename": "test.udl"},
            headers=headers,
        )
        assert response.status_code == 401
        assert "Invalid authentication token" in response.json()["detail"]

    @patch.dict(os.environ, {"API_TOKEN": "test-secret-token"})
    @patch("main.rating_pipeline")
    def test_valid_token(self, mock_pipeline):
        """Test that valid tokens are accepted."""
        # Mock the rating pipeline
        mock_report = Mock()
        mock_report.overall_score = 0.85
        mock_report.confidence = 0.92
        mock_report.metric_scores = {"ConsistencyMetric": 0.8}
        mock_report.metric_formulas = {"ConsistencyMetric": "formula"}
        mock_report.computation_trace = []
        mock_pipeline.process_udl.return_value = mock_report

        headers = {"Authorization": "Bearer test-secret-token"}
        response = self.client.post(
            "/rate",
            json={"content": "grammar Test { rule = 'hello' }",
                  "filename": "test.udl"},
            headers=headers,
        )
        assert response.status_code == 200
        assert response.json()["overall_score"] == 0.85

    @patch.dict(os.environ, {}, clear=True)  # No API_TOKEN set
    def test_no_authentication_when_token_not_set(self):
        """Test that no authentication is required when API_TOKEN is not set."""
        with patch("main.rating_pipeline") as mock_pipeline:
            mock_report = Mock()
            mock_report.overall_score = 0.85
            mock_report.confidence = 0.92
            mock_report.metric_scores = {"ConsistencyMetric": 0.8}
            mock_report.metric_formulas = {"ConsistencyMetric": "formula"}
            mock_report.computation_trace = []
            mock_pipeline.process_udl.return_value = mock_report

            # Request without token should succeed
            response = self.client.post(
                "/rate",
                json={
                    "content": "grammar Test { rule = 'hello' }",
                    "filename": "test.udl",
                },
            )
            assert response.status_code == 200


class TestEndpointValidation:
    """Test endpoint input validation and error handling."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    @patch.dict(os.environ, {}, clear=True)  # No authentication
    def test_rate_endpoint_validation(self):
        """Test validation for /rate endpoint."""
        # Missing content
        response = self.client.post("/rate", json={})
        assert response.status_code == 422

        # Invalid content type
        response = self.client.post("/rate", json={"content": 123})
        assert response.status_code == 422

        # Empty content should be allowed
        with patch("main.rating_pipeline") as mock_pipeline:
            mock_report = Mock()
            mock_report.overall_score = 0.0
            mock_report.confidence = 0.5
            mock_report.metric_scores = {}
            mock_report.metric_formulas = {}
            mock_report.computation_trace = []
            mock_pipeline.process_udl.return_value = mock_report

            response = self.client.post("/rate", json={"content": ""})
            assert response.status_code == 200

    @patch.dict(os.environ, {}, clear=True)  # No authentication
    def test_file_upload_validation(self):
        """Test validation for /rate/file endpoint."""
        # Invalid file extension
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".invalid", delete=False
        ) as f:
            f.write("test content")
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                response = self.client.post(
                    "/rate/file", files={"file": ("test.invalid", f, "text/plain")}
                )
            assert response.status_code == 400
            assert "Unsupported file type" in response.json()["detail"]
        finally:
            Path(temp_path).unlink()

        # Missing file
        response = self.client.post("/rate/file")
        assert response.status_code == 422

    @patch.dict(os.environ, {}, clear=True)  # No authentication
    def test_batch_endpoint_validation(self):
        """Test validation for /rate/batch endpoint."""
        # Empty batch
        response = self.client.post("/rate/batch", json={"udls": []})
        assert response.status_code == 200  # Empty batch is allowed

        # Too many UDLs
        udls = [{"content": f"grammar Test{i} {{ rule = 'test' }}"}
                for i in range(15)]
        response = self.client.post("/rate/batch", json={"udls": udls})
        assert response.status_code == 400
        assert "Batch size limited" in response.json()["detail"]

        # Invalid UDL structure
        response = self.client.post(
            "/rate/batch", json={"udls": [{"invalid": "structure"}]}
        )
        assert response.status_code == 422


class TestErrorHandling:
    """Test error handling and recovery scenarios."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    @patch.dict(os.environ, {}, clear=True)  # No authentication
    @patch("main.rating_pipeline", None)
    def test_pipeline_not_initialized(self):
        """Test behavior when rating pipeline is not initialized."""
        response = self.client.post(
            "/rate",
            json={"content": "grammar Test { rule = 'hello' }",
                  "filename": "test.udl"},
        )
        # The API catches the 503 error and returns 500
        assert response.status_code == 500
        assert "Failed to rate UDL" in response.json()["detail"]

        response = self.client.get("/metrics")
        assert response.status_code == 503
        assert "Rating pipeline not initialized" in response.json()["detail"]

    @patch.dict(os.environ, {}, clear=True)  # No authentication
    def test_pipeline_processing_error(self):
        """Test handling of processing errors."""
        with patch("main.rating_pipeline") as mock_pipeline:
            mock_pipeline.process_udl.side_effect = Exception(
                "Processing failed")

            response = self.client.post(
                "/rate",
                json={
                    "content": "grammar Test { rule = 'hello' }",
                    "filename": "test.udl",
                },
            )
            assert response.status_code == 500
            assert "Failed to rate UDL" in response.json()["detail"]

    @patch.dict(os.environ, {}, clear=True)  # No authentication
    def test_file_encoding_error(self):
        """Test handling of file encoding errors."""
        # Create a file with invalid UTF-8
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".udl", delete=False) as f:
            f.write(b"\xff\xfe\x00\x00invalid utf-8")
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                response = self.client.post(
                    "/rate/file", files={"file": ("test.udl", f, "text/plain")}
                )
            assert response.status_code == 400
            assert "valid UTF-8" in response.json()["detail"]
        finally:
            Path(temp_path).unlink()


class TestBatchProcessing:
    """Test batch processing functionality."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    @patch.dict(os.environ, {}, clear=True)  # No authentication
    @patch("main.rating_pipeline")
    def test_batch_parallel_processing(self, mock_pipeline):
        """Test parallel batch processing."""
        mock_report = Mock()
        mock_report.overall_score = 0.8
        mock_report.confidence = 0.9
        mock_report.metric_scores = {"ConsistencyMetric": 0.8}
        mock_report.metric_formulas = {"ConsistencyMetric": "formula"}
        mock_report.computation_trace = []
        mock_pipeline.process_udl.return_value = mock_report

        udls = [
            {
                "content": f"grammar Test{i} {{ rule = 'hello{i}' }}",
                "filename": f"test{i}.udl",
                "use_ctm": False,
                "include_trace": False,
            }
            for i in range(5)
        ]

        response = self.client.post(
            "/rate/batch", json={"udls": udls, "parallel": True}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["successful"] == 5
        assert data["failed"] == 0
        assert len(data["results"]) == 5
        assert data["total_processing_time"] > 0

    @patch.dict(os.environ, {}, clear=True)  # No authentication
    @patch("main.rating_pipeline")
    def test_batch_sequential_processing(self, mock_pipeline):
        """Test sequential batch processing."""
        mock_report = Mock()
        mock_report.overall_score = 0.8
        mock_report.confidence = 0.9
        mock_report.metric_scores = {"ConsistencyMetric": 0.8}
        mock_report.metric_formulas = {"ConsistencyMetric": "formula"}
        mock_report.computation_trace = []
        mock_pipeline.process_udl.return_value = mock_report

        udls = [
            {
                "content": f"grammar Test{i} {{ rule = 'hello{i}' }}",
                "filename": f"test{i}.udl",
            }
            for i in range(3)
        ]

        response = self.client.post(
            "/rate/batch", json={"udls": udls, "parallel": False}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["successful"] == 3
        assert data["failed"] == 0
        assert len(data["results"]) == 3

    @patch.dict(os.environ, {}, clear=True)  # No authentication
    def test_batch_partial_failure(self):
        """Test batch processing with some failures."""
        with patch("main.rating_pipeline") as mock_pipeline:
            # First call succeeds, second fails, third succeeds
            mock_report = Mock()
            mock_report.overall_score = 0.8
            mock_report.confidence = 0.9
            mock_report.metric_scores = {"ConsistencyMetric": 0.8}
            mock_report.metric_formulas = {"ConsistencyMetric": "formula"}
            mock_report.computation_trace = []

            def side_effect(*args, **kwargs):
                if hasattr(side_effect, "call_count"):
                    side_effect.call_count += 1
                else:
                    side_effect.call_count = 1

                if side_effect.call_count == 2:
                    raise Exception("Processing failed")
                return mock_report

            mock_pipeline.process_udl.side_effect = side_effect

            udls = [
                {
                    "content": "grammar Test1 { rule = 'hello1' }",
                    "filename": "test1.udl",
                },
                {
                    "content": "grammar Test2 { rule = 'hello2' }",
                    "filename": "test2.udl",
                },
                {
                    "content": "grammar Test3 { rule = 'hello3' }",
                    "filename": "test3.udl",
                },
            ]

            response = self.client.post(
                "/rate/batch", json={"udls": udls, "parallel": False}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["successful"] == 2
            assert data["failed"] == 1
            assert len(data["results"]) == 2


class TestFileUploadFunctionality:
    """Test file upload and processing functionality."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    @patch.dict(os.environ, {}, clear=True)  # No authentication
    @patch("main.rating_pipeline")
    def test_supported_file_extensions(self, mock_pipeline):
        """Test that all supported file extensions work."""
        mock_report = Mock()
        mock_report.overall_score = 0.8
        mock_report.confidence = 0.9
        mock_report.metric_scores = {"ConsistencyMetric": 0.8}
        mock_report.metric_formulas = {"ConsistencyMetric": "formula"}
        mock_report.computation_trace = []
        mock_pipeline.process_udl.return_value = mock_report

        supported_extensions = [".udl", ".dsl", ".grammar", ".ebnf", ".txt"]

        for ext in supported_extensions:
            with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False) as f:
                f.write("grammar Test { rule = 'hello' }")
                temp_path = f.name

            try:
                with open(temp_path, "rb") as f:
                    response = self.client.post(
                        "/rate/file",
                        files={"file": (f"test{ext}", f, "text/plain")},
                        data={"use_ctm": "false", "include_trace": "false"},
                    )
                assert response.status_code == 200, f"Failed for extension {ext}"
                assert response.json()["overall_score"] == 0.8
            finally:
                Path(temp_path).unlink()

    @patch.dict(os.environ, {}, clear=True)  # No authentication
    @patch("main.rating_pipeline")
    def test_file_upload_with_trace(self, mock_pipeline):
        """Test file upload with computation trace enabled."""
        mock_report = Mock()
        mock_report.overall_score = 0.8
        mock_report.confidence = 0.9
        mock_report.metric_scores = {"ConsistencyMetric": 0.8}
        mock_report.metric_formulas = {"ConsistencyMetric": "formula"}

        # Mock computation trace
        mock_step = Mock()
        mock_step.step_number = 1
        mock_step.operation = "tokenization"
        mock_step.formula = "T(x) = tokens"
        mock_step.output = "tokens"
        mock_report.computation_trace = [mock_step]

        mock_pipeline.process_udl.return_value = mock_report

        with tempfile.NamedTemporaryFile(mode="w", suffix=".udl", delete=False) as f:
            f.write("grammar Test { rule = 'hello' }")
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                response = self.client.post(
                    "/rate/file",
                    files={"file": ("test.udl", f, "text/plain")},
                    data={"use_ctm": "false", "include_trace": "true"},
                )

            assert response.status_code == 200
            data = response.json()
            assert data["trace"] is not None
            assert len(data["trace"]) == 1
            assert data["trace"][0]["step"] == 1
            assert data["trace"][0]["operation"] == "tokenization"
        finally:
            Path(temp_path).unlink()


class TestMetricsEndpoint:
    """Test metrics information endpoint."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    @patch.dict(os.environ, {}, clear=True)  # No authentication
    @patch("main.rating_pipeline")
    def test_metrics_with_real_pipeline(self, mock_pipeline):
        """Test metrics endpoint with mocked pipeline."""
        # Mock metrics
        mock_metric1 = Mock()
        mock_metric1.__class__.__name__ = "ConsistencyMetric"
        mock_metric1.get_formula.return_value = (
            "1 - (|Contradictions| + |Cycles|) / (|Rules| + 1)"
        )
        mock_metric1.get_properties.return_value = {
            "bounded": True,
            "monotonic": False,
            "additive": False,
            "continuous": True,
        }

        mock_metric2 = Mock()
        mock_metric2.__class__.__name__ = "CompletenessMetric"
        mock_metric2.get_formula.return_value = "|Defined| / |Required|"
        mock_metric2.get_properties.return_value = {
            "bounded": True,
            "monotonic": True,
            "additive": False,
            "continuous": True,
        }

        mock_pipeline.metrics = [mock_metric1, mock_metric2]

        response = self.client.get("/metrics")
        assert response.status_code == 200

        data = response.json()
        assert "metrics" in data
        assert len(data["metrics"]) >= 2

        # Check that we have the expected metrics
        metric_names = [m["name"] for m in data["metrics"]]
        assert "ConsistencyMetric" in metric_names
        assert "CompletenessMetric" in metric_names

    @patch.dict(os.environ, {}, clear=True)  # No authentication
    @patch("main.rating_pipeline")
    def test_metrics_with_empty_pipeline(self, mock_pipeline):
        """Test metrics endpoint with empty pipeline."""
        mock_pipeline.metrics = []

        response = self.client.get("/metrics")
        assert response.status_code == 200

        data = response.json()
        assert "metrics" in data
        # Should return default mock metrics
        assert len(data["metrics"]) >= 2

    @patch.dict(os.environ, {}, clear=True)  # No authentication
    @patch("main.rating_pipeline")
    def test_metrics_with_faulty_metric(self, mock_pipeline):
        """Test metrics endpoint with a metric that raises exceptions."""
        mock_metric = Mock()
        mock_metric.__class__.__name__ = "FaultyMetric"
        mock_metric.get_formula.side_effect = Exception("Formula error")
        mock_metric.get_properties.side_effect = Exception("Properties error")

        mock_pipeline.metrics = [mock_metric]

        response = self.client.get("/metrics")
        assert response.status_code == 200

        data = response.json()
        assert "metrics" in data
        # Should handle the faulty metric gracefully
        faulty_metric = next(
            (m for m in data["metrics"] if m["name"] == "FaultyMetric"), None
        )
        if faulty_metric:
            assert faulty_metric["formula"] == "N/A"
            assert faulty_metric["properties"] == {}


class TestHealthEndpoint:
    """Test health check endpoint."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_health_check_basic(self):
        """Test basic health check functionality."""
        response = self.client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "model_loaded" in data
        assert "uptime" in data

        assert data["status"] == "healthy"
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["uptime"], (int, float))
        assert data["uptime"] >= 0

    @patch("main.ctm_model", Mock())
    def test_health_check_with_model(self):
        """Test health check when CTM model is loaded."""
        response = self.client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["model_loaded"] is True

    @patch("main.ctm_model", None)
    def test_health_check_without_model(self):
        """Test health check when CTM model is not loaded."""
        response = self.client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["model_loaded"] is False


class TestRateLimiting:
    """Test rate limiting functionality."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    @patch.dict(os.environ, {}, clear=True)  # No authentication
    @patch("main.rating_pipeline")
    def test_rate_limiting_not_triggered_under_limit(self, mock_pipeline):
        """Test that rate limiting doesn't trigger under normal usage."""
        mock_report = Mock()
        mock_report.overall_score = 0.8
        mock_report.confidence = 0.9
        mock_report.metric_scores = {"ConsistencyMetric": 0.8}
        mock_report.metric_formulas = {"ConsistencyMetric": "formula"}
        mock_report.computation_trace = []
        mock_pipeline.process_udl.return_value = mock_report

        # Make several requests under the limit
        for i in range(3):
            response = self.client.post(
                "/rate",
                json={
                    "content": f"grammar Test{i} {{ rule = 'hello{i}' }}",
                    "filename": f"test{i}.udl",
                },
            )
            assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__])
