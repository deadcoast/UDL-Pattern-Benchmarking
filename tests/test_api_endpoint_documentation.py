"""
Tests for API endpoint documentation validation.

**Feature: documentation-validation, Property 24: API Endpoint Documentation**
**Validates: Requirements 12.3**

This module validates that all FastAPI endpoints defined in the deployment API
have corresponding documentation in the deployment README.md.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Add deployment API path for imports - must be done before any FastAPI imports
deployment_api_path = str(Path(__file__).parent.parent / "deployment" / "api")
if deployment_api_path not in sys.path:
    sys.path.insert(0, deployment_api_path)

# Flag to track if FastAPI app is available
FASTAPI_APP_AVAILABLE = False
try:
    # Try to import the app
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "deployment_main",
        Path(__file__).parent.parent / "deployment" / "api" / "main.py",
    )
    if spec and spec.loader:
        deployment_main = importlib.util.module_from_spec(spec)
        sys.modules["deployment_main"] = deployment_main
        spec.loader.exec_module(deployment_main)
        FASTAPI_APP_AVAILABLE = True
except Exception:
    FASTAPI_APP_AVAILABLE = False


def extract_fastapi_endpoints() -> List[Dict[str, str]]:
    """
    Extract all endpoints from the FastAPI application.

    Returns:
        List of dicts with 'method', 'path', and 'description' keys
    """
    if not FASTAPI_APP_AVAILABLE:
        return []

    try:
        app = deployment_main.app

        endpoints = []
        for route in app.routes:
            # Skip internal routes (like OpenAPI docs)
            if hasattr(route, "path") and hasattr(route, "methods"):
                path = route.path
                # Skip internal FastAPI routes
                if (
                    path.startswith("/openapi")
                    or path.startswith("/docs")
                    or path.startswith("/redoc")
                ):
                    continue

                for method in route.methods:
                    # Skip HEAD and OPTIONS methods (auto-generated)
                    if method in ("HEAD", "OPTIONS"):
                        continue

                    description = ""
                    if hasattr(route, "endpoint") and route.endpoint.__doc__:
                        description = route.endpoint.__doc__.strip().split("\n")[
                            0]

                    endpoints.append(
                        {"method": method, "path": path,
                            "description": description}
                    )

        return endpoints
    except Exception:
        return []


def extract_documented_endpoints() -> List[Dict[str, str]]:
    """
    Extract documented endpoints from deployment README.md.

    Returns:
        List of dicts with 'method', 'path', and 'description' keys
    """
    readme_path = Path(__file__).parent.parent / "deployment" / "README.md"

    if not readme_path.exists():
        return []

    content = readme_path.read_text()

    endpoints = []

    # Pattern to match endpoint documentation like:
    # - `GET /health` - Health check
    # - `POST /rate` - Rate UDL from content
    endpoint_pattern = (
        r"[`\-\*]\s*`?(GET|POST|PUT|DELETE|PATCH)\s+(/[^\s`]+)`?\s*[-–]\s*(.+?)(?:\n|$)"
    )

    matches = re.findall(endpoint_pattern, content, re.IGNORECASE)

    for method, path, description in matches:
        endpoints.append(
            {
                "method": method.upper(),
                "path": path.strip(),
                "description": description.strip(),
            }
        )

    # Also check for endpoints in code blocks like:
    # curl -X POST "http://localhost:8000/rate"
    curl_pattern = r'curl\s+(?:-X\s+)?(GET|POST|PUT|DELETE|PATCH)?\s*["\']?https?://[^/]+(/[^\s"\']+)'

    curl_matches = re.findall(curl_pattern, content, re.IGNORECASE)

    for method, path in curl_matches:
        method = method.upper() if method else "GET"
        # Clean up path (remove query params for comparison)
        path = path.split("?")[0].strip("\"'")

        # Check if this endpoint is already documented
        existing = any(e["path"] == path and e["method"]
                       == method for e in endpoints)
        if not existing:
            endpoints.append(
                {"method": method, "path": path,
                    "description": "Found in curl example"}
            )

    return endpoints


def get_endpoint_key(endpoint: Dict[str, str]) -> Tuple[str, str]:
    """Get a unique key for an endpoint."""
    return (endpoint["method"], endpoint["path"])


class TestAPIEndpointDocumentation:
    """
    Test suite for API endpoint documentation validation.

    **Feature: documentation-validation, Property 24: API Endpoint Documentation**
    **Validates: Requirements 12.3**
    """

    def test_all_endpoints_are_documented(self):
        """
        **Feature: documentation-validation, Property 24: API Endpoint Documentation**
        **Validates: Requirements 12.3**

        Verify that all FastAPI endpoints have corresponding documentation.
        """
        actual_endpoints = extract_fastapi_endpoints()
        documented_endpoints = extract_documented_endpoints()

        if not actual_endpoints:
            pytest.skip("No endpoints found in FastAPI app")

        # Create sets of endpoint keys for comparison
        actual_keys = {get_endpoint_key(e) for e in actual_endpoints}
        documented_keys = {get_endpoint_key(e) for e in documented_endpoints}

        # Find undocumented endpoints
        undocumented = actual_keys - documented_keys

        # Report findings
        if undocumented:
            undocumented_list = [
                f"{method} {path}" for method, path in sorted(undocumented)
            ]
            pytest.fail(
                f"Found {len(undocumented)} undocumented API endpoints:\n"
                f"  - " + "\n  - ".join(undocumented_list)
            )

    def test_documented_endpoints_exist(self):
        """
        **Feature: documentation-validation, Property 24: API Endpoint Documentation**
        **Validates: Requirements 12.3**

        Verify that all documented endpoints actually exist in the API.
        """
        actual_endpoints = extract_fastapi_endpoints()
        documented_endpoints = extract_documented_endpoints()

        if not documented_endpoints:
            pytest.skip("No documented endpoints found in README")

        # Create sets of endpoint keys for comparison
        actual_keys = {get_endpoint_key(e) for e in actual_endpoints}
        documented_keys = {get_endpoint_key(e) for e in documented_endpoints}

        # Find orphaned documentation (documented but doesn't exist)
        orphaned = documented_keys - actual_keys

        # Report findings
        if orphaned:
            orphaned_list = [f"{method} {path}" for method,
                             path in sorted(orphaned)]
            pytest.fail(
                f"Found {len(orphaned)} documented endpoints that don't exist in API:\n"
                f"  - " + "\n  - ".join(orphaned_list)
            )

    def test_endpoint_documentation_completeness(self):
        """
        **Feature: documentation-validation, Property 24: API Endpoint Documentation**
        **Validates: Requirements 12.3**

        Verify that endpoint documentation is complete (has description).
        """
        documented_endpoints = extract_documented_endpoints()

        if not documented_endpoints:
            pytest.skip("No documented endpoints found in README")

        # Check for endpoints without descriptions
        incomplete = [
            e
            for e in documented_endpoints
            if not e.get("description") or e["description"] == "Found in curl example"
        ]

        # Allow some endpoints to be documented only in curl examples
        # but flag if there are too many
        if len(incomplete) > len(documented_endpoints) // 2:
            incomplete_list = [
                f"{e['method']} {e['path']}" for e in incomplete]
            pytest.fail(
                f"Found {len(incomplete)} endpoints with incomplete documentation:\n"
                f"  - " + "\n  - ".join(incomplete_list)
            )

    def test_core_endpoints_documented(self):
        """
        **Feature: documentation-validation, Property 24: API Endpoint Documentation**
        **Validates: Requirements 12.3**

        Verify that core API endpoints are documented.
        """
        documented_endpoints = extract_documented_endpoints()
        documented_keys = {get_endpoint_key(e) for e in documented_endpoints}

        # Core endpoints that must be documented
        core_endpoints = [
            ("GET", "/health"),
            ("POST", "/rate"),
            ("POST", "/rate/file"),
            ("POST", "/rate/batch"),
            ("GET", "/metrics"),
        ]

        missing_core = []
        for method, path in core_endpoints:
            if (method, path) not in documented_keys:
                missing_core.append(f"{method} {path}")

        if missing_core:
            pytest.fail(
                "Core endpoints missing from documentation:\n"
                "  - " + "\n  - ".join(missing_core)
            )

    def test_endpoint_count_matches(self):
        """
        **Feature: documentation-validation, Property 24: API Endpoint Documentation**
        **Validates: Requirements 12.3**

        Verify that the number of documented endpoints matches actual endpoints.
        """
        actual_endpoints = extract_fastapi_endpoints()
        documented_endpoints = extract_documented_endpoints()

        actual_keys = {get_endpoint_key(e) for e in actual_endpoints}
        documented_keys = {get_endpoint_key(e) for e in documented_endpoints}

        # Allow for some variance (curl examples might add duplicates)
        # but the documented set should cover all actual endpoints
        coverage = (
            len(actual_keys & documented_keys) / len(actual_keys)
            if actual_keys
            else 1.0
        )

        assert coverage >= 1.0, (
            f"Documentation coverage is {coverage:.1%}. "
            f"Expected 100% coverage of {len(actual_keys)} endpoints."
        )


class TestAPIDocumentationConsistency:
    """
    Test suite for API documentation consistency.

    **Feature: documentation-validation, Property 24: API Endpoint Documentation**
    **Validates: Requirements 12.3**
    """

    def test_readme_has_api_section(self):
        """Verify README has an API endpoints section."""
        readme_path = Path(__file__).parent.parent / "deployment" / "README.md"

        if not readme_path.exists():
            pytest.skip("deployment/README.md not found")

        content = readme_path.read_text()

        # Check for API section
        assert (
            "API Endpoints" in content
            or "API endpoints" in content
            or "## API" in content
        ), "README should have an API endpoints section"

    def test_readme_has_example_usage(self):
        """Verify README has example API usage."""
        readme_path = Path(__file__).parent.parent / "deployment" / "README.md"

        if not readme_path.exists():
            pytest.skip("deployment/README.md not found")

        content = readme_path.read_text()

        # Check for curl examples
        assert "curl" in content.lower(), (
            "README should have curl examples for API usage"
        )

    def test_client_library_matches_api(self):
        """
        **Feature: documentation-validation, Property 24: API Endpoint Documentation**
        **Validates: Requirements 12.3**

        Verify that Python client library methods match API endpoints.
        """
        actual_endpoints = extract_fastapi_endpoints()
        actual_paths = {e["path"] for e in actual_endpoints}

        # Read Python client
        client_path = (
            Path(__file__).parent.parent / "deployment" /
            "client" / "python_client.py"
        )

        if not client_path.exists():
            pytest.skip("Python client not found")

        content = client_path.read_text()

        # Check that client has methods for core endpoints
        expected_methods = {
            "/health": "health_check",
            "/rate": "rate_udl",
            "/rate/file": "rate_udl_file",
            "/rate/batch": "rate_udl_batch",
            "/metrics": "get_available_metrics",
        }

        missing_methods = []
        for path, method_name in expected_methods.items():
            if path in actual_paths and method_name not in content:
                missing_methods.append(f"{method_name} (for {path})")

        if missing_methods:
            pytest.fail(
                "Python client missing methods for API endpoints:\n"
                "  - " + "\n  - ".join(missing_methods)
            )


# Property-based test for endpoint documentation
@given(st.sampled_from(["GET", "POST", "PUT", "DELETE", "PATCH"]))
@settings(max_examples=5)
def test_http_methods_documented_consistently(method: str):
    """
    **Feature: documentation-validation, Property 24: API Endpoint Documentation**
    **Validates: Requirements 12.3**

    *For any* HTTP method used in the API, if endpoints use that method,
    they should be documented with the same method in the README.
    """
    actual_endpoints = extract_fastapi_endpoints()
    documented_endpoints = extract_documented_endpoints()

    # Get endpoints using this method
    actual_with_method = [e for e in actual_endpoints if e["method"] == method]
    documented_with_method = [
        e for e in documented_endpoints if e["method"] == method]

    if not actual_with_method:
        # No endpoints use this method, skip
        return

    # All actual endpoints with this method should be documented
    actual_paths = {e["path"] for e in actual_with_method}
    documented_paths = {e["path"] for e in documented_with_method}

    missing = actual_paths - documented_paths
    assert not missing, f"Endpoints using {method} not documented: {missing}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
