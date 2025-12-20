"""Setup script for backward compatibility.

This project uses pyproject.toml as the primary configuration.
This setup.py exists only for compatibility with older pip versions
that don't fully support PEP 517/518.

For development, use: uv sync
For installation, use: pip install .
"""

from setuptools import setup

# All configuration is in pyproject.toml
# This file just triggers the build system
setup()
