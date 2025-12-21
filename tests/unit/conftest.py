"""
Pytest configuration for unit tests.

This conftest is separate from the main tests/conftest.py to avoid
torch import issues when testing the UDL rating framework.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
