"""Pytest configuration: add project root to sys.path for imports."""

import sys
import os

# Add project root so that package imports (configs, data, models, etc.) resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
