"""Ensures the project root is importable as ``spectradet`` during tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
