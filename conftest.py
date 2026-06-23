"""Root conftest: add src/ to sys.path so ptmc is importable in all tests."""
import sys
from pathlib import Path

_SRC = str(Path(__file__).parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
