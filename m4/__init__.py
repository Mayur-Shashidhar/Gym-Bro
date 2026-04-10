import sys
from pathlib import Path

_M4_DIR = Path(__file__).parent
if str(_M4_DIR) not in sys.path:
    sys.path.insert(0, str(_M4_DIR))

from movement_quality_scorer import score_quality  # noqa: F401
