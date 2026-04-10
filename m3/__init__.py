import sys
from pathlib import Path

# Add m3/ directory to path so bare imports inside m3 files resolve correctly
_M3_DIR = Path(__file__).parent
if str(_M3_DIR) not in sys.path:
    sys.path.insert(0, str(_M3_DIR))

from m3_api import (  # noqa: F401
    compute_named_angles,
    exercise_signal_angle,
    RepCounter,
    RepState,
    default_rep_counter,
    list_supported_exercises,
)
