import sys
from pathlib import Path

# Add m2/ directory to path so bare imports inside m2 files resolve correctly
_M2_DIR = Path(__file__).parent
if str(_M2_DIR) not in sys.path:
    sys.path.insert(0, str(_M2_DIR))

from m2_api import SkeletonTracker, SkeletonFrame, stream_webcam, stream_video  # noqa: F401
from landmark_utils import EXERCISE_JOINTS, LANDMARK_NAMES, VISIBILITY_THRESHOLD  # noqa: F401
