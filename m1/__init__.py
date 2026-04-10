import sys
from pathlib import Path

_M1_DIR = Path(__file__).parent
if str(_M1_DIR) not in sys.path:
    sys.path.insert(0, str(_M1_DIR))
