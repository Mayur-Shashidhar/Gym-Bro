import math
from typing import Dict, Optional

# ── MediaPipe landmark indices ──────────────────────────────────────────────
# Full list of the 33 BlazePose landmarks with human-readable names.

LANDMARK_NAMES = [
    "NOSE",
    "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR",
    "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_PINKY", "RIGHT_PINKY",
    "LEFT_INDEX", "RIGHT_INDEX",
    "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]

# Index lookup: name → mediapipe index
LANDMARK_INDEX: Dict[str, int] = {name: i for i, name in enumerate(LANDMARK_NAMES)}

# Joints most relevant to exercise tracking (used by M3 and M4)
EXERCISE_JOINTS = [
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW",    "RIGHT_ELBOW",
    "LEFT_WRIST",    "RIGHT_WRIST",
    "LEFT_HIP",      "RIGHT_HIP",
    "LEFT_KNEE",     "RIGHT_KNEE",
    "LEFT_ANKLE",    "RIGHT_ANKLE",
    "NOSE",
]

# Skeleton connections for drawing (pairs of landmark names)
POSE_CONNECTIONS = [
    # Torso
    ("LEFT_SHOULDER",  "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER",  "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP",       "RIGHT_HIP"),
    # Left arm
    ("LEFT_SHOULDER",  "LEFT_ELBOW"),
    ("LEFT_ELBOW",     "LEFT_WRIST"),
    # Right arm
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW",    "RIGHT_WRIST"),
    # Left leg
    ("LEFT_HIP",       "LEFT_KNEE"),
    ("LEFT_KNEE",      "LEFT_ANKLE"),
    # Right leg
    ("RIGHT_HIP",      "RIGHT_KNEE"),
    ("RIGHT_KNEE",     "RIGHT_ANKLE"),
    # Face to shoulders
    ("NOSE",           "LEFT_SHOULDER"),
    ("NOSE",           "RIGHT_SHOULDER"),
]

# Visibility threshold: below this → joint treated as missing
VISIBILITY_THRESHOLD = 0.5


# ── Raw landmark dict ────────────────────────────────────────────────────────

def extract_raw_landmarks(mp_results) -> Optional[Dict[str, Dict]]:
    """
    Convert a MediaPipe Pose results object (solutions API) into a plain dict.
    Used by unit tests via stub objects.
    SkeletonTracker (Tasks API) handles extraction directly in _extract_raw().

    Returns None if no pose was detected.
    Each entry: { "x": float, "y": float, "z": float, "visibility": float }
    """
    if not mp_results or not mp_results.pose_landmarks:
        return None

    landmarks = {}
    for idx, name in enumerate(LANDMARK_NAMES):
        lm = mp_results.pose_landmarks.landmark[idx]
        landmarks[name] = {
            "x":          float(lm.x),
            "y":          float(lm.y),
            "z":          float(lm.z),
            "visibility": float(lm.visibility),
        }
    return landmarks


def extract_world_landmarks(mp_results) -> Optional[Dict[str, Dict]]:
    """
    Extract world (metric) landmarks from MediaPipe.

    World landmarks have the origin at the hip midpoint and are measured in
    metres, making them useful for absolute angle calculations.
    Returns None if no pose was detected.
    """
    if not mp_results or not mp_results.pose_world_landmarks:
        return None

    landmarks = {}
    for idx, name in enumerate(LANDMARK_NAMES):
        lm = mp_results.pose_world_landmarks.landmark[idx]
        landmarks[name] = {
            "x":          float(lm.x),
            "y":          float(lm.y),
            "z":          float(lm.z),
            "visibility": float(lm.visibility),
        }
    return landmarks


# ── Visibility mask ──────────────────────────────────────────────────────────

def build_visibility_mask(
    raw_landmarks: Dict[str, Dict],
    threshold: float = VISIBILITY_THRESHOLD,
) -> Dict[str, bool]:
    """
    Return a bool dict indicating which joints are reliably detected.

    M3 (rep counter) and M4 (form checker) should skip logic for joints
    whose visibility flag is False.
    """
    return {
        name: lm["visibility"] >= threshold
        for name, lm in raw_landmarks.items()
    }


# ── Normalisation ────────────────────────────────────────────────────────────

def normalize_landmarks(
    raw_landmarks: Dict[str, Dict],
    visibility_mask: Dict[str, bool],
) -> Dict[str, Dict]:
    """
    Produce scale- and translation-invariant landmark coordinates.

    Strategy:
      1. Origin → hip midpoint (average of LEFT_HIP and RIGHT_HIP).
      2. Scale  → torso height (distance from hip midpoint to shoulder midpoint).
               Falls back to 1.0 if torso height is degenerate (< 1e-6).

    Only x, y, z are normalised (no visibility score — use the mask instead).

    Why normalise?
    - M1 (classifier) must be robust to where the person stands in frame.
    - M4 (form checker) angle rules work on body-relative coordinates.
    - M6 (dataset) features are comparable across different users and cameras.
    """
    norm: Dict[str, Dict] = {}

    # ── Compute origin (hip midpoint) ──
    left_hip  = raw_landmarks.get("LEFT_HIP")
    right_hip = raw_landmarks.get("RIGHT_HIP")

    if (left_hip and right_hip
            and visibility_mask.get("LEFT_HIP")
            and visibility_mask.get("RIGHT_HIP")):
        ox = (left_hip["x"] + right_hip["x"]) / 2.0
        oy = (left_hip["y"] + right_hip["y"]) / 2.0
        oz = (left_hip["z"] + right_hip["z"]) / 2.0
    else:
        # Fallback: no translation if hips not visible
        ox, oy, oz = 0.0, 0.0, 0.0

    # ── Compute scale (torso height) ──
    left_sh  = raw_landmarks.get("LEFT_SHOULDER")
    right_sh = raw_landmarks.get("RIGHT_SHOULDER")
    scale = 1.0

    if (left_sh and right_sh
            and visibility_mask.get("LEFT_SHOULDER")
            and visibility_mask.get("RIGHT_SHOULDER")):
        sh_mx = (left_sh["x"] + right_sh["x"]) / 2.0
        sh_my = (left_sh["y"] + right_sh["y"]) / 2.0
        sh_mz = (left_sh["z"] + right_sh["z"]) / 2.0
        torso_h = math.sqrt(
            (sh_mx - ox) ** 2 +
            (sh_my - oy) ** 2 +
            (sh_mz - oz) ** 2
        )
        if torso_h > 1e-6:
            scale = torso_h

    # ── Apply transform ──
    for name, lm in raw_landmarks.items():
        norm[name] = {
            "x": (lm["x"] - ox) / scale,
            "y": (lm["y"] - oy) / scale,
            "z": (lm["z"] - oz) / scale,
        }

    return norm


# ── Flat feature vector (for M1 / M6) ────────────────────────────────────────

def to_feature_vector(
    norm_landmarks: Dict[str, Dict],
    visibility_mask: Dict[str, bool],
    joints: list = EXERCISE_JOINTS,
) -> list:
    """
    Flatten normalised landmarks into a 1-D feature vector for ML.

    Only includes joints listed in `joints`.
    Missing joints (visibility False) are filled with zeros.

    Returns a list of floats: [x0, y0, z0, x1, y1, z1, ...]
    Length = len(joints) * 3
    """
    vector = []
    for name in joints:
        lm = norm_landmarks.get(name)
        if lm and visibility_mask.get(name, False):
            vector.extend([lm["x"], lm["y"], lm["z"]])
        else:
            vector.extend([0.0, 0.0, 0.0])
    return vector