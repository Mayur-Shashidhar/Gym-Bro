from typing import Dict, Optional

from m3_vector_utils import (
    angle_at_joint_deg,
    bilateral_mean,
    bilateral_max,
    bilateral_min,
    both_visible,
)
from m3_exercise_config import get_exercise_profile


ANGLE_TRIPLETS = {
    "left_knee": ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
    "right_knee": ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
    "left_hip": ("LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"),
    "right_hip": ("RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"),
    "left_elbow": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
    "right_elbow": ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
}


def compute_named_angles(
    landmarks_norm: Dict[str, Dict],
    visible: Dict[str, bool],
) -> Dict[str, Optional[float]]:
    """
    Compute a fixed set of useful joint angles.
    Invisible triplets return None.
    """
    angles: Dict[str, Optional[float]] = {}

    for name, (a, b, c) in ANGLE_TRIPLETS.items():
        if not both_visible(visible, (a, b, c)):
            angles[name] = None
            continue
        angles[name] = angle_at_joint_deg(landmarks_norm, a, b, c)

    return angles


def exercise_signal_angle(
    exercise_name: str,
    angles: Dict[str, Optional[float]],
) -> Optional[float]:
    """
    Convert named angles into one smooth scalar rep signal.

    Signal convention:
    - Higher angle => more extended position
    - Lower angle  => more flexed position
    """
    profile = get_exercise_profile(exercise_name)
    if profile is None:
        return None

    left = angles.get(profile.signal_left_angle)
    right = angles.get(profile.signal_right_angle)

    if profile.signal_aggregate == "min":
        return bilateral_min(left, right)
    if profile.signal_aggregate == "max":
        return bilateral_max(left, right)
    if profile.signal_aggregate == "left":
        return left
    if profile.signal_aggregate == "right":
        return right

    return bilateral_mean(left, right)
