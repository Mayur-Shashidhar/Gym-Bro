from typing import Dict, Optional, Tuple

import numpy as np


def joint_to_vector(landmarks_norm: Dict[str, Dict], joint_name: str) -> Optional[np.ndarray]:
    """Return a 3D vector for one normalized joint (x, y, z)."""
    lm = landmarks_norm.get(joint_name)
    if not lm:
        return None
    return np.array([lm["x"], lm["y"], lm["z"]], dtype=np.float32)


def segment_vector(
    landmarks_norm: Dict[str, Dict],
    start_joint: str,
    end_joint: str,
) -> Optional[np.ndarray]:
    """Return vector from start_joint -> end_joint."""
    p0 = joint_to_vector(landmarks_norm, start_joint)
    p1 = joint_to_vector(landmarks_norm, end_joint)
    if p0 is None or p1 is None:
        return None
    return p1 - p0


def angle_between_vectors_deg(v1: np.ndarray, v2: np.ndarray) -> Optional[float]:
    """
    Return angle in degrees [0, 180] between vectors.
    Returns None for degenerate vectors.
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None

    cosine = float(np.dot(v1, v2) / (n1 * n2))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    angle = float(np.degrees(np.arccos(cosine)))
    return angle


def angle_at_joint_deg(
    landmarks_norm: Dict[str, Dict],
    joint_a: str,
    joint_b: str,
    joint_c: str,
) -> Optional[float]:
    """
    Angle ABC in degrees, where B is the center joint.
    """
    ba = segment_vector(landmarks_norm, joint_b, joint_a)
    bc = segment_vector(landmarks_norm, joint_b, joint_c)
    if ba is None or bc is None:
        return None
    return angle_between_vectors_deg(ba, bc)


def bilateral_mean(left_value: Optional[float], right_value: Optional[float]) -> Optional[float]:
    """Return mean of available left/right values."""
    vals = [v for v in (left_value, right_value) if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def bilateral_min(left_value: Optional[float], right_value: Optional[float]) -> Optional[float]:
    """Return minimum of available left/right values."""
    vals = [v for v in (left_value, right_value) if v is not None]
    if not vals:
        return None
    return float(np.min(vals))


def bilateral_max(left_value: Optional[float], right_value: Optional[float]) -> Optional[float]:
    """Return maximum of available left/right values."""
    vals = [v for v in (left_value, right_value) if v is not None]
    if not vals:
        return None
    return float(np.max(vals))


def both_visible(visible: Dict[str, bool], joints: Tuple[str, ...]) -> bool:
    """Return True when all joints are marked visible."""
    return all(visible.get(j, False) for j in joints)
