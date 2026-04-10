from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class ExerciseProfile:
    name: str
    aliases: Tuple[str, ...]
    signal_left_angle: str
    signal_right_angle: str
    signal_aggregate: str          # min | mean | max | left | right
    visibility_joints: Tuple[str, ...]
    side_gap_left_angle: Optional[str]
    side_gap_right_angle: Optional[str]
    down_threshold: float          # min deviation from rest to enter active phase (°)
    up_threshold: float            # min return from extreme to count rep (°)
    smooth_window: int
    min_frames_between_reps: int
    dynamic_calibration: bool
    calibration_frames: int
    max_side_gap_deg: float        # max AVERAGE left/right asymmetry allowed (°)
    phase_timeout_frames: int      # kept for API compat
    signal_direction: str = "min_first"   # "min_first" or "max_first"


EXERCISE_PROFILES: Dict[str, ExerciseProfile] = {

    # ── Squat ──────────────────────────────────────────────────────────────
    # Signal: minimum of left/right knee angle (°)
    # Rest (standing):  ~165-175°
    # Active (parallel): ~85-95°
    # Entry: knee drops 35° from standing → we know they're going down
    # Count:  knee rises 45° from their deepest point → rep complete
    "squat": ExerciseProfile(
        name="squat",
        aliases=("squat", "air_squat", "bodyweight_squat"),
        signal_left_angle="left_knee",
        signal_right_angle="right_knee",
        signal_aggregate="mean",
        visibility_joints=(
            "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE",
            "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE",
        ),
        side_gap_left_angle="left_knee",
        side_gap_right_angle="right_knee",
        down_threshold=20.0,      # entry: knee bends 20° below standing baseline
        up_threshold=15.0,        # count:  knee rises 15° from deepest point
        smooth_window=3,
        min_frames_between_reps=5,
        dynamic_calibration=False,
        calibration_frames=75,
        max_side_gap_deg=45.0,
        phase_timeout_frames=120,
        signal_direction="min_first",
    ),

    # ── Push-up ────────────────────────────────────────────────────────────
    "pushup": ExerciseProfile(
        name="pushup",
        aliases=("pushup", "push_up", "push-up"),
        signal_left_angle="left_elbow",
        signal_right_angle="right_elbow",
        signal_aggregate="mean",
        visibility_joints=(
            "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST",
            "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST",
        ),
        side_gap_left_angle="left_elbow",
        side_gap_right_angle="right_elbow",
        down_threshold=25.0,      # entry: elbows bend 25° from extended baseline
        up_threshold=20.0,        # count:  elbows extend 20° from deepest point
        smooth_window=3,
        min_frames_between_reps=5,
        dynamic_calibration=False,
        calibration_frames=75,
        max_side_gap_deg=45.0,
        phase_timeout_frames=120,
        signal_direction="min_first",
    ),

    # ── Shoulder Press ─────────────────────────────────────────────────────
    "shoulder_press": ExerciseProfile(
        name="shoulder_press",
        aliases=("shoulder_press", "overhead_press", "ohp"),
        signal_left_angle="left_elbow",
        signal_right_angle="right_elbow",
        signal_aggregate="mean",
        visibility_joints=(
            "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST",
            "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST",
        ),
        side_gap_left_angle="left_elbow",
        side_gap_right_angle="right_elbow",
        down_threshold=20.0,      # entry: elbows extend 20° above rest baseline
        up_threshold=15.0,        # count:  elbows flex 15° from peak overhead
        smooth_window=3,
        min_frames_between_reps=5,
        dynamic_calibration=False,
        calibration_frames=75,
        max_side_gap_deg=45.0,
        phase_timeout_frames=120,
        signal_direction="max_first",
    ),
}


def normalize_exercise_name(exercise_name: str) -> str:
    return exercise_name.strip().lower().replace(" ", "_")


def get_exercise_profile(exercise_name: str) -> Optional[ExerciseProfile]:
    normalized = normalize_exercise_name(exercise_name)
    for profile in EXERCISE_PROFILES.values():
        if normalized in profile.aliases:
            return profile
    return None


def list_supported_exercises() -> Tuple[str, ...]:
    return tuple(EXERCISE_PROFILES.keys())
