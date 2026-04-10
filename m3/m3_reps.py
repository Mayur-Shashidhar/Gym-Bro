from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional
from collections import deque

import numpy as np

from m3_angles import compute_named_angles, exercise_signal_angle
from m3_exercise_config import ExerciseProfile, get_exercise_profile


@dataclass
class RepState:
    rep_count: int = 0
    phase: str = "start"          # start → down → up
    last_signal: Optional[float] = None
    smoothed_signal: Optional[float] = None
    last_rep_frame: int = -10**9
    valid_signal_frames: int = 0
    last_rep_confidence: float = 0.0
    rep_confidences: List[float] = field(default_factory=list)
    rejected_reps: int = 0
    calibration_complete: bool = False
    calibration_frames_seen: int = 0
    active_down_threshold: Optional[float] = None
    active_up_threshold: Optional[float] = None


@dataclass
class RepCounter:
    # ── Required ──────────────────────────────────────────────────────────
    exercise_name: str
    down_threshold: float          # min deviation from baseline to enter rep
    up_threshold: float            # min return from extreme to count/reject

    # ── Optional public ───────────────────────────────────────────────────
    smooth_window: int = 10        # EMA window (frames); larger = smoother
    min_frames_between_reps: int = 12
    dynamic_calibration: bool = False   # kept for API compat, unused
    calibration_frames: int = 75        # kept for API compat, unused
    calibration_low_percentile: float = 20.0
    calibration_high_percentile: float = 85.0
    calibration_threshold_margin: float = 5.0
    min_threshold_gap: float = 20.0
    max_side_gap_deg: float = 35.0
    phase_timeout_frames: int = 120    # kept for API compat, unused
    signal_direction: str = "min_first"  # "min_first" or "max_first"

    # ── Internal ──────────────────────────────────────────────────────────
    _buffer: Deque[float] = field(default_factory=lambda: deque(maxlen=10))
    _calibration_values: List[float] = field(default_factory=list)
    _phase_start_frame: int = -1

    # Cycle quality metrics (reset each rep)
    _cycle_side_gap_sum: float = 0.0
    _cycle_side_gap_count: int = 0
    _cycle_signal_min: Optional[float] = None
    _cycle_signal_max: Optional[float] = None
    _cycle_visibility_sum: float = 0.0
    _cycle_visibility_count: int = 0
    _cycle_roughness_sum: float = 0.0
    _cycle_last_smooth: Optional[float] = None

    # Adaptive baseline & rep tracking
    _baseline: Optional[float] = None
    _baseline_frames_seen: int = 0
    _active_extreme: Optional[float] = None   # deepest / highest point in rep

    _profile: Optional[ExerciseProfile] = None
    state: RepState = field(default_factory=RepState)

    # ── Init ──────────────────────────────────────────────────────────────

    def __post_init__(self):
        self._profile = get_exercise_profile(self.exercise_name)
        # Override signal_direction if profile provides it
        if self._profile is not None and hasattr(self._profile, "signal_direction"):
            object.__setattr__(self, "signal_direction", self._profile.signal_direction) \
                if hasattr(type(self), "__dataclass_fields__") else None
            self.signal_direction = self._profile.signal_direction
        self._buffer = deque(maxlen=max(1, int(self.smooth_window)))
        self.state.active_down_threshold = float(self.down_threshold)
        self.state.active_up_threshold = float(self.up_threshold)
        self.state.calibration_complete = True   # always skip old calibration
        self.state.phase = "start"

    def reset(self) -> None:
        self._buffer.clear()
        self.state = RepState()
        self._calibration_values.clear()
        self._phase_start_frame = -1
        self._baseline = None
        self._baseline_frames_seen = 0
        self._active_extreme = None
        self._reset_cycle_metrics()
        self.state.active_down_threshold = float(self.down_threshold)
        self.state.active_up_threshold = float(self.up_threshold)
        self.state.calibration_complete = True
        self.state.phase = "start"

    # ── Cycle helpers ─────────────────────────────────────────────────────

    def _reset_cycle_metrics(self) -> None:
        self._cycle_side_gap_sum = 0.0
        self._cycle_side_gap_count = 0
        self._cycle_signal_min = None
        self._cycle_signal_max = None
        self._cycle_visibility_sum = 0.0
        self._cycle_visibility_count = 0
        self._cycle_roughness_sum = 0.0
        self._cycle_last_smooth = None
        self._active_extreme = None

    def _set_phase(self, phase: str, frame_id: int) -> None:
        self.state.phase = phase
        self._phase_start_frame = frame_id
        if phase == "down":
            self._reset_cycle_metrics()

    # ── Side-gap helper ───────────────────────────────────────────────────

    def _exercise_side_gap(self, angles: Dict[str, Optional[float]]) -> Optional[float]:
        if self._profile is None:
            return None
        left_name = self._profile.side_gap_left_angle
        right_name = self._profile.side_gap_right_angle
        if left_name is None or right_name is None:
            return None
        left = angles.get(left_name)
        right = angles.get(right_name)
        if left is None or right is None:
            return None
        return abs(float(left) - float(right))

    def _relevant_visibility_ratio(self, visible: Dict[str, bool]) -> float:
        joints = self._profile.visibility_joints if self._profile else list(visible.keys())
        if not joints:
            return 0.0
        seen = sum(1 for j in joints if visible.get(j, False))
        return float(seen / len(joints))

    # ── Per-frame cycle metric accumulator ────────────────────────────────

    def _update_cycle_metrics(
        self,
        smooth_signal: float,
        visibility_ratio: float,
        side_gap: Optional[float],
    ) -> None:
        if self._cycle_signal_min is None:
            self._cycle_signal_min = smooth_signal
            self._cycle_signal_max = smooth_signal
        else:
            self._cycle_signal_min = min(self._cycle_signal_min, smooth_signal)
            self._cycle_signal_max = max(self._cycle_signal_max, smooth_signal)

        if self._cycle_last_smooth is not None:
            self._cycle_roughness_sum += abs(smooth_signal - self._cycle_last_smooth)
        self._cycle_last_smooth = smooth_signal

        self._cycle_visibility_sum += visibility_ratio
        self._cycle_visibility_count += 1

        if side_gap is not None:
            self._cycle_side_gap_sum += side_gap
            self._cycle_side_gap_count += 1

    # ── Rep confidence ────────────────────────────────────────────────────

    def _compute_rep_confidence(self) -> float:
        if self._cycle_signal_min is None or self._cycle_signal_max is None:
            return 0.0

        amplitude = self._cycle_signal_max - self._cycle_signal_min
        # Expected total ROM ≈ entry_threshold + return_threshold
        expected_rom = max(self.down_threshold + self.up_threshold, 1e-6)
        amplitude_score = float(np.clip(amplitude / expected_rom, 0.0, 1.0))

        steps = max(1, self._cycle_visibility_count - 1)
        smooth_den = max(1e-6, amplitude * steps)
        smoothness = float(np.clip(1.0 - (self._cycle_roughness_sum / smooth_den), 0.0, 1.0))

        visibility_score = 0.0
        if self._cycle_visibility_count > 0:
            visibility_score = float(np.clip(
                self._cycle_visibility_sum / self._cycle_visibility_count,
                0.0, 1.0,
            ))

        confidence = (0.45 * amplitude_score) + (0.30 * smoothness) + (0.25 * visibility_score)
        return float(np.clip(confidence, 0.0, 1.0))

    # ── Main update ───────────────────────────────────────────────────────

    def update_from_landmarks(
        self,
        landmarks_norm: Dict[str, Dict],
        visible: Dict[str, bool],
        frame_id: int,
    ) -> RepState:
        """
        Update rep count from normalized landmarks for one frame.

        Uses amplitude-hysteresis relative to each person's own rest position
        so fixed angles never need to be hard-coded.
        """
        angles = compute_named_angles(landmarks_norm, visible)
        signal = exercise_signal_angle(self.exercise_name, angles)
        side_gap = self._exercise_side_gap(angles)

        self.state.last_signal = signal
        if signal is None:
            return self.state

        signal_value = float(signal)
        self._buffer.append(signal_value)
        smooth_signal = float(np.mean(self._buffer))
        self.state.smoothed_signal = smooth_signal
        self.state.valid_signal_frames += 1

        # ── Step 1: Adaptive baseline (person's own rest position) ────────
        # min_first (squat / pushup): rest = high angle  → baseline tracks peak
        # max_first (shoulder_press): rest = low angle   → baseline tracks trough
        if self._baseline is None:
            self._baseline = smooth_signal
            self._baseline_frames_seen = 1
            return self.state

        self._baseline_frames_seen += 1

        if self.signal_direction == "min_first":
            if smooth_signal > self._baseline:
                # Fast pull towards new high (person standing up)
                self._baseline = 0.85 * self._baseline + 0.15 * smooth_signal
            else:
                # Very slow downward drift – don't chase the rep
                self._baseline = 0.998 * self._baseline + 0.002 * smooth_signal
        else:  # max_first
            if smooth_signal < self._baseline:
                # Fast pull towards new low (bar at rest)
                self._baseline = 0.85 * self._baseline + 0.15 * smooth_signal
            else:
                # Very slow upward drift
                self._baseline = 0.998 * self._baseline + 0.002 * smooth_signal

        # Give baseline 15 frames to stabilise before counting reps
        if self._baseline_frames_seen < 15:
            return self.state

        visibility_ratio = self._relevant_visibility_ratio(visible)

        # Deviation = how far signal has moved away from rest
        if self.signal_direction == "min_first":
            deviation = self._baseline - smooth_signal   # positive = bending
        else:
            deviation = smooth_signal - self._baseline   # positive = extending

        # ── Step 2: State machine ─────────────────────────────────────────

        if self.state.phase in {"start", "up"}:
            # Enter active (down) phase once deviation is large enough
            if deviation >= self.down_threshold:
                self._set_phase("down", frame_id)
                self._active_extreme = smooth_signal
                self._update_cycle_metrics(smooth_signal, visibility_ratio, side_gap)

        elif self.state.phase == "down":
            # Track the extreme (deepest bend / highest extension)
            if self._active_extreme is None:
                self._active_extreme = smooth_signal
            elif self.signal_direction == "min_first":
                self._active_extreme = min(self._active_extreme, smooth_signal)
            else:
                self._active_extreme = max(self._active_extreme, smooth_signal)

            self._update_cycle_metrics(smooth_signal, visibility_ratio, side_gap)

            # Measure return from extreme
            if self.signal_direction == "min_first":
                return_amount = smooth_signal - self._active_extreme
            else:
                return_amount = self._active_extreme - smooth_signal

            if (
                return_amount >= self.up_threshold
                and (frame_id - self.state.last_rep_frame) >= self.min_frames_between_reps
            ):
                # ── Form quality gate ─────────────────────────────────────
                # Use AVERAGE asymmetry over the whole rep (not worst single frame)
                avg_side_gap = (
                    self._cycle_side_gap_sum / self._cycle_side_gap_count
                    if self._cycle_side_gap_count > 0 else 0.0
                )
                good_form = avg_side_gap <= self.max_side_gap_deg

                self.state.last_rep_frame = frame_id
                self.state.last_rep_confidence = self._compute_rep_confidence()
                self.state.rep_confidences.append(self.state.last_rep_confidence)

                if good_form:
                    self.state.rep_count += 1
                else:
                    self.state.rejected_reps += 1

                self._set_phase("up", frame_id)
                self._reset_cycle_metrics()

        return self.state

    def update_from_skeleton(self, skeleton) -> RepState:
        """
        Convenience adapter for M2 SkeletonFrame-like objects.
        Requires: .landmarks_norm, .visible, .frame_id, .pose_detected
        """
        if not getattr(skeleton, "pose_detected", False):
            return self.state

        return self.update_from_landmarks(
            landmarks_norm=skeleton.landmarks_norm,
            visible=skeleton.visible,
            frame_id=skeleton.frame_id,
        )


def default_rep_counter(exercise_name: str) -> RepCounter:
    """Factory – creates a RepCounter with per-exercise defaults."""
    profile = get_exercise_profile(exercise_name)
    if profile is None:
        return RepCounter(
            exercise_name=exercise_name,
            down_threshold=30.0,
            up_threshold=40.0,
            smooth_window=10,
            min_frames_between_reps=12,
            max_side_gap_deg=35.0,
            signal_direction="min_first",
        )

    return RepCounter(
        exercise_name=exercise_name,
        down_threshold=profile.down_threshold,
        up_threshold=profile.up_threshold,
        smooth_window=profile.smooth_window,
        min_frames_between_reps=profile.min_frames_between_reps,
        dynamic_calibration=profile.dynamic_calibration,
        calibration_frames=profile.calibration_frames,
        max_side_gap_deg=profile.max_side_gap_deg,
        phase_timeout_frames=profile.phase_timeout_frames,
        signal_direction=profile.signal_direction,
    )
