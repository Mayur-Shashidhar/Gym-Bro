import sys
import unittest
from types import SimpleNamespace

from m3_api import (
    RepCounter,
    angle_at_joint_deg,
    compute_named_angles,
    default_rep_counter,
    exercise_signal_angle,
    list_supported_exercises,
)


def _lm(x, y, z=0.0):
    return {"x": float(x), "y": float(y), "z": float(z)}


def _all_visible(landmarks):
    return {k: True for k in landmarks.keys()}


def _squat_landmarks(knee_angle_deg: float, right_knee_angle_deg: float = None):
    """
    Build simple left/right lower body landmarks whose knee angle is controlled.
    Hip and knee are fixed; ankle x-offset controls angle at knee.
    """
    import math

    # Geometry around left knee at (0,0), hip at (0,1)
    # For angle theta at knee between vectors to hip and ankle:
    # ankle = (sin(theta), cos(theta))
    right_angle = knee_angle_deg if right_knee_angle_deg is None else right_knee_angle_deg

    t_left = math.radians(knee_angle_deg)
    ax_left = math.sin(t_left)
    ay_left = math.cos(t_left)

    t_right = math.radians(right_angle)
    ax_right = math.sin(t_right)
    ay_right = math.cos(t_right)

    landmarks = {
        "LEFT_HIP": _lm(0.0, 1.0),
        "LEFT_KNEE": _lm(0.0, 0.0),
        "LEFT_ANKLE": _lm(ax_left, ay_left),
        "RIGHT_HIP": _lm(1.0, 1.0),
        "RIGHT_KNEE": _lm(1.0, 0.0),
        "RIGHT_ANKLE": _lm(1.0 + ax_right, ay_right),
        "LEFT_SHOULDER": _lm(0.0, 2.0),
        "RIGHT_SHOULDER": _lm(1.0, 2.0),
        "LEFT_ELBOW": _lm(0.0, 2.5),
        "RIGHT_ELBOW": _lm(1.0, 2.5),
        "LEFT_WRIST": _lm(0.0, 3.0),
        "RIGHT_WRIST": _lm(1.0, 3.0),
    }
    return landmarks


class TestVectorAngle(unittest.TestCase):
    def test_angle_at_joint_90(self):
        landmarks = {
            "A": _lm(1.0, 0.0),
            "B": _lm(0.0, 0.0),
            "C": _lm(0.0, 1.0),
        }
        angle = angle_at_joint_deg(landmarks, "A", "B", "C")
        self.assertIsNotNone(angle)
        assert angle is not None
        self.assertAlmostEqual(angle, 90.0, places=4)


class TestAngles(unittest.TestCase):
    def test_compute_named_angles_knees(self):
        landmarks = _squat_landmarks(100.0)
        visible = _all_visible(landmarks)
        angles = compute_named_angles(landmarks, visible)

        self.assertIsNotNone(angles["left_knee"])
        self.assertIsNotNone(angles["right_knee"])
        left_knee = angles["left_knee"]
        right_knee = angles["right_knee"]
        assert left_knee is not None and right_knee is not None
        self.assertAlmostEqual(left_knee, 100.0, places=1)
        self.assertAlmostEqual(right_knee, 100.0, places=1)

    def test_exercise_alias_signal_resolution(self):
        landmarks = _squat_landmarks(100.0)
        visible = _all_visible(landmarks)
        angles = compute_named_angles(landmarks, visible)
        signal = exercise_signal_angle("air_squat", angles)

        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertAlmostEqual(signal, 100.0, places=1)

    def test_supported_exercise_list(self):
        supported = list_supported_exercises()
        self.assertIn("squat", supported)
        self.assertIn("pushup", supported)
        self.assertIn("shoulder_press", supported)


class TestRepCounter(unittest.TestCase):
    @staticmethod
    def _make_squat_counter_no_calibration(**kwargs):
        down_threshold = kwargs.pop("down_threshold", 95.0)
        up_threshold = kwargs.pop("up_threshold", 155.0)
        smooth_window = kwargs.pop("smooth_window", 5)
        min_frames_between_reps = kwargs.pop("min_frames_between_reps", 8)
        return RepCounter(
            exercise_name="squat",
            down_threshold=down_threshold,
            up_threshold=up_threshold,
            smooth_window=smooth_window,
            min_frames_between_reps=min_frames_between_reps,
            dynamic_calibration=False,
            **kwargs,
        )

    def test_squat_rep_counting(self):
        counter = self._make_squat_counter_no_calibration()

        # One clean rep pattern: high -> low -> high
        sequence = [165, 160, 145, 120, 95, 85, 95, 120, 145, 160, 168]

        state = counter.state
        for frame_id, knee_angle in enumerate(sequence):
            landmarks = _squat_landmarks(knee_angle)
            visible = _all_visible(landmarks)
            state = counter.update_from_landmarks(landmarks, visible, frame_id)

        self.assertEqual(state.rep_count, 1)

    def test_rep_confidence_is_recorded(self):
        counter = self._make_squat_counter_no_calibration()
        sequence = [165, 160, 145, 120, 95, 85, 95, 120, 145, 160, 168]

        state = counter.state
        for frame_id, knee_angle in enumerate(sequence):
            landmarks = _squat_landmarks(knee_angle)
            visible = _all_visible(landmarks)
            state = counter.update_from_landmarks(landmarks, visible, frame_id)

        self.assertEqual(state.rep_count, 1)
        self.assertGreater(state.last_rep_confidence, 0.0)
        self.assertLessEqual(state.last_rep_confidence, 1.0)
        self.assertEqual(len(state.rep_confidences), 1)

    def test_side_inconsistency_rejects_rep(self):
        counter = self._make_squat_counter_no_calibration(max_side_gap_deg=8.0)
        left_sequence = [165, 150, 120, 95, 85, 95, 120, 150, 165]

        state = counter.state
        for frame_id, left_angle in enumerate(left_sequence):
            landmarks = _squat_landmarks(left_angle, right_knee_angle_deg=165.0)
            visible = _all_visible(landmarks)
            state = counter.update_from_landmarks(landmarks, visible, frame_id)

        self.assertEqual(state.rep_count, 0)
        self.assertGreaterEqual(state.rejected_reps, 1)

    def test_timeout_resets_down_phase(self):
        counter = self._make_squat_counter_no_calibration(phase_timeout_frames=1, smooth_window=1)

        seq = [165, 90, 90, 90]
        state = counter.state
        for frame_id, knee_angle in enumerate(seq):
            landmarks = _squat_landmarks(knee_angle)
            visible = _all_visible(landmarks)
            state = counter.update_from_landmarks(landmarks, visible, frame_id)

        self.assertEqual(state.phase, "start")
        self.assertEqual(state.rep_count, 0)

    def test_dynamic_calibration_updates_thresholds(self):
        counter = default_rep_counter("squat")

        state = counter.state
        for frame_id in range(counter.calibration_frames + 5):
            knee_angle = 85.0 + (frame_id % 80)
            landmarks = _squat_landmarks(knee_angle)
            visible = _all_visible(landmarks)
            state = counter.update_from_landmarks(landmarks, visible, frame_id)

        self.assertTrue(state.calibration_complete)
        self.assertNotEqual(counter.down_threshold, 95.0)
        self.assertNotEqual(counter.up_threshold, 155.0)

    def test_no_pose_keeps_count(self):
        counter = self._make_squat_counter_no_calibration()
        fake = SimpleNamespace(
            pose_detected=False,
            landmarks_norm={},
            visible={},
            frame_id=0,
        )
        state = counter.update_from_skeleton(fake)
        self.assertEqual(state.rep_count, 0)


if __name__ == "__main__":
    print("Running M3 unit tests...\n")
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
