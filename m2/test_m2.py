import math
import sys
import unittest


# ── Stubs so we can test landmark_utils without mediapipe installed ─────────
class _FakeLandmark:
    def __init__(self, x, y, z, visibility):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility

class _FakePoseLandmarks:
    def __init__(self, landmarks):
        self.landmark = landmarks

class _FakeResults:
    def __init__(self, landmarks=None):
        self.pose_landmarks       = _FakePoseLandmarks(landmarks) if landmarks else None
        self.pose_world_landmarks = _FakePoseLandmarks(landmarks) if landmarks else None


from landmark_utils import (
    LANDMARK_NAMES,
    EXERCISE_JOINTS,
    extract_raw_landmarks,
    build_visibility_mask,
    normalize_landmarks,
    to_feature_vector,
    VISIBILITY_THRESHOLD,
)


def _make_fake_results(visibility=0.9):
    """33 landmarks positioned in a T-pose for testing."""
    lms = []
    for i, name in enumerate(LANDMARK_NAMES):
        x = 0.5 + 0.01 * i
        y = 0.5 - 0.01 * i
        z = 0.0
        lms.append(_FakeLandmark(x, y, z, visibility))
    return _FakeResults(lms)


class TestExtractRaw(unittest.TestCase):
    def test_returns_none_on_empty(self):
        self.assertIsNone(extract_raw_landmarks(None))
        self.assertIsNone(extract_raw_landmarks(_FakeResults()))

    def test_all_names_present(self):
        raw = extract_raw_landmarks(_make_fake_results())
        self.assertIsNotNone(raw)
        for name in LANDMARK_NAMES:
            self.assertIn(name, raw)

    def test_fields_present(self):
        raw = extract_raw_landmarks(_make_fake_results())
        lm  = raw["LEFT_SHOULDER"]
        for field in ("x", "y", "z", "visibility"):
            self.assertIn(field, lm)

    def test_visibility_preserved(self):
        raw = extract_raw_landmarks(_make_fake_results(visibility=0.3))
        self.assertAlmostEqual(raw["NOSE"]["visibility"], 0.3, places=5)


class TestVisibilityMask(unittest.TestCase):
    def test_high_visibility_is_true(self):
        raw  = extract_raw_landmarks(_make_fake_results(visibility=0.9))
        mask = build_visibility_mask(raw)
        self.assertTrue(all(mask.values()))

    def test_low_visibility_is_false(self):
        raw  = extract_raw_landmarks(_make_fake_results(visibility=0.1))
        mask = build_visibility_mask(raw)
        self.assertFalse(any(mask.values()))

    def test_threshold_boundary(self):
        raw  = extract_raw_landmarks(_make_fake_results(visibility=VISIBILITY_THRESHOLD))
        mask = build_visibility_mask(raw, threshold=VISIBILITY_THRESHOLD)
        self.assertTrue(all(mask.values()))


class TestNormalize(unittest.TestCase):
    def _run(self, visibility=0.9):
        raw  = extract_raw_landmarks(_make_fake_results(visibility=visibility))
        mask = build_visibility_mask(raw)
        norm = normalize_landmarks(raw, mask)
        return norm

    def test_keys_match_raw(self):
        raw  = extract_raw_landmarks(_make_fake_results())
        mask = build_visibility_mask(raw)
        norm = normalize_landmarks(raw, mask)
        self.assertEqual(set(norm.keys()), set(raw.keys()))

    def test_hip_midpoint_near_zero(self):
        norm = self._run()
        # hip midpoint in normalized space should be ~0
        lh = norm["LEFT_HIP"]
        rh = norm["RIGHT_HIP"]
        mx = (lh["x"] + rh["x"]) / 2
        my = (lh["y"] + rh["y"]) / 2
        self.assertAlmostEqual(mx, 0.0, places=5)
        self.assertAlmostEqual(my, 0.0, places=5)

    def test_torso_height_approx_one(self):
        norm = self._run()
        lh = norm["LEFT_HIP"]
        rh = norm["RIGHT_HIP"]
        ls = norm["LEFT_SHOULDER"]
        rs = norm["RIGHT_SHOULDER"]
        hip_mx = (lh["x"] + rh["x"]) / 2
        hip_my = (lh["y"] + rh["y"]) / 2
        hip_mz = (lh["z"] + rh["z"]) / 2
        sh_mx  = (ls["x"] + rs["x"]) / 2
        sh_my  = (ls["y"] + rs["y"]) / 2
        sh_mz  = (ls["z"] + rs["z"]) / 2
        torso  = math.sqrt(
            (sh_mx - hip_mx) ** 2 +
            (sh_my - hip_my) ** 2 +
            (sh_mz - hip_mz) ** 2
        )
        self.assertAlmostEqual(torso, 1.0, places=4)

    def test_no_visibility_gives_zero_origin(self):
        raw  = extract_raw_landmarks(_make_fake_results(visibility=0.0))
        mask = build_visibility_mask(raw)
        # All invisible → origin at 0, scale at 1 → raw coords returned as-is
        norm = normalize_landmarks(raw, mask)
        self.assertIn("NOSE", norm)


class TestFeatureVector(unittest.TestCase):
    def test_length(self):
        raw  = extract_raw_landmarks(_make_fake_results())
        mask = build_visibility_mask(raw)
        norm = normalize_landmarks(raw, mask)
        vec  = to_feature_vector(norm, mask, joints=EXERCISE_JOINTS)
        self.assertEqual(len(vec), len(EXERCISE_JOINTS) * 3)

    def test_invisible_joints_are_zero(self):
        raw  = extract_raw_landmarks(_make_fake_results(visibility=0.0))
        mask = build_visibility_mask(raw)
        norm = normalize_landmarks(raw, mask)
        vec  = to_feature_vector(norm, mask, joints=EXERCISE_JOINTS)
        self.assertTrue(all(v == 0.0 for v in vec))

    def test_all_floats(self):
        raw  = extract_raw_landmarks(_make_fake_results())
        mask = build_visibility_mask(raw)
        norm = normalize_landmarks(raw, mask)
        vec  = to_feature_vector(norm, mask, joints=EXERCISE_JOINTS)
        self.assertTrue(all(isinstance(v, float) for v in vec))


if __name__ == "__main__":
    print("Running M2 unit tests (no camera/mediapipe model required)...\n")
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
