import time
import urllib.request
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional, List

import cv2
import mediapipe as mp
from mediapipe.tasks.python.core        import base_options as mp_base
from mediapipe.tasks.python.vision      import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarksConnections,
)
from mediapipe.tasks.python.vision.core import vision_task_running_mode as rm_module

from landmark_utils import (
    LANDMARK_NAMES,
    EXERCISE_JOINTS,
    build_visibility_mask,
    normalize_landmarks,
    to_feature_vector,
    VISIBILITY_THRESHOLD,
)

# ── Model auto-download ───────────────────────────────────────────────────────

_MODEL_URLS = {
    0: ("pose_landmarker_lite.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"),
    1: ("pose_landmarker_full.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task"),
    2: ("pose_landmarker_heavy.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"),
}

_MODEL_CACHE_DIR = Path(__file__).parent / ".mp_models"


def _ensure_model(complexity: int) -> str:
    _MODEL_CACHE_DIR.mkdir(exist_ok=True)
    filename, url = _MODEL_URLS[complexity]
    dest = _MODEL_CACHE_DIR / filename
    if not dest.exists():
        print(f"[M2] Downloading '{filename}' (~30 MB, first run only)…")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"[M2] Saved to {dest}")
        except Exception as e:
            raise RuntimeError(
                f"[M2] Download failed: {e}\n"
                f"Manual fix: download the file and place it at {dest}"
            )
    return str(dest)


# ── SkeletonFrame ─────────────────────────────────────────────────────────────

@dataclass
class SkeletonFrame:
    """
    One processed frame. Contract between M2 and every other module:
      M3  → landmarks_norm  (angles)
      M4  → landmarks_norm + visible  (form rules)
      M1  → feature_vector  (classification)
      M6  → to_dict()  (dataset)
      M5  → annotated_frame  (display)
    """
    frame_id:       int   = 0
    timestamp:      float = 0.0
    pose_detected:  bool  = False

    landmarks_raw:   Dict[str, Dict] = field(default_factory=dict)
    landmarks_world: Dict[str, Dict] = field(default_factory=dict)
    landmarks_norm:  Dict[str, Dict] = field(default_factory=dict)
    visible:         Dict[str, bool] = field(default_factory=dict)
    feature_vector:  List[float]     = field(default_factory=list)

    annotated_frame: Optional[object] = None  # numpy ndarray

    def to_dict(self) -> dict:
        return {
            "frame_id":       self.frame_id,
            "timestamp":      self.timestamp,
            "pose_detected":  self.pose_detected,
            "landmarks_raw":  self.landmarks_raw,
            "landmarks_norm": self.landmarks_norm,
            "visible":        self.visible,
            "feature_vector": self.feature_vector,
        }


# ── SkeletonTracker ───────────────────────────────────────────────────────────

class SkeletonTracker:
    """
    Wraps MediaPipe PoseLandmarker (Tasks API, mediapipe>=0.10).

    Usage
    -----
    tracker = SkeletonTracker()
    frame   = tracker.process_frame(bgr_frame)
    tracker.release()

    with SkeletonTracker() as tracker:
        frame = tracker.process_frame(bgr)
    """

    def __init__(
        self,
        model_complexity:         int   = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence:  float = 0.5,
        visibility_threshold:     float = VISIBILITY_THRESHOLD,
        draw_skeleton:            bool  = True,
    ):
        """
        model_complexity: 0=lite (fastest), 1=full (default), 2=heavy (best accuracy)
        """
        self.visibility_threshold = visibility_threshold
        self.draw_skeleton_flag   = draw_skeleton
        self._frame_id            = 0
        self._start_time: Optional[float] = None

        model_path = _ensure_model(model_complexity)

        options = PoseLandmarkerOptions(
            base_options=mp_base.BaseOptions(model_asset_path=model_path),
            running_mode=rm_module.VisionTaskRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)

        # Build connection name-pairs once from MediaPipe's own connection list
        self._connections = [
            (LANDMARK_NAMES[c.start], LANDMARK_NAMES[c.end])
            for c in PoseLandmarksConnections.POSE_LANDMARKS
            if c.start < len(LANDMARK_NAMES) and c.end < len(LANDMARK_NAMES)
        ]

    # ── Main entry point ──────────────────────────────────────────────────────

    def process_frame(self, bgr_frame) -> SkeletonFrame:
        """
        Process one BGR OpenCV frame → SkeletonFrame.
        Only method other modules need to call.
        """
        if self._start_time is None:
            self._start_time = time.time()

        timestamp    = time.time() - self._start_time
        frame_id     = self._frame_id
        self._frame_id += 1
        timestamp_ms = int(timestamp * 1000)

        rgb  = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_img, timestamp_ms)

        skeleton = SkeletonFrame(frame_id=frame_id, timestamp=timestamp)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            skeleton.pose_detected = True
            raw   = self._extract_raw(result)
            world = self._extract_world(result)
            skeleton.landmarks_raw   = raw
            skeleton.landmarks_world = world
            skeleton.visible         = build_visibility_mask(raw, self.visibility_threshold)
            skeleton.landmarks_norm  = normalize_landmarks(raw, skeleton.visible)
            skeleton.feature_vector  = to_feature_vector(
                skeleton.landmarks_norm, skeleton.visible, joints=EXERCISE_JOINTS
            )
        else:
            skeleton.pose_detected  = False
            skeleton.feature_vector = [0.0] * (len(EXERCISE_JOINTS) * 3)

        annotated = bgr_frame.copy()
        if self.draw_skeleton_flag and skeleton.pose_detected:
            self._draw(annotated, skeleton)
        skeleton.annotated_frame = annotated

        return skeleton

    # ── Extraction ────────────────────────────────────────────────────────────

    def _extract_raw(self, result) -> Dict[str, Dict]:
        lms = result.pose_landmarks[0]
        return {
            LANDMARK_NAMES[i]: {
                "x":          float(lm.x),
                "y":          float(lm.y),
                "z":          float(lm.z),
                "visibility": float(lm.visibility or 1.0),
            }
            for i, lm in enumerate(lms) if i < len(LANDMARK_NAMES)
        }

    def _extract_world(self, result) -> Dict[str, Dict]:
        if not result.pose_world_landmarks or len(result.pose_world_landmarks) == 0:
            return {}
        lms = result.pose_world_landmarks[0]
        return {
            LANDMARK_NAMES[i]: {
                "x": float(lm.x), "y": float(lm.y), "z": float(lm.z),
                "visibility": float(lm.visibility or 1.0),
            }
            for i, lm in enumerate(lms) if i < len(LANDMARK_NAMES)
        }

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw(self, bgr_frame, skeleton: SkeletonFrame) -> None:
        h, w = bgr_frame.shape[:2]
        for (a, b) in self._connections:
            if skeleton.visible.get(a) and skeleton.visible.get(b):
                la, lb = skeleton.landmarks_raw[a], skeleton.landmarks_raw[b]
                pa = (int(la["x"] * w), int(la["y"] * h))
                pb = (int(lb["x"] * w), int(lb["y"] * h))
                cv2.line(bgr_frame, pa, pb, (0, 230, 120), 2, cv2.LINE_AA)
        for name, lm in skeleton.landmarks_raw.items():
            if skeleton.visible.get(name):
                px, py = int(lm["x"] * w), int(lm["y"] * h)
                color  = (0, 200, 255) if name in EXERCISE_JOINTS else (160, 160, 160)
                r      = 6 if name in EXERCISE_JOINTS else 4
                cv2.circle(bgr_frame, (px, py), r, color,   -1, cv2.LINE_AA)
                cv2.circle(bgr_frame, (px, py), r, (0,0,0),  1, cv2.LINE_AA)

    def draw_skeleton(self, bgr_frame, skeleton: SkeletonFrame) -> None:
        """Public helper for M5: draw onto any frame."""
        if skeleton.pose_detected:
            self._draw(bgr_frame, skeleton)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset_timer(self) -> None:
        self._frame_id   = 0
        self._start_time = None

    def release(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()