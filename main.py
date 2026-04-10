import sys
from pathlib import Path
from collections import deque
from typing import List, Optional, Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np

from m2.m2_api import SkeletonTracker, stream_webcam
from m3.m3_api import default_rep_counter, compute_named_angles
from m4.movement_quality_scorer import score_quality
from m1.classifier import predict_exercise_from_frame, detect_fatigue

# ── Constants ──────────────────────────────────────────────────────────────────
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Map classifier labels → movement_quality_scorer keys
EXERCISE_KEY_MAP = {
    "squat":          "squat",
    "pushup":         "push_up",
    "shoulder_press": "shoulder_press",
}

# Display labels for UI
EXERCISE_LABELS = {
    "squat": "Squat",
    "pushup": "Push Up",
    "shoulder_press": "Shoulder Press",
}

# Features needed by movement_quality_scorer per exercise
QUALITY_FEATURE_MAP = {
    "squat":          ("knee_angle",       "left_knee"),
    "push_up":        ("elbow_angle",      "left_elbow"),
    "shoulder_press": ("shoulder_angle",   "left_shoulder"),
}


def _build_quality_features(angles: dict, exercise_key: str) -> dict:
    """
    Build the feature dict that movement_quality_scorer.score_quality() expects.
    It needs: joint angle + symmetry_index + stability + smoothness.
    We compute what we can from M3 angles; stability/smoothness default to 0.03.
    """
    features = {}

    if exercise_key == "squat":
        lk = angles.get("left_knee") or 0.0
        rk = angles.get("right_knee") or 0.0
        features["knee_angle"] = min(lk, rk) if lk and rk else (lk or rk)
        avg = (lk + rk) / 2.0 if lk and rk else 1.0
        features["symmetry_index"] = abs(lk - rk) / avg if avg > 0 else 0.0

    elif exercise_key == "push_up":
        le = angles.get("left_elbow") or 0.0
        re = angles.get("right_elbow") or 0.0
        features["elbow_angle"] = min(le, re) if le and re else (le or re)
        avg = (le + re) / 2.0 if le and re else 1.0
        features["symmetry_index"] = abs(le - re) / avg if avg > 0 else 0.0

    elif exercise_key == "shoulder_press":
        le = angles.get("left_elbow") or 0.0
        re = angles.get("right_elbow") or 0.0
        features["elbow_angle"] = min(le, re) if le and re else (le or re)
        avg = (le + re) / 2.0 if le and re else 1.0
        features["symmetry_index"] = abs(le - re) / avg if avg > 0 else 0.0

    # Stability and smoothness — fixed reasonable defaults (live scoring)
    features["stability"]  = 0.03
    features["smoothness"] = 0.10
    return features, exercise_key


class GymBroPipeline:
    """Reusable pipeline for Streamlit and CLI use."""

    def __init__(self, vote_window: int = 30):
        self.current_exercise: Optional[str] = None
        self.rep_counter = None
        self.rep_scores: List[int] = []
        self.last_rep_count = 0
        self.quality_report: Dict[str, Any] = {
            "overall": 0,
            "feedback_text": "Stand in frame to begin",
            "rom": 0,
            "symmetry": 0,
            "stability": 0,
            "smoothness": 0,
        }
        self.fatigue_level = "low"
        self.exercise_votes: deque = deque(maxlen=vote_window)
        self.last_state = None

    def update(self, skeleton, models_ready: bool = True) -> Dict[str, Any]:
        """
        Run one pipeline step from an M2 SkeletonFrame.
        Returns UI-ready data.
        """
        if getattr(skeleton, "pose_detected", False) and models_ready:
            label = predict_exercise_from_frame(skeleton)
            if label != "unknown":
                self.exercise_votes.append(label)

            voted = (
                max(set(self.exercise_votes), key=list(self.exercise_votes).count)
                if self.exercise_votes else None
            )

            if voted and voted != self.current_exercise:
                self.current_exercise = voted
                self.rep_counter = default_rep_counter(voted)
                self.last_state = self.rep_counter.state
                self.rep_scores = []
                self.last_rep_count = 0

            if self.rep_counter is not None:
                state = self.rep_counter.update_from_skeleton(skeleton)
                self.last_state = state

                if state.rep_count > self.last_rep_count:
                    self.last_rep_count = state.rep_count

                    angles = compute_named_angles(
                        skeleton.landmarks_norm, skeleton.visible
                    )
                    ex_key = EXERCISE_KEY_MAP.get(self.current_exercise, "squat")
                    features, ex_key_used = _build_quality_features(angles, ex_key)
                    self.quality_report = score_quality(features, ex_key_used)
                    self.rep_scores.append(self.quality_report["overall"])
                    self.fatigue_level = detect_fatigue(self.rep_scores)

        display_name = EXERCISE_LABELS.get(self.current_exercise, "Detecting")
        rep_count = self.last_state.rep_count if self.last_state is not None else 0

        return {
            "exercise_label": display_name,
            "rep_count": rep_count,
            "quality_report": self.quality_report,
            "fatigue_level": self.fatigue_level,
            "rep_scores": self.rep_scores,
        }


def _draw_hud(frame, exercise, state, quality, fatigue):
    h, w = frame.shape[:2]

    # Dark panel on right
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - 270, 0), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    x, y = w - 255, 35

    def put(text, py, scale=0.52, color=(210, 210, 210), bold=False):
        t = 2 if bold else 1
        cv2.putText(frame, text, (x, py), FONT, scale, (0, 0, 0), t + 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x, py), FONT, scale, color, t, cv2.LINE_AA)

    # Exercise name
    ex_disp = (exercise or "Detecting...").replace("_", " ").title()
    put("Exercise", y, 0.42, (130, 130, 130))
    put(ex_disp, y + 24, 0.68, (0, 230, 100), bold=True)

    # Phase / calibration
    phase = state.phase if state else "—"
    if phase == "calibrating":
        pct = min(100, int((state.calibration_frames_seen / 75) * 100))
        put(f"Calibrating... {pct}%", y + 58, 0.46, (40, 200, 255))
    else:
        put(f"Phase: {phase}", y + 58, 0.46, (170, 170, 170))

    # Rep count
    reps = state.rep_count if state else 0
    put("Reps", y + 92, 0.42, (130, 130, 130))
    put(str(reps), y + 120, 1.1, (255, 200, 0), bold=True)

    # Overall quality score
    score = quality.get("overall", 0)
    s_color = (80, 220, 80) if score >= 80 else (40, 210, 210) if score >= 60 else (60, 60, 220)
    put("Quality", y + 165, 0.42, (130, 130, 130))
    put(f"{score}/100", y + 190, 0.82, s_color, bold=True)

    # Sub-scores
    put(f"ROM:        {quality.get('rom', 0)}", y + 220, 0.42)
    put(f"Symmetry:   {quality.get('symmetry', 0)}", y + 238, 0.42)
    put(f"Stability:  {quality.get('stability', 0)}", y + 256, 0.42)
    put(f"Smoothness: {quality.get('smoothness', 0)}", y + 274, 0.42)

    # Fatigue
    f_color = (80, 220, 80) if fatigue == "low" else (40, 200, 255) if fatigue == "medium" else (60, 60, 220)
    put("Fatigue", y + 308, 0.42, (130, 130, 130))
    put(fatigue.upper(), y + 330, 0.62, f_color, bold=True)

    # Feedback
    feedback = quality.get("feedback_text", "")
    if feedback:
        # Word-wrap to ~28 chars
        words = feedback.split()
        line, lines = "", []
        for w_word in words:
            if len(line) + len(w_word) + 1 <= 28:
                line = (line + " " + w_word).strip()
            else:
                lines.append(line)
                line = w_word
        if line:
            lines.append(line)
        put("Feedback:", y + 365, 0.42, (130, 130, 130))
        for i, ln in enumerate(lines[:3]):
            put(ln, y + 383 + i * 16, 0.38, (200, 200, 200))

    # Controls hint at bottom
    put("[q] quit", h - 15, 0.38, (90, 90, 90))


def main():
    print("=== GymBro Exercise Analyzer ===")
    print("Loading models...")

    # Pre-load classifier model
    from m1.classifier import _load_models
    try:
        _load_models()
        print("  Classifier loaded.")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("  Run: python m6/extract_dataset.py --augment")
        print("  Then: python m1/classifier.py")
        return

    tracker = SkeletonTracker(model_complexity=1, draw_skeleton=True)
    print("  MediaPipe loaded.")
    print("\nStarting webcam... press q to quit.\n")

    # State
    current_exercise: Optional[str] = None
    rep_counter = None
    rep_scores: List[int] = []
    last_rep_count = 0
    quality_report = {
        "overall": 0, "feedback_text": "Stand in frame to begin",
        "rom": 0, "symmetry": 0, "stability": 0, "smoothness": 0
    }
    fatigue_level = "low"
    exercise_votes: deque = deque(maxlen=30)

    import time
    fps_buf: deque = deque(maxlen=20)
    last_t = time.time()

    for skeleton in stream_webcam(tracker=tracker, show_preview=False):

        # ── FPS ──────────────────────────────────────────────────────────────
        now = time.time()
        fps_buf.append(1.0 / max(now - last_t, 1e-6))
        last_t = now
        fps = sum(fps_buf) / len(fps_buf)

        frame = skeleton.annotated_frame.copy()

        if skeleton.pose_detected:

            # ── Step 1: Classify exercise (smoothed over 30 frames) ───────────
            label = predict_exercise_from_frame(skeleton)
            if label != "unknown":
                exercise_votes.append(label)

            voted = (
                max(set(exercise_votes), key=list(exercise_votes).count)
                if exercise_votes else None
            )

            # ── Step 2: Reset rep counter if exercise changed ─────────────────
            if voted and voted != current_exercise:
                current_exercise = voted
                rep_counter = default_rep_counter(current_exercise)
                rep_scores = []
                last_rep_count = 0
                print(f"  Exercise detected: {current_exercise}")

            # ── Step 3: Count reps ────────────────────────────────────────────
            state = None
            if rep_counter is not None:
                state = rep_counter.update_from_skeleton(skeleton)

                # New rep completed
                if state.rep_count > last_rep_count:
                    last_rep_count = state.rep_count

                    # ── Step 4: Score quality ─────────────────────────────────
                    angles = compute_named_angles(
                        skeleton.landmarks_norm, skeleton.visible
                    )
                    ex_key = EXERCISE_KEY_MAP.get(current_exercise, "squat")
                    try:
                        features, ex_key_used = _build_quality_features(angles, ex_key)
                        quality_report = score_quality(features, ex_key_used)
                        rep_scores.append(quality_report["overall"])
                        fatigue_level = detect_fatigue(rep_scores)
                        print(
                            f"  Rep {state.rep_count}: "
                            f"score={quality_report['overall']} "
                            f"fatigue={fatigue_level}"
                        )
                    except Exception as e:
                        print(f"  [WARN] Quality score failed: {e}")

            # ── Step 5: Draw HUD ──────────────────────────────────────────────
            _draw_hud(frame, current_exercise, state, quality_report, fatigue_level)

        else:
            # No pose — show waiting message
            cv2.putText(frame, "Stand fully in frame", (20, 40),
                        FONT, 0.7, (0, 80, 220), 2, cv2.LINE_AA)

        # FPS counter
        cv2.putText(frame, f"{fps:.0f} fps", (10, frame.shape[0] - 10),
                    FONT, 0.45, (100, 100, 100), 1, cv2.LINE_AA)

        cv2.imshow("GymBro Exercise Analyzer", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    tracker.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
