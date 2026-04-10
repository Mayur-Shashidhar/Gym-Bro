import time
from typing import Iterator, Optional, Union

import cv2

from skeleton_tracker import SkeletonFrame, SkeletonTracker


def stream_webcam(
    camera_index: int = 0,
    width: int = 640,
    height: int = 480,
    fps_cap: Optional[float] = 30.0,
    tracker: Optional[SkeletonTracker] = None,
    show_preview: bool = False,
) -> Iterator[SkeletonFrame]:
    """
    Open the webcam and yield one SkeletonFrame per captured frame.

    Parameters
    ----------
    camera_index
        OpenCV camera index (0 = default webcam).
    width / height
        Requested capture resolution. Camera may use a different actual size.
    fps_cap
        Soft FPS cap — sleeps to avoid hammering the CPU. None = uncapped.
    tracker
        Pass an existing SkeletonTracker to reuse its MediaPipe session.
        If None, a new one is created (and released on StopIteration).
    show_preview
        Show an OpenCV window with the annotated frame (useful for debugging).
        Press 'q' to stop.

    Yields
    ------
    SkeletonFrame
        One per captured frame. Check .pose_detected before reading landmarks.
    """
    owns_tracker = tracker is None
    if tracker is None:
        tracker = SkeletonTracker()

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        if owns_tracker:
            tracker.release()
        raise RuntimeError(f"Could not open camera index {camera_index}.")

    frame_interval = (1.0 / fps_cap) if fps_cap else 0.0
    last_time      = 0.0

    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break

            # FPS cap
            now = time.time()
            elapsed = now - last_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            last_time = time.time()

            skeleton = tracker.process_frame(bgr)

            if show_preview and skeleton.annotated_frame is not None:
                _draw_hud(skeleton.annotated_frame, skeleton)
                cv2.imshow("M2 Skeleton Preview", skeleton.annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            yield skeleton

    finally:
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()
        if owns_tracker:
            tracker.release()


def stream_video(
    video_path: str,
    tracker: Optional[SkeletonTracker] = None,
    realtime: bool = False,
) -> Iterator[SkeletonFrame]:
    """
    Process a video file and yield SkeletonFrames (used by M6 for dataset prep).

    Parameters
    ----------
    video_path
        Path to the video file (.mp4, .avi, etc.).
    tracker
        Pass an existing SkeletonTracker to reuse its MediaPipe session.
    realtime
        If True, sleep between frames to match the video's native FPS.
        If False, process as fast as possible (better for M6 batch runs).

    Yields
    ------
    SkeletonFrame
        One per video frame.
    """
    owns_tracker = tracker is None
    if tracker is None:
        tracker = SkeletonTracker()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if owns_tracker:
            tracker.release()
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = (1.0 / native_fps) if realtime else 0.0
    last_time      = time.time()

    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break

            if realtime:
                now = time.time()
                elapsed = now - last_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                last_time = time.time()

            yield tracker.process_frame(bgr)

    finally:
        cap.release()
        if owns_tracker:
            tracker.release()


# ── HUD overlay ────────────────────────────────────────────────────────────────

def _draw_hud(bgr_frame, skeleton: SkeletonFrame) -> None:
    """Draw a minimal HUD onto the frame for preview/debug mode."""
    import cv2 as _cv2

    color_ok   = (0, 230, 80)
    color_warn = (0, 80, 230)
    font       = _cv2.FONT_HERSHEY_SIMPLEX

    status  = "POSE DETECTED" if skeleton.pose_detected else "NO POSE"
    color   = color_ok if skeleton.pose_detected else color_warn
    _cv2.putText(bgr_frame, status, (12, 28), font, 0.7, (0, 0, 0), 4, _cv2.LINE_AA)
    _cv2.putText(bgr_frame, status, (12, 28), font, 0.7, color,     2, _cv2.LINE_AA)

    fid_str = f"frame {skeleton.frame_id}  t={skeleton.timestamp:.2f}s"
    _cv2.putText(bgr_frame, fid_str, (12, 54), font, 0.5, (0, 0, 0), 3, _cv2.LINE_AA)
    _cv2.putText(bgr_frame, fid_str, (12, 54), font, 0.5, (220, 220, 220), 1, _cv2.LINE_AA)

    # Visibility summary
    if skeleton.pose_detected:
        n_visible = sum(skeleton.visible.values())
        n_total   = len(skeleton.visible)
        vis_str   = f"joints visible: {n_visible}/{n_total}"
        _cv2.putText(bgr_frame, vis_str, (12, 76), font, 0.5, (0, 0, 0), 3, _cv2.LINE_AA)
        _cv2.putText(bgr_frame, vis_str, (12, 76), font, 0.5, (200, 200, 200), 1, _cv2.LINE_AA)
