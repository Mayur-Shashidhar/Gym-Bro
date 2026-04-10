import sys
import os
import time
import json
from pathlib import Path

import cv2
import numpy as np

# ── Import M2 (same directory) ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from skeleton_tracker import SkeletonTracker
from landmark_utils   import EXERCISE_JOINTS, POSE_CONNECTIONS


# ── Layout constants ─────────────────────────────────────────────────────────
CAM_W, CAM_H  = 640, 480
PANEL_W       = 340
WIN_W         = CAM_W + PANEL_W
WIN_H         = CAM_H
FONT          = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL    = 0.42
FONT_MED      = 0.52
FONT_LARGE    = 0.72
SNAPSHOTS_DIR = Path("m2_snapshots")

# Joints to show normalised coords for in the panel
COORD_JOINTS = [
    "LEFT_SHOULDER",  "RIGHT_SHOULDER",
    "LEFT_HIP",       "RIGHT_HIP",
    "LEFT_KNEE",      "RIGHT_KNEE",
    "LEFT_WRIST",     "RIGHT_WRIST",
]


# ── Colour palette (BGR) ─────────────────────────────────────────────────────
C_BG        = (18,  18,  18)
C_PANEL_BG  = (28,  28,  28)
C_TEXT      = (210, 210, 210)
C_TEXT_DIM  = (110, 110, 110)
C_GREEN     = (80,  220, 80)
C_RED       = (60,  60,  220)
C_YELLOW    = (40,  210, 210)
C_ACCENT    = (180, 100, 255)
C_WHITE     = (240, 240, 240)
C_DIVIDER   = (50,  50,  50)


# ── Helpers ───────────────────────────────────────────────────────────────────

def put(img, text, xy, scale=FONT_SMALL, color=C_TEXT, bold=False):
    thick = 2 if bold else 1
    cv2.putText(img, text, xy, FONT, scale, color, thick, cv2.LINE_AA)


def hline(img, y, x0=0, x1=None):
    if x1 is None:
        x1 = img.shape[1]
    cv2.line(img, (x0, y), (x1, y), C_DIVIDER, 1)


def bar(img, x, y, w, h, frac, fg, bg=(45, 45, 45)):
    """Draw a horizontal fill bar."""
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    fill = max(1, int(frac * w))
    cv2.rectangle(img, (x, y), (x + fill, y + h), fg, -1)


def fvec_heatmap(img, x, y, w, h, fvec):
    """Render 39-value feature vector as a row of tiny colour cells."""
    n    = len(fvec)
    if n == 0:
        return
    cw   = max(1, w // n)
    vmax = max(abs(v) for v in fvec) or 1.0
    for i, v in enumerate(fvec):
        norm = v / vmax                          # –1 … +1
        if norm >= 0:
            r, g, b = int(norm * 60), int(norm * 200), int(norm * 80)
        else:
            r, g, b = int(-norm * 220), int(-norm * 60), int(-norm * 60)
        cx = x + i * cw
        cv2.rectangle(img, (cx, y), (cx + cw - 1, y + h), (b, g, r), -1)


# ── Data panel renderer ───────────────────────────────────────────────────────

def draw_panel(panel, skeleton, fps, model_complexity, show_panel):
    panel[:] = C_PANEL_BG

    if not show_panel:
        put(panel, "panel hidden  [d] to show", (10, CAM_H // 2), color=C_TEXT_DIM)
        return

    y = 14

    # ── Header ──
    put(panel, "M2 Skeleton Tracker", (10, y), FONT_MED, C_WHITE, bold=True)
    y += 20
    put(panel, f"complexity={model_complexity}  fps={fps:.1f}", (10, y), FONT_SMALL, C_TEXT_DIM)
    y += 14
    hline(panel, y); y += 10

    # ── Detection status ──
    if skeleton and skeleton.pose_detected:
        put(panel, "POSE DETECTED", (10, y), FONT_MED, C_GREEN, bold=True)
    else:
        put(panel, "NO POSE", (10, y), FONT_MED, C_RED, bold=True)
        y += 20
        put(panel, "Stand fully in frame,", (10, y), FONT_SMALL, C_TEXT_DIM)
        y += 14
        put(panel, "ensure good lighting.", (10, y), FONT_SMALL, C_TEXT_DIM)
        return

    fid = skeleton.frame_id
    ts  = skeleton.timestamp
    put(panel, f"frame {fid}   t={ts:.2f}s", (180, y), FONT_SMALL, C_TEXT_DIM)
    y += 22
    hline(panel, y); y += 10

    # ── Joint visibility bars ──
    put(panel, "Joint visibility", (10, y), FONT_SMALL, C_ACCENT, bold=True)
    y += 16

    for jname in EXERCISE_JOINTS:
        vis     = skeleton.visible.get(jname, False)
        raw_vis = skeleton.landmarks_raw.get(jname, {}).get("visibility", 0.0)
        color   = C_GREEN if vis else C_RED
        label   = jname.replace("_", " ").lower()

        put(panel, label, (10, y + 8), FONT_SMALL, C_TEXT)
        bar(panel, 175, y, 140, 12, raw_vis, color)
        put(panel, f"{raw_vis:.2f}", (320, y + 8), FONT_SMALL,
            C_GREEN if vis else C_RED)
        y += 16

    hline(panel, y); y += 10

    # ── Normalised coordinates table ──
    put(panel, "Normalised coords (hip=origin)", (10, y), FONT_SMALL, C_ACCENT, bold=True)
    y += 16
    put(panel, "Joint              x       y       z", (10, y), FONT_SMALL, C_TEXT_DIM)
    y += 14

    for jname in COORD_JOINTS:
        if not skeleton.visible.get(jname):
            continue
        lm    = skeleton.landmarks_norm.get(jname, {})
        label = jname.replace("_", " ").lower()[:17]
        put(panel, f"{label:<17}", (10,  y), FONT_SMALL, C_TEXT)
        put(panel, f"{lm.get('x', 0): .2f}", (156, y), FONT_SMALL, C_YELLOW)
        put(panel, f"{lm.get('y', 0): .2f}", (210, y), FONT_SMALL, C_YELLOW)
        put(panel, f"{lm.get('z', 0): .2f}", (264, y), FONT_SMALL, C_YELLOW)
        y += 14
        if y > CAM_H - 70:
            put(panel, "...", (10, y), FONT_SMALL, C_TEXT_DIM)
            break

    hline(panel, y); y += 8

    # ── Feature vector heatmap ──
    put(panel, f"Feature vector ({len(skeleton.feature_vector)} values)", (10, y),
        FONT_SMALL, C_ACCENT, bold=True)
    y += 16
    fvec_heatmap(panel, 10, y, PANEL_W - 20, 20, skeleton.feature_vector)
    y += 26
    put(panel, "green=+  red=−  intensity=magnitude", (10, y), FONT_SMALL, C_TEXT_DIM)
    y += 18

    # ── Controls reminder ──
    hline(panel, CAM_H - 40)
    put(panel, "[q]uit  [s]napshot  [r]eset  [d]ata  [1/2/3]model",
        (10, CAM_H - 24), FONT_SMALL, C_TEXT_DIM)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    print("\n=== M2 Live Test Harness ===")
    print("Opening webcam …  (press q in the window to quit)\n")

    SNAPSHOTS_DIR.mkdir(exist_ok=True)

    model_complexity = 2
    tracker = SkeletonTracker(
        model_complexity=model_complexity,
        draw_skeleton=True,
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    if not cap.isOpened():
        print("ERROR: Could not open webcam (index 0).")
        print("Try changing camera_index in test_live.py if you have multiple cameras.")
        tracker.release()
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened — actual resolution: {actual_w}×{actual_h}")
    print(f"Model complexity: {model_complexity}  (keys 1/2/3 to change)\n")

    # FPS tracking
    fps_buf   = []
    last_time = time.time()

    show_panel = True
    last_skeleton = None

    while True:
        ret, bgr = cap.read()
        if not ret:
            print("Camera read failed — exiting.")
            break

        # Mirror for natural selfie feel
        bgr = cv2.flip(bgr, 1)

        # ── Process ──
        skeleton     = tracker.process_frame(bgr)
        last_skeleton = skeleton

        # ── FPS ──
        now    = time.time()
        fps_buf.append(1.0 / max(now - last_time, 1e-6))
        last_time = now
        if len(fps_buf) > 20:
            fps_buf.pop(0)
        fps = sum(fps_buf) / len(fps_buf)

        # ── Compose window ──
        canvas = np.full((CAM_H, WIN_W, 3), C_BG, dtype=np.uint8)

        # Left: annotated frame (resize to CAM_W×CAM_H if camera res differs)
        ann = skeleton.annotated_frame
        if ann.shape[:2] != (CAM_H, CAM_W):
            ann = cv2.resize(ann, (CAM_W, CAM_H))
        canvas[:, :CAM_W] = ann

        # Right: data panel
        panel = canvas[:, CAM_W:]
        draw_panel(panel, skeleton, fps, model_complexity, show_panel)

        # ── FPS overlay on video ──
        put(canvas, f"{fps:.0f} fps", (CAM_W - 72, 22), FONT_MED,
            C_GREEN if fps >= 20 else C_YELLOW, bold=True)

        cv2.imshow("M2 Live Test", canvas)

        # ── Key handling ──
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Quit.")
            break

        elif key == ord("s"):
            # Save snapshot
            ts_str = time.strftime("%Y%m%d_%H%M%S")
            img_path  = SNAPSHOTS_DIR / f"snapshot_{ts_str}.jpg"
            json_path = SNAPSHOTS_DIR / f"snapshot_{ts_str}.json"
            cv2.imwrite(str(img_path), canvas)
            if last_skeleton:
                with open(json_path, "w") as f:
                    json.dump(last_skeleton.to_dict(), f, indent=2)
            print(f"Snapshot saved → {img_path}  +  {json_path}")

        elif key == ord("r"):
            tracker.reset_timer()
            fps_buf.clear()
            print("Timer and frame counter reset.")

        elif key == ord("d"):
            show_panel = not show_panel

        elif key == ord("1"):
            model_complexity = 0
            tracker.release()
            tracker = SkeletonTracker(model_complexity=0, draw_skeleton=True)
            print("Switched to model_complexity=0 (fastest)")

        elif key == ord("2"):
            model_complexity = 1
            tracker.release()
            tracker = SkeletonTracker(model_complexity=1, draw_skeleton=True)
            print("Switched to model_complexity=1 (balanced)")

        elif key == ord("3"):
            model_complexity = 2
            tracker.release()
            tracker = SkeletonTracker(model_complexity=2, draw_skeleton=True)
            print("Switched to model_complexity=2 (accurate)")

    cap.release()
    tracker.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()