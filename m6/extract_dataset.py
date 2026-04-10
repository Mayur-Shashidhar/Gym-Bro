import sys
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

# ── M2 imports ────────────────────────────────────────────────────────────────

from m2.m2_api import stream_video, SkeletonFrame
from m2.landmark_utils import EXERCISE_JOINTS

# ── M3 imports ────────────────────────────────────────────────────────────────
from m3.m3_api import compute_named_angles

# ── Supported exercises ───────────────────────────────────────────────────────
SUPPORTED_EXERCISES = ["squat", "pushup", "shoulder_press"]

# ── CSV column definitions ────────────────────────────────────────────────────

# Raw feature vector columns (39 values: 13 joints x xyz) — from M2 directly
FV_COLS = []
for jname in EXERCISE_JOINTS:
    short = jname.lower()
    FV_COLS += [f"{short}_x", f"{short}_y", f"{short}_z"]

# Joint angle columns — computed by M3's compute_named_angles()
ANGLE_COLS = [
    "left_knee",
    "right_knee",
    "left_hip",
    "right_hip",
    "left_elbow",
    "right_elbow",
]

# Symmetry columns — derived from M3 angles
SYMMETRY_COLS = [
    "knee_symmetry",
    "hip_symmetry",
    "elbow_symmetry",
]

# Meta columns — NO quality column, that is handled by movement_quality_scorer.py
META_COLS = ["exercise", "video_file", "frame_id"]

ALL_COLS = META_COLS + FV_COLS + ANGLE_COLS + SYMMETRY_COLS


# ─────────────────────────────────────────────────────────────────────────────
# Symmetry helper
# ─────────────────────────────────────────────────────────────────────────────

def _symmetry(left: Optional[float], right: Optional[float]) -> float:
    """
    Symmetry index: 0 = perfect symmetry, higher = more asymmetry.
    Returns 0.0 if either value is missing.
    """
    if left is None or right is None:
        return 0.0
    avg = (abs(left) + abs(right)) / 2.0
    if avg < 1e-9:
        return 0.0
    return abs(left - right) / avg


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_frame_features(frame: SkeletonFrame) -> Optional[Dict]:
    """
    Given one SkeletonFrame from M2, return a flat feature dict.
    Uses M3's compute_named_angles for all angle computation.
    Returns None if pose not detected.
    """
    if not frame.pose_detected:
        return None

    row = {}

    # ── Raw feature vector from M2 ────────────────────────────────────────────
    fv = frame.feature_vector  # 39 floats, already normalised by M2
    for i, col in enumerate(FV_COLS):
        row[col] = round(fv[i], 6) if i < len(fv) else 0.0

    # ── Joint angles via M3 ───────────────────────────────────────────────────
    angles = compute_named_angles(frame.landmarks_norm, frame.visible)
    for col in ANGLE_COLS:
        val = angles.get(col)
        row[col] = round(val, 4) if val is not None else 0.0

    # ── Symmetry indices ──────────────────────────────────────────────────────
    row["knee_symmetry"]  = round(_symmetry(angles.get("left_knee"),  angles.get("right_knee")),  4)
    row["hip_symmetry"]   = round(_symmetry(angles.get("left_hip"),   angles.get("right_hip")),   4)
    row["elbow_symmetry"] = round(_symmetry(angles.get("left_elbow"), angles.get("right_elbow")), 4)

    return row


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation: horizontal flip
# ─────────────────────────────────────────────────────────────────────────────

def flip_row(row: Dict) -> Dict:
    """
    Mirror a feature row horizontally.
    - Negates all _x coordinates in the raw feature vector
    - Swaps left <-> right angle values so labels stay correct
    Doubles dataset size for free — simulates a mirrored camera angle.
    """
    flipped = dict(row)

    # Negate all x values
    for col in FV_COLS:
        if col.endswith("_x"):
            flipped[col] = -row[col]

    # Swap left <-> right angles
    swap_pairs = [
        ("left_knee",  "right_knee"),
        ("left_hip",   "right_hip"),
        ("left_elbow", "right_elbow"),
    ]
    for left_col, right_col in swap_pairs:
        flipped[left_col]  = row[right_col]
        flipped[right_col] = row[left_col]

    # Symmetry values are symmetric by definition — no swap needed
    flipped["video_file"] = row["video_file"] + "_flipped"
    return flipped


# ─────────────────────────────────────────────────────────────────────────────
# Per-video processing
# ─────────────────────────────────────────────────────────────────────────────

def process_video(
    video_path: str,
    exercise: str,
    augment: bool = False,
) -> List[Dict]:
    """
    Run M2's stream_video on one file, extract features per frame.
    Returns list of feature dicts — one per valid frame.
    If augment=True, also appends horizontally flipped copies.
    """
    rows = []
    video_name = Path(video_path).name
    print(f"  Processing: {video_name} ({exercise})", end="", flush=True)

    try:
        for frame in stream_video(video_path):
            features = extract_frame_features(frame)
            if features is None:
                continue
            features["exercise"]   = exercise
            features["video_file"] = video_name
            features["frame_id"]   = frame.frame_id
            rows.append(features)

            if augment:
                rows.append(flip_row(features))

    except Exception as e:
        print(f"\n    WARNING: skipped {video_name} — {e}")
        return []

    print(f" -> {len(rows)} rows")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction loop
# ─────────────────────────────────────────────────────────────────────────────

def extract_all(
    video_dir: str = "videos",
    output_csv: str = "data/feature_vectors.csv",
    augment: bool = False,
):
    """
    Walk the video_dir folder tree and extract features from every video.

    Expected folder structure:
        video_dir/
            squat/              *.mp4
            pushup/             *.mp4
            shoulder_press/     *.mp4
    """
    video_dir  = Path(video_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}

    for exercise in SUPPORTED_EXERCISES:
        ex_dir = video_dir / exercise
        if not ex_dir.exists():
            print(f"[WARN] Folder not found, skipping: {ex_dir}")
            continue

        video_files = [
            f for f in ex_dir.iterdir()
            if f.suffix.lower() in video_extensions
        ]

        if not video_files:
            print(f"[WARN] No videos found in {ex_dir}")
            continue

        print(f"\n[{exercise.upper()}] — {len(video_files)} videos")

        for vf in sorted(video_files):
            rows = process_video(str(vf), exercise, augment=augment)
            all_rows.extend(rows)

    if not all_rows:
        print("\nERROR: No data extracted. Check your video folder structure.")
        return

    # ── Write CSV ─────────────────────────────────────────────────────────────
    print(f"\nWriting {len(all_rows)} rows -> {output_csv} ...")

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({col: row.get(col, 0.0) for col in ALL_COLS})

    print(f"Done. Dataset saved to: {output_csv}")
    _print_summary(all_rows)


def _print_summary(rows: List[Dict]):
    from collections import Counter
    counts = Counter(r["exercise"] for r in rows)
    print("\n-- Dataset Summary -------------------------")
    print(f"{'Exercise':<20} {'Frames':>8}")
    print("-" * 30)
    for ex, count in sorted(counts.items()):
        print(f"{ex:<20} {count:>8}")
    print("-" * 30)
    print(f"{'TOTAL':<20} {len(rows):>8}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M6 Dataset Extractor")
    parser.add_argument("--video_dir",  default="videos",
                        help="Root folder containing exercise subfolders")
    parser.add_argument("--output_csv", default="data/feature_vectors.csv",
                        help="Output CSV path")
    parser.add_argument("--augment", action="store_true",
                        help="Add horizontally flipped copies (doubles dataset)")
    args = parser.parse_args()

    extract_all(
        video_dir=args.video_dir,
        output_csv=args.output_csv,
        augment=args.augment,
    )
