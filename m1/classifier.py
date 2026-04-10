import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.metrics         import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

warnings.filterwarnings("ignore")


# Paths


MODEL_DIR         = Path(__file__).resolve().parents[1] / "models"
CLASSIFIER_PATH   = MODEL_DIR / "exercise_classifier.pkl"
SCALER_PATH       = MODEL_DIR / "scaler.pkl"
LABEL_ENC_PATH    = MODEL_DIR / "label_encoder.pkl"
FEATURE_COLS_PATH = MODEL_DIR / "feature_cols.json"
DEFAULT_CSV = Path(__file__).resolve().parent / "data" / "feature_vectors.csv"
# ─────────────────────────────────────────────────────────────────────────────
# Feature columns
# Must match exactly what extract_dataset.py writes into the CSV
# ─────────────────────────────────────────────────────────────────────────────

# Raw feature vector from M2 (39 values: 13 joints x xyz)
_EXERCISE_JOINTS = [
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW",    "RIGHT_ELBOW",
    "LEFT_WRIST",    "RIGHT_WRIST",
    "LEFT_HIP",      "RIGHT_HIP",
    "LEFT_KNEE",     "RIGHT_KNEE",
    "LEFT_ANKLE",    "RIGHT_ANKLE",
    "NOSE",
]
RAW_FV_COLS = []
for jname in _EXERCISE_JOINTS:
    short = jname.lower()
    RAW_FV_COLS += [f"{short}_x", f"{short}_y", f"{short}_z"]

# M3 angle columns (matches ANGLE_COLS in extract_dataset.py)
ANGLE_COLS = [
    "left_knee",
    "right_knee",
    "left_hip",
    "right_hip",
    "left_elbow",
    "right_elbow",
]

# Symmetry columns (matches SYMMETRY_COLS in extract_dataset.py)
SYMMETRY_COLS = [
    "knee_symmetry",
    "hip_symmetry",
    "elbow_symmetry",
]

# Full feature set for the classifier
FEATURE_COLS = RAW_FV_COLS + ANGLE_COLS + SYMMETRY_COLS


# Data loading


def load_data(csv_path: str = str(DEFAULT_CSV)):
    """
    Load feature_vectors.csv from M6.
    Returns X (features), y (exercise labels), df (full dataframe).
    """
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df)}   Columns: {len(df.columns)}")

    # Fill any missing feature columns with 0
    for col in FEATURE_COLS:
        if col not in df.columns:
            print(f"  [WARN] Missing column '{col}' — filling with 0")
            df[col] = 0.0

    # Drop rows with no exercise label
    df = df.dropna(subset=["exercise"])

    # Show class balance — important to catch imbalanced datasets early
    print("\n  Class balance:")
    for ex, count in df["exercise"].value_counts().items():
        print(f"    {ex:<25} {count} rows")

    X = df[FEATURE_COLS].fillna(0.0).values
    y = df["exercise"].values
    return X, y, df



# Training


def train(csv_path: str = str(DEFAULT_CSV)):
    """
    Train RandomForest exercise classifier.
    Saves model, scaler, label encoder, and feature column list to models/.
    """
    MODEL_DIR.mkdir(exist_ok=True)

    X, y, _ = load_data(csv_path)

    # Encode string labels to integers
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(f"\n  Classes: {list(le.classes_)}")

    # Stratified split — ensures all exercise classes appear in test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=0.2,
        random_state=42,
        stratify=y_enc,
    )
    print(f"  Train: {len(X_train)} rows   Test: {len(X_test)} rows")

    # Scale features — important for RandomForest with mixed-scale inputs
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Train RandomForest ────────────────────────────────────────────────────
    # Chosen because:
    #   - Trains in seconds on small datasets
    #   - No hyperparameter tuning needed
    #   - class_weight='balanced' handles uneven class sizes automatically
    #   - n_jobs=-1 uses all CPU cores
    print("\nTraining RandomForestClassifier ...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_s, y_train)

    # Evaluate
    y_pred = model.predict(X_test_s)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\n{'─'*50}")
    print(f"  Test Accuracy : {acc*100:.1f}%")
    print(f"{'─'*50}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Labels : {list(le.classes_)}")
    print(cm)

    print("\nCross-validation (5-fold) ...")
    cv = cross_val_score(model, X_train_s, y_train, cv=5, scoring="accuracy")
    print(f"  CV Accuracy : {cv.mean()*100:.1f}% +/- {cv.std()*100:.1f}%")

    # Warnings
    if acc < 0.75:
        print("\n[WARN] Accuracy below 75%. Try:")
        print("  1. Run extract_dataset.py --augment to double your data")
        print("  2. Record more videos (at least 10 per exercise)")
        print("  3. Check class balance above — very uneven counts hurt accuracy")

    # Save everything M7 needs 
    joblib.dump(model,  CLASSIFIER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le,     LABEL_ENC_PATH)

    with open(FEATURE_COLS_PATH, "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)

    print(f"\nSaved:")
    print(f"  {CLASSIFIER_PATH}")
    print(f"  {SCALER_PATH}")
    print(f"  {LABEL_ENC_PATH}")
    print(f"  {FEATURE_COLS_PATH}")

    return acc


# Module-level model cache (load once, reuse every frame)


_model  = None
_scaler = None
_le     = None


def _load_models():
    global _model, _scaler, _le
    if _model is None:
        if not CLASSIFIER_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {CLASSIFIER_PATH}.\n"
                f"Run: python classifier.py"
            )
        _model  = joblib.load(CLASSIFIER_PATH)
        _scaler = joblib.load(SCALER_PATH)
        _le     = joblib.load(LABEL_ENC_PATH)



# Prediction API  <-- M7 uses these two functions


def predict_exercise(feature_vector: list) -> str:
    """
    Predict which exercise is being performed.

    Parameters
    ----------
    feature_vector : list of float
        Feature values in FEATURE_COLS order.
        Length must match len(FEATURE_COLS) = 51.
        Shorter vectors are zero-padded; longer ones are trimmed.

    Returns
    -------
    str : one of "squat", "pushup", "shoulder_press"
    """
    _load_models()

    arr      = np.array(feature_vector, dtype=float).reshape(1, -1)
    expected = len(FEATURE_COLS)

    # Pad or trim to expected length
    if arr.shape[1] < expected:
        arr = np.pad(arr, ((0, 0), (0, expected - arr.shape[1])))
    elif arr.shape[1] > expected:
        arr = arr[:, :expected]

    arr_scaled = _scaler.transform(arr)
    pred_idx   = _model.predict(arr_scaled)[0]
    return str(_le.inverse_transform([pred_idx])[0])


def predict_exercise_from_frame(skeleton_frame) -> str:
    """
    Predict exercise directly from an M2 SkeletonFrame.
    Builds the feature vector internally using extract_frame_features.

    Parameters
    ----------
    skeleton_frame : SkeletonFrame
        M2's SkeletonFrame object with pose_detected=True.

    Returns
    -------
    str : exercise label, or "unknown" if pose not detected.
    """
    from m6.extract_dataset import extract_frame_features

    features = extract_frame_features(skeleton_frame)
    if features is None:
        return "unknown"

    fv = [features.get(col, 0.0) for col in FEATURE_COLS]
    return predict_exercise(fv)



# Fatigue detection — rule-based, no ML needed

def detect_fatigue(rep_scores: List[int]) -> str:
    """
    Detect fatigue from rep-over-rep quality score history.

    Works with scores produced by movement_quality_scorer.score_quality().

    Parameters
    ----------
    rep_scores : list of int
        Overall quality scores per completed rep, in order.
        e.g. [88, 85, 79, 70, 62]

    Returns
    -------
    str : "low", "medium", or "high"
    """
    if len(rep_scores) < 3:
        return "low"

    last3 = rep_scores[-3:]
    drop  = last3[0] - last3[-1]

    if drop >= 25:
        return "high"
    elif drop >= 15:
        return "medium"
    else:
        return "low"



# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M1 Exercise Classifier")
    parser.add_argument("--csv",      default=str(DEFAULT_CSV),
                        help="Path to feature_vectors.csv from M6")
    parser.add_argument("--evaluate", action="store_true",
                        help="Load saved model and re-evaluate without retraining")
    args = parser.parse_args()

    if args.evaluate:
        print("Loading saved model for evaluation ...")
        _load_models()
        X, y, _ = load_data(args.csv)
        le_loaded = joblib.load(LABEL_ENC_PATH)
        sc_loaded = joblib.load(SCALER_PATH)
        X_s       = sc_loaded.transform(X)
        y_enc     = le_loaded.transform(y)
        y_pred    = _model.predict(X_s)
        print(classification_report(y_enc, y_pred, target_names=le_loaded.classes_))
    else:
        train(args.csv)
