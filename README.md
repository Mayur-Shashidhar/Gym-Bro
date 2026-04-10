# Gym Bro

Gym Bro is a real-time exercise analyzer that combines pose estimation, rep counting, exercise classification, and movement quality scoring. It supports a CLI pipeline and a Streamlit dashboard for live coaching-style feedback.

## What it does

- Tracks a full-body skeleton with MediaPipe (M2).
- Computes joint angles and rep counts with a signal-based state machine (M3).
- Classifies exercises from per-frame feature vectors (M1).
- Scores rep quality with an explainable rule-based scorer (M4).
- Offers a live Streamlit dashboard UI (M5).
- Builds training datasets from videos (M6).

## Project layout

- main.py: End-to-end live pipeline (M2 -> M3 -> M1 -> M4) using the webcam.
- m1/: Exercise classifier (RandomForest) and fatigue detection.
- m2/: Pose tracking and landmark utilities (MediaPipe Tasks API).
- m3/: Angle extraction and rep counting.
- m4/: Movement quality scoring rules.
- m5/: Streamlit dashboard UI and styling.
- m6/: Dataset extraction from exercise videos.
- data/: feature_vectors.csv (training data).
- models/: saved classifier artifacts (generated).

## Requirements

Install dependencies from the project root:

```bash
pip install -r requirements.txt
```

Notes:
- MediaPipe will auto-download a pose model on first run (~30 MB).
- A working webcam is required for live demos.

## Quick start

### 1) Run the CLI pipeline

```bash
python main.py
```

This opens the webcam, detects the exercise, counts reps, and overlays live quality feedback. Press `q` to quit.

### 2) Run the Streamlit dashboard

```bash
streamlit run m5/app.py
```

The dashboard shows the live video feed, rep counts, and quality metrics.

## Train the classifier (M1)

If the classifier model is missing, train it from a dataset:

1) Build `data/feature_vectors.csv` from videos:

```bash
python m6/extract_dataset.py --video_dir videos --output_csv data/feature_vectors.csv --augment
```

Expected folder structure:

```
videos/
  squat/
  pushup/
  shoulder_press/
```

2) Train the classifier:

```bash
python m1/classifier.py
```

This saves model artifacts to `models/`.

## Tests and utilities

- M2 landmark utilities tests:

```bash
python m2/test_m2.py
```

- M3 rep counter tests:

```bash
python m3/test_m3.py
```

- Quick webcam sanity check:

```bash
python m5/test_cam.py
```

## Supported exercises

- Squat
- Push-up
- Shoulder press

## Troubleshooting

- If the classifier cannot load, run the training step above to create `models/exercise_classifier.pkl`.
- If the webcam cannot open, verify permissions and try the camera test script.

## License

MIT License

Copyright (c) 2026 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
