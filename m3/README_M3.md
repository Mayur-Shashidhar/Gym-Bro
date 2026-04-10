# M3 Overview (Angles + Rep Detection)

M3 converts normalized skeleton joints (from M2) into angle signals, then detects exercise reps using a state machine.

## What M3 does

- Computes joint angles using vector math (fast NumPy operations).
- Builds one movement signal per exercise (example: squat uses knee-angle signal).
- Detects reps with:
  - dynamic calibration (first frames),
  - down/up thresholds,
  - side-consistency checks,
  - timeout reset to avoid stuck states.
- Produces per-rep confidence score based on amplitude, smoothness, and visibility.

## Files

- m3_exercise_config.py: central config for supported exercises and defaults.
- m3_vector_utils.py: vector and angle helper functions.
- m3_angles.py: named angle extraction and exercise signal selection.
- m3_reps.py: RepCounter state machine, calibration, confidence, and filtering.
- m3_api.py: public import surface for other modules.
- test_m3.py: unit tests for M3 logic.

## Add a new exercise (config-driven)

You only need to edit m3_exercise_config.py:

1) Add one ExerciseProfile entry in EXERCISE_PROFILES.
2) Set signal_left_angle, signal_right_angle, and signal_aggregate.
3) Set visibility_joints and side-gap angle pair.
4) Set default thresholds and timing values.

No changes are required in m3_angles.py or m3_reps.py for a normal rep-based exercise.

## Inputs expected from M2

M3 expects M2 SkeletonFrame-like data:

- landmarks_norm: normalized joint coordinates dictionary.
- visible: joint visibility dictionary.
- frame_id: increasing frame index.
- pose_detected: boolean.

## Public imports for other modules (M1/M4/M5/M6/M7)

Use only these imports from m3_api.py:

- compute_named_angles
- exercise_signal_angle
- RepCounter
- RepState
- default_rep_counter
- list_supported_exercises

## Minimal integration pattern

1) Create a counter:

counter = default_rep_counter("squat")

2) For each frame from M2:

state = counter.update_from_skeleton(skeleton)

3) Use state values:

- state.rep_count
- state.phase
- state.last_rep_confidence
- state.rep_confidences
- state.rejected_reps
- state.calibration_complete
- state.active_down_threshold
- state.active_up_threshold

## What to observe during runtime

- Calibration phase:
  - state.phase is "calibrating" until enough frames are collected.
  - Reps are counted after calibration completes.
- Rep quality:
  - last_rep_confidence in range [0, 1].
- Symmetry/noise:
  - rejected_reps increases when side mismatch is too high.
- Stability:
  - timeout reset returns phase to "start" if user pauses too long in "down".

## Notes for other M modules

- M4 can reuse compute_named_angles for form checks.
- M5 should display rep_count, phase, confidence, and calibration status.
- M1/M6 can optionally store angle signal and confidence as extra features.
- M7 should treat m3_api.py as the only stable contract.
