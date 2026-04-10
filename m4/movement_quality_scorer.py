from __future__ import annotations
import math
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# 1. THRESHOLD CONFIGURATION
#    Each exercise defines four dimensions.
#    Units: angles in degrees, symmetry as ratio [0–1], stability/smoothness
#    in the same units your feature extractor produces (normalised by default).
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLDS: dict[str, dict[str, Any]] = {

    "squat": {
        "rom": {
            # Knee flexion angle at bottom of squat.
            # ~90° = parallel (gold standard); shallower = insufficient depth.
            # Values outside [min_good, max_good] are penalised linearly.
            "joint": "knee_angle",
            "min_good": 80.0,   # degrees – below this starts losing points
            "max_good": 110.0,  # degrees – above this (too shallow) loses points
            "min_acceptable": 60.0,   # hard floor – score → 0
            "max_acceptable": 150.0,  # hard ceiling (standing, no rep)
            "ideal": 90.0,      # peak score target
            "feedback": {
                "too_low":  "Knees bent too far — reduce depth slightly.",
                "good":     "Good squat depth — parallel or below.",
                "too_high": "Go lower — aim for thighs parallel to ground.",
            },
        },
        "symmetry": {
            # Symmetry Index = |left − right| / ((left + right) / 2)
            # < 0.05 = excellent bilateral balance
            # > 0.20 = clinically meaningful asymmetry
            "perfect":     0.05,
            "acceptable":  0.15,
            "poor":        0.25,
            "feedback": {
                "good":   "Even load distribution — great symmetry.",
                "medium": "Slight left/right imbalance detected.",
                "poor":   "Significant asymmetry — check hip/knee alignment.",
            },
        },
        "stability": {
            # Standard deviation of the hip/CoM trajectory (vertical axis).
            # Low std = controlled descent/ascent.
            "perfect":    0.02,   # < 2 % of body height = very stable
            "acceptable": 0.06,
            "poor":       0.12,
            "feedback": {
                "good":   "Stable movement — solid core engagement.",
                "medium": "Some wobble detected — brace your core.",
                "poor":   "Unstable movement — slow down and control the rep.",
            },
        },
        "smoothness": {
            # Mean absolute jerk (rate of change of acceleration), normalised.
            # Low jerk = smooth, controlled force application.
            "perfect":    0.10,
            "acceptable": 0.25,
            "poor":       0.45,
            "feedback": {
                "good":   "Smooth, controlled movement — excellent tempo.",
                "medium": "Slightly jerky — focus on a steady pace.",
                "poor":   "Very jerky movement — use a 3-second descent.",
            },
        },
        "weights": {"rom": 0.30, "symmetry": 0.25, "stability": 0.25, "smoothness": 0.20},
    },

    "push_up": {
        "rom": {
            # Elbow angle at bottom of push-up.
            # ~90° = full depth; higher angle = incomplete range.
            "joint": "elbow_angle",
            "min_good": 80.0,
            "max_good": 100.0,
            "min_acceptable": 60.0,
            "max_acceptable": 160.0,
            "ideal": 90.0,
            "feedback": {
                "too_low":  "Elbows too deep — risk of shoulder impingement.",
                "good":     "Full push-up depth — great range of motion.",
                "too_high": "Go lower — chest should nearly touch the ground.",
            },
        },
        "symmetry": {
            "perfect":     0.05,
            "acceptable":  0.15,
            "poor":        0.25,
            "feedback": {
                "good":   "Balanced push-up — both sides working equally.",
                "medium": "Minor left/right difference — check hand position.",
                "poor":   "One-sided push-up — align wrists under shoulders.",
            },
        },
        "stability": {
            # Hip sag / pike — measured as deviation of hip from straight line.
            "perfect":    0.03,
            "acceptable": 0.07,
            "poor":       0.14,
            "feedback": {
                "good":   "Rigid plank body line — great stability.",
                "medium": "Slight hip drop — squeeze glutes and abs.",
                "poor":   "Significant hip sag — stop and reset your plank.",
            },
        },
        "smoothness": {
            "perfect":    0.10,
            "acceptable": 0.28,
            "poor":       0.50,
            "feedback": {
                "good":   "Smooth push-up — great tempo control.",
                "medium": "Slightly rushed — slow the descent to 2 seconds.",
                "poor":   "Collapsing on descent — control the lowering phase.",
            },
        },
        "weights": {"rom": 0.30, "symmetry": 0.25, "stability": 0.25, "smoothness": 0.20},
    },

    "shoulder_press": {
        "rom": {
            # Elbow angle at the bottom (start) of the press.
            # ~90° = correct rack position; higher angle = bar too high / no setup;
            # lower angle = excessive windup, risk of shoulder impingement.
            # At the top, full elbow extension is the target — tracked separately
            # via the lock-out angle feature if available.
            "joint": "elbow_angle",
            "min_good": 85.0,    # degrees – below 85° = too much windup at start
            "max_good": 100.0,   # degrees – above 100° = shallow start position
            "min_acceptable": 65.0,   # hard floor
            "max_acceptable": 160.0,  # hard ceiling (bar barely moves)
            "ideal": 90.0,       # 90° rack position is the gold standard
            "feedback": {
                "too_low":  "Elbows too far forward — bring bar to shoulder rack position.",
                "good":     "Good rack position — full pressing range of motion.",
                "too_high": "Start lower — bring the bar to shoulder height before pressing.",
            },
        },
        "symmetry": {
            # Left/right elbow angle difference during the press.
            # Asymmetry indicates one shoulder dominating, causing bar tilt
            # and increased rotator cuff strain on the weaker side.
            "perfect":     0.05,
            "acceptable":  0.15,
            "poor":        0.25,
            "feedback": {
                "good":   "Balanced press — both shoulders driving equally.",
                "medium": "Slight left/right imbalance — focus on the weaker side.",
                "poor":   "Significant shoulder asymmetry — reduce load and address imbalance.",
            },
        },
        "stability": {
            # Lateral sway of the torso / head trajectory during the press.
            # Excessive trunk lean (lumbar hyperextension) is a common compensation
            # when load is too heavy, shifting stress from shoulders to lower back.
            "perfect":    0.02,
            "acceptable": 0.06,
            "poor":       0.12,
            "feedback": {
                "good":   "Stable torso — great core bracing throughout.",
                "medium": "Slight trunk sway — brace your core and keep ribs down.",
                "poor":   "Excessive lean — reduce weight and engage your core.",
            },
        },
        "smoothness": {
            # Mean absolute jerk of the bar/wrist trajectory.
            # A smooth, controlled press protects the rotator cuff.
            # Jerky movement (momentum-driven) masks true shoulder strength.
            "perfect":    0.10,
            "acceptable": 0.25,
            "poor":       0.45,
            "feedback": {
                "good":   "Smooth, controlled press — excellent shoulder stability.",
                "medium": "Slightly jerky — press with steady, deliberate force.",
                "poor":   "Using momentum — slow down and press under full control.",
            },
        },
        "weights": {"rom": 0.30, "symmetry": 0.25, "stability": 0.25, "smoothness": 0.20},
    },

    # ── TEMPLATE: copy this block to add a new exercise ──────────────────────
    # "deadlift": {
    #     "rom": { "joint": "hip_angle", "min_good": 45, "max_good": 70,
    #              "min_acceptable": 30, "max_acceptable": 120, "ideal": 55,
    #              "feedback": {"too_low": "...", "good": "...", "too_high": "..."} },
    #     "symmetry": { "perfect": 0.05, "acceptable": 0.15, "poor": 0.25,
    #                   "feedback": {"good": "...", "medium": "...", "poor": "..."} },
    #     "stability": { "perfect": 0.02, "acceptable": 0.06, "poor": 0.12,
    #                    "feedback": {"good": "...", "medium": "...", "poor": "..."} },
    #     "smoothness": { "perfect": 0.10, "acceptable": 0.25, "poor": 0.45,
    #                     "feedback": {"good": "...", "medium": "...", "poor": "..."} },
    #     "weights": {"rom": 0.30, "symmetry": 0.25, "stability": 0.25, "smoothness": 0.20},
    # },
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. NORMALISATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _score_rom(angle: float, cfg: dict) -> float:
    """
    Map a joint angle to 0–100.

    Logic (biomechanics):
      • Inside [min_good, max_good] → score scaled toward ideal (max at ideal)
      • Outside acceptable range   → 0 (no credit for a non-rep)
      • Linear interpolation in the grey zones

    Returns a float in [0.0, 100.0].
    """
    min_a = cfg["min_acceptable"]
    max_a = cfg["max_acceptable"]
    min_g = cfg["min_good"]
    max_g = cfg["max_good"]
    ideal = cfg["ideal"]

    # Outside hard limits → 0
    if angle <= min_a or angle >= max_a:
        return 0.0

    # Inside ideal band → interpolate toward 100 at the ideal point
    if min_g <= angle <= max_g:
        # Distance from ideal as fraction of half-band width
        half_band = max(ideal - min_g, max_g - ideal, 1e-6)
        dist = abs(angle - ideal)
        return max(0.0, 100.0 - (dist / half_band) * 20.0)  # -20 at band edge

    # In grey zone (below min_good)
    if angle < min_g:
        frac = (angle - min_a) / max(min_g - min_a, 1e-6)
        return frac * 80.0  # max 80 at boundary of good zone

    # In grey zone (above max_good)
    frac = (max_a - angle) / max(max_a - max_g, 1e-6)
    return frac * 80.0


def _score_metric(value: float, cfg: dict) -> float:
    """
    Generic scorer for symmetry, stability, smoothness.
    All three share the same shape: lower value = better.

    Breakpoints:
      value ≤ perfect    → 100
      value ≤ acceptable → linear 60–100
      value ≤ poor       → linear 20–60
      value > poor       → linear 0–20
    """
    p = cfg["perfect"]
    a = cfg["acceptable"]
    r = cfg["poor"]

    if value <= p:
        return 100.0
    if value <= a:
        frac = (value - p) / max(a - p, 1e-9)
        return 100.0 - frac * 40.0          # 100 → 60
    if value <= r:
        frac = (value - a) / max(r - a, 1e-9)
        return 60.0 - frac * 40.0           # 60 → 20
    # Beyond poor
    frac = min((value - r) / max(r, 1e-9), 1.0)
    return max(0.0, 20.0 - frac * 20.0)     # 20 → 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEEDBACK GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _feedback_rom(angle: float, cfg: dict) -> str:
    fb = cfg["feedback"]
    if angle < cfg["min_good"]:
        return fb["too_low"]
    if angle > cfg["max_good"]:
        return fb["too_high"]
    return fb["good"]


def _feedback_metric(value: float, cfg: dict) -> str:
    fb = cfg["feedback"]
    if value <= cfg["perfect"]:
        return fb["good"]
    if value <= cfg["acceptable"]:
        return fb["medium"]
    return fb["poor"]


def _combine_feedback(parts: list[str]) -> str:
    """Deduplicate and join feedback sentences."""
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return " ".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN SCORING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def score_quality(features: dict[str, float], exercise: str) -> dict[str, Any]:
    """
    Score a single exercise rep from extracted pose features.

    Parameters
    ----------
    features : dict
        Keys expected (all values are floats):
          - <joint_name>  e.g. "knee_angle", "elbow_angle"   (degrees)
          - "symmetry_index"   ratio in [0, 1]
          - "stability"        std-dev of CoM trajectory
          - "smoothness"       mean absolute jerk (normalised)

    exercise : str
        Must be a key in THRESHOLDS (e.g. "squat", "push_up", "lunge").

    Returns
    -------
    dict with keys: rom, symmetry, stability, smoothness, overall (0–100 int),
    feedback_text (str).

    Raises
    ------
    ValueError if exercise is not configured.
    """
    if exercise not in THRESHOLDS:
        raise ValueError(
            f"Exercise '{exercise}' not configured. "
            f"Available: {list(THRESHOLDS.keys())}"
        )

    cfg = THRESHOLDS[exercise]
    weights = cfg["weights"]

    # ── ROM ──────────────────────────────────────────────────────────────────
    rom_cfg = cfg["rom"]
    joint_key = rom_cfg["joint"]
    if joint_key not in features:
        raise KeyError(f"Feature '{joint_key}' required for {exercise} ROM scoring.")
    angle = features[joint_key]
    rom_score = _score_rom(angle, rom_cfg)
    rom_fb = _feedback_rom(angle, rom_cfg)

    # ── SYMMETRY ─────────────────────────────────────────────────────────────
    sym_val = features.get("symmetry_index", 0.0)
    sym_score = _score_metric(sym_val, cfg["symmetry"])
    sym_fb = _feedback_metric(sym_val, cfg["symmetry"])

    # ── STABILITY ────────────────────────────────────────────────────────────
    stab_val = features.get("stability", 0.0)
    stab_score = _score_metric(stab_val, cfg["stability"])
    stab_fb = _feedback_metric(stab_val, cfg["stability"])

    # ── SMOOTHNESS ───────────────────────────────────────────────────────────
    smooth_val = features.get("smoothness", 0.0)
    smooth_score = _score_metric(smooth_val, cfg["smoothness"])
    smooth_fb = _feedback_metric(smooth_val, cfg["smoothness"])

    # ── WEIGHTED OVERALL ─────────────────────────────────────────────────────
    overall = (
        rom_score    * weights["rom"]
        + sym_score  * weights["symmetry"]
        + stab_score * weights["stability"]
        + smooth_score * weights["smoothness"]
    )

    # ── FEEDBACK ASSEMBLY ────────────────────────────────────────────────────
    feedback_text = _combine_feedback([rom_fb, sym_fb, stab_fb, smooth_fb])

    return {
        "rom":           int(round(rom_score)),
        "symmetry":      int(round(sym_score)),
        "stability":     int(round(stab_score)),
        "smoothness":    int(round(smooth_score)),
        "overall":       int(round(overall)),
        "feedback_text": feedback_text,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. TEST CASES
# ─────────────────────────────────────────────────────────────────────────────

def _run_tests() -> None:
    separator = "─" * 60

    test_cases = [
        # ── SQUAT ─────────────────────────────────────────────────────────
        {
            "label": "SQUAT — Good Rep",
            "exercise": "squat",
            "features": {
                "knee_angle":      92.0,   # near ideal 90°
                "symmetry_index":  0.03,   # excellent bilateral balance
                "stability":       0.018,  # very stable CoM
                "smoothness":      0.08,   # smooth velocity profile
            },
        },
        {
            "label": "SQUAT — Bad Rep (shallow + asymmetric + jerky)",
            "exercise": "squat",
            "features": {
                "knee_angle":      138.0,  # far too shallow (barely a quarter squat)
                "symmetry_index":  0.22,   # clinically meaningful asymmetry
                "stability":       0.11,   # unstable, compensatory sway
                "smoothness":      0.42,   # very jerky descent/ascent
            },
        },

        # ── PUSH-UP ───────────────────────────────────────────────────────
        {
            "label": "PUSH-UP — Good Rep",
            "exercise": "push_up",
            "features": {
                "elbow_angle":     88.0,   # full depth
                "symmetry_index":  0.04,
                "stability":       0.025,  # rigid plank
                "smoothness":      0.09,
            },
        },
        {
            "label": "PUSH-UP — Bad Rep (incomplete + hip sag)",
            "exercise": "push_up",
            "features": {
                "elbow_angle":     145.0,  # barely bent — almost no range
                "symmetry_index":  0.18,
                "stability":       0.12,   # severe hip drop
                "smoothness":      0.46,
            },
        },

        # ── SHOULDER PRESS ────────────────────────────────────────────────
        {
            "label": "SHOULDER PRESS — Good Rep",
            "exercise": "shoulder_press",
            "features": {
                "elbow_angle":     91.0,   # near-ideal 90° rack position
                "symmetry_index":  0.03,   # excellent bilateral shoulder balance
                "stability":       0.018,  # minimal trunk sway
                "smoothness":      0.09,   # controlled pressing velocity
            },
        },
        {
            "label": "SHOULDER PRESS — Bad Rep (shallow + asymmetric + momentum)",
            "exercise": "shoulder_press",
            "features": {
                "elbow_angle":     148.0,  # bar barely lowered — almost no ROM
                "symmetry_index":  0.23,   # one shoulder dominant
                "stability":       0.11,   # excessive lumbar lean/sway
                "smoothness":      0.43,   # momentum-driven, jerky press
            },
        },
    ]

    print("\n" + "=" * 60)
    print("  MOVEMENT QUALITY SCORER — TEST RESULTS")
    print("=" * 60)

    for tc in test_cases:
        result = score_quality(tc["features"], tc["exercise"])
        print(f"\n{separator}")
        print(f"  {tc['label']}")
        print(separator)
        print(f"  Input features : {tc['features']}")
        print(f"  ROM            : {result['rom']:>3}/100")
        print(f"  Symmetry       : {result['symmetry']:>3}/100")
        print(f"  Stability      : {result['stability']:>3}/100")
        print(f"  Smoothness     : {result['smoothness']:>3}/100")
        print(f"  ── OVERALL     : {result['overall']:>3}/100  ──")
        print(f"  Feedback       : {result['feedback_text']}")

    print(f"\n{separator}")
    print("  Score gap validation (good vs bad, each exercise):")
    for ex in ["squat", "push_up", "shoulder_press"]:
        good_cases = [t for t in test_cases if t["exercise"] == ex and "Good" in t["label"]]
        bad_cases  = [t for t in test_cases if t["exercise"] == ex and "Bad"  in t["label"]]
        g = score_quality(good_cases[0]["features"], ex)["overall"]
        b = score_quality(bad_cases[0]["features"],  ex)["overall"]
        gap = g - b
        status = "✓ PASS" if gap >= 15 else "✗ FAIL"
        print(f"  [{status}]  {ex:10s}  good={g:>3}  bad={b:>3}  gap={gap:>3}")
    print(separator + "\n")


if __name__ == "__main__":
    _run_tests()
