import time
import sys
from pathlib import Path
from collections import deque

import streamlit as st

from components import (
    render_control_bar,
    render_live_tracking,
    render_coach_panel,
    render_analytics_section,
)

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from m2.m2_api import SkeletonTracker, stream_webcam
from m1.classifier import _load_models
from main import GymBroPipeline

st.set_page_config(
    page_title="AI Gym Coach Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_CSS_PATH = Path(__file__).with_name("styles.css")
with open(_CSS_PATH, "r", encoding="utf-8") as css_file:
    st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="page-wrap">
        <div class="header">
            <div>
                <div class="title">AI Gym Coach Dashboard</div>
                <div class="subtitle">Coach-grade movement feedback with real-time biomechanical insights</div>
            </div>
        </div>
        <div class="divider"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

def _init_state() -> None:
    st.session_state.setdefault("start_session", False)
    st.session_state.setdefault("tracker", None)
    st.session_state.setdefault("stream", None)
    st.session_state.setdefault("current_exercise", None)
    st.session_state.setdefault("rep_counter", None)
    st.session_state.setdefault("last_state", None)
    st.session_state.setdefault("last_frame", None)
    st.session_state.setdefault("pipeline", GymBroPipeline())
    st.session_state.setdefault("rep_scores", [])
    st.session_state.setdefault("last_rep_count", 0)
    st.session_state.setdefault(
        "quality_report",
        {
            "overall": 0,
            "feedback_text": "Stand in frame to begin",
            "rom": 0,
            "symmetry": 0,
            "stability": 0,
            "smoothness": 0,
        },
    )
    st.session_state.setdefault("fatigue_level", "low")
    st.session_state.setdefault("exercise_votes", deque(maxlen=30))
    st.session_state.setdefault("models_ready", None)


def _start_stream() -> None:
    if st.session_state.tracker is None:
        st.session_state.tracker = SkeletonTracker(model_complexity=1, draw_skeleton=True)
    if st.session_state.stream is None:
        st.session_state.stream = stream_webcam(
            tracker=st.session_state.tracker,
            show_preview=False,
            fps_cap=30.0,
        )


def _stop_stream() -> None:
    if st.session_state.stream is not None:
        try:
            st.session_state.stream.close()
        except Exception:
            pass
        st.session_state.stream = None
    if st.session_state.tracker is not None:
        st.session_state.tracker.release()
        st.session_state.tracker = None


def _ensure_models() -> bool:
    if st.session_state.models_ready is not None:
        return bool(st.session_state.models_ready)
    try:
        _load_models()
        st.session_state.models_ready = True
    except FileNotFoundError:
        st.session_state.models_ready = False
    return bool(st.session_state.models_ready)


_init_state()

button_cols = st.columns([1, 1, 6])
with button_cols[0]:
    if st.button("Start Session", type="primary", use_container_width=True):
        st.session_state.start_session = True
with button_cols[1]:
    if st.button("Stop", use_container_width=True):
        st.session_state.start_session = False

start_session = bool(st.session_state.start_session)
mode = "Live" if start_session else "Demo"

control_placeholder = st.empty()

st.markdown("<div class=\"section-spacer\"></div>", unsafe_allow_html=True)

left_col, right_col = st.columns([7, 3], gap="large")
video_placeholder = left_col.empty()
coach_placeholder = right_col.empty()

st.markdown("<div class=\"section-spacer-lg\"></div>", unsafe_allow_html=True)

analytics_placeholder = st.empty()

if start_session:
    models_ready = _ensure_models()
    if not models_ready:
        st.warning(
            "Classifier model not found. Run: python m1/classifier.py to enable auto-detection."
        )
    _start_stream()
    while st.session_state.start_session:
        try:
            skeleton = next(st.session_state.stream)
        except StopIteration:
            _stop_stream()
            break
        except Exception as exc:
            st.error(f"Webcam error: {exc}")
            _stop_stream()
            break

        frame = None
        if skeleton is not None:
            frame = skeleton.annotated_frame
            st.session_state.last_frame = frame

            result = st.session_state.pipeline.update(
                skeleton,
                models_ready=models_ready,
            )
            st.session_state.quality_report = result["quality_report"]
            st.session_state.fatigue_level = result["fatigue_level"]
            st.session_state.rep_scores = result.get("rep_scores", [])
            display_name = result["exercise_label"]
            rep_count = result["rep_count"]
        else:
            display_name = "Detecting"
            rep_count = 0

        # Only re-render heavy UI (coach, analytics, control) if state changed
        # to prevent the Streamlit frontend from lagging/freezing at 30 fps.
        rep_changed = rep_count != st.session_state.get("last_rendered_rep", -1)
        ex_changed = display_name != st.session_state.get("last_rendered_ex", "")
        
        if rep_changed or ex_changed:
            st.session_state["last_rendered_rep"] = rep_count
            st.session_state["last_rendered_ex"] = display_name

            with control_placeholder.container():
                render_control_bar(display_name, True, mode)

            with coach_placeholder.container():
                render_coach_panel(
                    exercise_name=display_name,
                    rep_count=rep_count,
                    overall_score=st.session_state.quality_report.get("overall", 0),
                    fatigue_level=st.session_state.fatigue_level.title(),
                    feedback_text=st.session_state.quality_report.get("feedback_text", ""),
                    status="Active",
                )

            with analytics_placeholder.container():
                render_analytics_section(
                    rom=st.session_state.quality_report.get("rom", 0),
                    symmetry=st.session_state.quality_report.get("symmetry", 0),
                    stability=st.session_state.quality_report.get("stability", 0),
                    smoothness=st.session_state.quality_report.get("smoothness", 0),
                    overall=st.session_state.quality_report.get("overall", 0),
                    start_session=True,
                    rep_scores=st.session_state.rep_scores,
                )

        # Always update video
        with video_placeholder.container():
            render_live_tracking(
                True,
                frame=frame if frame is not None else st.session_state.last_frame,
                status_text="Stand in frame to begin",
            )

        time.sleep(0.03)
else:
    _stop_stream()

    with control_placeholder.container():
        render_control_bar("Detecting", False, mode)

    with video_placeholder.container():
        render_live_tracking(False)

    with coach_placeholder.container():
        render_coach_panel(
            exercise_name="Detecting",
            rep_count=0,
            overall_score=0,
            fatigue_level="Low",
            feedback_text="Start your session to begin",
            status="Idle",
        )

    with analytics_placeholder.container():
        render_analytics_section(
            rom=0,
            symmetry=0,
            stability=0,
            smoothness=0,
            overall=0,
            start_session=False,
            rep_scores=None,
        )
