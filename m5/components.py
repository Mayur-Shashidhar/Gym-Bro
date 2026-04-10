import base64
import random

import cv2
import streamlit as st
import plotly.graph_objects as go


def render_control_bar(exercise_name: str, start_session: bool, mode: str) -> None:
    col1, col2, col3, col4 = st.columns([3, 2, 2, 2], gap="small")
    with col1:
        st.markdown(
            f"<div class=\"control-label\">Exercise</div><div class=\"control-value accent\">{exercise_name}</div>",
            unsafe_allow_html=True,
        )
    with col2:
        status = "Active" if start_session else "Idle"
        st.markdown(
            f"<div class=\"control-label\">Session</div><div class=\"control-value\">{status}</div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"<div class=\"badge badge-mode\">{mode}</div>",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            "<div class=\"badge badge-timer\">00:00</div>",
            unsafe_allow_html=True,
        )


def render_live_tracking(is_active: bool, frame=None, status_text: str = "") -> None:
    live_class = "live" if is_active else "idle"
    inner_html = ""
    if frame is not None:
        ok, buf = cv2.imencode(".jpg", frame)
        if ok:
            img_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            inner_html = f"<img src=\"data:image/jpeg;base64,{img_b64}\" class=\"live-frame\" />"
    if not inner_html:
        msg = status_text or "Start your session to begin"
        inner_html = (
            "<div class=\"video-empty\">"
            f"<div class=\"video-empty-title\">{msg}</div>"
            "<div class=\"video-empty-sub\">Live movement tracking will appear here.</div>"
            "</div>"
        )

    live_badge = "<div class='live-indicator'><span class='dot'></span>LIVE</div>" if is_active else ""
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">Live Movement Tracking</div>
            <div class="video-shell {live_class}">
                {live_badge}
                {inner_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_coach_panel(
    exercise_name: str,
    rep_count: int,
    overall_score: int,
    fatigue_level: str,
    feedback_text: str,
    status: str,
) -> None:
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">Coach Panel</div>
            <div class="status-row">
                <span class="status-badge">{status}</span>
            </div>
            <div class="exercise-name">{exercise_name.upper()}</div>
            <div class="score-card">
                <div class="score-value">{overall_score}</div>
                <div class="score-label">Movement quality index</div>
            </div>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">Rep Count</div>
                    <div class="metric-value">{rep_count}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Fatigue Level</div>
                    <div class="metric-value">{fatigue_level}</div>
                </div>
            </div>
            <div class="feedback-box">
                {feedback_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _chart_layout() -> dict:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#111827", size=12, family="'Space Grotesk', sans-serif"),
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(
            color="#6B7280",
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            color="#6B7280",
            showgrid=True,
            gridcolor="rgba(255,212,0,0.25)",
            zeroline=False,
        ),
    )


def render_analytics_section(
    rom: int,
    symmetry: int,
    stability: int,
    smoothness: int,
    overall: int,
    start_session: bool,
    rep_scores: list[int] | None = None,
) -> None:
    with st.expander("Analytics", expanded=True):
        left, right = st.columns([7, 3], gap="large")
        with left:
            st.markdown("<div class=\"chart-title\">Rep quality scores</div>", unsafe_allow_html=True)
            bar_chart = _bar_chart(start_session, rep_scores=rep_scores)
            st.plotly_chart(
                bar_chart,
                use_container_width=True,
                config={"displayModeBar": False},
            )

        with right:
            st.markdown("<div class=\"metric-stack\">", unsafe_allow_html=True)
            _metric_card("ROM", rom)
            _metric_card("Symmetry", symmetry)
            _metric_card("Stability", stability)
            _metric_card("Smoothness", smoothness)
            _metric_card("Overall", overall)
            st.markdown("</div>", unsafe_allow_html=True)


def _line_chart(start_session: bool) -> go.Figure:
    x_values = list(range(1, 31))
    base = 40 if start_session else 30
    y_values = [base + (random.random() * 20) for _ in x_values]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines",
            line=dict(color="#FFD400", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(255,212,0,0.2)",
        )
    )
    fig.update_layout(**_chart_layout())
    fig.update_yaxes(range=[0, 80])
    return fig


def _bar_chart(start_session: bool, rep_scores: list[int] | None = None) -> go.Figure:
    if rep_scores:
        values = rep_scores[-12:]
        reps = [f"R{i}" for i in range(len(rep_scores) - len(values) + 1, len(rep_scores) + 1)]
    else:
        reps = ["R1", "R2", "R3", "R4", "R5", "R6"]
        base = 70 if start_session else 55
        values = [base + random.randint(-8, 8) for _ in reps]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=reps,
            y=values,
            marker=dict(color="#FFD400"),
            opacity=0.9,
        )
    )
    fig.update_layout(**_chart_layout())
    fig.update_yaxes(range=[0, 100])
    return fig


def _gauge_chart(value: int) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"font": {"size": 28, "color": "#FFD400"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "rgba(107,114,128,0.6)"},
                "bar": {"color": "#FFD400"},
                "bgcolor": "#FFFFFF",
                "bordercolor": "#FFE066",
                "steps": [
                    {"range": [0, 60], "color": "rgba(255,224,102,0.35)"},
                    {"range": [60, 85], "color": "rgba(255,212,0,0.35)"},
                ],
                "threshold": {"line": {"color": "#22c55e", "width": 3}, "value": 90},
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#111827", family="'Space Grotesk', sans-serif"),
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


def _metric_card(label: str, value: int) -> None:
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="mini-label">{label}</div>
            <div class="mini-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
