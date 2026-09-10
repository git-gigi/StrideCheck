import streamlit as st
import tempfile
import os
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.pipeline import AnalysisPipeline
from src.report_generator import ReportGenerator
from src.video_annotator import VideoAnnotator
from src.config import THRESHOLDS

# Page Configuration
st.set_page_config(
    page_title="StrideCheck — Running Form Analyzer",
    page_icon="🏃",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    /* Score card styling */
    .score-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin-bottom: 1rem;
    }
    .score-card .score-value {
        font-size: 4rem;
        font-weight: 800;
        line-height: 1;
        margin: 0.5rem 0;
    }
    .score-card .score-label {
        font-size: 1rem;
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .score-green { color: #00e676; }
    .score-yellow { color: #ffea00; }
    .score-red { color: #ff1744; }

    /* Error card */
    .error-card {
        background: #1e1e2f;
        border-left: 4px solid #ff1744;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        color: #e0e0e0;
    }
    .error-card h4 {
        color: #ff8a80;
        margin: 0 0 0.5rem 0;
    }
    .error-card .suggestion {
        color: #80cbc4;
        font-style: italic;
    }

    /* Metric highlight */
    div[data-testid="stMetric"] {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/running--v1.png", width=64)
    st.title("StrideCheck")
    st.caption("AI Running Form Analyzer")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload a running video",
        type=["mp4", "mov"],
        help="Upload a video of a runner (side view recommended)",
    )

    analyze_button = st.button(
        "🔍 Analyze",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    )

    st.divider()
    st.caption("StrideCheck v0.1 · Built with MediaPipe & OpenCV")

# Helper Functions

def get_score_color_class(score: float) -> str:
    """Returns CSS class name based on score threshold."""
    if score >= 0.8:
        return "score-green"
    elif score >= 0.6:
        return "score-yellow"
    return "score-red"


def render_score_card(score: float, total_cycles: int, clean_cycles: int):
    """Renders the main score card with overall metrics."""
    percentage = int(score * 100)
    color_class = get_score_color_class(score)
    st.markdown(f"""
    <div class="score-card">
        <div class="score-label">Running Form Score</div>
        <div class="score-value {color_class}">{percentage}</div>
        <div class="score-label">{clean_cycles} / {total_cycles} clean cycles</div>
    </div>
    """, unsafe_allow_html=True)


def render_error_card(error_type: str, count: int, suggestion: str | None = None):
    """Renders an error summary card."""
    suggestion_html = ""
    if suggestion:
        suggestion_html = f'<p class="suggestion">💡 {suggestion}</p>'
    st.markdown(f"""
    <div class="error-card">
        <h4>⚠️ {error_type.replace("_", " ").title()}</h4>
        <p>Detected in <strong>{count}</strong> cycle{"s" if count > 1 else ""}</p>
        {suggestion_html}
    </div>
    """, unsafe_allow_html=True)


def build_knee_angle_chart(angles, cycles, cycle_results, fps):
    time = np.arange(len(angles["knee_l"])) / fps

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time, y=angles["knee_l"], name="Left Knee",
        line=dict(color="#00bcd4", width=1.5), opacity=0.8,
    ))
    fig.add_trace(go.Scatter(
        x=time, y=angles["knee_r"], name="Right Knee",
        line=dict(color="#ff9800", width=1.5), opacity=0.8,
    ))

    for cycle, cr in zip(cycles, cycle_results):
        t0 = cycle.start_frame / fps
        t1 = cycle.end_frame / fps
        color = "rgba(255,23,68,0.1)" if cr.errors else "rgba(0,230,118,0.05)"
        fig.add_vrect(x0=t0, x1=t1, fillcolor=color, line_width=0)

    fig.update_layout(
        title="Knee Angle Over Time",
        xaxis_title="Time (s)", yaxis_title="Angle (°)",
        template="plotly_dark", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def build_trunk_lean_chart(angles, cycles, cycle_results, fps):
    time = np.arange(len(angles["trunk_lean"])) / fps

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time, y=angles["trunk_lean"], name="Trunk Lean",
        line=dict(color="#7c4dff", width=1.5),
    ))

    threshold = THRESHOLDS["trunk_lean_max"]
    fig.add_hline(
        y=threshold, line_dash="dash", line_color="#ff1744",
        annotation_text=f"Threshold ({threshold:.0f}°)",
        annotation_position="top left",
    )

    for cycle, cr in zip(cycles, cycle_results):
        t0 = cycle.start_frame / fps
        t1 = cycle.end_frame / fps
        color = "rgba(255,23,68,0.1)" if cr.errors else "rgba(0,230,118,0.05)"
        fig.add_vrect(x0=t0, x1=t1, fillcolor=color, line_width=0)

    fig.update_layout(
        title="Trunk Lean Over Time",
        xaxis_title="Time (s)", yaxis_title="Angle (°)",
        template="plotly_dark", height=400,
    )
    return fig


# Main Area

if uploaded_file is None:
    # Landing page
    st.markdown("# 🏃 StrideCheck")
    st.markdown("### Analyze your running form with AI-powered biomechanics")
    st.info("👈 Upload a video from the sidebar to get started.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 📹 Upload")
        st.markdown("Record yourself running from a **side view** and upload the video.")
    with col2:
        st.markdown("#### 🔬 Analyze")
        st.markdown("Our pipeline detects **pose landmarks**, segments gait cycles, and evaluates your form.")
    with col3:
        st.markdown("#### 📊 Results")
        st.markdown("Get a detailed report with **score**, error breakdown, and corrective suggestions.")

elif analyze_button:
    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    annotated_video_path = None

    try:
        # Step 1: Run Pipeline
        with st.spinner("⚙️ Analyzing your running form..."):
            pipeline = AnalysisPipeline()
            result, cleaned_landmarks, metadata, angles, cycles, features = pipeline.run(temp_path)

        # Step 2: Generate Report
        report_gen = ReportGenerator()
        summary = report_gen.generate_summary(result)

        # Step 3: Extract Key Frames + Annotated Video
        with st.spinner("🎨 Generating annotated video..."):
            annotator = VideoAnnotator()
            key_frames = annotator.extract_key_frames(
                video_path=temp_path,
                landmarks=cleaned_landmarks,
                cycle_results=result.cycle_results,
            )

            annotated_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            annotated_video_path = annotated_tmp.name
            annotated_tmp.close()
            annotator.generate_annotated_video(
                video_path=temp_path,
                output_path=annotated_video_path,
                landmarks=cleaned_landmarks,
                cycles=cycles,
                cycle_results=result.cycle_results,
            )

        # Display Results
        st.markdown("---")

        # Score Section
        score_col, metrics_col = st.columns([1, 2])

        with score_col:
            render_score_card(summary.overall_score, summary.total_cycles, summary.clean_cycles)

        if result.cycle_results:
            avg_cadence = (summary.total_cycles / metadata.duration_sec) * 60 
        else:
            avg_cadence = 0

        with metrics_col:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Cycles", summary.total_cycles)
            m2.metric("Clean Cycles", summary.clean_cycles, delta=None)
            m3.metric("Issues Found", len(summary.error_counts))
            m4.metric("Average Cadence", f"{avg_cadence:.0f} spm")

        st.markdown("---")

        # Tabs for different views
        tab_errors, tab_keyframes, tab_charts, tab_cycles, tab_video = st.tabs(
            ["🔴 Error Breakdown", "🖼️ Key Frames", "📈 Joint Angles", "📋 Cycle Details", "📹 Video"]
        )

        # Tab 1: Error Breakdown
        with tab_errors:
            if not summary.error_counts:
                st.success("✅ No form errors detected! Your running form looks great.")
            else:
                st.subheader("Detected Issues")
                # Build a mapping from error_type to suggestion
                suggestion_map = {}
                for cycle in result.cycle_results:
                    for err in cycle.errors:
                        if err.error_type not in suggestion_map:
                            suggestion_map[err.error_type] = err.suggestion

                for error_type, count in summary.error_counts.items():
                    render_error_card(
                        error_type=error_type,
                        count=count,
                        suggestion=suggestion_map.get(error_type),
                    )

                if summary.recommendations:
                    st.subheader("💡 Recommendations")
                    for i, rec in enumerate(summary.recommendations, 1):
                        st.markdown(f"**{i}.** {rec}")

        # Tab 2: Key Frames
        with tab_keyframes:
            if not key_frames:
                st.info("No key frames to display (all cycles are clean).")
            else:
                st.subheader(f"Annotated Key Frames ({len(key_frames)} found)")
                for frame_idx, frame_img, error in key_frames:
                    col_img, col_info = st.columns([2, 1])
                    with col_img:
                        # OpenCV uses BGR, Streamlit expects RGB
                        frame_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
                        st.image(
                            frame_rgb,
                            caption=f"Frame #{frame_idx}",
                            use_container_width=True,
                        )
                    with col_info:
                        st.markdown(f"**Error:** {error.error_type.replace('_', ' ').title()}")
                        st.markdown(f"**Severity:** {error.severity}")
                        st.markdown(f"**Details:** {error.message}")
                        st.markdown(f"💡 *{error.suggestion}*")
                    st.divider()

        # Tab 3: Joint Angle Charts
        with tab_charts:
            st.subheader("Joint Angles Over Time")
            st.caption("Green regions = clean cycles · Red regions = cycles with errors")

            st.plotly_chart(
                build_knee_angle_chart(angles, cycles, result.cycle_results, metadata.fps),
                use_container_width=True,
            )
            st.plotly_chart(
                build_trunk_lean_chart(angles, cycles, result.cycle_results, metadata.fps),
                use_container_width=True,
            )

        # Tab 4: Cycle Details
        with tab_cycles:
            st.subheader("Per-Cycle Feature Breakdown")
            rows = []
            for i, (cycle, feat, cr) in enumerate(zip(cycles, features, result.cycle_results)):
                errors_str = ", ".join(e.error_type for e in cr.errors) or "—"
                rows.append({
                    "#": i + 1,
                    "Side": cycle.side.capitalize(),
                    "Cadence (spm)": f"{feat.cadence:.0f}",
                    "Knee Peak (°)": f"{feat.knee_peak:.1f}",
                    "Knee ROM (°)": f"{feat.knee_rom:.1f}",
                    "Trunk Lean (°)": f"{feat.trunk_lean_mean:.1f}",
                    "Vert. Osc.": f"{feat.vertical_oscillation:.3f}",
                    "Overstride": f"{feat.overstride_index:.3f}",
                    "Symmetry": f"{feat.stride_symmetry:.2f}",
                    "Label": cr.label.replace("_", " ").title(),
                    "Errors": errors_str,
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Tab 5: Video
        with tab_video:
            vid_col1, vid_col2 = st.columns(2)
            with vid_col1:
                st.subheader("Original")
                st.video(temp_path)
            with vid_col2:
                st.subheader("Annotated")
                if annotated_video_path and os.path.exists(annotated_video_path):
                    st.video(annotated_video_path)
                else:
                    st.warning("Annotated video could not be generated.")

            st.caption(
                f"Resolution: {metadata.width}×{metadata.height} · "
                f"FPS: {metadata.fps:.1f} · "
                f"Duration: {metadata.duration_sec:.1f}s · "
                f"Frames: {metadata.total_frames}"
            )

    finally:
        # Cleanup temp files
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        if annotated_video_path and os.path.exists(annotated_video_path):
            os.unlink(annotated_video_path)

else:
    # File uploaded but not yet analyzed
    st.markdown("# 🏃 StrideCheck")
    st.video(uploaded_file)
    st.info("👆 Video loaded! Click **Analyze** in the sidebar to start.")
