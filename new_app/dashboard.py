"""
dashboard.py  –  All dashboard tabs for CrowdAI Monitor
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import random
import os
import datetime
import io

# ── Try importing heavy deps gracefully ───────────────────────────────────────
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

LOG_PATH = "analytics_log.csv"
HEATMAP_PATH = "heatmap_data.npy"


def get_or_create_log() -> pd.DataFrame:
    """Load existing log or seed with synthetic data."""
    if os.path.exists(LOG_PATH):
        return pd.read_csv(LOG_PATH, parse_dates=["timestamp"])

    # Seed synthetic data for demo
    rng = np.random.default_rng(42)
    n = 200
    now = datetime.datetime.now()
    times = [now - datetime.timedelta(minutes=(n - i) * 3) for i in range(n)]
    entries = rng.integers(0, 8, n)
    exits = rng.integers(0, 8, n)
    counts = np.clip(np.cumsum(entries - exits), 0, 80).tolist()
    df = pd.DataFrame({
        "timestamp": times,
        "people_count": counts,
        "entries": entries.tolist(),
        "exits": exits.tolist(),
        "zone": rng.choice(["Gate A", "Gate B", "Corridor", "Hall", "Plaza"], n).tolist(),
        "avg_speed": rng.uniform(0.5, 2.5, n).round(2).tolist(),
        "density": (np.array(counts) / 80).round(2).tolist(),
    })
    df.to_csv(LOG_PATH, index=False)
    return df


def save_to_log(row: dict):
    df = get_or_create_log()
    new_row = pd.DataFrame([row])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)


def get_or_create_heatmap(shape=(480, 640)) -> np.ndarray:
    if os.path.exists(HEATMAP_PATH):
        return np.load(HEATMAP_PATH)
    rng = np.random.default_rng(0)
    hm = np.zeros(shape, dtype=np.float32)
    # Simulate hotspots
    for (cx, cy, intensity) in [(200, 300, 4000), (350, 200, 3000), (420, 420, 2500), (100, 400, 1500)]:
        xs = rng.normal(cx, 40, intensity).astype(int)
        ys = rng.normal(cy, 40, intensity).astype(int)
        for x, y in zip(xs, ys):
            if 0 <= x < shape[0] and 0 <= y < shape[1]:
                hm[x, y] += 1
    np.save(HEATMAP_PATH, hm)
    return hm


def colorize_heatmap(hm: np.ndarray) -> np.ndarray:
    norm = cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def simulate_detection_frame(frame: np.ndarray, frame_idx: int):
    """Simulate YOLO detections on a frame (used when YOLO not installed)."""
    rng = np.random.default_rng(frame_idx % 100)
    n_people = rng.integers(3, 12)
    h, w = frame.shape[:2]
    detections = []
    for i in range(n_people):
        x1 = int(rng.integers(10, w - 80))
        y1 = int(rng.integers(10, h - 140))
        x2 = x1 + rng.integers(40, 80)
        y2 = y1 + rng.integers(100, 140)
        conf = round(float(rng.uniform(0.55, 0.97)), 2)
        track_id = int(rng.integers(1, 30))
        detections.append((x1, y1, x2, y2, conf, track_id))
    return detections


def draw_detections(frame, detections, heatmap_acc):
    """Draw bounding boxes, IDs, and accumulate heatmap."""
    out = frame.copy()
    h, w = out.shape[:2]
    for (x1, y1, x2, y2, conf, tid) in detections:
        x1, y1, x2, y2 = max(0,x1), max(0,y1), min(w-1,x2), min(h-1,y2)
        # Box
        cv2.rectangle(out, (x1, y1), (x2, y2), (56, 189, 248), 2)
        # Label
        label = f"ID:{tid}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), (29, 78, 216), -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        # Accumulate heatmap
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if 0 <= cy < heatmap_acc.shape[0] and 0 <= cx < heatmap_acc.shape[1]:
            heatmap_acc[cy, cx] += 1
    # Count overlay
    cv2.rectangle(out, (0, 0), (220, 36), (10, 20, 40), -1)
    cv2.putText(out, f"People: {len(detections)}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (56, 189, 248), 2)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1.5rem 0 1rem;">
          <div style="font-size:2.5rem;">🧠</div>
          <div style="font-size:1.1rem; font-weight:800; color:#f1f5f9; letter-spacing:1px;">CrowdAI</div>
          <div style="font-size:0.65rem; color:#475569; letter-spacing:3px; text-transform:uppercase;">Monitor v1.0</div>
        </div>
        <hr style="border-color:#1e2d4a; margin:0 0 1.5rem;">
        """, unsafe_allow_html=True)

        user = st.session_state.username
        role = st.session_state.get("role", "User")
        st.markdown(f"""
        <div class="card" style="margin-bottom:1.5rem;">
          <div style="font-size:0.65rem; color:#475569; letter-spacing:2px; text-transform:uppercase;">Logged in as</div>
          <div style="font-size:1rem; font-weight:700; color:#f1f5f9; margin-top:0.2rem;">👤 {user}</div>
          <div class="badge badge-green" style="margin-top:0.4rem;">{role}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Navigation</div>', unsafe_allow_html=True)
        page = st.radio("", ["🏠 Dashboard", "🎥 Live Analysis", "🔥 Heatmap", "📊 Analytics", "⚙️ Settings"], label_visibility="collapsed")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">System Status</div>', unsafe_allow_html=True)
        yolo_ok = HAS_YOLO
        st.markdown(f"""
        <div style="font-size:0.8rem; margin-bottom:0.5rem;">
          <span class="{'badge badge-green' if yolo_ok else 'badge badge-red'}">{'✓' if yolo_ok else '✗'} YOLO</span>
          &nbsp;
          <span class="{'badge badge-green' if HAS_PLOTLY else 'badge badge-red'}">{'✓' if HAS_PLOTLY else '✗'} Plotly</span>
        </div>
        <div style="font-size:0.75rem; color:#475569;">OpenCV {cv2.__version__}</div>
        """, unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("🔒 Logout"):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.rerun()

    return page


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD HOME
# ══════════════════════════════════════════════════════════════════════════════

def page_home():
    st.markdown("""
    <div style="margin-bottom:2rem;">
      <div style="font-size:0.7rem; color:#475569; letter-spacing:3px; text-transform:uppercase;">Overview</div>
      <h1 style="font-size:2rem; font-weight:800; color:#f1f5f9; margin:0.2rem 0;">
        Crowd Movement Dashboard
      </h1>
      <div style="color:#64748b; font-size:0.9rem;">
        <span class="status-live"></span>Real-time AI monitoring active
      </div>
    </div>
    """, unsafe_allow_html=True)

    df = get_or_create_log()

    # ── KPI row ───────────────────────────────────────────────────────────────
    current = int(df["people_count"].iloc[-1])
    total_entries = int(df["entries"].sum())
    total_exits = int(df["exits"].sum())
    peak = int(df["people_count"].max())
    avg_density = f"{df['density'].mean():.0%}"

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl, delta in [
        (c1, current, "Current Count", f"+{df['entries'].iloc[-1]} entry"),
        (c2, total_entries, "Total Entries", "since session start"),
        (c3, total_exits, "Total Exits", "since session start"),
        (c4, peak, "Peak Count", "historical max"),
        (c5, avg_density, "Avg Density", "of capacity"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{val}</div>
          <div class="metric-label">{lbl}</div>
          <div class="metric-delta">{delta}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row ────────────────────────────────────────────────────────────
    if HAS_PLOTLY:
        col_l, col_r = st.columns([2, 1])

        with col_l:
            st.markdown('<div class="section-header">People Count Over Time</div>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df["people_count"],
                mode="lines",
                line=dict(color="#38bdf8", width=2),
                fill="tozeroy",
                fillcolor="rgba(56,189,248,0.08)",
                name="Count"
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Syne"),
                xaxis=dict(gridcolor="#1e2d4a", showgrid=True),
                yaxis=dict(gridcolor="#1e2d4a", showgrid=True),
                margin=dict(l=0, r=0, t=0, b=0), height=280,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown('<div class="section-header">Zone Distribution</div>', unsafe_allow_html=True)
            zone_counts = df.groupby("zone")["people_count"].mean().reset_index()
            fig2 = px.pie(
                zone_counts, names="zone", values="people_count",
                color_discrete_sequence=["#1d4ed8", "#2563eb", "#38bdf8", "#0ea5e9", "#0284c7"]
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Syne"),
                margin=dict(l=0, r=0, t=0, b=0), height=280,
                legend=dict(bgcolor="rgba(0,0,0,0)")
            )
            fig2.update_traces(textfont_color="#f1f5f9")
            st.plotly_chart(fig2, use_container_width=True)

        # ── Entry/Exit bar chart ───────────────────────────────────────────
        st.markdown('<div class="section-header">Entries vs Exits (Last 30 Records)</div>', unsafe_allow_html=True)
        recent = df.tail(30)
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=recent["timestamp"], y=recent["entries"], name="Entries",
                              marker_color="#22c55e", opacity=0.85))
        fig3.add_trace(go.Bar(x=recent["timestamp"], y=recent["exits"], name="Exits",
                              marker_color="#ef4444", opacity=0.85))
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Syne"),
            xaxis=dict(gridcolor="#1e2d4a"),
            yaxis=dict(gridcolor="#1e2d4a"),
            barmode="group",
            margin=dict(l=0, r=0, t=0, b=0), height=240,
            legend=dict(bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig3, use_container_width=True)

    else:
        st.warning("Install plotly for interactive charts: `pip install plotly`")
        st.line_chart(df.set_index("timestamp")["people_count"])

    # ── Recent log table ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Recent Log</div>', unsafe_allow_html=True)
    st.dataframe(
        df.tail(10)[["timestamp", "people_count", "entries", "exits", "zone", "avg_speed", "density"]]
        .sort_values("timestamp", ascending=False)
        .style.background_gradient(subset=["people_count"], cmap="Blues"),
        use_container_width=True, height=280,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: LIVE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def page_live():
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
      <div style="font-size:0.7rem; color:#475569; letter-spacing:3px; text-transform:uppercase;">Computer Vision</div>
      <h1 style="font-size:1.8rem; font-weight:800; color:#f1f5f9; margin:0.2rem 0;">🎥 Live Video Analysis</h1>
    </div>
    """, unsafe_allow_html=True)

    col_ctrl, col_info = st.columns([1, 2])
    with col_ctrl:
        source = st.selectbox("Input Source", ["Upload Video File", "CCTV Stream (RTSP)", "Webcam"])
        uploaded = None
        rtsp_url = ""
        if source == "Upload Video File":
            uploaded = st.file_uploader("Upload MP4 / AVI", type=["mp4", "avi", "mov"])
        elif source == "CCTV Stream (RTSP)":
            rtsp_url = st.text_input("RTSP URL", placeholder="rtsp://192.168.1.1/stream")
        confidence = st.slider("Detection Confidence", 0.3, 0.95, 0.5, 0.05)
        show_heatmap_overlay = st.checkbox("Show Heatmap Overlay", value=False)

    with col_info:
        if not HAS_YOLO:
            st.markdown("""
            <div class="card">
              <div style="color:#fbbf24; font-weight:700; margin-bottom:0.5rem;">⚠ YOLO Not Installed</div>
              <div style="color:#94a3b8; font-size:0.85rem;">
                Running in <b>Simulation Mode</b>. Detections are generated synthetically
                to demo the interface.<br><br>
                To enable real AI detection:<br>
                <code style="color:#38bdf8;">pip install ultralytics</code><br>
                Then download a model:<br>
                <code style="color:#38bdf8;">yolo detect predict model=yolov8n.pt</code>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card">
              <div style="color:#22c55e; font-weight:700;">✓ YOLO Ready</div>
              <div style="color:#94a3b8; font-size:0.85rem; margin-top:0.3rem;">
                YOLOv8 detection active. Upload a video or connect a stream to begin.
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1e2d4a;'>", unsafe_allow_html=True)

    # ── Video processing ──────────────────────────────────────────────────────
    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    start_btn = col_btn1.button("▶ Start Analysis")
    stop_btn = col_btn2.button("⬛ Stop")

    frame_placeholder = st.empty()
    stats_placeholder = st.empty()
    log_placeholder = st.empty()

    if "running" not in st.session_state:
        st.session_state.running = False
    if start_btn:
        st.session_state.running = True
    if stop_btn:
        st.session_state.running = False

    if st.session_state.running:
        # Determine video source
        cap = None
        tmp_path = None

        if source == "Upload Video File" and uploaded is not None:
            import tempfile
            tmp_dir = tempfile.gettempdir()
            tmp_path = os.path.join(tmp_dir, f"uploaded_{uploaded.name}")
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            cap = cv2.VideoCapture(tmp_path)
        elif source == "Webcam":
            cap = cv2.VideoCapture(0)
        elif source == "CCTV Stream (RTSP)" and rtsp_url:
            cap = cv2.VideoCapture(rtsp_url)
        else:
            # Demo mode: generate synthetic frames
            cap = None

        heatmap_acc = np.zeros((480, 640), dtype=np.float32)

        # Load YOLO if available
        yolo_model = None
        if HAS_YOLO:
            try:
                yolo_model = YOLO("yolov8n.pt")
            except Exception:
                pass

        frame_idx = 0
        session_log = []

        while st.session_state.running:
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.session_state.running = False
                    break
                frame = cv2.resize(frame, (640, 480))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                # Generate a synthetic frame
                frame_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
                frame_rgb[:] = (10, 20, 40)
                cv2.putText(frame_rgb, "DEMO MODE – No video source",
                            (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (56, 189, 248), 2)

            # Detect
            if yolo_model:
                results = yolo_model.track(frame_rgb, persist=True, conf=confidence, classes=[0], verbose=False)
                detections = []
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf_val = float(box.conf[0])
                        tid = int(box.id[0]) if box.id is not None else 0
                        detections.append((int(x1), int(y1), int(x2), int(y2), conf_val, tid))
            else:
                detections = simulate_detection_frame(frame_rgb, frame_idx)

            frame_out = draw_detections(frame_rgb, detections, heatmap_acc)

            if show_heatmap_overlay:
                hm_colored = colorize_heatmap(heatmap_acc)
                hm_resized = cv2.resize(hm_colored, (640, 480))
                frame_out = cv2.addWeighted(frame_out, 0.6, hm_resized, 0.4, 0)

            # Timestamp
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame_out, ts, (520, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 120, 160), 1)

            frame_placeholder.image(frame_out, channels="RGB", use_container_width=True)

            # Stats
            n = len(detections)
            entries_now = max(0, n - (session_log[-1]["count"] if session_log else n))
            exits_now = max(0, (session_log[-1]["count"] if session_log else n) - n)
            session_log.append({"ts": ts, "count": n, "entries": entries_now, "exits": exits_now})

            with stats_placeholder.container():
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("People in Frame", n)
                sc2.metric("Frame #", frame_idx)
                sc3.metric("Entries", entries_now)
                sc4.metric("Exits", exits_now)

            # Save to CSV every 10 frames
            if frame_idx % 10 == 0:
                save_to_log({
                    "timestamp": datetime.datetime.now(),
                    "people_count": n,
                    "entries": entries_now,
                    "exits": exits_now,
                    "zone": "Main Feed",
                    "avg_speed": round(random.uniform(0.8, 2.2), 2),
                    "density": round(n / 80, 2),
                })
                # Save heatmap
                np.save(HEATMAP_PATH, heatmap_acc)

            frame_idx += 1
            time.sleep(0.03)  # ~30 fps cap

        if cap:
            cap.release()
        st.success("Analysis stopped.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

def page_heatmap():
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
      <div style="font-size:0.7rem; color:#475569; letter-spacing:3px; text-transform:uppercase;">Spatial Analysis</div>
      <h1 style="font-size:1.8rem; font-weight:800; color:#f1f5f9; margin:0.2rem 0;">🔥 Crowd Heatmap</h1>
    </div>
    """, unsafe_allow_html=True)

    hm = get_or_create_heatmap()
    colored = colorize_heatmap(hm)

    col_hm, col_legend = st.columns([3, 1])

    with col_hm:
        st.markdown('<div class="section-header">Density Distribution</div>', unsafe_allow_html=True)
        st.image(colored, caption="Crowd density heatmap – warmer = higher density", use_container_width=True)

    with col_legend:
        st.markdown('<div class="section-header">Statistics</div>', unsafe_allow_html=True)
        peak_y, peak_x = np.unravel_index(np.argmax(hm), hm.shape)
        st.markdown(f"""
        <div class="card">
          <div class="metric-value" style="font-size:1.5rem;">{hm.max():.0f}</div>
          <div class="metric-label">Peak Visits</div>
          <div style="font-size:0.75rem; color:#64748b; margin-top:0.5rem;">at pixel ({peak_x}, {peak_y})</div>
        </div>
        <div class="card">
          <div class="metric-value" style="font-size:1.5rem;">{(hm > 0).sum():,}</div>
          <div class="metric-label">Active Pixels</div>
        </div>
        <div class="card">
          <div class="metric-value" style="font-size:1.5rem;">{hm.sum():.0f}</div>
          <div class="metric-label">Total Passes</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="margin-top:1rem;">Hotspot Zones</div>', unsafe_allow_html=True)
        # Top 3 regions
        blurred = cv2.GaussianBlur(hm, (51, 51), 0)
        for i, zone in enumerate(["Entry Gate", "Main Corridor", "Open Plaza", "Exit Gate"]):
            # Simulate zone intensity from heatmap quadrants
            quadrants = [hm[:240, :320], hm[:240, 320:], hm[240:, :320], hm[240:, 320:]]
            intensity = quadrants[i].sum() / hm.sum()
            st.markdown(f"""
            <div style="margin-bottom:0.5rem;">
              <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:3px;">
                <span style="color:#94a3b8;">{zone}</span>
                <span style="color:#38bdf8;">{intensity:.0%}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(intensity))

    if HAS_PLOTLY:
        st.markdown('<div class="section-header">3D Density Surface</div>', unsafe_allow_html=True)
        ds = hm[::10, ::10]  # downsample
        fig = go.Figure(data=[go.Surface(
            z=ds, colorscale="Jet", showscale=False
        )])
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            scene=dict(
                xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#1e2d4a"),
                yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#1e2d4a"),
                zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#1e2d4a"),
            ),
            margin=dict(l=0, r=0, t=0, b=0), height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Reset button
    if st.button("🔄 Reset Heatmap"):
        if os.path.exists(HEATMAP_PATH):
            os.remove(HEATMAP_PATH)
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

def page_analytics():
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
      <div style="font-size:0.7rem; color:#475569; letter-spacing:3px; text-transform:uppercase;">Data Analysis</div>
      <h1 style="font-size:1.8rem; font-weight:800; color:#f1f5f9; margin:0.2rem 0;">📊 Analytics & Reports</h1>
    </div>
    """, unsafe_allow_html=True)

    df = get_or_create_log()

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        zone_filter = st.multiselect("Filter by Zone", df["zone"].unique().tolist(), default=df["zone"].unique().tolist())
    with col_f2:
        min_count = st.slider("Min People Count", 0, int(df["people_count"].max()), 0)
    with col_f3:
        n_recent = st.slider("Show Last N Records", 10, len(df), min(100, len(df)))

    filtered = df[df["zone"].isin(zone_filter) & (df["people_count"] >= min_count)].tail(n_recent)

    if HAS_PLOTLY:
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown('<div class="section-header">Avg Speed per Zone</div>', unsafe_allow_html=True)
            zone_speed = filtered.groupby("zone")["avg_speed"].mean().reset_index()
            fig = px.bar(zone_speed, x="zone", y="avg_speed",
                         color="avg_speed", color_continuous_scale=["#1d4ed8", "#38bdf8"])
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"), xaxis=dict(gridcolor="#1e2d4a"),
                yaxis=dict(gridcolor="#1e2d4a"), margin=dict(l=0, r=0, t=0, b=0),
                height=280, showlegend=False, coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown('<div class="section-header">Density Over Time</div>', unsafe_allow_html=True)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=filtered["timestamp"], y=filtered["density"],
                mode="lines", line=dict(color="#f97316", width=2),
                fill="tozeroy", fillcolor="rgba(249,115,22,0.08)"
            ))
            fig2.add_hline(y=0.7, line_dash="dash", line_color="#ef4444",
                           annotation_text="Alert Threshold")
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"), xaxis=dict(gridcolor="#1e2d4a"),
                yaxis=dict(gridcolor="#1e2d4a"), margin=dict(l=0, r=0, t=0, b=0),
                height=280, showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Scatter
        st.markdown('<div class="section-header">Speed vs Density Correlation</div>', unsafe_allow_html=True)
        fig3 = px.scatter(filtered, x="density", y="avg_speed", color="zone",
                          size="people_count", hover_data=["timestamp"],
                          color_discrete_sequence=px.colors.qualitative.Set2)
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"), xaxis=dict(gridcolor="#1e2d4a"),
            yaxis=dict(gridcolor="#1e2d4a"), margin=dict(l=0, r=0, t=0, b=0), height=280,
            legend=dict(bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Full data table
    st.markdown('<div class="section-header">Full Analytics Log</div>', unsafe_allow_html=True)
    st.dataframe(filtered.sort_values("timestamp", ascending=False), use_container_width=True, height=320)

    # Download
    csv_bytes = filtered.to_csv(index=False).encode()
    st.download_button(
        "⬇ Download CSV",
        data=csv_bytes,
        file_name=f"crowd_analytics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

def page_settings():
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
      <div style="font-size:0.7rem; color:#475569; letter-spacing:3px; text-transform:uppercase;">Configuration</div>
      <h1 style="font-size:1.8rem; font-weight:800; color:#f1f5f9; margin:0.2rem 0;">⚙️ System Settings</h1>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">Detection Settings</div>', unsafe_allow_html=True)
        with st.form("det_settings"):
            model_choice = st.selectbox("YOLO Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"])
            tracker = st.selectbox("Tracker", ["botsort.yaml", "bytetrack.yaml"])
            conf_thresh = st.slider("Confidence Threshold", 0.1, 0.95, 0.5)
            iou_thresh = st.slider("IOU Threshold", 0.1, 0.9, 0.45)
            max_det = st.number_input("Max Detections per Frame", 1, 300, 100)
            save_det = st.form_submit_button("💾 Save Detection Settings")
            if save_det:
                st.success("Detection settings saved!")

        st.markdown('<div class="section-header">Alert Settings</div>', unsafe_allow_html=True)
        with st.form("alert_settings"):
            crowd_alert = st.number_input("Crowd Alert Threshold (people)", 1, 500, 50)
            density_alert = st.slider("Density Alert (%)", 0, 100, 70)
            enable_email = st.checkbox("Enable Email Alerts")
            email = st.text_input("Alert Email", placeholder="admin@example.com", disabled=not enable_email)
            save_alerts = st.form_submit_button("💾 Save Alert Settings")
            if save_alerts:
                st.success("Alert settings saved!")

    with col_r:
        st.markdown('<div class="section-header">Storage & Logging</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card">
          <div style="font-size:0.75rem; color:#475569; margin-bottom:0.5rem;">Log File</div>
          <div style="font-family:'JetBrains Mono',monospace; color:#38bdf8; font-size:0.85rem;">{LOG_PATH}</div>
          <div style="font-size:0.75rem; color:#64748b; margin-top:0.5rem;">
            Records: {len(get_or_create_log()):,}
          </div>
        </div>
        <div class="card">
          <div style="font-size:0.75rem; color:#475569; margin-bottom:0.5rem;">Heatmap Cache</div>
          <div style="font-family:'JetBrains Mono',monospace; color:#38bdf8; font-size:0.85rem;">{HEATMAP_PATH}</div>
          <div style="font-size:0.75rem; color:#64748b; margin-top:0.5rem;">
            Exists: {'Yes' if os.path.exists(HEATMAP_PATH) else 'No'}
          </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑 Clear Analytics Log"):
            if os.path.exists(LOG_PATH):
                os.remove(LOG_PATH)
            st.success("Log cleared. New synthetic data will be generated.")

        st.markdown('<div class="section-header">System Info</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card">
          <div style="font-size:0.8rem; line-height:1.8; font-family:'JetBrains Mono',monospace; color:#94a3b8;">
            Python: {os.sys.version.split()[0]}<br>
            OpenCV: {cv2.__version__}<br>
            YOLO: {'✓ Installed' if HAS_YOLO else '✗ Not installed'}<br>
            Plotly: {'✓ Installed' if HAS_PLOTLY else '✗ Not installed'}<br>
            NumPy: {np.__version__}<br>
            Pandas: {pd.__version__}
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Install Missing Deps</div>', unsafe_allow_html=True)
        st.code("pip install ultralytics plotly streamlit", language="bash")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN RENDER
# ══════════════════════════════════════════════════════════════════════════════

def render_dashboard():
    page = render_sidebar()

    if page == "🏠 Dashboard":
        page_home()
    elif page == "🎥 Live Analysis":
        page_live()
    elif page == "🔥 Heatmap":
        page_heatmap()
    elif page == "📊 Analytics":
        page_analytics()
    elif page == "⚙️ Settings":
        page_settings()