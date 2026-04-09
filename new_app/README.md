# 🧠 CrowdAI Monitor – Setup Guide

AI-Based Crowd Movement Analysis Dashboard built with **Streamlit + YOLOv8 + OpenCV**.

---

## 📁 Project Structure

```
crowd_dashboard/
├── app.py              ← Entry point (login + routing)
├── dashboard.py        ← All dashboard pages
├── requirements.txt    ← Python dependencies
├── .streamlit/
│   └── config.toml    ← Dark theme config
└── README.md
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

> For GPU support (optional):
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. Open in browser
```
http://localhost:8501
```

---

## 🔐 Login Credentials

| Username | Password   | Role          |
|----------|------------|---------------|
| admin    | admin123   | Administrator |

> To add more users, edit the `USERS` dict in `app.py`.

---

## 🎥 Using Live Analysis

1. Go to **🎥 Live Analysis** tab
2. Choose input source:
   - **Upload Video File** – upload your `campus_feed.mp4`
   - **CCTV Stream** – enter RTSP URL
   - **Webcam** – uses camera index 0
3. Adjust confidence threshold
4. Click **▶ Start Analysis**

> Without YOLO installed, the app runs in **Simulation Mode** — it generates synthetic detections so you can demo the full interface.

---

## 📊 Features

| Feature | Description |
|--------|-------------|
| 🔐 Login | Admin auth with username/password |
| 🏠 Dashboard | Live KPIs, charts, entry/exit flow |
| 🎥 Live Analysis | Real-time YOLO detection + tracking |
| 🔥 Heatmap | 2D + 3D crowd density visualization |
| 📊 Analytics | Filterable data, CSV export |
| ⚙️ Settings | Model config, alerts, system info |

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| UI Framework | Streamlit |
| AI Detection | Ultralytics YOLOv8 |
| Tracking | BoT-SORT / ByteTrack |
| Vision | OpenCV |
| Charts | Plotly |
| Data | Pandas, NumPy |

---

## 📋 Auto-Generated Files

The app auto-creates these on first run:
- `analytics_log.csv` – timestamped detection data
- `heatmap_data.npy` – accumulated heatmap array

---

## 🚀 Deployment

```bash
# Deploy to Streamlit Cloud
# 1. Push to GitHub
# 2. Go to share.streamlit.io
# 3. Connect your repo
# 4. Set main file: app.py
```
