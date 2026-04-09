import streamlit as st
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CrowdAI Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Credentials (hardcoded as requested) ─────────────────────────────────────
USERS = {
    "admin": {"password": "admin123", "role": "Administrator"},
}

# ── Session state init ────────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

  /* Dark background */
  .stApp { background: #0a0e1a; color: #e2e8f0; }

  /* Sidebar */
  [data-testid="stSidebar"] { background: #0d1220 !important; border-right: 1px solid #1e2d4a; }

  /* Hide default header */
  header[data-testid="stHeader"] { background: transparent; }

  /* Cards */
  .card {
    background: linear-gradient(135deg, #0f1729 0%, #111d35 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
  }
  .metric-card {
    background: linear-gradient(135deg, #0f1729 0%, #0e2040 100%);
    border: 1px solid #1e4a7a;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    box-shadow: 0 0 20px rgba(59,130,246,0.08);
  }
  .metric-card:hover { border-color: #3b82f6; box-shadow: 0 0 30px rgba(59,130,246,0.2); }
  .metric-value { font-size: 2.4rem; font-weight: 800; color: #38bdf8; font-family: 'JetBrains Mono', monospace; }
  .metric-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 2px; margin-top: 0.3rem; }
  .metric-delta { font-size: 0.8rem; color: #22c55e; margin-top: 0.2rem; }

  /* Login box */
  .login-container {
    max-width: 420px;
    margin: 8vh auto 0;
    background: linear-gradient(135deg, #0f1729 0%, #111d35 100%);
    border: 1px solid #1e3a5f;
    border-radius: 20px;
    padding: 3rem 2.5rem;
    box-shadow: 0 20px 60px rgba(0,0,0,0.6), 0 0 80px rgba(59,130,246,0.06);
  }
  .login-title {
    font-size: 1.8rem;
    font-weight: 800;
    color: #f1f5f9;
    text-align: center;
    margin-bottom: 0.3rem;
  }
  .login-sub { font-size: 0.85rem; color: #475569; text-align: center; margin-bottom: 2rem; letter-spacing: 1px; }
  .logo-text {
    text-align: center;
    font-size: 3rem;
    margin-bottom: 1rem;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2rem !important;
    font-weight: 700 !important;
    font-family: 'Syne', sans-serif !important;
    letter-spacing: 1px !important;
    width: 100% !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(37,99,235,0.3) !important;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    box-shadow: 0 6px 28px rgba(59,130,246,0.45) !important;
    transform: translateY(-1px) !important;
  }

  /* Inputs */
  .stTextInput > div > div > input {
    background: #0a0e1a !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
  }
  .stTextInput > div > div > input:focus { border-color: #3b82f6 !important; }

  /* Alert */
  .alert-box {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    color: #fca5a5;
    font-size: 0.85rem;
    margin-top: 0.5rem;
  }
  .success-box {
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    color: #86efac;
    font-size: 0.85rem;
  }

  /* Badge */
  .badge {
    display: inline-block;
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.3);
    color: #60a5fa;
    border-radius: 50px;
    padding: 0.2rem 0.8rem;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
  }
  .badge-green {
    background: rgba(34,197,94,0.15);
    border-color: rgba(34,197,94,0.3);
    color: #86efac;
  }
  .badge-red {
    background: rgba(239,68,68,0.15);
    border-color: rgba(239,68,68,0.3);
    color: #fca5a5;
  }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] { background: #0d1220; border-radius: 10px; padding: 4px; }
  .stTabs [data-baseweb="tab"] { color: #64748b; border-radius: 8px; font-family: 'Syne', sans-serif; font-weight: 600; }
  .stTabs [aria-selected="true"] { background: #1e3a5f !important; color: #38bdf8 !important; }

  /* Section header */
  .section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2d4a;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0a0e1a; }
  ::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }

  /* Progress bar */
  .stProgress > div > div > div { background: linear-gradient(90deg, #1d4ed8, #38bdf8) !important; }

  /* Status dot */
  .status-live { display: inline-block; width: 8px; height: 8px; background: #22c55e; border-radius: 50%; margin-right: 6px; animation: pulse 1.5s infinite; }
  @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(1.2)} }

  /* Selectbox */
  .stSelectbox > div > div { background: #0f1729 !important; border-color: #1e3a5f !important; color: #e2e8f0 !important; }

  /* Slider */
  .stSlider > div > div > div { background: #1e3a5f !important; }
  .stSlider > div > div > div > div { background: #3b82f6 !important; }

  /* File uploader */
  .stFileUploader > div { background: #0f1729; border: 2px dashed #1e3a5f; border-radius: 12px; }
  .stFileUploader > div:hover { border-color: #3b82f6; }

  /* DataFrame */
  .stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  LOGIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
def login_page():
    st.markdown("""
    <div class="login-container">
      <div class="logo-text">🧠</div>
      <div class="login-title">CrowdAI Monitor</div>
      <div class="login-sub">AI-BASED CROWD MOVEMENT ANALYSIS</div>
    </div>
    """, unsafe_allow_html=True)

    # Centre the form
    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        username = st.text_input("Username", placeholder="admin", key="login_user")
        password = st.text_input("Password", type="password", placeholder="••••••••", key="login_pass")
        st.markdown("<br>", unsafe_allow_html=True)
        login_btn = st.button("LOGIN →", key="login_btn")

        if login_btn:
            if username in USERS and USERS[username]["password"] == password:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.role = USERS[username]["role"]
                st.rerun()
            else:
                st.markdown('<div class="alert-box">⚠ Invalid username or password</div>', unsafe_allow_html=True)

        st.markdown("""
        <br>
        <div style="text-align:center; color:#334155; font-size:0.75rem;">
          Default credentials &nbsp;|&nbsp; admin / admin123
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ROUTER
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.authenticated:
    login_page()
else:
    # Import dashboard only after login (avoids heavy imports at startup)
    from dashboard import render_dashboard
    render_dashboard()
