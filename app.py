"""DEPTHMAP AI — Streamlit entry point.

Run:  streamlit run app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="DEPTHMAP AI · Edge-Preserving Depth Upsampling",
                   page_icon=str(Path(__file__).parent / "assets" / "logo.png"),
                   layout="wide", initial_sidebar_state="collapsed")

from src.ui import init_state, inject_css  # noqa: E402

inject_css()
init_state()

pages = [
    st.Page("views/home.py", title="Home", icon=":material/blur_on:", default=True),
    st.Page("views/dashboard.py", title="Dashboard", icon=":material/space_dashboard:"),
    st.Page("views/reconstruction.py", title="Reconstruction Studio", icon=":material/auto_fix_high:"),
    st.Page("views/comparison.py", title="Comparison Lab", icon=":material/compare:"),
    st.Page("views/analytics.py", title="Analytics", icon=":material/monitoring:"),
    st.Page("views/model_info.py", title="Model", icon=":material/neurology:"),
    st.Page("views/about.py", title="About", icon=":material/info:"),
]
st.navigation(pages, position="top").run()
