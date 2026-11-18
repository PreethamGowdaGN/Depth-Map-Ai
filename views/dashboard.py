"""Page 2 — Dashboard (real session activity only)."""
import pandas as pd
import streamlit as st

from config import METHOD_LABELS
from src.ui import footer, load_benchmark, page_header, section, stat

page_header("Overview", "Dashboard", "Your current session at a glance. Activity below is real — it is recorded only when you run something.")
ss = st.session_state
runs = ss.runs

c = st.columns(4)
with c[0]:
    stat("Processed this session", len(runs), "reconstructions + comparison runs", cyan=True)
with c[1]:
    stat("Selected method", METHOD_LABELS.get(ss.settings.get("method"), "—").split(" (")[0], "from the Studio")
with c[2]:
    stat("Selected scale", f"{ss.settings.get('scale', 4)}×", "upsampling factor")
with c[3]:
    stat("Status", ss.get("status", "Idle"), "last reconstruction")

st.write("")
q1, q2, q3 = st.columns(3)
if q1.button("Open Reconstruction Studio →", type="primary", width="stretch"):
    st.switch_page("views/reconstruction.py")
if q2.button("Compare all methods", width="stretch"):
    st.switch_page("views/comparison.py")
if q3.button("View analytics", width="stretch"):
    st.switch_page("views/analytics.py")

section("Recent activity", "This browser session only. Nothing is stored on the server.", "Activity")
if runs:
    df = pd.DataFrame(runs)
    st.dataframe(df, width="stretch", hide_index=True)
    lat = df.groupby("Method")["Latency (s)"].mean().sort_values()
    st.caption("Mean latency per method this session: " + " · ".join(f"{k} {v:.2f}s" for k, v in lat.items()))
else:
    st.info("No activity yet. Run a reconstruction and it will appear here.")

section("Project overview", "", "Project")
st.markdown("""
**DEPTHMAP AI** reconstructs dense, high-resolution depth from sparse, noisy or low-resolution depth plus an optional RGB image.

- **Classical CV** — nearest / bilinear / bicubic baselines, guided filter, joint bilateral upsampling, edge detection, hole filling, edge-aware denoising, boundary refinement.
- **Deep learning** — DepthNet, a dual-branch guided CNN (PyTorch).
- **Hybrid** — confidence-driven fusion of both, with texture-copy protection.
""")
B = load_benchmark()
if B:
    s = B.get("summary", {})
    st.markdown(f"Offline benchmark ({B['n_scenes']} unseen scenes, {B['generated_utc'][:10]}): hybrid reduces RMSE by "
                f"**{s.get('hybrid_vs_bicubic_rmse_reduction_pct', float('nan')):.1f}%** and boundary RMSE by "
                f"**{s.get('hybrid_vs_bicubic_boundary_rmse_reduction_pct', float('nan')):.1f}%** vs bicubic (mean over 2×/4×/8×).")
footer()
