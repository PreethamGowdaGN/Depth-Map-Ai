"""Page 5 — Performance Analytics."""
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import METHOD_LABELS, RESULTS_DIR
from src.ui import PLOT_COLORS, footer, load_benchmark, page_header, section, style_fig

page_header("Measurement", "Performance Analytics",
            "Accuracy, boundary quality, latency and memory — from your live comparison in this session and from the saved offline benchmark.")
ss = st.session_state

# ------------------------------------------------------------------------- #
section("Live session", "From the last run of the Comparison Lab in this browser session.", "Session")
C = ss.get("comparison")
if not C:
    st.info("No comparison has been run in this session yet. Open the **Comparison Lab** and press *Run comparison*.")
else:
    df = pd.DataFrame(C["rows"])
    df["Short"] = df["Method"].str.split(" \\(").str[0]
    st.caption(f"Input: {C['source']} · scale {C['scale']}×")
    if not C["has_gt"]:
        st.warning("Ground-truth accuracy metrics are **unavailable** for this input (no ground truth). Showing latency and memory only.")
    else:
        metrics = [("RMSE", "m", False), ("MAE", "m", False), ("PSNR", "dB", True), ("SSIM", "", True),
                   ("Boundary RMSE", "m · boundary accuracy", False), ("Edge F1", "edge preservation", True)]
        for i in range(0, 6, 3):
            cols = st.columns(3)
            for col, (m, unit, hib) in zip(cols, metrics[i:i + 3]):
                best = df[m].idxmax() if hib else df[m].idxmin()
                colors = ["#00e5ff" if j == best else "#3b4a7a" for j in range(len(df))]
                f = go.Figure(go.Bar(x=df["Short"], y=df[m], marker_color=colors,
                                     text=[f"{v:.3f}" for v in df[m]], textposition="outside"))
                f.update_layout(title=f"{m} ({'higher' if hib else 'lower'} is better)", yaxis_title=unit, showlegend=False)
                col.plotly_chart(style_fig(f, 320), width="stretch")
        f = px.scatter(df, x="Latency (s)", y="RMSE", text="Short", size=[18] * len(df),
                       color="Short", color_discrete_sequence=PLOT_COLORS)
        f.update_traces(textposition="top center")
        f.update_layout(title="Accuracy vs latency trade-off", showlegend=False)
        st.plotly_chart(style_fig(f, 380), width="stretch")
    cols = st.columns(2)
    f = go.Figure(go.Bar(x=df["Short"], y=df["Latency (s)"], marker_color="#3b82f6"))
    f.update_layout(title="Processing latency (s)")
    cols[0].plotly_chart(style_fig(f, 320), width="stretch")
    f = go.Figure([go.Bar(x=df["Short"], y=df["Peak mem NumPy (MB)"], name="Peak NumPy/Python (tracemalloc)", marker_color="#8b5cf6"),
                   go.Bar(x=df["Short"], y=df["RSS Δ (MB)"], name="Process RSS increase", marker_color="#f472b6")])
    f.update_layout(title="Memory consumption (MB)", barmode="group")
    cols[1].plotly_chart(style_fig(f, 320), width="stretch")
    st.caption("tracemalloc measures NumPy/Python allocations; PyTorch's allocator is only visible in the RSS increase "
               "(which is 0 when memory was already reserved by an earlier run).")

# ------------------------------------------------------------------------- #
section("Offline benchmark", "Saved by `python -m evaluation.benchmark` — same held-out scenes and settings for every method.", "Benchmark")
B = load_benchmark()
if not B:
    st.info("No saved benchmark found (results/benchmark.json). Run `python -m evaluation.benchmark`.")
else:
    rows = pd.DataFrame(B["rows"])
    rows["Method name"] = rows["method"].map(lambda k: METHOD_LABELS[k].split(" (")[0])
    st.markdown(f"**{B['n_scenes']} unseen test scenes** · {B['size'][1]}×{B['size'][0]} · scales {', '.join(str(s)+'×' for s in B['scales'])} · "
                f"degradation: noise σ={B['degradation']['noise_std']}, holes {B['degradation']['hole_fraction']*100:.0f}% · "
                f"generated {B['generated_utc']}")
    pick = st.selectbox("Metric", ["RMSE", "MAE", "PSNR", "SSIM", "Edge F1", "Boundary RMSE", "δ<1.25", "Texture-copy idx", "Latency (s)", "Peak mem (MB)"])
    f = px.line(rows, x="scale", y=pick, color="Method name", markers=True, color_discrete_sequence=PLOT_COLORS)
    f.update_layout(title=f"{pick} across upsampling factors", xaxis=dict(tickvals=B["scales"], ticktext=[f"{s}×" for s in B["scales"]]))
    st.plotly_chart(style_fig(f, 420), width="stretch")
    piv = rows.pivot_table(index="Method name", columns="scale", values=pick)
    piv.columns = [f"{c}×" for c in piv.columns]
    st.dataframe(piv.style.format("{:.4f}"), width="stretch")

    ab = RESULTS_DIR / "ablation.json"
    if ab.exists():
        A = json.loads(ab.read_text())
        section("Ablation study", f"Hybrid pipeline with one component removed at a time · {A['scale']}× · {A['n_scenes']} scenes.", "Ablation")
        adf = pd.DataFrame(A["rows"])
        f = go.Figure([go.Bar(x=adf["variant"], y=adf["RMSE"], name="RMSE", marker_color="#00e5ff"),
                       go.Bar(x=adf["variant"], y=adf["Boundary RMSE"], name="Boundary RMSE", marker_color="#8b5cf6")])
        f.update_layout(barmode="group", title="Removing a component — lower is better", yaxis_title="metres")
        st.plotly_chart(style_fig(f, 400), width="stretch")
        st.dataframe(adf.set_index("variant").style.format("{:.4f}"), width="stretch")

    fa = RESULTS_DIR / "failure_analysis.json"
    if fa.exists():
        F = json.loads(fa.read_text())
        section("Failure cases", "Stress tests — hybrid vs bicubic on each hard condition.", "Robustness")
        fdf = pd.DataFrame(F["rows"])
        f = px.bar(fdf, x="case", y="RMSE", color="method", barmode="group", color_discrete_sequence=PLOT_COLORS)
        f.update_layout(title=f"RMSE per failure case ({F['scale']}×)")
        st.plotly_chart(style_fig(f, 400), width="stretch")
        st.caption("Read the honest discussion of each case in eval_P4_TeamID.md.")

footer()
