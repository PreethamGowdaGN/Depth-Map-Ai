"""Shared Streamlit UI helpers: theme, cards, session state, sample loading."""
from __future__ import annotations

import base64
import json
from datetime import datetime

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from config import ASSETS_DIR, METHOD_LABELS, RESULTS_DIR, SAMPLES_DIR
from src.preprocessing import degrade_depth
from src.visualization import colorize, png_bytes

PLOT_COLORS = ["#00e5ff", "#8b5cf6", "#3b82f6", "#34d399", "#f472b6", "#fbbf24", "#f87171"]


def inject_css():
    css = (ASSETS_DIR / "style.css").read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def init_state():
    ss = st.session_state
    ss.setdefault("runs", [])            # real activity log (this session only)
    ss.setdefault("inputs", None)        # current input bundle
    ss.setdefault("result", None)        # last reconstruction
    ss.setdefault("comparison", None)    # last method comparison
    ss.setdefault("settings", {"method": "hybrid", "scale": 4})
    ss.setdefault("status", "Idle")


def log_run(kind: str, method: str, scale: int, latency: float, source: str, rmse=None):
    st.session_state.runs.insert(0, {
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Action": kind,
        "Method": METHOD_LABELS.get(method, method),
        "Scale": f"{scale}×",
        "Latency (s)": round(latency, 3),
        "Input": source,
        "RMSE (m)": None if rmse is None or not np.isfinite(rmse) else round(rmse, 4),
    })


# --------------------------------------------------------------------------- #
# HTML building blocks
# --------------------------------------------------------------------------- #
def section(title: str, sub: str = "", kicker: str = ""):
    k = f'<div class="dm-kicker">{kicker}</div>' if kicker else ""
    s = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(f'<div class="dm-section">{k}<h2>{title}</h2>{s}</div>', unsafe_allow_html=True)


def card(icon: str, title: str, body: str):
    st.markdown(f'<div class="dm-card"><div class="dm-icon">{icon}</div><h4>{title}</h4><p>{body}</p></div>',
                unsafe_allow_html=True)


def stat(label: str, value, sub: str = "", cyan: bool = False):
    st.markdown(f'<div class="dm-stat"><div class="k">{label}</div><div class="v {"cyan" if cyan else ""}">{value}</div>'
                f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def pill(text: str, kind: str = "") -> str:
    return f'<span class="dm-pill {kind}">{text}</span>'


def page_header(kicker: str, title: str, sub: str):
    init_state()  # pages can also be run standalone (tests)
    st.markdown(f'<div class="dm-kicker">{kicker}</div><h1 style="margin:.2rem 0 .3rem">{title}</h1>'
                f'<p style="color:var(--muted);max-width:820px">{sub}</p>', unsafe_allow_html=True)


def footer():
    st.markdown('<div class="dm-footer"><div><b style="color:var(--text)">DEPTHMAP AI</b> · Beyond Pixels. Preserving Geometry.</div>'
                '<div>Computer Vision Hackathon 2026 · Problem Statement 4 (P4)</div></div>', unsafe_allow_html=True)


def b64_png(img: np.ndarray) -> str:
    return base64.b64encode(png_bytes(img)).decode()


def compare_slider(before: np.ndarray, after: np.ndarray, label_a="Input", label_b="Reconstruction", height=None):
    """Interactive before/after slider (drag to reveal)."""
    h, w = after.shape[:2]
    if before.shape[:2] != (h, w):
        before = cv2.resize(before, (w, h), interpolation=cv2.INTER_NEAREST)
    a, b = b64_png(before), b64_png(after)
    box_h = 600
    maxw = int(min(1200, box_h * w / h))
    html = f"""
    <style>
    .cmp{{position:relative;width:100%;max-width:{maxw}px;margin:0 auto;aspect-ratio:{w}/{h};border-radius:14px;overflow:hidden;border:1px solid rgba(120,160,255,.2);background:#000;font-family:Inter,sans-serif}}
    .cmp img{{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;image-rendering:auto;user-select:none;pointer-events:none}}
    .cmp .top{{clip-path:inset(0 0 0 50%)}}
    .cmp input{{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:ew-resize;margin:0}}
    .cmp .bar{{position:absolute;top:0;bottom:0;left:50%;width:2px;background:#00e5ff;box-shadow:0 0 14px #00e5ff;pointer-events:none}}
    .cmp .knob{{position:absolute;top:50%;left:50%;width:34px;height:34px;margin:-17px 0 0 -17px;border-radius:50%;background:#050816;border:2px solid #00e5ff;color:#00e5ff;display:grid;place-items:center;font-size:14px;pointer-events:none}}
    .cmp .lab{{position:absolute;top:10px;padding:4px 10px;border-radius:999px;background:rgba(5,8,22,.75);color:#e6ecff;font-size:12px;font-weight:600}}
    </style>
    <div class="cmp" id="c">
      <img src="data:image/png;base64,{a}"><img class="top" id="t" src="data:image/png;base64,{b}">
      <div class="bar" id="bar"></div><div class="knob" id="k">⇆</div>
      <span class="lab" style="left:10px">{label_a}</span><span class="lab" style="right:10px">{label_b}</span>
      <input type="range" min="0" max="100" value="50" id="r">
    </div>
    <script>
      const r=document.getElementById('r'),t=document.getElementById('t'),bar=document.getElementById('bar'),k=document.getElementById('k');
      r.addEventListener('input',e=>{{const v=e.target.value;t.style.clipPath='inset(0 0 0 '+v+'%)';bar.style.left=v+'%';k.style.left=v+'%';}});
    </script>"""
    hh = height or int(maxw * h / w) + 8
    if hasattr(st, "iframe"):  # Streamlit >= 1.5x: st.iframe replaces components.html
        st.iframe(html, height=hh)  # our own generated HTML (two encoded images), never user HTML
    else:
        components.html(html, height=hh)


def zoom_figure(img: np.ndarray, title: str = "", height: int = 520) -> go.Figure:
    fig = go.Figure(go.Image(z=img))
    fig.update_layout(title=title, height=height, margin=dict(l=0, r=0, t=30 if title else 0, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#e6ecff", family="Inter"), dragmode="zoom")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,20,48,0.45)",
                      font=dict(color="#c9d3f5", family="Inter"), margin=dict(l=10, r=10, t=46, b=10),
                      legend=dict(bgcolor="rgba(0,0,0,0)"), colorway=PLOT_COLORS)
    fig.update_xaxes(gridcolor="rgba(120,160,255,0.10)", zerolinecolor="rgba(120,160,255,0.2)")
    fig.update_yaxes(gridcolor="rgba(120,160,255,0.10)", zerolinecolor="rgba(120,160,255,0.2)")
    return fig


# --------------------------------------------------------------------------- #
# Samples & saved results
# --------------------------------------------------------------------------- #
def list_samples() -> list[dict]:
    out = []
    for p in sorted(SAMPLES_DIR.glob("*/meta.json")):
        m = json.loads(p.read_text())
        m["dir"] = str(p.parent)
        out.append(m)
    return out


@st.cache_data(show_spinner=False)
def load_sample(sample_dir: str, scale: int) -> dict:
    """Load a sample's GT + RGB and simulate the sensor deterministically."""
    from pathlib import Path
    d = Path(sample_dir)
    meta = json.loads((d / "meta.json").read_text())
    rgb = cv2.cvtColor(cv2.imread(str(d / "rgb.png")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    gt = cv2.imread(str(d / "depth_gt.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    gt_mask = gt > 0
    deg = meta["degradation"]
    lr, m = degrade_depth(gt, scale, deg["noise_std"], deg["hole_fraction"], deg["sparsity"],
                          np.random.default_rng(meta.get("seed", 0)), gt_mask)
    sh = d / "sensor_holes.png"
    if sh.exists():
        holes = cv2.imread(str(sh), cv2.IMREAD_GRAYSCALE) > 0
        off = scale // 2
        hl = holes[off::scale, off::scale]
        m &= ~hl
        lr[~m] = 0
    rgb_shift = meta.get("rgb_shift_px", 0)
    if rgb_shift:
        rgb = np.roll(rgb, rgb_shift, axis=1)
    return {"rgb": rgb, "lr": lr, "mask": m, "gt": gt, "gt_mask": gt_mask, "scale": scale,
            "source": f"Sample · {meta['name']}", "is_metric": True, "meta": meta}


def load_benchmark():
    p = RESULTS_DIR / "benchmark.json"
    return json.loads(p.read_text()) if p.exists() else None


def depth_range(inputs: dict):
    ref = inputs.get("gt") if inputs.get("gt") is not None else inputs["lr"]
    refm = (inputs.get("gt_mask") if inputs.get("gt") is not None else inputs["mask"])
    vals = ref[refm]
    return float(np.percentile(vals, 1)), float(np.percentile(vals, 99))


def colorize_in(inputs, depth, mask=None):
    lo, hi = depth_range(inputs)
    return colorize(depth, mask, lo, hi)
