"""Page 1 — Landing page."""
import numpy as np
import streamlit as st

from src.ui import card, footer, list_samples, load_benchmark, load_sample, section, stat


@st.cache_data(show_spinner=False)
def hero_demo():
    """Real pipeline output on a held-out scene (computed, not a stock image)."""
    from src.hybrid import Settings, reconstruct
    from src.visualization import colorize, to_uint8_rgb
    import cv2
    samples = list_samples()
    if not samples:
        return None
    s = load_sample(samples[0]["dir"], 8)
    lo, hi = np.percentile(s["gt"], 1), np.percentile(s["gt"], 99)
    r = reconstruct("hybrid", s["lr"], s["mask"], 8, s["rgb"], Settings())
    lr_vis = cv2.resize(colorize(s["lr"], s["mask"], lo, hi), s["gt"].shape[::-1], interpolation=cv2.INTER_NEAREST)
    return [to_uint8_rgb(s["rgb"]), lr_vis, colorize(r.depth, r.valid, lo, hi), colorize(s["gt"], None, lo, hi)]


st.markdown("""
<div class="dm-hero">
  <div class="grid"></div>
  <span class="dm-eyebrow"><span class="dot"></span>Problem Statement 4 · Edge-Preserving Depth Upsampling</span>
  <div class="dm-title">See Beyond <span class="dm-grad">the Surface.</span></div>
  <div class="dm-tagline">Beyond Pixels. Preserving Geometry.</div>
  <p class="dm-lead" style="margin-top:1rem">
    DEPTHMAP AI turns sparse, noisy, low-resolution depth maps into dense high-resolution depth —
    keeping object boundaries crisp, filling holes, and refusing to copy RGB texture into geometry.
    Classical computer vision, a guided deep network and a confidence-driven hybrid engine, side by side.
  </p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns([1.1, 1.1, 3])
with c1:
    if st.button("Get Started  →", type="primary", width="stretch"):
        st.switch_page("views/reconstruction.py")
with c2:
    if st.button("Compare methods", width="stretch"):
        st.switch_page("views/comparison.py")

st.write("")
demo = hero_demo()
if demo is not None:
    cols = st.columns(4)
    caps = ["RGB guidance", "Input: 8× low-res, noisy, holes", "DEPTHMAP AI hybrid output", "Ground truth"]
    for col, img, cap in zip(cols, demo, caps):
        with col:
            st.image(img, width="stretch")
            st.markdown(f'<div class="dm-caption">{cap}</div>', unsafe_allow_html=True)
    st.caption("Live output of the hybrid pipeline on an unseen test scene (synthetic, seed 200 003). "
               "Warm = near, cool = far, black = missing measurement.")

bench = load_benchmark()
section("At a glance", "Numbers below are read from the saved offline benchmark (results/benchmark.json).", "Capabilities")
s1, s2, s3, s4 = st.columns(4)
with s1:
    stat("Reconstruction methods", "7", "3 interpolation · 2 edge-aware · deep · hybrid", cyan=True)
with s2:
    stat("Upsampling factors", "2× · 4× · 8×", "plus hole filling & denoising")
if bench:
    best = bench.get("summary", {}).get("hybrid_vs_bicubic_rmse_reduction_pct")
    with s3:
        stat("RMSE reduction vs bicubic", f"{best:.0f}%" if best is not None else "—", "hybrid, mean over test set & scales", cyan=True)
    with s4:
        stat("Test scenes", bench.get("n_scenes", "—"), "held-out, never seen in training")
else:
    with s3:
        stat("Benchmark", "not run yet", "python -m evaluation.benchmark")
    with s4:
        stat("Metrics", "9", "RMSE · MAE · PSNR · SSIM · Edge F1 …")

section("What it does", "Every feature below is implemented and runs in this app.", "Features")
feats = [
    ("◧", "RGB-guided upsampling", "Guided filter and joint bilateral upsampling use the colour image to place depth edges exactly on object outlines."),
    ("✦", "Texture-copy guard", "RGB enters only through a learned gate; classical corrections use depth, not colour, as their guide — stripes on a flat wall stay flat."),
    ("◎", "Hole filling", "Edge-aware diffusion and multi-scale push-pull fill missing measurements without leaking across boundaries."),
    ("≋", "Edge-aware denoising", "Mask-aware median + depth bilateral filter remove sensor noise but keep discontinuities."),
    ("⧉", "DepthNet CNN", "Dual-branch encoder–decoder with gated RGB–depth fusion and residual multi-scale output (PyTorch)."),
    ("⚡", "Hybrid engine", "DepthNet prediction corrected against the real measurements by a depth-guided joint bilateral filter — edges stay put, bias disappears."),
]
for row in (feats[:3], feats[3:]):
    cols = st.columns(3)
    for col, f in zip(cols, row):
        with col:
            card(*f)
    st.write("")

section("Technology", "", "Stack")
st.markdown(" ".join(f'<span class="dm-pill cyan">{t}</span>' for t in
            ["Python", "Streamlit", "PyTorch", "OpenCV", "NumPy", "SciPy", "scikit-image", "Plotly"]),
            unsafe_allow_html=True)

section("Mission", "", "Why")
st.markdown("""
<div class="dm-card"><p style="font-size:1rem">
Depth sensors on phones, robots and cars capture far fewer pixels than their cameras, and they drop out on glass,
dark or shiny surfaces. Naïve resizing smears foreground into background — creating "flying pixels" that make robots
mis-grasp and AR objects float. Our mission: <b style="color:var(--text)">reconstruct depth that respects geometry</b>,
and measure honestly where it still fails.
</p></div>""", unsafe_allow_html=True)

footer()
