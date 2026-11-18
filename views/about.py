"""Page 7 — About the project."""
import streamlit as st

from src.ui import card, footer, page_header, section

page_header("Project", "About DEPTHMAP AI",
            "Computer Vision Hackathon 2026 · Problem Statement 4 (P4) — Edge-Preserving Depth Map Upsampling and Refinement.")

section("The problem", "", "Problem Statement 4")
st.markdown("""
Depth sensors (Kinect, ToF, phone LiDAR, automotive LiDAR) deliver depth that is **low-resolution, noisy, and full of holes**,
while the paired RGB camera is sharp. The challenge: build a system that turns such depth into **dense, high-resolution depth**
that preserves object boundaries, geometric structure and depth discontinuities — using RGB guidance *without copying RGB texture*
— and support 2×, 4× and 8× upsampling, evaluated on unseen inputs.
""")

section("Why it is hard", "", "Motivation")
cols = st.columns(3)
with cols[0]:
    card("⇲", "Flying pixels", "Interpolating across an object edge invents depths between foreground and background that exist nowhere in the scene.")
with cols[1]:
    card("▦", "Texture copying", "A poster on a wall has strong colour edges but no depth edge. Naïve RGB guidance carves the poster into the wall.")
with cols[2]:
    card("◌", "Missing data", "Glass, dark and shiny surfaces return nothing. Holes must be filled from the right side of the boundary.")

section("Technical approach", "", "Method")
t = st.tabs(["Classical CV", "Deep learning", "Hybrid", "Evaluation"])
with t[0]:
    st.markdown("""
| Component | What it does | Where |
|---|---|---|
| Nearest / bilinear / bicubic | Mask-aware baselines (normalised convolution — missing pixels never count as 0 m) | `src/interpolation.py` |
| Guided filter | Output is locally a linear function of RGB → edges follow the image | `src/filters.py` |
| Joint bilateral upsampling | Averages low-res samples weighted by distance *and* colour similarity | `src/filters.py` |
| Edge detection | Sobel, Canny, relative depth gradient, discontinuity detection, *structure edges* | `src/edge_detection.py` |
| Denoising | Mask-aware median + bilateral filter on depth values | `src/filters.py` |
| Hole filling | RGB-guided iterative diffusion + multi-scale push-pull | `src/hole_filling.py` |
| Boundary refinement | Depth-guided residual correction (hybrid) and optional weighted-median flying-pixel snapping | `src/hybrid.py`, `src/refinement.py` |
""")
with t[1]:
    st.markdown("**DepthNet** — dual-branch residual encoder–decoder with gated RGB–depth fusion at four scales and coarse-to-fine residual heads. See the *Model* page.")
with t[2]:
    st.markdown("""
1. Mask-aware bicubic initialisation → 2. **DepthNet** on the raw low-res depth + RGB →
3. depth-discontinuity analysis → 4. **edge-aware data-fidelity correction**: the CNN's error at every real measurement
is spread with a joint bilateral filter guided by the predicted *depth*, then added back.

Optional stages (RGB-JBU fusion at structure edges, flying-pixel snapping) are available, but our ablation showed they
*hurt* once DepthNet was trained, so they are off by default. Every design choice was made on validation scenes before
the test benchmark was run.
""")
with t[3]:
    st.markdown("RMSE, MAE, PSNR, SSIM, δ<1.25, Edge F1, boundary RMSE, texture-copy index, latency and memory — on held-out scenes at 2×, 4× and 8×, plus an ablation study and failure-case analysis.")

section("Applications", "", "Impact")
cols = st.columns(4)
for col, (i, a, b) in zip(cols, [("🤖", "Robotics", "Grasping and obstacle avoidance need exact object edges."),
                                ("🥽", "AR / VR", "Occlusion of virtual objects behind real ones."),
                                ("🚗", "Autonomous driving", "Densifying sparse LiDAR with camera guidance."),
                                ("📱", "Computational photography", "Portrait-mode blur without halo artefacts.")]):
    with col:
        card(i, a, b)

section("Ethics & limitations", "", "Responsible AI")
st.markdown("""
- Depth reconstruction can **hallucinate plausible but wrong geometry** in large holes; outputs are flagged with validity masks and should not be used for safety-critical decisions without real-sensor validation.
- The shipped model is trained on synthetic data; performance on real sensors has not been measured in this build.
- Uploaded images are processed in memory and are not stored.
""")

section("Technologies", "", "Stack")
st.markdown("Python · Streamlit · PyTorch · OpenCV · NumPy · SciPy · scikit-image · Plotly · pytest · Docker")
footer()
