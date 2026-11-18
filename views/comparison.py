"""Page 4 — Method Comparison Lab."""
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import METHOD_LABELS, METHODS, SCALES
from src.edge_detection import depth_discontinuities
from src.hybrid import Settings, reconstruct
from src.metrics import HIGHER_IS_BETTER, evaluate, measure
from src.ui import (colorize_in, footer, list_samples, load_sample, log_run, page_header, pill, style_fig)
from src.visualization import error_map, overlay_edges, png_bytes, side_by_side, to_uint8_rgb

page_header("Laboratory", "Method Comparison Lab",
            "Run all seven methods on the same input with identical settings and compare them visually and numerically.")
ss = st.session_state

c1, c2, c3 = st.columns([1.4, 1, 1])
with c1:
    samples = list_samples()
    opts = (["Current studio input"] if ss.get("inputs") is not None else []) + [s["name"] for s in samples]
    src = st.selectbox("Input", opts)
with c2:
    scale = st.segmented_control("Scale", SCALES, default=4, format_func=lambda s: f"{s}×") or 4
with c3:
    chosen = st.multiselect("Methods", list(METHODS), default=list(METHODS))

with st.expander("Shared settings (applied identically to every method)"):
    a, b, c, d = st.columns(4)
    noise = a.slider("Noise reduction", 0.0, 1.0, 0.0, 0.05, help="Classical pre-denoising. Helps classical methods on very noisy input; DepthNet always gets raw depth.")
    edge = b.slider("Edge preservation", 0.0, 1.0, 0.6, 0.05)
    holes_on = c.toggle("Hole filling", True)
    rgb_on = d.toggle("RGB guidance", True)
settings = Settings(noise_reduction=noise, edge_preservation=edge, hole_filling=holes_on, use_rgb=rgb_on)

if src == "Current studio input":
    inputs = ss.inputs
    if inputs["scale"] != scale:
        st.warning(f"The studio input was prepared for {inputs['scale']}×; using {inputs['scale']}×.")
        scale = inputs["scale"]
else:
    meta = samples[[s["name"] for s in samples].index(src)]
    inputs = load_sample(meta["dir"], scale)

H, W = inputs["gt"].shape if inputs.get("gt") is not None else (inputs["lr"].shape[0] * scale, inputs["lr"].shape[1] * scale)
has_gt = inputs.get("gt") is not None
st.markdown(pill(inputs["source"], "cyan") + pill(f"{scale}× → {W}×{H}") +
            pill("ground truth ✓" if has_gt else "no ground truth", "ok" if has_gt else "warn"), unsafe_allow_html=True)

if st.button("Run comparison", type="primary", width="stretch", disabled=not chosen):
    rows, outs = [], {}
    prog = st.progress(0.0, "Starting…")
    for i, label in enumerate(chosen):
        key = METHODS[label]
        prog.progress(i / len(chosen), f"Running {label} …")
        with measure() as mm:
            r = reconstruct(key, inputs["lr"], inputs["mask"], scale, inputs["rgb"], settings, (H, W))
        row = {"Method": label, "Latency (s)": r.latency_s, "Peak mem NumPy (MB)": mm["peak_mem_mb"],
               "RSS Δ (MB)": mm.get("rss_delta_mb", np.nan)}
        if has_gt:
            row.update(evaluate(r.depth, inputs["gt"], inputs["gt_mask"], r.valid, inputs["rgb"]))
        rows.append(row)
        outs[key] = r
        log_run("Comparison", key, scale, r.latency_s, inputs["source"], row.get("RMSE"))
    prog.progress(1.0, "Done")
    ss.comparison = {"rows": rows, "outs": outs, "source": inputs["source"], "scale": scale, "has_gt": has_gt,
                     "settings": settings.to_dict(), "inputs": inputs}

C = ss.get("comparison")
if C and C["source"] == inputs["source"] and C["scale"] == scale:
    outs, df = C["outs"], pd.DataFrame(C["rows"]).set_index("Method")
    for r in outs.values():
        for n in r.notes:
            st.warning(f"{METHOD_LABELS[r.method]}: {n}")

    st.markdown("#### Outputs")
    tiles = [("Input (nearest view)", cv2.resize(colorize_in(inputs, inputs["lr"], inputs["mask"]), (W, H), interpolation=cv2.INTER_NEAREST))]
    tiles += [(METHOD_LABELS[k], colorize_in(inputs, r.depth, r.valid)) for k, r in outs.items()]
    if has_gt:
        tiles.append(("Ground truth", colorize_in(inputs, inputs["gt"], inputs["gt_mask"])))
    for i in range(0, len(tiles), 3):
        cols = st.columns(3)
        for col, (name, img) in zip(cols, tiles[i:i + 3]):
            col.image(img, caption=name, width="stretch")
    st.download_button("Download comparison sheet (PNG)",
                       png_bytes(side_by_side([t[1] for t in tiles], [t[0].split(" (")[0] for t in tiles])),
                       f"comparison_x{scale}.png", "image/png")

    if has_gt:
        st.markdown("#### Metric table")
        show = ["RMSE", "MAE", "PSNR", "SSIM", "δ<1.25", "Edge F1", "Boundary RMSE", "Texture-copy idx", "Coverage", "Latency (s)", "Peak mem NumPy (MB)"]
        t = df[[c for c in show if c in df]]
        hi = [c for c in t.columns if c in HIGHER_IS_BETTER]
        lo = [c for c in t.columns if c not in HIGHER_IS_BETTER]
        st.dataframe(t.style.format("{:.4f}").highlight_max(subset=hi, color="#0b4f5c").highlight_min(subset=lo, color="#0b4f5c"),
                     width="stretch")
        st.caption("Best value per column highlighted. RMSE/MAE/Boundary RMSE in metres. Lower texture-copy index = less RGB texture leaked into flat depth.")

        st.markdown("#### Difference maps |prediction − ground truth|")
        vmax = float(np.nanpercentile(np.abs(outs.get("bicubic", list(outs.values())[0]).depth - inputs["gt"])[inputs["gt_mask"]], 99))
        keys = list(outs)
        for i in range(0, len(keys), 4):
            cols = st.columns(4)
            for col, k in zip(cols, keys[i:i + 4]):
                r = outs[k]
                col.image(error_map(r.depth, inputs["gt"], inputs["gt_mask"] & r.valid, vmax), caption=METHOD_LABELS[k], width="stretch")
        st.caption(f"Same colour scale for every method (bright = error ≥ {vmax:.2f} m).")

    st.markdown("#### Edge preservation close-up")
    ref = inputs["gt"] if has_gt else outs[list(outs)[-1]].depth
    disc = depth_discontinuities(ref)
    ch, cw = max(H // 4, 32), max(W // 4, 32)
    dens = cv2.boxFilter(disc.astype(np.float32), -1, (cw, ch), normalize=False)
    y, x = np.unravel_index(np.argmax(dens), dens.shape)
    y0, x0 = int(np.clip(y - ch // 2, 0, H - ch)), int(np.clip(x - cw // 2, 0, W - cw))
    crop = lambda img: cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (cw * 3, ch * 3), interpolation=cv2.INTER_NEAREST)
    st.caption(f"3× zoom on the region with the most depth discontinuities (x={x0}, y={y0}). Cyan = detected discontinuities in each output.")
    crops = []
    if inputs["rgb"] is not None:
        crops.append(("RGB", to_uint8_rgb(inputs["rgb"])))
    crops += [(METHOD_LABELS[k], overlay_edges(colorize_in(inputs, r.depth, r.valid), depth_discontinuities(r.depth, r.valid))) for k, r in outs.items()]
    if has_gt:
        crops.append(("Ground truth", overlay_edges(colorize_in(inputs, inputs["gt"]), disc)))
    for i in range(0, len(crops), 4):
        cols = st.columns(4)
        for col, (n, img) in zip(cols, crops[i:i + 4]):
            col.image(crop(img), caption=n, width="stretch")

    st.markdown("#### Charts")
    g1, g2 = st.columns(2)
    names = [n.split(" (")[0] for n in df.index]
    if has_gt:
        f = go.Figure([go.Bar(x=names, y=df["RMSE"], name="RMSE", marker_color="#00e5ff"),
                       go.Bar(x=names, y=df["Boundary RMSE"], name="Boundary RMSE", marker_color="#8b5cf6")])
        f.update_layout(title="Accuracy (lower is better)", barmode="group", yaxis_title="metres")
        g1.plotly_chart(style_fig(f), width="stretch")
        f2 = go.Figure(go.Bar(x=names, y=df["Edge F1"], marker_color="#34d399"))
        f2.update_layout(title="Edge preservation · Edge F1 (higher is better)", yaxis_range=[0, 1])
        g2.plotly_chart(style_fig(f2), width="stretch")
    f3 = go.Figure(go.Bar(x=names, y=df["Latency (s)"], marker_color="#3b82f6"))
    f3.update_layout(title="Processing latency (s)")
    (g1 if not has_gt else st).plotly_chart(style_fig(f3, 320), width="stretch")
    if not has_gt:
        st.info("Ground-truth accuracy metrics are unavailable for this input — only latency and visual comparison are shown.")
else:
    st.info("Press **Run comparison** to evaluate the selected methods on this input.")

footer()
