"""Page 3 — Reconstruction Studio (main workspace)."""
import cv2
import numpy as np
import streamlit as st

from config import MAX_OUTPUT_PIXELS, METHOD_LABELS, METHODS, SCALES
from src.hybrid import Settings, reconstruct
from src.metrics import evaluate, measure
from src.preprocessing import (InputError, check_alignment, degrade_depth, edge_alignment_score,
                               fit_rgb_to, load_depth, load_rgb)
from src.ui import (colorize_in, compare_slider, footer, list_samples, load_sample, log_run,
                    page_header, pill, zoom_figure)
from src.visualization import (colorize, depth_png16_bytes, error_map, gray_to_rgb, npy_bytes,
                               overlay_edges, png_bytes, side_by_side, to_uint8_rgb)

page_header("Workspace", "Reconstruction Studio",
            "Load an RGB guidance image and a sparse / noisy / low-resolution depth map, choose a method and scale, "
            "and reconstruct a dense high-resolution depth map. Add ground truth to unlock accuracy metrics.")

ss = st.session_state

# =========================================================================== #
# 1. INPUTS
# =========================================================================== #
left, right = st.columns([1.05, 1], gap="large")
with left:
    st.markdown("#### 1 · Input")
    mode = st.radio("Input source", ["Sample scene", "Upload low-res depth", "Upload full-res depth & simulate sensor"],
                    horizontal=False, label_visibility="collapsed",
                    help="Samples are held-out test scenes. 'Simulate sensor' degrades your full-resolution depth so metrics can be computed.")

with right:
    st.markdown("#### 2 · Settings")
    method_label = st.selectbox("Reconstruction method", list(METHODS), index=list(METHODS.values()).index(ss.settings.get("method", "hybrid")))
    method = METHODS[method_label]
    scale = st.segmented_control("Upsampling factor", SCALES, default=ss.settings.get("scale", 4),
                                 format_func=lambda s: f"{s}×") or 4
    c1, c2 = st.columns(2)
    with c1:
        noise = st.slider("Noise reduction", 0.0, 1.0, 0.0, 0.05,
                          help="Classical edge-aware denoising (median + depth-bilateral) of the input for classical methods. "
                               "Useful for very noisy depth; at low noise it can round off corners. DepthNet always receives raw depth "
                               "(it learned its own denoising); in Hybrid this widens the residual-correction filter.")
        holes_on = st.toggle("Hole filling", True, help="Fill missing measurements. Off = holes stay marked invalid in the output.")
    with c2:
        edge = st.slider("Edge preservation", 0.0, 1.0, 0.6, 0.05,
                         help="Sharper discontinuities: lower guided-filter ε, narrower JBU range kernel, stronger boundary snapping.")
        rgb_on = st.toggle("RGB guidance", True, help="Use the colour image to guide edges. Off = depth-only processing.")
    if method == "deep":
        st.caption("DepthNet receives the raw depth and denoises / completes it itself; *edge preservation* does not apply.")
    if method in ("nearest", "bilinear", "bicubic"):
        st.caption("Interpolation baselines ignore *edge preservation* and *RGB guidance*; "
                   "noise reduction and hole filling are applied to their input like for every method.")
    ss.settings.update(method=method, scale=scale)

settings = Settings(noise_reduction=noise, edge_preservation=edge, hole_filling=holes_on, use_rgb=rgb_on)

inputs = None
errors: list[str] = []
warnings: list[str] = []

with left:
    if mode == "Sample scene":
        samples = list_samples()
        if not samples:
            st.error("No samples found. Run `python data/make_samples.py`.")
        else:
            names = [s["name"] for s in samples]
            pick = st.selectbox("Scene", names, help="All samples are unseen test scenes (seeds ≥ 200 000).")
            meta = samples[names.index(pick)]
            st.caption(meta["description"])
            inputs = load_sample(meta["dir"], scale)
            d = meta["degradation"]
            st.markdown(pill(f"noise σ={d['noise_std']}") + pill(f"holes {d['hole_fraction']*100:.0f}%") +
                        pill(f"kept {d['sparsity']*100:.0f}%") + pill("ground truth ✓", "ok"), unsafe_allow_html=True)

    else:
        upm = st.number_input("Raw depth units per metre (integer files)", min_value=1.0, value=1000.0, step=1.0,
                              help="1000 for millimetre PNGs (NYU, Kinect, Middlebury exports); 256 for KITTI. Float TIFF/NPY are assumed metres.")
        u1, u2 = st.columns(2)
        with u1:
            f_rgb = st.file_uploader("RGB guidance (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if mode == "Upload low-res depth":
            with u2:
                f_d = st.file_uploader("Low-res / sparse depth", type=["png", "tif", "tiff", "npy", "npz"])
            f_gt = st.file_uploader("Ground-truth depth (optional, output size)", type=["png", "tif", "tiff", "npy", "npz"])
        else:
            with u2:
                f_d = st.file_uploader("Full-res depth (becomes ground truth)", type=["png", "tif", "tiff", "npy", "npz"])
            f_gt = None
            k1, k2, k3 = st.columns(3)
            sim_noise = k1.slider("Sensor noise σ", 0.0, 0.05, 0.01, 0.005)
            sim_holes = k2.slider("Holes", 0.0, 0.6, 0.1, 0.05)
            sim_keep = k3.slider("Kept samples", 0.05, 1.0, 1.0, 0.05)

        try:
            rgb = load_rgb(f_rgb) if f_rgb else None
            if f_d:
                dd = load_depth(f_d, upm)
                warnings += dd.warnings
                if mode == "Upload low-res depth":
                    lr, m = dd.depth, dd.mask
                    out_shape = (lr.shape[0] * scale, lr.shape[1] * scale)
                    gt = gt_mask = None
                    if f_gt:
                        g = load_depth(f_gt, upm)
                        warnings += [f"Ground truth: {w}" for w in g.warnings]
                        gt, gt_mask = g.depth, g.mask
                        if gt.shape != out_shape:
                            warnings.append(f"Ground truth is {gt.shape[1]}×{gt.shape[0]} but output is "
                                            f"{out_shape[1]}×{out_shape[0]}; GT resized (nearest) for metrics.")
                            gt = cv2.resize(gt, out_shape[::-1], interpolation=cv2.INTER_NEAREST)
                            gt_mask = cv2.resize(gt_mask.astype(np.uint8), out_shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
                    src = f"Upload · {getattr(f_d, 'name', 'depth')}"
                else:
                    gt, gt_mask = dd.depth, dd.mask
                    hh, ww = (gt.shape[0] // scale) * scale, (gt.shape[1] // scale) * scale
                    gt, gt_mask = gt[:hh, :ww], gt_mask[:hh, :ww]
                    lr, m = degrade_depth(gt, scale, sim_noise, sim_holes, sim_keep, np.random.default_rng(0), gt_mask)
                    out_shape = gt.shape
                    if rgb is not None and rgb.shape[:2] == dd.depth.shape:
                        rgb = rgb[:hh, :ww]
                    elif rgb is not None:
                        warnings.append("RGB and depth sizes differ; RGB resized to the depth size.")
                    src = f"Simulated · {getattr(f_d, 'name', 'depth')}"
                if out_shape[0] * out_shape[1] > MAX_OUTPUT_PIXELS:
                    errors.append(f"Output {out_shape[1]}×{out_shape[0]} exceeds the hosted limit of "
                                  f"{MAX_OUTPUT_PIXELS/1e6:.1f} MP. Choose a smaller scale or a smaller input.")
                if rgb is not None:
                    warnings += check_alignment(rgb.shape, lr.shape, scale) if mode == "Upload low-res depth" else []
                    rgb = fit_rgb_to(rgb, out_shape)
                if not errors:
                    inputs = {"rgb": rgb, "lr": lr, "mask": m, "gt": gt, "gt_mask": gt_mask, "scale": scale,
                              "source": src, "is_metric": dd.is_metric}
            elif f_rgb:
                st.info("Now upload a depth map. A normal photo is not a depth map — depth must contain distances.")
        except InputError as e:
            errors.append(str(e))
        except Exception as e:  # unexpected decode problems
            errors.append(f"Could not read the files: {e}")

    for e in errors:
        st.error(e)
    for w in warnings:
        st.warning(w)

# =========================================================================== #
# Input previews
# =========================================================================== #
if inputs is not None:
    ss.inputs = inputs
    lr, m, rgb = inputs["lr"], inputs["mask"], inputs["rgb"]
    H, W = (lr.shape[0] * scale, lr.shape[1] * scale) if inputs.get("gt") is None else inputs["gt"].shape
    st.markdown("#### Input preview")
    p = st.columns(3)
    lr_vis = colorize_in(inputs, lr, m)
    with p[0]:
        if rgb is not None:
            st.image(to_uint8_rgb(rgb), caption=f"RGB guidance · {rgb.shape[1]}×{rgb.shape[0]}", width="stretch")
        else:
            st.info("No RGB guidance — guided methods fall back to depth-only filtering.")
    with p[1]:
        st.image(cv2.resize(lr_vis, (W, H), interpolation=cv2.INTER_NEAREST),
                 caption=f"Depth input · {lr.shape[1]}×{lr.shape[0]} · {m.mean()*100:.0f}% valid", width="stretch")
    with p[2]:
        if inputs.get("gt") is not None:
            st.image(colorize_in(inputs, inputs["gt"], inputs["gt_mask"]),
                     caption=f"Ground truth · {W}×{H}", width="stretch")
        else:
            st.info("No ground truth — accuracy metrics will be unavailable; latency and visual results still work.")
    vals = lr[m]
    unit = "m" if inputs.get("is_metric", True) else "rel."
    info = [pill(f"output {W}×{H}", "cyan"), pill(f"depth {vals.min():.2f}–{vals.max():.2f} {unit}"),
            pill(f"{(~m).sum():,} missing LR pixels", "warn" if (~m).any() else "ok")]
    if rgb is not None:
        lr_rgb = cv2.resize(rgb, (lr.shape[1], lr.shape[0]), interpolation=cv2.INTER_AREA)
        score = edge_alignment_score(lr_rgb, lr, m)
        if np.isfinite(score):
            info.append(pill(f"RGB–depth edge agreement {score*100:.0f}%", "ok" if score > 0.2 else "warn"))
    if not inputs.get("is_metric", True):
        info.append(pill("relative depth (no units)", "warn"))
    st.markdown(" ".join(info), unsafe_allow_html=True)

# =========================================================================== #
# 2. RUN
# =========================================================================== #
run = st.button("Run Reconstruction", type="primary", disabled=inputs is None, width="stretch")
if run and inputs is not None:
    ss.status = "Processing"
    with st.status(f"Running {method_label} at {scale}× …", expanded=True) as status:
        st.write("Pre-processing: denoising and hole filling" if (noise > 0 or holes_on) else "Pre-processing skipped")
        try:
            with measure() as mm:
                res = reconstruct(method, inputs["lr"], inputs["mask"], scale, inputs["rgb"], settings, (H, W))
            st.write(f"Reconstruction finished in {res.latency_s:.2f} s")
            metrics = None
            if inputs.get("gt") is not None:
                st.write("Computing metrics against ground truth")
                metrics = evaluate(res.depth, inputs["gt"], inputs["gt_mask"], res.valid, inputs["rgb"])
            ss.result = {"res": res, "metrics": metrics, "mem": mm, "method": method, "scale": scale,
                         "settings": settings.to_dict(), "source": inputs["source"]}
            log_run("Reconstruction", method, scale, res.latency_s, inputs["source"],
                    metrics["RMSE"] if metrics else None)
            ss.status = "Completed"
            status.update(label=f"Done · {res.latency_s:.2f} s", state="complete", expanded=False)
        except Exception as e:
            ss.status = "Failed"
            status.update(label="Reconstruction failed", state="error")
            st.error(f"Reconstruction failed: {e}")

# =========================================================================== #
# 3. RESULTS
# =========================================================================== #
R = ss.get("result")
if R and inputs is not None and R["res"].depth.shape == (H, W) and R["source"] == inputs["source"]:
    res, metrics = R["res"], R["metrics"]
    st.markdown("#### 3 · Result")
    for n in res.notes:
        st.warning(n)
    k = st.columns(5)
    k[0].metric("Method", {"jbu": "JBU", "guided": "Guided", "deep": "DepthNet", "hybrid": "Hybrid", "nearest": "Nearest", "bilinear": "Bilinear", "bicubic": "Bicubic"}[R["method"]])
    k[1].metric("Scale", f"{R['scale']}×")
    k[2].metric("Processing time", f"{res.latency_s:.2f} s")
    k[3].metric("Peak memory (NumPy)", f"{R['mem']['peak_mem_mb']:.0f} MB")
    k[4].metric("Valid output", f"{res.valid.mean()*100:.1f}%")

    out_vis = colorize_in(inputs, res.depth, res.valid)
    in_vis = cv2.resize(colorize_in(inputs, inputs["lr"], inputs["mask"]), (W, H), interpolation=cv2.INTER_NEAREST)
    tabs = st.tabs(["Before / after", "Zoom", "Edges", "Pipeline stages", "Metrics", "Download"])
    with tabs[0]:
        compare_slider(in_vis, out_vis, f"Input ({scale}× nearest view)", METHOD_LABELS[R["method"]])
        st.caption("Drag the slider. Warm = near, cool = far, black = missing.")
    with tabs[1]:
        which = st.radio("View", ["Reconstruction", "Input", "Ground truth", "RGB"], horizontal=True)
        img = {"Reconstruction": out_vis, "Input": in_vis,
               "Ground truth": colorize_in(inputs, inputs["gt"], inputs["gt_mask"]) if inputs.get("gt") is not None else None,
               "RGB": to_uint8_rgb(inputs["rgb"]) if inputs["rgb"] is not None else None}[which]
        if img is None:
            st.info(f"{which} not available for this input.")
        else:
            st.plotly_chart(zoom_figure(img), width="stretch", config={"displaylogo": False})
            st.caption("Drag a box to zoom, double-click to reset.")
    with tabs[2]:
        from src.edge_detection import canny_rgb, depth_discontinuities, sobel_magnitude, structure_edges
        e1, e2, e3 = st.columns(3)
        disc = depth_discontinuities(res.depth, res.valid)
        e1.image(overlay_edges(out_vis // 2, disc), caption="Detected depth discontinuities", width="stretch")
        e2.image(gray_to_rgb(sobel_magnitude(res.depth)), caption="Depth Sobel magnitude", width="stretch")
        if inputs["rgb"] is not None:
            se = structure_edges(inputs["rgb"], res.depth, res.valid)
            ce = canny_rgb(inputs["rgb"])
            vis = to_uint8_rgb(inputs["rgb"]) // 3
            vis[ce] = (120, 120, 160)
            vis[se] = (0, 240, 255)
            e3.image(vis, caption="RGB Canny (grey) vs structure edges (cyan) — only cyan steer depth", width="stretch")
        else:
            e3.info("RGB edges need an RGB image.")
    with tabs[3]:
        if res.stages:
            items = list(res.stages.items())
            for i in range(0, len(items), 3):
                cols = st.columns(3)
                for col, (name, arr) in zip(cols, items[i:i + 3]):
                    if arr.shape == (H, W) and ("edges" in name.lower() or "weight" in name.lower() or "band" in name.lower()):
                        col.image(gray_to_rgb(arr, normalize=False), caption=name, width="stretch")
                    elif arr.shape == (H, W):
                        col.image(colorize_in(inputs, arr), caption=name, width="stretch")
                    else:
                        col.image(cv2.resize(colorize_in(inputs, arr), (W, H), interpolation=cv2.INTER_NEAREST),
                                  caption=f"{name} (low-res)", width="stretch")
        else:
            st.info("This method has no intermediate stages (enable noise reduction / hole filling or pick Hybrid).")
    with tabs[4]:
        if metrics is None:
            st.info("Ground-truth accuracy metrics are **unavailable**: no ground truth was provided. "
                    "Use a sample scene or 'simulate sensor' mode to measure RMSE, MAE, PSNR, SSIM and edge accuracy.")
        else:
            u = "m" if inputs.get("is_metric", True) else "rel."
            a = st.columns(4)
            a[0].metric(f"RMSE ({u})", f"{metrics['RMSE']:.4f}")
            a[1].metric(f"MAE ({u})", f"{metrics['MAE']:.4f}")
            a[2].metric("PSNR (dB)", f"{metrics['PSNR']:.2f}")
            a[3].metric("SSIM", f"{metrics['SSIM']:.4f}")
            b = st.columns(4)
            b[0].metric("Edge F1", f"{metrics['Edge F1']:.3f}")
            b[1].metric(f"Boundary RMSE ({u})", f"{metrics['Boundary RMSE']:.4f}")
            b[2].metric("δ < 1.25", f"{metrics['δ<1.25']*100:.2f}%")
            tci = metrics["Texture-copy idx"]
            b[3].metric("Texture-copy index", "n/a" if not np.isfinite(tci) else f"{tci:.4f}")
            st.image(error_map(res.depth, inputs["gt"], inputs["gt_mask"] & res.valid),
                     caption="Absolute error |pred − GT| (bright = larger error)", width="stretch")
    with tabs[5]:
        base = f"depthmap_{R['method']}_x{R['scale']}"
        d1, d2, d3, d4 = st.columns(4)
        d1.download_button("Depth · 16-bit PNG (mm)", depth_png16_bytes(np.where(res.valid, res.depth, 0)),
                           f"{base}_mm.png", "image/png", width="stretch")
        d2.download_button("Depth · NumPy (m)", npy_bytes(np.where(res.valid, res.depth, np.nan)),
                           f"{base}.npy", "application/octet-stream", width="stretch")
        d3.download_button("Colourised PNG", png_bytes(out_vis), f"{base}_color.png", "image/png", width="stretch")
        panels = [in_vis, out_vis] + ([colorize_in(inputs, inputs["gt"], inputs["gt_mask"])] if inputs.get("gt") is not None else [])
        labels = ["Input", METHOD_LABELS[R["method"]]] + (["Ground truth"] if inputs.get("gt") is not None else [])
        d4.download_button("Visualization (side-by-side)", png_bytes(side_by_side(panels, labels)),
                           f"{base}_comparison.png", "image/png", width="stretch")
        st.caption("16-bit PNG stores millimetres (0 = invalid); .npy stores float32 metres (NaN = invalid).")
elif inputs is None:
    st.info("Choose a sample scene or upload files to begin.")

footer()
