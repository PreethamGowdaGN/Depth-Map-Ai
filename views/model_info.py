"""Page 6 — Model information (honest checkpoint status)."""
import plotly.graph_objects as go
import streamlit as st

from config import CHECKPOINT_PATH
from src.ui import footer, page_header, pill, section, style_fig

page_header("Deep learning", "DepthNet — model information",
            "A lightweight RGB-guided depth upsampling & completion CNN written in PyTorch.")


@st.cache_resource(show_spinner="Loading model…")
def runner():
    from models.inference import get_runner
    return get_runner()


r = runner()
meta = r.meta
if r.trained:
    st.markdown(pill("checkpoint loaded", "ok") + pill(f"{r.n_params:,} parameters", "cyan") + pill(f"device: {r.device}") +
                pill(f"trained {meta.get('iterations', '?')} iterations · {meta.get('train_minutes', '?')} min on {meta.get('device', '?')}"),
                unsafe_allow_html=True)
else:
    st.markdown(pill("UNTRAINED — no checkpoint", "bad") + pill(f"{r.n_params:,} parameters", "cyan"), unsafe_allow_html=True)
    st.error(f"No checkpoint at `{CHECKPOINT_PATH.relative_to(CHECKPOINT_PATH.parents[2])}`. The deep model is running with random "
             "weights and its outputs are meaningless. Train it with `python -m training.train`.")

section("Architecture", "Dual-branch encoder · gated fusion · multi-scale residual decoder", "Network")
st.graphviz_chart("""
digraph G {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#0e1430", color="#00e5ff", fontcolor="#e6ecff", fontname="Helvetica", fontsize=11];
  edge [color="#8b95b8"];
  rgb [label="RGB guide\\n3×H×W"]; dep [label="Init depth (bicubic)\\n+ confidence\\n2×H×W"];
  subgraph cluster_e { label="Encoders (1, ½, ¼, ⅛)"; fontcolor="#8b95b8"; color="#26305a";
    er [label="RGB branch\\nconv + ResBlocks"]; ed [label="Depth branch\\nconv + ResBlocks"]; }
  fu [label="Gated fusion ×4\\nd + σ(g)·proj(rgb)", color="#8b5cf6"];
  bt [label="Bottleneck\\n2 ResBlocks @ ⅛"];
  de [label="Decoder ¼ → ½ → 1\\nskip connections"];
  hd [label="Residual heads\\n¼, ½, 1 (coarse→fine)", color="#8b5cf6"];
  out [label="Depth = init + residual\\n1×H×W", color="#34d399"];
  rgb -> er; dep -> ed; er -> fu; ed -> fu; fu -> bt -> de -> hd -> out; dep -> out [style=dashed, label="skip"];
}""")

c1, c2 = st.columns(2)
with c1:
    st.markdown("""
**Input** (all at output resolution)
- RGB guidance image, values 0–1 (zeros if RGB guidance is off)
- Initial depth: hole-filled low-res depth, bicubic-upsampled, normalised per image
- Confidence map: up-sampled validity mask (1 = real measurement)

**Output** — dense depth at output resolution, de-normalised back to metres.
""")
with c2:
    st.markdown("""
**Why this design**
- *Residual learning* — the CNN only learns corrections to bicubic; smooth areas stay stable.
- *Gated fusion* — a learned gate decides where RGB features may influence depth, suppressing texture copying.
- *Scale-agnostic* — trained on random 2×/4×/8× inputs, so one model serves all factors.
- *Loss* — L1 + gradient loss + 4× weight within 2 px of depth edges, with deep supervision at ½ and ¼.
""")

section("Training", "", "Data & status")
if r.trained:
    st.markdown(f"""
| Item | Value |
|---|---|
| Dataset | {meta.get('dataset')} |
| Iterations · batch · crop | {meta.get('iterations')} · {meta.get('batch_size')} · {meta.get('crop')}×{meta.get('crop')} |
| Hardware | {meta.get('device')} · {meta.get('train_minutes')} min |
| Validation (4×, σ=0.01, 10 % holes) | normalised RMSE **{meta.get('val_rmse_normalised', float('nan')):.4f}** vs bicubic {meta.get('val_rmse_bicubic_normalised', float('nan')):.4f} |
| Saved | {meta.get('saved_utc')} |
""")
    from models.inference import training_history
    h = training_history()
    if h and h["iter"]:
        f = go.Figure([go.Scatter(x=h["iter"], y=h["val_rmse_model"], name="DepthNet (val)", mode="lines+markers"),
                       go.Scatter(x=h["iter"], y=h["val_rmse_bicubic"], name="Bicubic (val)", mode="lines", line=dict(dash="dash")),
                       go.Scatter(x=h["iter"], y=h["train_loss"], name="Train loss", mode="lines", yaxis="y2", opacity=.6)])
        f.update_layout(title="Training history", xaxis_title="iteration", yaxis_title="val RMSE (normalised)",
                        yaxis2=dict(overlaying="y", side="right", title="train loss", showgrid=False))
        st.plotly_chart(style_fig(f, 380), width="stretch")
else:
    st.info("No training history yet.")

section("Inference", "", "Runtime")
st.markdown("""
1. Low-res depth → (optional) denoise → multi-scale push-pull fill → bicubic upsample → normalise.
2. Pad to a multiple of 8, run DepthNet once (no tiling) on CPU or GPU (`cuda` used automatically when available).
3. De-normalise, crop padding, clip to the measured depth range.
""")

section("Limitations — read before judging", "", "Honesty")
st.warning("""
- **Trained on procedural synthetic scenes only.** NYU Depth V2, Middlebury and KITTI could not be downloaded in the
  build environment. Synthetic scenes contain the right *kinds* of structure (planes, curved objects, thin poles,
  texture, camouflage) but not real sensor artefacts, so accuracy on real Kinect / LiDAR data is **not yet measured**.
- Trained for a short CPU budget (~40 min). A GPU run on NYU Depth V2 is expected to improve it substantially —
  see *Fine-tuning on real data* below.
- The model cannot recover objects that are completely absent from the low-res input (e.g. a 2-px pole at 8×).
""")
with st.expander("Fine-tuning on real data (Google Colab, free GPU)"):
    st.code("""# 1. In Colab: Runtime → Change runtime type → GPU
!git clone https://github.com/<you>/DEPTHMAP_AI.git && cd DEPTHMAP_AI && pip install -r requirements.txt h5py
# 2. NYU Depth V2 labelled subset (≈2.8 GB, 1449 RGB-D pairs)
!wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat -P data/
!python training/prepare_nyu.py --mat data/nyu_depth_v2_labeled.mat --out data/nyu
# 3. Fine-tune from the synthetic checkpoint
!python -m training.train --data folder --root data/nyu/train --val-root data/nyu/val \\
        --iters 20000 --batch 16 --crop 256 --resume
# 4. Evaluate on the official test split
!python -m training.evaluate --root data/nyu/test""", language="bash")
footer()
