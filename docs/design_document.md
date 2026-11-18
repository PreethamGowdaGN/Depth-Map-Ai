# DEPTHMAP AI — Design Document

**Problem Statement 4 (P4): Edge-Preserving Depth Map Upsampling and Refinement**
Computer Vision Hackathon 2026

---

## 1. Problem formulation

Given

* a low-resolution depth map $D_{LR} \in \mathbb{R}^{h \times w}$ with a validity mask $M_{LR} \in \{0,1\}^{h\times w}$
  (0 = missing measurement), possibly noisy,
* an optional high-resolution RGB guide $I \in [0,1]^{H \times W \times 3}$ with $H = s\,h,\ W = s\,w,\ s \in \{2,4,8\}$,

estimate a dense depth map $\hat D \in \mathbb{R}^{H\times W}$ that minimises error to the true depth $D^*$ **and**
keeps depth discontinuities at the right place, without

* *flying pixels* — values between foreground and background depth that exist on neither surface,
* *texture copying* — depth edges created by colour edges that have no geometric cause,
* *oversmoothing* — blurred boundaries and thin structures.

**Sensor model used for evaluation** (`src/preprocessing.py::degrade_depth`): point-sampling
$D_{LR}[i,j] = D^*[s i + \lfloor s/2\rfloor,\ s j + \lfloor s/2 \rfloor]$ (no anti-alias blur — real low-res
sensors do not average across objects), multiplicative Gaussian noise $\sigma \cdot \max(D, 0.1)$, blob-shaped
holes, and optional random sparsification.

## 2. System design

* **Single process Streamlit app** (`app.py`, `views/`) — chosen for deployability (one command, free hosting,
  no separate backend). All computation is synchronous NumPy/OpenCV/PyTorch on CPU.
* **One engine entry point**: `src/hybrid.py::reconstruct(method, lr, mask, scale, rgb, settings)` is used by the
  UI, the tests and every evaluation script, so what is measured is exactly what is demonstrated.
* **Depth is data, not an image**: loaders keep 16-bit / float values, convert to metres with an explicit unit scale,
  build a validity mask from 0 / NaN / ±inf, and flag 8-bit or colour depth files as *relative*.
* **Validity masks everywhere**: interpolation uses normalised convolution, filters ignore invalid samples, metrics
  evaluate only on pixels valid in both GT and prediction and report *coverage* separately.
* **Settings** (`Settings` dataclass) are shared by all methods — see the mapping table in §5. Identical settings
  are used for every method in comparisons and benchmarks.
* **Performance budget**: output capped at 1.2 MP on the hosted app (rubric: response ≈ 10 s).

See `docs/architecture.md` for diagrams.

## 3. Algorithms

### 3.1 Interpolation baselines (`src/interpolation.py`)
Nearest, bilinear, bicubic via `cv2.remap` with sampling coordinates that exactly match the sensor model.
Missing samples are excluded: $\hat D = \text{interp}(D\cdot M) / \text{interp}(M)$, invalid where support < 0.5.
Bicubic overshoot is clipped to the measured range.

### 3.2 Guided filter (`src/filters.py::guided_filter`)
He et al.: in every $(2r+1)^2$ window $q = a^\top I + b$ with $a = (\Sigma_I + \varepsilon U)^{-1}\operatorname{cov}(I,p)$
(full 3×3 colour covariance, solved per pixel), $b = \bar p - a^\top \bar I$. Input $p$ = bicubic of the hole-filled
LR depth; $r = s+1$; $\varepsilon = 10^{-1.5-2.5e}$ for edge-preservation $e\in[0,1]$. Without RGB the bicubic depth
guides itself.

### 3.3 Joint bilateral upsampling (`src/filters.py::joint_bilateral_upsample`)
Kopf et al.: $\hat D(p) = \frac{\sum_{q} D_{LR}(q)\,M(q)\,f_s(\lVert p_\downarrow - q\rVert)\,f_r(\lVert I(p) - I(q_\uparrow)\rVert)}{\sum_q M(q) f_s f_r}$,
Gaussian $f_s$ ($\sigma_s$ = 1 LR px) and $f_r$ ($\sigma_r = 0.25 - 0.2e$), 4×4 LR neighbourhood, fully vectorised.

### 3.4 Edge detection (`src/edge_detection.py`)
Sobel magnitude; Canny on RGB and on normalised depth; **relative depth gradient** $|\nabla D| / D$ (scale invariant);
**depth discontinuities** = max 4-neighbour jump > 4 % of local depth (slanted planes are not flagged);
**structure edges** = RGB Canny edges inside a dilated band around depth discontinuities of the current estimate —
the key texture-copy guard.

### 3.5 Noise removal (`src/filters.py::denoise_depth`)
Mask-aware 3×3/5×5 median (removes speckle/outliers) then a bilateral filter whose range kernel is on *depth values*
($\sigma_r$ = 1–7 % of median depth): flat regions are smoothed, depth jumps are not, and RGB is not involved
(no texture can leak in).

### 3.6 Hole filling (`src/hole_filling.py`)
`guided_fill`: iterative normalised convolution — each hole pixel adjacent to known data takes a spatial × colour
weighted average of valid neighbours; repeated 12× it grows inward without crossing colour boundaries. Leftovers are
finished by `push_pull_fill`: a pyramid of (sum, count) that is averaged until every cell has support, then pulled back
into the holes only.

### 3.7 Boundary-aware refinement
*Depth-guided residual correction* (hybrid step 4, §5) keeps every correction on its own side of a depth edge.
*Flying-pixel snapping* (`src/refinement.py`): inside a band of 1–4 px around discontinuities, pixels whose value lies
well between the two surfaces are replaced by the **joint weighted median** of their 7×7 neighbourhood (spatial ×
colour weights, 0.15× for uncertain band pixels). A median returns a depth that exists on one side, so ramps collapse
into clean steps. Useful for classical outputs; off by default in the hybrid (see ablation).

## 4. Model architecture — DepthNet (`models/depth_model.py`)

| Part | Detail |
|---|---|
| Inputs | RGB (3 ch), normalised bicubic init depth + confidence (2 ch), all at output resolution |
| Encoders | RGB and depth branches, 4 scales (1, ½, ¼, ⅛), channels 16/32/64/96, strided conv + residual blocks |
| Fusion | `GatedFusion` at every scale: $f = f_d + \sigma(W[f_{rgb}, f_d]) \odot P f_{rgb}$ |
| Bottleneck | 2 residual blocks at ⅛ (receptive field for large holes) |
| Decoder | bilinear up-sampling + skip concatenation at ¼, ½, 1 |
| Output | coarse-to-fine residual heads; $\hat D = D_{init} + r_0$, $r_0 = h_0 + \uparrow r_1$, $r_1 = h_1 + \uparrow r_2$ |
| Size | 1.52 M parameters, ≈ 6 MB checkpoint |

**Normalisation**: per image, $(D - \min)/\max(\max-\min,\ 0.3\cdot\text{mean})$ of the LR measurements (the floor
prevents explosion on nearly flat crops — found during development).
**Loss**: boundary-weighted L1 (×4 within 2 px of GT edges) + 0.5·gradient L1, plus 0.25× the same on the ½ and ¼
outputs (deep supervision).
**Training data**: procedural RGB-D scenes (`src/synthetic.py`; tilted wall + floor, boxes, ellipsoids, polygons,
thin poles, textures, camouflage, Lambertian shading). Seeds 0–99 999 train, 100 000–100 999 val, ≥ 200 000 test.
Random degradation per sample: scale ∈ {2,4,8}, noise, holes, sparsity; augmentation: flip, colour jitter, global depth
scale, RGB dropout. AdamW, one-cycle LR 5·10⁻⁴, gradient clipping, batch 8, 128×128 crops, 2 600 iterations (24 min on 2 CPU cores). A first attempt at LR 2·10⁻³ diverged; it was restarted at 5·10⁻⁴ with a bad-batch guard.

`FolderRGBD` + `training/prepare_nyu.py` provide the same pipeline for NYU Depth V2 (or any RGB + 16-bit depth folder).

## 5. Hybrid pipeline (`src/hybrid.py::_hybrid`)

Final design (defaults in `Settings`):

1. **Initial interpolation** — mask-aware bicubic of the push-pull-filled LR depth (also the CNN's input encoding).
2. **Deep prediction** — DepthNet on the *raw* LR depth, its validity mask and RGB. It denoises, completes holes and
   places edges with RGB help through its learned gates.
3. **Edge analysis** — depth discontinuities of the CNN output (shown in the UI; drive optional stages).
4. **Edge-aware data-fidelity correction** — residual at every real measurement
   $r(q) = D_{LR}(q) - D_{cnn}(s q + \lfloor s/2 \rfloor)$, upsampled with joint bilateral upsampling whose range kernel
   is the *predicted depth* (normalised), $\sigma_r = 0.04 - 0.03e$, $\sigma_s = 1 + 2n$, radius 3; output
   $\hat D = D_{cnn} + \mathrm{JBU}_{D_{cnn}}(r)$.
   * Removes low-frequency CNN bias and pulls the result towards the actual sensor readings.
   * Corrections cannot cross depth edges (range kernel), and RGB is not used here, so texture cannot be copied.
5. *(optional, off)* RGB-JBU fusion within ~2 px of structure edges (RGB Canny ∩ depth-discontinuity band).
6. *(optional, off)* weighted-median flying-pixel snapping — only pixels whose value lies > 20 % of the local jump
   away from both surfaces are replaced.

If no trained checkpoint exists, an RGB joint-bilateral result replaces step 2 and the UI says so.

### How the design was chosen (validation data only)

The first design fused CNN and RGB-JBU near structure edges and then snapped boundaries. Once DepthNet was trained,
measurements on **validation** scenes (seeds 100 000–100 007, 2/4/8×) showed:

| Variant (validation, mean of 24 runs) | RMSE | Boundary RMSE | Edge F1 |
|---|---|---|---|
| DepthNet only | 0.1192 | 0.3118 | 0.9585 |
| DepthNet + median snapping (original) | 0.1323 | 0.3495 | 0.9073 |
| DepthNet + RGB-JBU fusion (original) | 0.2667 | 0.7297 | 0.9144 |
| DepthNet + depth-guided residual correction | 0.1147 | 0.3000 | 0.9587 |

We also found that classical pre-denoising of the CNN's input doubled its error (it was trained on raw noisy depth),
and that the residual must use the raw measurements. The final design follows these measurements; the test benchmark
was run only afterwards. The ablation in `eval_P4_TeamID.md` repeats the comparison on test scenes.

### Settings mapping
| UI setting | Classical methods | DepthNet | Hybrid |
|---|---|---|---|
| Noise reduction | median + depth-bilateral pre-denoise | — (raw input) | wider residual filter (σ_s) |
| Edge preservation | guided ε, JBU σ_r | — | narrower residual range kernel (σ_r) |
| Hole filling | guided fill / holes stay empty | holes stay empty if off | holes stay empty if off |
| RGB guidance | guide on/off | RGB input zeroed (trained with RGB dropout) | same as DepthNet |

## 6. Evaluation methodology

* **Test data**: held-out procedural scenes (seeds ≥ 200 500 for the benchmark; disjoint from training/validation and
  from the demo samples), 320×240, degraded with σ = 0.01, 10 % holes; 2×, 4×, 8×. Identical inputs and settings for
  all 7 methods (`evaluation/benchmark.py`).
* **Metrics** (`src/metrics.py`), on pixels valid in GT ∧ prediction:
  RMSE, MAE (m); PSNR with peak = GT range; SSIM on GT-range-normalised depth averaged over valid pixels; δ<1.25;
  **Edge F1** between predicted and GT discontinuity maps (2 px tolerance) = edge preservation;
  **Boundary RMSE** within 3 px of GT discontinuities = boundary accuracy;
  **Texture-copy index** = mean relative depth gradient of the prediction on GT-flat pixels that carry RGB edges;
  coverage; latency (wall-clock) and peak memory (tracemalloc; RSS delta for PyTorch).
* **Ablation** (`evaluation/ablation.py`): hybrid with one component removed at a time, 4×.
* **Failure cases** (`evaluation/failure_analysis.py`): eight stress conditions, bicubic vs JBU vs hybrid, with
  visual sheets in `results/failure_cases/`.
* **Real data**: `training/evaluate.py` runs the same protocol on any RGB-D folder (e.g. NYU test split). Not run in this
  build because the datasets could not be downloaded in the build environment.

## 7. Limitations & ethics

* Model trained on synthetic data only; real-sensor performance unmeasured.
* Hole filling and the CNN can produce plausible but wrong geometry in large holes — validity masks are exported with the
  `.npy` (NaN) and PNG (0) outputs; downstream safety-critical use needs real validation.
* Uploaded data is processed in memory and not stored; no credentials are used or committed.
