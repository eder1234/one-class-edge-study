# FaceMoCap One-Class Surface Deformation Forecasting (v2)

This repository contains a single end-to-end script:

- `oneclass_surface_deformation_v2.py` fileciteturn0file0

It implements a **movement-specific one-class forecasting** pipeline for FaceMoCap sequences using a **template facial mesh topology**, **ROI selection from healthy subjects**, **masked resampling**, and a **masked Temporal Convolutional Network (TCN)** forecaster. The model is trained **only on healthy** sequences and produces **anomaly scores** for both healthy and pathological samples.

---

## 1) Problem setup

Each FaceMoCap recording is a time sequence of 3D markers. After removing 3 head-reference markers (dental support), the remaining 105 markers define the facial point cloud.

For each movement \(m \in \{1,\dots,5\}\), the goal is to detect deviations from normal dynamics using:

1. a **movement-specific region of interest (ROI)** on the face (markers + edges),
2. a **forecaster** trained on healthy sequences only,
3. an **anomaly score** based on forecast errors.

Movements are assumed to be:
- M1: soft blink
- M2: forced blink
- M3: “o” sound
- M4: “pu” sound
- M5: smile

---

## 2) Expected input format

### 2.1 Metadata CSV

The script expects a CSV (e.g., `facemocap_metadata.csv`) with at least:

- `complete_filepath`: path to the sample CSV file
- `participant_id`: subject identifier (used for subject-level splitting)
- `condition`: `healthy` or `pathological`
- `single_movement`: `1` if the file contains a single movement
- `facial_movement`: integer in `[1..5]`
- `valid_for_processing`: `1` if the sample is allowed for processing

Filtering (default):
- `valid_for_processing == 1`
- `single_movement == 1`
- `facial_movement ∈ {1..5}`
- `condition ∈ {healthy, pathological}`

### 2.2 Sample CSV files

Each sample CSV is read as:

- skip first `--skiprows` rows (default `5`)
- read `--usecols_count` numeric columns starting at `--usecols_start` (defaults: `start=2`, `count=324`)

These 324 values per frame represent 108 3D points:

- reshape: \((T,324) \rightarrow (T,108,3)\)

Marker indexing:
- markers `0..2`: head reference (dental support)
- markers `3..107`: facial markers (105 markers)
- the facial markers are re-indexed as `0..104` internally when using the template topology (`faces.npy`).

---

## 3) Output directory structure

A single run produces:

```
out_dir/
  run_YYYY-MM-DD_HH-MM-SS/
    00_dataset_audit/
    01_preproc_sanity/
    02_roi_zscore_selection/
    03_roi_edges/
    04_oneclass_forecasting/
      M1/
      M2/
      M3/
      M4/
      M5/
    dropped_samples.csv
    global_summary.csv
    run_args.json
    topology_info.json
```

---

## 4) Methodological pipeline (with equations)

### Stage 0 — Dataset audit (`00_dataset_audit/`)

Produces counts per movement and condition and a bar plot:
- `counts_by_movement_condition.csv`
- `participants_by_movement_condition.csv`
- `counts_bar.png`

This is a sanity check; it does not change the dataset.

---

### Stage 1 — Head-frame normalization (per frame)

To reduce global head motion effects, each frame is mapped to a **head coordinate system** derived from the 3 head reference markers \((p_0, p_1, p_2)\).

Let:

- \(p_0\) be the origin,
- \(x = \frac{p_1 - p_0}{\|p_1 - p_0\|}\),
- \(z = \frac{x \times (p_2 - p_0)}{\|x \times (p_2 - p_0)\|}\),
- \(y = z \times x\).

Then the rotation matrix (columns are axes) is:

```latex
R = [x \; y \; z] \in \mathbb{R}^{3\times 3}
```

A facial marker \(q\) (in world coordinates) is mapped to head coordinates by:

```latex
q^{(head)} = R^\top (q - p_0)
```

If the head frame is invalid (e.g., degenerate configuration), the frame is marked as missing.

---

### Stage 2 — ROI marker selection by healthy z-scores (`02_roi_zscore_selection/`)

This stage uses **healthy samples only** to select the most movement-relevant markers.

#### 2.1 Per-sample per-marker peak displacement

For a marker \(j\) in head coordinates over time, define a rest pose as the mean of its first \(K\) valid frames:

```latex
r_j = \frac{1}{K} \sum_{t \in \mathcal{T}_{j,K}} q_{t,j}^{(head)}
```

Then the displacement time series:

```latex
d_{t,j} = \| q_{t,j}^{(head)} - r_j \|_2
```

A robust peak displacement is extracted using a high percentile (default 95th):

```latex
p_j = \mathrm{percentile}_{95}( \{ d_{t,j} : t \in \mathcal{T}_j \} )
```

#### 2.2 Movement-wise marker means and global marker statistics

Let \(m\) denote a movement, and let \(\mu_m(j)\) be the mean peak displacement of marker \(j\) across healthy samples for movement \(m\).
Let \(\mu_g(j)\) and \(\sigma_g(j)\) be the global mean and standard deviation of marker \(j\) across all movements.

#### 2.3 Z-score per movement and marker

```latex
z(m,j) = \frac{\mu_m(j) - \mu_g(j)}{\sigma_g(j) + \varepsilon}
```

Markers are selected as the top \(K\) indices with highest \(z(m,j)\) for each movement (default `--K_markers 25`).

Artifacts:
- `stepA_marker_peaks_long.csv`: long-format peaks table
- `zscore_table.csv`: z-score table (movement × marker)
- `roi_markers_topK.json` + `roi_markers_Mk.csv`: selected markers
- optional visualization: `top_roi_markers_visualization.png` (if the template file path is valid)

---

### Stage 3 — ROI edges from template topology (`03_roi_edges/`)

The file `faces.npy` is a triangular mesh connectivity over the 105 facial markers (indices \(0..104\)). Each triangle \((a,b,c)\) yields three undirected edges:
\((a,b)\), \((b,c)\), \((c,a)\).

Edges are deduplicated after sorting endpoints.

```latex
E = \mathrm{unique}\Big( \{ \{a,b\}, \{b,c\}, \{c,a\} : (a,b,c)\in \mathrm{faces} \} \Big)
```

For each movement \(m\), start from the ROI marker set \(S_m\) and keep only edges fully inside the set:

```latex
E_m = \{ (u,v)\in E : u\in S_m \land v\in S_m \}
```

If `|E_m|` is too small (`--min_roi_edges`, default 40), the marker set is expanded by `--expand_hops` neighbor hops in the mesh adjacency.

Artifacts:
- `roi_edges_Mk.npy`: final ROI edges per movement
- `roi_markers_final_Mk.csv`: final marker list after expansion
- `roi_edge_stats.json`: counts and settings

---

### Stage 4 — Per-sample edge features and active-window cropping

For each sample and movement, a **neutral frame** is selected in the first 10% of frames as the frame with:
- valid head reference markers, and
- maximal number of valid facial markers (ties broken by earliest frame).

#### 4.1 Edge-length deformation feature (vs neutral)

For an edge \(e=(i,j)\), define neutral length \(d_0(e)\) at the neutral frame \(t_0\):

```latex
d_0(e) = \| q_{t_0,i}^{(head)} - q_{t_0,j}^{(head)} \|_2
```

At time \(t\), define \(d_t(e)\) similarly. The feature can be absolute or relative:

- absolute (`--edge_feature abs`):
```latex
x_t(e) = d_t(e) - d_0(e)
```

- relative (`--edge_feature rel`, default):
```latex
x_t(e) = \frac{d_t(e) - d_0(e)}{d_0(e) + \varepsilon}
```

A binary mask \(m_t(e)\in\{0,1\}\) indicates whether both endpoints are valid at time \(t\) and at \(t_0\).

#### 4.2 Motion energy per frame

A per-frame “energy” signal is computed from available features:

```latex
\mathrm{energy}(t) = \frac{\sum_e |x_t(e)|\, m_t(e)}{\max(1, \sum_e m_t(e))}
```

#### 4.3 Active-window selection

Frames are selected where \(\mathrm{energy}(t)\) exceeds a fraction of its maximum:

```latex
\mathcal{A} = \{ t : \mathrm{energy}(t) \ge \alpha \cdot \max_t \mathrm{energy}(t) \}
```

The active window \([a,b)\) is chosen around \(\mathcal{A}\) with padding (`--win_pad`) and a minimum length (`--win_min_len`).
If \(\mathcal{A}\) is empty, a peak-centered window is used.

---

### Stage 5 — Resampling with missingness constraints (fixed length)

Each sample is resampled to a fixed length `--T_out` (default 50). Resampling is **feature-wise** with linear interpolation, but it is **masked** when interpolation spans large gaps.

Let the original valid indices for feature \(j\) be \(\{t_k\}\). For a target time \(\tau\), linear interpolation uses the nearest bracketing valid indices. The target is considered valid only if:

```latex
(t_{k+1} - t_k) \le \texttt{max_gap_frames}
```

Otherwise the target point is treated as missing (mask = 0).

This prevents “bridging” long missing segments.

---

### Stage 6 — Robust normalization (median/IQR) on healthy train only

For each movement \(m\), a robust scaler is fit on **healthy train samples only**, ignoring masked values.

For each feature dimension \(d\):

```latex
\mathrm{median}_d = \mathrm{nanmedian}(X_{\cdot,d}), \qquad
\mathrm{IQR}_d = Q_{75}(X_{\cdot,d}) - Q_{25}(X_{\cdot,d})
```

Normalization:

```latex
\tilde{X}_{t,d} = \frac{X_{t,d} - \mathrm{median}_d}{\mathrm{IQR}_d + \varepsilon}\, M_{t,d}
```

The multiplication by \(M\) keeps missing entries at zero.

The scaler statistics are saved as:
- `04_oneclass_forecasting/Mk/scaler_stats.npz`

---

### Stage 7 — Masked TCN forecaster (per movement)

A separate forecaster is trained for each movement \(m\).

#### 7.1 Model input

At each time, the model receives concatenated channels:

```latex
\mathrm{input}(t) = [\tilde{X}(t) \; || \; M(t)] \in \mathbb{R}^{2D}
```

A 1×1 Conv projects \(2D\to H\), then residual dilated 1D convolutions (dilations \(2^k\)) operate along time, followed by a 1×1 Conv producing predictions \(\hat{X}(t)\in\mathbb{R}^D\).

#### 7.2 Training loss (masked Huber)

The script trains a “next-step” version by ignoring \(t=0\) and computing a masked Huber loss for \(t\ge 1\).

Huber per element:

```latex
L_\delta(r) =
\begin{cases}
\frac{1}{2}r^2, & |r|\le \delta\\
\delta(|r|-\frac{1}{2}\delta), & |r|>\delta
\end{cases}
```

Masked loss:

```latex
\mathcal{L} =
\frac{\sum_{t=1}^{T-1}\sum_{d=1}^{D} L_\delta(\hat{X}_{t,d}-X_{t,d})\, M_{t,d}}
{\max\left(1,\sum_{t=1}^{T-1}\sum_{d=1}^{D} M_{t,d}\right)}
```

Optimization uses AdamW.

---

### Stage 8 — Anomaly scoring (motion-weighted forecast error)

For evaluation, the script computes a masked per-timestep MAE:

```latex
e_t = \frac{\sum_d |\hat{X}_{t,d} - X_{t,d}|\, M_{t,d}}{\max(1,\sum_d M_{t,d})}
\qquad (t=1,\dots,T-1)
```

It also computes a motion-derived weight \(w_t\in[0,1]\) from the resampled energy signal, downweighted by per-frame availability:

```latex
w_t \propto \mathrm{energy}(t)\times \frac{\sum_d M_{t,d}}{D}, \quad \text{then normalized to } [0,1]
```

Final anomaly score (weighted mean error):

```latex
s = \frac{\sum_{t=1}^{T-1} w_t\, e_t}{\max(\varepsilon, \sum_{t=1}^{T-1} w_t)}
```

A secondary score is also computed (peak error):
```latex
s_{\max} = \max_t e_t
```

---

### Stage 9 — Threshold selection (validation percentile, healthy-only)

For each movement \(m\), a subject-level split is computed on **healthy participant IDs**:
- train / val / test (defaults: 0.7 / 0.15 / 0.15)

The threshold is set to the `--thr_percentile` percentile (default 95th) of the **validation healthy** weighted scores:

```latex
\tau = \mathrm{percentile}_{p}( \{ s_i : i \in \text{val healthy} \} )
```

Prediction:
- anomalous if \(s > \tau\)

Per-movement artifacts:
- `scores.csv`
- `summary_at_threshold.csv`
- `run_config.json`
- figures in `figs/` and histograms `score_hist*.png`

---

## 5) Running the script

Minimal example (mirrors the header comment in the script):

```bash
python oneclass_surface_deformation_v2.py \
  --metadata facemocap_metadata.csv \
  --faces faces.npy \
  --out_dir outputs_oneclass_edges_v2 \
  --root_override "/media/rodriguez/easystore/Data_FaceMoCap" \
  --skiprows 5 --usecols_start 2 --usecols_count 324 \
  --K_markers 25 \
  --use_cache --device cuda --verbose
```

### Key arguments

- **Data parsing**
  - `--skiprows`, `--usecols_start`, `--usecols_count`
- **ROI markers**
  - `--K_markers`, `--rest_first_valid_frames`, `--marker_peak_q`
- **ROI edges**
  - `--min_roi_edges`, `--expand_hops`
- **Active window**
  - `--win_thr_frac`, `--win_min_len`, `--win_pad`
- **Resampling**
  - `--T_out`, `--max_gap`
- **Feature definition**
  - `--edge_feature {abs,rel}`
- **Training**
  - `--epochs`, `--batch`, `--lr`, `--wd`, `--hidden`, `--tcn_layers`, `--dropout`, `--huber_delta`
- **Threshold**
  - `--thr_percentile`
- **Compute**
  - `--device {cpu,cuda}`
- **Caching**
  - `--use_cache`, `--cache_dir`

---

## 6) Reproducibility & caching

- A global seed is set by `--seed`.
- If `--use_cache` is enabled, each preprocessed sample is saved as a `.npz` keyed by:
  - file identity hash
  - movement id
  - ROI hash tag
  - preprocessing parameters (T_out, gap, feature type, window params, read params)

This is intended to prevent mixing caches across incompatible runs.

---

## 7) Practical constraints and known caveats

These are properties of the current code (not future work):

1. **GitHub math rendering**: GitHub Markdown does not reliably render LaTeX in README without additional tooling. Equations are provided as LaTeX blocks for copy/paste into a paper.
2. **ROI visualization path**: the ROI marker visualization uses a hard-coded `template_path` in the script. If that file does not exist in your environment, the plot is skipped with a warning.
3. **Head-frame failures**: if the three head markers become invalid or degenerate, the entire frame’s head transform fails (facial points become NaN → masked out).
4. **Interpolation policy**: the resampling explicitly avoids bridging gaps larger than `--max_gap`. This can increase missingness in sequences with long dropout.
5. **One-class assumption**: only healthy samples are used for model fitting and threshold estimation; pathological samples are only for scoring.

---

## 8) Citation / attribution

If you use this pipeline in academic writing, you likely want to cite:
- the FaceMoCap dataset (your internal reference)
- masked forecasting / one-class anomaly detection literature
- temporal convolutional networks (TCN)

This README intentionally does not include external citations; add them to match your target venue.

---

## 9) License

Add your preferred license (e.g., MIT, Apache-2.0) in `LICENSE`. If your data are restricted (clinical context), clarify usage restrictions in a `DATA.md`.
