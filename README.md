# FaceMoCap One-Class Surface Deformation Forecasting (v2)

This repository contains an end-to-end pipeline for one-class (healthy-only) anomaly detection on FaceMoCap 3D facial motion recordings. It builds movement-specific regions of interest (ROIs) from healthy data, extracts deformation features on a fixed facial topology, trains a movement-wise forecaster on healthy samples only, and produces anomaly scores for both healthy and pathological samples.

## What’s inside

- `oneclass_surface_deformation_v2.py` — main script (data audit → ROI selection → feature extraction → one-class training → scoring → reports)
- `faces.npy` — facial topology (triangulation over the 105 facial markers), used to derive ROI edges
- `facemocap_metadata.csv` — dataset index/metadata (paths + labels + filtering flags)

> Notes
> - The script expects FaceMoCap sample CSVs with 108 3D points per frame (324 numeric values) and removes the first 3 points as a non-facial support.
> - The pipeline is movement-specific (M1–M5). A separate ROI and model are built per movement.

## Requirements

- Python 3.9+ (recommended)
- Common scientific stack: `numpy`, `pandas`, `scipy`, `matplotlib`
- Deep learning: `torch` (PyTorch)
- Optional: `tqdm` (progress bars)

Install example:

```bash
pip install numpy pandas scipy matplotlib torch tqdm
```

## Data layout and inputs

### 1) Metadata CSV

The script expects a metadata CSV containing at least:

- `complete_filepath`: absolute or relative path to a sample CSV
- `participant_id`: subject identifier (used for subject-level splitting)
- `condition`: `healthy` or `pathological`
- `single_movement`: `1` if the file contains a single movement
- `facial_movement`: integer in `[1..5]`
- `valid_for_processing`: `1` if the sample is allowed for processing

Only rows that pass filtering are used.

### 2) Sample CSV format

Each sample CSV is read with:

- `--skiprows` (default 5)
- `--usecols_start` (default 2)
- `--usecols_count` (default 324)

Those 324 values per frame represent 108 points × (x,y,z). The first 3 points are removed, leaving 105 facial points.

### 3) Topology file (`faces.npy`)

`faces.npy` should contain triangular connectivity over the 105 facial markers. The script converts triangles into a deduplicated set of undirected edges and then keeps edges fully contained in each movement ROI.

## What the pipeline does (high level)

For each movement:

1. Audit the dataset (counts by movement and condition).
2. Normalize frames using head reference markers (reduces rigid head motion).
3. Select ROI markers from healthy data using movement-specific statistics.
4. Build ROI edges from the facial topology (expand ROI if too small).
5. Extract edge deformation features relative to a neutral frame.
6. Crop an active temporal window where motion is strongest.
7. Resample to a fixed length with missingness-aware interpolation.
8. Normalize robustly using healthy train data only.
9. Train a movement-specific one-class forecaster.
10. Score sequences with forecast errors and choose thresholds from healthy validation data.
11. Export per-movement reports, plots, and a global summary.

## Running

Minimal example:

```bash
python oneclass_surface_deformation_v2.py   --metadata facemocap_metadata.csv   --faces faces.npy   --out_dir outputs_oneclass_edges_v2   --root_override "/media/rodriguez/easystore/Data_FaceMoCap"   --skiprows 5 --usecols_start 2 --usecols_count 324   --K_markers 25   --use_cache --device cuda --verbose
```

### Common options

- Data parsing: `--skiprows`, `--usecols_start`, `--usecols_count`, `--root_override`
- ROI selection: `--K_markers`, `--rest_first_valid_frames`, `--marker_peak_q`
- ROI edges: `--min_roi_edges`, `--expand_hops`
- Active window: `--win_thr_frac`, `--win_min_len`, `--win_pad`
- Resampling: `--T_out`, `--max_gap`
- Feature type: `--edge_feature {abs,rel}`
- Training: `--epochs`, `--batch`, `--lr`, `--wd`, `--hidden`, `--tcn_layers`, `--dropout`
- Thresholding: `--thr_percentile`
- Compute: `--device {cpu,cuda}`
- Caching: `--use_cache`, `--cache_dir`
- Reproducibility: `--seed`

## Outputs

A run creates a timestamped folder under `--out_dir`, typically containing:

- dataset audit tables/plots
- ROI marker and edge definitions per movement
- preprocessing sanity checks
- per-movement training logs and saved model artifacts
- per-movement score files and threshold summaries
- a global run summary and the CLI args used

## Reproducibility and caching

- Subject-level splits are performed using `participant_id`.
- If caching is enabled (`--use_cache`), intermediate preprocessed arrays are stored so repeated runs with identical settings are faster.
- Use `--seed` for reproducible splitting and training initialization.

## Known limitations

- If the head reference markers are missing/degenerate for a frame, that frame is effectively unusable for head-normalized features.
- Very long missing segments can reduce usable signal after gap-aware interpolation.
- The ROI marker visualization (if enabled in the script) may depend on an optional template asset path; missing assets are handled with warnings.

## License

Add a `LICENSE` file (MIT/Apache-2.0/etc.) depending on your intended distribution.

## Contact

For questions about data formatting or expected metadata fields, open an issue or contact the repository owner.
