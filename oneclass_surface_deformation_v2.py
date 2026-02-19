#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FaceMoCap one-class forecasting (v2):
- Template topology faces.npy built on 105 FACIAL markers (after dropping the 3 head refs).
- Active markers per movement are selected using the user's z-score strategy (healthy only).
- Movement-specific ROI edges are derived from active markers.
- Active-window cropping is applied (per sample, per movement) based on ROI motion energy.
- Masked TCN forecaster per movement (M1..M5), trained on healthy train/val, tested on healthy+pathological.
- Anomaly score uses motion-weighted mean forecasting error.

Expected CSV layout:
- Read 324 numeric columns -> 108 markers (x,y,z)
- First 3 markers = head reference (dental support)
- Next 105 markers = facial markers (these align with faces.npy indices 0..104)

Outputs:
run_xxx/
  00_dataset_audit/
  01_preproc_sanity/
  02_roi_zscore_selection/
  03_roi_edges/
  04_oneclass_forecasting/Mk/...

Run example:
  python oneclass_surface_deformation_v2.py \
    --metadata facemocap_metadata.csv \
    --faces faces.npy \
    --out_dir outputs_oneclass_edges_v2 \
    --root_override "/media/rodriguez/easystore/Data_FaceMoCap" \
    --skiprows 5 --usecols_start 2 --usecols_count 324 \
    --K_markers 25 \
    --use_cache --device cuda --verbose

Notes:
- This script assumes faces.npy indices are 0..104 (105 facial markers).
- No "remove last 4 markers" in v2 (because your faces.npy was rebuilt with 105).
"""

from __future__ import annotations

import os
import re
import json
import math
import time
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import hashlib

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# --------------------------
# Utility / Repro
# --------------------------
def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_roi_selection_integrated(template_pts: np.ndarray, roi_dict: Dict[int, List[int]], out_png: str, K: int):
    movements = {1: "soft blink", 2: "forced blink", 3: "o sound", 4: "pu sound", 5: "board smile"}
    fig = plt.figure(figsize=(25, 6))
    ELEV, AZIM = -70.75319029406856, -90.04874299604377
    
    for i, mv_id in enumerate(range(1, 6)):
        ax = fig.add_subplot(1, 5, i + 1, projection='3d')
        active_indices = roi_dict.get(mv_id, [])
        ax.scatter(template_pts[:, 0], template_pts[:, 1], template_pts[:, 2], 
                   c='lightgray', s=15, alpha=0.5, depthshade=True)
        if active_indices:
            valid = [idx for idx in active_indices if idx < len(template_pts)]
            ax.scatter(template_pts[valid, 0], template_pts[valid, 1], template_pts[valid, 2], 
                       c='blue', s=50, edgecolors='black', alpha=1.0)
        ax.set_title(f"{movements[mv_id].capitalize()}", fontsize=14, fontweight='bold')
        ax.view_init(elev=ELEV, azim=AZIM)
        ax.set_axis_off()
        ax.set_box_aspect([1, 1, 0.5]) 

    plt.suptitle(f"Top {K} ROI Markers per Movement", fontsize=18, y=1.05)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def now_str() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


# --------------------------
# Geometry: head coordinates
# --------------------------
def make_head_frame(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, eps: float = 1e-9) -> Optional[np.ndarray]:
    """
    Build a 3x3 rotation matrix R whose columns are unit axes (x,y,z) of the head frame.
    origin = p0
    x axis along p1 - p0
    z axis = normalize(cross(x, p2-p0))
    y axis = cross(z, x)
    """
    x = p1 - p0
    nx = np.linalg.norm(x)
    if not np.isfinite(nx) or nx < eps:
        return None
    x = x / nx

    v = p2 - p0
    nv = np.linalg.norm(v)
    if not np.isfinite(nv) or nv < eps:
        return None

    z = np.cross(x, v)
    nz = np.linalg.norm(z)
    if not np.isfinite(nz) or nz < eps:
        return None
    z = z / nz

    y = np.cross(z, x)
    ny = np.linalg.norm(y)
    if not np.isfinite(ny) or ny < eps:
        return None
    y = y / ny

    R = np.stack([x, y, z], axis=1)  # columns
    if not np.all(np.isfinite(R)):
        return None
    return R


def facial_world_to_head(facial_world: np.ndarray, head3_world: np.ndarray) -> np.ndarray:
    """
    facial_world: (Nf,3)
    head3_world: (3,3) (p0,p1,p2)
    returns facial in head coords: (Nf,3), NaNs if head invalid
    """
    if head3_world.shape != (3, 3):
        raise ValueError("head3_world must be (3,3)")
    if not np.isfinite(head3_world).all():
        return np.full_like(facial_world, np.nan, dtype=np.float32)
    p0, p1, p2 = head3_world[0], head3_world[1], head3_world[2]
    R = make_head_frame(p0, p1, p2)
    if R is None:
        return np.full_like(facial_world, np.nan, dtype=np.float32)
    X = facial_world - p0[None, :]
    return (R.T @ X.T).T.astype(np.float32)


# --------------------------
# Template topology -> edges
# --------------------------
def faces_to_undirected_edges(faces: np.ndarray) -> np.ndarray:
    f = faces.astype(np.int64)
    e01 = f[:, [0, 1]]
    e12 = f[:, [1, 2]]
    e20 = f[:, [2, 0]]
    edges = np.vstack([e01, e12, e20])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    return edges.astype(np.int32)


def build_edge_adjacency(edges: np.ndarray, n_verts: int) -> List[List[int]]:
    adj = [[] for _ in range(n_verts)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def roi_edges_from_markers(
    edges_all: np.ndarray,
    active_markers: List[int],
    n_verts: int,
    min_edges: int,
    adj: Optional[List[List[int]]] = None,
    expand_hops: int = 1
) -> Tuple[np.ndarray, List[int]]:
    """
    Keep edges whose endpoints are in active_markers.
    If too few edges, expand markers by neighbor hops in the mesh adjacency.
    Returns (roi_edges, final_marker_list).
    """
    S = set(active_markers)
    def edges_for(Sset):
        m = np.array([(a in Sset) and (b in Sset) for a, b in edges_all], dtype=bool)
        return edges_all[m]

    roi = edges_for(S)
    if roi.shape[0] >= min_edges or adj is None or expand_hops <= 0:
        return roi.astype(np.int32), sorted(S)

    # expand markers
    expanded = set(S)
    frontier = set(S)
    for _ in range(expand_hops):
        nxt = set()
        for v in frontier:
            for nb in adj[v]:
                nxt.add(nb)
        nxt -= expanded
        expanded |= nxt
        frontier = nxt
        roi = edges_for(expanded)
        if roi.shape[0] >= min_edges:
            break

    return roi.astype(np.int32), sorted(expanded)


# --------------------------
# Resampling with NaNs (limited interpolation)
# --------------------------
def resample_feature_matrix_with_mask(
    X: np.ndarray,         # (T, D), NaNs allowed
    M: np.ndarray,         # (T, D), 0/1
    T_out: int,
    max_gap_frames: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample each feature independently to fixed length T_out.
    Linear interpolation, but target points inside a gap > max_gap_frames are marked missing.
    No extrapolation treated as valid.
    """
    T, D = X.shape
    if T == 0:
        return np.zeros((T_out, D), np.float32), np.zeros((T_out, D), np.uint8)

    t_src = np.arange(T, dtype=np.float32)
    t_tgt = np.linspace(0, T - 1, T_out, dtype=np.float32)

    Xr = np.zeros((T_out, D), dtype=np.float32)
    Mr = np.zeros((T_out, D), dtype=np.uint8)

    for j in range(D):
        mj = (M[:, j] > 0) & np.isfinite(X[:, j])
        idx = np.where(mj)[0]
        if idx.size < 2:
            continue
        y = X[idx, j].astype(np.float32)
        yi = np.interp(t_tgt, t_src[idx], y).astype(np.float32)

        valid = np.zeros(T_out, dtype=bool)
        pos = np.searchsorted(idx, t_tgt, side="left")
        for k in range(T_out):
            p = pos[k]
            if p == 0 or p >= idx.size:
                continue
            left = idx[p - 1]
            right = idx[p]
            gap = right - left
            if gap <= max_gap_frames:
                valid[k] = True

        Mr[:, j] = valid.astype(np.uint8)
        Xr[:, j] = yi * Mr[:, j].astype(np.float32)

    return Xr, Mr


# --------------------------
# Robust normalization (median/IQR ignoring mask)
# --------------------------
@dataclass
class RobustScaler:
    median: np.ndarray  # (D,)
    iqr: np.ndarray     # (D,)

    def transform(self, X: np.ndarray, M: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        Xn = (X - self.median[None, :]) / (self.iqr[None, :] + eps)
        return (Xn * M.astype(np.float32)).astype(np.float32)


def fit_robust_scaler(train_X: List[np.ndarray], train_M: List[np.ndarray], eps: float = 1e-6) -> RobustScaler:
    all_vals = []
    for X, M in zip(train_X, train_M):
        all_vals.append(np.where(M.astype(bool), X, np.nan))
    A = np.concatenate(all_vals, axis=0)

    med = np.nanmedian(A, axis=0).astype(np.float32)
    q25 = np.nanpercentile(A, 25, axis=0).astype(np.float32)
    q75 = np.nanpercentile(A, 75, axis=0).astype(np.float32)
    iqr = (q75 - q25).astype(np.float32)
    iqr = np.where(np.isfinite(iqr) & (iqr > eps), iqr, 1.0).astype(np.float32)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
    return RobustScaler(median=med, iqr=iqr)


# --------------------------
# Feature extraction per sample
# --------------------------
@dataclass
class SampleProcessed:
    participant_id: str
    condition: str                # "healthy" or "pathological"
    movement: int                 # 1..5
    filepath: str

    X: np.ndarray                 # (T_out, D) float32 (normalized later)
    M: np.ndarray                 # (T_out, D) uint8
    W: np.ndarray                 # (T_out,) float32 motion weights (0..1)

    neutral_frame_idx: int
    completeness_first10pct: Dict[str, float]


def load_sample_raw_points(csv_path: str, skiprows: int, usecols_start: int, usecols_count: int) -> np.ndarray:
    usecols = list(range(usecols_start, usecols_start + usecols_count))
    df = pd.read_csv(
        csv_path,
        skiprows=skiprows,
        header=None,
        usecols=usecols,
        engine="python",
        na_values=["", " ", "NA", "NaN", "nan", "None"],
    )
    return df.to_numpy(dtype=np.float32)  # (T,324)


def map_324_to_points_108(A324: np.ndarray) -> np.ndarray:
    T, C = A324.shape
    if C != 324:
        raise ValueError(f"Expected 324 cols, got {C}")
    return A324.reshape(T, 108, 3).astype(np.float32)


def pick_neutral_frame(P108: np.ndarray, first_frac: float = 0.10) -> Tuple[int, Dict[str, float]]:
    """
    P108: (T,108,3)
    head = 0..2
    facial = 3..107 (105)
    Choose most complete facial among first 10%, requiring head valid.
    """
    T = P108.shape[0]
    K = max(1, int(math.ceil(T * first_frac)))

    head = P108[:K, :3, :]
    facial = P108[:K, 3:, :]  # (K,105,3)

    head_ok = np.isfinite(head).all(axis=(1, 2))
    facial_ok = np.isfinite(facial).all(axis=2)
    facial_count = facial_ok.sum(axis=1).astype(int)

    idx_candidates = np.where(head_ok)[0]
    if idx_candidates.size > 0:
        best = idx_candidates[np.argmax(facial_count[idx_candidates])]
        maxv = facial_count[best]
        bests = idx_candidates[facial_count[idx_candidates] == maxv]
        best = int(bests.min())
    else:
        best = int(np.argmax(facial_count))

    stats = {
        "T": float(T),
        "K_first10pct": float(K),
        "best_idx": float(best),
        "best_facial_valid": float(facial_count[best]),
        "head_ok_rate_first10pct": float(head_ok.mean()),
        "facial_valid_med_first10pct": float(np.median(facial_count)),
        "facial_valid_min_first10pct": float(facial_count.min()),
        "facial_valid_max_first10pct": float(facial_count.max()),
    }
    return best, stats


def compute_edge_features_vs_neutral(
    facial_head: np.ndarray,  # (T,105,3) NaNs allowed
    edges: np.ndarray,        # (D,2) indices in 0..104
    neutral_idx: int,
    edge_feature: str = "rel",
    eps: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    X(t,e) = delta edge length vs neutral.
    Mask requires endpoints valid at t AND neutral endpoints valid.
    """
    T = facial_head.shape[0]
    D = edges.shape[0]
    i = edges[:, 0].astype(np.int64)
    j = edges[:, 1].astype(np.int64)

    V0 = facial_head[neutral_idx]
    v0_ok = np.isfinite(V0).all(axis=1)
    e0_ok = v0_ok[i] & v0_ok[j]

    d0 = np.full(D, np.nan, dtype=np.float32)
    if e0_ok.any():
        diff0 = V0[i[e0_ok]] - V0[j[e0_ok]]
        d0[e0_ok] = np.linalg.norm(diff0, axis=1).astype(np.float32)

    X = np.zeros((T, D), dtype=np.float32)
    M = np.zeros((T, D), dtype=np.uint8)

    for t in range(T):
        Vt = facial_head[t]
        vt_ok = np.isfinite(Vt).all(axis=1)
        et_ok = vt_ok[i] & vt_ok[j] & e0_ok & np.isfinite(d0)
        if not et_ok.any():
            continue
        diff = Vt[i[et_ok]] - Vt[j[et_ok]]
        dt = np.linalg.norm(diff, axis=1).astype(np.float32)

        if edge_feature == "abs":
            feat = dt - d0[et_ok]
        else:
            feat = (dt - d0[et_ok]) / (d0[et_ok] + eps)

        X[t, et_ok] = feat
        M[t, et_ok] = 1

    return X, M


def compute_marker_peak_displacements(
    facial_head: np.ndarray,   # (T,105,3)
    rest_first_valid_frames: int = 10,
    peak_q: float = 95.0
) -> np.ndarray:
    """
    For each marker j:
      rest pose = mean over first K frames where that marker is valid (in head coords)
      displacement(t) = ||p(t) - rest||
      peak = percentile( displacement(valid_times), peak_q )
    returns: peaks (105,) float32, NaN if insufficient data
    """
    T, N, _ = facial_head.shape
    peaks = np.full((N,), np.nan, dtype=np.float32)

    for j in range(N):
        pj = facial_head[:, j, :]  # (T,3)
        ok = np.isfinite(pj).all(axis=1)
        idx = np.where(ok)[0]
        if idx.size < max(3, rest_first_valid_frames):
            continue
        idx_rest = idx[:rest_first_valid_frames]
        rest = np.mean(pj[idx_rest], axis=0)
        d = np.linalg.norm(pj[idx] - rest[None, :], axis=1)
        if d.size < 3:
            continue
        peaks[j] = np.percentile(d, peak_q).astype(np.float32)

    return peaks


def active_window_from_energy(
    energy: np.ndarray,              # (T,) float
    thr_frac: float,
    min_len: int,
    pad: int
) -> Tuple[int, int]:
    """
    Select [a,b) indices around activity:
    - if energy has any finite, compute max
    - keep indices where energy >= thr_frac * max
    - take min/max, pad, enforce min_len, clamp to [0,T)
    """
    T = energy.shape[0]
    if T == 0 or not np.isfinite(energy).any():
        return 0, T
    e = np.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0)
    mx = float(e.max())
    if mx <= 0:
        return 0, T
    thr = thr_frac * mx
    idx = np.where(e >= thr)[0]
    if idx.size == 0:
        # fallback to peak-centered
        t0 = int(np.argmax(e))
        a = max(0, t0 - min_len // 2)
        b = min(T, a + min_len)
        a = max(0, b - min_len)
        return a, b

    a = int(idx.min()) - pad
    b = int(idx.max()) + 1 + pad
    a = max(0, a)
    b = min(T, b)
    if (b - a) < min_len:
        # expand around peak inside [a,b)
        t0 = int(np.argmax(e))
        a = max(0, t0 - min_len // 2)
        b = min(T, a + min_len)
        a = max(0, b - min_len)
    return a, b


def compute_motion_energy(
    X: np.ndarray, M: np.ndarray, mode: str = "absmean"
) -> np.ndarray:
    """
    X: (T,D), M: (T,D)
    energy per frame based on available edges.
    """
    if X.size == 0:
        return np.zeros((X.shape[0],), np.float32)
    if mode == "absmean":
        num = (np.abs(X) * M.astype(np.float32)).sum(axis=1)
        den = np.clip(M.sum(axis=1).astype(np.float32), 1.0, None)
        return (num / den).astype(np.float32)
    raise ValueError("unknown energy mode")


def normalize_weights(w: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    w = np.nan_to_num(w.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mx = float(w.max()) if w.size else 0.0
    if mx <= eps:
        return np.zeros_like(w, dtype=np.float32)
    return (w / mx).astype(np.float32)


def process_one_sample_v2(
    row: pd.Series,
    roi_edges: np.ndarray,
    args,
    cache_dir: str,
    cache_tag: str
) -> Optional[SampleProcessed]:
    """
    For one sample:
      load -> P108 -> neutral idx -> facial head coords -> edge features -> energy -> crop -> resample -> weights
    Cache is movement+roi-specific via cache_tag.
    """
    fp = str(row["complete_filepath"])
    if args.root_override:
        m = re.search(r"(.*?/Data_FaceMoCap/)(.*)$", fp)
        if m:
            fp = os.path.join(args.root_override, m.group(2))

    if not os.path.exists(fp):
        return None

    participant_id = str(row["participant_id"])
    condition = str(row["condition"]).strip().lower()
    movement = int(float(row["facial_movement"]))
    filepath = fp

    rel_fp = os.path.relpath(fp, args.root_override) if args.root_override else os.path.abspath(fp)
    rel_fp = rel_fp.replace("\\", "/")  # normalize
    fp_hash = hashlib.sha1(rel_fp.encode("utf-8")).hexdigest()[:12]

    cache_key = (
        f"{fp_hash}__mv{movement}__{cache_tag}"
        f"__T{args.T_out}__gap{args.max_gap}__feat{args.edge_feature}"
        f"__skip{args.skiprows}__uc{args.usecols_start}-{args.usecols_count}"
        f"__win{args.win_thr_frac}-{args.win_min_len}-{args.win_pad}"
    )
    cache_path = os.path.join(cache_dir, participant_id, cache_key + ".npz")

    ensure_dir(os.path.dirname(cache_path))

    if args.use_cache and os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=False)
        stats_json = z["stats_json"].tobytes().decode("utf-8")
        return SampleProcessed(
            participant_id=participant_id,
            condition=condition,
            movement=movement,
            filepath=filepath,
            X=z["X"].astype(np.float32),
            M=z["M"].astype(np.uint8),
            W=z["W"].astype(np.float32),
            neutral_frame_idx=int(z["neutral_frame_idx"]),
            completeness_first10pct=json.loads(stats_json),
        )

    try:
        A324 = load_sample_raw_points(fp, args.skiprows, args.usecols_start, args.usecols_count)
    except Exception:
        return None
    if A324.ndim != 2 or A324.shape[1] != 324:
        return None

    P108 = map_324_to_points_108(A324)
    T = P108.shape[0]
    if T < 5:
        return None

    neutral_idx, stats = pick_neutral_frame(P108, first_frac=0.10)

    head_world = P108[:, :3, :]      # (T,3,3)
    facial_world = P108[:, 3:, :]    # (T,105,3)

    facial_head = np.full_like(facial_world, np.nan, dtype=np.float32)
    for t in range(T):
        facial_head[t] = facial_world_to_head(facial_world[t], head_world[t])

    # Edge features
    X, M = compute_edge_features_vs_neutral(
        facial_head=facial_head,
        edges=roi_edges,
        neutral_idx=neutral_idx,
        edge_feature=args.edge_feature,
    )

    # Energy and active window crop
    energy = compute_motion_energy(X, M)
    a, b = active_window_from_energy(
        energy=energy,
        thr_frac=args.win_thr_frac,
        min_len=args.win_min_len,
        pad=args.win_pad
    )
    Xc = X[a:b, :]
    Mc = M[a:b, :]
    ec = energy[a:b]

    # Resample to fixed length
    Xr, Mr = resample_feature_matrix_with_mask(Xc, Mc, T_out=args.T_out, max_gap_frames=args.max_gap)

    # Resample weights (energy): simple linear interp on cropped segment, then normalize to [0,1]
    if ec.size >= 2:
        t_src = np.arange(ec.size, dtype=np.float32)
        t_tgt = np.linspace(0, ec.size - 1, args.T_out, dtype=np.float32)
        Wr = np.interp(t_tgt, t_src, ec.astype(np.float32)).astype(np.float32)
    else:
        Wr = np.zeros((args.T_out,), dtype=np.float32)

    # Combine with availability (downweight frames with no data)
    avail = (Mr.sum(axis=1) / max(Mr.shape[1], 1)).astype(np.float32)
    Wr = normalize_weights(Wr) * avail
    Wr = normalize_weights(Wr)

    sp = SampleProcessed(
        participant_id=participant_id,
        condition=condition,
        movement=movement,
        filepath=filepath,
        X=Xr,
        M=Mr,
        W=Wr,
        neutral_frame_idx=neutral_idx,
        completeness_first10pct=stats,
    )

    if args.use_cache:
        stats_json = json.dumps(sp.completeness_first10pct).encode("utf-8")
        np.savez_compressed(
            cache_path,
            X=sp.X.astype(np.float32),
            M=sp.M.astype(np.uint8),
            W=sp.W.astype(np.float32),
            neutral_frame_idx=np.int32(sp.neutral_frame_idx),
            stats_json=np.frombuffer(stats_json, dtype=np.uint8),
        )

    return sp

# --------------------------
# Z-score ROI selection (healthy only)
# --------------------------
def compute_roi_markers_zscore(
    df: pd.DataFrame,
    args,
    out_dir: str
) -> Dict[int, List[int]]:
    """
    Implements the user's relevant_markers.py logic in a consistent pipeline:
    - For each healthy sample, compute per-marker peak displacement (95th percentile) in head coords.
    - Aggregate across movements and compute z-scores per marker per movement:
        z(m,j) = (mean_m(j) - mean_global(j)) / (std_global(j) + eps)
    - Select top K markers per movement by z-score.
    """
    ensure_dir(out_dir)

    healthy = df[df["condition"] == "healthy"].copy()
    if len(healthy) == 0:
        raise RuntimeError("No healthy samples after filtering; cannot compute ROI markers.")

    rows = []
    dropped = []

    for _, row in healthy.iterrows():
        fp = str(row["complete_filepath"])
        if args.root_override:
            m = re.search(r"(.*?/Data_FaceMoCap/)(.*)$", fp)
            if m:
                fp = os.path.join(args.root_override, m.group(2))
        if not os.path.exists(fp):
            dropped.append({"filepath": fp, "reason": "missing_path"})
            continue

        try:
            A324 = load_sample_raw_points(fp, args.skiprows, args.usecols_start, args.usecols_count)
            P108 = map_324_to_points_108(A324)
        except Exception:
            dropped.append({"filepath": fp, "reason": "load_fail"})
            continue

        if P108.shape[0] < 5:
            dropped.append({"filepath": fp, "reason": "too_short"})
            continue

        # head coords (for stable marker displacement)
        head_world = P108[:, :3, :]
        facial_world = P108[:, 3:, :]   # (T,105,3)

        T = P108.shape[0]
        facial_head = np.full_like(facial_world, np.nan, dtype=np.float32)
        for t in range(T):
            facial_head[t] = facial_world_to_head(facial_world[t], head_world[t])

        peaks = compute_marker_peak_displacements(
            facial_head=facial_head,
            rest_first_valid_frames=args.rest_first_valid_frames,
            peak_q=args.marker_peak_q
        )

        mv = int(float(row["facial_movement"]))
        pid = str(row["participant_id"])

        # store row: one per marker
        for j in range(peaks.shape[0]):
            rows.append({
                "participant_id": pid,
                "movement": mv,
                "marker": j,
                "peak_disp": float(peaks[j]) if np.isfinite(peaks[j]) else np.nan,
                "filepath": fp,
            })

    feats = pd.DataFrame(rows)
    feats.to_csv(os.path.join(out_dir, "stepA_marker_peaks_long.csv"), index=False)
    pd.DataFrame(dropped).to_csv(os.path.join(out_dir, "stepA_dropped.csv"), index=False)

    # pivot: (sample, marker)
    # compute mean per movement per marker, and global mean/std per marker
    mean_m = feats.groupby(["movement", "marker"])["peak_disp"].mean().unstack("marker")
    mean_g = feats.groupby(["marker"])["peak_disp"].mean()
    std_g = feats.groupby(["marker"])["peak_disp"].std()

    # align indices
    mean_g = mean_g.reindex(mean_m.columns)
    std_g = std_g.reindex(mean_m.columns)

    eps = 1e-6
    mg = mean_g.to_numpy(dtype=np.float32)[None, :]
    sg = std_g.to_numpy(dtype=np.float32)[None, :]
    z = (mean_m.to_numpy(dtype=np.float32) - mg) / (sg + eps)

    # keep z as DataFrame with same index/columns
    z = pd.DataFrame(z, index=mean_m.index, columns=mean_m.columns)


    z.to_csv(os.path.join(out_dir, "zscore_table.csv"), index=True)

    roi: Dict[int, List[int]] = {}
    rankings = {}

    for mv in sorted(z.index.tolist()):
        zv = z.loc[mv].to_numpy(dtype=np.float32)  # (105,)
        # handle NaNs: treat as very low
        zv = np.nan_to_num(zv, nan=-1e9, posinf=-1e9, neginf=-1e9)
        top = np.argsort(-zv)[:args.K_markers].astype(int).tolist()
        roi[int(mv)] = top
        rankings[int(mv)] = [{"marker": int(i), "z": float(zv[i])} for i in top]

    # ... (existing code saving json and csv files)
    with open(os.path.join(out_dir, "roi_markers_topK.json"), "w") as f:
        json.dump({"K": args.K_markers, "roi": roi, "rankings": rankings}, f, indent=2)

    # also save per movement csv
    for mv, top in roi.items():
        pd.DataFrame({"marker": top}).to_csv(os.path.join(out_dir, f"roi_markers_M{mv}.csv"), index=False)

    # --- Robust Integration ---
    template_path = '/media/rodriguez/easystore/Data_FaceMoCap/Sujets_Sains/SIMOVI/AD01/AD01_M5.csv'
    try:
        raw_df = pd.read_csv(template_path, skiprows=args.skiprows, header=None)
        raw_pts = raw_df.iloc[:, args.usecols_start : args.usecols_start + args.usecols_count].values.reshape(-1, 108, 3)
        
        template_aligned = None
        for frame in raw_pts:
            if not np.isnan(frame[:3]).any(): # Only dental markers must be valid
                p0, p1, p2 = frame[0], frame[1], frame[2]
                pts_centered = frame - p0
                u_x = (p1 - p0) / np.linalg.norm(p1 - p0)
                u_z = np.cross(u_x, (p2 - p0))
                u_z /= np.linalg.norm(u_z)
                u_y = np.cross(u_z, u_x)
                R = np.stack([u_x, u_y, u_z])
                template_aligned = (pts_centered @ R.T)[3:] # Keep facial only
                break
        
        if template_aligned is not None:
            plot_roi_selection_integrated(template_aligned, roi, os.path.join(out_dir, "top_roi_markers_visualization.png"), args.K_markers)
    except Exception as e:
        print(f"Warning: ROI visualization failed: {e}")
    # ---------------------------

    return roi
# --------------------------
# Dataset / splitting
# --------------------------
def subject_split(ids: List[str], seed: int, ratios=(0.7, 0.15, 0.15)) -> Tuple[set, set, set]:
    ids = sorted(set(ids))
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_tr = int(round(n * ratios[0]))
    n_va = int(round(n * ratios[1]))
    tr = set(ids[:n_tr])
    va = set(ids[n_tr:n_tr + n_va])
    te = set(ids[n_tr + n_va:])
    return tr, va, te


def audit_and_plot_dataset(meta: pd.DataFrame, out_dir: str) -> None:
    ensure_dir(out_dir)
    df = meta.copy()
    df["condition"] = df["condition"].astype(str).str.lower().str.strip()
    df["single_movement"] = pd.to_numeric(df["single_movement"], errors="coerce")
    df["facial_movement"] = pd.to_numeric(df["facial_movement"], errors="coerce")
    df["valid_for_processing"] = pd.to_numeric(df["valid_for_processing"], errors="coerce")
    df = df.dropna(subset=["single_movement", "facial_movement", "valid_for_processing"])

    df["single_movement"] = df["single_movement"].astype(int)
    df["facial_movement"] = df["facial_movement"].astype(int)
    df["valid_for_processing"] = df["valid_for_processing"].astype(int)

    df = df[(df["valid_for_processing"] == 1)]
    df = df[(df["single_movement"] == 1)]
    df = df[df["facial_movement"].between(1, 5)]
    df = df[df["condition"].isin(["healthy", "pathological"])]

    ct = df.groupby(["facial_movement", "condition"])["complete_filepath"].count().unstack(fill_value=0).reset_index()
    ct.to_csv(os.path.join(out_dir, "counts_by_movement_condition.csv"), index=False)

    pt = df.groupby(["facial_movement", "condition"])["participant_id"].nunique().unstack(fill_value=0).reset_index()
    pt.to_csv(os.path.join(out_dir, "participants_by_movement_condition.csv"), index=False)

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    mv = ct["facial_movement"].values
    healthy = ct.get("healthy", pd.Series([0]*len(ct))).values
    path = ct.get("pathological", pd.Series([0]*len(ct))).values
    ax.bar(mv - 0.2, healthy, width=0.4, label="healthy")
    ax.bar(mv + 0.2, path, width=0.4, label="pathological")
    ax.set_xticks(mv)
    ax.set_xlabel("Movement (M1..M5)")
    ax.set_ylabel("Samples")
    ax.set_title("Sample counts (single_movement=1, valid_for_processing=1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "counts_bar.png"), dpi=200)
    plt.close(fig)


# --------------------------
# PyTorch datasets
# --------------------------
class SequenceDataset(Dataset):
    def __init__(self, samples: List[SampleProcessed]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return {
            "X": torch.from_numpy(s.X).float(),   # (T,D)
            "M": torch.from_numpy(s.M).float(),   # (T,D)
            "W": torch.from_numpy(s.W).float(),   # (T,)
            "participant_id": s.participant_id,
            "condition": s.condition,
            "filepath": s.filepath,
        }


# --------------------------
# Masked TCN forecaster
# --------------------------
class ResBlockTCN(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.gn1 = nn.GroupNorm(8 if channels >= 8 else 1, channels)
        self.gn2 = nn.GroupNorm(8 if channels >= 8 else 1, channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        r = x
        x = self.conv1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = F.relu(x)
        x = self.drop(x)
        return x + r


class MaskedTCNForecaster(nn.Module):
    """
    Input: (values || mask) as channels, output predicted values at each time.
    """
    def __init__(self, D: int, hidden: int = 64, layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.D = D
        self.in_proj = nn.Conv1d(2 * D, hidden, kernel_size=1)
        blocks = []
        for k in range(layers):
            blocks.append(ResBlockTCN(hidden, dilation=2 ** k, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)
        self.out = nn.Conv1d(hidden, D, kernel_size=1)

    def forward(self, x_vals: torch.Tensor, x_mask: torch.Tensor):
        # (B,T,D)
        x = torch.cat([x_vals, x_mask], dim=-1)  # (B,T,2D)
        x = x.transpose(1, 2)                    # (B,2D,T)
        x = self.in_proj(x)
        x = F.relu(x)
        x = self.blocks(x)
        y = self.out(x)                          # (B,D,T)
        return y.transpose(1, 2)                 # (B,T,D)


def masked_huber_loss(yhat: torch.Tensor, y: torch.Tensor, m: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    # next-step (t=1..)
    yhat = yhat[:, 1:, :]
    y = y[:, 1:, :]
    m = m[:, 1:, :]

    diff = yhat - y
    abs_diff = torch.abs(diff)
    quad = torch.minimum(abs_diff, torch.tensor(delta, device=diff.device))
    lin = abs_diff - quad
    huber = 0.5 * quad * quad + delta * lin

    denom = torch.clamp(m.sum(), min=1.0)
    return (huber * m).sum() / denom


def masked_mae_per_timestep(yhat: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    yhat = yhat[:, 1:, :]
    y = y[:, 1:, :]
    m = m[:, 1:, :]
    err = torch.abs(yhat - y) * m
    denom = torch.clamp(m.sum(dim=2), min=1.0)  # (B,T-1)
    return err.sum(dim=2) / denom


# --------------------------
# Visualization helpers
# --------------------------
def plot_missingness_heatmap(M: np.ndarray, out_png: str, title: str):
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.imshow(M.T, aspect="auto", interpolation="nearest")
    ax.set_xlabel("time (resampled)")
    ax.set_ylabel("edge feature index")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_feature_heatmap(X: np.ndarray, out_png: str, title: str, vmax_pctl: float = 99.0):
    A = np.abs(X.copy())
    vmax = np.nanpercentile(A, vmax_pctl) if np.isfinite(A).any() else 1.0
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.imshow(X.T, aspect="auto", interpolation="nearest", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("time (resampled)")
    ax.set_ylabel("edge feature index")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_loss_curves(train_losses: List[float], val_losses: List[float], out_png: str, title: str):
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("masked huber loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_score_hist(scores_h: np.ndarray, scores_p: np.ndarray, thr: float, out_png: str, title: str):
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.hist(scores_h, bins=30, alpha=0.7, label="healthy")
    ax.hist(scores_p, bins=30, alpha=0.7, label="pathological")
    ax.axvline(thr, linestyle="--", linewidth=2, color="black", label=f"thr={thr:.3g}")
    
    # Calculate means
    mean_h = np.mean(scores_h) if len(scores_h) > 0 else np.nan
    mean_p = np.mean(scores_p) if len(scores_p) > 0 else np.nan
    
    # Add dotted lines for means
    if not np.isnan(mean_h):
        ax.axvline(mean_h, linestyle=":", linewidth=2, color="blue", label=f"healthy mean={mean_h:.3g}")
    if not np.isnan(mean_p):
        ax.axvline(mean_p, linestyle=":", linewidth=2, color="red", label=f"pathological mean={mean_p:.3g}")
    
    ax.set_xlabel("anomaly score")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_weights(w: np.ndarray, out_png: str, title: str):
    fig = plt.figure(figsize=(7, 3))
    ax = fig.add_subplot(111)
    ax.plot(w)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.set_xlabel("time (resampled)")
    ax.set_ylabel("weight (0..1)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# --------------------------
# Training / evaluation (weighted anomaly score)
# --------------------------
@dataclass
class MovementResult:
    movement: int
    threshold: float
    scores_df: pd.DataFrame
    train_curve: Dict[str, List[float]]


def weighted_score(et: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    et: (B,T-1) per-timestep error
    w:  (B,T)   weights 0..1 (we use t=1..)
    returns: (B,) weighted mean error
    """
    w2 = w[:, 1:]  # (B,T-1)
    num = (et * w2).sum(dim=1)
    den = torch.clamp(w2.sum(dim=1), min=1e-6)
    return num / den


def train_one_movement(
    movement: int,
    samples: List[SampleProcessed],
    scaler: RobustScaler,
    out_dir: str,
    args,
    device: torch.device
) -> MovementResult:
    mv_dir = ensure_dir(os.path.join(out_dir, f"M{movement}"))
    ensure_dir(os.path.join(mv_dir, "figs"))

    healthy = [s for s in samples if s.condition == "healthy"]
    patho = [s for s in samples if s.condition == "pathological"]

    healthy_ids = [s.participant_id for s in healthy]
    tr_ids, va_ids, te_ids = subject_split(healthy_ids, seed=args.seed + 1000 * movement, ratios=args.split)

    tr = [s for s in healthy if s.participant_id in tr_ids]
    va = [s for s in healthy if s.participant_id in va_ids]
    te_h = [s for s in healthy if s.participant_id in te_ids]
    te_p = patho

    def norm_copy(ss: List[SampleProcessed]) -> List[SampleProcessed]:
        out = []
        for s in ss:
            Xn = scaler.transform(s.X, s.M)
            out.append(SampleProcessed(
                participant_id=s.participant_id,
                condition=s.condition,
                movement=s.movement,
                filepath=s.filepath,
                X=Xn,
                M=s.M.copy(),
                W=s.W.copy(),
                neutral_frame_idx=s.neutral_frame_idx,
                completeness_first10pct=s.completeness_first10pct,
            ))
        return out

    trN, vaN, te_hN, te_pN = map(norm_copy, [tr, va, te_h, te_p])

    # quick sanity figs
    for tag, group in [("train", trN), ("val", vaN), ("test_healthy", te_hN), ("test_patho", te_pN)]:
        if len(group) == 0:
            continue
        s0 = group[0]
        plot_missingness_heatmap(s0.M, os.path.join(mv_dir, "figs", f"{tag}_missingness.png"),
                                 title=f"M{movement} {tag} missingness (T={args.T_out}, D={s0.X.shape[1]})")
        plot_feature_heatmap(s0.X, os.path.join(mv_dir, "figs", f"{tag}_features.png"),
                             title=f"M{movement} {tag} ROI edge features (normalized)")
        plot_weights(s0.W, os.path.join(mv_dir, "figs", f"{tag}_weights.png"),
                     title=f"M{movement} {tag} weights (motion energy)")

    D = trN[0].X.shape[1] if len(trN) else (vaN[0].X.shape[1] if len(vaN) else None)
    if D is None:
        raise RuntimeError(f"No data for movement {movement}")

    model = MaskedTCNForecaster(D=D, hidden=args.hidden, layers=args.tcn_layers, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    tr_loader = DataLoader(SequenceDataset(trN), batch_size=args.batch, shuffle=True, num_workers=0)
    va_loader = DataLoader(SequenceDataset(vaN), batch_size=args.batch, shuffle=False, num_workers=0)

    train_losses, val_losses = [], []
    best_val = float("inf")
    best_path = os.path.join(mv_dir, "best_model.pt")

    for epoch in range(args.epochs):
        model.train()
        tl = 0.0
        n = 0
        for batch in tr_loader:
            X = batch["X"].to(device)
            M = batch["M"].to(device)
            yhat = model(X, M)
            loss = masked_huber_loss(yhat, X, M, delta=args.huber_delta)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            tl += loss.item()
            n += 1
        tl = tl / max(n, 1)
        train_losses.append(tl)

        model.eval()
        vl = 0.0
        n = 0
        with torch.no_grad():
            for batch in va_loader:
                X = batch["X"].to(device)
                M = batch["M"].to(device)
                yhat = model(X, M)
                loss = masked_huber_loss(yhat, X, M, delta=args.huber_delta)
                vl += loss.item()
                n += 1
        vl = vl / max(n, 1)
        val_losses.append(vl)

        if vl < best_val:
            best_val = vl
            torch.save({"model": model.state_dict(), "D": D}, best_path)

        if args.verbose:
            print(f"[M{movement}] epoch {epoch+1:03d}/{args.epochs} train={tl:.4g} val={vl:.4g}")

    plot_loss_curves(train_losses, val_losses, os.path.join(mv_dir, "loss_curves.png"),
                     title=f"M{movement} training")

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    def score_group(group: List[SampleProcessed], tag: str) -> pd.DataFrame:
        loader = DataLoader(SequenceDataset(group), batch_size=args.batch, shuffle=False, num_workers=0)
        rows = []
        with torch.no_grad():
            for batch in loader:
                X = batch["X"].to(device)
                M = batch["M"].to(device)
                W = batch["W"].to(device)
                yhat = model(X, M)
                et = masked_mae_per_timestep(yhat, X, M)  # (B,T-1)

                s_mean = weighted_score(et, W).detach().cpu().numpy()
                s_peak = et.max(dim=1).values.detach().cpu().numpy()

                for i in range(X.shape[0]):
                    rows.append({
                        "movement": movement,
                        "set": tag,
                        "participant_id": batch["participant_id"][i],
                        "condition": batch["condition"][i],
                        "filepath": batch["filepath"][i],
                        "score_weighted_mean": float(s_mean[i]),
                        "score_peak": float(s_peak[i]),
                        "w_sum": float(W[i].sum().detach().cpu().item()),
                    })
        return pd.DataFrame(rows)

    df_tr = score_group(trN, "train_healthy")
    df_va = score_group(vaN, "val_healthy")
    df_th = score_group(te_hN, "test_healthy")
    df_tp = score_group(te_pN, "test_pathological")

    thr = float(np.percentile(df_va["score_weighted_mean"].values, args.thr_percentile)) if len(df_va) else float("inf")

    scores_df = pd.concat([df_tr, df_va, df_th, df_tp], axis=0, ignore_index=True)
    scores_df.to_csv(os.path.join(mv_dir, "scores.csv"), index=False)

    scores_h = scores_df[scores_df["condition"] == "healthy"]["score_weighted_mean"].values
    scores_p = scores_df[scores_df["condition"] == "pathological"]["score_weighted_mean"].values
    plot_score_hist(scores_h, scores_p, thr, os.path.join(mv_dir, "score_hist.png"),
                    title=f"M{movement} weighted scores")

    # --- New code for test-only histogram ---
    # Filter for test sets only
    df_test_only = scores_df[scores_df["set"].isin(["test_healthy", "test_pathological"])]

    # Separate scores by condition for the test set
    scores_h_test = df_test_only[df_test_only["condition"] == "healthy"]["score_weighted_mean"].values
    scores_p_test = df_test_only[df_test_only["condition"] == "pathological"]["score_weighted_mean"].values

    # Generate the test-only plot
    plot_score_hist(
        scores_h_test, 
        scores_p_test, 
        thr, 
        os.path.join(mv_dir, "score_hist_test.png"),
        title=f"M{movement} weighted scores (Test Only)"
    )
    # ----------------------------------------

    scores_df["pred_anom"] = (scores_df["score_weighted_mean"] > thr).astype(int)
    summ = scores_df.groupby(["set", "condition"])["pred_anom"].agg(["count", "mean"]).reset_index()
    summ.rename(columns={"mean": "anom_rate_at_thr"}, inplace=True)
    summ.to_csv(os.path.join(mv_dir, "summary_at_threshold.csv"), index=False)

    with open(os.path.join(mv_dir, "run_config.json"), "w") as f:
        json.dump({
            "movement": movement,
            "threshold": thr,
            "best_val_loss": best_val,
            "split": args.split,
            "roi_edges": int(args.roi_edge_counts.get(movement, -1)),
            "model": {"hidden": args.hidden, "layers": args.tcn_layers, "dropout": args.dropout},
            "preproc": {
                "T_out": args.T_out, "max_gap": args.max_gap, "edge_feature": args.edge_feature,
                "win_thr_frac": args.win_thr_frac, "win_min_len": args.win_min_len, "win_pad": args.win_pad
            },
        }, f, indent=2)

    return MovementResult(
        movement=movement,
        threshold=thr,
        scores_df=scores_df,
        train_curve={"train": train_losses, "val": val_losses},
    )


# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--faces", required=True, help="faces.npy indexing 105 facial markers (0..104)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--root_override", default=None)

    ap.add_argument("--skiprows", type=int, default=5)
    ap.add_argument("--usecols_start", type=int, default=2)
    ap.add_argument("--usecols_count", type=int, default=324)

    # ROI selection
    ap.add_argument("--K_markers", type=int, default=25)
    ap.add_argument("--rest_first_valid_frames", type=int, default=10)
    ap.add_argument("--marker_peak_q", type=float, default=95.0)

    # ROI edges
    ap.add_argument("--min_roi_edges", type=int, default=40)
    ap.add_argument("--expand_hops", type=int, default=1)

    # Active window
    ap.add_argument("--win_thr_frac", type=float, default=0.20)
    ap.add_argument("--win_min_len", type=int, default=20)
    ap.add_argument("--win_pad", type=int, default=3)

    # Resample + edge feature
    ap.add_argument("--T_out", type=int, default=50)
    ap.add_argument("--max_gap", type=int, default=3)
    ap.add_argument("--edge_feature", choices=["abs", "rel"], default="rel")

    # Training
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--tcn_layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--huber_delta", type=float, default=1.0)

    ap.add_argument("--thr_percentile", type=float, default=95.0)
    ap.add_argument("--split", type=float, nargs=3, default=[0.7, 0.15, 0.15])

    # Filtering
    ap.add_argument("--only_single_movement", type=int, default=1)
    ap.add_argument("--valid_only", type=int, default=1)

    # Cache
    ap.add_argument("--use_cache", action="store_true")
    ap.add_argument("--cache_dir", default=None)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    seed_all(args.seed)

    out_dir = ensure_dir(args.out_dir)
    run_dir = ensure_dir(os.path.join(out_dir, f"run_{now_str()}"))
    audit_dir = ensure_dir(os.path.join(run_dir, "00_dataset_audit"))
    sanity_dir = ensure_dir(os.path.join(run_dir, "01_preproc_sanity"))
    roi_dir = ensure_dir(os.path.join(run_dir, "02_roi_zscore_selection"))
    roi_edges_dir = ensure_dir(os.path.join(run_dir, "03_roi_edges"))
    train_root = ensure_dir(os.path.join(run_dir, "04_oneclass_forecasting"))

    cache_dir = args.cache_dir or os.path.join(run_dir, "cache")
    ensure_dir(cache_dir)

    # Load metadata robustly
    meta = pd.read_csv(args.metadata)
    meta["condition"] = meta["condition"].astype(str).str.lower().str.strip()
    for col in ["single_movement", "facial_movement", "valid_for_processing"]:
        meta[col] = pd.to_numeric(meta[col], errors="coerce")
    meta = meta.dropna(subset=["single_movement", "facial_movement", "valid_for_processing"])
    meta["single_movement"] = meta["single_movement"].astype(int)
    meta["facial_movement"] = meta["facial_movement"].astype(int)
    meta["valid_for_processing"] = meta["valid_for_processing"].astype(int)

    df = meta.copy()
    if args.valid_only == 1:
        df = df[df["valid_for_processing"] == 1]
    if args.only_single_movement == 1:
        df = df[df["single_movement"] == 1]
    df = df[df["facial_movement"].between(1, 5)]
    df = df[df["condition"].isin(["healthy", "pathological"])]

    # Audit
    audit_and_plot_dataset(meta=df, out_dir=audit_dir)

    # Load faces -> edges (global)
    faces = np.load(args.faces)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces.npy must be (F,3). Got {faces.shape}")
    faces = faces.astype(np.int32)
    if faces.min() < 0 or faces.max() >= 105:
        raise ValueError("faces.npy indices must be within 0..104 (105 facial markers).")

    edges_all = faces_to_undirected_edges(faces)
    adj = build_edge_adjacency(edges_all, n_verts=105)

    with open(os.path.join(run_dir, "topology_info.json"), "w") as f:
        json.dump({"faces": int(faces.shape[0]), "edges_all": int(edges_all.shape[0])}, f, indent=2)

    # Stage A: ROI markers via z-score (healthy only)
    roi_markers = compute_roi_markers_zscore(df=df, args=args, out_dir=roi_dir)

    # Stage B: ROI edges per movement
    roi_edges = {}
    roi_marker_final = {}
    roi_edge_counts = {}
    roi_edge_files = {}

    for mv in range(1, 6):
        active = roi_markers.get(mv, [])
        re, mlist = roi_edges_from_markers(
            edges_all=edges_all,
            active_markers=active,
            n_verts=105,
            min_edges=args.min_roi_edges,
            adj=adj,
            expand_hops=args.expand_hops
        )
        roi_edges[mv] = re
        roi_marker_final[mv] = mlist
        roi_edge_counts[mv] = int(re.shape[0])

        npy_path = os.path.join(roi_edges_dir, f"roi_edges_M{mv}.npy")
        np.save(npy_path, re.astype(np.int32))
        roi_edge_files[mv] = npy_path

        pd.DataFrame({"marker": mlist}).to_csv(os.path.join(roi_edges_dir, f"roi_markers_final_M{mv}.csv"), index=False)

    with open(os.path.join(roi_edges_dir, "roi_edge_stats.json"), "w") as f:
        json.dump({
            "K_markers": args.K_markers,
            "min_roi_edges": args.min_roi_edges,
            "expand_hops": args.expand_hops,
            "roi_edges_count": roi_edge_counts,
        }, f, indent=2)

    # stash counts on args for logging in run_config.json later
    args.roi_edge_counts = roi_edge_counts

    # Device
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # Stage C/D: process samples per movement using ROI edges + active window
    processed_by_mv: Dict[int, List[SampleProcessed]] = {m: [] for m in range(1, 6)}
    dropped = []
    neutral_stats_rows = []

    # cache tag must include ROI definition to avoid mixing with v1
    roi_hash_tag = f"ROIzk{args.K_markers}_minE{args.min_roi_edges}_hop{args.expand_hops}_faces{faces.shape[0]}_edges{edges_all.shape[0]}"

    for _, row in df.iterrows():
        mv = int(row["facial_movement"])
        re = roi_edges[mv]
        if re.shape[0] == 0:
            dropped.append({"complete_filepath": row["complete_filepath"], "reason": f"no_roi_edges_M{mv}"})
            continue

        sp = process_one_sample_v2(
            row=row,
            roi_edges=re,
            args=args,
            cache_dir=cache_dir,
            cache_tag=roi_hash_tag
        )
        if sp is None:
            dropped.append({"complete_filepath": row["complete_filepath"], "reason": "load_or_preproc_failed"})
            continue

        valid_ratio = float(sp.M.mean())
        if valid_ratio < 0.02:
            dropped.append({"complete_filepath": sp.filepath, "reason": f"too_sparse_mask_ratio={valid_ratio:.4f}"})
            continue

        # if weights sum ~0, active window likely failed / no motion
        if float(sp.W.sum()) < 1e-4:
            dropped.append({"complete_filepath": sp.filepath, "reason": "weights_all_zero"})
            continue

        processed_by_mv[mv].append(sp)

        neutral_stats_rows.append({
            "movement": mv,
            "participant_id": sp.participant_id,
            "condition": sp.condition,
            "filepath": sp.filepath,
            "neutral_frame_idx": sp.neutral_frame_idx,
            "mask_ratio_after_resample": float(sp.M.mean()),
            "w_sum": float(sp.W.sum()),
            "roi_edges": int(re.shape[0]),
            **{f"neutral_{k}": v for k, v in sp.completeness_first10pct.items()},
        })

    pd.DataFrame(dropped).to_csv(os.path.join(run_dir, "dropped_samples.csv"), index=False)
    pd.DataFrame(neutral_stats_rows).to_csv(os.path.join(sanity_dir, "neutral_frame_mask_weight_stats.csv"), index=False)

    # Stage E: train per movement
    all_results = []
    for mv in range(1, 6):
        samples = processed_by_mv[mv]
        if len(samples) == 0:
            print(f"[M{mv}] no processed samples. Skipping.")
            continue

        mv_train_dir = ensure_dir(os.path.join(train_root, f"M{mv}"))

        healthy = [s for s in samples if s.condition == "healthy"]
        if len(healthy) < 10:
            print(f"[M{mv}] too few healthy samples ({len(healthy)}). Expect unstable results.")

        ids = [s.participant_id for s in healthy]
        tr_ids, va_ids, te_ids = subject_split(ids, seed=args.seed + 1000 * mv, ratios=args.split)
        healthy_tr = [s for s in healthy if s.participant_id in tr_ids]

        scaler = fit_robust_scaler([s.X for s in healthy_tr], [s.M for s in healthy_tr])
        np.savez_compressed(os.path.join(mv_train_dir, "scaler_stats.npz"), median=scaler.median, iqr=scaler.iqr)

        res = train_one_movement(
            movement=mv,
            samples=samples,
            scaler=scaler,
            out_dir=train_root,
            args=args,
            device=device,
        )
        all_results.append(res)

    # Global summary on test sets
    rows = []
    for r in all_results:
        mv = r.movement
        sdf = r.scores_df.copy()
        sdf["gt_patho"] = (sdf["condition"] == "pathological").astype(int)
        sdf["pred"] = (sdf["score_weighted_mean"] > r.threshold).astype(int)
        test = sdf[sdf["set"].isin(["test_healthy", "test_pathological"])]

        tp = int(((test["gt_patho"] == 1) & (test["pred"] == 1)).sum())
        tn = int(((test["gt_patho"] == 0) & (test["pred"] == 0)).sum())
        fp = int(((test["gt_patho"] == 0) & (test["pred"] == 1)).sum())
        fn = int(((test["gt_patho"] == 1) & (test["pred"] == 0)).sum())

        rows.append({
            "movement": mv,
            "threshold": r.threshold,
            "roi_edges": int(args.roi_edge_counts.get(mv, -1)),
            "test_tp": tp, "test_tn": tn, "test_fp": fp, "test_fn": fn,
            "test_acc": (tp + tn) / max(tp + tn + fp + fn, 1),
            "test_tpr": tp / max(tp + fn, 1),
            "test_fpr": fp / max(fp + tn, 1),
        })

    pd.DataFrame(rows).to_csv(os.path.join(run_dir, "global_summary.csv"), index=False)

    with open(os.path.join(run_dir, "run_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nDONE. Outputs in: {run_dir}")


if __name__ == "__main__":
    main()

