import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.io import loadmat
from scipy.spatial import cKDTree
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Constants
"""
CUDA_VISIBLE_DEVICES=2 python train/train_lcz_graph_encoder_v5.py \
  --base_dir /mnt/disk1/workspace_jym/LCZ/data \
  --work_dir /mnt/disk1/workspace_jym/LCZ/work_dirs \
  --rs_norm_dir /mnt/disk1/workspace_jym/LCZ/data/Satellite/processed/norm \
  --building_shp /mnt/disk1/workspace_jym/LCZ/data/Building/AL_11_D010_20200502/AL_11_D010_20200502.shp \
  --story_col A10 \
  --split_dir /mnt/disk1/workspace_jym/LCZ/work_dirs/cnn_rs_baseline/splits \
  --exp_name graph_v5_lightweight_hyperedge_500m \
  --patch_size 33 \
  --graph_context_m 500 \
  --num_rings 3 \
  --num_sectors 8 \
  --graph_hidden 64 \
  --graph_out 128 \
  --graph_layers 2 \
  --graph_heads 2 \
  --batch_size 256 \
  --epochs 150 \
  --patience 25 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --dropout 0.5 \
  --class_weight \
  --rebuild_graph_cache
"""
# ============================================================

DEFAULT_BASE_DIR = "/mnt/disk1/workspace_jym/LCZ/data"
DEFAULT_WORK_DIR = "/mnt/disk1/workspace_jym/LCZ/work_dirs"

LCZ_CLASSES = [1, 2, 3, 4, 5, 6, 8, 101, 102, 104, 107]
LCZ_CLASS_NAMES = {
    1: "LCZ1", 2: "LCZ2", 3: "LCZ3", 4: "LCZ4", 5: "LCZ5", 6: "LCZ6", 8: "LCZ8",
    101: "LCZA", 102: "LCZB", 104: "LCZD", 107: "LCZG",
}


# ============================================================
# Reproducibility / IO
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_feature_stack(norm_dir: Path):
    mat_paths = sorted(norm_dir.glob("b*_norm.mat"))
    if len(mat_paths) == 0:
        raise FileNotFoundError(f"No b*_norm.mat files found in {norm_dir}")

    bands = []
    print("========== Load normalized RS feature bands ==========")
    for p in mat_paths:
        data = loadmat(p)
        if "norm" not in data:
            raise KeyError(f"'norm' variable not found in {p}")
        arr = data["norm"].astype(np.float32)
        arr[~np.isfinite(arr)] = 0.0
        bands.append(arr)
        print(f"{p.name}: shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")

    stack = np.stack(bands, axis=0).astype(np.float32)
    print("RS feature stack:", stack.shape)
    return stack, mat_paths


def count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ============================================================
# Lightweight HyperEdge Graph Provider
# ============================================================

class LightweightHyperGraphProvider:
    """
    V5 graph representation.

    Instead of using hundreds of building nodes, this provider summarizes surrounding buildings
    into a small fixed-size center-aware hyperedge graph:

      - 1 center node
      - ring-sector hyperedge nodes, e.g. 3 rings x 8 sectors = 24 nodes
      - story-bin hyperedge nodes, e.g. unknown / low / mid / high = 4 nodes
      - 1 global context node

    This keeps graph attention cheap while preserving LCZ-relevant information:
    density, total built area, height/story distribution, direction/ring distribution.
    """

    def __init__(
        self,
        building_shp: Path,
        story_col: str,
        gt_path: Path,
        graph_context_m: float = 500.0,
        num_rings: int = 3,
        num_sectors: int = 8,
        target_crs: str = "EPSG:32652",
    ):
        self.building_shp = Path(building_shp)
        self.story_col = story_col
        self.gt_path = Path(gt_path)
        self.graph_context_m = float(graph_context_m)
        self.half = self.graph_context_m / 2.0
        self.num_rings = int(num_rings)
        self.num_sectors = int(num_sectors)
        self.target_crs = target_crs

        self.num_center = 1
        self.num_sector_nodes = self.num_rings * self.num_sectors
        self.num_story_nodes = 4
        self.num_global = 1
        self.num_nodes = self.num_center + self.num_sector_nodes + self.num_story_nodes + self.num_global

        self.center_idx = 0
        self.sector_start = 1
        self.story_start = self.sector_start + self.num_sector_nodes
        self.global_idx = self.story_start + self.num_story_nodes

        self.node_dim = 24

        self._load_gt_grid()
        self._load_buildings()
        self.base_adj = self._make_base_adjacency()

    def _load_gt_grid(self):
        with rasterio.open(self.gt_path) as src:
            self.gt_transform = src.transform
            self.gt_crs = src.crs
            self.gt_bounds = src.bounds
            self.gt_width = src.width
            self.gt_height = src.height

    def _load_buildings(self):
        print("========== Load building data for V5 lightweight hypergraph ==========")
        print("SHP:", self.building_shp)
        if not self.building_shp.exists():
            raise FileNotFoundError(f"Building SHP not found: {self.building_shp}")

        try:
            gdf = gpd.read_file(self.building_shp)
        except UnicodeDecodeError:
            gdf = gpd.read_file(self.building_shp, encoding="cp949")

        print("Original CRS:", gdf.crs)
        print("Original buildings:", len(gdf))

        if self.story_col not in gdf.columns:
            raise KeyError(f"story_col={self.story_col} not found. Available: {list(gdf.columns)}")

        gdf = gdf.to_crs(self.target_crs)

        # Expand clipping by context margin to avoid losing buildings near GT boundary.
        margin = self.half + 50.0
        xmin = self.gt_bounds.left - margin
        xmax = self.gt_bounds.right + margin
        ymin = self.gt_bounds.bottom - margin
        ymax = self.gt_bounds.top + margin
        gdf = gdf.cx[xmin:xmax, ymin:ymax].copy()

        gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()
        gdf["geometry"] = gdf.geometry.buffer(0)
        gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()

        gdf["area_m2"] = gdf.geometry.area.astype(np.float32)
        gdf["perimeter_m"] = gdf.geometry.length.astype(np.float32)
        bounds = gdf.geometry.bounds
        gdf["bbox_w"] = (bounds["maxx"] - bounds["minx"]).astype(np.float32)
        gdf["bbox_h"] = (bounds["maxy"] - bounds["miny"]).astype(np.float32)
        gdf["story"] = pd.to_numeric(gdf[self.story_col], errors="coerce").fillna(0).astype(np.float32)
        gdf.loc[gdf["story"] < 0, "story"] = 0

        gdf = gdf[gdf["area_m2"] > 0].copy()

        cent = gdf.geometry.centroid
        self.xy = np.vstack([cent.x.values, cent.y.values]).T.astype(np.float32)
        self.area = gdf["area_m2"].values.astype(np.float32)
        self.story = gdf["story"].values.astype(np.float32)
        self.perimeter = gdf["perimeter_m"].values.astype(np.float32)
        self.bbox_w = gdf["bbox_w"].values.astype(np.float32)
        self.bbox_h = gdf["bbox_h"].values.astype(np.float32)

        self.tree = cKDTree(self.xy)

        # Normalization scales from building inventory.
        log_area = np.log1p(self.area)
        positive_story = self.story[self.story > 0]
        self.area_scale = float(np.percentile(log_area, 99)) if len(log_area) else 1.0
        self.story_scale = float(np.percentile(positive_story, 99)) if len(positive_story) else 1.0
        self.area_scale = max(self.area_scale, 1.0)
        self.story_scale = max(self.story_scale, 1.0)

        print("Clipped valid buildings:", len(self.area))
        print("Area scale log1p p99:", self.area_scale)
        print("Story scale p99:", self.story_scale)
        print("Story positive ratio:", float((self.story > 0).mean()))
        print("Fixed graph nodes:", self.num_nodes, "node_dim:", self.node_dim)

    def center_xy_from_rowcol(self, row50: int, col50: int) -> Tuple[float, float]:
        # 50m pixel center from raster transform.
        x = self.gt_transform.c + (col50 + 0.5) * self.gt_transform.a
        y = self.gt_transform.f + (row50 + 0.5) * self.gt_transform.e
        return float(x), float(y)

    def _query_buildings(self, cx: float, cy: float) -> np.ndarray:
        # Query with circle covering the square, then exact square filter.
        radius = self.half * math.sqrt(2.0)
        cand = self.tree.query_ball_point([cx, cy], r=radius)
        if len(cand) == 0:
            return np.empty((0,), dtype=np.int64)
        cand = np.asarray(cand, dtype=np.int64)
        dx = self.xy[cand, 0] - cx
        dy = self.xy[cand, 1] - cy
        keep = (np.abs(dx) <= self.half) & (np.abs(dy) <= self.half)
        return cand[keep]

    def _summary_feature(
        self,
        idx: np.ndarray,
        dx_all: np.ndarray,
        dy_all: np.ndarray,
        node_pos_x: float,
        node_pos_y: float,
        ring_norm: float,
        sector_cos: float,
        sector_sin: float,
        type_flags: Tuple[float, float, float, float],
    ) -> np.ndarray:
        # Feature layout, dim=24:
        # 0 count_norm, 1 total_area_ratio, 2 built_ratio,
        # 3 mean_log_area_norm, 4 max_log_area_norm, 5 std_log_area_norm,
        # 6 mean_story_norm, 7 max_story_norm, 8 std_story_norm, 9 high_story_ratio,
        # 10 mean_dist_norm, 11 std_dist_norm, 12 mean_abs_dx_norm, 13 mean_abs_dy_norm,
        # 14 node_pos_x, 15 node_pos_y, 16 ring_norm, 17 sector_cos, 18 sector_sin,
        # 19 is_center, 20 is_sector, 21 is_storybin, 22 is_global, 23 nonempty.
        feat = np.zeros((self.node_dim,), dtype=np.float32)
        feat[14] = float(node_pos_x)
        feat[15] = float(node_pos_y)
        feat[16] = float(ring_norm)
        feat[17] = float(sector_cos)
        feat[18] = float(sector_sin)
        feat[19:23] = np.asarray(type_flags, dtype=np.float32)

        n = len(idx)
        if n == 0:
            feat[23] = 0.0
            return feat

        area = self.area[idx]
        story = self.story[idx]
        log_area = np.log1p(area)
        dist = np.sqrt(dx_all * dx_all + dy_all * dy_all)
        context_area = self.graph_context_m * self.graph_context_m

        story_pos = story[story > 0]
        if len(story_pos) == 0:
            story_pos = np.array([0.0], dtype=np.float32)

        feat[0] = min(n / 200.0, 1.0)
        feat[1] = min(float(area.sum()) / max(context_area, 1.0), 1.5)
        feat[2] = min(float(area.sum()) / max(context_area, 1.0), 1.0)
        feat[3] = min(float(log_area.mean()) / self.area_scale, 1.5)
        feat[4] = min(float(log_area.max()) / self.area_scale, 1.5)
        feat[5] = min(float(log_area.std()) / self.area_scale, 1.5)
        feat[6] = min(float(story_pos.mean()) / self.story_scale, 1.5)
        feat[7] = min(float(story_pos.max()) / self.story_scale, 1.5)
        feat[8] = min(float(story_pos.std()) / self.story_scale, 1.5)
        feat[9] = float((story >= 10).sum()) / max(n, 1)
        feat[10] = min(float(dist.mean()) / max(self.half, 1.0), 2.0)
        feat[11] = min(float(dist.std()) / max(self.half, 1.0), 2.0)
        feat[12] = min(float(np.abs(dx_all).mean()) / max(self.half, 1.0), 1.0)
        feat[13] = min(float(np.abs(dy_all).mean()) / max(self.half, 1.0), 1.0)
        feat[23] = 1.0
        return feat.astype(np.float32)

    def _story_bin_indices(self, idx: np.ndarray) -> List[np.ndarray]:
        story = self.story[idx]
        bins = []
        bins.append(idx[story <= 0])
        bins.append(idx[(story > 0) & (story <= 3)])
        bins.append(idx[(story >= 4) & (story <= 9)])
        bins.append(idx[story >= 10])
        return bins

    def _make_base_adjacency(self) -> np.ndarray:
        N = self.num_nodes
        A = np.zeros((N, N), dtype=np.float32)

        # self-loop
        np.fill_diagonal(A, 1.0)

        # center <-> all summary nodes
        for j in range(1, N):
            A[self.center_idx, j] = 1.0
            A[j, self.center_idx] = 1.0

        # global <-> all sector and story nodes
        for j in range(1, N - 1):
            A[self.global_idx, j] = 1.0
            A[j, self.global_idx] = 1.0

        # ring-sector topology: angular neighbors and radial neighbors
        for r in range(self.num_rings):
            for s in range(self.num_sectors):
                idx = self.sector_start + r * self.num_sectors + s
                # angular neighbors
                for ss in [(s - 1) % self.num_sectors, (s + 1) % self.num_sectors]:
                    j = self.sector_start + r * self.num_sectors + ss
                    A[idx, j] = 1.0
                    A[j, idx] = 1.0
                # radial neighbors
                for rr in [r - 1, r + 1]:
                    if 0 <= rr < self.num_rings:
                        j = self.sector_start + rr * self.num_sectors + s
                        A[idx, j] = 1.0
                        A[j, idx] = 1.0

        # story bins fully connected to each other
        for i in range(self.num_story_nodes):
            for j in range(self.num_story_nodes):
                a = self.story_start + i
                b = self.story_start + j
                A[a, b] = 1.0

        return A

    def build_graph(self, row50: int, col50: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cx, cy = self.center_xy_from_rowcol(row50, col50)
        idx = self._query_buildings(cx, cy)

        x = np.zeros((self.num_nodes, self.node_dim), dtype=np.float32)
        A = self.base_adj.copy()
        mask = np.ones((self.num_nodes,), dtype=np.float32)

        # Center node.
        x[self.center_idx] = self._summary_feature(
            idx=np.empty((0,), dtype=np.int64),
            dx_all=np.empty((0,), dtype=np.float32),
            dy_all=np.empty((0,), dtype=np.float32),
            node_pos_x=0.0,
            node_pos_y=0.0,
            ring_norm=0.0,
            sector_cos=1.0,
            sector_sin=0.0,
            type_flags=(1.0, 0.0, 0.0, 0.0),
        )
        x[self.center_idx, 23] = 1.0

        if len(idx) > 0:
            dx = self.xy[idx, 0] - cx
            dy = self.xy[idx, 1] - cy
            dist = np.sqrt(dx * dx + dy * dy)
            angle = np.arctan2(dy, dx)
            angle_pos = (angle + 2 * np.pi) % (2 * np.pi)

            # Global node.
            x[self.global_idx] = self._summary_feature(
                idx=idx,
                dx_all=dx,
                dy_all=dy,
                node_pos_x=0.0,
                node_pos_y=0.0,
                ring_norm=1.0,
                sector_cos=1.0,
                sector_sin=0.0,
                type_flags=(0.0, 0.0, 0.0, 1.0),
            )

            # Ring-sector hyperedge nodes.
            ring_edges = np.linspace(0.0, self.half, self.num_rings + 1)
            # Include boundary points in last bin.
            ring_id = np.searchsorted(ring_edges[1:], dist, side="right")
            ring_id = np.clip(ring_id, 0, self.num_rings - 1)
            sector_id = np.floor(angle_pos / (2 * np.pi / self.num_sectors)).astype(np.int64)
            sector_id = np.clip(sector_id, 0, self.num_sectors - 1)

            for r in range(self.num_rings):
                r_mid = (ring_edges[r] + ring_edges[r + 1]) / 2.0
                r_norm = r_mid / max(self.half, 1.0)
                for s in range(self.num_sectors):
                    node = self.sector_start + r * self.num_sectors + s
                    sel = (ring_id == r) & (sector_id == s)
                    sub_idx = idx[sel]
                    sub_dx = dx[sel]
                    sub_dy = dy[sel]
                    theta = (s + 0.5) * (2 * np.pi / self.num_sectors)
                    node_pos_x = r_norm * math.cos(theta)
                    node_pos_y = r_norm * math.sin(theta)
                    x[node] = self._summary_feature(
                        idx=sub_idx,
                        dx_all=sub_dx,
                        dy_all=sub_dy,
                        node_pos_x=node_pos_x,
                        node_pos_y=node_pos_y,
                        ring_norm=r_norm,
                        sector_cos=math.cos(theta),
                        sector_sin=math.sin(theta),
                        type_flags=(0.0, 1.0, 0.0, 0.0),
                    )

            # Story-bin hyperedge nodes.
            story_bins = self._story_bin_indices(idx)
            for b, sub_idx in enumerate(story_bins):
                node = self.story_start + b
                # Need dx/dy for selected global idx.
                if len(sub_idx) > 0:
                    sub_xy = self.xy[sub_idx]
                    sub_dx = sub_xy[:, 0] - cx
                    sub_dy = sub_xy[:, 1] - cy
                else:
                    sub_dx = np.empty((0,), dtype=np.float32)
                    sub_dy = np.empty((0,), dtype=np.float32)
                # story bin node positions are abstract; spread them on a line.
                pos_x = -1.0 + 2.0 * b / max(self.num_story_nodes - 1, 1)
                x[node] = self._summary_feature(
                    idx=sub_idx,
                    dx_all=sub_dx,
                    dy_all=sub_dy,
                    node_pos_x=pos_x,
                    node_pos_y=1.15,
                    ring_norm=1.0,
                    sector_cos=0.0,
                    sector_sin=1.0,
                    type_flags=(0.0, 0.0, 1.0, 0.0),
                )
        else:
            # Empty global / summary nodes still keep their type/position encodings.
            x[self.global_idx] = self._summary_feature(
                idx=np.empty((0,), dtype=np.int64),
                dx_all=np.empty((0,), dtype=np.float32),
                dy_all=np.empty((0,), dtype=np.float32),
                node_pos_x=0.0,
                node_pos_y=0.0,
                ring_norm=1.0,
                sector_cos=1.0,
                sector_sin=0.0,
                type_flags=(0.0, 0.0, 0.0, 1.0),
            )
            ring_edges = np.linspace(0.0, self.half, self.num_rings + 1)
            for r in range(self.num_rings):
                r_mid = (ring_edges[r] + ring_edges[r + 1]) / 2.0
                r_norm = r_mid / max(self.half, 1.0)
                for s in range(self.num_sectors):
                    node = self.sector_start + r * self.num_sectors + s
                    theta = (s + 0.5) * (2 * np.pi / self.num_sectors)
                    x[node] = self._summary_feature(
                        idx=np.empty((0,), dtype=np.int64),
                        dx_all=np.empty((0,), dtype=np.float32),
                        dy_all=np.empty((0,), dtype=np.float32),
                        node_pos_x=r_norm * math.cos(theta),
                        node_pos_y=r_norm * math.sin(theta),
                        ring_norm=r_norm,
                        sector_cos=math.cos(theta),
                        sector_sin=math.sin(theta),
                        type_flags=(0.0, 1.0, 0.0, 0.0),
                    )
            for b in range(self.num_story_nodes):
                node = self.story_start + b
                pos_x = -1.0 + 2.0 * b / max(self.num_story_nodes - 1, 1)
                x[node] = self._summary_feature(
                    idx=np.empty((0,), dtype=np.int64),
                    dx_all=np.empty((0,), dtype=np.float32),
                    dy_all=np.empty((0,), dtype=np.float32),
                    node_pos_x=pos_x,
                    node_pos_y=1.15,
                    ring_norm=1.0,
                    sector_cos=0.0,
                    sector_sin=1.0,
                    type_flags=(0.0, 0.0, 1.0, 0.0),
                )

        global_feat = x[self.global_idx].copy()
        return x.astype(np.float32), A.astype(np.float32), mask.astype(np.float32), global_feat.astype(np.float32)


# ============================================================
# Graph cache
# ============================================================

def build_graph_cache(provider: LightweightHyperGraphProvider, sample_npz: Path, out_pt: Path):
    data = np.load(sample_npz)
    rows = data["rows"].astype(np.int64)
    cols = data["cols"].astype(np.int64)
    labels = data["labels"].astype(np.int64)

    xs, adjs, masks, globals_ = [], [], [], []
    print(f"Build V5 graph cache: {sample_npz.name}, samples={len(labels)}")
    for i, (r, c) in enumerate(zip(rows, cols)):
        x, adj, mask, gfeat = provider.build_graph(int(r), int(c))
        xs.append(x)
        adjs.append(adj)
        masks.append(mask)
        globals_.append(gfeat)
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(labels)}")

    payload = {
        "node_feats": torch.tensor(np.stack(xs), dtype=torch.float32),
        "adj": torch.tensor(np.stack(adjs), dtype=torch.float32),
        "node_mask": torch.tensor(np.stack(masks), dtype=torch.float32),
        "global_feats": torch.tensor(np.stack(globals_), dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
        "rows": torch.tensor(rows, dtype=torch.long),
        "cols": torch.tensor(cols, dtype=torch.long),
        "meta": {
            "num_nodes": provider.num_nodes,
            "node_dim": provider.node_dim,
            "graph_context_m": provider.graph_context_m,
            "num_rings": provider.num_rings,
            "num_sectors": provider.num_sectors,
        },
    }
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_pt)
    print("Saved graph cache:", out_pt)


# ============================================================
# Dataset
# ============================================================

class LCZHyperGraphDataset(Dataset):
    def __init__(self, feature_stack: np.ndarray, sample_npz: Path, graph_pt: Path, patch_size: int = 33):
        self.feature_stack = feature_stack
        self.patch_size = int(patch_size)
        self.radius = self.patch_size // 2
        self.graph = torch.load(graph_pt, map_location="cpu")

        data = np.load(sample_npz)
        self.rows50 = data["rows"].astype(np.int64)
        self.cols50 = data["cols"].astype(np.int64)
        self.labels = data["labels"].astype(np.int64)

        if len(self.labels) != int(self.graph["labels"].shape[0]):
            raise ValueError("Sample NPZ and graph cache length mismatch")

        self.padded = np.pad(
            self.feature_stack,
            pad_width=((0, 0), (self.radius, self.radius), (self.radius, self.radius)),
            mode="constant",
            constant_values=0.0,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        r50 = int(self.rows50[idx])
        c50 = int(self.cols50[idx])
        r10 = r50 * 5 + 2 + self.radius
        c10 = c50 * 5 + 2 + self.radius
        patch = self.padded[:, r10 - self.radius:r10 + self.radius + 1, c10 - self.radius:c10 + self.radius + 1]

        return {
            "rs": torch.from_numpy(patch.copy()).float(),
            "node_feats": self.graph["node_feats"][idx].float(),
            "adj": self.graph["adj"][idx].float(),
            "node_mask": self.graph["node_mask"][idx].float(),
            "global_feats": self.graph["global_feats"][idx].float(),
            "label": torch.tensor(self.labels[idx]).long(),
        }


# ============================================================
# Model
# ============================================================

class RSCNNEncoder(nn.Module):
    def __init__(self, in_channels: int, patch_size: int = 33, out_dim: int = 256, dropout: float = 0.5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, patch_size, patch_size)
            feat_dim = self.encoder(dummy).view(1, -1).shape[1]
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(feat_dim, out_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))

    def forward(self, x):
        return self.proj(self.encoder(x))


class MaskedGraphAttentionLayer(nn.Module):
    def __init__(self, dim: int, heads: int = 2, dropout: float = 0.5):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj, mask):
        B, N, D = x.shape
        h = self.heads
        q = self.q(self.norm1(x)).view(B, N, h, self.head_dim).transpose(1, 2)
        k = self.k(self.norm1(x)).view(B, N, h, self.head_dim).transpose(1, 2)
        v = self.v(self.norm1(x)).view(B, N, h, self.head_dim).transpose(1, 2)

        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_mask = (adj > 0).unsqueeze(1)
        node_mask = (mask > 0).unsqueeze(1).unsqueeze(2)
        score = score.masked_fill(~attn_mask, -1e4)
        score = score.masked_fill(~node_mask, -1e4)
        attn = torch.softmax(score, dim=-1)
        attn = self.dropout(attn)
        msg = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, D)
        x = x + self.dropout(self.out(msg))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class LightweightHyperGraphEncoder(nn.Module):
    def __init__(self, node_dim: int, global_dim: int, hidden: int = 64, out_dim: int = 128, layers: int = 2, heads: int = 2, dropout: float = 0.5):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(node_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.layers = nn.ModuleList([MaskedGraphAttentionLayer(hidden, heads=heads, dropout=dropout) for _ in range(layers)])
        self.pool_attn = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.global_proj = nn.Sequential(nn.Linear(global_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.out = nn.Sequential(nn.Linear(hidden * 4, out_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))

    def forward(self, node_feats, adj, node_mask, global_feats):
        x = self.input_proj(node_feats)
        for layer in self.layers:
            x = layer(x, adj, node_mask)

        center = x[:, 0, :]
        global_node = x[:, -1, :]

        mask = node_mask > 0
        masked_x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        mean_pool = masked_x.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1).float()

        attn_score = self.pool_attn(x).squeeze(-1).masked_fill(~mask, -1e4)
        attn = torch.softmax(attn_score, dim=-1).unsqueeze(-1)
        attn_pool = (x * attn).sum(dim=1)

        g = self.global_proj(global_feats)
        return self.out(torch.cat([center, global_node, attn_pool, g], dim=1))


class LCZGraphV5Model(nn.Module):
    def __init__(self, rs_channels: int, node_dim: int, global_dim: int, num_classes: int, patch_size: int = 33, graph_hidden: int = 64, graph_out: int = 128, graph_layers: int = 2, graph_heads: int = 2, dropout: float = 0.5):
        super().__init__()
        self.rs_encoder = RSCNNEncoder(rs_channels, patch_size=patch_size, out_dim=256, dropout=dropout)
        self.graph_encoder = LightweightHyperGraphEncoder(node_dim, global_dim, hidden=graph_hidden, out_dim=graph_out, layers=graph_layers, heads=graph_heads, dropout=dropout)

        self.graph_to_rs = nn.Linear(graph_out, 256)
        self.gate = nn.Sequential(nn.Linear(256 + graph_out, 128), nn.ReLU(inplace=True), nn.Linear(128, 256), nn.Sigmoid())
        self.classifier = nn.Sequential(
            nn.Linear(256 + graph_out, 256), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(256, num_classes)
        )

    def forward(self, rs, node_feats, adj, node_mask, global_feats):
        rs_feat = self.rs_encoder(rs)
        graph_feat = self.graph_encoder(node_feats, adj, node_mask, global_feats)
        gate = self.gate(torch.cat([rs_feat, graph_feat], dim=1))
        rs_guided = rs_feat + gate * self.graph_to_rs(graph_feat)
        return self.classifier(torch.cat([rs_guided, graph_feat], dim=1))


# ============================================================
# Train / Eval
# ============================================================

def compute_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader, device, criterion, num_classes):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n_total = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            rs = batch["rs"].to(device, non_blocking=True)
            node_feats = batch["node_feats"].to(device, non_blocking=True)
            adj = batch["adj"].to(device, non_blocking=True)
            node_mask = batch["node_mask"].to(device, non_blocking=True)
            global_feats = batch["global_feats"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            logits = model(rs, node_feats, adj, node_mask, global_feats)
            loss = criterion(logits, y)
            pred = logits.argmax(dim=1)
            total_loss += loss.item() * y.size(0)
            n_total += y.size(0)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return {
        "loss": total_loss / max(n_total, 1),
        "oa": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "cm": confusion_matrix(all_labels, all_preds, labels=list(range(num_classes))),
        "preds": all_preds,
        "labels": all_labels,
    }


def train(args):
    set_seed(args.seed)
    base_dir = Path(args.base_dir)
    work_dir = Path(args.work_dir)
    exp_dir = work_dir / args.exp_name
    split_dir = Path(args.split_dir)
    ckpt_dir = exp_dir / "checkpoints"
    result_dir = exp_dir / "results"
    cache_dir = exp_dir / "graph_cache_v5"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Device:", device)

    gt_path = base_dir / "GT" / "seoul_LCZ.tif"
    rs_features, band_paths = load_feature_stack(Path(args.rs_norm_dir))

    # Build graph cache if needed.
    cache_paths = {s: cache_dir / f"{s}_graphs.pt" for s in ["train", "val", "test"]}
    if args.rebuild_graph_cache or not all(p.exists() for p in cache_paths.values()):
        provider = LightweightHyperGraphProvider(
            building_shp=Path(args.building_shp),
            story_col=args.story_col,
            gt_path=gt_path,
            graph_context_m=args.graph_context_m,
            num_rings=args.num_rings,
            num_sectors=args.num_sectors,
        )
        for split in ["train", "val", "test"]:
            build_graph_cache(provider, split_dir / f"{split}_samples.npz", cache_paths[split])

    meta = torch.load(cache_paths["train"], map_location="cpu")["meta"]
    node_dim = int(meta["node_dim"])
    global_dim = node_dim

    train_ds = LCZHyperGraphDataset(rs_features, split_dir / "train_samples.npz", cache_paths["train"], patch_size=args.patch_size)
    val_ds = LCZHyperGraphDataset(rs_features, split_dir / "val_samples.npz", cache_paths["val"], patch_size=args.patch_size)
    test_ds = LCZHyperGraphDataset(rs_features, split_dir / "test_samples.npz", cache_paths["test"], patch_size=args.patch_size)

    print("========== Dataset ==========")
    print("train:", len(train_ds), "val:", len(val_ds), "test:", len(test_ds))
    print("node_dim:", node_dim, "nodes:", meta["num_nodes"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = LCZGraphV5Model(
        rs_channels=rs_features.shape[0],
        node_dim=node_dim,
        global_dim=global_dim,
        num_classes=len(LCZ_CLASSES),
        patch_size=args.patch_size,
        graph_hidden=args.graph_hidden,
        graph_out=args.graph_out,
        graph_layers=args.graph_layers,
        graph_heads=args.graph_heads,
        dropout=args.dropout,
    ).to(device)

    total_params, trainable_params = count_params(model)
    print("Params total:", total_params, "trainable:", trainable_params)

    train_labels = np.load(split_dir / "train_samples.npz")["labels"]
    if args.class_weight:
        weights = compute_class_weights(train_labels, len(LCZ_CLASSES)).to(device)
        print("Class weights:", weights.detach().cpu().numpy())
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_oa = -1.0
    best_epoch = -1
    patience_count = 0
    log_records = []

    print("========== Training ==========")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_seen = 0.0, 0
        for batch in train_loader:
            rs = batch["rs"].to(device, non_blocking=True)
            node_feats = batch["node_feats"].to(device, non_blocking=True)
            adj = batch["adj"].to(device, non_blocking=True)
            node_mask = batch["node_mask"].to(device, non_blocking=True)
            global_feats = batch["global_feats"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(rs, node_feats, adj, node_mask, global_feats)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            n_seen += y.size(0)

        train_loss = total_loss / max(n_seen, 1)
        val_result = evaluate(model, val_loader, device, criterion, len(LCZ_CLASSES))
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_result['loss']:.4f} | "
            f"val_OA={val_result['oa']:.4f} | val_macroF1={val_result['macro_f1']:.4f}"
        )
        log_records.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_result["loss"],
            "val_oa": val_result["oa"],
            "val_macro_f1": val_result["macro_f1"],
            "val_weighted_f1": val_result["weighted_f1"],
        })

        if val_result["oa"] > best_val_oa:
            best_val_oa = val_result["oa"]
            best_epoch = epoch
            patience_count = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "params_total": total_params,
                "params_trainable": trainable_params,
                "val_oa": best_val_oa,
                "lcz_classes": LCZ_CLASSES,
                "band_files": [str(p) for p in band_paths],
            }, ckpt_dir / "best_model.pt")
            print("  Saved best model")
        else:
            patience_count += 1

        if patience_count >= args.patience:
            print("Early stopping at epoch", epoch)
            break

    pd.DataFrame(log_records).to_csv(result_dir / "train_log.csv", index=False, encoding="utf-8-sig")

    print("========== Test ==========")
    ckpt = torch.load(ckpt_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_result = evaluate(model, test_loader, device, criterion, len(LCZ_CLASSES))

    print("Best epoch:", best_epoch)
    print(f"Best val OA: {best_val_oa:.4f}")
    print(f"Test OA: {test_result['oa']:.4f}")
    print(f"Test Macro-F1: {test_result['macro_f1']:.4f}")
    print(f"Test Weighted-F1: {test_result['weighted_f1']:.4f}")

    np.savetxt(result_dir / "confusion_matrix.csv", test_result["cm"], delimiter=",", fmt="%d")
    target_names = [LCZ_CLASS_NAMES[c] for c in LCZ_CLASSES]
    report = classification_report(
        test_result["labels"], test_result["preds"], labels=list(range(len(LCZ_CLASSES))),
        target_names=target_names, zero_division=0, output_dict=True
    )
    with open(result_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    summary = {
        "model": "CNN_RS_BuildingGraphEncoder_V5_LightweightHyperEdge",
        "best_epoch": int(best_epoch),
        "best_val_oa": float(best_val_oa),
        "test_oa": float(test_result["oa"]),
        "test_macro_f1": float(test_result["macro_f1"]),
        "test_weighted_f1": float(test_result["weighted_f1"]),
        "params_total": int(total_params),
        "params_trainable": int(trainable_params),
        "classes": LCZ_CLASSES,
        "class_names": target_names,
        "graph_meta": meta,
    }
    with open(result_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("Saved results to:", result_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=DEFAULT_BASE_DIR)
    parser.add_argument("--work_dir", type=str, default=DEFAULT_WORK_DIR)
    parser.add_argument("--rs_norm_dir", type=str, required=True)
    parser.add_argument("--building_shp", type=str, required=True)
    parser.add_argument("--story_col", type=str, default="A10")
    parser.add_argument("--split_dir", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="graph_v5_lightweight_hyperedge")

    parser.add_argument("--patch_size", type=int, default=33)
    parser.add_argument("--graph_context_m", type=float, default=500.0)
    parser.add_argument("--num_rings", type=int, default=3)
    parser.add_argument("--num_sectors", type=int, default=8)

    parser.add_argument("--graph_hidden", type=int, default=64)
    parser.add_argument("--graph_out", type=int, default=128)
    parser.add_argument("--graph_layers", type=int, default=2)
    parser.add_argument("--graph_heads", type=int, default=2)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--class_weight", action="store_true")
    parser.add_argument("--rebuild_graph_cache", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
