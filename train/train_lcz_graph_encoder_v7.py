#!/usr/bin/env python3
"""
CNN_RS_BuildingGraphEncoder_V7

V7 proposal: UMP-aware salient-node + multi-scale morphology token graph + edge-type-aware attention.
Only graph construction / representation / fusion-related components are changed from the V2 family.
Training settings can be kept identical to the previous graph baseline.

Recommended first run:
CUDA_VISIBLE_DEVICES=0 python train/train_lcz_graph_encoder_v2.py \
  --base_dir /mnt/disk1/workspace_jym/LCZ/data \
  --work_dir /mnt/disk1/workspace_jym/LCZ/work_dirs \
  --rs_norm_dir /mnt/disk1/workspace_jym/LCZ/data/Satellite/processed/norm \
  --building_shp /mnt/disk1/workspace_jym/LCZ/data/Building/AL_11_D010_20200502/AL_11_D010_20200502.shp \
  --story_col A10 \
  --split_dir /mnt/disk1/workspace_jym/LCZ/work_dirs/cnn_rs_baseline/splits \
  --exp_name cnn_rs_building_graph_encoder_v2_hybrid500 \
  --patch_size 33 --graph_context_m 500 --edge_mode hybrid --knn 8 --radius_m 120 \
  --max_nodes 192 --graph_hidden 96 --graph_out 192 --graph_layers 3 --graph_heads 4 \
  --batch_size 96 --epochs 180 --patience 35 --lr 5e-4 --dropout 0.35 \
  --class_weight --monitor macro_f1 --rebuild_graph_cache
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.io import loadmat
from scipy.spatial import cKDTree
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DEFAULT_BASE_DIR = "/mnt/disk1/workspace_jym/LCZ/data"
DEFAULT_WORK_DIR = "/mnt/disk1/workspace_jym/LCZ/work_dirs"
DEFAULT_BUILDING_SHP = "/mnt/disk1/workspace_jym/LCZ/data/Building/AL_11_D010_20200502/AL_11_D010_20200502.shp"

LCZ_CLASSES = [1, 2, 3, 4, 5, 6, 8, 101, 102, 104, 107]
LCZ_CLASS_NAMES = {
    1: "LCZ1", 2: "LCZ2", 3: "LCZ3", 4: "LCZ4", 5: "LCZ5", 6: "LCZ6", 8: "LCZ8",
    101: "LCZA", 102: "LCZB", 104: "LCZD", 107: "LCZG",
}

# V7 node features:
# 0 area_norm_or_bsf, 1 story_norm_or_bh, 2 story_missing_or_missing_ratio,
# 3 rel_x, 4 rel_y, 5 dist_norm_or_scale_norm, 6 cos_theta, 7 sin_theta,
# 8 bbox_w_norm_or_bdf, 9 bbox_h_norm_or_bof, 10 aspect_norm_or_high_story_ratio,
# 11 rect_fill_or_density, 12 compactness_or_built_ratio,
# 13 local_density_80m_or_count, 14 local_story_80m_or_std_story,
# 15 perimeter_norm_or_bof, 16 local_built_ratio_or_bdf,
# 17 is_building, 18 is_ump_token, 19 scale_norm, 20 is_global_token, 21 has_node
NODE_DIM = 22

# Per-scale global features, repeated for each configured global scale:
# 0 count_norm, 1 density_norm, 2 total_area_norm, 3 built_ratio/BSF,
# 4 mean_area_norm, 5 max_area_norm,
# 6 mean_story_norm/BH, 7 max_story_norm, 8 std_story_norm,
# 9 mean_dist_norm, 10 min_dist_norm, 11 story_valid_ratio,
# 12 mean_perimeter_norm/BOF, 13 max_perimeter_norm,
# 14 BDF_norm, 15 BOF_norm, 16 high_story_ratio, 17 low_story_ratio
GLOBAL_BASE_DIM = 18
DEFAULT_GLOBAL_SCALES_M = [330.0, 500.0, 700.0]
GLOBAL_DIM = GLOBAL_BASE_DIM * len(DEFAULT_GLOBAL_SCALES_M)

# Edge types used by edge-type-aware graph attention.
# 0 none/padding, 1 self, 2 knn, 3 radius, 4 knn+radius, 5 building-UMP, 6 UMP-UMP
NUM_EDGE_TYPES = 7


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_rs_stack(rs_norm_dir: Path):
    print("========== Load RS bands ==========")
    paths = [rs_norm_dir / f"b{i}_norm.mat" for i in range(1, 9)]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing RS norm files: {missing}")
    bands = []
    for p in paths:
        data = loadmat(p)
        if "norm" not in data:
            raise KeyError(f"'norm' not found in {p}")
        arr = data["norm"].astype(np.float32)
        arr[~np.isfinite(arr)] = 0.0
        bands.append(arr)
        print(f"{p.name}: shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")
    stack = np.stack(bands, axis=0).astype(np.float32)
    print("RS stack:", stack.shape)
    return stack, paths


def make_polygon_split(component_csv: Path, out_dir: Path, seed: int = 42, train_ratio: float = 0.6, val_ratio: float = 0.2):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(component_csv)
    df["polygon_id"] = df["polygon_id"].astype(int)
    df["lcz_class"] = df["lcz_class"].astype(int)
    rng = np.random.default_rng(seed)
    records = []
    for cls in LCZ_CLASSES:
        ids = df.loc[df["lcz_class"] == cls, "polygon_id"].values.astype(int)
        if len(ids) == 0:
            print(f"[WARN] class {cls}: no polygons")
            continue
        rng.shuffle(ids)
        n = len(ids)
        if n >= 3:
            n_train = max(1, int(np.floor(n * train_ratio)))
            n_val = max(1, int(np.floor(n * val_ratio)))
            if n_train + n_val >= n:
                n_train = max(1, n - 2)
                n_val = 1
            train_ids = ids[:n_train]
            val_ids = ids[n_train:n_train + n_val]
            test_ids = ids[n_train + n_val:]
        elif n == 2:
            train_ids, val_ids, test_ids = ids[:1], ids[1:], np.array([], dtype=int)
        else:
            train_ids, val_ids, test_ids = ids, np.array([], dtype=int), np.array([], dtype=int)
        for pid in train_ids:
            records.append({"polygon_id": int(pid), "lcz_class": int(cls), "split": "train"})
        for pid in val_ids:
            records.append({"polygon_id": int(pid), "lcz_class": int(cls), "split": "val"})
        for pid in test_ids:
            records.append({"polygon_id": int(pid), "lcz_class": int(cls), "split": "test"})
        print(f"Class {cls}: total={n}, train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    split_df = pd.DataFrame(records)
    split_df.to_csv(out_dir / "polygon_split.csv", index=False, encoding="utf-8-sig")
    return split_df


def build_sample_index(gt_path: Path, polygon_tif: Path, split_df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(gt_path) as src:
        lcz = src.read(1)
        nodata = src.nodata
        gt_transform = src.transform
        gt_bounds = src.bounds
    with rasterio.open(polygon_tif) as src:
        poly = src.read(1)
    if lcz.shape != poly.shape:
        raise ValueError(f"Shape mismatch: lcz={lcz.shape}, poly={poly.shape}")
    class_to_idx = {cls: i for i, cls in enumerate(LCZ_CLASSES)}
    valid = np.isin(lcz, LCZ_CLASSES) & (poly > 0)
    if nodata is not None:
        valid &= (lcz != nodata)
    for split in ["train", "val", "test"]:
        pids = split_df.loc[split_df["split"] == split, "polygon_id"].values.astype(int)
        mask = np.isin(poly, pids) & valid
        rows, cols = np.where(mask)
        labels_raw = lcz[rows, cols].astype(int)
        labels = np.array([class_to_idx[int(v)] for v in labels_raw], dtype=np.int64)
        np.savez_compressed(out_dir / f"{split}_samples.npz", rows=rows.astype(np.int32), cols=cols.astype(np.int32), labels=labels, labels_raw=labels_raw.astype(np.int32))
        print(f"{split}: samples={len(labels)} -> {out_dir / f'{split}_samples.npz'}")
    meta = {
        "lcz_classes": LCZ_CLASSES,
        "class_to_idx": {str(k): v for k, v in class_to_idx.items()},
        "idx_to_class": {str(v): k for k, v in class_to_idx.items()},
        "gt_transform": list(gt_transform)[:6],
        "gt_bounds": list(gt_bounds),
    }
    with open(out_dir / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


class BuildingGraphProviderV7:
    """
    V7 graph provider.

    Key differences from V2:
    1) Salient node selection: keeps not only nearest buildings but also large-area,
       high-story, and ring-sector representative buildings.
    2) Multi-scale global statistics: summarizes building morphology at several
       context radii, while keeping node-level graph representation.
    3) Edge type matrix: distinguishes self, kNN, radius, hybrid, and building-to-UMP morphology-token edges.
    """
    def __init__(self,
                 shp_path: Path,
                 gt_path: Path,
                 story_col: str = "A10",
                 patch_size: int = 33,
                 graph_context_m: Optional[float] = None,
                 global_scales_m: Optional[list] = None,
                 max_nodes: int = 192,
                 knn: int = 8,
                 radius_m: float = 120.0,
                 edge_mode: str = "hybrid",
                 salient_near_ratio: float = 0.45,
                 salient_large_ratio: float = 0.20,
                 salient_high_ratio: float = 0.20,
                 num_sectors: int = 8):
        self.shp_path = shp_path
        self.story_col = story_col
        self.patch_size = patch_size
        self.context_m = float(graph_context_m) if graph_context_m is not None else float(patch_size * 10.0)
        self.global_scales_m = [float(v) for v in (global_scales_m if global_scales_m is not None else DEFAULT_GLOBAL_SCALES_M)]
        if self.context_m not in self.global_scales_m:
            self.global_scales_m = sorted(set(self.global_scales_m + [self.context_m]))
        # Keep GLOBAL_DIM fixed for compatibility. If custom scales are used, exactly 3 are expected.
        if len(self.global_scales_m) != len(DEFAULT_GLOBAL_SCALES_M):
            raise ValueError(f"V7 expects exactly {len(DEFAULT_GLOBAL_SCALES_M)} global scales. Got {self.global_scales_m}")

        self.max_nodes = int(max_nodes)
        self.knn = int(knn)
        self.radius_m = float(radius_m)
        self.edge_mode = edge_mode
        self.half_size_m = self.context_m / 2.0
        self.search_radius = self.half_size_m * np.sqrt(2.0)
        self.max_global_context_m = max(self.global_scales_m)
        self.max_global_half_m = self.max_global_context_m / 2.0
        self.max_global_search_radius = self.max_global_half_m * np.sqrt(2.0)
        self.num_sectors = int(num_sectors)
        self.salient_near_ratio = float(salient_near_ratio)
        self.salient_large_ratio = float(salient_large_ratio)
        self.salient_high_ratio = float(salient_high_ratio)

        with rasterio.open(gt_path) as src:
            self.gt_transform = src.transform
            self.gt_bounds = src.bounds
            self.gt_crs = src.crs

        print("========== Load building graph source ==========")
        print("SHP:", shp_path)
        try:
            gdf = gpd.read_file(shp_path)
        except UnicodeDecodeError:
            gdf = gpd.read_file(shp_path, encoding="cp949")
        print("Original CRS:", gdf.crs)
        print("Original features:", len(gdf))
        if story_col not in gdf.columns:
            raise KeyError(f"{story_col} not found in building SHP columns: {list(gdf.columns)}")

        gdf = gdf.to_crs(self.gt_crs)
        b = self.gt_bounds
        pad = self.max_global_half_m + 100.0
        gdf = gdf.cx[b.left - pad:b.right + pad, b.bottom - pad:b.top + pad].copy()
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
        gdf["geometry"] = gdf.geometry.buffer(0)
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
        gdf["area_m2"] = gdf.geometry.area.astype(np.float32)
        gdf["perimeter_m"] = gdf.geometry.length.astype(np.float32)
        gdf["story"] = pd.to_numeric(gdf[story_col], errors="coerce").fillna(0).astype(np.float32)
        gdf.loc[gdf["story"] < 0, "story"] = 0.0
        gdf = gdf[gdf["area_m2"] > 0].copy()
        gdf["centroid"] = gdf.geometry.centroid
        bounds = gdf.geometry.bounds
        gdf["cx"] = gdf["centroid"].x.astype(np.float32)
        gdf["cy"] = gdf["centroid"].y.astype(np.float32)
        gdf["bbox_w"] = (bounds["maxx"] - bounds["minx"]).astype(np.float32)
        gdf["bbox_h"] = (bounds["maxy"] - bounds["miny"]).astype(np.float32)
        print("Clipped valid features:", len(gdf))
        print("Area stats:\n", gdf["area_m2"].describe())
        print("Story stats:\n", gdf["story"].describe())

        self.cx = gdf["cx"].to_numpy(np.float32)
        self.cy = gdf["cy"].to_numpy(np.float32)
        self.area = gdf["area_m2"].to_numpy(np.float32)
        self.perimeter = gdf["perimeter_m"].to_numpy(np.float32)
        self.story = gdf["story"].to_numpy(np.float32)
        self.bbox_w = gdf["bbox_w"].to_numpy(np.float32)
        self.bbox_h = gdf["bbox_h"].to_numpy(np.float32)
        self.xy = np.stack([self.cx, self.cy], axis=1)
        self.tree = cKDTree(self.xy)

        self.log_area = np.log1p(self.area)
        self.area_scale = max(float(np.percentile(self.log_area, 99)), 1e-6)
        self.story_scale = max(float(np.percentile(self.story[self.story > 0], 99)) if np.any(self.story > 0) else 1.0, 1.0)
        self.bbox_scale = max(float(np.percentile(np.log1p(np.maximum(self.bbox_w, self.bbox_h)), 99)), 1e-6)
        self.perim_scale = max(float(np.percentile(np.log1p(self.perimeter), 99)), 1e-6)
        print(f"Context={self.context_m:.1f}m, global_scales={self.global_scales_m}, edge_mode={edge_mode}, knn={knn}, radius={radius_m}, max_nodes={max_nodes}")
        print(f"Salient ratios: near={self.salient_near_ratio}, large={self.salient_large_ratio}, high={self.salient_high_ratio}, sectors={self.num_sectors}")
        print(f"Scales: log_area={self.area_scale:.4f}, story={self.story_scale:.4f}, log_bbox={self.bbox_scale:.4f}, log_perim={self.perim_scale:.4f}")

    def cell_center_xy(self, row50: int, col50: int):
        x = self.gt_transform.c + col50 * self.gt_transform.a + self.gt_transform.a / 2.0
        y = self.gt_transform.f + row50 * self.gt_transform.e + self.gt_transform.e / 2.0
        return float(x), float(y)

    def empty_graph(self):
        node = np.zeros((1, NODE_DIM), dtype=np.float32)
        adj = np.ones((1, 1), dtype=np.float32)
        etype = np.ones((1, 1), dtype=np.int64)
        mask = np.ones((1,), dtype=np.float32)
        glob = np.zeros((GLOBAL_DIM,), dtype=np.float32)
        return node, adj, etype, mask, glob

    def select_salient_nodes(self, idx, dx, dy, dist):
        n = len(idx)
        if n <= self.max_nodes:
            return np.arange(n, dtype=np.int64)

        selected = []
        selected_set = set()

        def add_indices(local_indices):
            for ii in local_indices:
                if int(ii) not in selected_set:
                    selected.append(int(ii))
                    selected_set.add(int(ii))
                if len(selected) >= self.max_nodes:
                    break

        n_near = max(1, int(round(self.max_nodes * self.salient_near_ratio)))
        n_large = max(1, int(round(self.max_nodes * self.salient_large_ratio)))
        n_high = max(1, int(round(self.max_nodes * self.salient_high_ratio)))

        # 1) nearest buildings
        add_indices(np.argsort(dist)[:n_near])

        # 2) largest buildings
        area_score = self.area[idx]
        add_indices(np.argsort(-area_score)[:n_large])

        # 3) high-story buildings, with area as tie-breaker
        story_score = self.story[idx] + 0.05 * np.clip(np.log1p(self.area[idx]) / self.area_scale, 0, 1)
        add_indices(np.argsort(-story_score)[:n_high])

        # 4) sector-diverse representatives: keep spatial coverage.
        remaining_budget = self.max_nodes - len(selected)
        if remaining_budget > 0:
            theta = np.arctan2(dy, dx + 1e-6)
            sector = np.floor(((theta + np.pi) / (2.0 * np.pi)) * self.num_sectors).astype(int)
            sector = np.clip(sector, 0, self.num_sectors - 1)
            per_sector = max(1, int(np.ceil(remaining_budget / self.num_sectors)))
            combined_score = (
                0.45 * np.clip(1.0 - dist / max(self.search_radius, 1e-6), 0, 1)
                + 0.30 * np.clip(np.log1p(self.area[idx]) / self.area_scale, 0, 1)
                + 0.25 * np.clip(self.story[idx] / self.story_scale, 0, 1)
            )
            for s in range(self.num_sectors):
                cand = np.where(sector == s)[0]
                if len(cand) == 0:
                    continue
                order = cand[np.argsort(-combined_score[cand])[:per_sector]]
                add_indices(order)
                if len(selected) >= self.max_nodes:
                    break

        # 5) Fill remaining by combined saliency.
        if len(selected) < self.max_nodes:
            combined_score = (
                0.40 * np.clip(1.0 - dist / max(self.search_radius, 1e-6), 0, 1)
                + 0.35 * np.clip(np.log1p(self.area[idx]) / self.area_scale, 0, 1)
                + 0.25 * np.clip(self.story[idx] / self.story_scale, 0, 1)
            )
            add_indices(np.argsort(-combined_score))

        return np.asarray(selected[:self.max_nodes], dtype=np.int64)

    def get_graph(self, row50: int, col50: int):
        center_x, center_y = self.cell_center_xy(row50, col50)
        candidate = self.tree.query_ball_point([center_x, center_y], r=self.max_global_search_radius)
        if len(candidate) == 0:
            return self.empty_graph()

        idx_all = np.asarray(candidate, dtype=np.int64)
        dx_all = self.cx[idx_all] - center_x
        dy_all = self.cy[idx_all] - center_y
        dist_all = np.sqrt(dx_all * dx_all + dy_all * dy_all)

        # Main graph nodes use graph_context_m square.
        inside_main = (np.abs(dx_all) <= self.half_size_m) & (np.abs(dy_all) <= self.half_size_m)
        idx = idx_all[inside_main]
        dx = dx_all[inside_main]
        dy = dy_all[inside_main]
        dist = dist_all[inside_main]
        if len(idx) == 0:
            glob = self.make_multiscale_global_features(idx_all, dx_all, dy_all, dist_all)
            ump_nodes, _ = self.make_ump_token_nodes(idx_all, dx_all, dy_all, dist_all)
            if len(ump_nodes) > 0 and np.any(ump_nodes[:, -1] > 0):
                n = ump_nodes.shape[0]
                adj = np.eye(n, dtype=np.float32)
                etype = np.eye(n, dtype=np.int64)
                # Connect UMP tokens weakly to share multi-scale context.
                for i in range(n):
                    for j in range(i + 1, n):
                        adj[i, j] = adj[j, i] = 0.5
                        etype[i, j] = etype[j, i] = 6
                mask = np.ones((n,), dtype=np.float32)
                return ump_nodes.astype(np.float32), adj, etype, mask, glob
            node, adj, etype, mask, _ = self.empty_graph()
            return node, adj, etype, mask, glob

        keep = self.select_salient_nodes(idx, dx, dy, dist)
        idx, dx, dy, dist = idx[keep], dx[keep], dy[keep], dist[keep]

        area = self.area[idx]
        story = self.story[idx]
        bbox_w = self.bbox_w[idx]
        bbox_h = self.bbox_h[idx]
        perim = self.perimeter[idx]

        area_norm = np.clip(np.log1p(area) / self.area_scale, 0.0, 1.0)
        story_norm = np.clip(story / self.story_scale, 0.0, 1.0)
        story_missing = (story <= 0).astype(np.float32)
        rel_x = np.clip(dx / self.half_size_m, -1.0, 1.0)
        rel_y = np.clip(dy / self.half_size_m, -1.0, 1.0)
        dist_norm = np.clip(dist / self.search_radius, 0.0, 1.0)
        theta = np.arctan2(dy, dx + 1e-6)
        cos_t = np.cos(theta).astype(np.float32)
        sin_t = np.sin(theta).astype(np.float32)
        bbox_w_norm = np.clip(np.log1p(bbox_w) / self.bbox_scale, 0.0, 1.0)
        bbox_h_norm = np.clip(np.log1p(bbox_h) / self.bbox_scale, 0.0, 1.0)
        aspect = bbox_w / np.maximum(bbox_h, 1e-6)
        aspect_norm = np.clip(np.abs(np.log(np.maximum(aspect, 1e-6))) / np.log(10.0), 0.0, 1.0)
        rect_fill = np.clip(area / np.maximum(bbox_w * bbox_h, 1e-6), 0.0, 1.0)
        compactness = np.clip((4.0 * math.pi * area) / np.maximum(perim * perim, 1e-6), 0.0, 1.0)

        # Local morphology around each selected building. Cache-time only, so O(N^2) is acceptable.
        xy_rel = np.stack([dx, dy], axis=1).astype(np.float32)
        local_density = np.zeros_like(area_norm, dtype=np.float32)
        local_story = np.zeros_like(area_norm, dtype=np.float32)
        local_built_ratio = np.zeros_like(area_norm, dtype=np.float32)
        if len(idx) > 1:
            dmat = np.sqrt(((xy_rel[:, None, :] - xy_rel[None, :, :]) ** 2).sum(axis=2))
            neigh = (dmat <= 80.0) & (dmat > 0)
            local_count = neigh.sum(axis=1).astype(np.float32)
            local_density = np.clip(local_count / 20.0, 0.0, 1.0)
            local_area_den = math.pi * (80.0 ** 2)
            for i in range(len(idx)):
                vals = story[neigh[i] & (story > 0)]
                local_story[i] = float(np.clip(vals.mean() / self.story_scale, 0.0, 1.0)) if len(vals) else 0.0
                member = neigh[i].copy()
                member[i] = True
                local_built_ratio[i] = float(np.clip(area[member].sum() / max(local_area_den, 1.0), 0.0, 1.0))
        else:
            local_built_ratio[:] = np.clip(area / (math.pi * (80.0 ** 2)), 0.0, 1.0)

        perimeter_norm = np.clip(np.log1p(perim) / self.perim_scale, 0.0, 1.0)
        is_building = np.ones_like(area_norm, dtype=np.float32)
        is_ump = np.zeros_like(area_norm, dtype=np.float32)
        scale_norm_node = np.zeros_like(area_norm, dtype=np.float32)
        is_global = np.zeros_like(area_norm, dtype=np.float32)
        has_node = np.ones_like(area_norm, dtype=np.float32)

        building_node = np.stack([
            area_norm, story_norm, story_missing,
            rel_x, rel_y, dist_norm, cos_t, sin_t,
            bbox_w_norm, bbox_h_norm, aspect_norm,
            rect_fill, compactness,
            local_density, local_story,
            perimeter_norm, local_built_ratio,
            is_building, is_ump, scale_norm_node, is_global, has_node,
        ], axis=1).astype(np.float32)

        glob = self.make_multiscale_global_features(idx_all, dx_all, dy_all, dist_all)
        ump_nodes, _ = self.make_ump_token_nodes(idx_all, dx_all, dy_all, dist_all)
        ump_scale_masks = [((np.abs(dx) <= (s / 2.0)) & (np.abs(dy) <= (s / 2.0))) for s in self.global_scales_m]
        node = np.concatenate([building_node, ump_nodes], axis=0).astype(np.float32)
        adj, etype = self.make_adj_and_edge_type_with_ump(xy_rel, dx, dy, ump_scale_masks)
        mask = np.ones((node.shape[0],), dtype=np.float32)
        return node, adj, etype, mask, glob

    def make_adj_and_edge_type(self, xy_rel: np.ndarray):
        # Backward-compatible pairwise building graph.
        n = xy_rel.shape[0]
        adj = np.zeros((n, n), dtype=np.float32)
        etype = np.zeros((n, n), dtype=np.int64)
        if n == 1:
            adj[0, 0] = 1.0
            etype[0, 0] = 1
            return adj, etype
        d = np.sqrt(((xy_rel[:, None, :] - xy_rel[None, :, :]) ** 2).sum(axis=2))
        sigma = max(self.radius_m, self.half_size_m / 2.0, 1e-6)

        if self.edge_mode in ["knn", "hybrid"]:
            k = min(self.knn, n - 1)
            for i in range(n):
                nn_idx = np.argsort(d[i])[1:k + 1]
                weights = np.exp(-d[i, nn_idx] / sigma).astype(np.float32)
                adj[i, nn_idx] = np.maximum(adj[i, nn_idx], weights)
                adj[nn_idx, i] = np.maximum(adj[nn_idx, i], weights)
                etype[i, nn_idx] = np.maximum(etype[i, nn_idx], 2)
                etype[nn_idx, i] = np.maximum(etype[nn_idx, i], 2)

        if self.edge_mode in ["radius", "hybrid"]:
            radius_mask = (d <= self.radius_m) & (d > 0)
            weights = np.exp(-d / sigma).astype(np.float32)
            already = (adj > 0) & radius_mask
            new = (adj <= 0) & radius_mask
            adj = np.maximum(adj, weights * radius_mask.astype(np.float32))
            etype[new] = 3
            etype[already] = 4

        np.fill_diagonal(adj, 1.0)
        np.fill_diagonal(etype, 1)
        return adj, etype

    def make_adj_and_edge_type_with_ump(self, xy_rel: np.ndarray, dx: np.ndarray, dy: np.ndarray, ump_scale_masks: list):
        nb = xy_rel.shape[0]
        nu = len(ump_scale_masks)
        n = nb + nu
        adj = np.zeros((n, n), dtype=np.float32)
        etype = np.zeros((n, n), dtype=np.int64)
        if nb > 0:
            b_adj, b_etype = self.make_adj_and_edge_type(xy_rel)
            adj[:nb, :nb] = b_adj
            etype[:nb, :nb] = b_etype
        # UMP morphology-token nodes aggregate group-level building structure.
        # Each UMP token connects to selected building nodes inside its corresponding scale.
        for u, member in enumerate(ump_scale_masks):
            ui = nb + u
            adj[ui, ui] = 1.0
            etype[ui, ui] = 1
            if nb > 0 and member is not None and len(member) == nb:
                dist = np.sqrt(dx * dx + dy * dy)
                scale = self.global_scales_m[u]
                sigma = max(scale / 2.0, 1.0)
                weights = np.exp(-dist / sigma).astype(np.float32)
                weights = np.maximum(weights, 0.25) * member.astype(np.float32)
                for bi in np.where(member)[0]:
                    adj[ui, bi] = adj[bi, ui] = max(float(weights[bi]), 0.25)
                    etype[ui, bi] = etype[bi, ui] = 5
        # Weakly connect UMP tokens to each other so scale summaries can communicate.
        for i in range(nu):
            for j in range(i + 1, nu):
                a = nb + i
                b = nb + j
                adj[a, b] = adj[b, a] = 0.5
                etype[a, b] = etype[b, a] = 6
        np.fill_diagonal(adj, 1.0)
        np.fill_diagonal(etype, 1)
        return adj, etype

    def make_ump_token_nodes(self, idx_all, dx_all, dy_all, dist_all):
        nodes = []
        selected_masks = []
        # Selected building nodes use graph_context coordinates. For each UMP scale,
        # the membership mask is computed later against selected dx/dy.
        for scale_m in self.global_scales_m:
            half = scale_m / 2.0
            inside = (np.abs(dx_all) <= half) & (np.abs(dy_all) <= half)
            idx = idx_all[inside]
            dist = dist_all[inside]
            if len(idx) == 0:
                feat = np.zeros((NODE_DIM,), dtype=np.float32)
                feat[18] = 1.0
                feat[19] = float(np.clip(scale_m / max(self.max_global_context_m, 1.0), 0.0, 1.0))
                feat[21] = 1.0
                nodes.append(feat)
                selected_masks.append(None)
                continue
            area = self.area[idx]
            story = self.story[idx]
            perim = self.perimeter[idx]
            scale_feat = self.make_global_features_for_scale(area, story, dist, scale_m, perim=perim)
            # Decode the most important UMP-like quantities into a node feature vector.
            count_norm, density_norm, total_area_norm, bsf = scale_feat[0], scale_feat[1], scale_feat[2], scale_feat[3]
            mean_area_norm, max_area_norm = scale_feat[4], scale_feat[5]
            bh, max_story, std_story = scale_feat[6], scale_feat[7], scale_feat[8]
            mean_perim, max_perim, bdf, bof = scale_feat[12], scale_feat[13], scale_feat[14], scale_feat[15]
            high_story_ratio, low_story_ratio = scale_feat[16], scale_feat[17]
            missing_ratio = 1.0 - scale_feat[11]
            feat = np.zeros((NODE_DIM,), dtype=np.float32)
            feat[0] = bsf                    # BSF-like
            feat[1] = bh                     # BH/story-like
            feat[2] = missing_ratio
            feat[5] = float(np.clip(scale_m / max(self.max_global_context_m, 1.0), 0.0, 1.0))
            feat[8] = bdf                    # BDF-like
            feat[9] = bof                    # BOF-like
            feat[10] = high_story_ratio
            feat[11] = density_norm
            feat[12] = bsf
            feat[13] = count_norm
            feat[14] = std_story
            feat[15] = mean_perim
            feat[16] = mean_area_norm
            feat[17] = 0.0
            feat[18] = 1.0                   # is_ump_token
            feat[19] = float(np.clip(scale_m / max(self.max_global_context_m, 1.0), 0.0, 1.0))
            feat[20] = 0.0
            feat[21] = 1.0
            nodes.append(feat)
            selected_masks.append(None)
        return np.stack(nodes, axis=0).astype(np.float32), selected_masks

    def make_multiscale_global_features(self, idx_all, dx_all, dy_all, dist_all):
        feats = []
        for scale_m in self.global_scales_m:
            half = scale_m / 2.0
            inside = (np.abs(dx_all) <= half) & (np.abs(dy_all) <= half)
            idx = idx_all[inside]
            dist = dist_all[inside]
            if len(idx) == 0:
                feats.append(np.zeros((GLOBAL_BASE_DIM,), dtype=np.float32))
                continue
            area = self.area[idx]
            story = self.story[idx]
            perim = self.perimeter[idx]
            feats.append(self.make_global_features_for_scale(area, story, dist, scale_m, perim=perim))
        return np.concatenate(feats, axis=0).astype(np.float32)

    def make_global_features_for_scale(self, area: np.ndarray, story: np.ndarray, dist: np.ndarray, scale_m: float, perim: Optional[np.ndarray] = None):
        n = len(area)
        if n == 0:
            return np.zeros((GLOBAL_BASE_DIM,), dtype=np.float32)
        context_area_m2 = max(scale_m * scale_m, 1.0)
        search_radius = max((scale_m / 2.0) * np.sqrt(2.0), 1.0)
        story_pos = story[story > 0]
        log_area_norm = np.clip(np.log1p(area) / self.area_scale, 0.0, 1.0)
        story_norm = np.clip(story_pos / self.story_scale, 0.0, 1.0) if len(story_pos) else np.array([], dtype=np.float32)
        count_norm = np.clip(n / max(self.max_nodes, 1), 0.0, 1.0)
        density_norm = np.clip((n / context_area_m2) / 0.001, 0.0, 1.0)
        total_area_norm = np.clip(np.log1p(float(area.sum())) / (self.area_scale + np.log1p(max(n, 1))), 0.0, 1.0)
        built_ratio = np.clip(float(area.sum()) / context_area_m2, 0.0, 1.0)  # BSF proxy
        mean_area_norm = float(log_area_norm.mean())
        max_area_norm = float(log_area_norm.max())
        mean_story_norm = float(story_norm.mean()) if len(story_norm) else 0.0  # BH/story proxy
        max_story_norm = float(story_norm.max()) if len(story_norm) else 0.0
        std_story_norm = float(story_norm.std()) if len(story_norm) else 0.0
        mean_dist_norm = float(np.clip(dist.mean() / search_radius, 0.0, 1.0))
        min_dist_norm = float(np.clip(dist.min() / search_radius, 0.0, 1.0))
        story_valid_ratio = float(len(story_pos) / max(n, 1))
        if perim is None:
            # Fallback: square-footprint perimeter approximation.
            perim = np.sqrt(np.maximum(area, 1e-6)) * 4.0
        perim_norm = np.clip(np.log1p(perim) / self.perim_scale, 0.0, 1.0)
        mean_perim_norm = float(perim_norm.mean())
        max_perim_norm = float(perim_norm.max())
        bdf_norm = mean_area_norm  # BDF proxy: mean building footprint scale in the region
        bof_norm = mean_perim_norm # BOF proxy: mean outline/perimeter scale in the region
        high_story_ratio = float(((story >= 10).sum()) / max(n, 1))
        low_story_ratio = float((((story > 0) & (story <= 3)).sum()) / max(n, 1))
        return np.array([
            count_norm, density_norm, total_area_norm, built_ratio,
            mean_area_norm, max_area_norm,
            mean_story_norm, max_story_norm, std_story_norm,
            mean_dist_norm, min_dist_norm, story_valid_ratio,
            mean_perim_norm, max_perim_norm, bdf_norm, bof_norm,
            high_story_ratio, low_story_ratio,
        ], dtype=np.float32)

def build_graph_cache_for_split(sample_npz: Path, graph_provider: BuildingGraphProviderV7, cache_path: Path, split_name: str):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.load(sample_npz)
    rows = data["rows"].astype(np.int64)
    cols = data["cols"].astype(np.int64)
    labels = data["labels"].astype(np.int64)
    labels_raw = data["labels_raw"].astype(np.int64) if "labels_raw" in data.files else None

    nodes, adjs, etypes, masks, globals_ = [], [], [], [], []
    node_counts = []
    print(f"\n========== Build graph cache: {split_name} ==========")
    print("sample_npz:", sample_npz)
    print("cache_path:", cache_path)
    print("num_samples:", len(labels))
    for i, (r50, c50) in enumerate(zip(rows, cols), start=1):
        node, adj, etype, mask, glob = graph_provider.get_graph(int(r50), int(c50))
        nodes.append(torch.from_numpy(node.astype(np.float16)))
        adjs.append(torch.from_numpy(adj.astype(np.float16)))
        etypes.append(torch.from_numpy(etype.astype(np.uint8)))
        masks.append(torch.from_numpy(mask.astype(np.uint8)))
        globals_.append(torch.from_numpy(glob.astype(np.float16)))
        node_counts.append(int(node.shape[0]))
        if i == 1 or i % 1000 == 0 or i == len(labels):
            print(f"  cached {i:>7}/{len(labels)} graphs | last_nodes={node.shape[0]}")
    payload = {
        "rows": torch.from_numpy(rows.astype(np.int32)),
        "cols": torch.from_numpy(cols.astype(np.int32)),
        "labels": torch.from_numpy(labels.astype(np.int64)),
        "labels_raw": torch.from_numpy(labels_raw.astype(np.int64)) if labels_raw is not None else None,
        "nodes": nodes,
        "adjs": adjs,
        "edge_types": etypes,
        "masks": masks,
        "globals": globals_,
        "meta": {
            "split_name": split_name,
            "num_samples": int(len(labels)),
            "patch_size": int(graph_provider.patch_size),
            "graph_context_m": float(graph_provider.context_m),
            "global_scales_m": [float(v) for v in graph_provider.global_scales_m],
            "max_nodes": int(graph_provider.max_nodes),
            "knn": int(graph_provider.knn),
            "radius_m": float(graph_provider.radius_m),
            "edge_mode": graph_provider.edge_mode,
            "node_feature_dim": NODE_DIM,
            "global_feature_dim": GLOBAL_DIM,
            "num_edge_types": NUM_EDGE_TYPES,
            "node_count_min": int(np.min(node_counts)) if node_counts else 0,
            "node_count_median": float(np.median(node_counts)) if node_counts else 0,
            "node_count_mean": float(np.mean(node_counts)) if node_counts else 0,
            "node_count_max": int(np.max(node_counts)) if node_counts else 0,
        },
    }
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(cache_path)
    print("Saved graph cache:", cache_path)
    print("Node count stats:", payload["meta"])


class CachedLCZGraphDatasetV7(Dataset):
    def __init__(self, rs_stack: np.ndarray, cache_pt: Path, patch_size: int = 33):
        self.rs_stack = rs_stack
        self.patch_size = patch_size
        self.radius = patch_size // 2
        self.cache_pt = cache_pt
        if not cache_pt.exists():
            raise FileNotFoundError(f"Graph cache not found: {cache_pt}")
        print(f"Load graph cache: {cache_pt}")
        data = torch.load(cache_pt, map_location="cpu")
        self.rows50 = data["rows"].long().numpy()
        self.cols50 = data["cols"].long().numpy()
        self.labels = data["labels"].long()
        self.nodes = data["nodes"]
        self.adjs = data["adjs"]
        self.edge_types = data["edge_types"]
        self.masks = data["masks"]
        self.globals = data["globals"]
        self.meta = data.get("meta", {})
        if not (len(self.rows50) == len(self.cols50) == len(self.labels) == len(self.nodes) == len(self.adjs) == len(self.edge_types) == len(self.masks) == len(self.globals)):
            raise RuntimeError("Cache length mismatch")
        if int(self.meta.get("node_feature_dim", -1)) != NODE_DIM:
            raise RuntimeError(f"Cache node_feature_dim mismatch. Expected {NODE_DIM}, got {self.meta.get('node_feature_dim')}. Use --rebuild_graph_cache.")
        if int(self.meta.get("global_feature_dim", -1)) != GLOBAL_DIM:
            raise RuntimeError(f"Cache global_feature_dim mismatch. Expected {GLOBAL_DIM}, got {self.meta.get('global_feature_dim')}. Use --rebuild_graph_cache.")
        print("  samples:", len(self.labels))
        print("  meta:", self.meta)
        self.rs_pad = np.pad(rs_stack, ((0, 0), (self.radius, self.radius), (self.radius, self.radius)), mode="constant")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        r50 = int(self.rows50[idx])
        c50 = int(self.cols50[idx])
        r10 = r50 * 5 + 2 + self.radius
        c10 = c50 * 5 + 2 + self.radius
        patch = self.rs_pad[:, r10 - self.radius:r10 + self.radius + 1, c10 - self.radius:c10 + self.radius + 1]
        return {
            "rs": torch.from_numpy(patch.copy()).float(),
            "node": self.nodes[idx].float(),
            "adj": self.adjs[idx].float(),
            "edge_type": self.edge_types[idx].long(),
            "mask": self.masks[idx].float(),
            "global": self.globals[idx].float(),
            "label": self.labels[idx].long(),
        }


def graph_collate_v7(batch):
    rs = torch.stack([b["rs"] for b in batch], dim=0)
    y = torch.stack([b["label"] for b in batch], dim=0)
    glob = torch.stack([b["global"] for b in batch], dim=0)
    max_n = max(b["node"].shape[0] for b in batch)
    feat_dim = batch[0]["node"].shape[1]
    nodes = torch.zeros(len(batch), max_n, feat_dim, dtype=torch.float32)
    adjs = torch.zeros(len(batch), max_n, max_n, dtype=torch.float32)
    edge_types = torch.zeros(len(batch), max_n, max_n, dtype=torch.long)
    masks = torch.zeros(len(batch), max_n, dtype=torch.float32)
    for i, b in enumerate(batch):
        n = b["node"].shape[0]
        nodes[i, :n] = b["node"]
        adjs[i, :n, :n] = b["adj"]
        edge_types[i, :n, :n] = b["edge_type"]
        masks[i, :n] = b["mask"]
    return rs, nodes, adjs, edge_types, masks, glob, y


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, out_dim: int = 256, base_channels: int = 32, dropout: float = 0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1), nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1), nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1), nn.BatchNorm2d(base_channels * 2), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1), nn.BatchNorm2d(base_channels * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, patch_size, patch_size)
            flat_dim = self.encoder(dummy).view(1, -1).shape[1]
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(flat_dim, out_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))

    def forward(self, x):
        return self.proj(self.encoder(x))


class GraphAttentionBlockV7(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1, num_edge_types: int = NUM_EDGE_TYPES):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim {dim} must be divisible by heads {heads}")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.edge_type_bias = nn.Embedding(num_edge_types, heads)
        nn.init.zeros_(self.edge_type_bias.weight)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj, edge_type, mask):
        residual = h
        h_norm = self.norm1(h)
        qkv = self.qkv(h_norm)
        q, k, v = qkv.chunk(3, dim=-1)
        B, N, D = q.shape
        q = q.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        edge_mask = (adj > 0) & (mask[:, :, None] > 0) & (mask[:, None, :] > 0)
        adj_bias = torch.log(adj.clamp_min(1e-6)).unsqueeze(1)
        type_bias = self.edge_type_bias(edge_type.clamp(min=0, max=NUM_EDGE_TYPES - 1)).permute(0, 3, 1, 2)
        logits = logits + adj_bias + type_bias
        logits = logits.masked_fill(~edge_mask.unsqueeze(1), -1e4)

        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, D)
        out = self.proj(out)
        h = residual + self.dropout(out)
        h = h * mask.unsqueeze(-1)
        h = h + self.ffn(self.norm2(h))
        h = h * mask.unsqueeze(-1)
        return h


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, h, mask):
        logits = self.score(h).squeeze(-1)
        logits = logits.masked_fill(mask <= 0, -1e4)
        w = torch.softmax(logits, dim=1).unsqueeze(-1)
        return (h * w).sum(dim=1)


class BuildingGraphEncoderV7(nn.Module):
    def __init__(self, in_dim: int = NODE_DIM, global_dim: int = GLOBAL_DIM, hidden_dim: int = 96, out_dim: int = 192, num_layers: int = 3, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.layers = nn.ModuleList([GraphAttentionBlockV7(hidden_dim, heads=heads, dropout=dropout) for _ in range(num_layers)])
        self.attn_pool = AttentionPool(hidden_dim)
        self.global_mlp = nn.Sequential(nn.Linear(global_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        pooled_dim = hidden_dim * 3 + hidden_dim
        self.out_mlp = nn.Sequential(
            nn.Linear(pooled_dim, out_dim), nn.GELU(), nn.LayerNorm(out_dim), nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim), nn.GELU(),
        )

    def forward(self, node, adj, edge_type, mask, glob):
        h = self.input_mlp(node) * mask.unsqueeze(-1)
        for layer in self.layers:
            h = layer(h, adj, edge_type, mask)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_pool = (h * mask.unsqueeze(-1)).sum(dim=1) / denom
        max_pool = h.masked_fill(mask.unsqueeze(-1) <= 0, -1e4).max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        attn_pool = self.attn_pool(h, mask)
        global_feat = self.global_mlp(glob)
        return self.out_mlp(torch.cat([mean_pool, max_pool, attn_pool, global_feat], dim=1))


class RSGraphLCZModelV7(nn.Module):
    def __init__(self, rs_channels: int, num_classes: int, patch_size: int = 33, graph_hidden: int = 96, graph_out: int = 192, graph_layers: int = 3, graph_heads: int = 4, dropout: float = 0.35):
        super().__init__()
        self.rs_encoder = ConvEncoder(rs_channels, patch_size, out_dim=256, base_channels=32, dropout=dropout * 0.3)
        self.graph_encoder = BuildingGraphEncoderV7(in_dim=NODE_DIM, global_dim=GLOBAL_DIM, hidden_dim=graph_hidden, out_dim=graph_out, num_layers=graph_layers, heads=graph_heads, dropout=dropout * 0.4)
        self.fusion = nn.Sequential(
            nn.LayerNorm(256 + graph_out),
            nn.Dropout(dropout),
            nn.Linear(256 + graph_out, 384), nn.GELU(),
            nn.LayerNorm(384), nn.Dropout(dropout),
            nn.Linear(384, 192), nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(192, num_classes),
        )

    def forward(self, rs, node, adj, edge_type, mask, glob):
        rs_feat = self.rs_encoder(rs)
        graph_feat = self.graph_encoder(node, adj, edge_type, mask, glob)
        return self.fusion(torch.cat([rs_feat, graph_feat], dim=1))


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=1.5, label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none", label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce).clamp(min=1e-6, max=1.0)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


def compute_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader, device, num_classes):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_preds, all_labels = [], []
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for rs, node, adj, edge_type, mask, glob, y in loader:
            rs = rs.to(device, non_blocking=True)
            node = node.to(device, non_blocking=True)
            adj = adj.to(device, non_blocking=True)
            edge_type = edge_type.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            glob = glob.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(rs, node, adj, edge_type, mask, glob)
            loss = criterion(logits, y)
            pred = logits.argmax(1)
            total_loss += loss.item() * y.size(0)
            n += y.size(0)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    labels = np.concatenate(all_labels)
    preds = np.concatenate(all_preds)
    return {
        "loss": total_loss / max(n, 1),
        "oa": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "cm": confusion_matrix(labels, preds, labels=list(range(num_classes))),
        "labels": labels,
        "preds": preds,
    }


def train(args):
    set_seed(args.seed)
    base_dir = Path(args.base_dir)
    work_dir = Path(args.work_dir)
    exp_dir = work_dir / args.exp_name
    split_dir = Path(args.split_dir) if args.split_dir else (work_dir / "cnn_rs_baseline" / "splits")
    ckpt_dir = exp_dir / "checkpoints"
    result_dir = exp_dir / "results"
    for d in [exp_dir / "splits", ckpt_dir, result_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if not all((split_dir / f"{s}_samples.npz").exists() for s in ["train", "val", "test"]):
        print(f"Split files not found in {split_dir}. Build new split there.")
        split_dir.mkdir(parents=True, exist_ok=True)
        split_df = make_polygon_split(base_dir / "GT" / "LCZ_class_from_components.csv", split_dir, args.seed, args.train_ratio, args.val_ratio)
        build_sample_index(base_dir / "GT" / "seoul_LCZ.tif", base_dir / "GT" / "lcz_polygonnumber_50m.tif", split_df, split_dir)
    print("Using split_dir:", split_dir)

    rs_stack, rs_paths = load_rs_stack(Path(args.rs_norm_dir))

    graph_cache_dir = Path(args.graph_cache_dir) if args.graph_cache_dir else (exp_dir / "graph_cache")
    graph_cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"v7_ctx{int(args.graph_context_m if args.graph_context_m else args.patch_size*10)}_gs{'-'.join(str(int(x)) for x in args.global_scales_m)}_{args.edge_mode}_k{args.knn}_r{int(args.radius_m)}_n{args.max_nodes}"
    cache_paths = {
        "train": graph_cache_dir / f"train_graphs_{tag}.pt",
        "val": graph_cache_dir / f"val_graphs_{tag}.pt",
        "test": graph_cache_dir / f"test_graphs_{tag}.pt",
    }
    need_cache = args.rebuild_graph_cache or any(not p.exists() for p in cache_paths.values())
    if need_cache:
        print("Graph cache missing or rebuild requested. Building graph cache once...")
        graph_provider = BuildingGraphProviderV7(
            Path(args.building_shp),
            base_dir / "GT" / "seoul_LCZ.tif",
            story_col=args.story_col,
            patch_size=args.patch_size,
            graph_context_m=args.graph_context_m,
            global_scales_m=args.global_scales_m,
            max_nodes=args.max_nodes,
            knn=args.knn,
            radius_m=args.radius_m,
            edge_mode=args.edge_mode,
            salient_near_ratio=args.salient_near_ratio,
            salient_large_ratio=args.salient_large_ratio,
            salient_high_ratio=args.salient_high_ratio,
            num_sectors=args.num_sectors,
        )
        for split_name in ["train", "val", "test"]:
            build_graph_cache_for_split(split_dir / f"{split_name}_samples.npz", graph_provider, cache_paths[split_name], split_name)
    else:
        print("Using existing graph cache:", graph_cache_dir)

    train_ds = CachedLCZGraphDatasetV7(rs_stack, cache_paths["train"], args.patch_size)
    val_ds = CachedLCZGraphDatasetV7(rs_stack, cache_paths["val"], args.patch_size)
    test_ds = CachedLCZGraphDatasetV7(rs_stack, cache_paths["test"], args.patch_size)
    print(f"Dataset: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=graph_collate_v7)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=graph_collate_v7)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=graph_collate_v7)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = RSGraphLCZModelV7(
        rs_stack.shape[0], len(LCZ_CLASSES), args.patch_size,
        graph_hidden=args.graph_hidden, graph_out=args.graph_out,
        graph_layers=args.graph_layers, graph_heads=args.graph_heads,
        dropout=args.dropout,
    ).to(device)
    num_params = count_parameters(model)
    print("Device:", device)
    print("Trainable parameters:", f"{num_params:,}")

    train_labels = np.load(split_dir / "train_samples.npz")["labels"]
    weight = compute_class_weights(train_labels, len(LCZ_CLASSES)).to(device) if args.class_weight else None
    if args.class_weight:
        print("Class weights:", weight.detach().cpu().numpy())
    if args.loss == "focal":
        criterion = FocalLoss(weight=weight, gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    best_metric, best_epoch, patience = -1.0, -1, 0
    logs = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for rs, node, adj, edge_type, mask, glob, y in train_loader:
            rs = rs.to(device, non_blocking=True)
            node = node.to(device, non_blocking=True)
            adj = adj.to(device, non_blocking=True)
            edge_type = edge_type.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            glob = glob.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(rs, node, adj, edge_type, mask, glob)
            loss = criterion(logits, y)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            n += y.size(0)
        scheduler.step()
        train_loss = total_loss / max(n, 1)
        val = evaluate(model, val_loader, device, len(LCZ_CLASSES))
        metric = val["macro_f1"] if args.monitor == "macro_f1" else val["oa"]
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val['loss']:.4f} | val_OA={val['oa']:.4f} | val_macroF1={val['macro_f1']:.4f} | monitor={metric:.4f}")
        logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "val_oa": val["oa"],
            "val_macro_f1": val["macro_f1"],
            "val_weighted_f1": val["weighted_f1"],
            "lr": optimizer.param_groups[0]["lr"],
        })
        if metric > best_metric:
            best_metric, best_epoch, patience = metric, epoch, 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "rs_channels": int(rs_stack.shape[0]),
                "num_classes": len(LCZ_CLASSES),
                "patch_size": args.patch_size,
                "lcz_classes": LCZ_CLASSES,
                "params": num_params,
                "best_metric": best_metric,
                "monitor": args.monitor,
                "rs_files": [str(p) for p in rs_paths],
                "building_shp": str(args.building_shp),
                "story_col": args.story_col,
                "max_nodes": args.max_nodes,
                "knn": args.knn,
                "radius_m": args.radius_m,
                "edge_mode": args.edge_mode,
                "graph_context_m": args.graph_context_m,
                "graph_hidden": args.graph_hidden,
                "graph_out": args.graph_out,
                "graph_layers": args.graph_layers,
                "graph_heads": args.graph_heads,
            }, ckpt_dir / "best_model.pt")
            print("  Saved best model")
        else:
            patience += 1
        if patience >= args.patience:
            print("Early stopping")
            break

    pd.DataFrame(logs).to_csv(result_dir / "train_log.csv", index=False, encoding="utf-8-sig")
    ckpt = torch.load(ckpt_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test = evaluate(model, test_loader, device, len(LCZ_CLASSES))
    np.savetxt(result_dir / "confusion_matrix.csv", test["cm"], delimiter=",", fmt="%d")
    target_names = [LCZ_CLASS_NAMES[c] for c in LCZ_CLASSES]
    report = classification_report(test["labels"], test["preds"], labels=list(range(len(LCZ_CLASSES))), target_names=target_names, zero_division=0, output_dict=True)
    with open(result_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    summary = {
        "model": "CNN_RS_BuildingGraphEncoder_V7",
        "best_epoch": int(best_epoch),
        "best_monitor": args.monitor,
        "best_metric": float(best_metric),
        "test_oa": float(test["oa"]),
        "test_macro_f1": float(test["macro_f1"]),
        "test_weighted_f1": float(test["weighted_f1"]),
        "params": int(num_params),
        "graph_cache_dir": str(graph_cache_dir),
        "max_nodes": int(args.max_nodes),
        "knn": int(args.knn),
        "radius_m": float(args.radius_m),
        "edge_mode": args.edge_mode,
        "graph_context_m": float(args.graph_context_m if args.graph_context_m is not None else args.patch_size * 10.0),
        "global_scales_m": [float(v) for v in args.global_scales_m],
        "salient_near_ratio": float(args.salient_near_ratio),
        "salient_large_ratio": float(args.salient_large_ratio),
        "salient_high_ratio": float(args.salient_high_ratio),
        "num_sectors": int(args.num_sectors),
        "graph_hidden": int(args.graph_hidden),
        "graph_out": int(args.graph_out),
        "graph_layers": int(args.graph_layers),
        "graph_heads": int(args.graph_heads),
        "loss": args.loss,
        "label_smoothing": float(args.label_smoothing),
        "classes": LCZ_CLASSES,
        "class_names": target_names,
    }
    with open(result_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("========== Test ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Saved results to:", result_dir)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", type=str, default=DEFAULT_BASE_DIR)
    p.add_argument("--work_dir", type=str, default=DEFAULT_WORK_DIR)
    p.add_argument("--rs_norm_dir", type=str, default="/mnt/disk1/workspace_jym/LCZ/data/Satellite/processed/norm")
    p.add_argument("--building_shp", type=str, default=DEFAULT_BUILDING_SHP)
    p.add_argument("--story_col", type=str, default="A10")
    p.add_argument("--split_dir", type=str, default=None)
    p.add_argument("--exp_name", type=str, default="cnn_rs_building_graph_encoder_v7")
    p.add_argument("--patch_size", type=int, default=33)
    p.add_argument("--graph_context_m", type=float, default=500.0)
    p.add_argument("--global_scales_m", type=float, nargs="+", default=DEFAULT_GLOBAL_SCALES_M)
    p.add_argument("--salient_near_ratio", type=float, default=0.45)
    p.add_argument("--salient_large_ratio", type=float, default=0.20)
    p.add_argument("--salient_high_ratio", type=float, default=0.20)
    p.add_argument("--num_sectors", type=int, default=8)
    p.add_argument("--edge_mode", type=str, default="hybrid", choices=["knn", "radius", "hybrid"])
    p.add_argument("--radius_m", type=float, default=120.0)
    p.add_argument("--max_nodes", type=int, default=192)
    p.add_argument("--knn", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=96)
    p.add_argument("--epochs", type=int, default=180)
    p.add_argument("--patience", type=int, default=35)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.35)
    p.add_argument("--graph_hidden", type=int, default=96)
    p.add_argument("--graph_out", type=int, default=192)
    p.add_argument("--graph_layers", type=int, default=3)
    p.add_argument("--graph_heads", type=int, default=4)
    p.add_argument("--loss", type=str, default="ce", choices=["ce", "focal"])
    p.add_argument("--focal_gamma", type=float, default=1.5)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--monitor", type=str, default="oa", choices=["oa", "macro_f1"])
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--graph_cache_dir", type=str, default=None)
    p.add_argument("--rebuild_graph_cache", action="store_true")
    p.add_argument("--train_ratio", type=float, default=0.6)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--class_weight", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
