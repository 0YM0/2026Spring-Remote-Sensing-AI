#!/usr/bin/env python3
"""
CNN_RS_BuildingGraphEncoder_V4

Graph-only improvement version.
Compared with the cached graph baseline, this version changes only graph construction,
graph representation, and fusion.

Main ideas:
1) Add a center/query node representing the target 50 m LCZ cell.
2) Add fixed radial-sector summary nodes to represent coarse building distribution.
3) Keep building polygons as object nodes.
4) Use typed spatial edges: self, center-building, center-sector, sector-building, building-kNN, building-radius, hybrid.
5) Use edge-aware graph attention.
6) Use center-aware graph readout and RS-guided gated fusion.

Default training settings are intentionally conservative:
- CE loss only
- AdamW optimizer
- no scheduler
- same split_dir reuse


CUDA_VISIBLE_DEVICES=0 python train/train_lcz_graph_encoder_v4.py \
  --base_dir /mnt/disk1/workspace_jym/LCZ/data \
  --work_dir /mnt/disk1/workspace_jym/LCZ/work_dirs \
  --rs_norm_dir /mnt/disk1/workspace_jym/LCZ/data/Satellite/processed/norm \
  --building_shp /mnt/disk1/workspace_jym/LCZ/data/Building/AL_11_D010_20200502/AL_11_D010_20200502.shp \
  --story_col A10 \
  --split_dir /mnt/disk1/workspace_jym/LCZ/work_dirs/cnn_rs_baseline/splits \
  --exp_name graph_v4_center_sector_hybrid500 \
  --patch_size 33 \
  --graph_context_m 500 \
  --edge_mode hybrid \
  --knn 6 \
  --radius_m 120 \
  --max_nodes 224 \
  --graph_hidden 96 \
  --graph_out 192 \
  --graph_layers 3 \
  --graph_heads 4 \
  --batch_size 128 \
  --epochs 150 \
  --patience 25 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --dropout 0.5 \
  --class_weight \
  --rebuild_graph_cache
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
DEFAULT_RS_NORM = "/mnt/disk1/workspace_jym/LCZ/data/Satellite/processed/norm"

LCZ_CLASSES = [1, 2, 3, 4, 5, 6, 8, 101, 102, 104, 107]
LCZ_CLASS_NAMES = {
    1: "LCZ1", 2: "LCZ2", 3: "LCZ3", 4: "LCZ4", 5: "LCZ5", 6: "LCZ6", 8: "LCZ8",
    101: "LCZA", 102: "LCZB", 104: "LCZD", 107: "LCZG",
}

# Node features:
# 0 is_center, 1 is_building, 2 is_sector,
# 3 area_norm, 4 story_norm, 5 story_missing,
# 6 rel_x, 7 rel_y, 8 dist_norm, 9 cos_theta, 10 sin_theta,
# 11 bbox_w_norm, 12 bbox_h_norm, 13 aspect_norm,
# 14 rect_fill, 15 compactness,
# 16 sector_count_norm, 17 sector_built_ratio
NODE_DIM = 18

# Global graph features:
# 0 count_norm, 1 density_norm, 2 total_area_norm, 3 built_ratio,
# 4 mean_area_norm, 5 std_area_norm, 6 max_area_norm,
# 7 mean_story_norm, 8 std_story_norm, 9 max_story_norm,
# 10 high_story_ratio, 11 mean_dist_norm, 12 min_dist_norm, 13 story_valid_ratio
GLOBAL_DIM = 14

# edge type ids
EDGE_NONE = 0
EDGE_SELF = 1
EDGE_CENTER = 2
EDGE_KNN = 3
EDGE_RADIUS = 4
EDGE_HYBRID = 5
EDGE_SECTOR = 6
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
        np.savez_compressed(
            out_dir / f"{split}_samples.npz",
            rows=rows.astype(np.int32),
            cols=cols.astype(np.int32),
            labels=labels,
            labels_raw=labels_raw.astype(np.int32),
        )
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


class CenterSectorBuildingGraphProvider:
    """
    V4 graph construction.

    Nodes:
      index 0: center/query node for the target 50 m LCZ cell
      index 1..S: fixed radial-sector summary nodes
      remaining: building polygon nodes

    Edges:
      self-loop
      center <-> sector
      center <-> building
      sector <-> building membership
      building <-> building hybrid spatial edges
    """

    def __init__(self,
                 shp_path: Path,
                 gt_path: Path,
                 story_col: str = "A10",
                 patch_size: int = 33,
                 graph_context_m: Optional[float] = None,
                 max_nodes: int = 224,
                 knn: int = 6,
                 radius_m: float = 120.0,
                 edge_mode: str = "hybrid",
                 num_rings: int = 3,
                 num_sectors: int = 4):
        self.shp_path = shp_path
        self.story_col = story_col
        self.patch_size = patch_size
        self.context_m = float(graph_context_m) if graph_context_m is not None else float(patch_size * 10.0)
        self.max_nodes = int(max_nodes)
        self.knn = int(knn)
        self.radius_m = float(radius_m)
        self.edge_mode = edge_mode
        self.num_rings = int(num_rings)
        self.num_sectors = int(num_sectors)
        self.num_sector_nodes = self.num_rings * self.num_sectors
        self.max_building_nodes = max(1, self.max_nodes - 1 - self.num_sector_nodes)
        self.half_size_m = self.context_m / 2.0
        self.search_radius = self.half_size_m * np.sqrt(2.0)
        self.context_area_m2 = self.context_m * self.context_m
        self.sector_area_m2 = self.context_area_m2 / max(self.num_sector_nodes, 1)

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
        pad = self.half_size_m + 50.0
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
        print(f"Context={self.context_m:.1f}m, rings={self.num_rings}, sectors={self.num_sectors}, edge_mode={edge_mode}, knn={knn}, radius={radius_m}, max_nodes={max_nodes}, max_buildings={self.max_building_nodes}")
        print(f"Scales: log_area={self.area_scale:.4f}, story={self.story_scale:.4f}, log_bbox={self.bbox_scale:.4f}")

    def cell_center_xy(self, row50: int, col50: int):
        x = self.gt_transform.c + col50 * self.gt_transform.a + self.gt_transform.a / 2.0
        y = self.gt_transform.f + row50 * self.gt_transform.e + self.gt_transform.e / 2.0
        return float(x), float(y)

    def compute_sector_ids(self, dx: np.ndarray, dy: np.ndarray, dist: np.ndarray):
        # ring: 0..num_rings-1 by normalized radial distance
        d_norm = np.clip(dist / max(self.search_radius, 1e-6), 0.0, 0.999999)
        ring = np.floor(d_norm * self.num_rings).astype(np.int64)
        # sector: 0..num_sectors-1 by angle
        theta = np.arctan2(dy, dx)  # -pi..pi
        sector = np.floor(((theta + math.pi) / (2.0 * math.pi)) * self.num_sectors).astype(np.int64)
        sector = np.clip(sector, 0, self.num_sectors - 1)
        return ring * self.num_sectors + sector

    def make_sector_nodes(self, area, story, dx, dy, dist, sector_ids):
        nodes = []
        for sid in range(self.num_sector_nodes):
            m = sector_ids == sid
            ring = sid // self.num_sectors
            sector = sid % self.num_sectors
            theta_center = -math.pi + (sector + 0.5) * (2.0 * math.pi / self.num_sectors)
            # normalized radius of the sector center, then convert to rel_x/rel_y feature scale
            r_norm = (ring + 0.5) / max(self.num_rings, 1)
            rel_x = float(np.clip(r_norm * math.cos(theta_center), -1.0, 1.0))
            rel_y = float(np.clip(r_norm * math.sin(theta_center), -1.0, 1.0))
            dist_norm = float(np.clip(r_norm, 0.0, 1.0))
            cos_t = float(math.cos(theta_center))
            sin_t = float(math.sin(theta_center))
            if m.any():
                a = area[m]
                s = story[m]
                sp = s[s > 0]
                area_norm = float(np.clip(np.log1p(a).mean() / self.area_scale, 0.0, 1.0))
                story_norm = float(np.clip((sp.mean() / self.story_scale) if len(sp) else 0.0, 0.0, 1.0))
                story_missing = 0.0 if len(sp) else 1.0
                count_norm = float(np.clip(m.sum() / max(self.max_building_nodes / max(self.num_sector_nodes, 1), 1.0), 0.0, 1.0))
                built_ratio = float(np.clip(a.sum() / max(self.sector_area_m2, 1.0), 0.0, 1.0))
            else:
                area_norm = story_norm = count_norm = built_ratio = 0.0
                story_missing = 1.0
            nodes.append([
                0.0, 0.0, 1.0,  # is_center, is_building, is_sector
                area_norm, story_norm, story_missing,
                rel_x, rel_y, dist_norm, cos_t, sin_t,
                0.0, 0.0, 0.0,
                0.0, 0.0,
                count_norm, built_ratio,
            ])
        return np.asarray(nodes, dtype=np.float32)

    def empty_graph(self):
        center_node = np.zeros((1, NODE_DIM), dtype=np.float32)
        center_node[0, 0] = 1.0
        sector_nodes = self.make_sector_nodes(
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
        node = np.concatenate([center_node, sector_nodes], axis=0)
        n = node.shape[0]
        adj = np.zeros((n, n), dtype=np.float32)
        edge_type = np.zeros((n, n), dtype=np.int64)
        np.fill_diagonal(adj, 1.0)
        np.fill_diagonal(edge_type, EDGE_SELF)
        # center-sector edges
        adj[0, 1:] = 1.0
        adj[1:, 0] = 1.0
        edge_type[0, 1:] = EDGE_SECTOR
        edge_type[1:, 0] = EDGE_SECTOR
        mask = np.ones((n,), dtype=np.float32)
        glob = np.zeros((GLOBAL_DIM,), dtype=np.float32)
        return node, adj, edge_type, mask, glob

    def get_graph(self, row50: int, col50: int):
        center_x, center_y = self.cell_center_xy(row50, col50)
        candidate = self.tree.query_ball_point([center_x, center_y], r=self.search_radius)
        if len(candidate) == 0:
            return self.empty_graph()
        idx = np.asarray(candidate, dtype=np.int64)
        dx = self.cx[idx] - center_x
        dy = self.cy[idx] - center_y
        inside = (np.abs(dx) <= self.half_size_m) & (np.abs(dy) <= self.half_size_m)
        idx, dx, dy = idx[inside], dx[inside], dy[inside]
        if len(idx) == 0:
            return self.empty_graph()
        dist = np.sqrt(dx * dx + dy * dy)
        if len(idx) > self.max_building_nodes:
            area_score = np.clip(np.log1p(self.area[idx]) / self.area_scale, 0, 1)
            story_score = np.clip(self.story[idx] / self.story_scale, 0, 1)
            # keep a mixture of central, large, and tall buildings
            score = dist / max(self.half_size_m, 1e-6) - 0.12 * area_score - 0.05 * story_score
            order = np.argsort(score)[:self.max_building_nodes]
            idx, dx, dy, dist = idx[order], dx[order], dy[order], dist[order]

        area = self.area[idx]
        story = self.story[idx]
        bbox_w = self.bbox_w[idx]
        bbox_h = self.bbox_h[idx]
        perim = self.perimeter[idx]
        sector_ids = self.compute_sector_ids(dx, dy, dist)

        center_node = np.zeros((1, NODE_DIM), dtype=np.float32)
        center_node[0, 0] = 1.0
        sector_nodes = self.make_sector_nodes(area, story, dx, dy, dist, sector_ids)
        building_nodes = self.make_building_node(area, story, bbox_w, bbox_h, perim, dx, dy, dist)
        node = np.concatenate([center_node, sector_nodes, building_nodes], axis=0)
        adj, edge_type = self.make_adj_with_center_sector(np.stack([dx, dy], axis=1).astype(np.float32), dist.astype(np.float32), sector_ids)
        mask = np.ones((node.shape[0],), dtype=np.float32)
        glob = self.make_global_features(area, story, dist)
        return node, adj, edge_type, mask, glob

    def make_building_node(self, area, story, bbox_w, bbox_h, perim, dx, dy, dist):
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
        is_center = np.zeros_like(area_norm, dtype=np.float32)
        is_building = np.ones_like(area_norm, dtype=np.float32)
        is_sector = np.zeros_like(area_norm, dtype=np.float32)
        count_norm = np.zeros_like(area_norm, dtype=np.float32)
        built_ratio = np.zeros_like(area_norm, dtype=np.float32)
        return np.stack([
            is_center, is_building, is_sector,
            area_norm, story_norm, story_missing,
            rel_x, rel_y, dist_norm, cos_t, sin_t,
            bbox_w_norm, bbox_h_norm, aspect_norm,
            rect_fill, compactness,
            count_norm, built_ratio,
        ], axis=1).astype(np.float32)

    def make_adj_with_center_sector(self, xy_rel: np.ndarray, dist_center: np.ndarray, sector_ids: np.ndarray):
        nb = xy_rel.shape[0]
        s = self.num_sector_nodes
        n = 1 + s + nb
        adj = np.zeros((n, n), dtype=np.float32)
        edge_type = np.zeros((n, n), dtype=np.int64)
        np.fill_diagonal(adj, 1.0)
        np.fill_diagonal(edge_type, EDGE_SELF)

        # center-sector edges
        if s > 0:
            adj[0, 1:1 + s] = 1.0
            adj[1:1 + s, 0] = 1.0
            edge_type[0, 1:1 + s] = EDGE_SECTOR
            edge_type[1:1 + s, 0] = EDGE_SECTOR

        if nb == 0:
            return adj, edge_type

        b0 = 1 + s
        center_sigma = max(self.half_size_m, 1e-6)
        cw = np.exp(-dist_center / center_sigma).astype(np.float32)
        adj[0, b0:] = cw
        adj[b0:, 0] = cw
        edge_type[0, b0:] = EDGE_CENTER
        edge_type[b0:, 0] = EDGE_CENTER

        # sector-building membership edges
        for bi, sid in enumerate(sector_ids):
            si = 1 + int(sid)
            bj = b0 + bi
            adj[si, bj] = 1.0
            adj[bj, si] = 1.0
            edge_type[si, bj] = EDGE_SECTOR
            edge_type[bj, si] = EDGE_SECTOR

        if nb == 1:
            return adj, edge_type

        d = np.sqrt(((xy_rel[:, None, :] - xy_rel[None, :, :]) ** 2).sum(axis=2))
        sigma = max(self.radius_m, self.half_size_m / 2.0, 1e-6)
        b_adj = np.zeros((nb, nb), dtype=np.float32)
        b_type = np.zeros((nb, nb), dtype=np.int64)

        if self.edge_mode in ["knn", "hybrid"]:
            k = min(self.knn, nb - 1)
            for i in range(nb):
                nn_idx = np.argsort(d[i])[1:k + 1]
                weights = np.exp(-d[i, nn_idx] / sigma).astype(np.float32)
                for jj, w in zip(nn_idx, weights):
                    et = EDGE_KNN
                    if b_adj[i, jj] > 0 and b_type[i, jj] == EDGE_RADIUS:
                        et = EDGE_HYBRID
                    b_adj[i, jj] = max(b_adj[i, jj], float(w))
                    b_adj[jj, i] = max(b_adj[jj, i], float(w))
                    b_type[i, jj] = et
                    b_type[jj, i] = et

        if self.edge_mode in ["radius", "hybrid"]:
            radius_mask = (d <= self.radius_m) & (d > 0)
            weights = np.exp(-d / sigma).astype(np.float32)
            ii, jj = np.where(radius_mask)
            for i, j in zip(ii, jj):
                et = EDGE_RADIUS
                if b_adj[i, j] > 0 and b_type[i, j] == EDGE_KNN:
                    et = EDGE_HYBRID
                b_adj[i, j] = max(b_adj[i, j], float(weights[i, j]))
                b_type[i, j] = et

        adj[b0:, b0:] = np.maximum(adj[b0:, b0:], b_adj)
        edge_type[b0:, b0:] = np.maximum(edge_type[b0:, b0:], b_type)
        np.fill_diagonal(adj, 1.0)
        np.fill_diagonal(edge_type, EDGE_SELF)
        return adj, edge_type

    def make_global_features(self, area: np.ndarray, story: np.ndarray, dist: np.ndarray):
        n = len(area)
        if n == 0:
            return np.zeros((GLOBAL_DIM,), dtype=np.float32)
        story_pos = story[story > 0]
        log_area_norm = np.clip(np.log1p(area) / self.area_scale, 0.0, 1.0)
        story_norm = np.clip(story_pos / self.story_scale, 0.0, 1.0) if len(story_pos) else np.array([], dtype=np.float32)
        count_norm = np.clip(n / max(self.max_building_nodes, 1), 0.0, 1.0)
        density_norm = np.clip((n / max(self.context_area_m2, 1.0)) / 0.001, 0.0, 1.0)
        total_area_norm = np.clip(np.log1p(float(area.sum())) / (self.area_scale + np.log1p(max(n, 1))), 0.0, 1.0)
        built_ratio = np.clip(float(area.sum()) / max(self.context_area_m2, 1.0), 0.0, 1.0)
        mean_area_norm = float(log_area_norm.mean())
        std_area_norm = float(log_area_norm.std())
        max_area_norm = float(log_area_norm.max())
        mean_story_norm = float(story_norm.mean()) if len(story_norm) else 0.0
        std_story_norm = float(story_norm.std()) if len(story_norm) else 0.0
        max_story_norm = float(story_norm.max()) if len(story_norm) else 0.0
        high_story_ratio = float((story_pos >= 5).sum() / max(len(story_pos), 1)) if len(story_pos) else 0.0
        mean_dist_norm = float(np.clip(dist.mean() / self.search_radius, 0.0, 1.0))
        min_dist_norm = float(np.clip(dist.min() / self.search_radius, 0.0, 1.0))
        story_valid_ratio = float(len(story_pos) / max(n, 1))
        return np.array([
            count_norm, density_norm, total_area_norm, built_ratio,
            mean_area_norm, std_area_norm, max_area_norm,
            mean_story_norm, std_story_norm, max_story_norm,
            high_story_ratio, mean_dist_norm, min_dist_norm, story_valid_ratio,
        ], dtype=np.float32)

def build_graph_cache_for_split(sample_npz: Path, graph_provider: CenterSectorBuildingGraphProvider, cache_path: Path, split_name: str):
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
        node, adj, edge_type, mask, glob = graph_provider.get_graph(int(r50), int(c50))
        nodes.append(torch.from_numpy(node.astype(np.float16)))
        adjs.append(torch.from_numpy(adj.astype(np.float16)))
        etypes.append(torch.from_numpy(edge_type.astype(np.int64)))
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
            "version": "v4_center_sector",
            "split_name": split_name,
            "num_samples": int(len(labels)),
            "patch_size": int(graph_provider.patch_size),
            "graph_context_m": float(graph_provider.context_m),
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


class CachedCenterGraphDataset(Dataset):
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
        if int(self.meta.get("node_feature_dim", -1)) != NODE_DIM:
            raise RuntimeError(f"Cache node_feature_dim mismatch. Expected {NODE_DIM}, got {self.meta.get('node_feature_dim')}. Use --rebuild_graph_cache.")
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


def center_graph_collate(batch):
    rs = torch.stack([b["rs"] for b in batch], dim=0)
    y = torch.stack([b["label"] for b in batch], dim=0)
    glob = torch.stack([b["global"] for b in batch], dim=0)
    max_n = max(b["node"].shape[0] for b in batch)
    feat_dim = batch[0]["node"].shape[1]
    nodes = torch.zeros(len(batch), max_n, feat_dim, dtype=torch.float32)
    adjs = torch.zeros(len(batch), max_n, max_n, dtype=torch.float32)
    etypes = torch.zeros(len(batch), max_n, max_n, dtype=torch.long)
    masks = torch.zeros(len(batch), max_n, dtype=torch.float32)
    for i, b in enumerate(batch):
        n = b["node"].shape[0]
        nodes[i, :n] = b["node"]
        adjs[i, :n, :n] = b["adj"]
        etypes[i, :n, :n] = b["edge_type"]
        masks[i, :n] = b["mask"]
    return rs, nodes, adjs, etypes, masks, glob, y


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, out_dim: int = 256, base_channels: int = 32, dropout: float = 0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, patch_size, patch_size)
            flat_dim = self.encoder(dummy).view(1, -1).shape[1]
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(flat_dim, out_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))

    def forward(self, x):
        return self.proj(self.encoder(x))


class EdgeAwareGraphAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, num_edge_types: int = NUM_EDGE_TYPES, dropout: float = 0.1):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim {dim} must be divisible by heads {heads}")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.edge_bias = nn.Embedding(num_edge_types, heads)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim), nn.Dropout(dropout))
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
        et_bias = self.edge_bias(edge_type.clamp_min(0).clamp_max(NUM_EDGE_TYPES - 1)).permute(0, 3, 1, 2)
        logits = logits + adj_bias + et_bias
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


class CenterAwareBuildingGraphEncoder(nn.Module):
    def __init__(self, in_dim: int = NODE_DIM, global_dim: int = GLOBAL_DIM, hidden_dim: int = 96, out_dim: int = 192, num_layers: int = 3, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.layers = nn.ModuleList([EdgeAwareGraphAttentionBlock(hidden_dim, heads=heads, dropout=dropout) for _ in range(num_layers)])
        self.attn_pool = AttentionPool(hidden_dim)
        self.global_mlp = nn.Sequential(nn.Linear(global_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        # center node + mean building pool + attention building pool + global stats
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, out_dim), nn.GELU(), nn.LayerNorm(out_dim), nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim), nn.GELU(),
        )

    def forward(self, node, adj, edge_type, mask, glob):
        h = self.input_mlp(node) * mask.unsqueeze(-1)
        for layer in self.layers:
            h = layer(h, adj, edge_type, mask)
        center = h[:, 0]
        bmask = mask.clone()
        bmask[:, 0] = 0.0
        denom = bmask.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_pool = (h * bmask.unsqueeze(-1)).sum(dim=1) / denom
        attn_pool = self.attn_pool(h, bmask + (mask[:, :1].repeat(1, mask.size(1)) * 0.0))
        # When there is no building node, attention pool may attend to masked logits. Guard with zeros.
        no_building = (bmask.sum(dim=1, keepdim=True) <= 0)
        attn_pool = torch.where(no_building, torch.zeros_like(attn_pool), attn_pool)
        global_feat = self.global_mlp(glob)
        return self.out_mlp(torch.cat([center, mean_pool, attn_pool, global_feat], dim=1))


class GatedFusion(nn.Module):
    def __init__(self, rs_dim: int = 256, graph_dim: int = 192, num_classes: int = 11, dropout: float = 0.5):
        super().__init__()
        self.graph_proj = nn.Sequential(nn.Linear(graph_dim, rs_dim), nn.GELU(), nn.LayerNorm(rs_dim))
        self.gate = nn.Sequential(nn.Linear(rs_dim + graph_dim, rs_dim), nn.GELU(), nn.Linear(rs_dim, rs_dim), nn.Sigmoid())
        self.classifier = nn.Sequential(
            nn.LayerNorm(rs_dim + graph_dim),
            nn.Dropout(dropout),
            nn.Linear(rs_dim + graph_dim, 384), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(384, 192), nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(192, num_classes),
        )

    def forward(self, rs_feat, graph_feat):
        gate = self.gate(torch.cat([rs_feat, graph_feat], dim=1))
        graph_residual = self.graph_proj(graph_feat)
        rs_guided = rs_feat + gate * graph_residual
        return self.classifier(torch.cat([rs_guided, graph_feat], dim=1))


class RSGraphLCZModelV4(nn.Module):
    def __init__(self, rs_channels: int, num_classes: int, patch_size: int = 33, graph_hidden: int = 96, graph_out: int = 192, graph_layers: int = 3, graph_heads: int = 4, dropout: float = 0.5):
        super().__init__()
        self.rs_encoder = ConvEncoder(rs_channels, patch_size, out_dim=256, base_channels=32, dropout=dropout * 0.3)
        self.graph_encoder = CenterAwareBuildingGraphEncoder(in_dim=NODE_DIM, global_dim=GLOBAL_DIM, hidden_dim=graph_hidden, out_dim=graph_out, num_layers=graph_layers, heads=graph_heads, dropout=dropout * 0.4)
        self.fusion = GatedFusion(rs_dim=256, graph_dim=graph_out, num_classes=num_classes, dropout=dropout)

    def forward(self, rs, node, adj, edge_type, mask, glob):
        rs_feat = self.rs_encoder(rs)
        graph_feat = self.graph_encoder(node, adj, edge_type, mask, glob)
        return self.fusion(rs_feat, graph_feat)


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
    context_tag = int(args.graph_context_m if args.graph_context_m is not None else args.patch_size * 10)
    tag = f"v4_ctx{context_tag}_{args.edge_mode}_k{args.knn}_r{int(args.radius_m)}_n{args.max_nodes}"
    cache_paths = {
        "train": graph_cache_dir / f"train_graphs_{tag}.pt",
        "val": graph_cache_dir / f"val_graphs_{tag}.pt",
        "test": graph_cache_dir / f"test_graphs_{tag}.pt",
    }
    need_cache = args.rebuild_graph_cache or any(not p.exists() for p in cache_paths.values())
    if need_cache:
        print("Graph cache missing or rebuild requested. Building graph cache once...")
        graph_provider = CenterSectorBuildingGraphProvider(
            Path(args.building_shp),
            base_dir / "GT" / "seoul_LCZ.tif",
            story_col=args.story_col,
            patch_size=args.patch_size,
            graph_context_m=args.graph_context_m,
            max_nodes=args.max_nodes,
            knn=args.knn,
            radius_m=args.radius_m,
            edge_mode=args.edge_mode,
        )
        for split_name in ["train", "val", "test"]:
            build_graph_cache_for_split(split_dir / f"{split_name}_samples.npz", graph_provider, cache_paths[split_name], split_name)
    else:
        print("Using existing graph cache:", graph_cache_dir)

    train_ds = CachedCenterGraphDataset(rs_stack, cache_paths["train"], args.patch_size)
    val_ds = CachedCenterGraphDataset(rs_stack, cache_paths["val"], args.patch_size)
    test_ds = CachedCenterGraphDataset(rs_stack, cache_paths["test"], args.patch_size)
    print(f"Dataset: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=center_graph_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=center_graph_collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=center_graph_collate)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = RSGraphLCZModelV4(
        rs_stack.shape[0], len(LCZ_CLASSES), args.patch_size,
        graph_hidden=args.graph_hidden,
        graph_out=args.graph_out,
        graph_layers=args.graph_layers,
        graph_heads=args.graph_heads,
        dropout=args.dropout,
    ).to(device)
    num_params = count_parameters(model)
    print("Device:", device)
    print("Trainable parameters:", f"{num_params:,}")

    train_labels = np.load(split_dir / "train_samples.npz")["labels"]
    weight = compute_class_weights(train_labels, len(LCZ_CLASSES)).to(device) if args.class_weight else None
    if args.class_weight:
        print("Class weights:", weight.detach().cpu().numpy())
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_oa, best_epoch, patience = -1.0, -1, 0
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
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            n += y.size(0)
        train_loss = total_loss / max(n, 1)
        val = evaluate(model, val_loader, device, len(LCZ_CLASSES))
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val['loss']:.4f} | val_OA={val['oa']:.4f} | val_macroF1={val['macro_f1']:.4f}")
        logs.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val["loss"], "val_oa": val["oa"], "val_macro_f1": val["macro_f1"], "val_weighted_f1": val["weighted_f1"]})
        if val["oa"] > best_val_oa:
            best_val_oa, best_epoch, patience = val["oa"], epoch, 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "rs_channels": int(rs_stack.shape[0]),
                "num_classes": len(LCZ_CLASSES),
                "patch_size": args.patch_size,
                "lcz_classes": LCZ_CLASSES,
                "params": num_params,
                "val_oa": best_val_oa,
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
                "model_version": "v4_center_sector_gated_fusion",
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
        "model": "CNN_RS_BuildingGraphEncoder_V4_CenterAwareGatedFusion",
        "best_epoch": int(best_epoch),
        "best_val_oa": float(best_val_oa),
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
        "graph_hidden": int(args.graph_hidden),
        "graph_out": int(args.graph_out),
        "graph_layers": int(args.graph_layers),
        "graph_heads": int(args.graph_heads),
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
    p.add_argument("--rs_norm_dir", type=str, default=DEFAULT_RS_NORM)
    p.add_argument("--building_shp", type=str, default=DEFAULT_BUILDING_SHP)
    p.add_argument("--story_col", type=str, default="A10")
    p.add_argument("--split_dir", type=str, default=None)
    p.add_argument("--exp_name", type=str, default="cnn_rs_building_graph_encoder_v4")
    p.add_argument("--patch_size", type=int, default=33)
    p.add_argument("--graph_context_m", type=float, default=500.0)
    p.add_argument("--edge_mode", type=str, default="hybrid", choices=["knn", "radius", "hybrid"])
    p.add_argument("--radius_m", type=float, default=120.0)
    p.add_argument("--max_nodes", type=int, default=224)
    p.add_argument("--knn", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--graph_hidden", type=int, default=96)
    p.add_argument("--graph_out", type=int, default=192)
    p.add_argument("--graph_layers", type=int, default=3)
    p.add_argument("--graph_heads", type=int, default=4)
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
