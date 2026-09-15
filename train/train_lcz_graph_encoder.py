#!/usr/bin/env python3
"""
CNN_RS_BuildingGraphEncoder
- RS branch: 10m RS patch, e.g. b1~b8_norm.mat
- Building graph branch: each building polygon in the 33x33 10m patch becomes a node
- Node feature: area, story(A10), relative centroid position, distance, bbox geometry, aspect ratio, existence flag
- Edge/adjacency: kNN among building centroids inside each patch
- Fusion: concat(RS feature, graph embedding) -> LCZ class

No torch_geometric dependency. Uses dense padded adjacency per batch.

Recommended run:

# 처음 실행
cd /mnt/disk1/workspace_jym/LCZ

CUDA_VISIBLE_DEVICES=2 python train/train_lcz_graph_encoder.py \
  --base_dir /mnt/disk1/workspace_jym/LCZ/data \
  --work_dir /mnt/disk1/workspace_jym/LCZ/work_dirs \
  --rs_norm_dir /mnt/disk1/workspace_jym/LCZ/data/Satellite/processed/norm \
  --building_shp /mnt/disk1/workspace_jym/LCZ/data/Building/AL_11_D010_20200502/AL_11_D010_20200502.shp \
  --story_col A10 \
  --split_dir /mnt/disk1/workspace_jym/LCZ/work_dirs/cnn_rs_baseline/splits \
  --exp_name cnn_rs_building_graph_encoder \
  --patch_size 33 \
  --batch_size 128 \
  --epochs 150 \
  --patience 25 \
  --max_nodes 128 \
  --knn 6 \
  --class_weight \
  --rebuild_graph_cache

# 이후 다시 실행
CUDA_VISIBLE_DEVICES=2 python train/train_lcz_graph_encoder.py \
  --base_dir /mnt/disk1/workspace_jym/LCZ/data \
  --work_dir /mnt/disk1/workspace_jym/LCZ/work_dirs \
  --rs_norm_dir /mnt/disk1/workspace_jym/LCZ/data/Satellite/processed/norm \
  --building_shp /mnt/disk1/workspace_jym/LCZ/data/Building/AL_11_D010_20200502/AL_11_D010_20200502.shp \
  --story_col A10 \
  --split_dir /mnt/disk1/workspace_jym/LCZ/work_dirs/cnn_rs_baseline/splits \
  --exp_name cnn_rs_building_graph_encoder \
  --patch_size 33 \
  --batch_size 128 \
  --epochs 150 \
  --patience 25 \
  --max_nodes 128 \
  --knn 6 \
  --class_weight

"""

import argparse
import json
import random
from pathlib import Path

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

DEFAULT_BASE_DIR = "/mnt/disk1/workspace_jym/LCZ/data"
DEFAULT_WORK_DIR = "/mnt/disk1/workspace_jym/LCZ/work_dirs"
DEFAULT_BUILDING_SHP = "/mnt/disk1/workspace_jym/LCZ/data/Building/AL_11_D010_20200502/AL_11_D010_20200502.shp"

LCZ_CLASSES = [1, 2, 3, 4, 5, 6, 8, 101, 102, 104, 107]
LCZ_CLASS_NAMES = {
    1: "LCZ1", 2: "LCZ2", 3: "LCZ3", 4: "LCZ4", 5: "LCZ5", 6: "LCZ6", 8: "LCZ8",
    101: "LCZA", 102: "LCZB", 104: "LCZD", 107: "LCZG",
}


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


class BuildingGraphProvider:
    def __init__(self, shp_path: Path, gt_path: Path, story_col: str = "A10", patch_size: int = 33, max_nodes: int = 128, knn: int = 6):
        self.shp_path = shp_path
        self.story_col = story_col
        self.patch_size = patch_size
        self.max_nodes = max_nodes
        self.knn = knn
        self.half_size_m = patch_size * 10.0 / 2.0
        self.search_radius = self.half_size_m * np.sqrt(2.0)

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
        # clip to GT bounds expanded by one patch radius
        b = self.gt_bounds
        pad = self.half_size_m + 20.0
        gdf = gdf.cx[b.left - pad:b.right + pad, b.bottom - pad:b.top + pad].copy()
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
        gdf["geometry"] = gdf.geometry.buffer(0)
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
        gdf["area_m2"] = gdf.geometry.area.astype(np.float32)
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
        self.story = gdf["story"].to_numpy(np.float32)
        self.bbox_w = gdf["bbox_w"].to_numpy(np.float32)
        self.bbox_h = gdf["bbox_h"].to_numpy(np.float32)
        self.xy = np.stack([self.cx, self.cy], axis=1)
        self.tree = cKDTree(self.xy)

        self.area_scale = float(np.percentile(np.log1p(self.area), 99))
        self.story_scale = max(float(np.percentile(self.story[self.story > 0], 99)) if np.any(self.story > 0) else 1.0, 1.0)
        self.bbox_scale = float(np.percentile(np.log1p(np.maximum(self.bbox_w, self.bbox_h)), 99))
        self.area_scale = max(self.area_scale, 1e-6)
        self.bbox_scale = max(self.bbox_scale, 1e-6)
        print(f"Scales: log_area={self.area_scale:.4f}, story={self.story_scale:.4f}, log_bbox={self.bbox_scale:.4f}")

    def cell_center_xy(self, row50: int, col50: int):
        # 50m raster Area pixel center
        x = self.gt_transform.c + col50 * self.gt_transform.a + self.gt_transform.a / 2.0
        y = self.gt_transform.f + row50 * self.gt_transform.e + self.gt_transform.e / 2.0
        return float(x), float(y)

    def get_graph(self, row50: int, col50: int):
        center_x, center_y = self.cell_center_xy(row50, col50)
        candidate = self.tree.query_ball_point([center_x, center_y], r=self.search_radius)
        if len(candidate) == 0:
            return self.empty_graph()
        idx = np.asarray(candidate, dtype=np.int64)
        dx = self.cx[idx] - center_x
        dy = self.cy[idx] - center_y
        inside = (np.abs(dx) <= self.half_size_m) & (np.abs(dy) <= self.half_size_m)
        idx = idx[inside]
        dx = dx[inside]
        dy = dy[inside]
        if len(idx) == 0:
            return self.empty_graph()

        dist = np.sqrt(dx * dx + dy * dy)
        # 가까운 건물 max_nodes개만 사용
        if len(idx) > self.max_nodes:
            order = np.argsort(dist)[:self.max_nodes]
            idx, dx, dy, dist = idx[order], dx[order], dy[order], dist[order]

        area_norm = np.clip(np.log1p(self.area[idx]) / self.area_scale, 0.0, 1.0)
        story_norm = np.clip(self.story[idx] / self.story_scale, 0.0, 1.0)
        rel_x = np.clip(dx / self.half_size_m, -1.0, 1.0)
        rel_y = np.clip(dy / self.half_size_m, -1.0, 1.0)
        dist_norm = np.clip(dist / self.search_radius, 0.0, 1.0)
        bbox_w_norm = np.clip(np.log1p(self.bbox_w[idx]) / self.bbox_scale, 0.0, 1.0)
        bbox_h_norm = np.clip(np.log1p(self.bbox_h[idx]) / self.bbox_scale, 0.0, 1.0)
        aspect = self.bbox_w[idx] / np.maximum(self.bbox_h[idx], 1e-6)
        aspect_norm = np.clip(np.log1p(aspect) / np.log1p(10.0), 0.0, 1.0)
        has_building = np.ones_like(area_norm, dtype=np.float32)

        node = np.stack([area_norm, story_norm, rel_x, rel_y, dist_norm, bbox_w_norm, bbox_h_norm, aspect_norm, has_building], axis=1).astype(np.float32)
        adj = self.make_knn_adj(np.stack([dx, dy], axis=1).astype(np.float32))
        mask = np.ones((node.shape[0],), dtype=np.float32)
        return node, adj, mask

    def empty_graph(self):
        node = np.zeros((1, 9), dtype=np.float32)
        adj = np.ones((1, 1), dtype=np.float32)
        mask = np.ones((1,), dtype=np.float32)
        return node, adj, mask

    def make_knn_adj(self, xy_rel: np.ndarray):
        n = xy_rel.shape[0]
        adj = np.zeros((n, n), dtype=np.float32)
        if n == 1:
            adj[0, 0] = 1.0
            return adj
        d = np.sqrt(((xy_rel[:, None, :] - xy_rel[None, :, :]) ** 2).sum(axis=2))
        k = min(self.knn, n - 1)
        for i in range(n):
            nn_idx = np.argsort(d[i])[1:k + 1]
            weights = np.exp(-d[i, nn_idx] / max(self.half_size_m, 1e-6)).astype(np.float32)
            adj[i, nn_idx] = weights
            adj[nn_idx, i] = np.maximum(adj[nn_idx, i], weights)
        np.fill_diagonal(adj, 1.0)
        return adj


class LCZGraphDataset(Dataset):
    def __init__(self, rs_stack: np.ndarray, sample_npz: Path, graph_provider: BuildingGraphProvider, patch_size: int = 33):
        self.rs_stack = rs_stack
        self.patch_size = patch_size
        self.radius = patch_size // 2
        self.graph_provider = graph_provider
        data = np.load(sample_npz)
        self.rows50 = data["rows"].astype(np.int64)
        self.cols50 = data["cols"].astype(np.int64)
        self.labels = data["labels"].astype(np.int64)
        self.rs_pad = np.pad(rs_stack, ((0, 0), (self.radius, self.radius), (self.radius, self.radius)), mode="constant")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        r50 = int(self.rows50[idx])
        c50 = int(self.cols50[idx])
        r10 = r50 * 5 + 2 + self.radius
        c10 = c50 * 5 + 2 + self.radius
        patch = self.rs_pad[:, r10 - self.radius:r10 + self.radius + 1, c10 - self.radius:c10 + self.radius + 1]
        node, adj, mask = self.graph_provider.get_graph(r50, c50)
        return {
            "rs": torch.from_numpy(patch.copy()).float(),
            "node": torch.from_numpy(node).float(),
            "adj": torch.from_numpy(adj).float(),
            "mask": torch.from_numpy(mask).float(),
            "label": torch.tensor(self.labels[idx]).long(),
        }



# ============================================================
# Graph cache
# ============================================================

def build_graph_cache_for_split(sample_npz: Path,
                                graph_provider: BuildingGraphProvider,
                                cache_path: Path,
                                split_name: str):
    """
    Precompute building graphs for one split and save them to a .pt file.

    This removes the expensive per-epoch KDTree query + kNN adjacency construction
    from Dataset.__getitem__(). The first cache generation can still take time, but
    all following epochs/runs reuse the cached tensors directly.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(sample_npz)
    rows = data["rows"].astype(np.int64)
    cols = data["cols"].astype(np.int64)
    labels = data["labels"].astype(np.int64)
    labels_raw = data["labels_raw"].astype(np.int64) if "labels_raw" in data.files else None

    nodes = []
    adjs = []
    masks = []

    print(f"\n========== Build graph cache: {split_name} ==========")
    print("sample_npz:", sample_npz)
    print("cache_path:", cache_path)
    print("num_samples:", len(labels))

    node_counts = []

    for i, (r50, c50) in enumerate(zip(rows, cols), start=1):
        node, adj, mask = graph_provider.get_graph(int(r50), int(c50))

        # Store in compact CPU tensors. They will be cast to float32 in collate.
        nodes.append(torch.from_numpy(node.astype(np.float16)))
        adjs.append(torch.from_numpy(adj.astype(np.float16)))
        masks.append(torch.from_numpy(mask.astype(np.uint8)))
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
        "masks": masks,
        "meta": {
            "split_name": split_name,
            "num_samples": int(len(labels)),
            "patch_size": int(graph_provider.patch_size),
            "max_nodes": int(graph_provider.max_nodes),
            "knn": int(graph_provider.knn),
            "node_feature_dim": 9,
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


class CachedLCZGraphDataset(Dataset):
    """
    Dataset that reads precomputed graph tensors from .pt cache.

    RS patches are still extracted from the normalized raster stack on the fly,
    but graph node features and adjacency matrices are no longer rebuilt every epoch.
    """
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
        self.masks = data["masks"]
        self.meta = data.get("meta", {})

        if not (len(self.rows50) == len(self.cols50) == len(self.labels) == len(self.nodes) == len(self.adjs) == len(self.masks)):
            raise RuntimeError("Cache length mismatch among rows/cols/labels/nodes/adjs/masks")

        print("  samples:", len(self.labels))
        print("  meta:", self.meta)

        self.rs_pad = np.pad(
            rs_stack,
            ((0, 0), (self.radius, self.radius), (self.radius, self.radius)),
            mode="constant",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        r50 = int(self.rows50[idx])
        c50 = int(self.cols50[idx])

        r10 = r50 * 5 + 2 + self.radius
        c10 = c50 * 5 + 2 + self.radius

        patch = self.rs_pad[
            :,
            r10 - self.radius:r10 + self.radius + 1,
            c10 - self.radius:c10 + self.radius + 1,
        ]

        return {
            "rs": torch.from_numpy(patch.copy()).float(),
            "node": self.nodes[idx].float(),
            "adj": self.adjs[idx].float(),
            "mask": self.masks[idx].float(),
            "label": self.labels[idx].long(),
        }

def graph_collate(batch):
    rs = torch.stack([b["rs"] for b in batch], dim=0)
    y = torch.stack([b["label"] for b in batch], dim=0)
    max_n = max(b["node"].shape[0] for b in batch)
    feat_dim = batch[0]["node"].shape[1]
    nodes = torch.zeros(len(batch), max_n, feat_dim, dtype=torch.float32)
    adjs = torch.zeros(len(batch), max_n, max_n, dtype=torch.float32)
    masks = torch.zeros(len(batch), max_n, dtype=torch.float32)
    for i, b in enumerate(batch):
        n = b["node"].shape[0]
        nodes[i, :n] = b["node"]
        adjs[i, :n, :n] = b["adj"]
        masks[i, :n] = b["mask"]
    return rs, nodes, adjs, masks, y


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, out_dim: int = 256, base_channels: int = 32):
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
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(flat_dim, out_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.proj(self.encoder(x))


class DenseGraphSAGELayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.self_lin = nn.Linear(dim, dim)
        self.neigh_lin = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, h, adj, mask):
        # h: B,N,D; adj: B,N,N; mask: B,N
        adj = adj * mask[:, None, :] * mask[:, :, None]
        deg = adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        neigh = torch.bmm(adj, h) / deg
        out = self.self_lin(h) + self.neigh_lin(neigh)
        out = self.act(self.norm(out))
        out = out * mask.unsqueeze(-1)
        return out


class BuildingGraphEncoder(nn.Module):
    def __init__(self, in_dim: int = 9, hidden_dim: int = 64, out_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.input_mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True), nn.LayerNorm(hidden_dim))
        self.layers = nn.ModuleList([DenseGraphSAGELayer(hidden_dim) for _ in range(num_layers)])
        self.out_mlp = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.ReLU(inplace=True))

    def forward(self, node, adj, mask):
        h = self.input_mlp(node) * mask.unsqueeze(-1)
        for layer in self.layers:
            h = layer(h, adj, mask)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / denom
        return self.out_mlp(pooled)


class RSGraphLCZModel(nn.Module):
    def __init__(self, rs_channels: int, num_classes: int, patch_size: int = 33, graph_hidden: int = 64, graph_out: int = 128, dropout: float = 0.5):
        super().__init__()
        self.rs_encoder = ConvEncoder(rs_channels, patch_size, out_dim=256, base_channels=32)
        self.graph_encoder = BuildingGraphEncoder(in_dim=9, hidden_dim=graph_hidden, out_dim=graph_out, num_layers=3)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 + graph_out, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, rs, node, adj, mask):
        rs_feat = self.rs_encoder(rs)
        graph_feat = self.graph_encoder(node, adj, mask)
        return self.classifier(torch.cat([rs_feat, graph_feat], dim=1))


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
        for rs, node, adj, mask, y in loader:
            rs, node, adj, mask, y = rs.to(device), node.to(device), adj.to(device), mask.to(device), y.to(device)
            logits = model(rs, node, adj, mask)
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

    # ------------------------------------------------------------
    # Build/load graph cache
    # ------------------------------------------------------------
    if args.graph_cache_dir is None:
        graph_cache_dir = exp_dir / "graph_cache"
    else:
        graph_cache_dir = Path(args.graph_cache_dir)
    graph_cache_dir.mkdir(parents=True, exist_ok=True)

    cache_paths = {
        "train": graph_cache_dir / "train_graphs.pt",
        "val": graph_cache_dir / "val_graphs.pt",
        "test": graph_cache_dir / "test_graphs.pt",
    }

    need_cache = args.rebuild_graph_cache or any(not p.exists() for p in cache_paths.values())

    if need_cache:
        print("Graph cache missing or rebuild requested. Building graph cache once...")
        graph_provider = BuildingGraphProvider(
            Path(args.building_shp),
            base_dir / "GT" / "seoul_LCZ.tif",
            story_col=args.story_col,
            patch_size=args.patch_size,
            max_nodes=args.max_nodes,
            knn=args.knn,
        )
        for split_name in ["train", "val", "test"]:
            build_graph_cache_for_split(
                sample_npz=split_dir / f"{split_name}_samples.npz",
                graph_provider=graph_provider,
                cache_path=cache_paths[split_name],
                split_name=split_name,
            )
    else:
        print("Using existing graph cache:", graph_cache_dir)

    train_ds = CachedLCZGraphDataset(rs_stack, cache_paths["train"], args.patch_size)
    val_ds = CachedLCZGraphDataset(rs_stack, cache_paths["val"], args.patch_size)
    test_ds = CachedLCZGraphDataset(rs_stack, cache_paths["test"], args.patch_size)
    print(f"Dataset: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=graph_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=graph_collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=graph_collate)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = RSGraphLCZModel(rs_stack.shape[0], len(LCZ_CLASSES), args.patch_size, args.graph_hidden, args.graph_out, args.dropout).to(device)
    num_params = count_parameters(model)
    print("Device:", device)
    print("Trainable parameters:", f"{num_params:,}")

    train_labels = np.load(split_dir / "train_samples.npz")["labels"]
    if args.class_weight:
        weights = compute_class_weights(train_labels, len(LCZ_CLASSES)).to(device)
        print("Class weights:", weights.detach().cpu().numpy())
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_oa, best_epoch, patience = -1.0, -1, 0
    logs = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for rs, node, adj, mask, y in train_loader:
            rs, node, adj, mask, y = rs.to(device), node.to(device), adj.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(rs, node, adj, mask), y)
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
                "graph_hidden": args.graph_hidden,
                "graph_out": args.graph_out,
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
        "model": "CNN_RS_BuildingGraphEncoder",
        "best_epoch": int(best_epoch),
        "best_val_oa": float(best_val_oa),
        "test_oa": float(test["oa"]),
        "test_macro_f1": float(test["macro_f1"]),
        "test_weighted_f1": float(test["weighted_f1"]),
        "params": int(num_params),
        "graph_cache_dir": str(graph_cache_dir),
        "max_nodes": int(args.max_nodes),
        "knn": int(args.knn),
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
    p.add_argument("--split_dir", type=str, default=None, help="Use the same split as baseline, e.g. work_dirs/cnn_rs_baseline/splits")
    p.add_argument("--exp_name", type=str, default="cnn_rs_building_graph_encoder")
    p.add_argument("--patch_size", type=int, default=33)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--graph_hidden", type=int, default=64)
    p.add_argument("--graph_out", type=int, default=128)
    p.add_argument("--max_nodes", type=int, default=128)
    p.add_argument("--knn", type=int, default=6)
    p.add_argument("--graph_cache_dir", type=str, default=None, help="Directory for precomputed graph .pt files. Default: <exp_dir>/graph_cache")
    p.add_argument("--rebuild_graph_cache", action="store_true", help="Rebuild graph cache even if .pt cache files already exist")
    p.add_argument("--train_ratio", type=float, default=0.6)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0, help="0 is safest because geopandas/kdtree objects are large")
    p.add_argument("--class_weight", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
