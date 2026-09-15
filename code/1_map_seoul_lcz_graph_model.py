#!/usr/bin/env python3
"""
Seoul-wide LCZ mapping for CNN_RS_BuildingGraphEncoder V7/V10-style models.

Purpose
-------
Generate a full LCZ GeoTIFF map by sliding over the 50 m LCZ grid. For each
output grid cell, the script builds:
  1) an RS patch from normalized Landsat/RS bands
  2) a building graph using the same V7 graph provider
and runs the trained graph-fusion LCZ classifier.

This script is intentionally independent from training. It imports the model
classes/functions from your train_lcz_graph_encoder_v7.py or v10.py file.

Example
------

#V7
# 방법 1. 일단 전체 rectangular extent 전부 예측 [이것만 사용하도록]
CUDA_VISIBLE_DEVICES=2 python code/map_seoul_lcz_graph_model.py \
  --model_py /mnt/disk1/workspace_jym/LCZ/train/train_lcz_graph_encoder_v7.py \
  --model_version v7 \
  --checkpoint /mnt/disk1/workspace_jym/LCZ/work_dirs/graph_v7_node_edge_scale_interaction_oa/checkpoints/best_model.pt \
  --base_dir /mnt/disk1/workspace_jym/LCZ/data \
  --rs_norm_dir /mnt/disk1/workspace_jym/LCZ/data/Satellite/processed/norm \
  --building_shp /mnt/disk1/workspace_jym/LCZ/data/Building/AL_11_D010_20200502/AL_11_D010_20200502.shp \
  --story_col A10 \
  --template_tif /mnt/disk1/workspace_jym/LCZ/data/GT/seoul_LCZ.tif \
  --out_tif /mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_v7_full_extent_raw.tif \
  --out_index_tif /mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_v7_full_extent_index_raw.tif \
  --out_conf_tif /mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_v7_full_extent_conf_raw.tif \
  --patch_size 33 \
  --graph_context_m 500 \
  --global_scales_m 330 500 700 \
  --edge_mode hybrid \
  --knn 8 \
  --radius_m 120 \
  --max_nodes 192 \
  --batch_size 256 \
  --valid_mode all

"""

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling


def import_train_module(model_py: Path):
    model_py = Path(model_py)
    if not model_py.exists():
        raise FileNotFoundError(f"model_py not found: {model_py}")
    spec = importlib.util.spec_from_file_location("lcz_train_module", str(model_py))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_valid_mask(template_arr: np.ndarray,
                     nodata,
                     valid_mode: str,
                     valid_mask_tif: Optional[Path] = None) -> np.ndarray:
    """Return boolean mask on output 50 m grid."""
    if valid_mask_tif is not None:
        with rasterio.open(valid_mask_tif) as src:
            mask = src.read(1)
            if mask.shape != template_arr.shape:
                raise ValueError(
                    f"valid_mask_tif shape {mask.shape} != template_tif shape {template_arr.shape}. "
                    "Resample the mask before running this script."
                )
        return mask > 0

    if valid_mode == "all":
        return np.ones(template_arr.shape, dtype=bool)
    if valid_mode == "template_nodata":
        valid = np.isfinite(template_arr.astype(np.float32))
        if nodata is not None:
            valid &= template_arr != nodata
        return valid
    if valid_mode == "template_nonzero":
        valid = np.isfinite(template_arr.astype(np.float32))
        if nodata is not None:
            valid &= template_arr != nodata
        valid &= template_arr != 0
        return valid
    if valid_mode == "lcz_classes":
        # Useful only when template_tif contains valid LCZ labels and outside area is 0/nodata.
        lcz_classes = np.array([1, 2, 3, 4, 5, 6, 8, 101, 102, 104, 107], dtype=template_arr.dtype)
        return np.isin(template_arr, lcz_classes)
    raise ValueError(f"Unknown valid_mode: {valid_mode}")


def make_graph_batch(items: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]], device):
    """Pad variable-size graph items and move tensors to device."""
    patches, nodes_list, adjs_list, etypes_list, masks_list, globals_list = [], [], [], [], [], []
    max_n = 1
    feat_dim = None
    for patch, node, adj, etype, mask, glob in items:
        patches.append(torch.from_numpy(patch.copy()).float())
        nodes_list.append(torch.from_numpy(node.astype(np.float32)).float())
        adjs_list.append(torch.from_numpy(adj.astype(np.float32)).float())
        etypes_list.append(torch.from_numpy(etype.astype(np.int64)).long())
        masks_list.append(torch.from_numpy(mask.astype(np.float32)).float())
        globals_list.append(torch.from_numpy(glob.astype(np.float32)).float())
        max_n = max(max_n, int(node.shape[0]))
        feat_dim = int(node.shape[1])

    B = len(items)
    rs = torch.stack(patches, dim=0)
    glob = torch.stack(globals_list, dim=0)
    nodes = torch.zeros(B, max_n, feat_dim, dtype=torch.float32)
    adjs = torch.zeros(B, max_n, max_n, dtype=torch.float32)
    edge_types = torch.zeros(B, max_n, max_n, dtype=torch.long)
    masks = torch.zeros(B, max_n, dtype=torch.float32)

    for i in range(B):
        n = nodes_list[i].shape[0]
        nodes[i, :n] = nodes_list[i]
        adjs[i, :n, :n] = adjs_list[i]
        edge_types[i, :n, :n] = etypes_list[i]
        masks[i, :n] = masks_list[i]

    return (
        rs.to(device, non_blocking=True),
        nodes.to(device, non_blocking=True),
        adjs.to(device, non_blocking=True),
        edge_types.to(device, non_blocking=True),
        masks.to(device, non_blocking=True),
        glob.to(device, non_blocking=True),
    )


def instantiate_model(module, model_version: str, ckpt: dict, rs_channels: int, num_classes: int, args):
    patch_size = int(ckpt.get("patch_size", args.patch_size))
    graph_hidden = int(ckpt.get("graph_hidden", args.graph_hidden))
    graph_out = int(ckpt.get("graph_out", args.graph_out))
    graph_layers = int(ckpt.get("graph_layers", args.graph_layers))
    graph_heads = int(ckpt.get("graph_heads", args.graph_heads))

    if model_version.lower() == "v7":
        if not hasattr(module, "RSGraphLCZModelV7"):
            raise AttributeError("RSGraphLCZModelV7 not found in model_py")
        return module.RSGraphLCZModelV7(
            rs_channels=rs_channels,
            num_classes=num_classes,
            patch_size=patch_size,
            graph_hidden=graph_hidden,
            graph_out=graph_out,
            graph_layers=graph_layers,
            graph_heads=graph_heads,
            dropout=args.dropout,
        )
    if model_version.lower() == "v10":
        if not hasattr(module, "RSGraphLCZModelV10"):
            raise AttributeError("RSGraphLCZModelV10 not found in model_py")
        return module.RSGraphLCZModelV10(
            rs_channels=rs_channels,
            num_classes=num_classes,
            patch_size=patch_size,
            graph_hidden=graph_hidden,
            graph_out=graph_out,
            graph_layers=graph_layers,
            graph_heads=graph_heads,
            film_hidden=int(ckpt.get("film_hidden", args.film_hidden)),
            film_residual_init=float(ckpt.get("film_residual_init", args.film_residual_init)),
            dropout=args.dropout,
        )
    raise ValueError(f"Unsupported model_version: {model_version}")


def save_geotiff(path: Path, arr: np.ndarray, profile: dict, nodata: int, dtype: str, descriptions: Optional[List[str]] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update({
        "driver": "GTiff",
        "height": arr.shape[-2],
        "width": arr.shape[-1],
        "count": 1 if arr.ndim == 2 else arr.shape[0],
        "dtype": dtype,
        "nodata": nodata,
        "compress": "lzw",
        "predictor": 2 if np.issubdtype(np.dtype(dtype), np.floating) else 1,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "IF_SAFER",
    })
    with rasterio.open(path, "w", **out_profile) as dst:
        if arr.ndim == 2:
            dst.write(arr.astype(dtype), 1)
            if descriptions:
                dst.set_band_description(1, descriptions[0])
        else:
            dst.write(arr.astype(dtype))
            if descriptions:
                for i, desc in enumerate(descriptions, start=1):
                    dst.set_band_description(i, desc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_py", type=str, required=True, help="Path to train_lcz_graph_encoder_v7.py or train_lcz_graph_encoder_v10.py")
    parser.add_argument("--model_version", type=str, required=True, choices=["v7", "v10"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--rs_norm_dir", type=str, required=True)
    parser.add_argument("--building_shp", type=str, required=True)
    parser.add_argument("--story_col", type=str, default="A10")
    parser.add_argument("--template_tif", type=str, required=True, help="50 m grid template GeoTIFF, e.g., seoul_LCZ.tif")
    parser.add_argument("--valid_mask_tif", type=str, default=None, help="Optional 50 m mask raster. >0 pixels are predicted.")
    parser.add_argument("--valid_mode", type=str, default="template_nodata", choices=["all", "template_nodata", "template_nonzero", "lcz_classes"])
    parser.add_argument("--out_tif", type=str, required=True)
    parser.add_argument("--out_index_tif", type=str, default=None, help="Optional class-index map: 0..C-1, nodata=-1")
    parser.add_argument("--out_conf_tif", type=str, default=None, help="Optional max probability/confidence map scaled to 0..10000 uint16")
    parser.add_argument("--save_probs", action="store_true", help="Save C-band probability GeoTIFF. Can be large.")
    parser.add_argument("--out_prob_tif", type=str, default=None)

    # Must match training/model settings.
    parser.add_argument("--patch_size", type=int, default=33)
    parser.add_argument("--graph_context_m", type=float, default=500.0)
    parser.add_argument("--global_scales_m", type=float, nargs="+", default=[330.0, 500.0, 700.0])
    parser.add_argument("--edge_mode", type=str, default="hybrid", choices=["knn", "radius", "hybrid"])
    parser.add_argument("--knn", type=int, default=8)
    parser.add_argument("--radius_m", type=float, default=120.0)
    parser.add_argument("--max_nodes", type=int, default=192)
    parser.add_argument("--salient_near_ratio", type=float, default=0.45)
    parser.add_argument("--salient_large_ratio", type=float, default=0.20)
    parser.add_argument("--salient_high_ratio", type=float, default=0.20)
    parser.add_argument("--num_sectors", type=int, default=8)

    # Model construction defaults; checkpoint values override most of these.
    parser.add_argument("--graph_hidden", type=int, default=96)
    parser.add_argument("--graph_out", type=int, default=192)
    parser.add_argument("--graph_layers", type=int, default=3)
    parser.add_argument("--graph_heads", type=int, default=4)
    parser.add_argument("--film_hidden", type=int, default=128)
    parser.add_argument("--film_residual_init", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.35)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0, help="Reserved for future use; graph building is currently in-process.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--progress_every", type=int, default=5000)
    parser.add_argument("--nodata", type=int, default=0, help="Nodata value for output LCZ-code map")
    args = parser.parse_args()

    model_py = Path(args.model_py)
    checkpoint = Path(args.checkpoint)
    base_dir = Path(args.base_dir)
    template_tif = Path(args.template_tif)
    out_tif = Path(args.out_tif)

    module = import_train_module(model_py)
    lcz_classes = list(getattr(module, "LCZ_CLASSES", [1, 2, 3, 4, 5, 6, 8, 101, 102, 104, 107]))
    num_classes = len(lcz_classes)

    print("========== Load template ==========")
    with rasterio.open(template_tif) as src:
        template = src.read(1)
        template_profile = src.profile.copy()
        template_nodata = src.nodata
        template_shape = template.shape
        print("template:", template_tif)
        print("shape:", template_shape)
        print("crs:", src.crs)
        print("transform:", src.transform)
        print("nodata:", template_nodata)

    valid_mask = build_valid_mask(
        template,
        template_nodata,
        args.valid_mode,
        Path(args.valid_mask_tif) if args.valid_mask_tif else None,
    )
    total_cells = int(valid_mask.sum())
    print(f"valid_mode={args.valid_mode}, cells_to_predict={total_cells:,} / {valid_mask.size:,}")

    print("========== Load RS stack ==========")
    rs_stack, _ = module.load_rs_stack(Path(args.rs_norm_dir))
    rs_channels, h10, w10 = rs_stack.shape
    print("rs_stack:", rs_stack.shape)

    print("========== Build graph provider ==========")
    graph_provider = module.BuildingGraphProviderV7(
        Path(args.building_shp),
        template_tif,
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

    print("========== Load model ==========")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = torch.load(checkpoint, map_location=device)
    model = instantiate_model(module, args.model_version, ckpt, rs_channels, num_classes, args).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print("device:", device)
    print("checkpoint:", checkpoint)
    print("model_version:", args.model_version)
    print("classes:", lcz_classes)

    # Pad RS stack once. This matches CachedLCZGraphDatasetV7.__getitem__.
    radius = args.patch_size // 2
    rs_pad = np.pad(rs_stack, ((0, 0), (radius, radius), (radius, radius)), mode="constant")

    pred_code = np.full(template_shape, args.nodata, dtype=np.int16)
    pred_index = np.full(template_shape, -1, dtype=np.int16)
    pred_conf = np.zeros(template_shape, dtype=np.uint16) if args.out_conf_tif else None
    prob_cube = None
    if args.save_probs:
        prob_cube = np.zeros((num_classes, template_shape[0], template_shape[1]), dtype=np.float32)

    rows, cols = np.where(valid_mask)
    coords = list(zip(rows.tolist(), cols.tolist()))
    batch_items = []
    batch_coords = []
    t0 = time.time()
    done = 0

    with torch.no_grad():
        for row50, col50 in coords:
            # Same 50 m -> 10 m center logic as training dataset.
            r10 = int(row50) * 5 + 2 + radius
            c10 = int(col50) * 5 + 2 + radius
            patch = rs_pad[:, r10 - radius:r10 + radius + 1, c10 - radius:c10 + radius + 1]
            if patch.shape != (rs_channels, args.patch_size, args.patch_size):
                # Should not happen with padding, but keep safe.
                continue

            node, adj, etype, mask, glob = graph_provider.get_graph(int(row50), int(col50))
            batch_items.append((patch, node, adj, etype, mask, glob))
            batch_coords.append((int(row50), int(col50)))

            if len(batch_items) >= args.batch_size:
                rs, node_t, adj_t, etype_t, mask_t, glob_t = make_graph_batch(batch_items, device)
                logits = model(rs, node_t, adj_t, etype_t, mask_t, glob_t)
                probs = torch.softmax(logits, dim=1)
                conf, pred = probs.max(dim=1)
                pred_np = pred.detach().cpu().numpy()
                conf_np = conf.detach().cpu().numpy()
                probs_np = probs.detach().cpu().numpy() if args.save_probs else None
                for k, (rr, cc) in enumerate(batch_coords):
                    pi = int(pred_np[k])
                    pred_index[rr, cc] = pi
                    pred_code[rr, cc] = int(lcz_classes[pi])
                    if pred_conf is not None:
                        pred_conf[rr, cc] = int(np.clip(round(conf_np[k] * 10000), 0, 10000))
                    if prob_cube is not None:
                        prob_cube[:, rr, cc] = probs_np[k]
                done += len(batch_items)
                if done % args.progress_every < args.batch_size:
                    elapsed = time.time() - t0
                    speed = done / max(elapsed, 1e-6)
                    remain = (total_cells - done) / max(speed, 1e-6)
                    print(f"predicted {done:,}/{total_cells:,} cells | {speed:.1f} cells/s | remain {remain/60:.1f} min")
                batch_items.clear()
                batch_coords.clear()

        if batch_items:
            rs, node_t, adj_t, etype_t, mask_t, glob_t = make_graph_batch(batch_items, device)
            logits = model(rs, node_t, adj_t, etype_t, mask_t, glob_t)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            pred_np = pred.detach().cpu().numpy()
            conf_np = conf.detach().cpu().numpy()
            probs_np = probs.detach().cpu().numpy() if args.save_probs else None
            for k, (rr, cc) in enumerate(batch_coords):
                pi = int(pred_np[k])
                pred_index[rr, cc] = pi
                pred_code[rr, cc] = int(lcz_classes[pi])
                if pred_conf is not None:
                    pred_conf[rr, cc] = int(np.clip(round(conf_np[k] * 10000), 0, 10000))
                if prob_cube is not None:
                    prob_cube[:, rr, cc] = probs_np[k]
            done += len(batch_items)

    print(f"Done prediction: {done:,} cells in {(time.time() - t0)/60:.1f} min")

    print("========== Save outputs ==========")
    save_geotiff(out_tif, pred_code, template_profile, nodata=args.nodata, dtype="int16", descriptions=["LCZ class code"])
    print("saved:", out_tif)

    if args.out_index_tif:
        save_geotiff(Path(args.out_index_tif), pred_index, template_profile, nodata=-1, dtype="int16", descriptions=["LCZ class index"])
        print("saved:", args.out_index_tif)

    if args.out_conf_tif and pred_conf is not None:
        save_geotiff(Path(args.out_conf_tif), pred_conf, template_profile, nodata=0, dtype="uint16", descriptions=["Max class probability x10000"])
        print("saved:", args.out_conf_tif)

    if args.save_probs:
        prob_path = Path(args.out_prob_tif) if args.out_prob_tif else out_tif.with_name(out_tif.stem + "_probs.tif")
        profile = template_profile.copy()
        profile.update({
            "driver": "GTiff",
            "height": template_shape[0],
            "width": template_shape[1],
            "count": num_classes,
            "dtype": "float32",
            "nodata": 0.0,
            "compress": "lzw",
            "predictor": 2,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "BIGTIFF": "IF_SAFER",
        })
        with rasterio.open(prob_path, "w", **profile) as dst:
            dst.write(prob_cube.astype(np.float32))
            for i, cls in enumerate(lcz_classes, start=1):
                dst.set_band_description(i, f"P(LCZ {cls})")
        print("saved:", prob_path)

    meta = {
        "model_version": args.model_version,
        "model_py": str(model_py),
        "checkpoint": str(checkpoint),
        "template_tif": str(template_tif),
        "rs_norm_dir": args.rs_norm_dir,
        "building_shp": args.building_shp,
        "valid_mode": args.valid_mode,
        "valid_mask_tif": args.valid_mask_tif,
        "lcz_classes": lcz_classes,
        "output_tif": str(out_tif),
        "num_predicted_cells": int(done),
        "patch_size": int(args.patch_size),
        "graph_context_m": float(args.graph_context_m),
        "global_scales_m": [float(x) for x in args.global_scales_m],
        "edge_mode": args.edge_mode,
        "knn": int(args.knn),
        "radius_m": float(args.radius_m),
        "max_nodes": int(args.max_nodes),
    }
    meta_path = out_tif.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("saved:", meta_path)


if __name__ == "__main__":
    main()
