#!/usr/bin/env python3
"""
Create a Seoul-wide LCZ map with a model trained by train_lcz_rs_cnn.py.

Both experiments from the training script are supported without separate model
branches:
  A. RS-only baseline: 8-channel norm directory
  B. RS + rasterized building baseline: 10-channel norm_rs_building directory

The checkpoint and normalized feature directory determine the variant. Feature
files are loaded through train_lcz_rs_cnn.py so their ordering exactly matches
training (notably, b10_norm.mat sorts before b1_norm.mat).

Examples
--------
# A. RS-only baseline
CUDA_VISIBLE_DEVICES=1 python code/1_map_seoul_lcz_rs_cnn.py \
  --model_py /mnt/disk1/workspace_jym/LCZ/train/train_lcz_rs_cnn.py \
  --checkpoint /mnt/disk1/workspace_jym/LCZ/work_dirs/cnn_rs_baseline/checkpoints/best_model.pt \
  --norm_dir /mnt/disk1/workspace_jym/LCZ/data/Satellite/processed/norm \
  --template_tif /mnt/disk1/workspace_jym/LCZ/data/GT/seoul_LCZ.tif \
  --out_tif /mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_cnn_rs_full_extent_raw.tif \
  --out_index_tif /mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_cnn_rs_full_extent_index_raw.tif \
  --out_conf_tif /mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_cnn_rs_full_extent_conf_raw.tif \
  --batch_size 256 --valid_mode all

# B. RS + rasterized building baseline
CUDA_VISIBLE_DEVICES=2 python code/1_map_seoul_lcz_rs_cnn.py \
  --model_py /mnt/disk1/workspace_jym/LCZ/train/train_lcz_rs_cnn.py \
  --checkpoint /mnt/disk1/workspace_jym/LCZ/work_dirs/cnn_rs_raster_building/checkpoints/best_model.pt \
  --norm_dir /mnt/disk1/workspace_jym/LCZ/data/Satellite/processed/norm_rs_building \
  --template_tif /mnt/disk1/workspace_jym/LCZ/data/GT/seoul_LCZ.tif \
  --out_tif /mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_rs_raster_building_full_extent_raw.tif \
  --out_index_tif /mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_rs_raster_building_full_extent_index_raw.tif \
  --out_conf_tif /mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_rs_raster_building_full_extent_conf_raw.tif \
  --batch_size 256 --valid_mode all
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import rasterio
import torch


DEFAULT_LCZ_CLASSES = [1, 2, 3, 4, 5, 6, 8, 101, 102, 104, 107]


def import_train_module(model_py: Path):
    if not model_py.exists():
        raise FileNotFoundError(f"model_py not found: {model_py}")
    spec = importlib.util.spec_from_file_location("lcz_rs_cnn_train_module", str(model_py))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import training module: {model_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_valid_mask(template: np.ndarray,
                     nodata,
                     valid_mode: str,
                     valid_mask_tif: Optional[Path],
                     template_profile: dict) -> np.ndarray:
    if valid_mask_tif is not None:
        with rasterio.open(valid_mask_tif) as src:
            mask = src.read(1)
            if mask.shape != template.shape:
                raise ValueError(
                    f"valid_mask_tif shape {mask.shape} != template shape {template.shape}"
                )
            if src.crs != template_profile["crs"] or src.transform != template_profile["transform"]:
                raise ValueError("valid_mask_tif must have the same CRS and transform as template_tif")
        return mask > 0

    finite = np.isfinite(template.astype(np.float32))
    if valid_mode == "all":
        return np.ones(template.shape, dtype=bool)
    if valid_mode == "template_nodata":
        return finite & (template != nodata) if nodata is not None else finite
    if valid_mode == "template_nonzero":
        valid = finite & (template != 0)
        return valid & (template != nodata) if nodata is not None else valid
    if valid_mode == "lcz_classes":
        return np.isin(template, DEFAULT_LCZ_CLASSES)
    raise ValueError(f"Unknown valid_mode: {valid_mode}")


def validate_inputs(feature_stack: np.ndarray,
                    band_paths: List[Path],
                    ckpt: dict,
                    template_shape,
                    patch_size: int) -> None:
    if feature_stack.ndim != 3:
        raise ValueError(f"Expected C x H x W feature stack, got {feature_stack.shape}")

    expected_channels = int(ckpt.get("in_channels", feature_stack.shape[0]))
    if feature_stack.shape[0] != expected_channels:
        raise ValueError(
            f"Checkpoint expects {expected_channels} channels, but norm_dir contains "
            f"{feature_stack.shape[0]} channels: {[p.name for p in band_paths]}"
        )

    expected_shape = (int(template_shape[0]) * 5, int(template_shape[1]) * 5)
    if tuple(feature_stack.shape[1:]) != expected_shape:
        raise ValueError(
            f"Feature shape {feature_stack.shape[1:]} does not match the training assumption "
            f"of 5x the 50 m template shape {template_shape}; expected {expected_shape}"
        )

    if patch_size <= 0 or patch_size % 2 == 0:
        raise ValueError(f"patch_size must be a positive odd integer, got {patch_size}")

    checkpoint_bands = [Path(p).name for p in ckpt.get("band_files", [])]
    loaded_bands = [Path(p).name for p in band_paths]
    if checkpoint_bands and checkpoint_bands != loaded_bands:
        raise ValueError(
            "Band order differs from training checkpoint.\n"
            f"checkpoint: {checkpoint_bands}\n"
            f"norm_dir:   {loaded_bands}"
        )


def save_geotiff(path: Path,
                 arr: np.ndarray,
                 profile: dict,
                 nodata,
                 dtype: str,
                 descriptions: Optional[List[str]] = None) -> None:
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
                for band_index, description in enumerate(descriptions, start=1):
                    dst.set_band_description(band_index, description)


def write_batch_predictions(model,
                            patches,
                            coords,
                            device,
                            lcz_classes,
                            pred_code,
                            pred_index,
                            pred_conf,
                            prob_cube) -> None:
    x = torch.from_numpy(np.stack(patches, axis=0)).float().to(device, non_blocking=True)
    probs = torch.softmax(model(x), dim=1)
    conf, pred = probs.max(dim=1)
    pred_np = pred.cpu().numpy()
    conf_np = conf.cpu().numpy()
    probs_np = probs.cpu().numpy() if prob_cube is not None else None

    for i, (row, col) in enumerate(coords):
        class_index = int(pred_np[i])
        pred_index[row, col] = class_index
        pred_code[row, col] = int(lcz_classes[class_index])
        if pred_conf is not None:
            pred_conf[row, col] = int(np.clip(round(float(conf_np[i]) * 10000), 0, 10000))
        if prob_cube is not None:
            prob_cube[:, row, col] = probs_np[i]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Map Seoul LCZ with RS-only or RS+rasterized-building LCZCNN checkpoints."
    )
    parser.add_argument("--model_py", type=str, required=True, help="Path to train_lcz_rs_cnn.py")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--norm_dir", type=str, required=True, help="Training norm_dir for experiment A or B")
    parser.add_argument("--template_tif", type=str, required=True, help="50 m output grid template")
    parser.add_argument("--valid_mask_tif", type=str, default=None, help="Optional aligned mask; values > 0 are mapped")
    parser.add_argument(
        "--valid_mode",
        choices=["all", "template_nodata", "template_nonzero", "lcz_classes"],
        default="template_nodata",
    )
    parser.add_argument("--out_tif", type=str, required=True, help="Output LCZ class-code GeoTIFF")
    parser.add_argument("--out_index_tif", type=str, default=None, help="Optional class-index GeoTIFF")
    parser.add_argument("--out_conf_tif", type=str, default=None, help="Optional confidence x10000 GeoTIFF")
    parser.add_argument("--save_probs", action="store_true", help="Save all class probabilities as a multiband GeoTIFF")
    parser.add_argument("--out_prob_tif", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5, help="Only affects construction; dropout is disabled at inference")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--progress_every", type=int, default=5000)
    parser.add_argument("--max_cells", type=int, default=None, help="Optional smoke-test limit; remaining cells stay nodata")
    parser.add_argument("--nodata", type=int, default=0, help="Nodata for the LCZ class-code output")
    return parser.parse_args()


def main():
    args = parse_args()
    model_py = Path(args.model_py)
    checkpoint_path = Path(args.checkpoint)
    norm_dir = Path(args.norm_dir)
    template_tif = Path(args.template_tif)
    out_tif = Path(args.out_tif)

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be greater than zero")
    if args.max_cells is not None and args.max_cells <= 0:
        raise ValueError("--max_cells must be greater than zero")

    module = import_train_module(model_py)
    if not hasattr(module, "load_feature_stack") or not hasattr(module, "LCZCNN"):
        raise AttributeError("model_py must provide load_feature_stack() and LCZCNN")

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
        template_profile,
    )
    rows, cols = np.where(valid_mask)
    if args.max_cells is not None:
        rows = rows[:args.max_cells]
        cols = cols[:args.max_cells]
    total_cells = len(rows)
    print(f"valid_mode={args.valid_mode}, cells_to_predict={total_cells:,}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = load_checkpoint(checkpoint_path, device)
    patch_size = int(checkpoint.get("patch_size", 33))
    lcz_classes = [int(v) for v in checkpoint.get(
        "lcz_classes", getattr(module, "LCZ_CLASSES", DEFAULT_LCZ_CLASSES)
    )]
    num_classes = int(checkpoint.get("num_classes", len(lcz_classes)))
    if num_classes != len(lcz_classes):
        raise ValueError(f"num_classes={num_classes} but lcz_classes has {len(lcz_classes)} entries")

    print("========== Load feature stack ==========")
    feature_stack, band_paths = module.load_feature_stack(norm_dir)
    band_paths = [Path(p) for p in band_paths]
    validate_inputs(feature_stack, band_paths, checkpoint, template_shape, patch_size)
    in_channels = int(feature_stack.shape[0])
    model_variant = "rs_only" if in_channels == 8 else "rs_raster_building"

    print("========== Load model ==========")
    model = module.LCZCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        patch_size=patch_size,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    print("device:", device)
    print("variant:", model_variant)
    print("checkpoint:", checkpoint_path)
    print("patch_size:", patch_size)
    print("bands:", [p.name for p in band_paths])
    print("classes:", lcz_classes)

    radius = patch_size // 2
    padded = np.pad(feature_stack, ((0, 0), (radius, radius), (radius, radius)), mode="constant")
    pred_code = np.full(template_shape, args.nodata, dtype=np.int16)
    pred_index = np.full(template_shape, -1, dtype=np.int16)
    pred_conf = np.zeros(template_shape, dtype=np.uint16) if args.out_conf_tif else None
    prob_cube = (
        np.zeros((num_classes, template_shape[0], template_shape[1]), dtype=np.float32)
        if args.save_probs else None
    )

    patches = []
    batch_coords = []
    done = 0
    last_report = 0
    started = time.time()

    print("========== Predict ==========")
    with torch.inference_mode():
        for row50, col50 in zip(rows.tolist(), cols.tolist()):
            row10 = int(row50) * 5 + 2 + radius
            col10 = int(col50) * 5 + 2 + radius
            patch = padded[
                :,
                row10 - radius:row10 + radius + 1,
                col10 - radius:col10 + radius + 1,
            ]
            if patch.shape != (in_channels, patch_size, patch_size):
                raise RuntimeError(f"Invalid patch shape at ({row50}, {col50}): {patch.shape}")
            patches.append(patch)
            batch_coords.append((int(row50), int(col50)))

            if len(patches) >= args.batch_size:
                write_batch_predictions(
                    model, patches, batch_coords, device, lcz_classes,
                    pred_code, pred_index, pred_conf, prob_cube,
                )
                done += len(patches)
                patches.clear()
                batch_coords.clear()

                if done - last_report >= args.progress_every:
                    elapsed = time.time() - started
                    speed = done / max(elapsed, 1e-6)
                    remain = (total_cells - done) / max(speed, 1e-6)
                    print(f"predicted {done:,}/{total_cells:,} | {speed:.1f} cells/s | remain {remain / 60:.1f} min")
                    last_report = done

        if patches:
            write_batch_predictions(
                model, patches, batch_coords, device, lcz_classes,
                pred_code, pred_index, pred_conf, prob_cube,
            )
            done += len(patches)

    elapsed = time.time() - started
    print(f"Done prediction: {done:,} cells in {elapsed / 60:.1f} min")

    print("========== Save outputs ==========")
    save_geotiff(out_tif, pred_code, template_profile, args.nodata, "int16", ["LCZ class code"])
    print("saved:", out_tif)

    if args.out_index_tif:
        save_geotiff(Path(args.out_index_tif), pred_index, template_profile, -1, "int16", ["LCZ class index"])
        print("saved:", args.out_index_tif)
    if args.out_conf_tif:
        save_geotiff(Path(args.out_conf_tif), pred_conf, template_profile, 0, "uint16", ["Max class probability x10000"])
        print("saved:", args.out_conf_tif)
    if args.save_probs:
        prob_path = Path(args.out_prob_tif) if args.out_prob_tif else out_tif.with_name(out_tif.stem + "_probs.tif")
        descriptions = [f"P(LCZ {class_code})" for class_code in lcz_classes]
        save_geotiff(prob_path, prob_cube, template_profile, 0.0, "float32", descriptions)
        print("saved:", prob_path)

    metadata = {
        "model_variant": model_variant,
        "model_py": str(model_py),
        "checkpoint": str(checkpoint_path),
        "norm_dir": str(norm_dir),
        "band_files": [str(p) for p in band_paths],
        "template_tif": str(template_tif),
        "valid_mode": args.valid_mode,
        "valid_mask_tif": args.valid_mask_tif,
        "output_tif": str(out_tif),
        "num_predicted_cells": int(done),
        "input_channels": in_channels,
        "patch_size": patch_size,
        "lcz_classes": lcz_classes,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_oa": checkpoint.get("val_oa"),
    }
    metadata_path = out_tif.with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    print("saved:", metadata_path)


if __name__ == "__main__":
    main()
