#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create paper-ready qualitative subset figures from full LCZ maps.

Outputs
-------
1) Full Proposed/V7 LCZ map with labeled subset boxes.
2) A five-column figure for each subset:
   Satellite RGB | RS-CNN | RasterConcat-CNN | BuildingVisual-CNN | BGF-LCZNet
3) Cropped LCZ GeoTIFFs for each model and subset.

You can pass boxes directly:
  --bbox "Gangnam=127.005,37.490,127.090,37.535"
Coordinates are interpreted by --bbox_crs. Default is EPSG:4326.

Example: postprocessed maps for paper figures
---------------------------------------------
python code/7_crop_subset_figures.py \
  --rgb_tifs \
    /mnt/disk1/workspace_jym/LCZ/data/Satellite/processed/10m/04_SR_B4_10m.tif \
    /mnt/disk1/workspace_jym/LCZ/data/Satellite/processed/10m/03_SR_B3_10m.tif \
    /mnt/disk1/workspace_jym/LCZ/data/Satellite/processed/10m/02_SR_B2_10m.tif \
  --rgb_label "Landsat 8 RGB" \
  --map "RS-CNN (Baseline)=/mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_cnn_rs_full_extent_raw_maj3.tif" \
  --map "RasterConcat-CNN=/mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_rs_raster_building_full_extent_raw_maj3.tif" \
  --map "BuildingVisual-CNN=/mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_building_visual_full_extent_raw_maj3.tif" \
  --map "BGF-LCZNet (Ours)=/mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_v7_full_extent_raw_maj3.tif" \
  --out_dir /mnt/disk1/workspace_jym/LCZ/work_dirs/maps/subsets_postprocessed_maj3 \
  --bbox "Gangnam=127.005,37.490,127.090,37.535" \
  --bbox "Yeouido=126.900,37.515,126.950,37.545" \
  --bbox "Seongsu=127.030,37.535,127.075,37.565" \
  --bbox "Guro=126.850,37.460,126.930,37.515" \
  --bbox "Bukhansan=126.950,37.610,127.050,37.700" \
  --bbox_crs EPSG:4326
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle


LCZ_PALETTE = {
    0:   (0, 0, 0, 0),
    1:   (140, 0, 0, 255),
    2:   (209, 0, 0, 255),
    3:   (255, 0, 0, 255),
    4:   (191, 77, 0, 255),
    5:   (255, 102, 0, 255),
    6:   (255, 179, 102, 255),
    7:   (255, 255, 0, 255),
    8:   (188, 188, 188, 255),
    9:   (255, 204, 170, 255),
    10:  (85, 85, 85, 255),
    101: (0, 100, 0, 255),
    102: (34, 139, 34, 255),
    103: (107, 142, 35, 255),
    104: (181, 219, 90, 255),
    105: (0, 0, 0, 255),
    106: (251, 236, 93, 255),
    107: (0, 0, 255, 255),
}

LCZ_NAME = {
    1: "LCZ 1", 2: "LCZ 2", 3: "LCZ 3", 4: "LCZ 4", 5: "LCZ 5", 6: "LCZ 6",
    7: "LCZ 7", 8: "LCZ 8", 9: "LCZ 9", 10: "LCZ 10",
    101: "LCZ A", 102: "LCZ B", 103: "LCZ C", 104: "LCZ D",
    105: "LCZ E", 106: "LCZ F", 107: "LCZ G",
}


def rgba01(code):
    r, g, b, a = LCZ_PALETTE[code]
    return (r / 255, g / 255, b / 255, a / 255)


def parse_map_arg(values):
    parsed = []
    for v in values:
        if "=" not in v:
            raise ValueError("--map must be NAME=/path/to/file.tif")
        name, path = v.split("=", 1)
        parsed.append((name, Path(path)))
    return parsed


def parse_bbox_arg(v):
    if "=" not in v:
        raise ValueError('--bbox format must be "Name=minx,miny,maxx,maxy"')
    name, coords = v.split("=", 1)
    vals = [float(x) for x in coords.split(",")]
    if len(vals) != 4:
        raise ValueError('--bbox coordinates must be minx,miny,maxx,maxy')
    return name, tuple(vals)


def compact_cmap(arrays):
    used = set()
    for arr in arrays:
        used.update([int(x) for x in np.unique(arr) if int(x) != 0])
    used = sorted([c for c in used if c in LCZ_PALETTE])
    colors = [rgba01(0)]
    code_to_idx = {0: 0}
    for i, code in enumerate(used, start=1):
        code_to_idx[code] = i
        colors.append(rgba01(code))
    return used, code_to_idx, ListedColormap(colors), BoundaryNorm(np.arange(-0.5, len(colors) + 0.5), len(colors))


def remap(arr, code_to_idx):
    out = np.zeros_like(arr, dtype=np.int16)
    for code, idx in code_to_idx.items():
        out[arr == code] = idx
    return out


def read_lcz_crop(src, bounds):
    win = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
    arr = src.read(1, window=win, boundless=True, fill_value=0)
    if src.nodata is not None:
        arr = np.where(arr == src.nodata, 0, arr)
    return arr, win


def stretch_rgb(rgb):
    out = np.zeros_like(rgb, dtype=np.float32)
    valid = np.all(np.isfinite(rgb), axis=2)
    for channel in range(3):
        values = rgb[:, :, channel][valid]
        values = values[values > 0]
        if len(values) == 0:
            continue
        low, high = np.percentile(values, [2, 98])
        if high <= low:
            high = low + 1.0
        out[:, :, channel] = np.clip((rgb[:, :, channel] - low) / (high - low), 0, 1)
    out[~valid] = 0
    return np.clip(out, 0, 1)


def read_rgb_crop(rgb_srcs, bounds, bounds_crs, multiband=False):
    if multiband:
        src = rgb_srcs[0]
        if src.count < 3:
            raise ValueError(f"--rgb_tif must contain at least 3 bands, got {src.count}")
        src_bounds = (
            transform_bounds(bounds_crs, src.crs, *bounds, densify_pts=21)
            if str(src.crs) != str(bounds_crs)
            else bounds
        )
        win = from_bounds(*src_bounds, transform=src.transform).round_offsets().round_lengths()
        rgb = src.read([1, 2, 3], window=win, boundless=True, fill_value=src.nodata or 0).astype(np.float32)
        if src.nodata is not None:
            rgb[rgb == src.nodata] = np.nan
        return stretch_rgb(np.moveaxis(rgb, 0, 2))

    arrays = []
    shape = None
    for src in rgb_srcs:
        src_bounds = (
            transform_bounds(bounds_crs, src.crs, *bounds, densify_pts=21)
            if str(src.crs) != str(bounds_crs)
            else bounds
        )
        win = from_bounds(*src_bounds, transform=src.transform).round_offsets().round_lengths()
        arr = src.read(1, window=win, boundless=True, fill_value=src.nodata or 0).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        if shape is None:
            shape = arr.shape
        elif arr.shape != shape:
            raise ValueError("All RGB rasters must have the same grid and crop shape")
        arrays.append(arr)
    return stretch_rgb(np.stack(arrays, axis=2))


def make_overview(src, bbox_items_projected, out_png, overview_name):
    arr = src.read(1)
    if src.nodata is not None:
        arr = np.where(arr == src.nodata, 0, arr)
    used, code_to_idx, cmap, norm = compact_cmap([arr])
    idx_arr = remap(arr, code_to_idx)

    fig, ax = plt.subplots(figsize=(11, 10))
    ax.imshow(idx_arr, cmap=cmap, norm=norm, interpolation="nearest")
    box_colors = ["#ffe600", "#ffffff", "#00e5ff", "#ff5ac8", "#7cff00", "#ff9f1c"]

    for index, (region_name, bounds) in enumerate(bbox_items_projected):
        win = from_bounds(*bounds, transform=src.transform)
        x = float(win.col_off)
        y = float(win.row_off)
        width = float(win.width)
        height = float(win.height)
        color = box_colors[index % len(box_colors)]
        ax.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor="black", linewidth=4.0))
        ax.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor=color, linewidth=2.2))
        ax.text(
            x + 3,
            y + 3,
            region_name,
            color="black",
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="left",
            bbox={"facecolor": color, "edgecolor": "black", "alpha": 0.9, "pad": 2.5},
        )

    handles = [mpatches.Patch(color=rgba01(code), label=LCZ_NAME.get(code, str(code))) for code in used]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(1.01, 0), frameon=False, fontsize=8)
    ax.set_title(f"{overview_name}: full LCZ map and selected subset areas", fontsize=14, fontweight="bold")
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--map", action="append", required=True, help="NAME=/path/to/lcz_map.tif. Use multiple times.")
    rgb_group = p.add_mutually_exclusive_group(required=True)
    rgb_group.add_argument(
        "--rgb_tif",
        help="A georeferenced multiband RGB GeoTIFF, such as an export from GEE.",
    )
    rgb_group.add_argument(
        "--rgb_tifs",
        nargs=3,
        metavar=("RED_TIF", "GREEN_TIF", "BLUE_TIF"),
        help="Three aligned satellite bands in natural-color R G B order.",
    )
    p.add_argument("--rgb_label", default="Satellite RGB")
    p.add_argument("--bbox", action="append", required=True, help='Name=minx,miny,maxx,maxy')
    p.add_argument("--bbox_crs", default="EPSG:4326")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--overview_map", default=None, help="Map name used for full-map subset boxes. Default: last --map.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    map_items = parse_map_arg(args.map)
    bbox_items = [parse_bbox_arg(x) for x in args.bbox]

    srcs = [(name, rasterio.open(path)) for name, path in map_items]
    rgb_paths = [args.rgb_tif] if args.rgb_tif else args.rgb_tifs
    rgb_srcs = [rasterio.open(path) for path in rgb_paths]
    rgb_is_multiband = args.rgb_tif is not None
    try:
        base_crs = srcs[0][1].crs
        for model_name, src in srcs[1:]:
            if src.crs != base_crs or src.shape != srcs[0][1].shape or src.transform != srcs[0][1].transform:
                raise ValueError(f"{model_name} map is not aligned with the first LCZ map")

        bbox_items_projected = [
            (
                name,
                transform_bounds(args.bbox_crs, base_crs, *bbox, densify_pts=21)
                if args.bbox_crs != str(base_crs)
                else bbox,
            )
            for name, bbox in bbox_items
        ]
        overview_name = args.overview_map or srcs[-1][0]
        overview_matches = [src for name, src in srcs if name == overview_name]
        if not overview_matches:
            raise ValueError(f"--overview_map {overview_name!r} is not one of {[name for name, _ in srcs]}")
        overview_png = out_dir / "full_map_with_subset_boxes.png"
        make_overview(overview_matches[0], bbox_items_projected, overview_png, overview_name)
        print("saved:", overview_png)

        for region_name, tb in bbox_items_projected:
            satellite_rgb = read_rgb_crop(rgb_srcs, tb, base_crs, multiband=rgb_is_multiband)

            arrays = []
            profiles = []
            for model_name, src in srcs:
                arr, win = read_lcz_crop(src, tb)
                arrays.append(arr)
                profiles.append((model_name, src.profile.copy(), src.window_transform(win)))

            used, code_to_idx, cmap, norm = compact_cmap(arrays)

            for arr, (model_name, profile, win_transform) in zip(arrays, profiles):
                profile.update(
                    height=arr.shape[0],
                    width=arr.shape[1],
                    transform=win_transform,
                    dtype="int16",
                    nodata=0,
                    count=1,
                    compress="lzw",
                )
                crop_tif = out_dir / f"{region_name}_{model_name}.tif"
                with rasterio.open(crop_tif, "w", **profile) as dst:
                    dst.write(arr.astype(np.int16), 1)
                    dst.write_colormap(1, {k: v for k, v in LCZ_PALETTE.items() if k <= 255})

            n = len(arrays) + 1
            fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 5.6))
            axes[0].imshow(satellite_rgb, interpolation="nearest")
            axes[0].set_title(args.rgb_label, fontsize=12, fontweight="bold")
            axes[0].set_axis_off()
            for ax, (model_name, _src), arr in zip(axes[1:], srcs, arrays):
                ax.imshow(remap(arr, code_to_idx), cmap=cmap, norm=norm, interpolation="nearest")
                ax.set_title(model_name, fontsize=12, fontweight="bold")
                ax.set_axis_off()

            handles = [mpatches.Patch(color=rgba01(code), label=LCZ_NAME.get(code, str(code))) for code in used]
            axes[-1].legend(handles=handles, loc="lower left", bbox_to_anchor=(1.02, 0.0), frameon=False, fontsize=8)
            fig.suptitle(region_name, fontsize=15, fontweight="bold")
            plt.tight_layout()
            out_png = out_dir / f"{region_name}_comparison.png"
            fig.savefig(out_png, dpi=300, bbox_inches="tight")
            plt.close(fig)

            print("saved:", out_png)

    finally:
        for _, src in srcs:
            src.close()
        for src in rgb_srcs:
            src.close()


if __name__ == "__main__":
    main()
