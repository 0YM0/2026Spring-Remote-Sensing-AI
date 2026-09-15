#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply LCZ categorical color table, optionally mask by Seoul boundary, export PNG, and compute area statistics.

Example: full extent map
------------------------
# A. RS-only 8채널 모델
python code/2_visualize_lcz_map_and_area.py \
  --input_tif /mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_cnn_rs_full_extent_raw.tif \
  --out_dir /mnt/disk1/workspace_jym/LCZ/work_dirs/maps/figures_cnn_rs_full \
  --title "RS-CNN (Baseline) LCZ map"

# B. RS + Rasterized Building 10채널 모델
python code/2_visualize_lcz_map_and_area.py \
  --input_tif /mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_rs_raster_building_full_extent_raw.tif \
  --out_dir /mnt/disk1/workspace_jym/LCZ/work_dirs/maps/figures_rs_raster_building_full \
  --title "RasterConcat-CNN LCZ map"

# C. RS + Building dual visual encoder
python code/2_visualize_lcz_map_and_area.py \
  --input_tif /mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_building_visual_full_extent_raw.tif \
  --out_dir /mnt/disk1/workspace_jym/LCZ/work_dirs/maps/figures_building_visual_full \
  --title "BuildingVisual-CNN LCZ map"

# D. Building graph fusion model
python code/2_visualize_lcz_map_and_area.py \
  --input_tif /mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_v7_full_extent_raw.tif \
  --out_dir /mnt/disk1/workspace_jym/LCZ/work_dirs/maps/figures_v7_full \
  --title "BGF-LCZNet (Ours) LCZ map"


"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches


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
    1: "LCZ 1 Compact high-rise",
    2: "LCZ 2 Compact mid-rise",
    3: "LCZ 3 Compact low-rise",
    4: "LCZ 4 Open high-rise",
    5: "LCZ 5 Open mid-rise",
    6: "LCZ 6 Open low-rise",
    7: "LCZ 7 Lightweight low-rise",
    8: "LCZ 8 Large low-rise",
    9: "LCZ 9 Sparsely built",
    10: "LCZ 10 Heavy industry",
    101: "LCZ A Dense trees",
    102: "LCZ B Scattered trees",
    103: "LCZ C Bush/scrub",
    104: "LCZ D Low plants",
    105: "LCZ E Bare rock/paved",
    106: "LCZ F Bare soil/sand",
    107: "LCZ G Water",
}


def rgba01(code):
    r, g, b, a = LCZ_PALETTE[code]
    return (r / 255, g / 255, b / 255, a / 255)


def add_color_table(output_tif: Path, arr: np.ndarray, profile: dict):
    output_tif.parent.mkdir(parents=True, exist_ok=True)
    profile = profile.copy()
    profile.update(dtype="int16", count=1, nodata=0, compress="lzw")
    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(arr.astype(np.int16), 1)
        dst.write_colormap(1, {k: v for k, v in LCZ_PALETTE.items() if k <= 255})


def make_png(arr, out_png: Path, title: str = "", crop_valid: bool = True, dpi: int = 300):
    valid = arr != 0
    plot_arr = arr.copy()

    if crop_valid and np.any(valid):
        rows, cols = np.where(valid)
        r0, r1 = max(rows.min() - 5, 0), min(rows.max() + 6, arr.shape[0])
        c0, c1 = max(cols.min() - 5, 0), min(cols.max() + 6, arr.shape[1])
        plot_arr = plot_arr[r0:r1, c0:c1]

    used_codes = [c for c in LCZ_PALETTE.keys() if c != 0 and np.any(plot_arr == c)]
    if not used_codes:
        raise RuntimeError("No valid LCZ class found.")

    code_to_idx = {0: 0}
    colors = [rgba01(0)]
    labels = []
    for idx, code in enumerate(used_codes, start=1):
        code_to_idx[code] = idx
        colors.append(rgba01(code))
        labels.append((idx, code, LCZ_NAME.get(code, str(code))))

    idx_arr = np.zeros_like(plot_arr, dtype=np.int16)
    for code, idx in code_to_idx.items():
        idx_arr[plot_arr == code] = idx

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5, 1), cmap.N)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(idx_arr, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=14)

    handles = [mpatches.Patch(color=colors[idx], label=name) for idx, code, name in labels]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(1.02, 0.0), frameon=False, fontsize=8)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_tif", required=True)
    p.add_argument("--mask_tif", default=None, help="Optional 1/0 valid mask aligned with input_tif.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--title", default="")
    p.add_argument("--prefix", default=None)
    p.add_argument("--no_crop_valid", action="store_true")
    args = p.parse_args()

    input_tif = Path(args.input_tif)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or input_tif.stem

    with rasterio.open(input_tif) as src:
        arr = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        nodata = src.nodata

    if nodata is not None:
        arr = np.where(arr == nodata, 0, arr)

    if args.mask_tif:
        with rasterio.open(args.mask_tif) as msrc:
            mask = msrc.read(1)
            if mask.shape != arr.shape:
                raise ValueError(f"Mask shape {mask.shape} != input shape {arr.shape}")
        arr = np.where(mask > 0, arr, 0)

    colored_tif = out_dir / f"{prefix}_colored.tif"
    png_path = out_dir / f"{prefix}.png"
    csv_path = out_dir / f"{prefix}_area_stats.csv"

    add_color_table(colored_tif, arr, profile)
    make_png(arr, png_path, title=args.title, crop_valid=(not args.no_crop_valid))

    pixel_area_m2 = abs(transform.a * transform.e)
    valid = arr != 0
    valid_total = int(valid.sum())

    rows = []
    for code in sorted([c for c in np.unique(arr).tolist() if c != 0]):
        count = int((arr == code).sum())
        rows.append({
            "class_code": int(code),
            "class_name": LCZ_NAME.get(int(code), str(code)),
            "pixel_count": count,
            "area_km2": count * pixel_area_m2 / 1_000_000,
            "percent": count / valid_total * 100 if valid_total > 0 else 0,
        })

    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("saved:", colored_tif)
    print("saved:", png_path)
    print("saved:", csv_path)
    print("valid cells:", valid_total)


if __name__ == "__main__":
    main()
