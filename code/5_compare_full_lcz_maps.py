#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare full LCZ maps from multiple models:
- side-by-side PNG
- pairwise disagreement maps
- pairwise change ratio CSV
- class area distribution CSV

Examples
--------
# Step 5: compare raw full-extent maps
python code/5_compare_full_lcz_maps.py \
  --map "RS-CNN (Baseline)=/mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_cnn_rs_full_extent_raw.tif" \
  --map "RasterConcat-CNN=/mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_rs_raster_building_full_extent_raw.tif" \
  --map "BuildingVisual-CNN=/mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_building_visual_full_extent_raw.tif" \
  --map "BGF-LCZNet (Ours)=/mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_v7_full_extent_raw.tif" \
  --out_dir /mnt/disk1/workspace_jym/LCZ/work_dirs/maps/comparison_raw_full \
  --title "Raw full-extent LCZ map comparison"

# Step 7: compare identically postprocessed maps
python code/5_compare_full_lcz_maps.py \
  --map "RS-CNN (Baseline)=/mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_cnn_rs_full_extent_raw_maj3.tif" \
  --map "RasterConcat-CNN=/mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_rs_raster_building_full_extent_raw_maj3.tif" \
  --map "BuildingVisual-CNN=/mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_building_visual_full_extent_raw_maj3.tif" \
  --map "BGF-LCZNet (Ours)=/mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_v7_full_extent_raw_maj3.tif" \
  --out_dir /mnt/disk1/workspace_jym/LCZ/work_dirs/maps/comparison_postprocessed_maj3 \
  --title "Postprocessed LCZ map comparison (3x3 majority)"
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


def load_map(path):
    with rasterio.open(path) as src:
        arr = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        nodata = src.nodata
    if nodata is not None:
        arr = np.where(arr == nodata, 0, arr)
    return arr, profile, transform


def compact_cmap_for_arrays(arrays):
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


def remap_to_idx(arr, code_to_idx):
    out = np.zeros(arr.shape, dtype=np.int16)
    for code, idx in code_to_idx.items():
        out[arr == code] = idx
    return out


def crop_valid_bbox(arrays, mask=None, pad=5):
    if mask is not None:
        valid = mask > 0
    else:
        valid = np.zeros_like(arrays[0], dtype=bool)
        for arr in arrays:
            valid |= (arr != 0)
    if not np.any(valid):
        return slice(0, arrays[0].shape[0]), slice(0, arrays[0].shape[1])
    rows, cols = np.where(valid)
    r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad + 1, arrays[0].shape[0])
    c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad + 1, arrays[0].shape[1])
    return slice(r0, r1), slice(c0, c1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--map", action="append", required=True, help="NAME=/path/to/lcz_map.tif. Use multiple times.")
    p.add_argument("--mask_tif", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--title", default="Seoul LCZ map comparison")
    p.add_argument("--no_crop_valid", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    map_items = parse_map_arg(args.map)
    names, arrays = [], []
    profile, transform = None, None

    for name, path in map_items:
        arr, prof, trans = load_map(path)
        if profile is None:
            profile, transform = prof, trans
        elif arr.shape != arrays[0].shape:
            raise ValueError(f"Shape mismatch for {name}: {arr.shape} != {arrays[0].shape}")
        names.append(name)
        arrays.append(arr)

    mask = None
    if args.mask_tif:
        with rasterio.open(args.mask_tif) as src:
            mask = src.read(1) > 0
        if mask.shape != arrays[0].shape:
            raise ValueError(f"Mask shape {mask.shape} != map shape {arrays[0].shape}")
        arrays = [np.where(mask, arr, 0) for arr in arrays]

    used, code_to_idx, cmap, norm = compact_cmap_for_arrays(arrays)
    rs, cs = crop_valid_bbox(arrays, mask=mask, pad=5) if not args.no_crop_valid else (slice(None), slice(None))

    n = len(arrays)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, name, arr in zip(axes, names, arrays):
        ax.imshow(remap_to_idx(arr[rs, cs], code_to_idx), cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(name)
        ax.set_axis_off()

    handles = [mpatches.Patch(color=rgba01(code), label=f"{LCZ_NAME.get(code, code)}") for code in used]
    axes[-1].legend(handles=handles, loc="lower left", bbox_to_anchor=(1.02, 0.0), frameon=False, fontsize=8)
    fig.suptitle(args.title, fontsize=14)
    plt.tight_layout()
    side_png = out_dir / "full_map_side_by_side.png"
    fig.savefig(side_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    pixel_area_m2 = abs(transform.a * transform.e)
    valid = mask if mask is not None else np.any(np.stack([(a != 0) for a in arrays], axis=0), axis=0)
    area_rows = []
    for name, arr in zip(names, arrays):
        valid_model = valid & (arr != 0)
        total_valid = valid_model.sum()
        for code in used:
            count = int((valid_model & (arr == code)).sum())
            area_rows.append({
                "model": name,
                "class_code": code,
                "class_name": LCZ_NAME.get(code, str(code)),
                "pixel_count": count,
                "area_km2": count * pixel_area_m2 / 1_000_000,
                "percent": count / total_valid * 100 if total_valid > 0 else 0,
            })
    pd.DataFrame(area_rows).to_csv(out_dir / "class_area_distribution.csv", index=False, encoding="utf-8-sig")

    pair_rows = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = arrays[i], arrays[j]
            pair_valid = valid & (a != 0) & (b != 0)
            diff = (a != b) & pair_valid
            same = (a == b) & pair_valid
            total = int(pair_valid.sum())
            pair_rows.append({
                "model_a": names[i],
                "model_b": names[j],
                "valid_pixels": total,
                "same_pixels": int(same.sum()),
                "different_pixels": int(diff.sum()),
                "different_percent": float(diff.sum() / total * 100) if total > 0 else 0.0,
            })

            out_diff = np.zeros_like(a, dtype=np.uint8)
            out_diff[same] = 1
            out_diff[diff] = 2

            diff_profile = profile.copy()
            diff_profile.update(dtype="uint8", count=1, nodata=0, compress="lzw")
            diff_tif = out_dir / f"diff_{names[i]}_vs_{names[j]}.tif"
            with rasterio.open(diff_tif, "w", **diff_profile) as dst:
                dst.write(out_diff, 1)
                dst.write_colormap(1, {
                    0: (0, 0, 0, 0),
                    1: (210, 210, 210, 255),
                    2: (230, 0, 0, 255),
                })

    pd.DataFrame(pair_rows).to_csv(out_dir / "pairwise_disagreement.csv", index=False, encoding="utf-8-sig")

    print("saved:", side_png)
    print("saved:", out_dir / "class_area_distribution.csv")
    print("saved:", out_dir / "pairwise_disagreement.csv")


if __name__ == "__main__":
    main()
