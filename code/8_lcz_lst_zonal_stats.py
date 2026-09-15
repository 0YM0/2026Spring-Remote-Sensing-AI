#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute LST statistics by LCZ class.

Example: final BGF-LCZNet (Ours) postprocessed map
-------------------------------------------
python code/8_lcz_lst_zonal_stats.py \
  --lcz_tif /mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_v7_full_extent_raw_maj3.tif \
  --lst_tif /mnt/disk1/workspace_jym/LCZ/data/Satellite/processed/10m/08_ST_B10_LST_C_10m.tif \
  --out_dir /mnt/disk1/workspace_jym/LCZ/work_dirs/lst_analysis/v7_maj3
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt


LCZ_NAME = {
    1: "LCZ 1", 2: "LCZ 2", 3: "LCZ 3", 4: "LCZ 4", 5: "LCZ 5", 6: "LCZ 6",
    7: "LCZ 7", 8: "LCZ 8", 9: "LCZ 9", 10: "LCZ 10",
    101: "LCZ A", 102: "LCZ B", 103: "LCZ C", 104: "LCZ D",
    105: "LCZ E", 106: "LCZ F", 107: "LCZ G",
}


def align_lst_to_lcz(lst_tif, lcz_profile, out_shape):
    with rasterio.open(lst_tif) as src:
        src_arr = src.read(1).astype(np.float32)
        src_nodata = src.nodata
        dst = np.full(out_shape, np.nan, dtype=np.float32)

        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=lcz_profile["transform"],
            dst_crs=lcz_profile["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    return dst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lcz_tif", required=True)
    p.add_argument("--lst_tif", required=True)
    p.add_argument("--mask_tif", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_samples_per_class", type=int, default=5000)
    p.add_argument("--lst_offset", type=float, default=0.0, help="Add offset to LST values. Example: -273.15 if Kelvin to Celsius.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.lcz_tif) as src:
        lcz = src.read(1)
        profile = src.profile.copy()
        nodata = src.nodata
    if nodata is not None:
        lcz = np.where(lcz == nodata, 0, lcz)

    lst = align_lst_to_lcz(args.lst_tif, profile, lcz.shape)
    lst = lst + args.lst_offset

    valid = (lcz != 0) & np.isfinite(lst)
    if args.mask_tif:
        with rasterio.open(args.mask_tif) as src:
            mask = src.read(1) > 0
        if mask.shape != lcz.shape:
            raise ValueError(f"Mask shape {mask.shape} != LCZ shape {lcz.shape}")
        valid &= mask

    rows = []
    box_data = []
    box_labels = []

    rng = np.random.default_rng(42)
    for code in sorted([int(c) for c in np.unique(lcz[valid]) if int(c) != 0]):
        vals = lst[valid & (lcz == code)]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue

        rows.append({
            "class_code": code,
            "class_name": LCZ_NAME.get(code, str(code)),
            "count": int(len(vals)),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
            "q25": float(np.percentile(vals, 25)),
            "q75": float(np.percentile(vals, 75)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        })

        if len(vals) > args.max_samples_per_class:
            vals_plot = rng.choice(vals, size=args.max_samples_per_class, replace=False)
        else:
            vals_plot = vals
        box_data.append(vals_plot)
        box_labels.append(LCZ_NAME.get(code, str(code)))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "lcz_lst_stats.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(max(10, len(box_data) * 0.8), 5))
    ax.boxplot(box_data, labels=box_labels, showfliers=False)
    ax.set_ylabel("LST")
    ax.set_title("LST distribution by LCZ class")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(out_dir / "lcz_lst_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("saved:", out_dir / "lcz_lst_stats.csv")
    print("saved:", out_dir / "lcz_lst_boxplot.png")


if __name__ == "__main__":
    main()
