#!/usr/bin/env python3
"""
Apply identical LCZ map post-processing to one or more model outputs.

Recommended use
---------------
Use this only for map visualization / LST analysis products after saving raw
model predictions. Keep raw maps for quantitative accuracy assessment.

Example
-------
python code/6_postprocess_lcz_majority_filter.py \
  --input_tifs \
    /mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_cnn_rs_full_extent_raw.tif \
    /mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_rs_raster_building_full_extent_raw.tif \
    /mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_building_visual_full_extent_raw.tif \
    /mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_v7_full_extent_raw.tif \
  --suffix _maj3 \
  --window 3 \
  --nodata 0
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import generic_filter


def majority_func(values, nodata):
    vals = values.astype(np.int64)
    center = vals[len(vals) // 2]
    if center == nodata:
        return nodata
    vals = vals[vals != nodata]
    if len(vals) == 0:
        return center
    uniq, cnt = np.unique(vals, return_counts=True)
    return int(uniq[np.argmax(cnt)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_tifs", nargs="+", required=True)
    p.add_argument("--suffix", type=str, default="_maj3")
    p.add_argument("--window", type=int, default=3, choices=[3, 5, 7])
    p.add_argument("--nodata", type=int, default=0)
    args = p.parse_args()

    for tif in args.input_tifs:
        in_path = Path(tif)
        out_path = in_path.with_name(in_path.stem + args.suffix + in_path.suffix)
        with rasterio.open(in_path) as src:
            arr = src.read(1)
            profile = src.profile.copy()
        filtered = generic_filter(
            arr,
            function=lambda x: majority_func(x, args.nodata),
            size=args.window,
            mode="nearest",
        ).astype(arr.dtype)
        profile.update(compress="lzw", tiled=True, blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(filtered, 1)
            dst.set_band_description(1, f"LCZ class code, {args.window}x{args.window} majority filtered")
        print("saved:", out_path)


if __name__ == "__main__":
    main()
