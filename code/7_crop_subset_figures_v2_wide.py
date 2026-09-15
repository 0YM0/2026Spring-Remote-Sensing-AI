#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create paper-ready qualitative subset figures with high-resolution imagery.

Outputs
-------
1) Full BGF-LCZNet LCZ map with labeled subset boxes.
2) A five-column figure for each subset:
   High-resolution imagery | RS-CNN | RasterConcat-CNN | BuildingVisual-CNN | BGF-LCZNet
3) Cached imagery tiles, georeferenced imagery crops, and cropped LCZ GeoTIFFs.

The default Esri World Imagery source provides high-resolution satellite and
aerial imagery where available. At Seoul latitude, zoom 17 has approximately
0.95 m pixel spacing. Source imagery resolution and acquisition date vary by
location. Keep the attribution shown in the output figure.

Example: high-resolution imagery with the same wide subset extents as v1
-----------------------------------------------------------------------
# maj3
python code/7_crop_subset_figures_v2_wide.py \
  --map "RS-CNN (Baseline)=/mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_cnn_rs_full_extent_raw_maj3.tif" \
  --map "RasterConcat-CNN=/mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_rs_raster_building_full_extent_raw_maj3.tif" \
  --map "BuildingVisual-CNN=/mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_building_visual_full_extent_raw_maj3.tif" \
  --map "BGF-LCZNet (Ours)=/mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_v7_full_extent_raw_maj3.tif" \
  --out_dir /mnt/disk1/workspace_jym/LCZ/work_dirs/maps/subsets_highres_v2_wide_z17_maj3 \
  --use_v1_bboxes \
  --bbox_crs EPSG:4326 \
  --zoom 17 \
  --max_tiles 5000

# 다른 위치
python code/7_crop_subset_figures_v2_wide.py \
  --map "RS-CNN (Baseline)=/mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_cnn_rs_full_extent_raw_maj3.tif" \
  --map "RasterConcat-CNN=/mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_rs_raster_building_full_extent_raw_maj3.tif" \
  --map "BuildingVisual-CNN=/mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_building_visual_full_extent_raw_maj3.tif" \
  --map "BGF-LCZNet (Ours)=/mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_v7_full_extent_raw_maj3.tif" \
  --out_dir /mnt/disk1/workspace_jym/LCZ/work_dirs/maps/subsets_highres_v2_wide_z17_maj3 \
  --bbox "Seongsu=127.030,37.535,127.075,37.565" \
  --bbox "Guro=126.865,37.465,126.915,37.505" \
  --bbox "Yeouido=126.890,37.510,126.960,37.555" \
  --bbox "Jamsil=127.055,37.485,127.115,37.525" \
  --bbox "Bukhansan=126.950,37.610,127.040,37.685" \
  --bbox_crs EPSG:4326 \
  --zoom 17 \
  --max_tiles 5000

# raw map
python code/7_crop_subset_figures_v2_wide.py \
  --map "RS-CNN (Baseline)=/mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_cnn_rs_full_extent_raw.tif" \
  --map "RasterConcat-CNN=/mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_rs_raster_building_full_extent_raw.tif" \
  --map "BuildingVisual-CNN=/mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_building_visual_full_extent_raw.tif" \
  --map "BGF-LCZNet (Ours)=/mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_v7_full_extent_raw.tif" \
  --out_dir /mnt/disk1/workspace_jym/LCZ/work_dirs/maps/subsets_highres_v2_wide_raw \
  --use_v1_bboxes \
  --bbox_crs EPSG:4326 \
  --zoom 16

# 다른 위치
python code/7_crop_subset_figures_v2_wide.py \
  --map "RS-CNN (Baseline)=/mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_cnn_rs_full_extent_raw.tif" \
  --map "RasterConcat-CNN=/mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_rs_raster_building_full_extent_raw.tif" \
  --map "BuildingVisual-CNN=/mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_building_visual_full_extent_raw.tif" \
  --map "BGF-LCZNet (Ours)=/mnt/disk1/workspace_jym/LCZ/work_dirs/maps/seoul_lcz_v7_full_extent_raw.tif" \
  --out_dir /mnt/disk1/workspace_jym/LCZ/work_dirs/maps/subsets_highres_v2_wide_z17_raw \
  --bbox "Seongsu=127.030,37.535,127.075,37.565" \
  --bbox "Guro=126.865,37.465,126.915,37.505" \
  --bbox "Yeouido=126.890,37.510,126.960,37.555" \
  --bbox "Jamsil=127.055,37.485,127.115,37.525" \
  --bbox "Bukhansan=126.950,37.610,127.040,37.685" \
  --bbox_crs EPSG:4326 \
  --zoom 17 \
  --max_tiles 5000
"""

import argparse
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import requests
import rasterio
from rasterio.transform import from_origin
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

DEFAULT_TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
DEFAULT_ATTRIBUTION = (
    "Sources: Esri, Vantor, GeoEye, Earthstar Geographics, CNES/Airbus DS, "
    "USDA, USGS, AeroGRID, IGN, OpenStreetMap contributors, and the GIS User Community"
)

# Same subset extents as 7_crop_subset_figures.py (v1).
# These are deliberately wider than the compact v2 example boxes.
V1_DEFAULT_BBOXES = [
    "Gangnam=127.005,37.490,127.090,37.535",
    "Yeouido=126.900,37.515,126.950,37.545",
    "Seongsu=127.030,37.535,127.075,37.565",
    "Guro=126.850,37.460,126.930,37.515",
    "Bukhansan=126.950,37.610,127.050,37.700",
]
WEB_MERCATOR_HALF_WORLD = 20037508.342789244
TILE_SIZE = 256


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


def lonlat_to_tile_fraction(lon, lat, zoom):
    lat = float(np.clip(lat, -85.05112878, 85.05112878))
    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def fetch_tile(session, tile_url, zoom, x, y, cache_dir, timeout):
    cache_path = cache_dir / str(zoom) / str(x) / f"{y}.jpg"
    if cache_path.exists():
        return np.asarray(Image.open(cache_path).convert("RGB"))

    url = tile_url.format(z=zoom, x=x, y=y)
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(cache_path, format="JPEG", quality=95)
    return np.asarray(image)


def save_imagery_geotiff(path, rgb, transform):
    profile = {
        "driver": "GTiff",
        "height": rgb.shape[0],
        "width": rgb.shape[1],
        "count": 3,
        "dtype": "uint8",
        "crs": "EPSG:3857",
        "transform": transform,
        "compress": "jpeg",
        "photometric": "RGB",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.moveaxis(rgb, 2, 0))
        dst.set_band_description(1, "Red")
        dst.set_band_description(2, "Green")
        dst.set_band_description(3, "Blue")


def read_highres_imagery(bounds, bounds_crs, zoom, tile_url, cache_dir, max_tiles, timeout):
    lon_min, lat_min, lon_max, lat_max = transform_bounds(
        bounds_crs, "EPSG:4326", *bounds, densify_pts=21
    )
    x_left, y_bottom = lonlat_to_tile_fraction(lon_min, lat_min, zoom)
    x_right, y_top = lonlat_to_tile_fraction(lon_max, lat_max, zoom)

    tile_x0 = math.floor(x_left)
    tile_x1 = math.ceil(x_right) - 1
    tile_y0 = math.floor(y_top)
    tile_y1 = math.ceil(y_bottom) - 1
    n_tiles = (tile_x1 - tile_x0 + 1) * (tile_y1 - tile_y0 + 1)
    if n_tiles > max_tiles:
        raise ValueError(
            f"Requested bbox needs {n_tiles} imagery tiles at zoom {zoom}, exceeding "
            f"--max_tiles={max_tiles}. Use a smaller bbox or lower --zoom."
        )

    mosaic = np.zeros(
        ((tile_y1 - tile_y0 + 1) * TILE_SIZE, (tile_x1 - tile_x0 + 1) * TILE_SIZE, 3),
        dtype=np.uint8,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "LCZ-paper-figure-generator/1.0"})
    for tile_y in range(tile_y0, tile_y1 + 1):
        for tile_x in range(tile_x0, tile_x1 + 1):
            tile = fetch_tile(session, tile_url, zoom, tile_x, tile_y, cache_dir, timeout)
            row = (tile_y - tile_y0) * TILE_SIZE
            col = (tile_x - tile_x0) * TILE_SIZE
            mosaic[row:row + TILE_SIZE, col:col + TILE_SIZE] = tile

    global_left = int(math.floor(x_left * TILE_SIZE))
    global_right = int(math.ceil(x_right * TILE_SIZE))
    global_top = int(math.floor(y_top * TILE_SIZE))
    global_bottom = int(math.ceil(y_bottom * TILE_SIZE))
    local_left = global_left - tile_x0 * TILE_SIZE
    local_right = global_right - tile_x0 * TILE_SIZE
    local_top = global_top - tile_y0 * TILE_SIZE
    local_bottom = global_bottom - tile_y0 * TILE_SIZE
    crop = mosaic[local_top:local_bottom, local_left:local_right]

    resolution = 2 * WEB_MERCATOR_HALF_WORLD / (TILE_SIZE * (2 ** zoom))
    mercator_left = -WEB_MERCATOR_HALF_WORLD + global_left * resolution
    mercator_top = WEB_MERCATOR_HALF_WORLD - global_top * resolution
    transform = from_origin(mercator_left, mercator_top, resolution, resolution)
    center_lat = (lat_min + lat_max) / 2.0
    ground_spacing = resolution * math.cos(math.radians(center_lat))
    return crop, transform, n_tiles, ground_spacing


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
    p.add_argument("--imagery_label", default="Esri World Imagery")
    # Zoom 16 is recommended for the wider v1 subset boxes.
    # It is still much sharper than 10 m Landsat RGB, while keeping tile counts manageable.
    p.add_argument("--zoom", type=int, default=16, choices=range(14, 20))
    p.add_argument("--tile_url", default=DEFAULT_TILE_URL, help="XYZ/ArcGIS tile URL with {z}, {y}, and {x}.")
    p.add_argument("--tile_cache_dir", default=None, help="Default: OUT_DIR/imagery_tile_cache")
    p.add_argument("--attribution", default=DEFAULT_ATTRIBUTION)
    p.add_argument("--max_tiles", type=int, default=2000)
    p.add_argument("--request_timeout", type=float, default=30.0)
    p.add_argument("--bbox", action="append", default=None, help='Name=minx,miny,maxx,maxy. If omitted, v1 default subset boxes are used.')
    p.add_argument("--use_v1_bboxes", action="store_true", help="Use the same wide subset boxes as 7_crop_subset_figures.py.")
    p.add_argument("--bbox_crs", default="EPSG:4326")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--overview_map", default=None, help="Map name used for full-map subset boxes. Default: last --map.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tile_cache_dir = Path(args.tile_cache_dir) if args.tile_cache_dir else out_dir / "imagery_tile_cache"

    map_items = parse_map_arg(args.map)
    bbox_strings = V1_DEFAULT_BBOXES if (args.use_v1_bboxes or not args.bbox) else args.bbox
    bbox_items = [parse_bbox_arg(x) for x in bbox_strings]
    print("========== Subset boxes ==========")
    for name, bbox in bbox_items:
        print(f"{name}: {bbox}")

    srcs = [(name, rasterio.open(path)) for name, path in map_items]
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
            highres_rgb, imagery_transform, n_tiles, ground_spacing = read_highres_imagery(
                bounds=tb,
                bounds_crs=base_crs,
                zoom=args.zoom,
                tile_url=args.tile_url,
                cache_dir=tile_cache_dir,
                max_tiles=args.max_tiles,
                timeout=args.request_timeout,
            )
            imagery_tif = out_dir / f"{region_name}_highres_imagery_z{args.zoom}.tif"
            save_imagery_geotiff(imagery_tif, highres_rgb, imagery_transform)
            print(
                f"{region_name}: imagery_tiles={n_tiles}, "
                f"pixel_spacing_at_center={ground_spacing:.2f} m"
            )
            print("saved:", imagery_tif)

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
            axes[0].imshow(highres_rgb, interpolation="nearest")
            axes[0].set_title(
                f"{args.imagery_label}\n(z{args.zoom}, {ground_spacing:.2f} m/px)",
                fontsize=11,
                fontweight="bold",
            )
            axes[0].set_axis_off()
            for ax, (model_name, _src), arr in zip(axes[1:], srcs, arrays):
                ax.imshow(remap(arr, code_to_idx), cmap=cmap, norm=norm, interpolation="nearest")
                ax.set_title(model_name, fontsize=12, fontweight="bold")
                ax.set_axis_off()

            handles = [mpatches.Patch(color=rgba01(code), label=LCZ_NAME.get(code, str(code))) for code in used]
            axes[-1].legend(handles=handles, loc="lower left", bbox_to_anchor=(1.02, 0.0), frameon=False, fontsize=8)
            fig.suptitle(region_name, fontsize=15, fontweight="bold")
            fig.text(0.01, 0.01, args.attribution, ha="left", va="bottom", fontsize=6.5, color="#555555")
            plt.tight_layout(rect=(0, 0.035, 1, 1))
            out_png = out_dir / f"{region_name}_comparison.png"
            fig.savefig(out_png, dpi=300, bbox_inches="tight")
            plt.close(fig)

            metadata = {
                "region": region_name,
                "bbox_projected": [float(v) for v in tb],
                "bbox_crs": str(base_crs),
                "imagery_tile_url": args.tile_url,
                "imagery_zoom": args.zoom,
                "estimated_ground_pixel_spacing_m": ground_spacing,
                "downloaded_or_cached_tile_count": n_tiles,
                "imagery_geotiff": str(imagery_tif),
                "attribution": args.attribution,
                "note": "Pixel spacing is computed from Web Mercator zoom; source imagery resolution varies by location.",
            }
            with open(out_dir / f"{region_name}_imagery_metadata.json", "w", encoding="utf-8") as file:
                json.dump(metadata, file, indent=2, ensure_ascii=False)
            print("saved:", out_png)

    finally:
        for _, src in srcs:
            src.close()


if __name__ == "__main__":
    main()
