"""
tiler.py — Sliding-window chip extractor with georeferencing and quality filtering.

Works with both projected (UTM) and geographic (WGS-84) CRS inputs.
"""

import os
import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS

import config as cfg


def calculate_tile_bounds(image_shape, tile_size, overlap):
    """Return list of (row_start, row_end, col_start, col_end) for sliding window."""
    height, width = image_shape[:2]
    stride = max(1, int(tile_size * (1 - overlap)))

    row_starts = list(range(0, height - tile_size + 1, stride))
    if not row_starts or row_starts[-1] + tile_size < height:
        row_starts.append(max(0, height - tile_size))

    col_starts = list(range(0, width - tile_size + 1, stride))
    if not col_starts or col_starts[-1] + tile_size < width:
        col_starts.append(max(0, width - tile_size))

    return [
        (r, r + tile_size, c, c + tile_size)
        for r in row_starts for c in col_starts
    ]


def extract_tile_with_georef(image_array, affine_transform, bounds):
    """Slice a tile from the image array and compute its affine transform."""
    row_start, row_end, col_start, col_end = bounds
    tile = image_array[row_start:row_end, col_start:col_end]
    tile_affine = affine_transform * Affine.translation(col_start, row_start)
    return tile, tile_affine


def pixel_to_latlon(affine_transform, pixel_coords, crs):
    """
    Convert pixel (col, row) pairs → (lat, lon) in WGS-84.

    Handles both projected (UTM) and geographic (EPSG:4326) input CRS.
    """
    from pyproj import Transformer

    crs_obj = CRS.from_string(crs) if isinstance(crs, str) else crs
    wgs84 = CRS.from_epsg(4326)

    need_transform = (crs_obj.to_epsg() != 4326)
    if need_transform:
        xformer = Transformer.from_crs(crs_obj, wgs84, always_xy=True)

    results = []
    for col, row in pixel_coords:
        x, y = affine_transform * (col, row)
        if need_transform:
            lon, lat = xformer.transform(x, y)
        else:
            lon, lat = x, y
        results.append((lat, lon))
    return results


def filter_garbage_tiles(tile_array, scl_band=None):
    """
    Reject tiles that are mostly empty, cloud-covered, or all-water.

    Returns (is_valid: bool, reason: str).
    """
    h, w = tile_array.shape[:2]
    total = h * w

    # NoData check — all channels == 0
    if tile_array.ndim == 3:
        black = np.all(tile_array == 0, axis=2)
    else:
        black = tile_array == 0

    if np.sum(black) / total > cfg.NODATA_THRESHOLD:
        return False, f"nodata_{np.sum(black)/total:.2f}"

    # SCL-based cloud / water rejection (Sentinel-2 only)
    if scl_band is not None:
        scl = cv2.resize(
            scl_band.astype(np.uint8), (w, h),
            interpolation=cv2.INTER_NEAREST
        )
        water_mask = (scl == 6)
        if np.sum(water_mask) / total > cfg.WATER_THRESHOLD:
            return False, f"water_{np.sum(water_mask)/total:.2f}"

        cloud_mask = (scl == 8) | (scl == 9)
        if np.sum(cloud_mask) / total > cfg.CLOUD_THRESHOLD:
            return False, f"cloud_{np.sum(cloud_mask)/total:.2f}"

    return True, "valid"


def save_geotiff(tile_array, tile_affine, crs, path):
    """Save a BGR uint8 tile as a georeferenced GeoTIFF (RGB band order)."""
    if tile_array.ndim == 2:
        tile_array = np.stack([tile_array] * 3, axis=2)

    h, w = tile_array.shape[:2]
    crs_obj = CRS.from_string(crs) if isinstance(crs, str) else crs

    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=h, width=w,
        count=3,
        dtype=tile_array.dtype,
        crs=crs_obj,
        transform=tile_affine,
        compress="lzw",
    ) as dst:
        # OpenCV stores BGR; GeoTIFF convention is RGB
        dst.write(tile_array[:, :, [2, 1, 0]].transpose(2, 0, 1))


def save_jpeg(tile_array, path):
    """Save a BGR tile as a high-quality JPEG for Roboflow upload."""
    cv2.imwrite(path, tile_array, [cv2.IMWRITE_JPEG_QUALITY, 92])


def generate_metadata_csv(records, path):
    pd.DataFrame(records).to_csv(path, index=False)


def tile_image(image_array, affine_transform, crs, scl_band,
               output_dir, segment_id, date_str):
    """
    Tile the full-scene image into 512×512 chips with 50% overlap.

    Saves:
      • .tif  — georeferenced GeoTIFF (for detection/inference with coordinates)
      • .jpg  — JPEG copy for Roboflow upload
      • metadata.csv — per-tile lat/lon corners and validity flags
    """
    tiles_dir = os.path.join(output_dir, segment_id, date_str, "tiles")
    os.makedirs(tiles_dir, exist_ok=True)

    bounds_list = calculate_tile_bounds(image_array.shape, cfg.TILE_SIZE, cfg.TILE_OVERLAP)
    total = len(bounds_list)
    print(f"   → {total} tiles to process ({cfg.TILE_SIZE}px, {cfg.TILE_OVERLAP*100:.0f}% overlap)")

    metadata, valid_count, rejected_count = [], 0, 0

    for idx, bounds in enumerate(bounds_list):
        tile_num = idx + 1
        base = f"tile_{tile_num:04d}"

        tile_arr, tile_aff = extract_tile_with_georef(image_array, affine_transform, bounds)

        # Skip tiles smaller than expected (edge tiles)
        if tile_arr.shape[0] < cfg.TILE_SIZE or tile_arr.shape[1] < cfg.TILE_SIZE:
            continue

        is_valid, reason = filter_garbage_tiles(tile_arr, scl_band)

        h, w = tile_arr.shape[:2]
        try:
            corners = pixel_to_latlon(tile_aff, [(0, 0), (w, 0), (w, h), (0, h)], crs)
            tl_lat, tl_lon = corners[0]
            br_lat, br_lon = corners[2]
        except Exception:
            tl_lat = tl_lon = br_lat = br_lon = None

        metadata.append({
            "tile_filename": base,
            "top_left_lat": tl_lat,
            "top_left_lon": tl_lon,
            "bottom_right_lat": br_lat,
            "bottom_right_lon": br_lon,
            "tile_width": w,
            "tile_height": h,
            "is_valid": is_valid,
            "rejection_reason": reason,
        })

        if is_valid:
            save_geotiff(tile_arr, tile_aff, crs,
                         os.path.join(tiles_dir, f"{base}.tif"))
            save_jpeg(tile_arr, os.path.join(tiles_dir, f"{base}.jpg"))
            valid_count += 1
            if tile_num % 10 == 1 or tile_num == total:
                print(f"   [{tile_num:04d}/{total}] ✓ saved  {base}.jpg")
        else:
            rejected_count += 1
            if tile_num % 10 == 1 or tile_num == total:
                print(f"   [{tile_num:04d}/{total}] ✗ rejected ({reason})")

    csv_path = os.path.join(tiles_dir, "metadata.csv")
    generate_metadata_csv(metadata, csv_path)

    return {
        "total_tiles": total,
        "valid_tiles": valid_count,
        "rejected_tiles": rejected_count,
        "output_directory": tiles_dir,
        "metadata_csv": csv_path,
    }
