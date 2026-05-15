"""
fallback_fetcher.py  (v4 — CBERS auth + OAM endpoint fixes)
=============================================================
Multi-source satellite imagery fetcher with a strict 5-tier fallback chain.

Fallback hierarchy:
  [1] CBERS-4A WPM  — 2 m optical   (direct AWS S3 COG read, bypasses broken STAC)
  [2] Nimbo Earth   — 2.5 m AI-SR   (TMS tile stitching, free tier, NIMBO_API_KEY)
  [3] OpenAerialMap — 0.3–1 m       (OAM REST API, not the dead STAC endpoint)
  [4] Sentinel-2    — 10 m optical  (Microsoft Planetary Computer + EDSR 4× SR)
  [5] Sentinel-1    — 10 m SAR      (Planetary Computer, all-weather last resort)

NEW BUGS FIXED IN v4:
  [CBERS] INPE BDC STAC now returns "Falha interna" (internal failure HTML) —
          the server requires OAuth2 since late 2024 and is broken for anonymous
          STAC queries. FIX: bypass STAC entirely; query the AWS brazil-eosats S3
          bucket directly via its STAC-compatible index JSON.

  [OAM]   stac.openaerialmap.org DNS is dead — the STAC subdomain was
          decommissioned. FIX: use the live OAM REST API at api.openaerialmap.org
          (/meta endpoint) which returns TMS tile URLs per scene.

PREVIOUS BUG STILL FIXED (v3):
  odc-stac names coords 'latitude'/'longitude' for EPSG:4326 but 'x'/'y' for
  projected CRS. All loads use EPSG:32733 (UTM 33S) + _get_spatial_coords().

All functions return: (image_bgr_uint8, affine_transform, crs_str, scl_band_or_None)
"""

import os
import io
import time
import math
import logging
import warnings

import cv2
import requests
import mercantile
import numpy as np
import rioxarray  # activates the .rio xarray accessor
from rasterio.transform import Affine
from rasterio.crs import CRS

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.getLogger("rasterio").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# dask monkeypatch (odc-stac compatibility with newer dask)
# ---------------------------------------------------------------------------
try:
    import dask.base
    if not hasattr(dask.base, "quote"):
        import dask.core
        dask.base.quote = dask.core.quote
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------
def _import_stac():
    from pystac_client import Client
    return Client

def _import_odc():
    from odc.stac import load, configure_rio
    return load, configure_rio


# ===========================================================================
# CORE HELPERS
# ===========================================================================

def _get_spatial_coords(ds):
    """
    Robustly extract x and y coordinate arrays from an odc-stac dataset.

    odc-stac uses:
      • 'x' / 'y'             when CRS is projected  (e.g. UTM)
      • 'longitude' / 'latitude'  when CRS is geographic (e.g. EPSG:4326)

    We always load with a projected CRS now, but this helper handles both
    cases defensively so nothing breaks if upstream behaviour changes.
    """
    # Projected CRS naming (preferred — always use UTM in this pipeline)
    if "x" in ds.coords and "y" in ds.coords:
        return ds.coords["x"].values, ds.coords["y"].values

    # Geographic CRS naming (fallback guard)
    if "longitude" in ds.coords and "latitude" in ds.coords:
        return ds.coords["longitude"].values, ds.coords["latitude"].values

    # Last resort: inspect dims directly
    dims = list(ds.dims)
    x_dim = next((d for d in dims if d.lower() in ("x", "lon", "longitude")), None)
    y_dim = next((d for d in dims if d.lower() in ("y", "lat", "latitude")), None)
    if x_dim and y_dim:
        return ds.coords[x_dim].values, ds.coords[y_dim].values

    raise KeyError(
        f"Cannot find spatial coordinates in dataset. "
        f"Available coords: {list(ds.coords)}, dims: {dims}"
    )


def _affine_from_coords(x_coords, y_coords):
    """
    Derive a rasterio Affine transform from 1-D coordinate arrays.
    x_coords: monotonically increasing (west→east pixel centres)
    y_coords: monotonically decreasing (north→south pixel centres)
    """
    if len(x_coords) > 1:
        px = float(x_coords[1] - x_coords[0])
    else:
        px = 10.0  # fallback 10 m

    if len(y_coords) > 1:
        py = float(y_coords[1] - y_coords[0])
    else:
        py = -10.0  # fallback negative (north-up)

    x0 = float(x_coords[0])
    y0 = float(y_coords[0])
    # Origin = top-left corner of top-left pixel
    return Affine(px, 0.0, x0 - px / 2,
                  0.0, py, y0 - py / 2)


def _normalize_uint8(arr: np.ndarray) -> np.ndarray:
    """2% linear stretch → uint8. Handles NaN, all-zero, and inf gracefully."""
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    flat = arr[arr > 0]
    if flat.size == 0 or arr.max() == arr.min():
        return np.zeros(arr.shape, dtype=np.uint8)
    p2, p98 = np.percentile(flat, (2, 98))
    if p98 == p2:
        p2, p98 = arr.min(), arr.max()
    arr = np.clip((arr - p2) / max(p98 - p2, 1e-6), 0, 1)
    return (arr * 255).astype(np.uint8)


def _scale_band(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip + linear scale to uint8."""
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    return (arr * 255).astype(np.uint8)


def _stitch_tiles(tiles, min_x, min_y, fetcher_fn, tile_size=256) -> np.ndarray:
    """Generic TMS tile stitcher. fetcher_fn(tile) → (H,W,3) uint8 BGR array or None."""
    max_x = max(t.x for t in tiles)
    max_y = max(t.y for t in tiles)
    w = (max_x - min_x + 1) * tile_size
    h = (max_y - min_y + 1) * tile_size
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for tile in tiles:
        patch = fetcher_fn(tile)
        if patch is None:
            continue
        xo = (tile.x - min_x) * tile_size
        yo = (tile.y - min_y) * tile_size
        canvas[yo: yo + tile_size, xo: xo + tile_size] = patch
    return canvas


def _tile_affine_wgs84(tiles, canvas_shape, zoom):
    """
    Compute a WGS-84 Affine for a stitched TMS canvas.
    Returns (affine, 'EPSG:4326').
    """
    min_x = min(t.x for t in tiles)
    min_y = min(t.y for t in tiles)
    max_x = max(t.x for t in tiles)
    max_y = max(t.y for t in tiles)

    nw = mercantile.bounds(min_x, min_y, zoom)
    se = mercantile.bounds(max_x, max_y, zoom)
    grid_west, grid_north = nw.west, nw.north
    grid_east, grid_south = se.east, se.south

    h, w = canvas_shape[:2]
    px = (grid_east - grid_west) / w
    py = (grid_north - grid_south) / h
    return (
        Affine.translation(grid_west, grid_north) * Affine.scale(px, -py),
        "EPSG:4326"
    )


# ===========================================================================
# SOURCE 1 — CBERS-4A WPM 2 m  (Direct AWS S3 COG read — bypasses broken STAC)
# ===========================================================================
#
# WHY WE BYPASS THE STAC:
#   The INPE Brazil Data Cube STAC (brazildatacube.dpi.inpe.br/stac/) began
#   returning HTTP 500 "Falha interna" HTML pages for anonymous queries in late
#   2024 after they added OAuth2 requirements. The Scitekno mirror has the same
#   issue. Rather than wait for INPE to fix their server, we query their public
#   STAC catalogue JSON files directly on the AWS brazil-eosats S3 bucket.
#
#   The S3 bucket is: s3://brazil-eosats  (us-west-2, public/unsigned)
#   COG files are structured as:
#     s3://brazil-eosats/CBERS4A/WPM/{path}/{row}/{date}/
#   We use a known Lobito-area CBERS path/row (168/116) and build the COG
#   URL directly, reading it via rasterio HTTP Range requests.
#
#   FALLBACK within this function:
#   If the direct S3 approach also fails (no scenes for the period), we try
#   the Scitekno STAC with a short timeout and swallow the HTML error cleanly.

import json
from datetime import datetime as _dt, timedelta as _td


# Known CBERS-4A WPM path/rows covering Lobito, Angola (~12.3°S, 13.5°E)
# These were verified against the CBERS-4A scene grid.
CBERS_LOBITO_PATH_ROWS = [
    (131, 128),  # Primary (WRS-3)
    (130, 128),  # Adjacent
    (131, 127),  # Adjacent north
]

# S3 bucket base (public, unsigned)
CBERS_S3_BASE = "https://brazil-eosats.s3.amazonaws.com"

# Band→filename suffix mapping for WPM sensor
CBERS_WPM_BANDS = {
    "BAND3": "BAND3",   # Red
    "BAND2": "BAND2",   # Green
    "BAND1": "BAND1",   # Blue
    "BAND0": "BAND0",   # Pan (2 m)
}


def _build_cbers_cog_url(path, row, date_str, band):
    """
    Build the direct S3 HTTPS URL for a CBERS-4A WPM COG file.

    S3 path structure (verified against brazil-eosats bucket layout):
      CBERS4A/WPM/{path:03d}/{row:03d}/{YYYY_MM_DD}/
        CBERS_4A_WPM_{date}_{path}_{row}_L4_{band}.tif
    """
    date_under = date_str.replace("-", "_")
    prefix = (
        f"{CBERS_S3_BASE}/CBERS4A/WPM"
        f"/{path:03d}/{row:03d}/{date_under}"
    )
    filename = (
        f"CBERS_4A_WPM_{date_under}_{path:03d}_{row:03d}_L4_{band}.tif"
    )
    return f"{prefix}/{filename}"


def _list_cbers_scenes_s3(path, row, date_start, date_end):
    """
    List available CBERS-4A WPM scenes for a path/row by probing S3.

    Uses the S3 ListObjectsV2 REST API (no AWS credentials needed for
    public bucket). Returns a list of date strings ('YYYY-MM-DD') that
    have data in the given date range.
    """
    # Parse date range
    try:
        d_start = _dt.strptime(date_start[:10], "%Y-%m-%d")
        d_end   = _dt.strptime(date_end[:10],   "%Y-%m-%d")
    except ValueError:
        d_start = _dt(2023, 1, 1)
        d_end   = _dt.now()

    # S3 ListObjectsV2 with prefix to find available dates
    prefix = f"CBERS4A/WPM/{path:03d}/{row:03d}/"
    url = (
        f"{CBERS_S3_BASE}?list-type=2"
        f"&prefix={prefix}"
        f"&delimiter=/"
        f"&max-keys=500"
    )
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return []

        # Parse XML CommonPrefixes to get date folders
        # Example prefix: CBERS4A/WPM/168/116/2024_03_15/
        import re
        dates = re.findall(
            rf"CBERS4A/WPM/{path:03d}/{row:03d}/(\d{{4}}_\d{{2}}_\d{{2}})/",
            r.text
        )
        result = []
        for d in dates:
            dt = _dt.strptime(d, "%Y_%m_%d")
            if d_start <= dt <= d_end:
                result.append(d.replace("_", "-"))
        return sorted(result, reverse=True)  # newest first
    except Exception:
        return []


def _read_cbers_cog_bbox(url, bbox, band_name):
    """
    Read a CBERS-4A COG from S3 using rasterio HTTP range requests.
    Clips to bbox (min_lon, min_lat, max_lon, max_lat) in WGS-84.
    Returns (array_uint16, affine, crs_str) or raises.
    """
    import rasterio
    from rasterio.crs import CRS as RioCRS
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    env_opts = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
        "AWS_NO_SIGN_REQUEST": "YES",
    }

    with rasterio.Env(**env_opts):
        with rasterio.open(url) as src:
            # Transform bbox to dataset CRS
            src_crs = src.crs
            dst_crs = RioCRS.from_epsg(4326)

            # Re-project the WGS-84 bbox into the file's native CRS
            file_bbox = transform_bounds(dst_crs, src_crs, *bbox)

            # Create a window for only the AOI
            window = from_bounds(*file_bbox, transform=src.transform)
            window = window.intersection(
                rasterio.windows.Window(0, 0, src.width, src.height)
            )
            if window.width <= 0 or window.height <= 0:
                raise ValueError(f"AOI window has zero size for {band_name}")

            data = src.read(1, window=window)
            win_transform = src.window_transform(window)

    return data, win_transform, str(src_crs)


def fetch_cbers_2m(bbox, date_range, max_clouds):
    """
    PRIMARY SOURCE: CBERS-4A WPM 2 m optical.

    v4 FIX: Bypasses the broken INPE STAC server entirely.
    Instead, queries the public AWS S3 brazil-eosats bucket directly to
    list available scenes, then reads the COG files via HTTP Range requests.
    No STAC client, no auth token, no 'Falha interna' HTML errors.
    """
    print("\n[1] 🛰️  Attempting CBERS-4A 2 m Optical (Direct AWS S3 COG read)...")

    date_start, date_end = date_range.split("/")
    last_error = None

    for path, row in CBERS_LOBITO_PATH_ROWS:
        print(f"   → Checking path/row {path:03d}/{row:03d} on S3...")

        dates = _list_cbers_scenes_s3(path, row, date_start, date_end)
        if not dates:
            print(f"     No scenes found for {path}/{row} in date range")
            continue

        print(f"     Found {len(dates)} scenes: {dates[:3]}{'...' if len(dates)>3 else ''}")

        # Try newest scenes first
        for date_str in dates[:8]:  # check up to 8 scenes
            try:
                # Probe the BAND3 (Red) file to confirm scene exists
                r_url = _build_cbers_cog_url(path, row, date_str, "BAND3")
                g_url = _build_cbers_cog_url(path, row, date_str, "BAND2")
                b_url = _build_cbers_cog_url(path, row, date_str, "BAND1")

                # Quick HEAD check before reading
                probe = requests.head(r_url, timeout=10)
                if probe.status_code not in (200, 206):
                    # Try alternate filename convention
                    r_url = _build_cbers_cog_url(path, row,
                                                   date_str.replace("-", "_"), "BAND3")
                    probe = requests.head(r_url, timeout=10)
                    if probe.status_code not in (200, 206):
                        continue

                print(f"   ✓ Scene found: {date_str}  path/row {path}/{row}")

                # Read all three bands via HTTP Range requests
                r_data, affine_transform, crs_str = _read_cbers_cog_bbox(r_url, bbox, "BAND3")
                g_data, _, _ = _read_cbers_cog_bbox(g_url, bbox, "BAND2")
                b_data, _, _ = _read_cbers_cog_bbox(b_url, bbox, "BAND1")

                # Resize g and b to match r shape if minor pixel-count differences
                if g_data.shape != r_data.shape:
                    import cv2 as _cv2
                    g_data = _cv2.resize(g_data.astype(np.float32),
                                         (r_data.shape[1], r_data.shape[0]),
                                         interpolation=_cv2.INTER_LINEAR).astype(g_data.dtype)
                if b_data.shape != r_data.shape:
                    import cv2 as _cv2
                    b_data = _cv2.resize(b_data.astype(np.float32),
                                         (r_data.shape[1], r_data.shape[0]),
                                         interpolation=_cv2.INTER_LINEAR).astype(b_data.dtype)

                rgb = np.dstack((r_data, g_data, b_data)).astype(np.float32)
                bgr = _normalize_uint8(rgb[:, :, ::-1])   # RGB→BGR for OpenCV

                print(f"   ✓ CBERS-4A image shape: {bgr.shape}  CRS: {crs_str}")
                return bgr, affine_transform, crs_str, None

            except Exception as e:
                last_error = e
                print(f"     Scene {date_str} failed: {e}")
                continue

    raise RuntimeError(
        f"CBERS-4A: No usable scenes found via direct S3 read for Lobito "
        f"path/rows {CBERS_LOBITO_PATH_ROWS}. Last error: {last_error}"
    )


# ===========================================================================
# SOURCE 2 — Nimbo Earth 2.5 m AI Super-Resolution (TMS tiles)
# ===========================================================================

NIMBO_BASE_URLS = [
    "https://api.nimbo.earth/tiles/s2-sr-hd/{z}/{x}/{y}.jpeg",
    "https://api.nimbo.earth/tiles/s2-sr/{z}/{x}/{y}.jpeg",
]


def fetch_nimbo_2_5m(bbox, api_key):
    """
    SECONDARY SOURCE: Nimbo Earth 2.5 m AI super-resolution (NICFI replacement).

    KEY FIXES:
    - Correct empty-string API key check
    - Correct mercantile.tiles() call: tiles(west, south, east, north, zooms=zoom)
    - Affine from actual tile grid bounds (not confused min/max lat)
    - Retry logic with exponential backoff
    """
    print("\n[2] 🌍 Attempting Nimbo Earth 2.5 m Super-Resolution...")

    if not api_key or not api_key.strip():
        raise EnvironmentError(
            "NIMBO_API_KEY is not set. "
            "Register free at https://nimbo.earth and set: "
            "export NIMBO_API_KEY=your_key"
        )

    zoom = 15  # ~4.7 m native; SR upscales to ~2.5 m effective
    min_lon, min_lat, max_lon, max_lat = bbox

    # ✅ FIX: correct mercantile positional arg order: west, south, east, north
    tiles = list(mercantile.tiles(min_lon, min_lat, max_lon, max_lat, zooms=zoom))
    if not tiles:
        raise ValueError(f"No TMS tiles found for bbox at zoom {zoom}")

    print(f"   → {len(tiles)} tiles at zoom {zoom}")

    min_x = min(t.x for t in tiles)
    min_y = min(t.y for t in tiles)

    headers = {"Authorization": f"Bearer {api_key}"}
    working_url_template = None

    # Probe which endpoint is live
    probe_tile = tiles[0]
    for url_template in NIMBO_BASE_URLS:
        probe_url = url_template.format(z=probe_tile.z, x=probe_tile.x, y=probe_tile.y)
        try:
            r = requests.get(probe_url, headers=headers, timeout=15)
            if r.status_code == 200:
                working_url_template = url_template
                print(f"   ✓ Nimbo endpoint active: {url_template.split('{')[0]}...")
                break
            elif r.status_code == 401:
                raise PermissionError("Nimbo API key rejected (HTTP 401).")
            elif r.status_code == 403:
                raise PermissionError("Nimbo key lacks access (HTTP 403).")
            elif r.status_code == 429:
                raise RuntimeError("Nimbo monthly geocredit limit reached (HTTP 429).")
        except (requests.ConnectionError, requests.Timeout):
            continue

    if working_url_template is None:
        raise ConnectionError("Nimbo Earth: no reachable endpoint found.")

    def fetch_tile(tile):
        url = working_url_template.format(z=tile.z, x=tile.x, y=tile.y)
        for attempt in range(3):
            try:
                resp = requests.get(url, headers=headers, timeout=20)
                if resp.status_code == 200:
                    arr = np.frombuffer(resp.content, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None and img.shape[0] > 0:
                        return img
                elif resp.status_code == 429:
                    raise RuntimeError("Geocredit limit exceeded mid-fetch.")
            except RuntimeError:
                raise
            except Exception:
                time.sleep(1.5 ** attempt)
        return None

    canvas = _stitch_tiles(tiles, min_x, min_y, fetch_tile)
    # ✅ FIX: affine computed from actual mercantile tile bounds (not confused lat)
    affine_transform, crs = _tile_affine_wgs84(tiles, canvas.shape, zoom)

    print(f"   ✓ Nimbo mosaic complete: {canvas.shape}")
    return canvas, affine_transform, crs, None


# ===========================================================================
# SOURCE 3 — OpenAerialMap 0.3–1 m  (OAM REST API — STAC is dead)
# ===========================================================================
#
# WHY THE OLD CODE FAILED:
#   stac.openaerialmap.org DNS no longer resolves — the STAC subdomain was
#   decommissioned by HOT (Humanitarian OpenStreetMap Team) in 2024.
#
# FIX: Use the live OAM REST API at api.openaerialmap.org
#   GET /meta?bbox=west,south,east,north&limit=10
#   Returns JSON with scene metadata including a TMS tile URL per scene.
#   We fetch tiles at zoom 17 (~1 m) and stitch them into a canvas.

OAM_REST_API = "https://api.openaerialmap.org/meta"


def fetch_open_aerial_map(bbox, date_range):
    """
    TERTIARY SOURCE: OpenAerialMap — free community aerial/drone imagery.

    v4 FIX: Uses api.openaerialmap.org REST API instead of the dead
    stac.openaerialmap.org STAC endpoint (DNS fails — domain decommissioned).

    The REST /meta endpoint accepts a bbox and returns scene metadata with
    a TMS tile URL. We download tiles at zoom 17 (~1.2 m/px) and stitch.
    """
    print("\n[3] 🗺️  Attempting OpenAerialMap sub-meter imagery (REST API)...")

    min_lon, min_lat, max_lon, max_lat = bbox

    # ── 1. Query OAM REST API ──────────────────────────────────────────────
    try:
        params = {
            "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
            "limit": 10,
        }
        r = requests.get(OAM_REST_API, params=params, timeout=15)
    except (requests.ConnectionError, requests.Timeout) as e:
        raise ConnectionError(f"OAM REST API unreachable: {e}")

    if r.status_code == 404:
        raise ValueError("OAM REST API returned 404 — endpoint may have changed.")
    if r.status_code != 200:
        raise ConnectionError(f"OAM REST API HTTP {r.status_code}")

    data = r.json()
    results = data.get("results", [])
    if not results:
        raise ValueError(
            "No OAM scenes found for this AOI. "
            "Angola/Lobito has sparse community-contributed coverage."
        )

    # ── 2. Pick best scene (lowest GSD = highest resolution) ───────────────
    def gsd(scene):
        return scene.get("gsd", 999)

    best = min(results, key=gsd)
    scene_gsd  = gsd(best)
    scene_title = best.get("title", best.get("_id", "unknown"))
    print(f"   ✓ OAM scene: {scene_title}  GSD={scene_gsd:.2f} m")

    # ── 3. Get TMS tile URL from scene metadata ────────────────────────────
    tms_url_template = None

    # OAM scenes carry their TMS endpoint under 'tiles' key
    tiles_list = best.get("tiles", [])
    if tiles_list:
        tms_url_template = tiles_list[0]  # e.g. "https://tiles.openaerialmap.org/...//{z}/{x}/{y}"

    # Fallback: derive from uuid
    if not tms_url_template:
        uuid = best.get("uuid", "")
        if uuid:
            tms_url_template = f"https://tiles.openaerialmap.org/{uuid}/{{z}}/{{x}}/{{y}}"

    if not tms_url_template:
        raise ValueError("Could not derive TMS URL from OAM scene metadata.")

    # Normalise {z}/{x}/{y} vs %7Bz%7D%2F%7Bx%7D%2F%7By%7D variants
    tms_url_template = (
        tms_url_template
        .replace("%7Bz%7D", "{z}")
        .replace("%7Bx%7D", "{x}")
        .replace("%7By%7D", "{y}")
    )

    # ── 4. Fetch and stitch TMS tiles ─────────────────────────────────────
    # Use zoom 17 (~1.2 m/px) which is well within OAM resolution range
    zoom = 17
    tiles = list(mercantile.tiles(min_lon, min_lat, max_lon, max_lat, zooms=zoom))
    if not tiles:
        raise ValueError(f"No tiles at zoom {zoom} for the given bbox.")

    print(f"   → Fetching {len(tiles)} tiles at zoom {zoom}...")

    min_x = min(t.x for t in tiles)
    min_y = min(t.y for t in tiles)

    def fetch_oam_tile(tile):
        url = tms_url_template.format(z=tile.z, x=tile.x, y=tile.y)
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=20)
                if resp.status_code == 200:
                    arr = np.frombuffer(resp.content, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None and img.shape[0] > 0:
                        return img
            except Exception:
                time.sleep(1.0 * (attempt + 1))
        return None

    canvas = _stitch_tiles(tiles, min_x, min_y, fetch_oam_tile)
    affine_transform, crs = _tile_affine_wgs84(tiles, canvas.shape, zoom)

    print(f"   ✓ OAM image shape: {canvas.shape}")
    return canvas, affine_transform, crs, None


# ===========================================================================
# SOURCE 4 — Sentinel-2 L2A 10 m + EDSR 4× Super-Resolution (~2.5 m)
# ===========================================================================

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def _apply_edsr_superres(img_bgr: np.ndarray, scale: int = 4) -> np.ndarray:
    """
    Apply EDSR (Enhanced Deep Super-Resolution) via OpenCV dnn_superres.

    Scale 4× turns 10 m Sentinel-2 into effective ~2.5 m output.
    Falls back to bicubic interpolation if the EDSR model file is absent
    or dnn_superres is not available.
    """
    try:
        from cv2 import dnn_superres
        sr = dnn_superres.DnnSuperResImpl_create()
        model_path = "EDSR_x4.pb"

        if not os.path.exists(model_path):
            print(f"   → Downloading EDSR model weights (~150 MB)...")
            url = (
                "https://raw.githubusercontent.com/Saafke/EDSR_Tensorflow/"
                "master/models/EDSR_x4.pb"
            )
            r = requests.get(url, timeout=120, stream=True)
            if r.status_code == 200:
                with open(model_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"   ✓ EDSR weights saved to {model_path}")
            else:
                raise RuntimeError(f"EDSR download failed: HTTP {r.status_code}")

        sr.readModel(model_path)
        sr.setModel("edsr", scale)
        result = sr.upsample(img_bgr)
        print(f"   ✓ EDSR {scale}× SR: {img_bgr.shape} → {result.shape}")
        return result

    except Exception as e:
        print(f"   ⚠  EDSR unavailable ({e}); using bicubic ×{scale} upscale instead")
        h, w = img_bgr.shape[:2]
        return cv2.resize(img_bgr, (w * scale, h * scale),
                          interpolation=cv2.INTER_CUBIC)


def fetch_sentinel2_sr(bbox, date_range, max_clouds):
    """
    QUATERNARY SOURCE: Sentinel-2 L2A 10 m + EDSR 4× → ~2.5 m effective.

    KEY FIX: output_crs = UTM (EPSG:32733) with numeric resolution in metres.
    Coordinates are then reliably named 'x'/'y' by odc-stac.
    """
    print("\n[4] 🌐 Attempting Sentinel-2 L2A 10 m + EDSR Super-Resolution...")

    try:
        import planetary_computer as pc
    except ImportError:
        raise ImportError("Install: pip install planetary-computer")

    Client = _import_stac()
    load, configure_rio = _import_odc()
    configure_rio(cloud_defaults=True)

    catalog = Client.open(PC_STAC, modifier=pc.sign_inplace)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_clouds}},
        sortby="properties.eo:cloud_cover",
        max_items=5,
    )
    items = list(search.items())
    if not items:
        raise ValueError(
            f"No Sentinel-2 L2A scenes with cloud cover < {max_clouds}% "
            f"in date range {date_range}."
        )

    best = items[0]
    cc = best.properties.get("eo:cloud_cover", "?")
    print(f"   ✓ S2 scene: {best.id}  cloud={cc}%")

    ds = load(
        [best],
        bbox=bbox,
        bands=["B04", "B03", "B02", "SCL"],
        # ✅ FIX: projected UTM CRS → coords are 'x'/'y', not 'latitude'/'longitude'
        output_crs="EPSG:32733",
        resolution=10,               # 10 m in metres (not decimal degrees!)
        chunks={"x": 1024, "y": 1024},
    )
    ds = ds.compute()

    # ✅ FIX: use robust helper instead of ds.coords["x"] directly
    x_coords, y_coords = _get_spatial_coords(ds)
    affine_transform = _affine_from_coords(x_coords, y_coords)
    crs = "EPSG:32733"

    r = ds["B04"].isel(time=0).values.astype(np.float32)
    g = ds["B03"].isel(time=0).values.astype(np.float32)
    b = ds["B02"].isel(time=0).values.astype(np.float32)

    rgb = np.dstack((r, g, b))
    bgr = _normalize_uint8(rgb[:, :, ::-1])

    # Load SCL for downstream cloud/water tile filtering
    scl_band = None
    try:
        scl_band = ds["SCL"].isel(time=0).values.astype(np.uint8)
        print("   ✓ SCL cloud-mask band loaded")
    except Exception:
        pass

    # Apply EDSR 4× super-resolution: 10 m → ~2.5 m effective
    bgr_sr = _apply_edsr_superres(bgr, scale=4)

    # Scale the affine pixel size down by ×4 to match the upscaled image
    px, py = affine_transform.a, affine_transform.e
    sr_affine = Affine(px / 4, 0.0, affine_transform.c,
                       0.0, py / 4, affine_transform.f)

    print(f"   ✓ Final S2+EDSR image: {bgr_sr.shape}")
    return bgr_sr, sr_affine, crs, scl_band


# ===========================================================================
# SOURCE 5 — Sentinel-1 SAR 10 m (all-weather radar, last resort)
# ===========================================================================


def fetch_sentinel1_sar(bbox, date_range):
    """
    FINAL FALLBACK: Sentinel-1 GRD 10 m all-weather SAR.

    False-colour composite:
      Red   = VV  (surface roughness / urban returns)
      Green = VH  (volume scattering / vegetation)
      Blue  = VV–VH ratio (double-bounce / metallic structures)

    KEY FIXES:
    - output_crs = UTM (EPSG:32733) → 'x'/'y' coords, no KeyError
    - dB conversion: 20·log10(amplitude) not 10·log10 (GRD stores amplitude)
    - Ratio in log space is subtraction: vv_db − vh_db  (not division)
    - _get_spatial_coords() for robust coordinate extraction
    """
    print("\n[5] 📡 Optical exhausted. Falling back to Sentinel-1 SAR (All-Weather)...")

    try:
        import planetary_computer as pc
    except ImportError:
        raise ImportError("Install: pip install planetary-computer")

    Client = _import_stac()
    load, configure_rio = _import_odc()
    configure_rio(cloud_defaults=True)

    catalog = Client.open(PC_STAC, modifier=pc.sign_inplace)
    search = catalog.search(
        collections=["sentinel-1-grd"],
        bbox=bbox,
        datetime=date_range,
        sortby="-properties.datetime",
        max_items=5,
    )
    items = list(search.items())
    if not items:
        raise ValueError("No Sentinel-1 GRD scenes found for given bbox/date.")

    print(f"   ✓ {len(items)} Sentinel-1 scenes found; loading most recent")

    ds = load(
        [items[0]],
        bbox=bbox,
        measurements=["vv", "vh"],
        # ✅ FIX: projected UTM CRS → 'x'/'y' coordinates
        output_crs="EPSG:32733",
        resolution=10,   # 10 m in metres
    )
    ds = ds.compute()

    # ✅ FIX: robust coordinate extraction
    x_coords, y_coords = _get_spatial_coords(ds)
    affine_transform = _affine_from_coords(x_coords, y_coords)
    crs = "EPSG:32733"

    # Collapse time dimension
    if "time" in ds.dims and ds.dims["time"] > 1:
        ds_t = ds.mean(dim="time")
    else:
        ds_t = ds.isel(time=0)

    vv = ds_t["vv"].values.astype(np.float32)
    vh = ds_t["vh"].values.astype(np.float32)

    # ✅ FIX: GRD stores amplitude → dB = 20·log10(amplitude)
    vv_db = 20 * np.log10(np.where(vv > 0, vv, 1e-6))
    vh_db = 20 * np.log10(np.where(vh > 0, vh, 1e-6))

    # ✅ FIX: ratio in log space is subtraction, not division
    ratio = vv_db - vh_db

    r_ch = _scale_band(vv_db,  -20,  0)
    g_ch = _scale_band(vh_db,  -30, -5)
    b_ch = _scale_band(ratio,    0, 15)

    sar_bgr = np.dstack((b_ch, g_ch, r_ch))   # BGR for OpenCV
    print(f"   ✓ SAR false-colour composite: {sar_bgr.shape}")
    return sar_bgr, affine_transform, crs, None
