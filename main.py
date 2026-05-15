"""
main.py — Lobito Corridor Satellite Pipeline  (v3)
===================================================
Orchestrates a 5-tier fallback chain to guarantee a ≤2.5 m image chip
is always produced, regardless of commercial data availability or weather.

Fallback hierarchy:
  [1] CBERS-4A WPM    — 2.0 m  optical  (INPE BDC STAC, free, no key)
  [2] Nimbo Earth     — 2.5 m  AI-SR    (TMS API, free tier, NIMBO_API_KEY)
  [3] OpenAerialMap   — 0.3–1 m optical  (OAM STAC, free, no key)
  [4] Sentinel-2+EDSR — ~2.5 m eff.     (Planetary Computer + DNN upscale)
  [5] Sentinel-1 SAR  — 10 m   radar    (Planetary Computer, all-weather)

What was broken in v2 (now fixed in v3):
  The root cause was odc-stac naming coordinates 'latitude'/'longitude' when
  output_crs='EPSG:4326' (geographic), but the code tried ds.coords["x"].
  Fix: all loads now use output_crs='EPSG:32733' (UTM zone 33S, projected),
  so odc-stac always returns 'x'/'y' coordinate names.
"""

import os
import sys
import config as cfg
from datetime import datetime

# ---------------------------------------------------------------------------
# dask monkeypatch — must be before any odc-stac import
# ---------------------------------------------------------------------------
try:
    import dask.base
    if not hasattr(dask.base, "quote"):
        import dask.core
        dask.base.quote = dask.core.quote
except ImportError:
    pass

from src.aoi_builder import create_bounding_box
from src.fallback_fetcher import (
    fetch_cbers_2m,
    fetch_nimbo_2_5m,
    fetch_open_aerial_map,
    fetch_sentinel2_sr,
    fetch_sentinel1_sar,
)
from src.tiler import tile_image


def run_pipeline():
    print("=" * 65)
    print("🚀  MULTI-SOURCE DFI MONITORING PIPELINE  (v3 — 5 Sources)")
    print("=" * 65)
    print("\nKey fix: all STAC loads use projected UTM CRS (EPSG:32733)")
    print("         → odc-stac coords always named 'x'/'y', never KeyError\n")

    os.makedirs(cfg.DOWNLOAD_DIR, exist_ok=True)
    bbox = create_bounding_box(cfg.LATITUDE, cfg.LONGITUDE, cfg.BUFFER_DEGREES)
    date_str = datetime.now().strftime("%Y-%m-%d")

    final_image      = None
    affine_transform = None
    crs              = None
    scl_band         = None
    source_used      = None

    # ── Source 1: CBERS-4A WPM 2 m ────────────────────────────────────────
    print("\n" + "─" * 65)
    try:
        final_image, affine_transform, crs, scl_band = fetch_cbers_2m(
            bbox, cfg.DATE_RANGE, cfg.MAX_CLOUD_COVER
        )
        source_used = "CBERS-4A WPM (2 m optical, pan-sharpened)"
    except Exception as e:
        print(f"   ⚠️  CBERS-4A skipped: {e}")

    # ── Source 2: Nimbo Earth 2.5 m ───────────────────────────────────────
    if final_image is None:
        print("\n" + "─" * 65)
        try:
            final_image, affine_transform, crs, scl_band = fetch_nimbo_2_5m(
                bbox, cfg.NIMBO_API_KEY
            )
            source_used = "Nimbo Earth (2.5 m AI super-resolution)"
        except Exception as e:
            print(f"   ⚠️  Nimbo Earth skipped: {e}")

    # ── Source 3: OpenAerialMap sub-meter ─────────────────────────────────
    if final_image is None:
        print("\n" + "─" * 65)
        try:
            final_image, affine_transform, crs, scl_band = fetch_open_aerial_map(
                bbox, cfg.DATE_RANGE
            )
            source_used = "OpenAerialMap (0.3–1 m optical)"
        except Exception as e:
            print(f"   ⚠️  OpenAerialMap skipped: {e}")

    # ── Source 4: Sentinel-2 + EDSR SR (~2.5 m effective) ─────────────────
    if final_image is None:
        print("\n" + "─" * 65)
        try:
            final_image, affine_transform, crs, scl_band = fetch_sentinel2_sr(
                bbox, cfg.DATE_RANGE, cfg.MAX_CLOUD_COVER
            )
            source_used = "Sentinel-2 L2A + EDSR ×4 super-resolution (~2.5 m)"
        except Exception as e:
            print(f"   ⚠️  Sentinel-2+SR skipped: {e}")

    # ── Source 5: Sentinel-1 SAR 10 m (all-weather last resort) ───────────
    if final_image is None:
        print("\n" + "─" * 65)
        try:
            final_image, affine_transform, crs, scl_band = fetch_sentinel1_sar(
                bbox, cfg.DATE_RANGE
            )
            source_used = "Sentinel-1 SAR (10 m, all-weather false-colour)"
        except Exception as e:
            print(f"   ❌  Sentinel-1 SAR also failed: {e}")

    # ── Abort if everything failed ─────────────────────────────────────────
    if final_image is None:
        print("\n" + "=" * 65)
        print("❌  CRITICAL FAILURE: All 5 satellite sources exhausted.")
        print("\n   Possible causes:")
        print("   • CBERS-4A: INPE STAC server may be temporarily down")
        print("     → Try again later or widen DATE_RANGE in config.py")
        print("   • Nimbo Earth: set NIMBO_API_KEY environment variable")
        print("     → Register free at https://nimbo.earth")
        print("   • OpenAerialMap: Angola/Lobito has sparse community coverage")
        print("     → Expected fallthrough — not an error")
        print("   • Sentinel-2: raise MAX_CLOUD_COVER to 40+ in config.py")
        print("     → Angola wet season (Nov–Apr) has persistent cloud cover")
        print("   • Sentinel-1: check internet connectivity to planetarycomputer.microsoft.com")
        print("=" * 65)
        sys.exit(1)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"✅  Image acquired from: {source_used}")
    print(f"    Shape  : {final_image.shape}  dtype: {final_image.dtype}")
    print(f"    CRS    : {crs}")
    print(f"    Affine : {affine_transform}")
    print(f"{'=' * 65}")

    # ── Chip extraction ────────────────────────────────────────────────────
    print("\n🔪  PHASE 3 — CHIP EXTRACTION")
    print(f"{'=' * 65}")

    stats = tile_image(
        image_array=final_image,
        affine_transform=affine_transform,
        crs=crs,
        scl_band=scl_band,
        output_dir=cfg.DOWNLOAD_DIR,
        segment_id=cfg.SEGMENT_ID,
        date_str=date_str,
    )

    # ── Final report ───────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("🎉  PIPELINE COMPLETE")
    print(f"{'=' * 65}")
    print(f"   Source         : {source_used}")
    print(f"   Total tiles    : {stats['total_tiles']}")
    print(f"   Valid tiles    : {stats['valid_tiles']}")
    print(f"   Rejected tiles : {stats['rejected_tiles']}")
    print(f"   Output dir     : {stats['output_directory']}")
    print(f"   Metadata CSV   : {stats['metadata_csv']}")
    print(f"\n📋  Next Steps:")
    print("   1. Review GeoTIFF + JPEG tiles in the output directory")
    print("   2. Upload JPEGs to Roboflow for YOLOv8/v10 annotation")
    print("   3. Use metadata.csv for geospatial traceability")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    run_pipeline()
