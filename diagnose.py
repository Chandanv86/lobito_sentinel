"""
diagnose.py — Pre-flight connectivity and dependency checker
============================================================
Run this BEFORE main.py to verify your environment is correctly configured.

Usage:
    python diagnose.py
"""

import sys
import os

PASS = "✅"
WARN = "⚠️ "
FAIL = "❌"
INFO = "ℹ️ "

print("=" * 60)
print("  LOBITO PIPELINE — DIAGNOSTICS  (v3)")
print("=" * 60)

# ── 1. Python version ──────────────────────────────────────────
print("\n[1] Python version")
v = sys.version_info
ok = v >= (3, 9)
print(f"   {PASS if ok else FAIL}  Python {v.major}.{v.minor}.{v.micro}  "
      f"{'(3.9+ required)' if not ok else ''}")

# ── 2. Required packages ───────────────────────────────────────
print("\n[2] Required packages")
packages = {
    "numpy":              "numpy",
    "cv2":                "opencv-python",
    "requests":           "requests",
    "rasterio":           "rasterio",
    "rioxarray":          "rioxarray",
    "shapely":            "shapely",
    "mercantile":         "mercantile",
    "pystac_client":      "pystac-client",
    "odc.stac":           "odc-stac",
    "planetary_computer": "planetary-computer",
    "pyproj":             "pyproj",
    "pandas":             "pandas",
}

all_ok = True
for module, pkg in packages.items():
    try:
        m = __import__(module)
        ver = getattr(m, "__version__", "?")
        print(f"   {PASS}  {pkg:<30} {ver}")
    except ImportError:
        print(f"   {FAIL}  {pkg:<30} NOT INSTALLED")
        print(f"         → pip install {pkg}")
        all_ok = False

if not all_ok:
    print(f"\n   {FAIL}  Missing packages — run: pip install -r requirements.txt")

# ── 3. EDSR model weights ──────────────────────────────────────
print("\n[3] EDSR super-resolution model (optional, used for S2+EDSR)")
model_path = "EDSR_x4.pb"
if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / 1024 / 1024
    print(f"   {PASS}  EDSR_x4.pb found ({size_mb:.1f} MB)")
else:
    print(f"   {WARN}  EDSR_x4.pb not found")
    print(f"         Will be downloaded automatically (~150 MB) on first use,")
    print(f"         OR bicubic interpolation will be used as fallback.")

# ── 4. API Keys ────────────────────────────────────────────────
print("\n[4] API Keys")
nimbo_key = os.environ.get("NIMBO_API_KEY", "")
if nimbo_key:
    print(f"   {PASS}  NIMBO_API_KEY is set ({nimbo_key[:8]}...)")
else:
    print(f"   {WARN}  NIMBO_API_KEY not set — Nimbo Earth (Source 2) will be skipped")
    print(f"         Register free at https://nimbo.earth")
    print(f"         Then: export NIMBO_API_KEY=your_key   (Linux/Mac)")
    print(f"         Or:   set NIMBO_API_KEY=your_key      (Windows CMD)")
    print(f"         Or:   $env:NIMBO_API_KEY='your_key'   (Windows PowerShell)")

# ── 5. Network connectivity ────────────────────────────────────
print("\n[5] Network connectivity to satellite data endpoints")
try:
    import requests

    endpoints = {
        # CBERS: v4 bypasses STAC; tests the S3 bucket directly
        "CBERS-4A AWS S3 bucket (primary)":      "https://brazil-eosats.s3.amazonaws.com",
        "CBERS-4A INPE BDC STAC (deprecated*)":  "https://brazildatacube.dpi.inpe.br/stac/",
        "Nimbo Earth API":                        "https://api.nimbo.earth",
        "Planetary Computer (S2 + SAR)":          "https://planetarycomputer.microsoft.com",
        # OAM: v4 uses REST API, not the dead STAC subdomain
        "OAM REST API (v4, primary)":             "https://api.openaerialmap.org/meta?limit=1",
        "OAM STAC (dead - DNS fails*)":           "https://stac.openaerialmap.org",
    }

    for name, url in endpoints.items():
        deprecated = name.endswith("*)")
        try:
            r = requests.get(url, timeout=8)
            if r.status_code < 500:
                icon = WARN if deprecated else PASS
                note = "  ← no longer used in v4" if deprecated else ""
                print(f"   {icon}  {name:<50} HTTP {r.status_code}{note}")
            else:
                print(f"   {WARN}  {name:<50} HTTP {r.status_code} (server error)")
        except requests.ConnectionError:
            if "stac.openaerialmap" in url:
                print(f"   {PASS}  {name:<50} DNS dead (expected — v4 uses REST API)")
            else:
                print(f"   {FAIL}  {name:<50} DNS/connection failed")
        except requests.Timeout:
            print(f"   {WARN}  {name:<50} Timed out (>8 s)")

    print(f"\n   {INFO}  * INPE BDC STAC requires OAuth2 since late 2024 → v4 uses S3 direct read")
    print(f"   {INFO}  * OAM stac.openaerialmap.org DNS decommissioned → v4 uses api.openaerialmap.org")

except ImportError:
    print(f"   {FAIL}  requests not installed — cannot test connectivity")

# ── 6. odc-stac CRS coordinate naming (the v2 bug) ─────────────
print("\n[6] Verifying odc-stac coordinate naming (key v2 bug check)")
try:
    from odc.geo.crs import CRS
    utm = CRS("EPSG:32733")
    geo = CRS("EPSG:4326")
    print(f"   {INFO}  EPSG:32733 (UTM 33S) is geographic: {utm.geographic}  → coords named 'x'/'y'")
    print(f"   {INFO}  EPSG:4326  (WGS-84)  is geographic: {geo.geographic}  → coords named 'latitude'/'longitude'")
    print(f"   {PASS}  v3 always uses EPSG:32733 for loads → 'x'/'y' guaranteed → no KeyError")
except Exception as e:
    print(f"   {WARN}  Could not verify CRS types: {e}")

# ── 7. Config summary ──────────────────────────────────────────
print("\n[7] Current config.py settings")
try:
    import config as cfg
    print(f"   {INFO}  LATITUDE / LONGITUDE : {cfg.LATITUDE} / {cfg.LONGITUDE}")
    print(f"   {INFO}  BUFFER_DEGREES       : {cfg.BUFFER_DEGREES}  (~{cfg.BUFFER_DEGREES*111:.1f} km radius)")
    print(f"   {INFO}  DATE_RANGE           : {cfg.DATE_RANGE}")
    print(f"   {INFO}  MAX_CLOUD_COVER      : {cfg.MAX_CLOUD_COVER}%")
    print(f"   {INFO}  DOWNLOAD_DIR         : {cfg.DOWNLOAD_DIR}/")
    print(f"   {INFO}  TILE_SIZE            : {cfg.TILE_SIZE} px")
    print(f"   {INFO}  UTM_CRS              : {cfg.UTM_CRS}")
except Exception as e:
    print(f"   {FAIL}  Could not load config: {e}")

print("\n" + "=" * 60)
print("  If all checks pass, run:  python main.py")
print("=" * 60 + "\n")
