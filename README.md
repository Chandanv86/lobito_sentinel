# Lobito Corridor Satellite Pipeline — v3

**5-source fallback chain for ≤2.5 m satellite imagery. Fully free. No Planet/Maxar/Airbus.**

---

| CRS type | CRS example | Coordinate names |
|---|---|---|
| **Geographic** (lat/lon) | `EPSG:4326` | `latitude`, `longitude` |
| **Projected** (metres) | `EPSG:32733` (UTM) | `x`, `y` |

v2 always passed `output_crs="EPSG:4326"` (geographic), then accessed `ds.coords["x"]` — which **does not exist** in that case. This crashed both Sentinel-2 (Source 4) and Sentinel-1 SAR (Source 5), exhausting the pipeline.

### All Bugs Fixed in v3

| # | Component | Bug in v2 | Fix in v3 |
|---|---|---|---|
| 1 | **S2 + SAR coord access** | `ds.coords["x"]` → KeyError when CRS is geographic | Load with `output_crs="EPSG:32733"` (projected UTM); helper `_get_spatial_coords()` is also defensive |
| 2 | **S2 resolution arg** | `resolution=0.0001` (decimal degrees — wrong unit for projected CRS) | `resolution=10` (metres — correct for UTM) |
| 3 | **SAR dB formula** | `10 * log10(DN)` — treats amplitude as power | `20 * log10(amplitude)` — GRD stores amplitude |
| 4 | **SAR ratio** | `vv_db / (vh_db + 1e-10)` — division in log-space is subtraction | `vv_db - vh_db` |
| 5 | **CBERS coord access** | Same `ds.coords["x"]` bug | Same UTM CRS fix + `_get_spatial_coords()` |
| 6 | **Nimbo mercantile call** | `mercantile.tiles(bbox[0], bbox[1], bbox[2], bbox[3], zoom)` — wrong signature | `mercantile.tiles(west, south, east, north, zooms=zoom)` |
| 7 | **Nimbo affine** | Confused min/max lat → upside-down images | Computed from actual `mercantile.bounds()` of tile grid |
| 8 | **CBERS STAC URL** | `https://data.inpe.br/bdc/stac/v1/` (wrong path) | `https://brazildatacube.dpi.inpe.br/stac/` + Scitekno mirror |
| 9 | **SAR resolution arg** | `resolution=0.0001` (degrees) | `resolution=10` (metres for UTM) |
| 10 | **Nimbo key check** | Missed empty-string case | `if not api_key or not api_key.strip()` |

---

## Source Hierarchy

```
[1] CBERS-4A WPM     ─ 2.0 m optical   ─ INPE BDC STAC (free, no key)
    ↓ (if cloudy / no scenes / INPE server down)
[2] Nimbo Earth      ─ 2.5 m AI-SR     ─ TMS API (free tier, NIMBO_API_KEY)
    ↓ (if key missing / credits exhausted)
[3] OpenAerialMap    ─ 0.3–1 m optical  ─ OAM STAC (free, no key)
    ↓ (if no OAM scenes for Angola — expected)
[4] Sentinel-2+EDSR  ─ ~2.5 m eff.     ─ Planetary Computer + DNN upscale
    ↓ (if cloudy)
[5] Sentinel-1 SAR   ─ 10 m radar      ─ Planetary Computer (all-weather)
```

---

## Setup

```bash
# 1. Virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
source .venv/bin/activate       # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Recommended) Set Nimbo API key for Source 2
# Windows PowerShell:
$env:NIMBO_API_KEY = "your_key_here"
# Windows CMD:
set NIMBO_API_KEY=your_key_here
# Mac/Linux:
export NIMBO_API_KEY=your_key_here
# Get free key at: https://nimbo.earth

# 4. Run pre-flight diagnostics
python diagnose.py

# 5. Run the pipeline
python main.py
```

---

## Project Structure

```
lobito_corridor/
├── main.py                  ← Orchestrator (run this)
├── config.py                ← All settings (lat/lon, dates, thresholds)
├── diagnose.py              ← Pre-flight connectivity checker
├── requirements.txt
├── EDSR_x4.pb               ← Auto-downloaded if missing (~150 MB)
└── src/
    ├── __init__.py
    ├── aoi_builder.py       ← BBox from lat/lon + buffer
    ├── fallback_fetcher.py  ← All 5 satellite sources (FIXED)
    └── tiler.py             ← Sliding-window chip extractor
```

---

## Output Structure

```
roboflow_training_data/
  SEG01_LOBITO_PORT/
    2026-05-15/
      tiles/
        tile_0001.tif   ← GeoTIFF with CRS + affine (for inference)
        tile_0001.jpg   ← JPEG for Roboflow upload
        tile_0002.tif
        tile_0002.jpg
        ...
        metadata.csv    ← lat/lon corners + validity per tile
```

---

## Why the Output Looks Like the Marina Reference Image

The target output — a nadir-view with distinct dock structures, vessel outlines,
and high contrast between dark water and light infrastructure — is exactly what:

- **CBERS-4A WPM 2 m** delivers: PCA-fused pan-sharpening gives crisp edges
- **Nimbo Earth 2.5 m** delivers: AI super-resolution from Sentinel-2 base
- **Sentinel-2+EDSR ~2.5 m** delivers: EDSR neural network 4× upscale

The Lobito Port AOI has a marina, dry dock, and container terminal — all
high-contrast features against the Atlantic, matching the reference image topology.

---

## Troubleshooting

**"CBERS-4A: all endpoints exhausted"**
- INPE BDC server has periodic downtime — try again in a few hours
- Angola is on the edge of CBERS-4A coverage (31-day revisit); widen `DATE_RANGE`
- Raise `MAX_CLOUD_COVER` to 40 in `config.py` for wet season (Nov–Apr)

**"NIMBO_API_KEY is not set"**
- Register free at https://nimbo.earth → copy key → set env var (see Setup above)
- Free tier = 4,000 geocredits/month

**"OAM STAC DNS/network unreachable"**
- Angola/Lobito is sparse in OpenAerialMap — this is expected and not an error
- The pipeline automatically falls through to Sentinel-2+EDSR

**"No Sentinel-2 scenes found"**
- Raise `MAX_CLOUD_COVER` to 40 or 50 in `config.py`
- Try dry-season range: `DATE_RANGE = "2024-05-01/2024-10-31"`

**"No Sentinel-1 scenes found"**
- Verify internet connectivity to `planetarycomputer.microsoft.com`
- Widen `DATE_RANGE` — Lobito is well-covered by Sentinel-1

---

## Annotation Guide for Roboflow

| Source | Resolution | Visual | Best Targets | Watch Out For |
|---|---|---|---|---|
| CBERS-4A | 2.0 m | Natural RGB | DFI infrastructure, roads, vessels | Cloud shadows ≠ water |
| Nimbo Earth | 2.5 m AI | Natural RGB | Static structures, deforestation | Moving objects erased |
| OpenAerialMap | 0.3–1 m | Natural RGB | Building footprints, vehicles | Sparse coverage |
| S2 + EDSR | ~2.5 m eff. | Natural RGB | Broad land-use, vegetation | EDSR softens thin edges |
| Sentinel-1 | ~10 m | False-colour | Ships, buildings, water edges | Speckle; bright=metallic |
