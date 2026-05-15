import os

# =============================================================================
# LOBITO CORRIDOR SATELLITE PIPELINE — CONFIG
# =============================================================================

# 1. Target Coordinates (Lobito Port, Angola)
LATITUDE        = 12.3380  # Adjusted slightly north for better port coverage
LONGITUDE       =  13.5450  # Adjusted slightly east for port center
BUFFER_DEGREES  =  0.025     # Increased slightly for better context (~2.7 km)

# 2. Date Range & Cloud Filters
DATE_RANGE      = "2024-05-01/2026-05-14"  # Focus on most recent dry season + current
MAX_CLOUD_COVER = 15.0          # Lowered for higher quality scenes

# 3. API Keys  (set as env vars — never hard-code secrets)
NIMBO_API_KEY   = "750bf73bb0f6cb9639286ea471b05e335dd4595ce1"

# 4. Output
DOWNLOAD_DIR    = "roboflow_training_data"
SEGMENT_ID      = "SEG01_LOBITO_PORT"

# 5. Tiling
TILE_SIZE       = 512
TILE_OVERLAP    = 0.5

# 6. Garbage-tile rejection thresholds
NODATA_THRESHOLD = 0.95
WATER_THRESHOLD  = 0.90
CLOUD_THRESHOLD  = 0.85

# 7. UTM zone for Lobito/Angola — EPSG:32733 (WGS 84 / UTM zone 33S)
#    Using a projected CRS avoids the odc-stac latitude/longitude coord-naming bug
UTM_CRS         = "EPSG:32733"
UTM_RESOLUTION  = 10   # metres — for S2 and SAR native resolution
UTM_RESOLUTION_CBERS = 2  # metres — CBERS-4A WPM fused
