from shapely.geometry import Point

def create_bounding_box(lat, lon, buffer_deg):
    """Convert a lat/lon centre point into a WGS-84 square bounding box tuple."""
    print(f"\n🗺️  Calculating bounding box for [{lat}, {lon}]...")
    bbox = Point(lon, lat).buffer(buffer_deg).bounds  # (min_lon, min_lat, max_lon, max_lat)
    print(f"   ✓ BBox: [{bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f}]")
    return bbox
