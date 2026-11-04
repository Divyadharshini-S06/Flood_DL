"""
Geo/terrain helpers that work for ANY coordinate in India (not just the
calibrated basins) — this is what makes Tier B (national coverage) possible.

Beyond elevation/slope/river-distance, this now also estimates:
  - impervious_surface_density: how "paved over" the area is (built-up /
    residential / industrial land nearby via OpenStreetMap) — a real driver
    of urban flooding independent of rainfall or drainage alone (e.g. 2005
    Mumbai, 2022 Bengaluru, 2020 Hyderabad were all worsened by unplanned
    urban surface runoff, not just rainfall totals).
  - distance_to_coast_km: coastal low-lying areas drain differently
    (tidal backwater effects) than inland river basins.
  - is_monsoon_season: India's June-September (+ NE monsoon Oct-Dec for the
    southeast coast) rainfall regime behaves differently than the dry season.
"""
import math
from datetime import datetime

import requests

OPEN_METEO_ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# A coarse set of reference points along India's coastline, used only to
# estimate rough distance-to-coast (NOT a precise coastline geometry).
COASTLINE_REFERENCE_POINTS = [
    (22.5, 88.3), (20.3, 86.6), (17.7, 83.3), (13.8, 80.3), (13.0, 80.3),
    (11.0, 79.8), (9.9, 78.1), (8.1, 77.5), (9.9, 76.3), (11.9, 75.4),
    (15.5, 73.8), (19.1, 72.9), (21.2, 72.9), (22.8, 69.9), (8.7, 76.9),
    (21.6, 87.9), (14.4, 80.0), (16.5, 82.2),
]


def get_elevation(lat: float, lon: float) -> float:
    try:
        r = requests.get(OPEN_METEO_ELEVATION_URL,
                          params={"latitude": lat, "longitude": lon}, timeout=5)
        r.raise_for_status()
        return r.json()["elevation"][0]
    except Exception:
        return 25.0


def get_slope(lat: float, lon: float, delta_deg: float = 0.01) -> float:
    try:
        center = get_elevation(lat, lon)
        north = get_elevation(lat + delta_deg, lon)
        east = get_elevation(lat, lon + delta_deg)
        dist_m = delta_deg * 111_000
        grad_ns = abs(north - center) / dist_m
        grad_ew = abs(east - center) / dist_m
        slope_pct = max(grad_ns, grad_ew) * 100
        return round(slope_pct, 2)
    except Exception:
        return 1.0


def get_distance_to_nearest_river_km(lat: float, lon: float, radius_m: int = 5000) -> float:
    query = f"""
    [out:json][timeout:10];
    (
      way["waterway"](around:{radius_m},{lat},{lon});
      way["natural"="water"](around:{radius_m},{lat},{lon});
    );
    out center 10;
    """
    try:
        r = requests.post(OVERPASS_URL, data={"data": query}, timeout=10)
        r.raise_for_status()
        elements = r.json().get("elements", [])
        if not elements:
            return round(radius_m / 1000, 2)
        best = float("inf")
        for el in elements:
            c = el.get("center")
            if not c:
                continue
            d = haversine_km(lat, lon, c["lat"], c["lon"])
            best = min(best, d)
        return round(best, 2) if best != float("inf") else round(radius_m / 1000, 2)
    except Exception:
        return 2.0


def get_impervious_surface_density(lat: float, lon: float, radius_m: int = 1500) -> float:
    """Fraction (0-1) of nearby mapped OSM land that is built-up/paved
    (residential, commercial, industrial, or a road) vs. permeable
    (farmland, forest, grass, water). Higher = less natural absorption of
    rainfall = higher runoff risk independent of rainfall totals alone.
    Falls back to a neutral 0.4 if Overpass is unreachable."""
    query = f"""
    [out:json][timeout:10];
    (
      way["landuse"~"residential|commercial|industrial|retail"](around:{radius_m},{lat},{lon});
      way["highway"](around:{radius_m},{lat},{lon});
      way["landuse"~"farmland|forest|grass|meadow|orchard"](around:{radius_m},{lat},{lon});
      way["natural"~"wood|wetland"](around:{radius_m},{lat},{lon});
    );
    out tags 200;
    """
    try:
        r = requests.post(OVERPASS_URL, data={"data": query}, timeout=10)
        r.raise_for_status()
        elements = r.json().get("elements", [])
        if not elements:
            return 0.4
        built, permeable = 0, 0
        for el in elements:
            tags = el.get("tags", {})
            landuse = tags.get("landuse")
            if landuse in ("residential", "commercial", "industrial", "retail"):
                built += 1
            elif "highway" in tags:
                built += 1
            elif landuse in ("farmland", "forest", "grass", "meadow", "orchard"):
                permeable += 1
            elif tags.get("natural") in ("wood", "wetland"):
                permeable += 1
        total = built + permeable
        if total == 0:
            return 0.4
        return round(built / total, 3)
    except Exception:
        return 0.4


def get_distance_to_coast_km(lat: float, lon: float) -> float:
    best = min(haversine_km(lat, lon, clat, clon) for clat, clon in COASTLINE_REFERENCE_POINTS)
    return round(best, 1)


def is_monsoon_season(dt: datetime | None = None) -> bool:
    """South-West monsoon (Jun-Sep) covers most of India; kept simple and
    documented rather than modelling the NE monsoon's separate Oct-Dec
    window for the southeast coast."""
    dt = dt or datetime.utcnow()
    return dt.month in (6, 7, 8, 9)


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))
