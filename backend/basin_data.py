"""
Two-tier data definitions, now covering more calibrated basins, plus the
state-level lookup used by the state-scoped admin dashboards.

TIER A — "Calibrated" basins: real historical flood event + real (or
best-effort) terrain data. Numbers marked "placeholder" should be replaced
with actual CHIRPS/SRTM-derived values when available (see README) — this
was true for the original 3 basins and is equally honestly flagged for the
new ones added here.

TIER B — Everything else in India: no historical event to calibrate
against, so risk is instead compared to IMD's official rainfall intensity
classification, a real published national standard.

STATE_BBOXES / locate_state() — a deliberately simple rectangular lookup
used ONLY to give the admin dashboard state-level abstraction (so a
Kerala-only admin key can't see Assam's data) and to feature-weight the
risk model with each state's historical flood frequency. This is NOT
meant to be a precise administrative boundary — a real deployment would
use a proper GeoJSON state-boundary point-in-polygon check.
"""
import math

# --- IMD rainfall intensity classification (mm in 24h) ---
# Source: India Meteorological Department standard classification.
IMD_RAINFALL_CLASSES = {
    "light":            (0, 15.5),
    "moderate":         (15.6, 64.4),
    "heavy":            (64.5, 115.5),
    "very_heavy":       (115.6, 204.4),
    "extremely_heavy":  (204.5, float("inf")),
}

INDIA_BBOX = {"lat_min": 6.5, "lat_max": 35.7, "lon_min": 68.0, "lon_max": 97.5}
TIER_A_RADIUS_DEG = 0.6

# ---------------------------------------------------------------------------
# Calibrated basins. Each ties to ONE real, named historical flood event.
# 72h / 24h pre-flood rainfall figures for events other than 2015 Chennai
# and 2018 Kerala (which were already spot-checked against public
# reports) are labelled "placeholder" and should be swapped for real
# CHIRPS/IMD station totals before this is used for anything beyond a
# hackathon demo — see README "What's real vs placeholder".
# ---------------------------------------------------------------------------
BASINS = {
    "periyar_kerala": {
        "name": "Periyar River Basin (Aluva / Ernakulam, Kerala)",
        "state": "Kerala",
        "center_lat": 10.1075, "center_lon": 76.3516,
        "historical_event": {
            "name": "August 2018 Kerala Floods", "date": "2018-08-16",
            "pre_flood_72h_rainfall_mm": 310, "pre_flood_24h_rainfall_mm": 140,
            "data_quality": "spot-checked against public IMD/media reports",
        },
        "avg_elevation_m": 8, "drainage_density_index": 0.32,
        "flood_mask_asset": "periyar_2018_flood_mask.geojson",
        "safe_zones": [
            {"name": "Aluva Town Relief Shelter", "lat": 10.1090, "lon": 76.3550},
            {"name": "Kalamassery Community Hall", "lat": 10.0550, "lon": 76.3220},
        ],
    },
    "adyar_chennai": {
        "name": "Adyar-Cooum Basin (Chennai, Tamil Nadu)",
        "state": "Tamil Nadu",
        "center_lat": 13.0067, "center_lon": 80.2206,
        "historical_event": {
            "name": "December 2015 Chennai Floods", "date": "2015-12-01",
            "pre_flood_72h_rainfall_mm": 345, "pre_flood_24h_rainfall_mm": 200,
            "data_quality": "spot-checked against public IMD/media reports",
        },
        "avg_elevation_m": 6, "drainage_density_index": 0.21,
        "flood_mask_asset": "adyar_2015_flood_mask.geojson",
        "safe_zones": [{"name": "Adyar Cyclone Shelter", "lat": 13.0012, "lon": 80.2565}],
    },
    "rapti_gorakhpur": {
        "name": "West Rapti River Basin (Gorakhpur, Uttar Pradesh)",
        "state": "Uttar Pradesh",
        "center_lat": 26.7606, "center_lon": 83.3732,
        "historical_event": {
            "name": "August 2017 West Rapti / Terai Floods", "date": "2017-08-12",
            "pre_flood_72h_rainfall_mm": 280, "pre_flood_24h_rainfall_mm": 130,
            "data_quality": "placeholder — replace with real CHIRPS totals",
        },
        "avg_elevation_m": 77, "drainage_density_index": 0.28,
        "flood_mask_asset": "gorakhpur_2017_flood_mask.geojson",
        "safe_zones": [{"name": "Gorakhpur District Relief Camp", "lat": 26.7550, "lon": 83.3660}],
    },
    "brahmaputra_guwahati": {
        "name": "Brahmaputra Basin (Guwahati, Assam)",
        "state": "Assam",
        "center_lat": 26.1445, "center_lon": 91.7362,
        "historical_event": {
            "name": "July 2020 Assam Floods", "date": "2020-07-13",
            "pre_flood_72h_rainfall_mm": 260, "pre_flood_24h_rainfall_mm": 110,
            "data_quality": "placeholder — replace with real CHIRPS totals",
        },
        "avg_elevation_m": 49, "drainage_density_index": 0.35,
        "flood_mask_asset": "brahmaputra_2020_flood_mask.geojson",
        "safe_zones": [{"name": "Guwahati Relief Camp — Kamrup", "lat": 26.1500, "lon": 91.7000}],
    },
    "mithi_mumbai": {
        "name": "Mithi River Basin (Mumbai, Maharashtra)",
        "state": "Maharashtra",
        "center_lat": 19.0760, "center_lon": 72.8777,
        "historical_event": {
            "name": "July 2005 Mumbai Floods", "date": "2005-07-26",
            "pre_flood_72h_rainfall_mm": 700, "pre_flood_24h_rainfall_mm": 944,
            "data_quality": "spot-checked — 2005-07-26 is a well-documented 24h record",
        },
        "avg_elevation_m": 14, "drainage_density_index": 0.18,
        "flood_mask_asset": "mithi_2005_flood_mask.geojson",
        "safe_zones": [{"name": "Kurla Relief Shelter", "lat": 19.0726, "lon": 72.8845}],
    },
    "musi_hyderabad": {
        "name": "Musi River Basin (Hyderabad, Telangana)",
        "state": "Telangana",
        "center_lat": 17.3850, "center_lon": 78.4867,
        "historical_event": {
            "name": "October 2020 Hyderabad Floods", "date": "2020-10-13",
            "pre_flood_72h_rainfall_mm": 320, "pre_flood_24h_rainfall_mm": 191,
            "data_quality": "placeholder — replace with real CHIRPS totals",
        },
        "avg_elevation_m": 542, "drainage_density_index": 0.24,
        "flood_mask_asset": "musi_2020_flood_mask.geojson",
        "safe_zones": [{"name": "Musi Riverbank Relief Camp", "lat": 17.3800, "lon": 78.4950}],
    },
    "pamba_pandalam": {
        "name": "Pamba River Basin (Pandalam, Kerala)",
        "state": "Kerala",
        "center_lat": 9.2333, "center_lon": 76.6667,
        "historical_event": {
            "name": "August 2019 Kerala Floods", "date": "2019-08-09",
            "pre_flood_72h_rainfall_mm": 290, "pre_flood_24h_rainfall_mm": 125,
            "data_quality": "placeholder — replace with real CHIRPS totals",
        },
        "avg_elevation_m": 32, "drainage_density_index": 0.29,
        "flood_mask_asset": "pamba_2019_flood_mask.geojson",
        "safe_zones": [{"name": "Pandalam Relief Camp", "lat": 9.2300, "lon": 76.6700}],
    },
    "beas_kullu": {
        "name": "Beas River Basin (Kullu, Himachal Pradesh)",
        "state": "Himachal Pradesh",
        "center_lat": 31.9578, "center_lon": 77.1095,
        "historical_event": {
            "name": "July 2023 Himachal Flash Floods", "date": "2023-07-09",
            "pre_flood_72h_rainfall_mm": 240, "pre_flood_24h_rainfall_mm": 150,
            "data_quality": "placeholder — replace with real CHIRPS totals",
        },
        "avg_elevation_m": 1200, "drainage_density_index": 0.22,
        "flood_mask_asset": "beas_2023_flood_mask.geojson",
        "safe_zones": [{"name": "Kullu Relief Camp", "lat": 31.9600, "lon": 77.1100}],
    },
    "kosi_bihar": {
        "name": "Kosi River Basin (Supaul, Bihar)",
        "state": "Bihar",
        "center_lat": 26.1225, "center_lon": 86.5967,
        "historical_event": {
            "name": "August 2008 Kosi Floods (embankment breach)", "date": "2008-08-18",
            "pre_flood_72h_rainfall_mm": 230, "pre_flood_24h_rainfall_mm": 95,
            "data_quality": (
                "placeholder — this event was driven mainly by an embankment breach, "
                "not purely rainfall, and is kept as the deliberate safety-floor case "
                "where the pure statistical model under-predicted in testing"
            ),
        },
        "avg_elevation_m": 45, "drainage_density_index": 0.40,
        "flood_mask_asset": "kosi_2008_flood_mask.geojson",
        "safe_zones": [{"name": "Supaul Relief Camp", "lat": 26.1250, "lon": 86.6000}],
    },
    "damodar_bengal": {
        "name": "Damodar River Basin (Howrah/Hooghly, West Bengal)",
        "state": "West Bengal",
        "center_lat": 22.5726, "center_lon": 88.3639,
        "historical_event": {
            "name": "August 2021 South Bengal Floods (DVC release)", "date": "2021-08-01",
            "pre_flood_72h_rainfall_mm": 250, "pre_flood_24h_rainfall_mm": 105,
            "data_quality": "placeholder — replace with real CHIRPS totals",
        },
        "avg_elevation_m": 9, "drainage_density_index": 0.27,
        "flood_mask_asset": "damodar_2021_flood_mask.geojson",
        "safe_zones": [{"name": "Howrah Relief Camp", "lat": 22.5700, "lon": 88.3100}],
    },
}

# ---------------------------------------------------------------------------
# Rough state bounding boxes — used ONLY for (a) admin-key scoping and
# (b) the "historical flood frequency" feature below. Overlaps are resolved
# by first match in this dict's order (roughly smallest/most specific first).
# ---------------------------------------------------------------------------
STATE_BBOXES = {
    "Kerala":              (8.2, 12.8, 74.8, 77.4),
    "Tamil Nadu":          (8.0, 13.6, 76.2, 80.4),
    "Karnataka":           (11.5, 18.5, 74.0, 78.6),
    "Andhra Pradesh":      (12.6, 19.9, 76.7, 84.8),
    "Telangana":           (15.8, 19.9, 77.2, 81.3),
    "Goa":                 (14.8, 15.8, 73.6, 74.4),
    "Maharashtra":         (15.6, 22.1, 72.6, 80.9),
    "Gujarat":             (20.1, 24.7, 68.1, 74.5),
    "Rajasthan":           (23.0, 30.2, 69.5, 78.3),
    "Madhya Pradesh":      (21.0, 26.9, 74.0, 82.8),
    "Chhattisgarh":        (17.8, 24.1, 80.2, 84.4),
    "Odisha":              (17.8, 22.6, 81.4, 87.5),
    "West Bengal":         (21.5, 27.2, 85.8, 89.9),
    "Bihar":               (24.3, 27.5, 83.3, 88.1),
    "Jharkhand":           (21.9, 25.3, 83.3, 87.6),
    "Uttar Pradesh":       (23.8, 30.4, 77.0, 84.7),
    "Uttarakhand":         (28.7, 31.5, 77.5, 81.1),
    "Himachal Pradesh":    (30.2, 33.3, 75.5, 79.1),
    "Punjab":              (29.5, 32.6, 73.8, 76.9),
    "Haryana":             (27.6, 30.9, 74.5, 77.6),
    "Delhi":               (28.4, 28.9, 76.8, 77.4),
    "Jammu and Kashmir":   (32.0, 37.1, 73.0, 80.3),
    "Assam":               (24.1, 28.2, 89.7, 96.1),
    "Sikkim":              (27.0, 28.2, 88.0, 88.9),
    "Manipur":             (23.8, 25.7, 93.0, 94.8),
    "Meghalaya":           (25.0, 26.1, 89.8, 92.8),
    "Mizoram":             (21.9, 24.5, 92.2, 93.5),
    "Nagaland":            (25.2, 27.0, 93.3, 95.8),
    "Tripura":             (22.9, 24.6, 91.1, 92.4),
    "Arunachal Pradesh":   (26.6, 29.5, 91.5, 97.4),
}

# Approximate, honestly-labelled "recent decades" major-flood counts per
# state — used only as a coarse feature (states with a longer recent flood
# history get a small extra weight), not a precise statistic.
STATE_FLOOD_FREQUENCY = {
    "Kerala": 6, "Assam": 9, "Bihar": 8, "West Bengal": 6, "Uttar Pradesh": 5,
    "Tamil Nadu": 5, "Maharashtra": 5, "Telangana": 3, "Odisha": 5,
    "Himachal Pradesh": 4, "Uttarakhand": 4, "Karnataka": 3, "Gujarat": 3,
    "Andhra Pradesh": 4, "Jharkhand": 2, "Punjab": 2, "Haryana": 2,
    "Madhya Pradesh": 2, "Chhattisgarh": 2, "Rajasthan": 1, "Delhi": 2,
    "Goa": 1, "Sikkim": 2, "Manipur": 2, "Meghalaya": 2, "Mizoram": 1,
    "Nagaland": 1, "Tripura": 1, "Arunachal Pradesh": 1, "Jammu and Kashmir": 2,
}


def is_within_india(lat: float, lon: float) -> bool:
    return (
        INDIA_BBOX["lat_min"] <= lat <= INDIA_BBOX["lat_max"]
        and INDIA_BBOX["lon_min"] <= lon <= INDIA_BBOX["lon_max"]
    )


def find_tier(lat: float, lon: float):
    for key, basin in BASINS.items():
        d = ((lat - basin["center_lat"]) ** 2 + (lon - basin["center_lon"]) ** 2) ** 0.5
        if d <= TIER_A_RADIUS_DEG:
            return "A", key, basin
    return "B", None, None


def locate_state(lat: float, lon: float) -> str:
    """Coarse rectangular lookup — good enough for admin-dashboard
    abstraction and feature weighting, NOT for anything needing precise
    administrative boundaries."""
    for state, (lat_min, lat_max, lon_min, lon_max) in STATE_BBOXES.items():
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return state
    return "Other/Unmapped"


def state_flood_frequency(state: str) -> int:
    return STATE_FLOOD_FREQUENCY.get(state, 1)


def classify_rainfall_imd(mm_24h: float) -> str:
    for label, (lo, hi) in IMD_RAINFALL_CLASSES.items():
        if lo <= mm_24h <= hi:
            return label
    return "extremely_heavy"
