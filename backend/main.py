"""
FloodXplain backend v3 - pan-India, two-tier (calibrated basin / national
general coverage), hybrid rule+ML risk model, state-scoped admin, and a
place+date historical evaluation tool.

Run:
  pip install -r ../requirements.txt
  python train_model.py        # train the ensemble once (see README)
  uvicorn main:app --reload --port 8000

Try:
  http://localhost:8000/risk?lat=10.10&lon=76.35   (Tier A - Kerala)
  http://localhost:8000/risk?lat=28.61&lon=77.20   (Tier B - Delhi, general coverage)
"""
import random
import time
from datetime import datetime, timedelta

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from basin_data import BASINS, find_tier, is_within_india, locate_state, state_flood_frequency
from geo_utils import (
    get_elevation, get_slope, get_distance_to_nearest_river_km,
    get_impervious_surface_density, get_distance_to_coast_km, is_monsoon_season,
)
from risk_model import compute_risk
from database import init_db
from auth_routes import router as auth_router

app = FastAPI(title="FloodXplain API v3")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(auth_router)
init_db()

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
GEOCODE_URL = "https://nominatim.openstreetmap.org/search"

CACHE_TTL_SECONDS = 30 * 60
_cache: dict = {}
_severity_history: dict = {}


def cache_key(lat: float, lon: float) -> str:
    return f"{round(lat, 2)}_{round(lon, 2)}"


def get_live_forecast(lat: float, lon: float):
    try:
        resp = requests.get(
            OPEN_METEO_URL,
            params={
                "latitude": lat, "longitude": lon,
                "hourly": "precipitation,windspeed_10m,cloudcover,temperature_2m",
                "past_days": 7, "forecast_days": 3, "timezone": "UTC",
            },
            timeout=6,
        )
        resp.raise_for_status()
        data = resp.json()["hourly"]
        hourly = [
            {"time": data["time"][i], "precipitation": data["precipitation"][i] or 0.0,
             "windspeed": data["windspeed_10m"][i], "cloudcover": data["cloudcover"][i],
             "temperature": data["temperature_2m"][i]}
            for i in range(len(data["time"]))
        ]
        return hourly, "live"
    except Exception:
        return generate_mock_forecast(), "mock (offline fallback)"


def generate_mock_forecast():
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(hours=168)
    hourly = []
    for i in range(168 + 72):
        hour_of_day = (now + timedelta(hours=i)).hour
        base_rain = max(0, 8 * pow(2.71828, -((hour_of_day - 14) ** 2) / 20))
        precipitation = round(base_rain + random.uniform(0, 2), 1)
        hourly.append({
            "time": (now + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M"),
            "precipitation": precipitation,
            "windspeed": round(random.uniform(5, 25), 1),
            "cloudcover": random.randint(20, 100),
            "temperature": round(27 + random.uniform(-3, 3), 1),
        })
    return hourly


def split_past_and_future(hourly):
    now_str = datetime.utcnow().strftime("%Y-%m-%dT%H:00")
    future = [h for h in hourly if h["time"] >= now_str]
    past = [h for h in hourly if h["time"] < now_str]
    daily_totals = []
    for day_start in range(len(past) - 24, -1, -24):
        day_hours = past[day_start:day_start + 24]
        daily_totals.append(sum(h["precipitation"] for h in day_hours))
    return daily_totals[:7], future


def gather_context_features(lat, lon, tier, basin):
    """Everything beyond rainfall/drainage that feeds the risk model:
    terrain, urban-surface, coastal proximity, season, and each state's
    historical flood frequency."""
    elevation_m = basin["avg_elevation_m"] if tier == "A" else get_elevation(lat, lon)
    slope_pct = get_slope(lat, lon)
    river_dist_km = get_distance_to_nearest_river_km(lat, lon)
    impervious_density = get_impervious_surface_density(lat, lon)
    coastal_dist_km = get_distance_to_coast_km(lat, lon)
    monsoon_flag = 1 if is_monsoon_season() else 0
    state = locate_state(lat, lon)
    flood_freq = state_flood_frequency(state)
    return {
        "elevation_m": elevation_m, "slope_pct": slope_pct, "river_dist_km": river_dist_km,
        "impervious_density": impervious_density, "coastal_dist_km": coastal_dist_km,
        "monsoon_flag": monsoon_flag, "state": state, "state_flood_freq": flood_freq,
    }


@app.get("/basins")
def list_basins():
    return BASINS


@app.get("/risk")
def get_risk(lat: float = Query(...), lon: float = Query(...), lang: str = "en", force_refresh: bool = False):
    if not is_within_india(lat, lon):
        raise HTTPException(status_code=400, detail="Location is outside the supported India coverage area.")

    key = f"{cache_key(lat, lon)}_{lang}"
    now = time.time()
    if not force_refresh and key in _cache:
        cached_time, cached_resp = _cache[key]
        if now - cached_time < CACHE_TTL_SECONDS:
            cached_resp["cache"] = {"hit": True, "age_seconds": int(now - cached_time)}
            return cached_resp

    tier, basin_key, basin = find_tier(lat, lon)
    forecast_hourly, source = get_live_forecast(lat, lon)
    past_daily, future_hourly = split_past_and_future(forecast_hourly)
    ctx = gather_context_features(lat, lon, tier, basin)

    previous_severity = _severity_history.get(cache_key(lat, lon))
    risk = compute_risk(
        tier=tier, basin=basin, forecast_hourly=future_hourly,
        elevation_m=ctx["elevation_m"], slope_pct=ctx["slope_pct"], river_dist_km=ctx["river_dist_km"],
        past_daily_rainfall_mm=past_daily, impervious_density=ctx["impervious_density"],
        coastal_dist_km=ctx["coastal_dist_km"], monsoon_flag=ctx["monsoon_flag"],
        state_flood_freq=ctx["state_flood_freq"], previous_severity=previous_severity, lang=lang,
    )
    _severity_history[cache_key(lat, lon)] = risk["severity"]

    response = {
        "location": {"lat": lat, "lon": lon},
        "state": ctx["state"],
        "confidence_tier": tier,
        "matched_basin": (
            {"key": basin_key, "name": basin["name"],
             "reference_event": basin["historical_event"]["name"],
             "safe_zones": basin.get("safe_zones", [])}
            if tier == "A" else None
        ),
        "forecast_source": source,
        "risk": risk,
        "hourly_preview": future_hourly[:12],
        "flood_mask_asset": basin["flood_mask_asset"] if tier == "A" else None,
        "last_updated": datetime.utcnow().isoformat(),
        "cache": {"hit": False, "age_seconds": 0},
    }
    _cache[key] = (now, response)
    return response


# ---------------------------------------------------------------------------
# DEMO / EVALUATION MODE
# ---------------------------------------------------------------------------
@app.get("/demo/basins")
def demo_basins():
    return {
        key: {
            "name": b["name"], "event_name": b["historical_event"]["name"],
            "event_date": b["historical_event"]["date"], "lat": b["center_lat"], "lon": b["center_lon"],
        }
        for key, b in BASINS.items()
    }


@app.get("/demo/geocode")
def demo_geocode(q: str):
    """Lets the frontend's evaluation panel turn a typed place name into
    lat/lon for the free-form place+date picker."""
    try:
        resp = requests.get(GEOCODE_URL, params={"format": "json", "q": f"{q}, India"},
                             headers={"User-Agent": "FloodXplain-Hackathon-Demo"}, timeout=8)
        resp.raise_for_status()
        results = resp.json()
        if not results:
            raise HTTPException(status_code=404, detail="Place not found")
        return {"lat": float(results[0]["lat"]), "lon": float(results[0]["lon"]), "display_name": results[0]["display_name"]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Geocoding failed: {e}")


def _run_historical_replay(lat: float, lon: float, event_date: datetime, tier: str, basin: dict | None, lang: str):
    start_date = (event_date - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = (event_date + timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        resp = requests.get(
            ARCHIVE_URL,
            params={"latitude": lat, "longitude": lon, "start_date": start_date, "end_date": end_date,
                    "hourly": "precipitation", "timezone": "UTC"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()["hourly"]
        hourly = [{"time": data["time"][i], "precipitation": data["precipitation"][i] or 0.0}
                  for i in range(len(data["time"]))]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Couldn't fetch historical archive data: {e}")

    future_hourly = [h for h in hourly if h["time"] >= (event_date - timedelta(days=3)).strftime("%Y-%m-%dT00:00")
                      and h["time"] < event_date.strftime("%Y-%m-%dT23:00")]
    past_hourly = [h for h in hourly if h["time"] < (event_date - timedelta(days=3)).strftime("%Y-%m-%dT00:00")]
    daily_totals = []
    for day_start in range(len(past_hourly) - 24, -1, -24):
        daily_totals.append(sum(h["precipitation"] for h in past_hourly[day_start:day_start + 24]))

    ctx = gather_context_features(lat, lon, tier, basin)
    risk = compute_risk(
        tier=tier, basin=basin, forecast_hourly=future_hourly,
        elevation_m=ctx["elevation_m"], slope_pct=ctx["slope_pct"], river_dist_km=ctx["river_dist_km"],
        past_daily_rainfall_mm=daily_totals[:7], impervious_density=ctx["impervious_density"],
        coastal_dist_km=ctx["coastal_dist_km"], monsoon_flag=1, state_flood_freq=ctx["state_flood_freq"],
        previous_severity=None, lang=lang,
    )
    return risk, future_hourly, ctx["state"]


@app.get("/demo/historical")
def demo_historical(basin_key: str, lang: str = "en"):
    """Quick shortcut: replay one of the pre-defined calibrated basins on
    its own real historical event date."""
    basin = BASINS.get(basin_key)
    if not basin:
        raise HTTPException(status_code=404, detail="Unknown basin key")
    event_date = datetime.strptime(basin["historical_event"]["date"], "%Y-%m-%d")
    risk, future_hourly, state = _run_historical_replay(
        basin["center_lat"], basin["center_lon"], event_date, "A", basin, lang
    )
    return {
        "historical_replay": True, "basin": {"key": basin_key, "name": basin["name"]},
        "event": basin["historical_event"], "state": state, "risk": risk,
        "hourly_preview": future_hourly[-24:] if len(future_hourly) >= 24 else future_hourly,
        "last_updated": datetime.utcnow().isoformat(),
    }


@app.get("/demo/evaluate")
def demo_evaluate(lat: float = Query(...), lon: float = Query(...), date: str = Query(...), lang: str = "en"):
    """Free-form evaluation: ANY India location + ANY date in roughly the
    past 20 years, replaying the real archived weather for that date
    through the model. This is what makes the evaluation tool a genuine
    place-and-calendar picker rather than being limited to the 10
    predefined basins."""
    if not is_within_india(lat, lon):
        raise HTTPException(status_code=400, detail="Location is outside the supported India coverage area.")
    try:
        event_date = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Date must be in YYYY-MM-DD format")

    earliest = datetime.utcnow() - timedelta(days=365 * 20)
    if event_date < earliest or event_date > datetime.utcnow():
        raise HTTPException(status_code=400, detail="Pick a date within roughly the past 20 years, not in the future.")

    tier, basin_key, basin = find_tier(lat, lon)
    risk, future_hourly, state = _run_historical_replay(lat, lon, event_date, tier, basin, lang)
    return {
        "historical_replay": True,
        "location": {"lat": lat, "lon": lon}, "state": state,
        "date": date, "confidence_tier": tier,
        "matched_basin": {"key": basin_key, "name": basin["name"]} if tier == "A" else None,
        "risk": risk,
        "hourly_preview": future_hourly[-24:] if len(future_hourly) >= 24 else future_hourly,
        "last_updated": datetime.utcnow().isoformat(),
    }
