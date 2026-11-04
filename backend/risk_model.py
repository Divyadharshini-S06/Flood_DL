"""
Risk scoring — hybrid of a physics/rules-based score (unchanged safety
floor logic) and a trained 3-model ensemble (RandomForest + Gradient
Boosting + a small MLP neural net; see train_model.py), combined by
taking the MORE CAUTIOUS of the two. This was a deliberate design choice
made after the pure statistical model alone under-predicted a real case
(Kosi/Bihar, 2008 — an embankment-breach flood that plain rainfall
thresholds miss) in testing — the rule-based floor exists specifically to
catch what the learned model might not.

Now also folds in factors beyond rainfall/drainage, per your note that
those two alone aren't a reliable basis: impervious-surface density
(urban runoff), distance to coast, monsoon-season flag, and each state's
historical flood frequency. See geo_utils.py / basin_data.py for how
each is computed.

Also implements hysteresis (severity only changes once the new score
meaningfully crosses a threshold, to avoid flicker) AND a real per-hour
severity series (`hourly_risk`) computed from a TRAILING 24h rainfall
window at each hour, not a cumulative-from-start-of-window sum. The
previous version's frontend computed its own per-hour colour by dividing
"rainfall so far" by "the currently visible window's total rainfall" —
which mathematically climbs toward 100% by the last hour shown no matter
how little rain actually fell, and is exactly why hours late in the strip
kept showing Severe/Extreme while the headline card correctly said Low.
Fix: the backend now computes each hour's severity from a real trailing
24h sum compared to the SAME reference threshold the headline uses, so
the hourly strip can never contradict the overall verdict.
"""
import os

import joblib
import numpy as np

from basin_data import classify_rainfall_imd

SEVERITY_ORDER = ["Low", "Moderate", "Severe", "Extreme"]
THRESHOLDS = {"Low": 0.0, "Moderate": 0.45, "Severe": 0.75, "Extreme": 1.0}
HYSTERESIS_BUFFER = 0.05

# Same probability cut points used for the ML ensemble's severity mapping,
# and for the per-hour trailing-window severity (both expressed as "how far
# past the reference threshold", so they stay on one consistent scale).
PCT_THRESHOLDS = {"Low": 0.0, "Moderate": 0.35, "Severe": 0.65, "Extreme": 0.9}

FEATURE_NAMES = [
    "forecast_24h_mm", "forecast_72h_mm", "peak_intensity_mm", "api_value",
    "elevation_m", "slope_pct", "river_dist_km", "impervious_density",
    "coastal_dist_km", "monsoon_flag", "state_flood_freq", "tier_a_flag",
]

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
_ENSEMBLE = None  # lazy-loaded: (scaler, {name: model})


def _load_ensemble():
    global _ENSEMBLE
    if _ENSEMBLE is not None:
        return _ENSEMBLE
    try:
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
        models = {
            name: joblib.load(os.path.join(MODELS_DIR, f"{name}.joblib"))
            for name in ("random_forest", "gradient_boosting", "mlp_neural_net")
        }
        _ENSEMBLE = (scaler, models)
    except Exception:
        # Models not trained yet — run `python train_model.py` in backend/.
        # The rule-based safety floor still works on its own without this.
        _ENSEMBLE = (None, None)
    return _ENSEMBLE


def ml_ensemble_probability(features: dict) -> float | None:
    scaler, models = _load_ensemble()
    if scaler is None:
        return None
    x = np.array([[features[f] for f in FEATURE_NAMES]], dtype=float)
    x_scaled = scaler.transform(x)
    probs = [m.predict_proba(x_scaled)[0, 1] for m in models.values()]
    return float(np.mean(probs))


def antecedent_precipitation_index(past_daily_rainfall_mm: list, decay: float = 0.9) -> float:
    return sum(r * (decay ** i) for i, r in enumerate(past_daily_rainfall_mm))


def rainfall_intensity_peak(hourly_forecast: list, window: int = 24) -> float:
    subset = hourly_forecast[:window]
    return max((h["precipitation"] for h in subset), default=0.0)


def _raw_score_tier_a(basin: dict, forecast_24h_mm: float, forecast_72h_mm: float,
                       peak_intensity_mm: float, api_value: float,
                       slope_pct: float, river_dist_km: float, impervious_density: float) -> float:
    hist = basin["historical_event"]
    rainfall_ratio = max(
        forecast_24h_mm / hist["pre_flood_24h_rainfall_mm"],
        forecast_72h_mm / hist["pre_flood_72h_rainfall_mm"],
    )
    elevation_factor = 1.15 if basin["avg_elevation_m"] < 10 else 1.0
    drainage_factor = 1.2 if basin["drainage_density_index"] < 0.3 else 1.0
    slope_factor = 1.1 if slope_pct < 2 or slope_pct > 8 else 1.0
    river_factor = 1.15 if river_dist_km < 1.0 else 1.0
    api_factor = 1.0 + min(api_value / 200, 0.25)
    intensity_factor = 1.1 if peak_intensity_mm > 20 else 1.0
    # Urban runoff: a heavily paved-over area sheds rainfall as surface
    # runoff faster than it can drain, independent of the rainfall total.
    impervious_factor = 1.0 + min(impervious_density * 0.2, 0.2)

    terrain_multiplier = elevation_factor * drainage_factor * slope_factor * river_factor * impervious_factor
    return rainfall_ratio * terrain_multiplier * api_factor * intensity_factor


def _raw_score_tier_b(forecast_24h_mm: float, elevation_m: float, slope_pct: float,
                       river_dist_km: float, api_value: float, peak_intensity_mm: float,
                       impervious_density: float) -> float:
    imd_very_heavy = 115.6
    rainfall_ratio = forecast_24h_mm / imd_very_heavy
    elevation_factor = 1.15 if elevation_m < 15 else 1.0
    slope_factor = 1.1 if slope_pct < 2 or slope_pct > 8 else 1.0
    river_factor = 1.15 if river_dist_km < 1.0 else 1.0
    api_factor = 1.0 + min(api_value / 200, 0.25)
    intensity_factor = 1.1 if peak_intensity_mm > 20 else 1.0
    impervious_factor = 1.0 + min(impervious_density * 0.2, 0.2)

    terrain_multiplier = elevation_factor * slope_factor * river_factor * impervious_factor
    return rainfall_ratio * terrain_multiplier * api_factor * intensity_factor


def score_to_severity(score: float) -> str:
    if score >= THRESHOLDS["Extreme"]:
        return "Extreme"
    if score >= THRESHOLDS["Severe"]:
        return "Severe"
    if score >= THRESHOLDS["Moderate"]:
        return "Moderate"
    return "Low"


def prob_to_severity(prob: float) -> str:
    if prob >= PCT_THRESHOLDS["Extreme"]:
        return "Extreme"
    if prob >= PCT_THRESHOLDS["Severe"]:
        return "Severe"
    if prob >= PCT_THRESHOLDS["Moderate"]:
        return "Moderate"
    return "Low"


def more_cautious(sev_a: str, sev_b: str) -> str:
    return sev_a if SEVERITY_ORDER.index(sev_a) >= SEVERITY_ORDER.index(sev_b) else sev_b


def apply_hysteresis(new_score: float, previous_severity: str | None) -> str:
    candidate = score_to_severity(new_score)
    if previous_severity is None or candidate == previous_severity:
        return candidate
    prev_idx = SEVERITY_ORDER.index(previous_severity)
    cand_idx = SEVERITY_ORDER.index(candidate)
    moving_up = cand_idx > prev_idx
    boundary_key = candidate if moving_up else previous_severity
    boundary = THRESHOLDS[boundary_key]
    buffer_ok = (new_score >= boundary + HYSTERESIS_BUFFER) if moving_up \
        else (new_score <= boundary - HYSTERESIS_BUFFER)
    return candidate if buffer_ok else previous_severity


def estimate_time_window(hourly_forecast: list, trigger_threshold_mm: float, severity: str) -> str:
    running = 0.0
    trigger_idx = None
    for i, h in enumerate(hourly_forecast[:24]):
        running += h["precipitation"]
        if running >= trigger_threshold_mm:
            trigger_idx = i
            break
    if trigger_idx is None:
        return "24-48h" if severity != "Low" else "no_significant_risk"
    if trigger_idx <= 6:
        return "0-6h"
    if trigger_idx <= 12:
        return "6-12h"
    return "12-24h"


def compute_hourly_risk(hourly_forecast: list, past_daily_rainfall_mm: list,
                         reference_threshold_24h_mm: float) -> list:
    """Per-hour severity using a TRAILING 24h rainfall window ending at
    each hour (carrying over yesterday's rainfall for the first few hours
    where fewer than 24 forecast hours exist yet), compared against the
    SAME reference threshold as the headline card. This can never diverge
    from the main verdict the way a cumulative-from-start sum could."""
    yesterday_total = past_daily_rainfall_mm[0] if past_daily_rainfall_mm else 0.0
    precip = [h["precipitation"] for h in hourly_forecast]
    out = []
    for i in range(len(hourly_forecast)):
        window_start = max(0, i - 23)
        forecast_part = sum(precip[window_start:i + 1])
        hours_covered = i - window_start + 1
        missing_hours = 24 - hours_covered
        carryover = yesterday_total * (missing_hours / 24) if missing_hours > 0 else 0.0
        trailing_24h_mm = forecast_part + carryover
        pct = trailing_24h_mm / reference_threshold_24h_mm if reference_threshold_24h_mm else 0.0
        sev = prob_to_severity(pct)
        out.append({
            "time": hourly_forecast[i]["time"],
            "severity": sev,
            "trailing_24h_mm": round(trailing_24h_mm, 1),
            "pct_of_reference": round(pct * 100, 1),
        })
    return out


EXPLANATION_I18N = {
    "en": {
        "template": (
            "Forecast rainfall over the next 24h is {mm}mm ({imd_class} by IMD classification), "
            "which is {pct}% of the reference level from {ref}. "
            "Antecedent rainfall (soil saturation proxy) adds further risk. "
            "Terrain factors: elevation {elevation}m, slope {slope}%, "
            "{river_dist}km from the nearest mapped waterway, "
            "{impervious}% built-up/paved surface nearby. "
            "The rule-based estimate and the trained model estimate were combined by "
            "taking the more cautious of the two. Combined estimated risk: {severity}."
        ),
        "imd_classes": {"light": "light", "moderate": "moderate", "heavy": "heavy",
                         "very_heavy": "very heavy", "extremely_heavy": "extremely heavy"},
        "severities": {"Low": "low", "Moderate": "moderate", "Severe": "severe", "Extreme": "extreme"},
    },
    "ta": {
        "template": (
            "அடுத்த 24 மணி நேரத்தில் எதிர்பார்க்கப்படும் மழையளவு {mm}மிமீ ({imd_class} — IMD வகைப்பாடு), "
            "இது {ref} நிகழ்வின் அளவின் {pct}% ஆகும். "
            "முந்தைய மழையளவு (மண் ஈரப்பதம்) கூடுதல் ஆபத்தை சேர்க்கிறது. "
            "நில அமைப்பு காரணிகள்: உயரம் {elevation}மீ, சரிவு {slope}%, "
            "அருகிலுள்ள நீர்நிலையிலிருந்து {river_dist}கிமீ தூரம், "
            "அருகில் {impervious}% கட்டப்பட்ட/மூடப்பட்ட மேற்பரப்பு. "
            "விதி-அடிப்படையிலான மதிப்பீடும் பயிற்சி பெற்ற மாடல் மதிப்பீடும் — எது கடுமையானதோ அதை எடுத்துக்கொள்ளப்பட்டது. "
            "மொத்த மதிப்பிடப்பட்ட ஆபத்து நிலை: {severity}."
        ),
        "imd_classes": {"light": "லேசான", "moderate": "மிதமான", "heavy": "கனமழை",
                         "very_heavy": "கடும் மழை", "extremely_heavy": "மிகக் கடும் மழை"},
        "severities": {"Low": "குறைவு", "Moderate": "மிதமானது", "Severe": "கடுமையானது", "Extreme": "மிக கடுமையானது"},
    },
    "hi": {
        "template": (
            "अगले 24 घंटों में अनुमानित वर्षा {mm}मिमी है ({imd_class} — IMD वर्गीकरण के अनुसार), "
            "जो {ref} की घटना के स्तर का {pct}% है। "
            "पिछली वर्षा (मिट्टी की नमी) जोखिम को और बढ़ाती है। "
            "भूभाग कारक: ऊंचाई {elevation}मी, ढलान {slope}%, "
            "निकटतम जल निकाय से {river_dist}किमी दूरी, "
            "आस-पास {impervious}% बना/पक्का क्षेत्र। "
            "नियम-आधारित अनुमान और प्रशिक्षित मॉडल अनुमान में से अधिक सतर्क वाले को चुना गया। "
            "कुल अनुमानित जोखिम: {severity}।"
        ),
        "imd_classes": {"light": "हल्की", "moderate": "मध्यम", "heavy": "भारी",
                         "very_heavy": "अत्यंत भारी", "extremely_heavy": "बेहद अत्यधिक भारी"},
        "severities": {"Low": "कम", "Moderate": "मध्यम", "Severe": "गंभीर", "Extreme": "अत्यधिक"},
    },
}


def build_explanation(lang, forecast_24h_mm, imd_class, pct_label, reference_label,
                       elevation_m, slope_pct, river_dist_km, impervious_density, severity):
    pack = EXPLANATION_I18N.get(lang, EXPLANATION_I18N["en"])
    return pack["template"].format(
        mm=f"{forecast_24h_mm:.0f}",
        imd_class=pack["imd_classes"].get(imd_class, imd_class),
        pct=pct_label,
        ref=reference_label,
        elevation=f"{elevation_m:.0f}",
        slope=f"{slope_pct:.1f}",
        river_dist=f"{river_dist_km:.1f}",
        impervious=f"{impervious_density * 100:.0f}",
        severity=pack["severities"].get(severity, severity.lower()),
    )


def compute_risk(tier: str, basin: dict | None, forecast_hourly: list,
                  elevation_m: float, slope_pct: float, river_dist_km: float,
                  past_daily_rainfall_mm: list, impervious_density: float = 0.4,
                  coastal_dist_km: float = 150.0, monsoon_flag: int = 0,
                  state_flood_freq: int = 1, previous_severity: str | None = None,
                  lang: str = "en"):
    next_24h = forecast_hourly[:24]
    next_72h = forecast_hourly[:72] if len(forecast_hourly) >= 72 else forecast_hourly
    forecast_24h_mm = sum(h["precipitation"] for h in next_24h)
    forecast_72h_mm = sum(h["precipitation"] for h in next_72h)
    peak_intensity = rainfall_intensity_peak(forecast_hourly)
    api_value = antecedent_precipitation_index(past_daily_rainfall_mm)

    if tier == "A":
        rule_score = _raw_score_tier_a(basin, forecast_24h_mm, forecast_72h_mm, peak_intensity,
                                        api_value, slope_pct, river_dist_km, impervious_density)
        reference_threshold_24h_mm = basin["historical_event"]["pre_flood_24h_rainfall_mm"]
        reference_label = basin["historical_event"]["name"]
        pct_label = round((forecast_24h_mm / reference_threshold_24h_mm) * 100, 1)
    else:
        rule_score = _raw_score_tier_b(forecast_24h_mm, elevation_m, slope_pct, river_dist_km,
                                        api_value, peak_intensity, impervious_density)
        reference_threshold_24h_mm = 115.6
        reference_label = "IMD 'very heavy rainfall' national threshold (115.6mm/24h)"
        pct_label = round((forecast_24h_mm / 115.6) * 100, 1)
    trigger_threshold = reference_threshold_24h_mm * 0.5

    rule_severity = apply_hysteresis(rule_score, previous_severity)

    ml_features = {
        "forecast_24h_mm": forecast_24h_mm, "forecast_72h_mm": forecast_72h_mm,
        "peak_intensity_mm": peak_intensity, "api_value": api_value,
        "elevation_m": elevation_m, "slope_pct": slope_pct, "river_dist_km": river_dist_km,
        "impervious_density": impervious_density, "coastal_dist_km": coastal_dist_km,
        "monsoon_flag": monsoon_flag, "state_flood_freq": state_flood_freq,
        "tier_a_flag": 1 if tier == "A" else 0,
    }
    ml_prob = ml_ensemble_probability(ml_features)
    ml_severity = prob_to_severity(ml_prob) if ml_prob is not None else None

    # The safety-floor design: take whichever estimate is more cautious.
    severity = more_cautious(rule_severity, ml_severity) if ml_severity else rule_severity

    time_window = estimate_time_window(forecast_hourly, trigger_threshold, severity)
    imd_class = classify_rainfall_imd(forecast_24h_mm)
    hourly_risk = compute_hourly_risk(forecast_hourly, past_daily_rainfall_mm, reference_threshold_24h_mm)

    explanation = build_explanation(
        lang, forecast_24h_mm, imd_class, pct_label, reference_label,
        elevation_m, slope_pct, river_dist_km, impervious_density, severity,
    )

    return {
        "severity": severity,
        "raw_score": round(rule_score, 3),
        "time_window": time_window,
        "forecast_24h_mm": round(forecast_24h_mm, 1),
        "forecast_72h_mm": round(forecast_72h_mm, 1),
        "peak_intensity_mm_per_hr": round(peak_intensity, 1),
        "antecedent_precipitation_index": round(api_value, 1),
        "imd_rainfall_class": imd_class,
        "pct_of_reference_threshold": pct_label,
        "reference_label": reference_label,
        "reference_threshold_24h_mm": reference_threshold_24h_mm,
        "explanation": explanation,
        "elevation_m": round(elevation_m, 1),
        "slope_pct": round(slope_pct, 1),
        "river_dist_km": round(river_dist_km, 1),
        "impervious_density": round(impervious_density, 2),
        "coastal_dist_km": round(coastal_dist_km, 1),
        "hourly_risk": hourly_risk,
        "model_ensemble": {
            "rule_based_severity": rule_severity,
            "ml_ensemble_severity": ml_severity,
            "ml_ensemble_probability": round(ml_prob, 3) if ml_prob is not None else None,
            "combination": "more-cautious-of-two (rule-based safety floor)",
            "ml_available": ml_prob is not None,
        },
    }
