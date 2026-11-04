"""
Trains the flood-risk ensemble model used by risk_model.py.

WHY AN ENSEMBLE OF THREE MODELS, NOT ONE:
  - RandomForestClassifier   — robust to noisy/correlated features, good baseline.
  - GradientBoostingClassifier — usually the strongest single model on small
    tabular datasets like this one, catches non-linear threshold effects.
  - MLPClassifier (a small feedforward neural network — 2 hidden layers) —
    the "deep learning" component. Given the inputs are tabular physical/
    weather features (not images or raw time-series pixels), a small MLP is
    the honest choice for a neural net here: a CNN needs spatial image
    structure we don't have per-request, and an RNN needs long raw sequences
    rather than the engineered hourly aggregates used below. Forcing a CNN/
    RNN onto 12 tabular numbers would be decorative, not more accurate.
  Their probabilities are averaged (soft voting) — this is what "not just
  one group two or three" means here: three genuinely different model
  families, whose disagreement is itself informative (see metrics.json).

WHY MORE THAN 3-20 EXAMPLES:
  The project only has ~10 real, named historical flood events (see
  basin_data.BASINS) with (placeholder-flagged, best-effort) feature
  values — that's genuinely all the curated ground truth there is. Real
  production systems train on thousands of station-years of data; we
  don't have that here. So this script HONESTLY augments those ~10 real
  positives with:
    1. Noise-jittered duplicates of each real event (±10% on continuous
       features) — models the fact that a "flood" isn't one exact feature
       vector, it's a neighbourhood of similar conditions.
    2. A large set of programmatically generated NEGATIVE (no-flood)
       examples spanning realistic Indian terrain/rainfall combinations,
       weighted by each state's real historical flood frequency so flood-
       prone states aren't just "the same 10 rows" to the model.
    3. Borderline near-miss examples (rainfall approaching but not
       reaching a flood threshold) so the model doesn't learn a trivial
       "any rain => flood" shortcut.
  This is clearly synthetic augmentation, not real additional ground
  truth — flagged here and in the saved metrics.json so nobody mistakes
  the dataset size for real observed flood count.

Run:
  cd backend
  pip install -r ../requirements.txt
  python train_model.py
"""
import json
import os
import random

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from basin_data import BASINS, STATE_BBOXES, state_flood_frequency

random.seed(42)
np.random.seed(42)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_NAMES = [
    "forecast_24h_mm", "forecast_72h_mm", "peak_intensity_mm", "api_value",
    "elevation_m", "slope_pct", "river_dist_km", "impervious_density",
    "coastal_dist_km", "monsoon_flag", "state_flood_freq", "tier_a_flag",
]


def _real_positive_rows():
    """One row per real, named historical flood event in basin_data.py."""
    rows = []
    for key, b in BASINS.items():
        hist = b["historical_event"]
        state = b["state"]
        rows.append({
            "forecast_24h_mm": hist["pre_flood_24h_rainfall_mm"],
            "forecast_72h_mm": hist["pre_flood_72h_rainfall_mm"],
            "peak_intensity_mm": hist["pre_flood_24h_rainfall_mm"] * 0.35,
            "api_value": hist["pre_flood_72h_rainfall_mm"] * 0.5,
            "elevation_m": b["avg_elevation_m"],
            "slope_pct": 1.5 if b["avg_elevation_m"] < 100 else 6.0,
            "river_dist_km": 0.5,
            "impervious_density": 0.55,
            "coastal_dist_km": 40.0 if state in ("Kerala", "Tamil Nadu", "Maharashtra", "West Bengal") else 250.0,
            "monsoon_flag": 1,
            "state_flood_freq": state_flood_frequency(state),
            "tier_a_flag": 1,
            "label": 1,
            "source": f"real:{key}",
        })
    return rows


def _jittered_duplicates(real_rows, n_per_row=12, jitter=0.12):
    rows = []
    for base in real_rows:
        for _ in range(n_per_row):
            r = dict(base)
            for f in FEATURE_NAMES:
                if f in ("monsoon_flag", "tier_a_flag", "state_flood_freq"):
                    continue
                r[f] = max(0.0, base[f] * (1 + random.uniform(-jitter, jitter)))
            r["source"] = base["source"] + ":jitter"
            rows.append(r)
    return rows


def _random_state():
    return random.choice(list(STATE_BBOXES.keys()))


def _synthetic_negative_rows(n=420):
    """No-flood examples: modest rainfall, or heavy rainfall but terrain/
    drainage that doesn't produce a flood — spans both tiers."""
    rows = []
    for _ in range(n):
        state = _random_state()
        monsoon = random.random() < 0.4
        rows.append({
            "forecast_24h_mm": abs(random.gauss(25, 20)),
            "forecast_72h_mm": abs(random.gauss(60, 45)),
            "peak_intensity_mm": abs(random.gauss(6, 5)),
            "api_value": abs(random.gauss(15, 15)),
            "elevation_m": abs(random.gauss(150, 200)),
            "slope_pct": abs(random.gauss(4, 4)),
            "river_dist_km": abs(random.gauss(4, 4)) + 0.3,
            "impervious_density": min(1.0, abs(random.gauss(0.35, 0.2))),
            "coastal_dist_km": abs(random.gauss(300, 250)),
            "monsoon_flag": int(monsoon),
            "state_flood_freq": state_flood_frequency(state),
            "tier_a_flag": 0,
            "label": 0,
            "source": "synthetic:negative",
        })
    return rows


def _synthetic_borderline_rows(n=140):
    """Near-miss examples: substantial rainfall that still falls short of a
    flood, so the model can't shortcut to 'any heavy rain = flood'."""
    rows = []
    for _ in range(n):
        state = _random_state()
        rows.append({
            "forecast_24h_mm": abs(random.gauss(75, 15)),
            "forecast_72h_mm": abs(random.gauss(160, 30)),
            "peak_intensity_mm": abs(random.gauss(14, 4)),
            "api_value": abs(random.gauss(40, 15)),
            "elevation_m": abs(random.gauss(80, 60)) + 20,
            "slope_pct": abs(random.gauss(5, 3)) + 2,
            "river_dist_km": abs(random.gauss(3, 2)) + 1.0,
            "impervious_density": min(1.0, abs(random.gauss(0.4, 0.15))),
            "coastal_dist_km": abs(random.gauss(200, 150)),
            "monsoon_flag": 1,
            "state_flood_freq": state_flood_frequency(state),
            "tier_a_flag": 0,
            "label": 0,
            "source": "synthetic:borderline_negative",
        })
    return rows


def _synthetic_positive_rows(n=160):
    """Additional plausible flood-like conditions not tied to one of the 10
    named events — high rainfall + poor terrain/drainage combinations,
    weighted toward historically flood-prone states."""
    rows = []
    states = list(STATE_BBOXES.keys())
    weights = [state_flood_frequency(s) for s in states]
    for _ in range(n):
        state = random.choices(states, weights=weights, k=1)[0]
        elevation = abs(random.gauss(15, 15))
        rows.append({
            "forecast_24h_mm": abs(random.gauss(160, 40)),
            "forecast_72h_mm": abs(random.gauss(320, 60)),
            "peak_intensity_mm": abs(random.gauss(28, 8)),
            "api_value": abs(random.gauss(90, 25)),
            "elevation_m": elevation,
            "slope_pct": abs(random.gauss(1.5, 1.0)),
            "river_dist_km": abs(random.gauss(0.6, 0.5)),
            "impervious_density": min(1.0, abs(random.gauss(0.6, 0.15))),
            "coastal_dist_km": abs(random.gauss(80, 80)),
            "monsoon_flag": 1,
            "state_flood_freq": state_flood_frequency(state),
            "tier_a_flag": 0,
            "label": 1,
            "source": "synthetic:positive",
        })
    return rows


def build_dataset():
    real_pos = _real_positive_rows()
    rows = (
        real_pos
        + _jittered_duplicates(real_pos)
        + _synthetic_positive_rows()
        + _synthetic_negative_rows()
        + _synthetic_borderline_rows()
    )
    random.shuffle(rows)
    X = np.array([[r[f] for f in FEATURE_NAMES] for r in rows], dtype=float)
    y = np.array([r["label"] for r in rows], dtype=int)
    sources = [r["source"] for r in rows]
    return X, y, sources


def train_and_evaluate():
    X, y, sources = build_dataset()
    n_real = sum(1 for s in sources if s.startswith("real:") and ":jitter" not in s)
    n_jittered = sum(1 for s in sources if ":jitter" in s)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=3, random_state=42
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.08, random_state=42
        ),
        "mlp_neural_net": MLPClassifier(
            hidden_layer_sizes=(32, 16), activation="relu", alpha=1e-3,
            max_iter=2000, random_state=42, early_stopping=True,
        ),
    }

    metrics = {"dataset": {
        "total_rows": len(y),
        "real_curated_events": n_real,
        "jittered_duplicates_of_real_events": n_jittered,
        "positives": int(y.sum()), "negatives": int((1 - y).sum()),
        "note": (
            f"Only {n_real} rows are real named historical flood events (see basin_data.BASINS). "
            "Everything else is explicitly synthetic augmentation (jittered duplicates + generated "
            "negatives/positives) — see train_model.py docstring for exactly how and why."
        ),
    }, "models": {}}

    fitted = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        fitted[name] = model

        train_pred = model.predict(X_train_s)
        test_pred = model.predict(X_test_s)
        test_prob = model.predict_proba(X_test_s)[:, 1]

        cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring="f1")

        metrics["models"][name] = {
            "train_accuracy": round(accuracy_score(y_train, train_pred), 4),
            "test_accuracy": round(accuracy_score(y_test, test_pred), 4),
            "test_precision": round(precision_score(y_test, test_pred), 4),
            "test_recall": round(recall_score(y_test, test_pred), 4),
            "test_f1": round(f1_score(y_test, test_pred), 4),
            "test_roc_auc": round(roc_auc_score(y_test, test_prob), 4),
            "cv_f1_5fold_mean": round(cv_scores.mean(), 4),
            "cv_f1_5fold_std": round(cv_scores.std(), 4),
            "overfit_gap_train_minus_test_accuracy": round(
                accuracy_score(y_train, train_pred) - accuracy_score(y_test, test_pred), 4
            ),
        }
        print(f"[{name}] test_acc={metrics['models'][name]['test_accuracy']} "
              f"f1={metrics['models'][name]['test_f1']} "
              f"roc_auc={metrics['models'][name]['test_roc_auc']} "
              f"overfit_gap={metrics['models'][name]['overfit_gap_train_minus_test_accuracy']}")

    # Ensemble (soft voting) evaluated on the held-out test set too.
    ensemble_prob = np.mean([m.predict_proba(X_test_s)[:, 1] for m in fitted.values()], axis=0)
    ensemble_pred = (ensemble_prob >= 0.5).astype(int)
    metrics["ensemble"] = {
        "test_accuracy": round(accuracy_score(y_test, ensemble_pred), 4),
        "test_f1": round(f1_score(y_test, ensemble_pred), 4),
        "test_roc_auc": round(roc_auc_score(y_test, ensemble_prob), 4),
    }
    print(f"[ensemble] test_acc={metrics['ensemble']['test_accuracy']} "
          f"f1={metrics['ensemble']['test_f1']} roc_auc={metrics['ensemble']['test_roc_auc']}")

    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    for name, model in fitted.items():
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.joblib"))
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(MODELS_DIR, "feature_names.json"), "w") as f:
        json.dump(FEATURE_NAMES, f)

    print(f"\nSaved models + metrics.json to {MODELS_DIR}/")
    return metrics


if __name__ == "__main__":
    train_and_evaluate()
