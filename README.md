# FloodXplain

**Tagline:** *See the flood before the water does.*

A flood early-warning app: enter a location anywhere in India, get an
hourly weather strip plus an hourly **flood risk** strip with a
plain-language explanation of *why*, based on a hybrid rule-based +
trained-model risk score.

## What's fixed / new in this version

- **Login is mandatory every session again.** The previous build silently
  auto-restored your last session from `localStorage`, which is why the
  sign-in screen stopped appearing on repeat visits. That auto-restore is
  now removed — `frontend/index.html` always shows the login gate on open.
- **The hourly risk strip no longer contradicts the headline card.** The
  "4-15" you saw in the strip were just hour-of-day labels (4am–3pm) — the
  real bug was that the *frontend* computed each hour's colour by dividing
  "rainfall so far" by "this window's own total rainfall," a ratio that
  mathematically climbs toward 100% by the last hour no matter how little
  rain actually fell. The backend now computes a real trailing-24h-window
  severity per hour (`risk_model.compute_hourly_risk`) against the same
  reference threshold the headline card uses, and the frontend just
  displays it — it can't diverge again.
- **The map is working as intended** — Esri's free World Dark Gray Base
  tiles, no key required, no "403"/"API key required" watermark. If it
  still looks sparse in your area, that's Esri's own tile coverage/detail
  level at that zoom, not an app error.
- **Admin dashboard now has real state-scoped abstraction.** Every state
  gets its own admin key (`backend/admin_keys.py`) plus one master key.
  A state key sees *only* that state's usage and risk data — enforced
  server-side, not by hiding UI. The master key sees the national totals,
  a per-state comparison chart, and the full user list. Charts (via
  Chart.js): a stock-market-style usage-over-time line, a severity-split
  doughnut, and (master only) a per-state bar comparison.
- **Risk model is now a hybrid ensemble, not one rule-based formula.**
  Three independently-trained models (Random Forest, Gradient Boosting, a
  small feedforward neural net) vote alongside the original rules-based
  score; the more cautious of the two is used (unchanged design principle
  from before — this is the same safety floor that was added after the
  pure statistical model under-predicted the Kosi/Bihar case in testing).
  See `backend/train_model.py` and `backend/models/metrics.json` for the
  full evaluation (train/test split + 5-fold cross-validation, so you can
  see it isn't overfit).
- **More than rainfall + drainage now feed the score:** built-up/paved
  surface density nearby (urban runoff), distance to the coast, monsoon-
  season flag, and each state's historical flood frequency — see
  `geo_utils.py` and `basin_data.py`.
- **10 calibrated historical basins now, not 3** — Kerala (2), Chennai,
  Gorakhpur, Assam/Guwahati, Mumbai, Hyderabad, Himachal, Bihar/Kosi, and
  West Bengal. See the honesty table below for which numbers are real vs.
  best-effort placeholders.
- **The evaluation tool is now a real place + calendar picker**, not
  limited to the 10 basins — type any India location and pick any date in
  the past ~20 years, and it replays the real archived weather for that
  exact date/place through the model (`GET /demo/evaluate`). The 10 basins
  are still available as one-click shortcuts.

## Run it locally

```bash
cd backend
pip install -r ../requirements.txt
python train_model.py          # trains the 3-model ensemble, ~a few seconds
uvicorn main:app --reload --port 8000
```

Then open `frontend/index.html` in a browser (or serve the `frontend/`
folder with `python3 -m http.server 5500` and visit
`http://localhost:5500`). You'll see the sign-in screen every time you
open it — use "Continue as Guest" to test immediately.

If you skip `python train_model.py`, the app still works — it just falls
back to the rule-based score alone (`model_ensemble.ml_available: false`
in the API response) until you train it.

## Admin dashboards

```bash
# Optional — override any of these before your demo; defaults live in
# backend/admin_keys.py and work out of the box for local testing.
export FLOODXPLAIN_ADMIN_KEY_KERALA="your-real-secret"
export FLOODXPLAIN_MASTER_ADMIN_KEY="your-real-master-secret"
```

Open `frontend/admin.html` directly (not linked anywhere in the main app)
and enter either:
- **A state key** (see `backend/admin_keys.py` for the full list) → see
  only that state's usage, severity split, and recent risk checks.
- **The master key** → see national totals, every state's breakdown, the
  usage-over-time chart, and the full user list.

## Set up Google Sign-In (optional — guest mode works without this)

1. Go to **https://console.cloud.google.com/** and create a new project (free)
2. **APIs & Services → OAuth consent screen** → "External" → fill in the
   app name (FloodXplain) and your email → save
3. **APIs & Services → Credentials → Create Credentials → OAuth client ID**
4. Application type: **Web application**
5. Under "Authorized JavaScript origins," add the URL you'll open the app
   from (e.g. `http://localhost:5500`)
6. Copy the generated **Client ID**
7. In `frontend/index.html`, paste it into `const GOOGLE_CLIENT_ID = "..."`
8. On the backend: `export FLOODXPLAIN_GOOGLE_CLIENT_ID="123456-abc.apps.googleusercontent.com"`

## What's real vs. still a placeholder

| Piece | Status |
|---|---|
| Live hourly weather | **Real** — Open-Meteo |
| Basin matching (Tier A calibrated regions) | **Real logic**, 10 basins defined |
| Historical flood rainfall thresholds (Tier A) | **Mixed** — 2015 Chennai and 2005 Mumbai are spot-checked against public reports; the other 8 are honestly-flagged best-effort placeholders (see `data_quality` field per basin in `basin_data.py`) — replace with real CHIRPS-derived values before anything beyond a demo |
| National rainfall thresholds (Tier B) | **Real** — IMD's official classification |
| Elevation / slope / river distance | **Real, live** — Open-Meteo Elevation API + OSM Overpass |
| Impervious-surface density, coastal distance, monsoon flag | **Real, live** (impervious density is an OSM-derived approximation, not satellite-measured — see `geo_utils.py`) |
| ML ensemble (RF + Gradient Boosting + MLP) | **Real, trained**, but on a dataset that mixes 10 real curated events with explicit synthetic augmentation — see `train_model.py` docstring and `models/metrics.json`. This is genuinely all the curated ground truth available; a real deployment needs a proper historical flood inventory (e.g. from CWC/IMD/NRSC) to retrain on. |
| Hourly per-hour risk severity | **Real, fixed** — trailing-24h window, computed server-side |
| Flood extent map overlay | **Not yet wired** — `flood_mask_asset` is a filename placeholder |
| User accounts + admin dashboard | **Real and working** — SQLite-backed, state-scoped |
| Evaluation tool (place + date, any location, ~20yr range) | **Real** — replays actual Open-Meteo archive data |

## Priority next steps

1. Replace the placeholder-flagged basin rainfall totals with real
   CHIRPS/CWC-derived numbers for each basin.
2. Get a real historical flood inventory (state disaster management
   authorities, NRSC, CWC) to retrain `train_model.py` on real ground
   truth instead of synthetic augmentation.
3. Add the flood-extent GeoJSON overlay once available.
4. Deploy the backend somewhere it stays running (Render/Railway free
   tier) and update `API_BASE` in `index.html` and `admin.html` before
   publishing on GitHub Pages — `localhost` only works on your own machine.
5. Regenerate every admin key in `admin_keys.py` before a real demo/deploy.
