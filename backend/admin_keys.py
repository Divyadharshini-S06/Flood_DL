"""
Admin key registry — state-scoped keys + one master key.

WHY THIS FILE EXISTS (per your request): each Indian state gets its own
admin key so a state disaster-management authority can log in and see
ONLY their state's usage/risk data — they can't see other states' data,
and don't need to know other states' keys. The master key (used by you /
a national coordinating body) can see everything, aggregated and broken
down per state.

SECURITY NOTE — same honesty as the rest of this project: this is a
shared-secret scheme appropriate for a hackathon prototype, not real
production-grade role-based auth. Before a real deployment:
  1. Move these keys out of source control (env vars / secrets manager).
  2. Regenerate every key below — the ones here are demo values.
  3. Add per-key rate limiting and audit logging.

Every state key below is verified unique at import time (see the
assertion at the bottom) — if you add a new state, the app will refuse to
start if you accidentally reuse an existing key.
"""
import os

# One unique key per state/UT. Override any of these via environment
# variables named FLOODXPLAIN_ADMIN_KEY_<STATE_SLUG> before your demo if
# you don't want to use the placeholder values below.
_DEFAULT_STATE_KEYS = {
    "Kerala":              "fx-kl-9247-monsoon",
    "Tamil Nadu":          "fx-tn-3381-cooum",
    "Karnataka":           "fx-ka-5560-cauvery",
    "Andhra Pradesh":      "fx-ap-7710-godavari",
    "Telangana":           "fx-tg-2294-musi",
    "Goa":                 "fx-ga-1187-mandovi",
    "Maharashtra":         "fx-mh-8823-mithi",
    "Gujarat":             "fx-gj-4456-narmada",
    "Rajasthan":           "fx-rj-6634-luni",
    "Madhya Pradesh":      "fx-mp-1129-betwa",
    "Chhattisgarh":        "fx-cg-5502-mahanadi",
    "Odisha":              "fx-od-3390-baitarani",
    "West Bengal":         "fx-wb-7712-damodar",
    "Bihar":               "fx-br-2266-kosi",
    "Jharkhand":           "fx-jh-4478-subarnarekha",
    "Uttar Pradesh":       "fx-up-9903-rapti",
    "Uttarakhand":         "fx-uk-5541-alaknanda",
    "Himachal Pradesh":    "fx-hp-8820-beas",
    "Punjab":              "fx-pb-3367-sutlej",
    "Haryana":             "fx-hr-1145-yamuna",
    "Delhi":               "fx-dl-7761-yamuna",
    "Jammu and Kashmir":   "fx-jk-6629-jhelum",
    "Assam":               "fx-as-2298-brahmaputra",
    "Sikkim":              "fx-sk-4471-teesta",
    "Manipur":             "fx-mn-8842-imphal",
    "Meghalaya":           "fx-ml-5563-umiam",
    "Mizoram":             "fx-mz-3396-tlawng",
    "Nagaland":            "fx-nl-7729-dhansiri",
    "Tripura":             "fx-tr-1153-gumti",
    "Arunachal Pradesh":   "fx-ar-9917-siang",
}


def _env_slug(state: str) -> str:
    return state.upper().replace(" ", "_").replace("/", "_")


def _load_state_keys() -> dict:
    keys = {}
    for state, default_key in _DEFAULT_STATE_KEYS.items():
        env_name = f"FLOODXPLAIN_ADMIN_KEY_{_env_slug(state)}"
        keys[state] = os.environ.get(env_name, default_key)
    return keys


STATE_ADMIN_KEYS: dict[str, str] = _load_state_keys()

# Master key — can view every state, aggregated national stats, and the
# per-state breakdown. Change this before your demo.
MASTER_ADMIN_KEY: str = os.environ.get("FLOODXPLAIN_MASTER_ADMIN_KEY", "fx-master-2026-flood-national")

# Build the reverse lookup once. If two states somehow end up with the same
# key (e.g. a careless env-var override), fail loudly at startup rather than
# silently letting one state see another's dashboard.
_KEY_TO_STATE: dict[str, str] = {}
for _state, _key in STATE_ADMIN_KEYS.items():
    if _key in _KEY_TO_STATE:
        raise RuntimeError(
            f"Admin key collision: '{_key}' is used by both "
            f"'{_KEY_TO_STATE[_key]}' and '{_state}'. Every state admin key "
            f"must be unique — fix the relevant FLOODXPLAIN_ADMIN_KEY_* env var."
        )
    if _key == MASTER_ADMIN_KEY:
        raise RuntimeError(f"State '{_state}' admin key collides with the master key.")
    _KEY_TO_STATE[_key] = _state


def resolve_admin_key(key: str):
    """Returns ('master', None) for the master key, ('state', state_name)
    for a valid state key, or (None, None) if the key is invalid."""
    if not key:
        return None, None
    if key == MASTER_ADMIN_KEY:
        return "master", None
    state = _KEY_TO_STATE.get(key)
    if state:
        return "state", state
    return None, None
