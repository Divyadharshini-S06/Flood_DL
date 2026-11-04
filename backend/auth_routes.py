"""
Auth + admin endpoints.

Two ways a person can use the app:
  1. "Continue with Google" - verified server-side using Google's own
     library, so we can trust the name/email it gives us.
  2. "Continue as Guest" - just a name, no verification, lower trust but
     zero setup friction for a hackathon demo.

Admin endpoint is protected by the state-scoped / master key scheme in
admin_keys.py. A state key only ever sees that state's data (real
abstraction, enforced server-side — the frontend never decides this); the
master key sees everything plus the per-state breakdown. This is a
shared-secret scheme appropriate for a hackathon prototype, not real
production role-based auth (see admin_keys.py for the honest caveats).
"""
import os
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from database import upsert_user, log_visit, get_state_stats, get_master_stats
from basin_data import locate_state, is_within_india
from admin_keys import resolve_admin_key

router = APIRouter()

GOOGLE_CLIENT_ID = os.environ.get("FLOODXPLAIN_GOOGLE_CLIENT_ID")


class GuestLoginRequest(BaseModel):
    name: str
    email: str | None = None


class GoogleLoginRequest(BaseModel):
    credential: str


class VisitLogRequest(BaseModel):
    user_id: int | None = None
    lat: float
    lon: float
    severity: str
    confidence_tier: str


@router.post("/auth/guest")
def auth_guest(body: GuestLoginRequest):
    if not body.name or not body.name.strip():
        raise HTTPException(status_code=400, detail="Name is required")
    user_id = upsert_user(name=body.name.strip(), email=body.email, google_sub=None, auth_type="guest")
    return {"user_id": user_id, "name": body.name.strip(), "auth_type": "guest"}


@router.post("/auth/google")
def auth_google(body: GoogleLoginRequest):
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(
            status_code=501,
            detail="Google sign-in isn't configured on this server yet. Set FLOODXPLAIN_GOOGLE_CLIENT_ID, "
                   "or use 'Continue as Guest' instead.",
        )
    try:
        from google.oauth2 import id_token
        from google.auth.transport import requests as google_requests

        info = id_token.verify_oauth2_token(body.credential, google_requests.Request(), GOOGLE_CLIENT_ID)
        name = info.get("name", "Google User")
        email = info.get("email")
        sub = info.get("sub")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid Google token: {e}")

    user_id = upsert_user(name=name, email=email, google_sub=sub, auth_type="google")
    return {"user_id": user_id, "name": name, "email": email, "auth_type": "google"}


@router.post("/log-visit")
def log_visit_endpoint(body: VisitLogRequest):
    state = locate_state(body.lat, body.lon) if is_within_india(body.lat, body.lon) else "Other/Unmapped"
    log_visit(body.user_id, body.lat, body.lon, state, body.severity, body.confidence_tier)
    return {"ok": True, "state": state}


@router.get("/admin/verify")
def admin_verify(x_admin_key: str = Header(default="")):
    """Lets the admin frontend find out ahead of time whether a key is a
    state key or the master key (and which state), without yet returning
    the full dataset."""
    scope, state = resolve_admin_key(x_admin_key)
    if not scope:
        raise HTTPException(status_code=403, detail="Invalid or missing admin key")
    return {"scope": scope, "state": state}


@router.get("/admin/stats")
def admin_stats(x_admin_key: str = Header(default="")):
    scope, state = resolve_admin_key(x_admin_key)
    if not scope:
        raise HTTPException(status_code=403, detail="Invalid or missing admin key")
    if scope == "master":
        return get_master_stats()
    return get_state_stats(state)
