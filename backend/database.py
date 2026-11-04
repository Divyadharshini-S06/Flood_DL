"""
SQLite database for user profiles + visit logging.

Tables:
  users  - one row per person (guest name-only, or Google sign-in)
  visits - one row per app open/location check, linked to a user, now
           tagged with the resolved `state` so the admin dashboard can be
           scoped per state without re-computing it on every read.

Kept as SQLite (no external service) so it runs anywhere with zero setup -
appropriate for a hackathon prototype. For real deployment, move to a
hosted database.
"""
import sqlite3
import os
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "floodxplain.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT,
            google_sub TEXT UNIQUE,
            auth_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS visits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            lat REAL,
            lon REAL,
            state TEXT,
            severity TEXT,
            confidence_tier TEXT,
            visited_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    # Migration for DBs created before the `state` column existed.
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(visits)").fetchall()]
    if "state" not in cols:
        conn.execute("ALTER TABLE visits ADD COLUMN state TEXT")
    conn.commit()
    conn.close()


def upsert_user(name: str, email: str | None, google_sub: str | None, auth_type: str) -> int:
    conn = get_conn()
    now = datetime.utcnow().isoformat()
    cur = conn.cursor()

    row = None
    if google_sub:
        row = cur.execute("SELECT id FROM users WHERE google_sub = ?", (google_sub,)).fetchone()

    if row:
        cur.execute("UPDATE users SET last_seen_at = ?, name = ? WHERE id = ?", (now, name, row["id"]))
        user_id = row["id"]
    else:
        cur.execute(
            "INSERT INTO users (name, email, google_sub, auth_type, created_at, last_seen_at) VALUES (?,?,?,?,?,?)",
            (name, email, google_sub, auth_type, now, now),
        )
        user_id = cur.lastrowid

    conn.commit()
    conn.close()
    return user_id


def log_visit(user_id: int | None, lat: float, lon: float, state: str, severity: str, confidence_tier: str):
    conn = get_conn()
    conn.execute(
        "INSERT INTO visits (user_id, lat, lon, state, severity, confidence_tier, visited_at) VALUES (?,?,?,?,?,?,?)",
        (user_id, lat, lon, state, severity, confidence_tier, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def _severity_breakdown(conn, state: str | None):
    q = "SELECT severity, COUNT(*) c FROM visits"
    params = ()
    if state:
        q += " WHERE state = ?"
        params = (state,)
    q += " GROUP BY severity"
    rows = conn.execute(q, params).fetchall()
    breakdown = {"Low": 0, "Moderate": 0, "Severe": 0, "Extreme": 0}
    for r in rows:
        if r["severity"] in breakdown:
            breakdown[r["severity"]] = r["c"]
    return breakdown


def _usage_timeseries(conn, state: str | None, days: int = 30):
    """Daily visit counts for the last `days` days — feeds the
    stock-market-style usage-over-time line chart in the admin dashboard."""
    since = (datetime.utcnow() - timedelta(days=days)).isoformat()
    q = "SELECT visited_at, severity FROM visits WHERE visited_at >= ?"
    params = [since]
    if state:
        q += " AND state = ?"
        params.append(state)
    rows = conn.execute(q, params).fetchall()

    buckets = {}
    for r in rows:
        day = r["visited_at"][:10]
        buckets.setdefault(day, {"total": 0, "high_severity": 0})
        buckets[day]["total"] += 1
        if r["severity"] in ("Severe", "Extreme"):
            buckets[day]["high_severity"] += 1

    # Fill in every day in the window (including zero-visit days) so the
    # chart has a continuous x-axis instead of gaps.
    series = []
    for i in range(days, -1, -1):
        day = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
        b = buckets.get(day, {"total": 0, "high_severity": 0})
        series.append({"date": day, "visits": b["total"], "high_severity_visits": b["high_severity"]})
    return series


def get_state_stats(state: str):
    conn = get_conn()
    total_visits = conn.execute("SELECT COUNT(*) c FROM visits WHERE state = ?", (state,)).fetchone()["c"]
    distinct_users = conn.execute(
        "SELECT COUNT(DISTINCT user_id) c FROM visits WHERE state = ? AND user_id IS NOT NULL", (state,)
    ).fetchone()["c"]
    recent_visits = conn.execute("""
        SELECT visits.id, users.name, visits.lat, visits.lon, visits.severity,
               visits.confidence_tier, visits.visited_at
        FROM visits LEFT JOIN users ON visits.user_id = users.id
        WHERE visits.state = ?
        ORDER BY visits.visited_at DESC LIMIT 200
    """, (state,)).fetchall()
    result = {
        "scope": "state",
        "state": state,
        "total_visits": total_visits,
        "distinct_users": distinct_users,
        "severity_breakdown": _severity_breakdown(conn, state),
        "usage_timeseries": _usage_timeseries(conn, state),
        "recent_visits": [dict(v) for v in recent_visits],
    }
    conn.close()
    return result


def get_master_stats():
    conn = get_conn()
    total_users = conn.execute("SELECT COUNT(*) c FROM users").fetchone()["c"]
    total_visits = conn.execute("SELECT COUNT(*) c FROM visits").fetchone()["c"]
    users = conn.execute(
        "SELECT id, name, email, auth_type, created_at, last_seen_at FROM users ORDER BY last_seen_at DESC LIMIT 200"
    ).fetchall()
    recent_visits = conn.execute("""
        SELECT visits.id, users.name, visits.lat, visits.lon, visits.state, visits.severity,
               visits.confidence_tier, visits.visited_at
        FROM visits LEFT JOIN users ON visits.user_id = users.id
        ORDER BY visits.visited_at DESC LIMIT 200
    """).fetchall()
    per_state_rows = conn.execute("""
        SELECT COALESCE(state, 'Other/Unmapped') state, COUNT(*) visits,
               SUM(CASE WHEN severity IN ('Severe','Extreme') THEN 1 ELSE 0 END) high_severity
        FROM visits GROUP BY state ORDER BY visits DESC
    """).fetchall()
    result = {
        "scope": "master",
        "total_users": total_users,
        "total_visits": total_visits,
        "severity_breakdown": _severity_breakdown(conn, None),
        "usage_timeseries": _usage_timeseries(conn, None),
        "per_state": [dict(r) for r in per_state_rows],
        "users": [dict(u) for u in users],
        "recent_visits": [dict(v) for v in recent_visits],
    }
    conn.close()
    return result
