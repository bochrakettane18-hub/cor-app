"""
auth.py - User accounts + per-user data store for the corrosion app.

Standard library only (sqlite3 + hashlib + secrets + hmac), so it adds no
dependencies. Design notes:

  * Passwords are never stored in clear text. Each account keeps a random salt
    and a PBKDF2-HMAC-SHA256 hash; verification is constant-time.
  * Every scored record is tagged with its owner and all reads are filtered by
    username, so one user can never see another user's data (confidentiality).
  * Uploads are *appended* - saving new rows never modifies or replaces the
    rows already stored.

Everything lives in a single SQLite file (app_data.db) next to the app.
"""

import json
import sqlite3
import hashlib
import hmac
import secrets
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "app_data.db"

_PBKDF2_ITERATIONS = 200_000
MIN_USERNAME_LEN = 3
MIN_PASSWORD_LEN = 6


@contextmanager
def _db():
    """Connection that commits on clean exit and always closes (thread-safe per call)."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create the users + records tables if they do not yet exist."""
    with _db() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS users (
                   username   TEXT PRIMARY KEY,
                   salt       TEXT NOT NULL,
                   pwd_hash   TEXT NOT NULL,
                   created_at TEXT NOT NULL
               )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS records (
                   id         INTEGER PRIMARY KEY AUTOINCREMENT,
                   username   TEXT NOT NULL,
                   created_at TEXT NOT NULL,
                   batch      TEXT,
                   payload    TEXT NOT NULL
               )"""
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_records_user ON records(username)")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash_password(password: str, salt: bytes) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", (password or "").encode("utf-8"),
                             salt, _PBKDF2_ITERATIONS)
    return dk.hex()


# ------------------------------------------------------------------ accounts
def user_exists(username: str) -> bool:
    with _db() as conn:
        row = conn.execute("SELECT 1 FROM users WHERE username = ?",
                           (username,)).fetchone()
    return row is not None


def create_user(username: str, password: str):
    """Register a new account. Returns (ok: bool, message_key: str)."""
    username = (username or "").strip()
    if len(username) < MIN_USERNAME_LEN:
        return False, "err_username_short"
    if len(password or "") < MIN_PASSWORD_LEN:
        return False, "err_password_short"
    if user_exists(username):
        return False, "err_user_taken"
    salt = secrets.token_bytes(16)
    pwd_hash = _hash_password(password, salt)
    with _db() as conn:
        conn.execute(
            "INSERT INTO users (username, salt, pwd_hash, created_at) VALUES (?, ?, ?, ?)",
            (username, salt.hex(), pwd_hash, _now()),
        )
    return True, "ok_signup"


def verify_user(username: str, password: str) -> bool:
    """Constant-time password check; False if the account is unknown."""
    username = (username or "").strip()
    with _db() as conn:
        row = conn.execute("SELECT salt, pwd_hash FROM users WHERE username = ?",
                           (username,)).fetchone()
    if row is None:
        # Hash anyway to blunt username-enumeration timing differences.
        _hash_password(password, b"\x00" * 16)
        return False
    calc = _hash_password(password, bytes.fromhex(row["salt"]))
    return hmac.compare_digest(calc, row["pwd_hash"])


# --------------------------------------------------------------- data store
def save_records(username: str, df: pd.DataFrame, batch: str = None) -> int:
    """Append scored rows for a user; returns the number of rows added.

    Inserts only - existing rows are never modified or replaced.
    """
    if df is None or len(df) == 0:
        return 0
    ts = _now()
    batch = batch or ts
    safe = df.where(pd.notnull(df), None)
    rows = [
        (username, ts, batch, json.dumps(rec, default=str, ensure_ascii=False))
        for rec in safe.to_dict(orient="records")
    ]
    with _db() as conn:
        conn.executemany(
            "INSERT INTO records (username, created_at, batch, payload) VALUES (?, ?, ?, ?)",
            rows,
        )
    return len(rows)


def count_records(username: str) -> int:
    with _db() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM records WHERE username = ?",
                           (username,)).fetchone()
    return int(row["n"]) if row else 0


def load_records(username: str) -> pd.DataFrame:
    """All stored rows for a user, newest first, as a DataFrame."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT created_at, batch, payload FROM records WHERE username = ? ORDER BY id DESC",
            (username,),
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    recs = []
    for r in rows:
        d = json.loads(r["payload"])
        d["_saved_at"] = r["created_at"]
        d["_source"] = r["batch"]
        recs.append(d)
    return pd.DataFrame(recs)
