"""
Lightweight SQLite log of past analyses, so the app remembers what you've
run instead of losing everything on refresh — turns this from a one-shot
demo into an actual tool with a record you can come back to.

No setup needed: the DB file is created automatically on first use.
"""
from __future__ import annotations
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Optional

# Override with HISTORY_DB_PATH env var (e.g. to point at a mounted Docker
# volume) — defaults to a plain file next to the project root.
DB_PATH = Path(os.environ.get("HISTORY_DB_PATH", str(Path(__file__).parent.parent / "history.db")))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,            -- 'classify' | 'change'
    engine TEXT,
    thumbnail_base64 TEXT,         -- small preview image, kept short
    stats_json TEXT NOT NULL       -- full stats/summary payload for this run
);
"""


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(_SCHEMA)
    return conn


def log_analysis(kind: str, engine: str, thumbnail_base64: str, stats: dict) -> int:
    with closing(_connect()) as conn:
        cur = conn.execute(
            "INSERT INTO analyses (created_at, kind, engine, thumbnail_base64, stats_json) VALUES (?, ?, ?, ?, ?)",
            (datetime.utcnow().isoformat(timespec="seconds") + "Z", kind, engine, thumbnail_base64, json.dumps(stats)),
        )
        conn.commit()
        return cur.lastrowid


def get_history(limit: int = 20) -> list[dict]:
    with closing(_connect()) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, created_at, kind, engine, thumbnail_base64, stats_json FROM analyses ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {
            "id": r["id"],
            "created_at": r["created_at"],
            "kind": r["kind"],
            "engine": r["engine"],
            "thumbnail_base64": r["thumbnail_base64"],
            "stats": json.loads(r["stats_json"]),
        }
        for r in rows
    ]


def get_analysis(analysis_id: int) -> Optional[dict]:
    with closing(_connect()) as conn:
        conn.row_factory = sqlite3.Row
        r = conn.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    if not r:
        return None
    return {
        "id": r["id"], "created_at": r["created_at"], "kind": r["kind"], "engine": r["engine"],
        "thumbnail_base64": r["thumbnail_base64"], "stats": json.loads(r["stats_json"]),
    }


def clear_history() -> None:
    with closing(_connect()) as conn:
        conn.execute("DELETE FROM analyses")
        conn.commit()
