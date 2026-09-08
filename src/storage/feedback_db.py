"""SQLite feedback storage (Phase 4).

One writer (`/feedback` in `src/api/main.py`) -- the UI never touches this
file directly, same reasoning as `/ingest` owning the query-cache clear (see
docs/pipeline.md §7.5 / the Phase 4 plan). `logs/runs.jsonl` still gets a
`log_run()` call too for feedback events -- this module is the durable,
queryable store the roadmap calls for; the JSONL line is the existing
observability trail, unchanged.
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from config.settings import FEEDBACK_DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    correlation_id TEXT,
    question TEXT,
    answer TEXT,
    rating TEXT NOT NULL,
    comment TEXT,
    created_at TEXT NOT NULL
);
"""


@contextmanager
def _connect():
    conn = sqlite3.connect(FEEDBACK_DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create the table if it doesn't exist. Idempotent -- safe to call on
    every app startup (called from `src/api/main.py`'s `lifespan`)."""
    with _connect() as conn:
        conn.execute(_SCHEMA)


def insert_feedback(
    correlation_id: str | None,
    question: str | None,
    answer: str | None,
    rating: str,
    comment: str | None,
) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO feedback (correlation_id, question, answer, rating, "
            "comment, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                correlation_id,
                question,
                answer,
                rating,
                comment,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )


def feedback_stats() -> dict:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT rating, COUNT(*) FROM feedback GROUP BY rating"
        ).fetchall()
    counts = {rating: n for rating, n in rows}
    up, down = counts.get("up", 0), counts.get("down", 0)
    return {"up": up, "down": down, "total": up + down}
