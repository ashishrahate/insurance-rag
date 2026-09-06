"""Console logging setup + append-only structured run log (JSON Lines).

The run log is written once per operation, after the work is done. Every write
is wrapped so a logging failure can never propagate into the query path.
"""
import json
import logging
import os
import platform
import uuid
from datetime import datetime, timezone

from config.settings import LOG_LEVEL, LOGS_DIR, OBS_ENABLED, RUN_ENV, RUN_LOG_FILE

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Console logger. Handlers are attached once per process."""
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-5s %(name)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        root = logging.getLogger("insurance_rag")
        root.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(f"insurance_rag.{name}")


def new_correlation_id() -> str:
    """Short id tying every stage of one operation together."""
    return uuid.uuid4().hex[:12]


# Computed once - cheap to attach to every record, makes runs from different
# machines self-describing when the logs are compared later.
_HOST = {
    "env": RUN_ENV,
    "platform": platform.system(),
    "machine": platform.machine(),
    "python": platform.python_version(),
    "cpu_count": os.cpu_count(),
}


def host_info() -> dict:
    return dict(_HOST)


def log_run(record: dict) -> None:
    """Append one JSON line to logs/runs.jsonl. Silent on any failure."""
    if not OBS_ENABLED:
        return
    try:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **_HOST,
            **record,
        }
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RUN_LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass
