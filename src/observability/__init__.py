"""Latency measurement and structured run logging.

Deliberately passive: timing is perf_counter deltas, the log write happens once
after a query completes, and nothing here can raise into the inference path.
"""
from src.observability.logger import get_logger, log_run, new_correlation_id
from src.observability.ollama_metrics import extract_ollama_metrics
from src.observability.timing import Stopwatch, stage

__all__ = [
    "Stopwatch",
    "stage",
    "get_logger",
    "log_run",
    "new_correlation_id",
    "extract_ollama_metrics",
]
