"""Extract Ollama's own timing counters from a normalized ChatResponse.

Ollama reports nanosecond durations and token counts alongside the result.
These are the numbers that actually separate CPU from GPU inference:
prefill throughput, generation throughput, and model load time. They live
under `resp["raw"]` -- the provider-native payload `ChatResponse` keeps
alongside its normalized fields (see `src/providers/base.py`) -- since only
Ollama reports them; a provider without them just yields an empty dict here.
"""
NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000


def _get(resp, key, default=None):
    """Works for both dicts and the ollama client's pydantic response models."""
    if isinstance(resp, dict):
        return resp.get(key, default)
    return getattr(resp, key, default)


def _ms(ns):
    return round(ns / NS_PER_MS, 2) if isinstance(ns, (int, float)) else None


def _tok_s(count, ns):
    if not count or not ns:
        return None
    return round(count / (ns / NS_PER_S), 2)


def extract_metrics(resp: dict) -> dict:
    """Return a flat dict of Ollama-reported timings. Never raises.

    `resp` is a `ChatResponse` (the normalized shape every Provider.chat()
    returns); this reads its `raw` field, which is only populated with these
    counters when the backend is Ollama.
    """
    try:
        raw = resp.get("raw") if isinstance(resp, dict) else None
        if raw is None:
            return {}
        prompt_tokens = _get(raw, "prompt_eval_count")
        prompt_ns = _get(raw, "prompt_eval_duration")
        output_tokens = _get(raw, "eval_count")
        output_ns = _get(raw, "eval_duration")
        return {
            "ollama_total_ms": _ms(_get(raw, "total_duration")),
            "ollama_load_ms": _ms(_get(raw, "load_duration")),
            "prompt_tokens": prompt_tokens,
            "prompt_eval_ms": _ms(prompt_ns),
            "prefill_tok_s": _tok_s(prompt_tokens, prompt_ns),
            "output_tokens": output_tokens,
            "output_eval_ms": _ms(output_ns),
            "gen_tok_s": _tok_s(output_tokens, output_ns),
        }
    except Exception:  # observability must never break a query
        return {}
