"""Summarise logs/runs.jsonl: latency percentiles and throughput per env.

Use this to compare hardware. Tag runs with RUN_ENV when you record them:
    RUN_ENV=local-cpu  python -m src.ask "..."
    RUN_ENV=colab-t4   python -m src.ask "..."

Then:
    python -m src.observability.report
    python -m src.observability.report --file colab_runs.jsonl
    python -m src.observability.report --group llm_model --last 50
"""
import argparse
import json
from collections import defaultdict

from config.settings import RUN_LOG_FILE

STAGES = [
    "embed_ms",
    "qdrant_client_ms",
    "qdrant_search_ms",
    "prompt_build_ms",
    "llm_ms",
    "total_ms",
]
THROUGHPUT = ["prefill_tok_s", "gen_tok_s", "prompt_tokens", "output_tokens",
              "ollama_load_ms"]


def load(path, last=None, op="query"):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if op and r.get("op") != op:
                continue
            rows.append(r)
    return rows[-last:] if last else rows


def pct(values, p):
    if not values:
        return None
    s = sorted(values)
    idx = min(int(round((p / 100.0) * (len(s) - 1))), len(s) - 1)
    return s[idx]


def fmt(v, unit=""):
    if v is None:
        return "-"
    return f"{v:,.1f}{unit}" if isinstance(v, float) else f"{v:,}{unit}"


def summarise(rows, group_key):
    groups = defaultdict(list)
    for r in rows:
        groups[r.get(group_key, "?")].append(r)

    for name, rs in sorted(groups.items()):
        statuses = defaultdict(int)
        for r in rs:
            statuses[r.get("status", "?")] += 1
        status_str = ", ".join(f"{k}={v}" for k, v in sorted(statuses.items()))
        print(f"\n=== {group_key}={name}  ({len(rs)} runs: {status_str}) ===")

        print(f"  {'stage':<20} {'p50':>10} {'p95':>10} {'mean':>10}")
        for stg in STAGES:
            vals = [r[stg] for r in rs if isinstance(r.get(stg), (int, float))]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            print(f"  {stg:<20} {fmt(pct(vals, 50)):>10} "
                  f"{fmt(pct(vals, 95)):>10} {fmt(mean):>10}")

        print(f"  {'-' * 52}")
        for key in THROUGHPUT:
            vals = [r[key] for r in rs if isinstance(r.get(key), (int, float))]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            print(f"  {key:<20} {fmt(pct(vals, 50)):>10} "
                  f"{fmt(pct(vals, 95)):>10} {fmt(mean):>10}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=str(RUN_LOG_FILE), help="jsonl to read")
    ap.add_argument("--group", default="env",
                    help="field to group by (env, llm_model, status, k)")
    ap.add_argument("--last", type=int, help="only the last N runs")
    ap.add_argument("--op", default="query", help="operation filter ('' for all)")
    args = ap.parse_args()

    try:
        rows = load(args.file, args.last, args.op or None)
    except FileNotFoundError:
        raise SystemExit(f"no log file at {args.file} - run a query first")

    if not rows:
        raise SystemExit(f"no '{args.op}' records in {args.file}")

    print(f"{len(rows)} run(s) from {args.file}")
    summarise(rows, args.group)
    print()


if __name__ == "__main__":
    main()
