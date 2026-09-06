"""Extract clean text from the downloaded CA bulletin PDFs.

Reads data/raw/ca/manifest.json, pulls text with pdfplumber, strips repeated
letterhead / footer / page-number lines, fixes hyphenated line breaks, and
writes per bulletin into data/processed/ca/:
  <doc_id>.json   canonical record (metadata + text)
  <doc_id>.txt    text only, for quick human review
plus a manifest.json envelope summarising the run.

Run from the repo root:
    python -m src.ingestion.parse_ca_bulletins
    python -m src.ingestion.parse_ca_bulletins --only CA_BULLETIN_2025_7
"""
import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone

import pdfplumber

from config.settings import CA_BULLETINS_URL, PROCESSED_CA_DIR, RAW_CA_DIR

PAGE_NUM_RE = re.compile(
    r"^\s*(?:page\s+)?-?\s*\d+\s*(?:of\s+\d+)?\s*-?\s*$", re.IGNORECASE
)
HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
MIN_USEFUL_CHARS = 200  # below this, assume a scanned / image-only PDF

# "DATE: February 25, 2025" in the bulletin header block
DATE_LINE_RE = re.compile(r"^\s*DATE:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
DATE_FORMATS = ("%B %d, %Y", "%b %d, %Y", "%b. %d, %Y", "%B %d %Y", "%m/%d/%Y")


def extract_date_issued(text: str) -> tuple[str | None, str | None]:
    """Pull the header DATE: line and normalise to an ISO date (YYYY-MM-DD).

    Returns (iso_date_or_None, raw_matched_string_or_None) so a failed parse
    is still visible for debugging.
    """
    m = DATE_LINE_RE.search(text)
    if not m:
        return None, None
    raw = re.sub(r"\s+", " ", m.group(1)).strip().rstrip(".")
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date().isoformat(), raw
        except ValueError:
            continue
    return None, raw


def load_manifest() -> list[dict]:
    data = json.loads((RAW_CA_DIR / "manifest.json").read_text(encoding="utf-8"))
    docs = data["documents"] if isinstance(data, dict) else data
    return [d for d in docs if d.get("status") in ("downloaded", "cached")]


def extract_pages(pdf_path) -> list[list[str]]:
    """Text lines per page."""
    pages: list[list[str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append([ln.rstrip() for ln in text.splitlines()])
    return pages


def edge_boilerplate(pages: list[list[str]], edge: int = 3, ratio: float = 0.5) -> set[str]:
    """Lines near the top/bottom of many pages -- headers, footers, address block."""
    if len(pages) < 3:
        return set()
    counter: Counter[str] = Counter()
    for lines in pages:
        window = lines[:edge] + lines[-edge:]
        for ln in {l.strip() for l in window if l.strip()}:
            counter[ln] += 1
    threshold = max(2, int(len(pages) * ratio))
    return {ln for ln, n in counter.items() if n >= threshold}


def clean(pages: list[list[str]]) -> str:
    drop = edge_boilerplate(pages)
    cleaned_pages = []
    for lines in pages:
        kept = [
            ln
            for ln in lines
            if ln.strip()
            and ln.strip() not in drop
            and not PAGE_NUM_RE.match(ln)
        ]
        cleaned_pages.append("\n".join(kept))
    text = "\n\n".join(cleaned_pages)
    text = HYPHEN_BREAK_RE.sub(r"\1\2", text)   # join words split across lines
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def parse_one(doc: dict) -> dict:
    pdf_path = RAW_CA_DIR / f"{doc['doc_id']}.pdf"
    pages = extract_pages(pdf_path)
    text = clean(pages)
    date_issued, date_issued_raw = extract_date_issued(text)
    return {
        "doc_id": doc["doc_id"],
        "state": "CA",
        "document_type": doc.get("document_type", "bulletin"),
        "bulletin_number": doc.get("bulletin_number"),
        "title": doc.get("title"),
        "year": doc.get("year"),
        "source_url": doc.get("source_url"),
        "date_issued": date_issued,
        "date_issued_raw": date_issued_raw,
        "date_effective": None,  # stated in prose; Phase 2 / manual
        "n_pages": len(pages),
        "n_chars": len(text),
        "status": "ok" if len(text) >= MIN_USEFUL_CHARS else "empty_or_scanned",
        "text": text,
        "parsed_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="parse just this doc_id")
    args = ap.parse_args()

    docs = load_manifest()
    if args.only:
        docs = [d for d in docs if d["doc_id"] == args.only]
        if not docs:
            raise SystemExit(f"{args.only} not in manifest")

    PROCESSED_CA_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for doc in docs:
        rec = parse_one(doc)
        (PROCESSED_CA_DIR / f"{rec['doc_id']}.json").write_text(
            json.dumps(rec, indent=2), encoding="utf-8"
        )
        (PROCESSED_CA_DIR / f"{rec['doc_id']}.txt").write_text(
            rec["text"], encoding="utf-8"
        )
        summaries.append(
            {k: rec[k] for k in ("doc_id", "n_pages", "n_chars", "status", "date_issued")}
        )
        flag = "" if rec["status"] == "ok" else "  <-- CHECK"
        if rec["date_issued"] is None:
            flag += f"  (no date; raw={rec['date_issued_raw']!r})"
        d = rec["date_issued"] or "??????????"
        print(f"  {rec['doc_id']:<26} {d}  {rec['n_pages']:>2}p  {rec['n_chars']:>7,} chars{flag}")

    envelope = {
        "source": "ca_doi_bulletins",
        "listing_url": CA_BULLETINS_URL,
        "parsed_at": datetime.now(timezone.utc).isoformat(),
        "document_count": len(summaries),
        "documents": summaries,
    }
    (PROCESSED_CA_DIR / "manifest.json").write_text(
        json.dumps(envelope, indent=2), encoding="utf-8"
    )
    bad = [s["doc_id"] for s in summaries if s["status"] != "ok"]
    total = sum(s["n_chars"] for s in summaries)
    print(f"\nParsed {len(summaries)} docs, {total:,} chars -> {PROCESSED_CA_DIR}")
    if bad:
        print(f"Needs attention ({len(bad)}): {', '.join(bad)}")


if __name__ == "__main__":
    main()
