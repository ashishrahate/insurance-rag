"""Download recent California DOI bulletin PDFs and write a manifest.

This does fetch-and-save only. Text extraction / cleaning happens in the
next step (parse), which reads data/raw/ca/manifest.json rather than the
website.

Run from the repo root:
    python -m src.ingestion.scrape_ca_bulletins            # 18 newest, year >= 2020
    python -m src.ingestion.scrape_ca_bulletins --list-only
    python -m src.ingestion.scrape_ca_bulletins --limit 25 --min-year 2018
"""
import argparse
import json
import re
import time
from datetime import datetime, timezone
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from config.settings import (
    CA_BULLETINS_URL,
    CA_DOI_BASE,
    HTTP_HEADERS,
    RAW_CA_DIR,
    REQUEST_DELAY_SEC,
)

# "Bulletin 2025-7", "Bulletin 2000-2)", non-breaking spaces, etc.
NUM_IN_TEXT = re.compile(r"bulletin\s*(\d{2,4})[-–](\d+)", re.IGNORECASE)
NUM_IN_URL = re.compile(r"bulletin[-_]?(\d{4})[-_](\d+)", re.IGNORECASE)


def fetch_listing() -> BeautifulSoup:
    resp = requests.get(CA_BULLETINS_URL, headers=HTTP_HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def preceding_year(anchor) -> int | None:
    """Year from the nearest <h3> heading above this anchor in document order."""
    for h3 in anchor.find_all_previous("h3"):
        m = re.search(r"(19|20)\d{2}", h3.get_text())
        if m:
            return int(m.group(0))
    return None


def clean_title(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip(" :)–-")


def parse_bulletin_number(title: str, url: str) -> str | None:
    m = NUM_IN_TEXT.search(title) or NUM_IN_URL.search(url)
    if not m:
        return None
    year, seq = m.group(1), m.group(2)
    if len(year) == 2:  # e.g. "96-12"
        year = ("19" if int(year) > 50 else "20") + year
    return f"{year}-{int(seq)}"


def slug_from_url(url: str) -> str:
    name = url.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")[:60]


def collect_candidates(soup: BeautifulSoup) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for a in soup.find_all("a", href=True):
        if not a["href"].lower().endswith(".pdf"):
            continue
        url = urljoin(CA_DOI_BASE, a["href"])
        if url in seen:
            continue
        seen.add(url)

        title = clean_title(a.get_text())
        number = parse_bulletin_number(title, url)
        year = (int(number[:4]) if number else None) or preceding_year(a)
        if not title or title == "�":
            title = slug_from_url(url).replace("_", " ")

        if number:
            doc_id = f"CA_BULLETIN_{number.replace('-', '_')}"
        else:
            doc_id = f"CA_BULLETIN_{slug_from_url(url)}"

        out.append(
            {
                "doc_id": doc_id,
                "bulletin_number": number,
                "title": title,
                "year": year,
                "document_type": "bulletin",
                "source_url": url,
            }
        )
    return out


def select(cands: list[dict], min_year: int, limit: int) -> list[dict]:
    kept = [c for c in cands if c["year"] and c["year"] >= min_year]
    # dedupe on doc_id, keeping the longest title
    by_id: dict[str, dict] = {}
    for c in kept:
        cur = by_id.get(c["doc_id"])
        if cur is None or len(c["title"]) > len(cur["title"]):
            by_id[c["doc_id"]] = c

    def sort_key(c: dict):
        seq = int(c["bulletin_number"].split("-")[1]) if c["bulletin_number"] else 0
        return (c["year"], seq)

    return sorted(by_id.values(), key=sort_key, reverse=True)[:limit]


def download_pdf(url: str, dest, force: bool) -> tuple[str, int]:
    if dest.exists() and not force:
        return "cached", dest.stat().st_size
    resp = requests.get(url, headers=HTTP_HEADERS, timeout=60)
    resp.raise_for_status()
    if not resp.content.startswith(b"%PDF"):
        raise ValueError(f"not a PDF (starts with {resp.content[:8]!r})")
    dest.write_bytes(resp.content)
    return "downloaded", len(resp.content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=18)
    parser.add_argument("--min-year", type=int, default=2020)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    soup = fetch_listing()
    selected = select(collect_candidates(soup), args.min_year, args.limit)

    print(f"Selected {len(selected)} bulletins (year >= {args.min_year}):")
    for c in selected:
        print(f"  {c['doc_id']:<28} {c['title'][:70]}")
    if args.list_only:
        return

    RAW_CA_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for c in selected:
        dest = RAW_CA_DIR / f"{c['doc_id']}.pdf"
        rec = {**c, "local_path": str(dest.relative_to(RAW_CA_DIR.parents[2]))}
        try:
            status, size = download_pdf(c["source_url"], dest, args.force)
            rec.update(status=status, bytes=size,
                       scraped_at=datetime.now(timezone.utc).isoformat())
            print(f"  [{status:>10}] {c['doc_id']} ({size:,} B)")
        except Exception as exc:  # noqa: BLE001 - record and move on
            rec.update(status="error", error=str(exc))
            print(f"  [     ERROR] {c['doc_id']}: {exc}")
        records.append(rec)
        if rec["status"] == "downloaded":
            time.sleep(REQUEST_DELAY_SEC)

    manifest = RAW_CA_DIR / "manifest.json"
    envelope = {
        "source": "ca_doi_bulletins",
        "listing_url": CA_BULLETINS_URL,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "document_count": len(records),
        "documents": records,
    }
    manifest.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    ok = sum(r["status"] in ("downloaded", "cached") for r in records)
    print(f"\n{ok}/{len(records)} available. Manifest: {manifest}")


if __name__ == "__main__":
    main()
