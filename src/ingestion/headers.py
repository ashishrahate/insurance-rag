"""Recover lightweight section structure from a cleaned CA bulletin.

The parser (`parse_ca_bulletins.py`) emits one flat `text` blob. These bulletins
do have structure, just not machine-readable structure:

  - a `RE:` subject line (often wrapping 2-3 physical lines) that is the single
    best one-line description of the document;
  - in the longer ones, `I.` / `II.` roman-numeral sections and `A.` / `B.`
    lettered subsections.

`split_into_sections` turns the blob into `[Section(headers, body)]`, where
`headers` is the heading path above that body (`[RE topic]`, or
`[RE topic, "II. ...", "A. ..."]`). The chunker chunks each section's body
independently and stamps every chunk with that section's `headers`
(the locked `parent_headers` payload field).

Why it matters: 6 of the 18 bulletins are wildfire-moratorium notices that share
~90% boilerplate. Prepending the heading path to the text we embed pushes each
one's chunks into a distinct region of vector space instead of piling them on
top of each other.
"""
import re
from dataclasses import dataclass, field

# "RE: <subject>" in the header block. Capture the rest of that physical line.
_RE_LINE = re.compile(r"^\s*RE:\s*(.+?)\s*$", re.IGNORECASE)

# "I. Background", "II. AB 144 Provides ..." -- roman numeral + title text.
_ROMAN_HEADING = re.compile(r"^\s*([IVXLC]{1,6})\.\s+(\S.*)$")

# "A. Previous CIC section 10112.2 ...", "D. Health Savings Account ..."
_LETTER_HEADING = re.compile(r"^\s*([A-H])\.\s+(\S.{3,})$")

# First line of the RE continuation that is actually body prose, not more title.
_BODY_START = re.compile(
    r"^(On |As enacted|As required|This Bulletin|In accordance|Following |"
    r"Pursuant |California Insurance Code|Proposition |Property and casualty|"
    r"Insurers |Under |Effective )",
)

_MAX_RE_LINES = 4


@dataclass
class Section:
    headers: list[str]
    body: str = ""
    _lines: list[str] = field(default_factory=list, repr=False)


def extract_re_topic(lines: list[str]) -> tuple[str | None, int]:
    """Return (RE subject, index of first line after the RE block).

    The subject can wrap: keep appending following lines until one looks like
    body prose, ends a sentence, or the cap is hit.
    """
    for i, ln in enumerate(lines):
        m = _RE_LINE.match(ln)
        if not m:
            continue
        parts = [m.group(1).strip()]
        j = i + 1
        while j < len(lines) and len(parts) < _MAX_RE_LINES:
            nxt = lines[j].strip()
            if not nxt or _BODY_START.match(nxt) or parts[-1].endswith("."):
                break
            parts.append(nxt)
            j += 1
        topic = re.sub(r"\s+", " ", " ".join(parts)).rstrip(".")
        return f"RE: {topic}", j
    return None, 0


def split_into_sections(text: str) -> list[Section]:
    """Split a cleaned bulletin into heading-keyed sections.

    Always returns at least one Section. Bulletins with no roman/letter headings
    (most of them) come back as a single section carrying just the RE topic.
    """
    lines = text.splitlines()
    re_topic, start = extract_re_topic(lines)
    root = [re_topic] if re_topic else []

    sections: list[Section] = [Section(headers=list(root))]
    roman_heading: str | None = None

    for ln in lines[start:]:
        rm = _ROMAN_HEADING.match(ln)
        lm = _LETTER_HEADING.match(ln)
        if rm:
            roman_heading = f"{rm.group(1)}. {rm.group(2).strip()}"
            sections.append(Section(headers=root + [roman_heading]))
        elif lm and roman_heading:
            letter = f"{lm.group(1)}. {lm.group(2).strip()}"
            sections.append(Section(headers=root + [roman_heading, letter]))
        else:
            sections[-1]._lines.append(ln)

    for s in sections:
        s.body = "\n".join(s._lines).strip()
    return [s for s in sections if s.body]


def format_for_embedding(headers: list[str], content: str) -> str:
    """Text actually sent to the embedder.

    With `EMBED_WITH_HEADERS` off (the Phase 2 default -- Change 1 showed the
    heading prefix hurts on this corpus), this is just the raw chunk. The
    heading path still rides along in the `parent_headers` payload either way.
    """
    from config.settings import EMBED_WITH_HEADERS

    if not headers or not EMBED_WITH_HEADERS:
        return content
    return " > ".join(headers) + "\n\n" + content
