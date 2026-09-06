"""Build the chat messages for a grounded RAG answer."""
from qdrant_client.models import ScoredPoint

SYSTEM_PROMPT = (
    "You answer questions about California insurance regulations using the "
    "numbered context passages provided by the user.\n"
    "- Base every factual claim on the passages. Do not add outside knowledge "
    "or guess at figures, dates, or code sections.\n"
    "- You MAY combine, compare, and summarise across passages to answer "
    "(for example, contrasting two bulletins).\n"
    "- Only if the passages do not address the topic at all, reply exactly: "
    '"The provided bulletins do not cover this."\n'
    "- Cite the bulletin number(s) you used, e.g. (Bulletin 2025-8).\n"
    "- Be concise and specific. Quote regulatory language when it matters."
)


def format_context(hits: list[ScoredPoint]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        p = h.payload
        num = p.get("bulletin_number") or p.get("doc_id")
        blocks.append(
            f"[{i}] Bulletin {num} - {p.get('title', '')}\n"
            f"Source: {p.get('source_url', '')}\n"
            f"{p.get('content', '').strip()}"
        )
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list[ScoredPoint]) -> list[dict]:
    user = (
        f"Context passages:\n\n{format_context(hits)}\n\n"
        f"Question: {question}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
