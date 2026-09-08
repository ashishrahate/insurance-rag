"""Streamlit UI for the CA insurance regulations RAG assistant (Phase 4).

Pure HTTP client of the FastAPI service (`src/api/main.py`) -- never imports
`generate.py`/`retrieval/` directly. See docs/pipeline.md / the Phase 4 plan
for why: one API surface, one writer for feedback/cache state, and this file
keeps working unmodified if the API ever runs on a different host.

Run (with the API already up), from the repo root:
    streamlit run ui/app.py
"""
import sys
from pathlib import Path

# Streamlit runs this script directly and only puts its own directory
# (ui/) on sys.path -- unlike `python -m src.ask`, which adds the repo root
# automatically. Without this, `from config...`/`from src...` below fail
# with ModuleNotFoundError regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
import streamlit as st
import pandas as pd

from config.settings import API_BASE_URL, REFUSAL_SCORE_CUTOFF

# A query above the refusal cutoff but still not very confident -- flagged in
# the admin tab as worth a look, not auto-refused. Not a new server-side
# constant: this is a UI display threshold only, kept close to the guardrail
# cutoff it's relative to.
LOW_CONFIDENCE_THRESHOLD = REFUSAL_SCORE_CUTOFF + 0.2

st.set_page_config(page_title="CA Insurance RAG", layout="wide")
st.title("California Insurance Regulations Assistant")

ask_tab, admin_tab = st.tabs(["Ask", "Admin"])

# ---------------------------------------------------------------------------
# Ask tab
# ---------------------------------------------------------------------------
with ask_tab:
    with st.form("ask_form"):
        question = st.text_input(
            "Ask a question about California insurance regulations",
            placeholder="e.g. What must insurers do about smoke damage claims?",
        )
        doc_type_choice = st.selectbox(
            "Document type",
            ["all", "bulletin"],  # the only real document_type in the corpus today
        )
        submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        document_type = None if doc_type_choice == "all" else doc_type_choice
        with st.spinner("Retrieving and generating an answer (CPU inference, ~30-45s)..."):
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/query",
                    json={"question": question, "document_type": document_type},
                    timeout=120,
                )
                resp.raise_for_status()
                st.session_state["last_response"] = resp.json()
                st.session_state["last_question"] = question
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")

    data = st.session_state.get("last_response")
    if data:
        st.subheader("Answer")
        # No `unsafe_allow_html` -- Streamlit's default markdown escaping is
        # the concrete answer to "sanitize AI output before showing it in a
        # browser": raw HTML/script tags in the LLM's answer render as inert
        # text, not executable markup.
        if data["refused"]:
            st.warning(data["answer"])
        else:
            st.markdown(data["answer"])

        if data["citations"]:
            st.subheader("Sources")
            for c in data["citations"]:
                title = c.get("title") or c["doc_id"]
                url = c.get("source_url")
                st.markdown(f"- [{title}]({url})" if url else f"- {title}")
                if c.get("parent_headers"):
                    st.caption(" > ".join(c["parent_headers"]))

        cid = data["correlation_id"]
        feedback_sent_key = f"feedback_sent_{cid}"
        if st.session_state.get(feedback_sent_key):
            st.caption("Thanks for the feedback!")
        else:
            col_up, col_down = st.columns(2)

            def _send_feedback(rating: str) -> None:
                try:
                    requests.post(
                        f"{API_BASE_URL}/feedback",
                        json={
                            "correlation_id": cid,
                            "question": st.session_state.get("last_question"),
                            "answer": data["answer"],
                            "citations": data["citations"],
                            "rating": rating,
                        },
                        timeout=10,
                    )
                    st.session_state[feedback_sent_key] = True
                except requests.RequestException as e:
                    st.error(f"Feedback failed: {e}")

            with col_up:
                if st.button("👍 Helpful", key=f"up_{cid}"):
                    _send_feedback("up")
                    st.rerun()
            with col_down:
                if st.button("👎 Not helpful", key=f"down_{cid}"):
                    _send_feedback("down")
                    st.rerun()

        if data["retrieved_chunks"]:
            st.sidebar.subheader("Retrieved passages")
            for ch in data["retrieved_chunks"]:
                with st.sidebar.expander(f"{ch['chunk_id']}  (score={ch['score']:.3f})"):
                    st.write(ch["content"])

# ---------------------------------------------------------------------------
# Admin tab
# ---------------------------------------------------------------------------
with admin_tab:
    st.subheader("Feedback")
    try:
        stats = requests.get(f"{API_BASE_URL}/admin/feedback_stats", timeout=10).json()
        c1, c2, c3 = st.columns(3)
        c1.metric("👍 Up", stats["up"])
        c2.metric("👎 Down", stats["down"])
        c3.metric("Total", stats["total"])
    except requests.RequestException as e:
        st.error(f"Could not load feedback stats: {e}")

    st.subheader("Recent queries")
    try:
        recent = requests.get(
            f"{API_BASE_URL}/admin/recent_queries", params={"limit": 50}, timeout=10
        ).json()
        if recent:
            df = pd.DataFrame(recent)

            def _flag(row) -> str:
                if row["status"] in ("refused", "refused_guardrail", "no_results"):
                    return "refused"
                if row["status"] == "malformed_llm_output":
                    return "malformed"
                if row["top_score"] is not None and row["top_score"] < LOW_CONFIDENCE_THRESHOLD:
                    return "low-confidence"
                return ""

            df["flag"] = df.apply(_flag, axis=1)
            st.dataframe(
                df[["ts", "question", "status", "top_score", "flag", "correlation_id"]],
                use_container_width=True,
            )
        else:
            st.caption("No queries logged yet.")
    except requests.RequestException as e:
        st.error(f"Could not load recent queries: {e}")
