"""Central configuration. Every other module imports its settings from here
so connection details, collection names, and model IDs are defined once.

Values can be overridden with environment variables of the same name. A
local `.env` file (gitignored, see `.env.example`) is loaded first if
present -- this is where secrets like OPENAI_API_KEY belong: never typed
into a shared terminal history or committed, and picked up automatically by
every `python`/`uvicorn`/`pytest` invocation without exporting anything by
hand each session.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_CA_DIR = DATA_DIR / "raw" / "ca"
PROCESSED_CA_DIR = DATA_DIR / "processed" / "ca"
EVAL_DIR = DATA_DIR / "eval"

# --- Source: California Department of Insurance bulletins ---
CA_DOI_BASE = "https://www.insurance.ca.gov"
CA_BULLETINS_URL = (
    "https://www.insurance.ca.gov/0250-insurers/0300-insurers/0200-bulletins/"
    "bulletin-notices-commiss-opinion/bulletins.cfm"
)

# --- Polite scraping ---
HTTP_HEADERS = {"User-Agent": "insurance-rag-bot/0.1 (educational RAG project)"}
REQUEST_DELAY_SEC = 1.0

# --- Chunking (naive fixed-size; Phase 2 Change 4 sweeps these) ---
CHUNK_SIZE_WORDS = int(os.getenv("CHUNK_SIZE_WORDS", "300"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "50"))

# Header-aware chunking splits each bulletin into heading-keyed sections and
# stamps every chunk with its heading path (`parent_headers`, a locked payload
# field used later for citations). Whether that heading path is also PREPENDED
# to the text we embed is a separate switch:
#   Phase 2 Change 1 measured it -> net negative on the CA corpus. The doc-level
#   "RE:" line is identical boilerplate across the near-duplicate pairs (Life vs
#   LTC PBR, the 6 moratorium clones), so prepending it dilutes the body signal
#   (fiscal year, dollar amounts) that was doing the disambiguation:
#   MRR 0.862 -> 0.804, Recall@3 0.957 -> 0.913. See Challenges & Learnings #5.
# So: keep the sections + populate parent_headers, but embed raw chunk text.
EMBED_WITH_HEADERS = os.getenv("EMBED_WITH_HEADERS", "0").lower() not in ("0", "false", "no")

# --- Qdrant ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
# Set only for environments without a Docker Qdrant server (e.g. Colab) --
# switches get_client() to embedded/local mode (RocksDB-backed, in-process).
# Unset by default so today's host/port behavior is unchanged. See
# docs/colab-gpu-plan.md §4.
QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH")

# Qdrant Cloud (docs/colab-gpu-plan.md §4 option 3): set both to point
# get_client() at a hosted cluster over HTTPS instead of local host/port or
# embedded mode. Takes precedence over QDRANT_LOCAL_PATH if both are set.
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# --- Collections (California first; Blue/Green swaps between _v1 and _v2) ---
CA_COLLECTION = os.getenv("CA_COLLECTION", "insurance_ca_v1")
CA_COLLECTION_ALIAS = "insurance_ca_live"

# --- Embedding model (Ollama) ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = 768  # nomic-embed-text output dimensionality

# --- Provider abstraction (Phase 3) ---
# Two independent axes, each with its own env var -- embed.py and
# generate.py never change, only which provider a given axis resolves to:
#   EMBED_PROVIDER -- what embeds documents at ingest time AND queries at
#     search time (these two MUST match: comparing a query vector to stored
#     vectors only works if both came from the same model/dimensionality).
#   LLM_PROVIDER -- what generates the answer from retrieved context.
# Decoupled on purpose: swapping the *generation* model (e.g. to compare
# gpt-4o-mini's answer quality against the llama3.2:3b baseline) is then a
# free comparison against the existing collection -- no re-embedding, no new
# collection, no OpenAI embedding spend, since retrieval doesn't change at
# all. Only changing EMBED_PROVIDER needs a new collection (see
# OPENAI_EMBED_DIM below) -- that's a deliberate, separate decision (Phase 5
# task 4/6), not a side effect of testing a different LLM.
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "ollama")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# --- OpenAI provider (Phase 5 task 5) ---
# text-embedding-3-small outputs 1536-dim vectors, not nomic-embed-text's 768
# -- switching LLM_PROVIDER to "openai" changes both chat and embedding, so
# it needs its own collection re-ingested at the right vector size, not a
# same-collection swap. See docs/pipeline.md / Blue/Green (Phase 5 task 4)
# for where that collection comes from.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_EMBED_DIM = 1536

# --- LLM (Ollama) ---
# Dev model is deliberately small: this machine has no Ollama-usable GPU, so
# inference is 100% CPU. Phase 1 only needs the pipeline to work; real answer
# quality comes from the Phase 3/5 OpenAI swap.
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
LLM_NUM_PREDICT = 300  # cap generated tokens (bounds worst-case latency)

# Keep Ollama models resident between calls so back-to-back questions don't
# pay the 5-20s reload tax (default keep_alive is only 5 min).
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

# --- Retrieval ---
RETRIEVE_K = 3  # chunks passed to the LLM; small corpus rarely needs more

# Reciprocal Rank Fusion damping constant (Change 2: BM25 + dense hybrid).
# Standard value from the original RRF paper (Cormack et al., 2009). RRF only
# uses rank position, not raw score, which is what lets us combine BM25 scores
# and cosine scores -- two incomparable scales -- without calibration.
RRF_K = 60

# Cross-encoder reranker (Change 3). Runs locally via sentence-transformers;
# ~2GB one-time download, ~1s/query on CPU for a 20-candidate pool.
RERANK_MODEL = "BAAI/bge-reranker-base"

# Refusal guardrail (Phase 3). Below this hybrid_rerank top-1 score, skip the
# LLM call entirely and return a structured refusal. Derived from
# data/eval/hybrid_rerank_scores.json (28-question eval set, see
# `retrieval_eval.py --dump-json`): in-scope top1 scores ranged
# [0.693, 1.000], out-of-scope top1 scores ranged [0.003, 0.286] -- a clean,
# non-overlapping gap. 0.5 sits at that gap's midpoint (~0.2 margin either
# side). Not a calibrated probability -- bge-reranker-base emits uncalibrated
# logits (see CLAUDE.md) -- and tuned on only 28 questions; revisit as the
# eval set grows.
REFUSAL_SCORE_CUTOFF = float(os.getenv("REFUSAL_SCORE_CUTOFF", "0.5"))

# Basic input hygiene, part of prompt-injection defense-in-depth (see
# docs/scaling-notes.md). Generous for a real question (typical ones are
# under 200 chars) but bounded -- caps worst-case embedding/rerank/prompt
# cost from a pathologically long question, injected or not. Enforced twice:
# QueryRequest (API) rejects with a clean 422 via Pydantic's max_length;
# answer_question() (generate.py) truncates-and-logs as defense-in-depth for
# the CLI path too, which has no schema validation of its own.
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "500"))

# --- Evaluation (Phase 2) ---
# Retrieval eval scores the top-EVAL_K distinct doc_ids per question. Kept equal
# to RETRIEVE_K so the number measured is the number the app actually uses.
EVAL_SET_FILE = EVAL_DIR / "ca_eval_set.json"
EVAL_RESULTS_FILE = EVAL_DIR / "results.md"
EVAL_K = 3

# --- LLM-as-judge (Phase 5) ---
# The judge runs as its own service (src/judge_service/) so which model judges
# is independently configurable from LLM_PROVIDER/LLM_MODEL above -- swapping
# the judge (e.g. to a frontier model, once the OpenAI provider lands) never
# touches the generation path it's scoring. Same ollama/llama3.2:3b default as
# generation for now, since only one provider exists yet.
JUDGE_LLM_PROVIDER = os.getenv("JUDGE_LLM_PROVIDER", "ollama")
JUDGE_LLM_MODEL = os.getenv("JUDGE_LLM_MODEL", "llama3.2:3b")
JUDGE_SERVICE_URL = os.getenv("JUDGE_SERVICE_URL", "http://localhost:8100")

ANSWER_EVAL_RESULTS_FILE = EVAL_DIR / "answer_eval_results.md"

# Quality gates (Phase 5): tuned against the eval set as it's run, never
# hardcoded from a guide -- same convention as REFUSAL_SCORE_CUTOFF above.
# Starting values are the roadmap's stated targets; revisit once a baseline
# run exists to tune against.
HIT_AT_K_GATE = float(os.getenv("HIT_AT_K_GATE", "0.80"))
FAITHFULNESS_GATE = float(os.getenv("FAITHFULNESS_GATE", "0.85"))

# --- Observability ---
# Structured run records land in logs/runs.jsonl, one JSON line per query.
# RUN_ENV tags each record so runs from different hardware stay comparable
# (e.g. RUN_ENV=colab-t4 python -m src.ask "...").
LOGS_DIR = PROJECT_ROOT / "logs"
RUN_LOG_FILE = LOGS_DIR / "runs.jsonl"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
RUN_ENV = os.getenv("RUN_ENV", "local-cpu")
OBS_ENABLED = os.getenv("OBS_ENABLED", "1").lower() not in ("0", "false", "no")

# --- UI & feedback (Phase 4) ---
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
FEEDBACK_DB_PATH = os.getenv("FEEDBACK_DB_PATH", str(PROJECT_ROOT / "feedback.db"))

# --- Payload fields that get a Qdrant index: field name -> schema type ---
PAYLOAD_INDEXES = {
    "state": "keyword",
    "document_type": "keyword",
    "date_effective": "datetime",
}
