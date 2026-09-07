"""Central configuration. Every other module imports its settings from here
so connection details, collection names, and model IDs are defined once.

Values can be overridden with environment variables of the same name.
"""
import os
from pathlib import Path

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

# --- Chunking (naive fixed-size; Phase 2 sweeps these) ---
CHUNK_SIZE_WORDS = 300
CHUNK_OVERLAP_WORDS = 50

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

# --- Collections (California first; Blue/Green swaps between _v1 and _v2) ---
CA_COLLECTION = os.getenv("CA_COLLECTION", "insurance_ca_v1")
CA_COLLECTION_ALIAS = "insurance_ca_live"

# --- Embedding model (Ollama) ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = 768  # nomic-embed-text output dimensionality

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

# --- Evaluation (Phase 2) ---
# Retrieval eval scores the top-EVAL_K distinct doc_ids per question. Kept equal
# to RETRIEVE_K so the number measured is the number the app actually uses.
EVAL_SET_FILE = EVAL_DIR / "ca_eval_set.json"
EVAL_RESULTS_FILE = EVAL_DIR / "results.md"
EVAL_K = 3

# --- Observability ---
# Structured run records land in logs/runs.jsonl, one JSON line per query.
# RUN_ENV tags each record so runs from different hardware stay comparable
# (e.g. RUN_ENV=colab-t4 python -m src.ask "...").
LOGS_DIR = PROJECT_ROOT / "logs"
RUN_LOG_FILE = LOGS_DIR / "runs.jsonl"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
RUN_ENV = os.getenv("RUN_ENV", "local-cpu")
OBS_ENABLED = os.getenv("OBS_ENABLED", "1").lower() not in ("0", "false", "no")

# --- Payload fields that get a Qdrant index: field name -> schema type ---
PAYLOAD_INDEXES = {
    "state": "keyword",
    "document_type": "keyword",
    "date_effective": "datetime",
}
