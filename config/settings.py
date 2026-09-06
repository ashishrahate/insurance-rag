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

# --- Payload fields that get a Qdrant index: field name -> schema type ---
PAYLOAD_INDEXES = {
    "state": "keyword",
    "document_type": "keyword",
    "date_effective": "datetime",
}
