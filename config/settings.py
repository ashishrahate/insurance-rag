"""Central configuration. Every other module imports its settings from here
so connection details, collection names, and model IDs are defined once.

Values can be overridden with environment variables of the same name.
"""
import os

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
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")

# --- Payload fields that get a Qdrant index: field name -> schema type ---
PAYLOAD_INDEXES = {
    "state": "keyword",
    "document_type": "keyword",
    "date_effective": "datetime",
}
