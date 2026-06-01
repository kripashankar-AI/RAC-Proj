"""Central configuration for the RAC project."""
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
INDEX_DIR = ROOT_DIR / "index"

FAQS_PATH = DATA_DIR / "ssa_faqs.json"
FAISS_INDEX_PATH = INDEX_DIR / "faqs.faiss"
META_PATH = INDEX_DIR / "faqs_meta.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 4
MIN_SCORE = 0.25  # cosine similarity floor for "I don't know"

# Second model used by the "New RAG new model" tab.
EMBEDDING_MODEL_V2 = "sentence-transformers/all-mpnet-base-v2"
FAISS_INDEX_V2_PATH = INDEX_DIR / "faqs_v2.faiss"
META_V2_PATH = INDEX_DIR / "faqs_v2_meta.json"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

SSA_FAQ_BASE = "https://faq.ssa.gov"
USER_AGENT = "RAG-SSA-FAQ-Bot/0.1 (+research; contact via repo)"
