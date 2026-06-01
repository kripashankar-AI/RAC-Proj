"""Build a FAISS index from data/ssa_faqs.json."""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rac import config


def load_faqs(path: Path = config.FAQS_PATH) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        faqs = json.load(f)
    if not isinstance(faqs, list) or not faqs:
        raise ValueError(f"No FAQs found in {path}")
    for i, item in enumerate(faqs):
        if "question" not in item or "answer" not in item:
            raise ValueError(f"FAQ #{i} missing 'question' or 'answer'")
    return faqs


def build_index(faqs: list[dict], model_name: str = config.EMBEDDING_MODEL):
    model = SentenceTransformer(model_name)
    texts = [f"{item['question']}\n{item['answer']}" for item in faqs]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


def main() -> None:
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faqs = load_faqs()
    print(f"Loaded {len(faqs)} FAQs from {config.FAQS_PATH}")

    index, _ = build_index(faqs)
    faiss.write_index(index, str(config.FAISS_INDEX_PATH))
    with config.META_PATH.open("w", encoding="utf-8") as f:
        json.dump(faqs, f, ensure_ascii=False, indent=2)

    print(f"Wrote FAISS index -> {config.FAISS_INDEX_PATH}")
    print(f"Wrote metadata    -> {config.META_PATH}")


if __name__ == "__main__":
    main()
