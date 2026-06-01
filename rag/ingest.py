"""Build FAISS indices from data/ssa_faqs.json.

By default builds both:
  - v1 index using ``EMBEDDING_MODEL``        (faqs.faiss)
  - v2 index using ``EMBEDDING_MODEL_V2``     (faqs_v2.faiss)

Use ``--version v1`` or ``--version v2`` to build just one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from rag import config


def load_faqs(path: Path = config.FAQS_PATH) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        faqs = json.load(f)
    if not isinstance(faqs, list) or not faqs:
        raise ValueError(f"No FAQs found in {path}")
    for i, item in enumerate(faqs):
        if "question" not in item or "answer" not in item:
            raise ValueError(f"FAQ #{i} missing 'question' or 'answer'")
    return faqs


def _build(faqs: list[dict], model_name: str, index_path: Path, meta_path: Path) -> None:
    print(f"\n>> Building '{model_name}' -> {index_path.name}")
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
    faiss.write_index(index, str(index_path))
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(faqs, f, ensure_ascii=False, indent=2)
    print(f"   wrote {index_path}\n   wrote {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS indices for RAG.")
    parser.add_argument("--version", choices=["v1", "v2", "all"], default="all")
    args = parser.parse_args()

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faqs = load_faqs()
    print(f"Loaded {len(faqs)} FAQs from {config.FAQS_PATH}")

    if args.version in {"v1", "all"}:
        _build(faqs, config.EMBEDDING_MODEL, config.FAISS_INDEX_PATH, config.META_PATH)
    if args.version in {"v2", "all"}:
        _build(
            faqs,
            config.EMBEDDING_MODEL_V2,
            config.FAISS_INDEX_V2_PATH,
            config.META_V2_PATH,
        )


if __name__ == "__main__":
    main()
