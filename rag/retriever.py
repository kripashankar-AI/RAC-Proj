"""Retrieval-augmented question answering over the SSA FAQ corpus.

Supports two embedding models (``v1`` = MiniLM, ``v2`` = MPNet) selected via
the ``version`` argument.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from rag import config


@dataclass
class RetrievedFAQ:
    question: str
    answer: str
    url: str
    category: str
    score: float


def _paths_for(version: str) -> tuple[Path, Path, str]:
    if version == "v1":
        return config.FAISS_INDEX_PATH, config.META_PATH, config.EMBEDDING_MODEL
    if version == "v2":
        return (
            config.FAISS_INDEX_V2_PATH,
            config.META_V2_PATH,
            config.EMBEDDING_MODEL_V2,
        )
    raise ValueError(f"Unknown version: {version}")


class Retriever:
    def __init__(self, version: str = "v1") -> None:
        index_path, meta_path, model_name = _paths_for(version)
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Index for {version} not found. Run "
                f"`python -m rag.ingest --version {version}` first."
            )
        self.version = version
        self.model_name = model_name
        self.index = faiss.read_index(str(index_path))
        with meta_path.open("r", encoding="utf-8") as f:
            self.faqs: list[dict] = json.load(f)
        self.model = SentenceTransformer(model_name)

    def search(self, query: str, top_k: int = config.TOP_K) -> list[RetrievedFAQ]:
        emb = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        scores, idxs = self.index.search(emb, top_k)
        results: list[RetrievedFAQ] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            item = self.faqs[idx]
            results.append(
                RetrievedFAQ(
                    question=item["question"],
                    answer=item["answer"],
                    url=item.get("url", ""),
                    category=item.get("category", ""),
                    score=float(score),
                )
            )
        return results


# ---- one cached retriever per version --------------------------------------

_retriever_cache: dict[str, Retriever] = {}


def _get_retriever(version: str = "v1") -> Retriever:
    if version not in _retriever_cache:
        _retriever_cache[version] = Retriever(version)
    return _retriever_cache[version]


def _rephrase_with_openai(query: str, hits: list[RetrievedFAQ]) -> str | None:
    if not config.OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    context = "\n\n".join(
        f"[{i + 1}] Q: {h.question}\nA: {h.answer}\nSource: {h.url}"
        for i, h in enumerate(hits)
    )
    system = (
        "You are a helpful assistant answering questions about the U.S. Social "
        "Security Administration. Answer ONLY using the provided FAQ context. "
        "If the context does not contain the answer, say you don't know and "
        "suggest visiting ssa.gov. Cite sources as [n]. Be concise."
    )
    user = f"Question: {query}\n\nFAQ context:\n{context}"
    try:
        resp = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content
    except Exception as exc:
        return f"(OpenAI rephrase failed: {exc})"


def answer(query: str, top_k: int = config.TOP_K, version: str = "v1") -> dict:
    retriever = _get_retriever(version)
    hits = retriever.search(query, top_k=top_k)
    if not hits or hits[0].score < config.MIN_SCORE:
        return {
            "answer": (
                "I don't have a confident answer to that in my SSA FAQ corpus. "
                "Please check ssa.gov or call 1-800-772-1213."
            ),
            "sources": [h.__dict__ for h in hits],
            "used_llm": False,
            "version": version,
            "model": retriever.model_name,
        }

    rephrased = _rephrase_with_openai(query, hits)
    if rephrased:
        return {
            "answer": rephrased,
            "sources": [h.__dict__ for h in hits],
            "used_llm": True,
            "version": version,
            "model": retriever.model_name,
        }
    return {
        "answer": hits[0].answer,
        "sources": [h.__dict__ for h in hits],
        "used_llm": False,
        "version": version,
        "model": retriever.model_name,
    }
