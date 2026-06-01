"""Retrieval-augmented question answering over the SSA FAQ corpus."""
from __future__ import annotations

import json
from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rac import config


@dataclass
class RetrievedFAQ:
    question: str
    answer: str
    url: str
    category: str
    score: float


class Retriever:
    def __init__(self) -> None:
        if not config.FAISS_INDEX_PATH.exists() or not config.META_PATH.exists():
            raise FileNotFoundError(
                "Index not found. Run `python -m rac.ingest` first."
            )
        self.index = faiss.read_index(str(config.FAISS_INDEX_PATH))
        with config.META_PATH.open("r", encoding="utf-8") as f:
            self.faqs: list[dict] = json.load(f)
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)

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
    except Exception as exc:  # network/auth/etc
        return f"(OpenAI rephrase failed: {exc})"


def answer(query: str, top_k: int = config.TOP_K) -> dict:
    retriever = _get_retriever()
    hits = retriever.search(query, top_k=top_k)
    if not hits or hits[0].score < config.MIN_SCORE:
        return {
            "answer": (
                "I don't have a confident answer to that in my SSA FAQ corpus. "
                "Please check ssa.gov or call 1-800-772-1213."
            ),
            "sources": [h.__dict__ for h in hits],
            "used_llm": False,
        }

    rephrased = _rephrase_with_openai(query, hits)
    if rephrased:
        return {
            "answer": rephrased,
            "sources": [h.__dict__ for h in hits],
            "used_llm": True,
        }
    return {
        "answer": hits[0].answer,
        "sources": [h.__dict__ for h in hits],
        "used_llm": False,
    }


_retriever_singleton: Retriever | None = None


def _get_retriever() -> Retriever:
    global _retriever_singleton
    if _retriever_singleton is None:
        _retriever_singleton = Retriever()
    return _retriever_singleton
