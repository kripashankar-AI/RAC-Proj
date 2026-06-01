"""Gradio web UI for the SSA FAQ chatbot."""
from __future__ import annotations

import gradio as gr

from rac import retriever


def _format_sources(sources: list[dict]) -> str:
    if not sources:
        return ""
    lines = ["**Sources**"]
    for i, s in enumerate(sources, 1):
        url = s.get("url") or ""
        title = s.get("question", "(untitled)")
        score = s.get("score", 0.0)
        if url:
            lines.append(f"{i}. [{title}]({url}) — score `{score:.2f}`")
        else:
            lines.append(f"{i}. {title} — score `{score:.2f}`")
    return "\n".join(lines)


def chat_fn(message: str, history: list[tuple[str, str]]) -> str:
    result = retriever.answer(message)
    body = result["answer"]
    sources = _format_sources(result.get("sources", []))
    if sources:
        return f"{body}\n\n{sources}"
    return body


def main() -> None:
    demo = gr.ChatInterface(
        fn=chat_fn,
        title="RAC — SSA FAQ Chatbot",
        description=(
            "Retrieval-Augmented Chatbot trained on Social Security Administration FAQs. "
            "**Demo only — not official SSA guidance.** Verify at "
            "[ssa.gov](https://www.ssa.gov)."
        ),
        examples=[
            "How do I apply for Social Security retirement benefits?",
            "What is my full retirement age?",
            "How do I replace a lost Social Security card?",
            "What's the difference between SSDI and SSI?",
            "Are Social Security benefits taxable?",
        ],
    )
    demo.launch()


if __name__ == "__main__":
    main()
