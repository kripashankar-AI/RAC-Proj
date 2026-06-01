"""Gradio dashboard for the SSA FAQ RAG chatbot.

Three menu tabs:
  - RAG                 : retrieval-augmented chatbot (model v1: MiniLM)
  - New RAG new model   : same chatbot using a different embedding model (v2: MPNet)
  - Analyser Visuals    : upload an image + question, get an answer about it
"""
from __future__ import annotations

import gradio as gr

from rag import config
from rag import retriever
from rag import vision


CSS = """
.gradio-container { max-width: 1100px !important; }
#title { text-align: center; }
"""


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


def _make_chat_fn(version: str):
    def chat_fn(message: str, history: list[tuple[str, str]]) -> str:
        result = retriever.answer(message, version=version)
        body = result["answer"]
        sources = _format_sources(result.get("sources", []))
        return f"{body}\n\n{sources}" if sources else body

    return chat_fn


EXAMPLES = [
    "How do I apply for Social Security retirement benefits?",
    "What is my full retirement age?",
    "How do I replace a lost Social Security card?",
    "What's the difference between SSDI and SSI?",
    "Are Social Security benefits taxable?",
]


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="RAG — SSA FAQ Dashboard") as demo:
        gr.Markdown(
            "# RAG — SSA FAQ Dashboard\n"
            "Retrieval-Augmented Generation chatbot trained on Social Security "
            "Administration FAQs. **Demo only — not official SSA guidance.** "
            "Verify at [ssa.gov](https://www.ssa.gov).",
            elem_id="title",
        )

        with gr.Tabs():
            with gr.Tab("RAG"):
                gr.Markdown(f"_Model:_ `{config.EMBEDDING_MODEL}`")
                gr.ChatInterface(fn=_make_chat_fn("v1"), examples=EXAMPLES)

            with gr.Tab("New RAG new model"):
                gr.Markdown(
                    f"_Model:_ `{config.EMBEDDING_MODEL_V2}` — a larger, "
                    "higher-quality sentence encoder. Build its index once "
                    "with `python -m rag.ingest --version v2`."
                )
                gr.ChatInterface(fn=_make_chat_fn("v2"), examples=EXAMPLES)

            with gr.Tab("Analyser Visuals"):
                gr.Markdown(
                    "### Analyse a chart, plot, or screenshot\n"
                    "Upload any image and ask a question about it. With "
                    "`OPENAI_API_KEY` set, the dashboard uses a vision model "
                    "to read values, axes, and trends. Without a key, it "
                    "falls back to a basic image descriptor."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        image_in = gr.Image(label="Visual", type="filepath", height=360)
                        vquestion_in = gr.Textbox(
                            label="Question",
                            placeholder="e.g. Which category has the most FAQs?",
                            lines=2,
                        )
                        analyse_btn = gr.Button("Analyse", variant="primary")
                    with gr.Column(scale=1):
                        analysis_out = gr.Markdown(
                            "_Upload an image and click **Analyse**._"
                        )

                gr.Examples(
                    examples=[
                        "Summarise this chart in two sentences.",
                        "Which category has the highest count?",
                        "What is the highest score and which FAQ does it belong to?",
                        "What trend does this visual show?",
                        "Are there any outliers? If so, which ones?",
                    ],
                    inputs=[vquestion_in],
                    label="Example questions",
                )

                analyse_btn.click(
                    fn=vision.analyse,
                    inputs=[image_in, vquestion_in],
                    outputs=analysis_out,
                )

    return demo


def main() -> None:
    build_ui().launch(theme=gr.themes.Soft(), css=CSS)


if __name__ == "__main__":
    main()
