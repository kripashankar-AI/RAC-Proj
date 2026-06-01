"""Gradio dashboard for the SSA FAQ chatbot.

Two menu tabs:
  - RAC               : retrieval-augmented chatbot
  - Analyser Visuals  : upload an image + question, get an answer about it
"""
from __future__ import annotations

import gradio as gr

from rac import retriever
from rac import vision


CSS = """
.gradio-container { max-width: 1100px !important; }
#title { text-align: center; }
"""


# ---------- RAC tab ----------------------------------------------------------

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
    return f"{body}\n\n{sources}" if sources else body


# ---------- UI ---------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="RAC — SSA FAQ Dashboard") as demo:
        gr.Markdown(
            "# RAC — SSA FAQ Dashboard\n"
            "Retrieval-Augmented Chatbot trained on Social Security Administration FAQs. "
            "**Demo only — not official SSA guidance.** Verify at "
            "[ssa.gov](https://www.ssa.gov).",
            elem_id="title",
        )

        with gr.Tabs():
            with gr.Tab("RAC"):
                gr.ChatInterface(
                    fn=chat_fn,
                    examples=[
                        "How do I apply for Social Security retirement benefits?",
                        "What is my full retirement age?",
                        "How do I replace a lost Social Security card?",
                        "What's the difference between SSDI and SSI?",
                        "Are Social Security benefits taxable?",
                    ],
                )

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
                        image_in = gr.Image(
                            label="Visual",
                            type="filepath",
                            height=360,
                        )
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
