"""Visual question answering — analyse an uploaded image and answer questions about it.

Uses OpenAI vision (gpt-4o family) when ``OPENAI_API_KEY`` is set; otherwise falls
back to a lightweight PIL-based descriptor (size, mode, dominant colors).
"""
from __future__ import annotations

import base64
import mimetypes
from collections import Counter
from pathlib import Path

from PIL import Image

from rac import config


# ---------- helpers ----------------------------------------------------------

def _encode_data_uri(image_path: str) -> str:
    mime, _ = mimetypes.guess_type(image_path)
    if not mime:
        mime = "image/png"
    data = Path(image_path).read_bytes()
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _dominant_colors(img: Image.Image, k: int = 5) -> list[tuple[tuple[int, int, int], float]]:
    small = img.convert("RGB").resize((96, 96))
    pixels = list(small.getdata())
    counts = Counter(pixels)
    total = sum(counts.values()) or 1
    return [(rgb, count / total) for rgb, count in counts.most_common(k)]


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


# ---------- OpenAI vision ----------------------------------------------------

def _openai_vision(image_path: str, question: str) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        return "_(openai package not installed)_"

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    data_uri = _encode_data_uri(image_path)
    system = (
        "You are a data-visualization analyst. Given a chart, plot, or "
        "screenshot, answer the user's question precisely. Quote exact "
        "values, axis labels, and categories you can read. If something is "
        "not visible in the image, say so explicitly. Keep the answer concise."
    )
    try:
        resp = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                },
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content or "(empty response)"
    except Exception as exc:
        return f"OpenAI vision call failed: `{exc}`"


# ---------- offline fallback -------------------------------------------------

def _basic_analysis(image_path: str, question: str) -> str:
    img = Image.open(image_path)
    w, h = img.size
    mode = img.mode
    colors = _dominant_colors(img)

    swatches = " ".join(
        f"`{_rgb_to_hex(rgb)}` ({pct * 100:.1f}%)" for rgb, pct in colors
    )
    note = (
        "_No `OPENAI_API_KEY` set, so I can only describe the image's basic "
        "properties. Set the env var and restart for full visual Q&A._"
    )
    return (
        f"**Question:** {question or '(no question — describing the image)'}\n\n"
        f"**Image properties**\n"
        f"- Dimensions: `{w} × {h}` px\n"
        f"- Mode: `{mode}`\n"
        f"- Dominant colors: {swatches}\n\n"
        f"{note}"
    )


# ---------- public API -------------------------------------------------------

def analyse(image_path: str | None, question: str) -> str:
    if not image_path:
        return "Please upload an image first."
    question = (question or "").strip() or "Describe this visual and its key insights."
    if config.OPENAI_API_KEY:
        return _openai_vision(image_path, question)
    return _basic_analysis(image_path, question)
