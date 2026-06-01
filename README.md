# RAG — Retrieval-Augmented Generation for SSA FAQs

A small, self-contained Retrieval-Augmented Generation (RAG) chatbot trained on
U.S. Social Security Administration (SSA) frequently asked questions.

It uses:

- **sentence-transformers** for embeddings (two models supported, swappable per tab):
  - **v1** — `all-MiniLM-L6-v2` (small, fast)
  - **v2** — `all-mpnet-base-v2` (larger, higher quality) — used by the
    *New RAG new model* tab.
- **FAISS** for fast similarity search.
- An optional **OpenAI** layer to rephrase the retrieved answer in natural language.
- A small **scraper** that pulls public FAQs from `https://faq.ssa.gov`.
- An **Analyser Visuals** tab that answers questions about an uploaded chart or
  screenshot (uses OpenAI vision when `OPENAI_API_KEY` is set).

> Disclaimer: Technical demo. **Not** affiliated with SSA. Verify with
> [ssa.gov](https://www.ssa.gov).

---

## Quick start

```powershell
# 1. Create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Scrape more FAQs
python -m rag.scraper --max-pages 50

# 4. Build the vector indices (both models)
python -m rag.ingest                  # builds v1 + v2
# or just one:
# python -m rag.ingest --version v1
# python -m rag.ingest --version v2

# 5. Chat in the terminal
python -m rag.cli                     # v1
python -m rag.cli --version v2        # v2

# 6. Launch the dashboard
python -m rag.app
```

The dashboard ships three tabs:

| Tab                  | Purpose                                          |
| -------------------- | ------------------------------------------------ |
| RAG                  | Chatbot using the v1 embedding model             |
| New RAG new model    | Same chatbot using the v2 embedding model        |
| Analyser Visuals     | Upload an image + question and get an analysis   |

---

## Project layout

```
RAG-Proj/
├── data/
│   └── ssa_faqs.json        # FAQ corpus (seed + scraped)
├── index/                   # FAISS indices + metadata (generated)
│   ├── faqs.faiss / faqs_meta.json         # v1
│   └── faqs_v2.faiss / faqs_v2_meta.json   # v2
├── rag/
│   ├── __init__.py
│   ├── config.py
│   ├── ingest.py            # builds the FAISS indices
│   ├── retriever.py         # query → top-k FAQs → answer (v1 or v2)
│   ├── scraper.py           # pulls FAQs from faq.ssa.gov
│   ├── vision.py            # image Q&A backend
│   ├── cli.py               # terminal chat
│   └── app.py               # Gradio dashboard
├── requirements.txt
└── README.md
```

## Adding your own FAQs

`data/ssa_faqs.json` is a list of objects:

```json
[
  { "question": "…", "answer": "…", "url": "https://…", "category": "Retirement" }
]
```

Append entries (or run the scraper), then re-run `python -m rag.ingest`.

## Optional: nicer answers via OpenAI

Set `OPENAI_API_KEY`. The retriever uses the model from `rag/config.py`
(`gpt-4o-mini` by default) to rephrase the top retrieved answer and to power the
Analyser Visuals tab.
