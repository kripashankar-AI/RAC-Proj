# RAC — Retrieval-Augmented Chatbot for SSA FAQs

A small, self-contained Retrieval-Augmented Chatbot (RAC) trained on U.S. Social
Security Administration (SSA) frequently asked questions.

It uses:

- **sentence-transformers** (`all-MiniLM-L6-v2`) for embeddings — runs locally, no API key needed.
- **FAISS** for fast similarity search.
- An optional **OpenAI** layer to rephrase the retrieved answer in natural language.
- A small **scraper** that pulls public FAQs from `https://faq.ssa.gov` so you can grow the corpus.

> Disclaimer: This project is a technical demo. It is **not** affiliated with the SSA
> and its answers are not official guidance. Always verify with [ssa.gov](https://www.ssa.gov).

---

## Quick start

```powershell
# 1. Create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Scrape more FAQs from faq.ssa.gov into data/ssa_faqs.json
python -m rac.scraper --max-pages 50

# 4. Build the vector index
python -m rac.ingest

# 5. Chat in the terminal
python -m rac.cli

# 6. Or launch the web UI
python -m rac.app
```

Type a question like:

```
> How do I apply for Social Security retirement benefits?
> What is my full retirement age?
> How do I replace a lost Social Security card?
```

---

## Project layout

```
RAC-Proj/
├── data/
│   └── ssa_faqs.json        # FAQ corpus (seed + scraped)
├── index/                   # FAISS index + metadata (generated)
├── rac/
│   ├── __init__.py
│   ├── config.py
│   ├── ingest.py            # builds the FAISS index
│   ├── retriever.py         # query → top-k FAQs → answer
│   ├── scraper.py           # pulls FAQs from faq.ssa.gov
│   ├── cli.py               # terminal chat
│   └── app.py               # Gradio web UI
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

Append entries (or run the scraper), then re-run `python -m rac.ingest`.

## Optional: nicer answers via OpenAI

Set `OPENAI_API_KEY` in your environment and the retriever will use the model
configured in `rac/config.py` to rephrase the top retrieved answer. Without a key,
it returns the retrieved answer verbatim.
