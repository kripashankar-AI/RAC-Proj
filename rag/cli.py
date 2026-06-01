"""Interactive terminal chat against the SSA FAQ corpus."""
from __future__ import annotations

import argparse

from rag import retriever


BANNER = """\
RAG — SSA FAQ Chatbot
---------------------
Ask a question about U.S. Social Security. Type 'exit' or Ctrl+C to quit.
This is a demo and not official SSA guidance — verify at ssa.gov.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["v1", "v2"], default="v1")
    args = parser.parse_args()

    print(BANNER)
    print(f"(model version: {args.version})\n")
    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit", ":q"}:
            break

        result = retriever.answer(query, version=args.version)
        print(f"\nBot: {result['answer']}")

        sources = result.get("sources", [])
        if sources:
            print("\nSources:")
            for i, s in enumerate(sources, 1):
                url = s.get("url") or "(no url)"
                print(f"  [{i}] {s['question']}  —  {url}  (score={s['score']:.2f})")


if __name__ == "__main__":
    main()
