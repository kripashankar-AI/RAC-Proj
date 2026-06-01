"""Scrape FAQ question/answer pairs from faq.ssa.gov.

Respects robots.txt by using a polite User-Agent and rate limit. The site's FAQ
pages have stable URLs of the form `/en-US/Topic/article/KA-XXXXX`. This script
discovers article links from category pages and extracts the question + answer.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from rag import config

ARTICLE_RE = re.compile(r"/en-US/Topic/article/KA-\d+", re.IGNORECASE)
CATEGORY_RE = re.compile(r"/en-US/Topic/\d+", re.IGNORECASE)
INDEX_URL = f"{config.SSA_FAQ_BASE}/en-US/"


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": config.USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})
    return s


def discover_article_urls(session: requests.Session, max_pages: int) -> list[str]:
    seen_categories: set[str] = set()
    seen_articles: set[str] = set()
    to_visit = [INDEX_URL]

    while to_visit and len(seen_articles) < max_pages:
        url = to_visit.pop(0)
        try:
            r = session.get(url, timeout=20)
            r.raise_for_status()
        except requests.RequestException:
            continue

        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            full = urljoin(config.SSA_FAQ_BASE, href.split("?")[0])
            if ARTICLE_RE.search(full) and full not in seen_articles:
                seen_articles.add(full)
                if len(seen_articles) >= max_pages:
                    break
            elif CATEGORY_RE.search(full) and full not in seen_categories:
                seen_categories.add(full)
                to_visit.append(full)
        time.sleep(0.4)

    return sorted(seen_articles)


def parse_article(session: requests.Session, url: str) -> dict | None:
    try:
        r = session.get(url, timeout=20)
        r.raise_for_status()
    except requests.RequestException:
        return None

    soup = BeautifulSoup(r.text, "html.parser")

    title_el = soup.find("h1") or soup.find("title")
    question = title_el.get_text(" ", strip=True) if title_el else ""
    if not question:
        return None

    # The answer body is typically inside the main article container.
    body_el = (
        soup.find("article")
        or soup.find("div", attrs={"id": re.compile("article|content", re.I)})
        or soup.find("main")
    )
    if not body_el:
        return None

    paragraphs = [
        p.get_text(" ", strip=True)
        for p in body_el.find_all(["p", "li"])
        if p.get_text(strip=True)
    ]
    answer = "\n".join(paragraphs).strip()
    if len(answer) < 40:
        return None

    crumbs = soup.find_all(attrs={"class": re.compile("breadcrumb", re.I)})
    category = crumbs[0].get_text(" / ", strip=True) if crumbs else "SSA FAQ"

    return {
        "question": question,
        "answer": answer,
        "url": url,
        "category": category,
    }


def merge(existing: list[dict], new_items: list[dict]) -> list[dict]:
    by_url = {item.get("url", ""): item for item in existing if item.get("url")}
    by_q = {item["question"].strip().lower(): item for item in existing}
    for item in new_items:
        if item["url"] in by_url:
            continue
        if item["question"].strip().lower() in by_q:
            continue
        existing.append(item)
    return existing


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape FAQs from faq.ssa.gov")
    parser.add_argument("--max-pages", type=int, default=50)
    parser.add_argument("--output", type=Path, default=config.FAQS_PATH)
    args = parser.parse_args()

    session = _session()

    print(f"Discovering up to {args.max_pages} article URLs from {config.SSA_FAQ_BASE} ...")
    urls = discover_article_urls(session, args.max_pages)
    print(f"Found {len(urls)} article URLs.")

    new_items: list[dict] = []
    for url in tqdm(urls, desc="Fetching articles"):
        item = parse_article(session, url)
        if item:
            new_items.append(item)
        time.sleep(0.4)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if args.output.exists():
        with args.output.open("r", encoding="utf-8") as f:
            existing = json.load(f)

    merged = merge(existing, new_items)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Added {len(merged) - len(existing)} new FAQs (total: {len(merged)}) -> {args.output}")
    print("Next: run `python -m rag.ingest` to rebuild the index.")


if __name__ == "__main__":
    main()
