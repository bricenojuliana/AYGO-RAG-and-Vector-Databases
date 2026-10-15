"""Download the sample knowledge base used by the notebook.

Fetches the plain-text version of a few English Wikipedia articles through the
MediaWiki API and saves each one as a .txt file in data/wikipedia/. The files
are committed to the repository, so this script only needs to be run again to
refresh them.

Usage:
    python scripts/download_wikipedia.py
"""

from pathlib import Path

import requests

ARTICLES = [
    "Retrieval-augmented generation",
    "Vector database",
    "Word embedding",
    "Large language model",
    "Transformer (deep learning architecture)",
    "Semantic search",
]

API_URL = "https://en.wikipedia.org/w/api.php"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "wikipedia"
# Wikipedia asks API clients to identify themselves with a descriptive User-Agent.
HEADERS = {"User-Agent": "AYGO-RAG-Workshop/1.0 (student project; educational use)"}


def fetch_article(title: str) -> tuple[str, str]:
    """Return (canonical_title, plain_text) for a Wikipedia article."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "titles": title,
    }
    response = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
    response.raise_for_status()
    page = next(iter(response.json()["query"]["pages"].values()))
    return page["title"], page["extract"]


def slugify(title: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in title]
    return "_".join(filter(None, "".join(keep).split("_")))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for title in ARTICLES:
        canonical, text = fetch_article(title)
        url = f"https://en.wikipedia.org/wiki/{canonical.replace(' ', '_')}"
        path = OUTPUT_DIR / f"{slugify(canonical)}.txt"
        # The first two lines carry the title and the source URL; the notebook
        # reads them back as document metadata.
        path.write_text(f"Title: {canonical}\nSource: {url}\n\n{text}\n", encoding="utf-8")
        print(f"{path.name:55s} {len(text):>7,d} chars")


if __name__ == "__main__":
    main()
