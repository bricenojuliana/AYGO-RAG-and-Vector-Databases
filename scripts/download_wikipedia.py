"""Download the sample knowledge base used by the notebook.

Fetches the plain-text version of a few English Wikipedia articles through the
MediaWiki API and saves each one as a .txt file in data/wikipedia/. The files
are committed to the repository, so this script only needs to be run again to
refresh them.

Usage:
    python scripts/download_wikipedia.py
"""

import re
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


DISPLAYSTYLE = re.compile(r"\s*\{\\displaystyle (.*)\}\s*")
# A formula rendered by `explaintext` is a run of lines that are empty or indented
# by 2+ spaces (one MathML symbol per line), ending in its LaTeX source.
FORMULA_LINE = re.compile(r"(?: {2,}.*| *)")


def clean_math(text: str) -> str:
    """Replace each formula's one-symbol-per-line MathML dump with its LaTeX source.

    Inline formulas become `$latex$` inside their sentence; formulas that stand
    on their own line become `$$latex$$` on that line.
    """
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        if not FORMULA_LINE.fullmatch(lines[i]):
            out.append(lines[i])
            i += 1
            continue
        j = i
        while j < len(lines) and FORMULA_LINE.fullmatch(lines[j]):
            j += 1
        run = lines[i:j]
        latex = [m.group(1) for line in run if (m := DISPLAYSTYLE.fullmatch(line))]
        if not latex:  # ordinary blank lines between paragraphs
            out.extend(run)
            i = j
            continue
        # Empty lines at the edges of the run are paragraph breaks, not part of the formula.
        lead = next(k for k, line in enumerate(run) if line)
        trail = len(run) - next(k for k, line in enumerate(reversed(run)) if line)
        out.extend(run[:lead])
        inline = lead == 0 and trail == len(run) and out and out[-1].strip() and j < len(lines)
        if inline:
            following = lines[j]
            glue = "" if following[:1] in ".,;:)" else " "
            out[-1] = f"{out[-1].rstrip()} {' '.join(f'${x}$' for x in latex)}{glue}{following.lstrip()}"
            i = j + 1
        else:
            out.extend(f"$${x}$$" for x in latex)
            out.extend(run[trail:])
            i = j
    return "\n".join(out)


def slugify(title: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in title]
    return "_".join(filter(None, "".join(keep).split("_")))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for title in ARTICLES:
        canonical, text = fetch_article(title)
        text = clean_math(text)
        url = f"https://en.wikipedia.org/wiki/{canonical.replace(' ', '_')}"
        path = OUTPUT_DIR / f"{slugify(canonical)}.txt"
        # The first two lines carry the title and the source URL; the notebook
        # reads them back as document metadata.
        path.write_text(f"Title: {canonical}\nSource: {url}\n\n{text}\n", encoding="utf-8")
        print(f"{path.name:55s} {len(text):>7,d} chars")


if __name__ == "__main__":
    main()
