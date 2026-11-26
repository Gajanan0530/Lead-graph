"""Download a few public National Archives pages and save them as plain text.

Why this exists: the project needs real source text, and it needs a record of
exactly where each piece came from (URL, time fetched, and a fingerprint of
the text). That record is the first link in our provenance chain.
"""
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "https://www.archives.gov/research/jfk/warren-commission-report/"

# The pages we are testing with. Chosen to be institutional and non-graphic.
SOURCES = [
    {"id": "wc_ch1", "title": "Warren Commission Report, Chapter 1: Summary and Conclusions",
     "url": BASE + "chapter-1.html"},
    {"id": "wc_ch6", "title": "Warren Commission Report, Chapter 6: Investigation of Possible Conspiracy",
     "url": BASE + "chapter-6.html"},
    {"id": "wc_app5", "title": "Warren Commission Report, Appendix 5: List of Witnesses",
     "url": BASE + "appendix5.html"},
]

# Paths are built from this file's location, so the script works from any folder.
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"          # raw text: kept OUT of git (see .gitignore)
MANIFEST = ROOT / "data" / "sources.json"  # the provenance record: kept IN git

# Identify ourselves politely to the website.
HEADERS = {"User-Agent": "lead-graph-hackathon/0.1 (student project; few polite requests)"}

# Everything after this line on the page is site footer, not report text.
STOP_MARKER = "This page was last reviewed"
# A site notice that appears inside the content area; we skip it.
SKIP_PREFIX = "The National Archives is conducting a review"
# Page elements that hold readable text.
BLOCK_TAGS = ["p", "li", "h2", "h3", "h4", "blockquote", "td", "th"]


def clean(text):
    """Turn odd spaces and line breaks into single spaces."""
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def extract_main_text(html):
    """Return the page title plus the text blocks that follow it."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()  # remove code that is not readable text
    h1 = soup.find("h1")
    if h1 is None:
        raise ValueError("Could not find the page title (h1).")
    lines = [clean(h1.get_text(" "))]
    for element in h1.find_all_next(BLOCK_TAGS):
        text = clean(element.get_text(" "))
        if not text:
            continue
        if text.startswith(STOP_MARKER):
            break  # reached the footer
        if text.startswith(SKIP_PREFIX):
            continue
        if text == lines[-1]:
            continue  # skip exact repeats from nested elements
        lines.append(text)
    return "\n".join(lines)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for source in SOURCES:
        print(f"Fetching {source['id']} ...")
        response = requests.get(source["url"], headers=HEADERS, timeout=30)
        response.raise_for_status()  # stop with a clear error if the page failed
        response.encoding = "utf-8"
        text = extract_main_text(response.text)

        out_path = RAW_DIR / f"{source['id']}.txt"
        out_path.write_text(text, encoding="utf-8")

        manifest.append({
            "id": source["id"],
            "title": source["title"],
            "url": source["url"],
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "characters": len(text),
            "words": len(text.split()),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })
        print(f"  saved {out_path.name}: {len(text.split())} words")
        print("  preview:", text[:200].replace("\n", " | "))
        time.sleep(2)  # be polite: pause between requests

    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Wrote provenance record:", MANIFEST)


if __name__ == "__main__":
    main()