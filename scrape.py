"""
scrape.py - Scrapes Incerro website pages and saves them as text files.
"""
import os
import re
import requests
from bs4 import BeautifulSoup

# Mapping of output filename -> source URL
URL_MAP = {
    "home.txt": "https://www.incerro.ai/",
    "services.txt": "https://www.incerro.ai/services",
    "about-us.txt": "https://www.incerro.ai/about-us",
    "geographic-intelligence.txt": "https://www.incerro.ai/products/geographic-intelligence",
    "4sight.txt": "https://www.incerro.ai/products/4sight",
    "document-intelligence.txt": "https://www.incerro.ai/products/document-intelligence",
    "data-intelligence.txt": "https://www.incerro.ai/products/data-intelligence",
    "financial-intelligence.txt": "https://www.incerro.ai/products/financial-intelligence",
    "mvp-insight.txt": "https://www.incerro.ai/insights/the-new-rule-for-mvps-in-2026-build-to-learn-not-just-launch",
    "contact-us.txt": "https://www.incerro.ai/contact-us",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Save URL map as metadata for ingestion
URL_MAP_FILE = os.path.join(DATA_DIR, "url_map.txt")


def clean_text(text: str) -> str:
    """Remove excessive whitespace and blank lines."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            cleaned.append(stripped)
    return "\n".join(cleaned)


def scrape_page(url: str) -> str | None:
    """Fetch and extract main text content from a URL."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"ERROR: Failed to fetch {url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove non-content tags
    for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Try to get main content
    main = soup.find("main") or soup.find("article") or soup.find("body")
    if main is None:
        main = soup

    text = main.get_text(separator="\n")
    text = clean_text(text)
    return text


def main():
    # Save URL map for ingestion
    with open(URL_MAP_FILE, "w", encoding="utf-8") as f:
        for filename, url in URL_MAP.items():
            f.write(f"{filename}|{url}\n")

    print(f"Scraping {len(URL_MAP)} pages...\n")

    low_content_threshold = 200  # characters

    for filename, url in URL_MAP.items():
        print(f"Scraping: {url}")
        text = scrape_page(url)

        if text is None:
            print(f"  SKIPPED (fetch error)\n")
            continue

        if len(text) < low_content_threshold:
            print(f"  WARNING: Low content extracted from {url}. Page may require JavaScript rendering.")
            print(f"  Extracted {len(text)} characters — skipping.\n")
            continue

        out_path = os.path.join(DATA_DIR, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"SOURCE_URL: {url}\n")
            f.write(f"PAGE_FILE: {filename}\n\n")
            f.write(text)

        print(f"  Saved {len(text)} chars -> {filename}\n")

    print("Scraping complete.")


if __name__ == "__main__":
    main()
