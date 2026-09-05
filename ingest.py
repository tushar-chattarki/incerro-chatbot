import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
CHROMA_DIR = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "incerro_docs"

CHUNK_SIZE = 1000       # target characters per chunk
CHUNK_OVERLAP = 120     # overlap characters


def load_url_map() -> dict[str, str]:
    """Load filename -> URL mapping saved by scrape.py."""
    url_map = {}
    map_file = DATA_DIR / "url_map.txt"
    if map_file.exists():
        with open(map_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "|" in line:
                    fname, url = line.split("|", 1)
                    url_map[fname.strip()] = url.strip()
    return url_map


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph breaks."""
    # Split on double newlines (paragraph breaks) first
    paragraphs = re.split(r"\n\n+", text)

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If adding this paragraph would exceed chunk_size, flush
        if current_chunk and len(current_chunk) + len(para) + 2 > chunk_size:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap from end of previous chunk
            current_chunk = current_chunk[-overlap:] + "\n\n" + para
        else:
            current_chunk = (current_chunk + "\n\n" + para).strip() if current_chunk else para

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # If a single paragraph is longer than chunk_size, split it by sentences
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > chunk_size * 1.5:
            # Split long chunks by sentence boundaries
            sentences = re.split(r"(?<=[.!?])\s+", chunk)
            sub_chunk = ""
            for sent in sentences:
                if sub_chunk and len(sub_chunk) + len(sent) + 1 > chunk_size:
                    final_chunks.append(sub_chunk.strip())
                    sub_chunk = sub_chunk[-overlap:] + " " + sent
                else:
                    sub_chunk = (sub_chunk + " " + sent).strip() if sub_chunk else sent
            if sub_chunk.strip():
                final_chunks.append(sub_chunk.strip())
        else:
            final_chunks.append(chunk)

    return [c for c in final_chunks if len(c) > 50]  # filter tiny fragments


def get_page_title(text: str, filename: str) -> str:
    """Extract a page title from the text or use filename."""
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if line and not line.startswith("SOURCE_URL") and not line.startswith("PAGE_FILE") and len(line) > 5:
            return line[:100]
    return filename.replace(".txt", "").replace("-", " ").title()


def strip_metadata_header(text: str) -> str:
    """Remove the SOURCE_URL / PAGE_FILE header lines added by scrape.py."""
    lines = text.split("\n")
    content_lines = []
    skip_empty = True
    for line in lines:
        if line.startswith("SOURCE_URL:") or line.startswith("PAGE_FILE:"):
            continue
        if skip_empty and not line.strip():
            continue
        skip_empty = False
        content_lines.append(line)
    return "\n".join(content_lines)


def main():
    import chromadb

    gemini_key = os.getenv("GEMINI_API_KEY")

    # Set up ChromaDB client
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection to avoid duplicate ingestion
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Cleared existing collection: {COLLECTION_NAME}")
    except Exception:
        pass

    # Set up embedding function
    # Use Chroma's default local embedding function (reliable, no API key needed)
    # Gemini embedding via ChromaDB wrapper has compatibility issues with current versions
    print("Using Chroma default local embedding function (sentence-transformers).")
    embedding_fn = None

    if embedding_fn:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn
        )
    else:
        collection = client.create_collection(name=COLLECTION_NAME)

    url_map = load_url_map()
    txt_files = sorted([f for f in DATA_DIR.iterdir() if f.suffix == ".txt" and f.name != "url_map.txt"])

    if not txt_files:
        print("ERROR: No .txt files found in data/. Run scrape.py first.")
        return

    total_chunks = 0

    for txt_file in txt_files:
        raw_text = txt_file.read_text(encoding="utf-8")
        source_url = url_map.get(txt_file.name, "https://www.incerro.ai/")

        # Strip the header metadata lines
        content = strip_metadata_header(raw_text)
        page_title = get_page_title(content, txt_file.name)
        chunks = chunk_text(content)

        if not chunks:
            print(f"{txt_file.name}: 0 chunks (skipped — insufficient content)")
            continue

        ids = [f"{txt_file.stem}_{i}" for i in range(len(chunks))]
        metadatas = [
            {"source_url": source_url, "page_title": page_title, "chunk_index": i}
            for i in range(len(chunks))
        ]

        collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        print(f"{txt_file.name}: {len(chunks)} chunks")
        total_chunks += len(chunks)

    print(f"\nTotal chunks ingested: {total_chunks}")
    print(f"Collection '{COLLECTION_NAME}' stored in: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
