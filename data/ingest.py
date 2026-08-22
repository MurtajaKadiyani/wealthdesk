"""
data/ingest.py
--------------
Ingests BNB policy documents into ChromaDB.

Run independently of seed.py (no dependency between them):
    python data/ingest.py

What this script does:
  1. Reads all .md files from data/documents/
  2. Splits them into overlapping chunks (better retrieval precision)
  3. Generates embeddings using a free local model (all-MiniLM-L6-v2)
  4. Writes the vector store to data/vectorstore/ (persisted to disk)

First run: downloads ~90 MB of embedding model weights to ~/.cache/huggingface/.
Subsequent runs: uses the cache. No internet connection required after first run.

The script is idempotent -- it deletes and rebuilds the vector store on every run.
This mirrors seed.py's DROP TABLE / CREATE TABLE pattern. Participants can add or
edit a document and re-run ingest.py to make the change searchable, without any
leftover stale chunks from the previous run.

Why local embeddings (HuggingFace) rather than OpenAI embeddings?
  Using OpenAI for embeddings would require participants to have an OpenAI key
  from Session 1. We delay the OpenAI key requirement to Session 6 (eval judge).
  all-MiniLM-L6-v2 is fast, free, and produces good retrieval quality for
  English policy documents of this length.

Why are rates NOT in these documents?
  Rates are in data/bnb_data.db (see seed.py). Putting "home loan: 8.5%" in a
  document would create two sources of truth. When the rate changes (seed.py is
  re-run), the document would still say 8.5% while the database says something
  else. The compliance node (US-08, Session 9) specifically checks that the
  agent never retrieves stale rate information from documents.

Windows path note:
  ChromaDB's persist_directory requires a string. We pass str(VECTOR_DIR) where
  VECTOR_DIR is a pathlib.Path object. This gives the correct path on Windows
  without manual string manipulation.
"""

import shutil
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

DATA_DIR   = Path(__file__).parent
DOCS_DIR   = DATA_DIR / "documents"
VECTOR_DIR = DATA_DIR / "vectorstore"

EMBED_MODEL   = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 500   # characters per chunk
CHUNK_OVERLAP = 50    # overlap between consecutive chunks

# Smaller chunks (500 chars) give more precise retrieval but may split context
# across chunks. 50-char overlap ensures a sentence that straddles a boundary
# is preserved in at least one chunk.
#
# Splitting is header-aware (see split_documents()): we first cut on markdown
# "#"/"##" boundaries so a section like "## Required Documents" never gets
# fused with the unrelated paragraph before it, then only fall back to plain
# character splitting for sections that are still too long. Each resulting
# chunk is prefixed with its document title so retrieval can tell "personal
# loan" chunks apart from "home loan" chunks even when the wording overlaps.


def load_documents() -> List[Document]:
    """Load every .md file in the documents directory.

    Each document is tagged with its filename in metadata["source"].
    That tag appears in LangSmith traces (from Session 4 onward, when basic
    tracing is introduced with US-03) so you can see exactly which document
    contributed to each agent response.
    """
    if not DOCS_DIR.exists():
        print(f"Error: Documents directory not found at {DOCS_DIR}", file=sys.stderr)
        print("Make sure you are running this script from the wealthdesk/ folder.", file=sys.stderr)
        sys.exit(1)

    docs = []
    md_files = sorted(DOCS_DIR.glob("*.md"))
    if not md_files:
        print(f"Error: No .md files found in {DOCS_DIR}", file=sys.stderr)
        sys.exit(1)

    for path in md_files:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            print(f"  Warning: {path.name} is empty -- skipping", file=sys.stderr)
            continue
        doc = Document(page_content=text, metadata={"source": path.name})
        docs.append(doc)
        print(f"  Loaded: {path.name:40s} ({len(text):,} chars)")

    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    """Split documents into chunks for retrieval.

    Two passes:
      1. MarkdownHeaderTextSplitter cuts each document on "#"/"##" boundaries
         first, so a section (e.g. "## Required Documents") never gets fused
         with the tail of an unrelated section. This was the bug that made
         "documents required for personal loan" miss the actual document
         list: with plain character splitting, that section landed in a
         chunk mostly made of the previous section's text, which pulled its
         embedding away from "documents".
      2. Any section still longer than CHUNK_SIZE falls back to
         RecursiveCharacterTextSplitter (paragraph -> sentence -> character
         boundaries), so long FAQ categories still get precise sub-chunks.

    Every chunk is prefixed with "<doc title> > <section>" before embedding.
    This keeps the product name (e.g. "BNB Personal Loan Guide") attached to
    every chunk from that document, so a personal-loan query doesn't get
    outranked by a lexically-similar but wrong-product chunk (e.g. a home
    loan FAQ entry that also happens to say "documents" and "loan").
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2")],
        strip_headers=True,
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    chunks: List[Document] = []
    for doc in docs:
        source = doc.metadata["source"]
        for section in header_splitter.split_text(doc.page_content):
            title   = section.metadata.get("h1", "")
            heading = section.metadata.get("h2", "")
            prefix  = f"{title} > {heading}\n\n" if heading else f"{title}\n\n"
            body    = section.page_content.strip()
            if not body:
                continue

            bodies = (
                [body] if len(prefix) + len(body) <= CHUNK_SIZE
                else char_splitter.split_text(body)
            )
            for b in bodies:
                chunks.append(Document(
                    page_content=prefix + b,
                    metadata={"source": source, "section": heading or title},
                ))
    return chunks


def main() -> None:
    print("Ingesting BNB documents into ChromaDB")
    print(f"  Source : {DOCS_DIR}")
    print(f"  Target : {VECTOR_DIR}\n")

    # Delete the existing vector store before rebuilding. This makes the script
    # idempotent: running it twice produces the same result as running it once.
    # Without this, Chroma.from_documents() appends chunks to any existing
    # collection, causing duplicate retrieval results in Session 4.
    if VECTOR_DIR.exists():
        shutil.rmtree(VECTOR_DIR)
        print(f"  Cleared existing vector store (idempotent reset)\n")

    print("Loading documents...")
    docs = load_documents()

    chunks = split_documents(docs)
    print(f"\nSplit {len(docs)} documents into {len(chunks)} chunks")
    print(f"  chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}\n")

    print(f"Loading embedding model: {EMBED_MODEL}")
    print("  First run downloads ~90 MB. Subsequent runs use cache.\n")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    print("Building vector store...")
   # Configure Chroma's HNSW index to use cosine similarity.
#
# During retrieval, Chroma compares the query embedding with every stored
# embedding using this metric. Cosine measures the angle between vectors,
# making it well-suited for semantic text similarity.
#
# This metric becomes part of the HNSW index structure when the collection
# is created and cannot be changed later. Switching to a different metric
# (e.g., L2) requires rebuilding the vector store.
#
# For the WealthDesk corpus, cosine also produces similarity scores that
# separate meaningful queries from noise more cleanly, making thresholding
# (e.g., 0.3) easier to tune.
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )

    print(f"\nDone. {len(chunks)} chunks stored at {VECTOR_DIR}")
    print("Run 'python data/ingest.py' again after adding or editing documents.")
    print("\nSetup complete. Run 'python s01/solution/main.py' to start WealthDesk.")


if __name__ == "__main__":
    main()
