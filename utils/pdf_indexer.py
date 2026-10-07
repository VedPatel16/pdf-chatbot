"""Phase 1: read a PDF, split it into chunks, and store them in a vector database."""
import uuid
from functools import lru_cache

import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# One in-memory database for the whole app (data is lost when the app closes).
# Every upload gets its own uniquely named collection, so several users
# on a public website never overwrite each other's data.
_client = chromadb.Client()


@lru_cache(maxsize=1)
def get_embedder():
    """Load the embedding model once. The first run downloads it (about 90 MB)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def drop_collection(name):
    """Delete a collection to free memory (ignored if it does not exist)."""
    if not name:
        return
    try:
        _client.delete_collection(name)
    except Exception:
        pass


def index_pdf(uploaded_file, old_collection_name=None):
    """Read the PDF, split it, embed the chunks, and store them.

    old_collection_name: this user's previous collection, deleted after the new one is ready.

    Returns (collection, number_of_pages, number_of_chunks).
    Raises ValueError if the PDF is unreadable or contains no text.
    """
    try:
        reader = PdfReader(uploaded_file)
        page_list = list(reader.pages)
    except Exception:
        raise ValueError("This file could not be read as a PDF.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    chunks, pages = [], []
    for number, page in enumerate(page_list, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        for piece in splitter.split_text(text):
            if piece.strip():
                chunks.append(piece)
                pages.append(number)

    if not chunks:
        raise ValueError(
            "No text could be read from this PDF. It may be a scanned "
            "(image-only) PDF, which this version does not support."
        )

    vectors = get_embedder().encode(chunks, show_progress_bar=False).tolist()

    collection = _client.create_collection(
        "pdf_" + uuid.uuid4().hex, metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=vectors,
        metadatas=[{"page": p} for p in pages],
    )
    drop_collection(old_collection_name)  # a new upload replaces this user's old data
    return collection, len(page_list), len(chunks)
