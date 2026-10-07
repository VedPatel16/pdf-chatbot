"""Phase 2: find the chunks that match a question and let the LLM answer from them."""
import os
import re

from dotenv import load_dotenv
from groq import Groq

from utils.pdf_indexer import get_embedder

load_dotenv()

TOP_K = 4
HISTORY_MESSAGES = 10
TEMPERATURE = 0.2
NOT_FOUND = "I could not find this information in the uploaded document."
_PAGES_LINE = re.compile(r"^[ \t>*_-]*pages used\W*(.*)$", re.IGNORECASE | re.MULTILINE)


def _split_answer(raw, retrieved_pages):
    """Separate the answer text from the 'Pages used: ...' line the model adds.

    Returns (answer, pages_actually_used). If the model forgot the line, fall back
    to every retrieved page. A 'not found' answer has no sources.
    """
    matches = list(_PAGES_LINE.finditer(raw))
    if matches:
        last = matches[-1]
        answer = raw[: last.start()].strip()
        used = {int(n) for n in re.findall(r"\d+", last.group(1))} & set(retrieved_pages)
    else:
        answer = raw.strip()
        used = set(retrieved_pages)
    if "could not find this information" in answer.lower():
        used = set()
    return answer, sorted(used)


def _get_client():
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key == "paste_your_key_here":
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add your key to the .env file and restart the app."
        )
    return Groq(api_key=api_key)


def _search_text(question, history):
    """Short follow-ups like 'explain that' carry no topic, so add the previous question."""
    if history and len(question.split()) <= 6:
        previous = [m["content"] for m in history if m["role"] == "user"]
        if previous:
            return previous[-1] + " " + question
    return question


def ask(question, collection, history):
    """Return (answer, source_pages, source_chunks), using only the pages the answer relied on.

    source_chunks is a list of (page_number, text) used to build the answer.
    """
    client = _get_client()
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()

    # 1. find the most related chunks
    q_vector = get_embedder().encode(
        [_search_text(question, history)], show_progress_bar=False
    ).tolist()
    found = collection.query(
        query_embeddings=q_vector, n_results=min(TOP_K, collection.count())
    )
    docs = found["documents"][0]
    pages = [m["page"] for m in found["metadatas"][0]]
    context = "\n\n".join(f"[Page {p}]\n{d}" for p, d in zip(pages, docs))

    # 2. build the prompt
    system_prompt = (
        "You are a helpful assistant that answers questions about a PDF document.\n"
        "Use ONLY the context below to answer. Do not use outside knowledge.\n"
        f"If the answer is not in the context, reply exactly: \"{NOT_FOUND}\"\n"
        "Keep the answer clear and to the point, in simple language.\n"
        "After the answer, add a last line in exactly this format: Pages used: 2, 3\n"
        "It must list only the page numbers (from the [Page N] labels) that contain "
        "the information you used. If you could not find the answer, write: Pages used: none\n\n"
        "Context:\n" + context
    )
    recent = [
        {"role": m["role"], "content": m["content"]}
        for m in history[-HISTORY_MESSAGES:]
    ]

    # 3. ask the LLM
    try:
        reply = client.chat.completions.create(
            model=model,
            temperature=TEMPERATURE,
            messages=[{"role": "system", "content": system_prompt}]
            + recent
            + [{"role": "user", "content": question}],
        )
    except Exception as e:
        raise RuntimeError(f"The LLM request failed: {e}")

    raw = (reply.choices[0].message.content or "").strip()
    if not raw:
        raise RuntimeError("The model returned an empty answer. Please try again.")
    answer, used_pages = _split_answer(raw, pages)
    used_chunks = [(p, d) for p, d in zip(pages, docs) if p in used_pages]
    return answer, used_pages, used_chunks
