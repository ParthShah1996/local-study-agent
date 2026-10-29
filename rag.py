# ---------------------------------------------------------------------------- #
#                                    IMPORTS                                   #
# ---------------------------------------------------------------------------- #
import re

import chromadb
import ollama

from config import *

SYSTEM_PROMPT = """You are a study tutor.
Answer ONLY from the CONTEXT given below.
Use simple, short, easy English.
After each fact, add the source number like [1] or [2].
If the context does not answer the question, say:
"I can not find this in your documents."
Do not make up facts."""


# -------------------------- GET COLLECTION FUNCTION ------------------------- #
_collection = None

def get_collection():
    """Open the Chroma collection one time and reuse it."""
    global _collection # Why do we define to _collection variables
    if _collection is None:
        client = chromadb.PersistentClient(path = str(DB_DIR))
        _collection = client.get_collection(COLLECTION)
    return _collection


# ------------------------------ CHUNK RETREIVAL ----------------------------- #
def retreive(question):
    """Return the closest chunks as a list of dicts."""
    col = get_collection()
    q_emb = ollama.embed(
        model=EMBED_MODEL, input="search_query: " + question
        )["embeddings"][0]
    res = col.query(
        query_embeddings=[q_emb],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, m, d in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        chunks.append(
            {
            "text": doc,
            "source": m["source"],
            "kind": m["kind"],
            "location": m["location"],
            "is_code": bool(m["is_code"]),
            "score": 1 -d,
            }
        )
    return chunks

# --------------------------- CHAT MESSAGE BUILDER --------------------------- #
def build_messages(question, chunks):
    """Make the numbered context and the chat messages"""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        text = f"```python\n{c['text']}\n```" if c["is_code"] else c["text"]
        blocks.append(f"{i} ({c['source']}, {c['kind']} {c['location']})\n{text}")
    context = "\n\n".join(blocks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"},
    ]

# ---------------------------- STREAM CHAT ANSWERS --------------------------- #
def stream_answer(question, chunks):
    """Yield the answer in small chunks"""
    stream = ollama.chat(
        model=CHAT_MODEL,
        messages=build_messages(question, chunks),
        options={"num_ctx": NUM_CTX, "temperature": 0.2},
        stream=True,
    )
    for part in stream:
        yield part["message"]["content"]

# ---------------------------- CITATION FORMATTING --------------------------- #
def format_sources(answer, chunks):
    """Show the sources the answers used. If none, show the closest matches."""
    used = []
    if "can not find" not in answer.lower():
        used = sorted(
            {int(n) for n in re.findall(r"\[(d+)]", answer) if 1 <= int(n) <= len(chunks)}
        )
    if used:
        title = "Sources:"
        numbers = used
    else:
        title = "Closest matches (not used in the answer): "
        numbers = list(range(1, len(chunks) + 1))

    lines = [title]
    for n in numbers:
        c = chunks[n - 1]
        lines.append(f"- [{n}] {c['source']}, {c['kind']} {c['location']} (match: {c['score']:.2f})")
    return "\n".join(lines)