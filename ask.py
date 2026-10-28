# ---------------------------------------------------------------------------- #
#                                    IMPORTS                                   #
# ---------------------------------------------------------------------------- #
import sys

import chromadb
import ollama

from config import *

# ---------------------------------------------------------------------------- #
#                                 SYSTEM PROMPT                                #
# ---------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are a study tutor.
Answer ONLY from the CONTEXT given below.
Use simple, short, easy English.
After each fact, add the source number like [1] or [2].
If the context does not answer the question, say:
"I can not find this in your documents."
Do not make up facts."""

# ---------------------------------------------------------------------------- #
#                                 MAIN FUNCTION                                #
# ---------------------------------------------------------------------------- #
def main():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        col = client.get_collection(COLLECTION)
    except Exception:
        print("No database found. Run python ingest.py")
        sys.exit(1)

    print("Study agent ready. Type 'exit' to stop.")

    # Infinite loop to start chat
    while True:
        question = input("\nYou:").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        # 1. Turn the question into an embedding
        q_emb = ollama.embed(
            model=EMBED_MODEL, input="search_query: " + question
            )["embeddings"][0]

        # 2. Find the closest chunks
        res = col.query(
            query_embeddings=[q_emb],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"],
        )

        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]

        # 3. Build the context with source numbers
        blocks = []
        for i, (doc, m) in enumerate(zip(docs, metas), start = 1):
            blocks.append(f"[{i}]({m['source']} , {m['kind']} {m['location']})\n{doc}")
        context = "\n\n".join(blocks)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"},
        ]

        # 4. Ask the model and stream the answer
        print("\nTutor: ", end="", flush=True)
        stream = ollama.chat(
            model=CHAT_MODEL,
            messages=messages,
            options={"num_ctx": NUM_CTX, "temperature": 0.2},
            stream=True,
        )

        for part in stream:
            print(part["message"]["content"], end="", flush=True)

        # 5. Show sources
        print("\n\nSources:")
        for i, (m,d) in enumerate(zip(metas, dists), start = 1):
            print(f"  [{i}] {m['source']}, {m['kind']} {m['location']}  (match: {1 - d:.2f})")


if __name__ == "__main__":
    main()