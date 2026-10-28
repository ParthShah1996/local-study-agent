# ---------------------------------------------------------------------------- #
#                                    IMPORTS                                   #
# ---------------------------------------------------------------------------- #
import math
import re

import chromadb
import ollama
import pymupdf
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from config import *

# ---------------------------------------------------------------------------- #
#                               DEFINE SPLITTERS                               #
# ---------------------------------------------------------------------------- #
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)

code_splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.PYTHON, chunk_size = CHUNK_SIZE, chunk_overlap = CHUNK_OVERLAP
)

# ---------------------------------------------------------------------------- #
#                               HELPER FUNCTIONS                               #
# ---------------------------------------------------------------------------- #
MONO_WORDS = (
    "mono", "courier", "consolas", "menlo", "typewriter", "code", "fixed"
)

def is_mono(font_name):
    f = font_name.lower()
    return any(w in f for w in MONO_WORDS)

def add_segments(segments, is_code, text):
    """Join neighbor pieces of the same type. """
    if segments and segments[-1][0] == is_code:
        segments[-1] = (is_code, segments[-1][1] + "\n" + text)
    else:
        segments.append((is_code, text))

def page_segments(page):
    """Split one PDF page into [(is_code, text), ...] inn reading order"""
    segments = []
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0: #skips images
            continue

        lines = [] # (is_code, text, x_start, font_size)
        for line in block["lines"]:
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = "".join(s["text"] for s in line["spans"]).rstrip()
            all_chars = sum(len(s["text"]) for s in spans)
            mono_chars = sum(len(s["text"]) for s in spans if is_mono(s["font"]))
            lines.append(
                (mono_chars > 0.6 * all_chars, text, line["bbox"][0], spans[0]["size"])
            )

        # group neighbor lines with the same type
        i = 0
        while i < len(lines):
            j = i
            while j < len(lines) and lines[j][0] == lines[i][0]:
                j += 1
            group = lines[i:j]
            if group[0][0]: # code group: rebuild indentation from x position
                min_x = min(g[2] for g in group)
                fixed = []
                for _, text, x, size in group:
                    if text[:1].isspace():
                        fixed.append(text)
                    else:
                        indent = round((x - min_x) / (size * 0.6))
                        fixed.append(" " * indent + text)
                add_segments(segments, True, "\n".join(fixed))
            else:
                add_segments(segments, False, "\n".join(g[1] for g in group))
            i = j
    return segments

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text):
    text = CONTROL_CHARS.sub("", text)
    # drop broken unicode characters
    return text.encode("utf-8", "ignore").decode("utf-8")


def bad_vector(vec):
    return any(not math.isfinite(x) for x in vec)

# ---------------------------------------------------------------------------- #
#                             DATA LOADER FUNCTIONS                            #
# ---------------------------------------------------------------------------- #

def load_pdf(path):
    chunks = []
    name = str(path.relative_to(DOCS_DIR))
    with pymupdf.open(path) as pdf:
        for page_no, page in enumerate(pdf, start=1):
            for is_code, text in page_segments(page):
                text = text.strip("\n")
                
                # SKIP TRIVIAL CHUNKS: Ignore snippets smaller than 25 chars unless they look like code
                if len(text.strip()) < 25 and not (is_code and any(keyword in text for keyword in ["return", "def", "if", "for", "="])):
                    continue

                if is_code:
                    pieces = (
                        [text]
                        if len(text) <= MAX_CODE_CHUNK
                        else code_splitter.split_text(text)
                    )
                else:
                    pieces = text_splitter.split_text(text)

                for piece in pieces:
                    # Double check piece length
                    if len(piece.strip()) < 10:
                        continue
                    chunks.append(
                        {
                            "text": piece,
                            "source": name,
                            "kind": "page",
                            "location": page_no,
                            "is_code": is_code,
                        }
                    )
    return chunks

def load_py(path):
    chunks = []
    name = str(path.relative_to(DOCS_DIR))
    text = path.read_text(encoding="utf-8", errors = "ignore")
    docs = code_splitter.create_documents([text], add_start_index=True)
    for d in docs:
        line_no = text.count("\n", 0, d.metadata["start_index"]) + 1
        chunks.append(
            {
                "text": d.page_content,
                "source": name,
                "kind": "line",
                "location": line_no,
                "is_code": True,
            }
        )
    return chunks

# ---------------------------------------------------------------------------- #
#                                 MAIN FUNCTION                                #
# ---------------------------------------------------------------------------- #
def main():
    DOCS_DIR.mkdir(exist_ok=True)
    files = sorted(
        p for p in DOCS_DIR.rglob("*") if p.suffix.lower() in {".pdf", ".py"}
    )
    if not files:
        print("No .pdf or .py files found in the docs folder. Add some and try again.")
        return

    all_chunks = []
    for f in files:
        print("Reading:", f.relative_to(DOCS_DIR))
        chunks = load_pdf(f) if f.suffix.lower() == ".pdf" else load_py(f)
        if not chunks:
            print("  Warning: no text found (maybe a scanned PDF).")
        all_chunks.extend(chunks)

    # clean every chunk and drop empty ones
    for c in all_chunks:
        c["text"] = clean_text(c["text"])
    all_chunks = [c for c in all_chunks if c["text"].strip()]

    code_count = sum(c["is_code"] for c in all_chunks)
    print(f"\nTotal chunks: {len(all_chunks)} (code chunks: {code_count})")

    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection(name=COLLECTION)  # rebuild from scratch each time
        print(f"Cleared old collection: {COLLECTION}")
    except Exception:  # noqa: BLE001, S110
        pass
    col = client.create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})

    saved = 0
    skipped = 0
    for start in range(0, len(all_chunks), BATCH):
        batch = all_chunks[start : start + BATCH]
        embeddings = ollama.embed(
            model=EMBED_MODEL,
            input=["search_document: " + c["text"] for c in batch],
        )["embeddings"]

        ids, vectors, documents, metadatas = [], [], [], []
        for i, c in enumerate(batch):
            if bad_vector(embeddings[i]):
                print(f"  Skipped chunk-{start + i}: bad embedding ({c['source']}, page {c['location']})")
                skipped += 1
                continue
            ids.append(f"chunk-{start + i}")
            vectors.append(embeddings[i])
            documents.append(c["text"])
            metadatas.append(
                {
                    "source": str(c["source"]),
                    "kind": str(c["kind"]),
                    "location": int(c["location"]),
                    "is_code": int(c["is_code"]),
                }
            )

        if ids:
            col.add(ids=ids, embeddings=vectors, documents=documents, metadatas=metadatas)
            saved += len(ids)
        print(f"  Saved {saved}/{len(all_chunks)}", flush=True)

    print(f"\nDone. Saved {col.count()} chunks. Skipped {skipped}.")
    print("Now run: python ask.py")

if __name__ == "__main__":
    main()