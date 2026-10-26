# ---------------------------------------------------------------------------- #
#                                    IMPORTS                                   #
# ---------------------------------------------------------------------------- #
import chromadb
import fitz
import ollama
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

MONO_WORDS = ("mono", "courier", "consolas", "menlo", "typewriter", "code", "fixed")

def is_mono(font_name):
    f = font_name.lower()
    return any(w in f for w in MONO_WORDS)

    