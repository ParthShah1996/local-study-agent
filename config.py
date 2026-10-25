from pathlib import Path

BASE = Path(__file__).parent
DOCS_DIR = BASE/ "docs" 
DB_DIR = BASE/ "chroma_db"

CHAT_MODEL = "qwen2.5:7b"
EMBED_MODEL = "nomic-embed-text"
COLLECTION = "study_docs" 

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MAX_CODE_CHUNK = 2000
TOP_K = 5
NUM_CTX = 4096 #SMALL CONTEXT SAVES VRAM
BATCH = 32

