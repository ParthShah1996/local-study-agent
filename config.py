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

MIN_SCORE = 0.58          # best match below this = "not in my documents"
MAX_HISTORY_TURNS = 3     # how many old question+answer pairs the rewriter sees


WEB_SEARCH = True         # False = never use the internet
WEB_MAX_RESULTS = 5       # how many web results to read

PROJECT_DIR = Path(__file__).parent
CURRICULUM_FILE = PROJECT_DIR/ "curriculum.json"
PROGRESS_FILE = PROJECT_DIR/ "progress.json"

QUIZ_QUESTIONS = 5        # questions per quiz
TOPIC_CHUNKS = 4          # chunks used for a lesson or quiz
TOPIC_CHUNKS_CHARS = 1200 # cut each chunk to the length (keeps the 4096 context safe)
PASS_SCORE = 0.7          # 70% or more topic done