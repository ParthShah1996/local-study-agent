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