# ---------------------------------------------------------------------------- #
#                                    IMPORTS                                   #
# ---------------------------------------------------------------------------- #
import re

import ollama
from ddgs import DDGS

from config import *

# ---------------------------------------------------------------------------- #
#                                HELPER FUNTIONS                               #
# ---------------------------------------------------------------------------- #
WEB_SYSTEM_PROMPT = """You are a study tutor.
The student's documents did not have the answer, so you can use WEB RESULT
Answer ONLY from the WEB RESULTS given below.
Use simple, short, easy English.
After each fact, add the source number like [1] or [2].
If the results do not answer the question, say:
"I can not find this on the web either."
Do not make up facts."""

def search(query):
    """Search the web. Only the query text is sent. Returns a list of dicts."""
    raw = DDGS().text(query, max_results=WEB_MAX_RESULTS)
    results = []
    for r in raw:
        results.append(
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "text": r.get("body" "")[:400] # keep it short for the 4096 context
            }
        )
    return results

def build_messages(question, results):
    blocks = [
        f"[{i}] {r['title']} ({r['url']})\n{r['text']}"
        for i, r in enumerate(results, start=1)
    ]
    context = "\n\n".join(blocks)
    return [
        {"role": "system", "content": WEB_SYSTEM_PROMPT},
        {"role": "user", "content": f"WEB RESULTS: \n{context}\n\n {question}"},
    ]

def stream_web_answer(question, results):
    """Yield the answer in small pieces."""
    stream = ollama.chat(
        model=CHAT_MODEL,
        messages=build_messages(question, results),
        options={"num_ctx": NUM_CTX, "temperature": 0.2},
        stream=True,
    )
    for part in stream:
        yield part["message"]["content"]

def format_web_sources(answer, results):
    """Show the web pages the answer used. If none, show what was checked."""
    used = sorted(
        int(n) for n in re.findall(r"\[(\d+)\]", answer) if 1 <= int(n) <= len(results) + 1
    )
    if used and "can not find" not in answer.lower():
        title, numbers = "Web sources:", used
    else:
        title, numbers = "Web results checked (not used in the answer):", range(1, len(results) + 1)
    
    lines = [title]
    for n in numbers:
        r = results[n - 1]
        lines.append(f"- [{n}] {r['title']} - {r['url']}")
    return "\n".join(lines)    