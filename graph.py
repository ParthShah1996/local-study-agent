# ---------------------------------------------------------------------------- #
#                                    IMPORTS                                   #
# ---------------------------------------------------------------------------- #
import operator
from typing import Annotated, TypedDict

import ollama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

import rag
import web
from config import *

# ---------------------------------- PROMPTS --------------------------------- #
REFUSAL = "i can not find"
WEB_LABEL = "From the web (not from your documents):\n\n"

REWRITE_PROMPT = """Rewrite the follow-up question so it makes sense alone.
Use the chat history to replace words like "it","that", or "this one".
Do NOT answer the question.
If the question is already clear, return it unchanged.
Return only the question in one line.

CHAT HISTORY:
{history}

FOLLOW-UP QUESTION: {question}"""

# ---------------------------------------------------------------------------- #
#                                     STATE                                    #
# ---------------------------------------------------------------------------- #
class State(TypedDict):
    question: str       # what the user typed
    standalone: str     # the question after rewriting
    chunks: list        # chunks found in Chroma
    answer: str         # the final answer
    found: bool         # did the documents answer the question?
    web_results: list   # results from the web (empty if none)
    # "history" is the memory. operator.add means: ADD new items to the
    # old list. Without it the old list would be replaced.
    history: Annotated[list, operator.add]


# ---------------------------------------------------------------------------- #
#                                     NODES                                    #
# ---------------------------------------------------------------------------- #
def rewrite(state: State):
    writer = get_stream_writer()
    writer({"status": "Rewriting your question..."})

    lines = []
    for m in state["history"][-2 * MAX_HISTORY_TURNS :]:
        who = "Student" if m["role"] == "user" else "Tutor"
        lines.append(f"{who} : {m['content'][:300]}")

    prompt = REWRITE_PROMPT.format(
        history = "\n".join(lines), question=state["question"]
    )
    res = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role":"user", "content":prompt}],
        options={"num_ctx": NUM_CTX, "temperature":0},
    )
    text = res["message"]["content"].strip().strip('"')
    if not text or len(text) > 300: # the model talked too much: use the original
        text = state["question"]

    writer({"status": f"Searching for: {text}"})
    return {"standalone": text}

def use_original(state: State):
    return {"standalone": state["question"]}

def retreive(state: State):
    get_stream_writer()({"status": "Searching your documents..."})
    return {"chunks": rag.retreive(state["standalone"])}

def generate(state:State):
    """Answer from your documents. If the model says 'not found', show nothing"""
    writer = get_stream_writer()
    writer({"status": "Writing the answer..."})

    answer, shown = "", 0
    for piece in rag.stream_answer(state["standalone"], state["chunks"]):
        answer += piece
        head = answer.lstrip(' "\'\n').lower()
        if len(head) < len(REFUSAL) and REFUSAL.startswith(head):
            continue # it may become "I can not find...": wait a little
        if head.startswith(REFUSAL):
            return {"answer": answer, "found": False}
        writer({"token": answer[shown:]})
        shown = len(answer)

    if not answer.strip():
        return {"answer": "", "found": False}
    if shown < len(answer): # a very short answer ended while we were waiting
        writer({"token": answer[shown:]})

    writer({"sources": rag.format_sources(answer, state["chunks"])})
    return {"answer": answer, "found": True}

def web_search(state:State):
    """Search the web. Only the question text leaves the laptop."""
    if not WEB_SEARCH:
        return {"web_results": []}

    writer = get_stream_writer()
    query = state["standalone"]
    writer({"status": f"Not in your documents. Searching the web for: {query}"})
    try:
        results = web.search(query)
    except Exception as e: # no internet, rate limit, library change... # noqa: BLE001
        writer({"status": f"Web search failed: {e}"})
        results = []
    return {"web_results": results}

def web_answer(state:State):
    writer = get_stream_writer()
    writer({"status":"Writing the answer from web results..."})

    full = WEB_LABEL
    writer({"token": WEB_LABEL})
    for piece in web.stream_web_answer(state["standalone"], state["web_results"]):
        full += piece
        writer({"token": piece})
    writer({"sources": web.format_web_sources(full, state["web_results"])})
    return {"answer": full}

def no_answer(state:State):
    writer = get_stream_writer()
    if WEB_SEARCH:
        message = "I can not find this in you documents or on the web."
    else:
        message = "I can not find this in you documents"
    writer({"token":message})
    writer({"sources": rag.format_sources(message, state["chunks"])})
    return {"answer": message}

def save_turn(state:State):
    return {
        "history": [
            {"role":"user", "content": state["question"]},
            {"role":"assistant", "content": state["answer"]},
        ]
    }

# ---------------------------------------------------------------------------- #
#                           CONDITIONAL EDGE ROUTERS                           #
# ---------------------------------------------------------------------------- #
def route_start(state:State):
    """Old messages? Then rewrite. First question? use it as it is."""
    return "rewrite" if state.get("history") else "use_original"

def route_after_retreive(state:State):
    """Good match? Answer. Weak match? Say we can not find it."""
    best = max((c["score"] for c in state["chunks"]), default=0)
    return "generate" if best >= MIN_SCORE else "web_search"

def route_after_generate(state:State):
    """Documents answered? Save. The model said 'not found'? Go to the web."""
    return "save_turn" if state["found"] else "web_search"

def route_after_web(state:State):
    """Web results? Answer from them. Nothing? Say we can not find it."""
    return "web_answer" if state["web_results"] else "no_answer"

# ---------------------------------------------------------------------------- #
#                                  BUILD GRAPH                                 #
# ---------------------------------------------------------------------------- #
def build_graph():
    g = StateGraph(State)

    g.add_node("rewrite", rewrite)
    g.add_node("use_original", use_original)
    g.add_node("retreive", retreive)
    g.add_node("generate", generate)
    g.add_node("web_search", web_search)
    g.add_node("web_answer", web_answer)
    g.add_node("no_answer", no_answer)
    g.add_node("save_turn", save_turn)

    g.add_conditional_edges(
        START, route_start, {"rewrite": "rewrite", "use_original": "use_original"}
    )
    g.add_edge("rewrite", "retreive")
    g.add_edge("use_original", "retreive")
    g.add_conditional_edges(
        "retreive",
        route_after_retreive,
        {"generate": "generate", "web_search": "web_search"},
    )
    g.add_conditional_edges(
        "generate",
        route_after_generate,
        {"save_turn": "save_turn", "web_search": "web_search"},
    )
    g.add_conditional_edges(
        "web_search",
        route_after_web,
        {"web_answer": "web_answer", "no_answer":"no_answer"},
    )
    g.add_edge("web_answer", "save_turn")
    g.add_edge("no_answer", "save_turn")
    g.add_edge("save_turn", END)

    # The checkpointer saves the state after every step, for each thread_id.
    return g.compile(checkpointer=InMemorySaver())


def ask_stream(graph, question, thread_id):
    """Run the graph. Yields small dicts: status, token, or sources."""
    config = {"configurable": {"thread_id": thread_id}}
    yield from graph.stream({"question": question}, config, stream_mode="custom")