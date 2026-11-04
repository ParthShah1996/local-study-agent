# ---------------------------------------------------------------------------- #
#                                    IMPORTS                                   #
# ---------------------------------------------------------------------------- #
import uuid

import gradio as gr

import rag
from graph import ask_stream, build_graph

# ------------------------------- CHAT FUNCTION ------------------------------ #
graph = build_graph()
_threads = {} # browser session -> chat thread id

EXAMPLES = [
    "What is the basic idea for reversing a linked list?",
    "When do I use a monotonic stack?",
    "Explain binary search on a rotated sorted array.",
]

def get_thread_id(request, history):
    """One thread per chat. An empty screen (first message or Clear) = new thread."""
    session = request.session_hash if request else "local"
    if not history or session not in _threads:
        _threads[session] = f"{session}-{uuid.uuid4().hex[:8]}"
    return _threads[session]

def chat(message, history, request: gr.Request):
    try:
        rag.get_collection()
    except Exception:
        yield "No database found. Run `python ingest.py` first."
        return

    thread_id = get_thread_id(request, history)
    answer, sources = "", ""
    yield "Thinking..."

    try:
        for event in ask_stream(graph, message, thread_id):
            if "status" in event and not answer:
                yield event["status"]
            elif "token" in event:
                answer += event["token"]
                yield answer
            elif "sources" in event:
                sources = event["sources"]
    except Exception as e:  # noqa: BLE001
        yield f"Error: {e}\n\nIs Ollama running? `ollama ps`."
        return

    yield answer + "\n\n" + sources

# ----------------------------------- DEMO ----------------------------------- #
demo = gr.ChatInterface(
    fn=chat,
    title="Local Study Tutor",
    description="Answers come from my own study documents. Everything runs on this laptop.",
    examples=EXAMPLES
)

if __name__ == "__main__":
    # Do NOT use share=True. It would make your laptop reachable from the internet.
    demo.launch(inbrowser=True)
