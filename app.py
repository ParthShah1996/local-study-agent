# ---------------------------------------------------------------------------- #
#                                    IMPORTS                                   #
# ---------------------------------------------------------------------------- #
import gradio as gr

import rag

# ------------------------------- CHAT FUNCTION ------------------------------ #
EXAMPLES = [
    "What is the basic idea for reversing a linked list?",
    "When do I use a monotonic stack?",
    "Explain binary search on a rotated sorted array.",
]

def chat(message, history):
    # Each question is answered alone for now. Memory functionality TBD
    yield "Searching your documents..."

    try:
        chunks = rag.retreive(message)
    except Exception as e:  # noqa: BLE001
        yield(
            f"Error: {e}\n\n"
            "Check 1: did you run `python ingest.py\n"
            "Check 2: is Ollama running? (`ollama ps`)"
        )
        return

    answer = ""
    for piece in rag.stream_answer(message, chunks):
        answer += piece
        yield answer

    yield answer + "\n\n" + rag.format_sources(answer, chunks)

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
