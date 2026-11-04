# ---------------------------------------------------------------------------- #
#                                    IMPORTS                                   #
# ---------------------------------------------------------------------------- #
import sys
import uuid

import rag
from graph import ask_stream, build_graph


# ---------------------------------------------------------------------------- #
#                                 MAIN FUNCTION                                #
# ---------------------------------------------------------------------------- #
def main():
    try:
        rag.get_collection()
    except Exception:  # noqa: BLE001
        print("No database found. Run python ingest.py")
        sys.exit(1)

    graph = build_graph()
    thread_id = uuid.uuid4().hex

    print("Study agent ready. Type 'exit' to stop. '/new' for a new chat.")

    # Infinite loop to start chat
    while True:
        question = input("\nYou:").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        if question.lower() == "/new":
            thread_id = uuid.uuid4().hex
            print("New chat started. Old messages are forgotten.")
            continue

        started = False
        for event in ask_stream(graph, question, thread_id):
            if "status" in event:
                print(f" ({event['status']})")
            elif "token" in event:
                if not started:
                    print("\nTutor: ", end="", flush=True)
                    started = True
                print(event["token"], end="", flush=True)
            elif "sources" in event:
                print("\n\n" + event["sources"])
                

if __name__ == "__main__":
    main()