# ---------------------------------------------------------------------------- #
#                                    IMPORTS                                   #
# ---------------------------------------------------------------------------- #
import sys

import rag


# ---------------------------------------------------------------------------- #
#                                 MAIN FUNCTION                                #
# ---------------------------------------------------------------------------- #
def main():
    try:
        rag.get_collection()
    except Exception:  # noqa: BLE001
        print("No database found. Run python ingest.py")
        sys.exit(1)

    print("Study agent ready. Type 'exit' to stop.")

    # Infinite loop to start chat
    while True:
        question = input("\nYou:").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        chunks = rag.retreive(question)

        print("\nTutor: ", end="", flush=True)
        answer = ""
        for piece in rag.stream_answer(question, chunks):
            answer += piece
            print(piece, end="", flush=True)

        print("\n\n" + rag.format_sources(answer, chunks))
        

if __name__ == "__main__":
    main()