# Local Study Agent

A free study chatbot that runs fully on my laptop.
It answers from my own PDF and Python study files first, and shows the source (file name + page) for every answer.

> Status: **Milestone 1 (simple RAG, command line)**. More milestones are coming.

## Why I built this

- Learn how RAG and AI agents work by building one step by step.
- Study from my own documents, not only from general model knowledge.
- Keep my private data on my own computer.
- Use it later for tutoring, quizzes, and interview practice.

## Features

- [x] Read PDF files (PyMuPDF) and Python files (code-aware splitter)
- [x] Local embeddings and a local vector database (Chroma)
- [x] Local chat model with Ollama
- [x] Answers with citations: file name + page (PDF) or line (Python)
- [ ] Chat screen (Gradio)
- [ ] LangGraph agent flow with memory
- [ ] Web search only when my documents are not enough
- [ ] Tutor mode: curriculum, teaching, quizzes, "what to study next"
- [ ] Interview preparation mode
- [ ] Daily task automation

## How it works

```
Your files (PDF, .py)
        |
   ingest.py: read -> cut into chunks -> embeddings -> save in Chroma
        |
   ask.py:    question -> find top 5 chunks -> give chunks to the model
        |
   Answer + sources
```

## Privacy

- The model, embeddings, and database all run on my laptop.
- My documents (`docs/`) and my database (`chroma_db/`) are **not** uploaded to GitHub (see `.gitignore`).
- In the web search milestone, only a short search query will leave my laptop. Never my files.

## Requirements

- Python 3.11
- [Ollama](https://ollama.com)
- A computer with about 16 GB RAM. A GPU with 6 GB VRAM helps but is not required. Slow replies are okay.

Tested on: Windows, RTX 3060 (6 GB VRAM), 16 GB RAM.

## Setup

1. Install Ollama, then download the models:

```
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

2. Clone the project and install packages:

```
git clone https://github.com/<your-username>/study-agent.git
cd study-agent
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, use `source .venv/bin/activate` instead of `.venv\Scripts\activate`.

## Run

1. Put your `.pdf` and `.py` files in the `docs/` folder.
2. Build the database (run again when you add or change files):

```
python ingest.py
```

3. Start the chatbot:

```
python ask.py
```

Type `exit` to stop.

## Settings

All settings are in `config.py`: model names, chunk size, number of chunks (top-K), and context size.
Slow or out of memory? Try a smaller model, for example `llama3.2:3b`, by changing `CHAT_MODEL`.

## Project files

| File | Job |
|---|---|
| `config.py` | All settings in one place |
| `ingest.py` | Read files, cut chunks, save embeddings in Chroma |
| `ask.py` | Ask questions and get answers with sources |
| `requirements.txt` | Python packages |
| `docs/` | My study files (not on GitHub) |
| `chroma_db/` | The database (not on GitHub) |

## Tech

Ollama, Chroma, PyMuPDF, LangChain text splitters. Coming next: LangGraph, Gradio, web search.

## Known limits

- Scanned PDFs (images with no text) are not read. They need OCR.
- A 7B model can make mistakes. Always check the sources.

## License

MIT. Add a `LICENSE` file if you want others to reuse the code.