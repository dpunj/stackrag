# Stackrag

Simple FastAPI backend for a RAG pipeline that processes PDFs and answers questions with Mistral AI.

## Features

- **PDF processing**: extracts and chunks text from PDFs
  - Uses 500/50 chunking strategy as a safe middle ground with PDFs (vs. 1000/10 or 300/30)
- **Query processing**: Classifies query intent to skip RAG when unnecessary (e.g., greetings, chitchat)
  - If classified as RAG, the user query is transformed to enhance retrieval
- **Semantic search**: Uses vector embeddings to find relevant document sections
  - If we wanted to incpororate keyword search we could use BM25 and use a hybrid search method that combines both
- **RAG**: Generates answers based on the retrieved chunks

### Stack

Uses a lightweight stack with a few deps:

- **Framework**: FastAPI
- **LLM**: Mistral AI
- **Embeddings**: Mistral Embed
- **PDF Processing**: [PyMuPDF](https://github.com/pymupdf/PyMuPDF) (fitz)
- **Vector Operations**: Numpy

## Getting Started

Clone this repo and setup & activate venv using [uv](https://github.com/astral-sh/uv), and install deps

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Try it!

```bash
fastapi dev main.py
```
