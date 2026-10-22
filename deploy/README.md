---
title: RAG PDF QA Bot
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "{gradio_version}"
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Ask questions about your PDFs, answered with sources
---

# RAG PDF QA Bot

Upload up to 5 PDF files and ask questions about them. Answers come only from
your documents and list the pages they were found on.

- **Answers:** Google Gemini (free API tier)
- **Embeddings:** IBM Granite Embedding 278M, running on this Space's CPU
- **Search:** Chroma vector store, one per upload, kept in memory

Uploaded files are deleted after an hour. The Space sleeps when unused; waking
it takes a minute or two while the embedding model loads.

Source code, tests and the local (Ollama) version:
https://github.com/hlmhlr/rag-pdf-qabot

This Space is deployed automatically from GitHub; changes made here are overwritten.
