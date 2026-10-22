"""Embed the PDFs and retrieve the chunks that match a question
(lab sections: embedding model, vector store, retriever).

Aim
    Check that retrieval finds the right part of the right file before an
    LLM is added in the next step.

What you'll see
    How long embedding took, then for each question the top chunks with
    their file, page, and the start of their text.

What to notice
    - The questions share almost no words with the matching text
      ("time off" vs "annual leave"); the match is by meaning.
    - Each question finds its answer in one of the two files.
    - Uses the embedding provider from .env (Ollama by default).

Run:  uv run python scripts/02_retrieve.py
      uv run python scripts/02_retrieve.py --pdf a.pdf --question "..." --k 2
"""

import argparse
import time
from pathlib import Path

from qabot.ingest import IngestError, load_pdfs, split_documents
from qabot.llm import MissingCloudPackages, describe, get_embeddings
from qabot.vectorstore import TOP_K, build_vector_store, get_retriever

SAMPLES = sorted((Path(__file__).resolve().parent.parent / "data" / "samples").glob("*.pdf"))
QUESTIONS = [
    "How much paid time off do I get each year?",
    "How long does it take to fully charge the battery?",
    "Is the frame covered if it breaks?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", nargs="+", type=Path, default=SAMPLES)
    parser.add_argument("--question", action="append", help="repeat for several questions")
    parser.add_argument("--k", type=int, default=2, help=f"chunks per question (app default: {TOP_K})")
    args = parser.parse_args()

    try:
        chunks = split_documents(load_pdfs(args.pdf))
    except IngestError as error:
        raise SystemExit(f"Cannot load the files: {error}")

    print(f"Using {describe()}")
    try:
        embeddings = get_embeddings()
    except MissingCloudPackages as error:
        raise SystemExit(str(error))

    start = time.perf_counter()
    vectordb = build_vector_store(chunks, embeddings)
    print(f"Embedded {len(chunks)} chunks in {time.perf_counter() - start:.1f} s\n")

    retriever = get_retriever(vectordb, k=args.k)
    for question in args.question or QUESTIONS:
        print(f"Q: {question}")
        for rank, doc in enumerate(retriever.invoke(question), start=1):
            snippet = " ".join(doc.page_content.split())[:110]
            print(f"  {rank}. {doc.metadata['file']} p.{doc.metadata['page'] + 1}: {snippet}...")
        print()


if __name__ == "__main__":
    main()