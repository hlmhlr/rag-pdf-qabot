"""Load PDFs and split them into chunks (lab sections: document loader, text splitter).

Aim
    See what the loader and the splitter produce before anything is embedded.

What you'll see
    Per file: pages loaded and characters of text. Then the chunk count,
    the shortest, average and longest chunk, and the first chunk with its
    metadata.

What to notice
    - No chunk is longer than 1000 characters (the lab's chunk_size).
    - Each chunk keeps "file" and "page", which later become the sources
      shown under an answer.
    - Try --chunk-size 300 and see the chunk count go up.

Run:  uv run python scripts/01_load_and_split.py                        (sample PDFs)
      uv run python scripts/01_load_and_split.py --pdf a.pdf b.pdf      (your own)
"""

import argparse
from collections import Counter
from pathlib import Path

from qabot.ingest import CHUNK_OVERLAP, CHUNK_SIZE, IngestError, load_pdfs, split_documents

SAMPLES = sorted((Path(__file__).resolve().parent.parent / "data" / "samples").glob("*.pdf"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", nargs="+", type=Path, default=SAMPLES, help="PDF files to load")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args()

    try:
        pages = load_pdfs(args.pdf)
    except IngestError as error:
        raise SystemExit(f"Cannot load the files: {error}")

    print("Loaded:")
    page_counts = Counter(page.metadata["file"] for page in pages)
    for file, count in page_counts.items():
        chars = sum(len(p.page_content) for p in pages if p.metadata["file"] == file)
        print(f"  {file}: {count} pages, {chars:,} characters")

    chunks = split_documents(pages, args.chunk_size, args.chunk_overlap)
    lengths = [len(chunk.page_content) for chunk in chunks]
    print(f"\nSplit into {len(chunks)} chunks (size {args.chunk_size}, overlap {args.chunk_overlap}): "
          f"shortest {min(lengths)}, average {sum(lengths) // len(lengths)}, longest {max(lengths)} characters")

    first = chunks[0]
    print(f"\nFirst chunk, from {first.metadata['file']} page {first.metadata['page'] + 1}:")
    print(first.page_content[:300] + ("..." if len(first.page_content) > 300 else ""))


if __name__ == "__main__":
    main()
