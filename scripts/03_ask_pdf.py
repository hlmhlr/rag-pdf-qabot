"""Ask questions about PDFs (lab section: question-answering chain).

Aim
    Run the complete pipeline: load, split, embed, retrieve, answer, with
    the index built once and reused for every later question.

What you'll see
    For each question: the answer, its sources (file and page), the time it
    took, and whether the index was built or reused.

What to notice
    - Only the first question builds the index; later ones skip embedding
      and are much faster.
    - The fourth question combines both files: the discount is in the
      handbook, the warranty details in the manual.
    - The last question isn't covered by the documents, so the bot should
      say it couldn't find the answer instead of inventing one.
    - With --compare-lab, the lab's RetrievalQA answers the first question
      from the same index, without sources.

Run:  uv run python scripts/03_ask_pdf.py
      uv run python scripts/03_ask_pdf.py --pdf a.pdf b.pdf --question "..."
      uv run python scripts/03_ask_pdf.py --compare-lab
"""

import argparse
from pathlib import Path

from qabot.index import QABot
from qabot.ingest import IngestError
from qabot.llm import MissingCloudPackages, describe, get_embeddings, get_llm

SAMPLES = sorted((Path(__file__).resolve().parent.parent / "data" / "samples").glob("*.pdf"))
QUESTIONS = [
    "How many days of annual leave do full-time employees get?",
    "How long does a full battery charge take?",
    "What does error E07 on the display mean?",
    "Can employees get a discount on an e-bike, and how long is the battery covered by warranty?",
    "Who is the CEO of Northwind Cycles?",
]


def compare_with_lab(bot: QABot, files: list[Path], question: str) -> None:
    """Answer one question with the lab's RetrievalQA, reusing the bot's index."""
    from langchain_classic.chains import RetrievalQA  # legacy chain, dev dependency only

    index, _ = bot.cache.get_or_build(files)
    qa = RetrievalQA.from_chain_type(llm=bot.llm, chain_type="stuff", retriever=index.retriever)
    print("Lab's RetrievalQA (same index, its default prompt, no sources):")
    print(f"Q: {question}\nA: {qa.invoke(question)['result'].strip()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", nargs="+", type=Path, default=SAMPLES)
    parser.add_argument("--question", action="append", help="repeat for several questions")
    parser.add_argument("--compare-lab", action="store_true", help="also run the lab's RetrievalQA")
    args = parser.parse_args()

    print(f"Using {describe()}\n")
    try:
        bot = QABot(get_llm(), get_embeddings())
    except (MissingCloudPackages, ValueError) as error:
        raise SystemExit(str(error))

    questions = args.question or QUESTIONS
    for question in questions:
        try:
            answer = bot.ask(args.pdf, question)
        except IngestError as error:
            raise SystemExit(f"Cannot load the files: {error}")

        index_note = "index reused" if answer.reused_index else "index built"
        print(f"Q: {question}\nA: {answer.text}")
        print(f"   Sources: {', '.join(answer.sources) or 'none'}")
        print(f"   {answer.seconds:.1f} s, {index_note}\n")

    if args.compare_lab:
        compare_with_lab(bot, args.pdf, questions[0])


if __name__ == "__main__":
    main()
