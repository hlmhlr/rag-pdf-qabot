"""Build each document set's index once and reuse it (extension 1), and the bot itself.

The lab's retriever_qa(file, query) loads, splits and embeds the PDF again
for every question. Embedding is the slow step (seconds on a GPU, longer on
the free CPU deployment), so asking three questions about one file paid
that cost three times.

Here the index is kept in memory, keyed by a fingerprint of the files'
contents:
    - the same files again    -> the stored index is reused, no embedding
    - a file added or changed -> a different fingerprint, so a new index
    - same file, new name     -> same content, so the index is reused
Only the most recent MAX_INDEXES indexes are kept, so memory stays bounded
when many people use the deployed app.

Contents
    fingerprint()  SHA-256 over the files' bytes, independent of order
    IndexCache     get_or_build(): reuse or build a retriever
    QABot          ask(files, question): the function behind the app
"""

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStoreRetriever

from qabot.ingest import check_files, load_pdfs, split_documents
from qabot.qa import NO_ANSWER, build_qa_chain, sources_of
from qabot.vectorstore import TOP_K, build_vector_store, get_retriever

MAX_INDEXES = 3


def fingerprint(paths: list[str | Path]) -> str:
    """One hash for a set of files, based only on their contents.

    Each file is hashed on its own, then the sorted file hashes are hashed
    together, so uploading the same files in a different order (or under a
    different name) gives the same fingerprint.
    """
    file_hashes = sorted(hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths)
    return hashlib.sha256("".join(file_hashes).encode()).hexdigest()


@dataclass
class IndexInfo:
    """What was built for one set of files, shown to the user and in tests."""

    retriever: VectorStoreRetriever
    files: list[str]
    chunks: int
    build_seconds: float


class IndexCache:
    """Keep the last few indexes in memory, least recently used dropped first."""

    def __init__(self, embeddings: Embeddings, max_indexes: int = MAX_INDEXES, k: int = TOP_K):
        self.embeddings = embeddings
        self.max_indexes = max_indexes
        self.k = k
        self._indexes: OrderedDict[str, IndexInfo] = OrderedDict()

    def __len__(self) -> int:
        return len(self._indexes)

    def get_or_build(self, paths: list[str | Path]) -> tuple[IndexInfo, bool]:
        """Return (index, reused). reused is True when nothing had to be embedded."""
        files = check_files(paths)  # validate before reading the bytes for the hash
        key = fingerprint(files)

        if key in self._indexes:
            self._indexes.move_to_end(key)  # mark as most recently used
            return self._indexes[key], True

        start = time.perf_counter()
        chunks = split_documents(load_pdfs(files))
        vectordb = build_vector_store(chunks, self.embeddings)
        info = IndexInfo(
            retriever=get_retriever(vectordb, k=self.k),
            files=[f.name for f in files],
            chunks=len(chunks),
            build_seconds=time.perf_counter() - start,
        )

        self._indexes[key] = info
        if len(self._indexes) > self.max_indexes:
            _, oldest = self._indexes.popitem(last=False)  # drop the least recently used
            oldest.retriever.vectorstore.delete_collection()  # and free its vectors
        return info, False


@dataclass
class Answer:
    """One answer, with what the app shows under it."""

    text: str
    sources: list[str] = field(default_factory=list)
    reused_index: bool = False
    seconds: float = 0.0


class QABot:
    """Answer questions about uploaded PDFs (the lab's retriever_qa, as an object).

    The models and the cache live as long as the bot, so one bot serves
    every question in the app.
    """

    def __init__(self, llm: BaseChatModel, embeddings: Embeddings, cache: IndexCache | None = None):
        self.llm = llm
        self.cache = cache or IndexCache(embeddings)

    def ask(self, paths: list[str | Path], question: str) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("Type a question first.")

        start = time.perf_counter()
        index, reused = self.cache.get_or_build(paths)
        result = build_qa_chain(index.retriever, self.llm).invoke({"question": question})
        text = result["answer"].strip()
        # Sources are the chunks that were retrieved. When the model says the
        # answer isn't there, listing them would suggest they support it.
        found = NO_ANSWER.lower().rstrip(".") not in text.lower()
        return Answer(
            text=text,
            sources=sources_of(result["docs"]) if found else [],
            reused_index=reused,
            seconds=time.perf_counter() - start,
        )
