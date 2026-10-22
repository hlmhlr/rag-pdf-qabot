"""Shared test helpers: sample PDF paths and a fake embedding model."""

from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"
HANDBOOK = SAMPLES / "northwind_employee_handbook.pdf"
MANUAL = SAMPLES / "volta_x2_ebike_manual.pdf"


class KeywordEmbeddings(Embeddings):
    """A tiny, predictable stand-in for a real embedding model.

    Each vector counts how often each keyword appears in the text, so texts
    that share keywords are "similar". No model, no network, same result
    every run.
    """

    KEYWORDS = ["leave", "charge", "battery", "warranty", "frame", "remote", "expenses", "motor"]

    def _embed(self, text: str) -> list[float]:
        words = text.lower()
        # The final 1.0 keeps every vector non-zero, which similarity search needs.
        return [float(words.count(k)) for k in self.KEYWORDS] + [1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture
def embeddings() -> KeywordEmbeddings:
    return KeywordEmbeddings()
