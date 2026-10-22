"""Shared test helpers: sample PDF paths, a fake embedding model and a fake chat model."""

from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"
HANDBOOK = SAMPLES / "northwind_employee_handbook.pdf"
MANUAL = SAMPLES / "volta_x2_ebike_manual.pdf"


class KeywordEmbeddings(Embeddings):
    """A tiny, predictable stand-in for a real embedding model.

    Each vector counts how often each keyword appears in the text, so texts
    that share keywords are "similar". No model, no network, same result
    every run. documents_embedded counts chunk embeddings, so tests can
    check when the cache skips embedding.
    """

    KEYWORDS = ["leave", "charge", "battery", "warranty", "frame", "remote", "expenses", "motor"]

    def __init__(self):
        self.documents_embedded = 0

    def _embed(self, text: str) -> list[float]:
        words = text.lower()
        # The final 1.0 keeps every vector non-zero, which similarity search needs.
        return [float(words.count(k)) for k in self.KEYWORDS] + [1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.documents_embedded += len(texts)
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class RecordingChatModel(BaseChatModel):
    """A fake chat model: always gives the same reply and records every prompt."""

    reply: str | list = "A fake answer."
    prompts: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "recording-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.prompts.append(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])


@pytest.fixture
def embeddings() -> KeywordEmbeddings:
    return KeywordEmbeddings()


@pytest.fixture
def fake_llm() -> RecordingChatModel:
    return RecordingChatModel(prompts=[])  # a fresh list per test
