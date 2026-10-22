"""Tests for the web app's handlers, with fake models and no browser."""

import gradio as gr
import pytest
from conftest import HANDBOOK, MANUAL, KeywordEmbeddings, RecordingChatModel

from qabot.index import QABot
from qabot.ui import START_MESSAGE, answer_question, build_app, index_files

FILES = [str(HANDBOOK), str(MANUAL)]


class BrokenEmbeddings(KeywordEmbeddings):
    """Embeddings that fail, like Ollama not running."""

    def embed_documents(self, texts):
        raise ConnectionError("connection refused")


class BrokenChatModel(RecordingChatModel):
    """A chat model that fails, like a Gemini rate limit."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")


@pytest.fixture
def bot(embeddings, fake_llm):
    return QABot(fake_llm, embeddings)


def test_no_files_shows_the_start_message(bot):
    assert index_files(bot, None) == START_MESSAGE
    assert answer_question(bot, [], "anything?") == ("", "", START_MESSAGE)


def test_upload_builds_then_reuses_the_index(bot):
    first = index_files(bot, FILES)
    assert first.startswith("✅") and "6 chunks" in first
    assert "already indexed" in index_files(bot, FILES)


def test_bad_upload_is_reported_not_raised(bot, tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("not a pdf")
    assert index_files(bot, [str(notes)]) == "⚠️ notes.txt is not a PDF file."


def test_answer_shows_text_sources_and_timing(bot):
    index_files(bot, FILES)  # as in the app: the upload indexes first
    text, sources, status = answer_question(bot, FILES, "How long does the battery take to charge?")
    assert text == "A fake answer."
    assert sources.startswith("**Retrieved passages**") and MANUAL.name in sources
    assert "index reused" in status


def test_empty_question_is_reported(bot):
    assert answer_question(bot, FILES, "  ") == ("", "", "⚠️ Type a question first.")


def test_model_failures_become_messages():
    broken_embeddings = QABot(RecordingChatModel(prompts=[]), BrokenEmbeddings())
    assert "ConnectionError" in index_files(broken_embeddings, FILES)

    broken_llm = QABot(BrokenChatModel(prompts=[]), KeywordEmbeddings())
    _, _, status = answer_question(broken_llm, FILES, "Is the frame covered?")
    assert status.startswith("⚠️ The model could not answer (RuntimeError)")


def test_build_app_creates_the_page(bot):
    app = build_app(bot, models="fake models", examples=[[FILES, "A question?"]])
    assert isinstance(app, gr.Blocks)
