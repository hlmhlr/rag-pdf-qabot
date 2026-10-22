"""Tests for the index cache (extension 1) and the bot, with fake models."""

import shutil

import pytest
from conftest import HANDBOOK, MANUAL

from qabot.index import IndexCache, QABot, fingerprint


def test_fingerprint_ignores_order_and_file_name(tmp_path):
    renamed = tmp_path / "renamed.pdf"
    shutil.copy(MANUAL, renamed)
    assert fingerprint([HANDBOOK, MANUAL]) == fingerprint([renamed, HANDBOOK])


def test_fingerprint_changes_with_the_file_set():
    assert fingerprint([HANDBOOK]) != fingerprint([HANDBOOK, MANUAL])


def test_same_files_are_embedded_only_once(embeddings):
    cache = IndexCache(embeddings)
    first, reused_first = cache.get_or_build([HANDBOOK, MANUAL])
    embedded = embeddings.documents_embedded

    second, reused_second = cache.get_or_build([MANUAL, HANDBOOK])  # same files, other order

    assert (reused_first, reused_second) == (False, True)
    assert second is first
    assert embeddings.documents_embedded == embedded  # nothing embedded the second time
    assert first.files == [HANDBOOK.name, MANUAL.name] and first.chunks == 6


def test_least_recently_used_index_is_dropped(embeddings):
    cache = IndexCache(embeddings, max_indexes=2)
    cache.get_or_build([HANDBOOK])
    cache.get_or_build([MANUAL])
    cache.get_or_build([HANDBOOK])  # use the handbook again: the manual is now the oldest
    cache.get_or_build([HANDBOOK, MANUAL])  # a third index: the manual's is dropped

    assert len(cache) == 2
    assert cache.get_or_build([HANDBOOK])[1] is True  # still cached
    assert cache.get_or_build([MANUAL])[1] is False  # had to be rebuilt


def test_bot_answers_with_sources_and_reuses_the_index(embeddings, fake_llm):
    bot = QABot(fake_llm, embeddings)

    first = bot.ask([HANDBOOK, MANUAL], "  How long does the battery take to charge?  ")
    second = bot.ask([HANDBOOK, MANUAL], "Is the frame under warranty?")

    assert first.text == "A fake answer."
    assert first.sources[0].startswith(MANUAL.name)
    assert (first.reused_index, second.reused_index) == (False, True)
    assert fake_llm.prompts[0][1].content == "How long does the battery take to charge?"  # trimmed


def test_no_sources_when_the_answer_is_not_found(embeddings, fake_llm):
    fake_llm.reply = "I couldn't find the answer in the uploaded documents"  # no final period
    answer = QABot(fake_llm, embeddings).ask([HANDBOOK], "Who is the CEO?")
    assert answer.sources == []


def test_bot_rejects_an_empty_question(embeddings, fake_llm):
    with pytest.raises(ValueError, match="Type a question"):
        QABot(fake_llm, embeddings).ask([HANDBOOK], "   ")
