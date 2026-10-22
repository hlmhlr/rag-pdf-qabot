"""Tests for the QA prompt, sources and chain, with fake models."""

from conftest import HANDBOOK, MANUAL
from langchain_core.documents import Document

from qabot.ingest import load_pdfs, split_documents
from qabot.qa import NO_ANSWER, build_qa_chain, format_docs, sources_of
from qabot.vectorstore import build_vector_store, get_retriever


def doc(file, page, text="text"):
    return Document(page_content=text, metadata={"file": file, "page": page})


def test_format_docs_numbers_and_labels_chunks():
    context = format_docs([doc("a.pdf", 0, "first"), doc("b.pdf", 4, "second")])
    assert context == "[1] (a.pdf p.1)\nfirst\n\n[2] (b.pdf p.5)\nsecond"


def test_sources_are_distinct_and_in_ranking_order():
    docs = [doc("b.pdf", 2), doc("a.pdf", 0), doc("b.pdf", 2)]
    assert sources_of(docs) == ["b.pdf p.3", "a.pdf p.1"]


def test_chain_puts_retrieved_context_and_question_in_the_prompt(embeddings, fake_llm):
    chunks = split_documents(load_pdfs([HANDBOOK, MANUAL]))
    retriever = get_retriever(build_vector_store(chunks, embeddings), k=1)

    result = build_qa_chain(retriever, fake_llm).invoke({"question": "battery charge time?"})

    assert result["answer"] == "A fake answer."
    assert result["docs"][0].metadata["file"] == MANUAL.name
    system, human = fake_llm.prompts[0]
    assert "4.5 hours" in system.content  # the retrieved chunk is in the context
    assert NO_ANSWER in system.content  # the "don't guess" instruction
    assert human.content == "battery charge time?"


def test_chain_handles_list_content_like_gemini(embeddings, fake_llm):
    fake_llm.reply = [{"type": "text", "text": "Four and a half hours."}]
    retriever = get_retriever(build_vector_store(split_documents(load_pdfs([MANUAL])), embeddings))
    result = build_qa_chain(retriever, fake_llm).invoke({"question": "charge time?"})
    assert result["answer"] == "Four and a half hours."
