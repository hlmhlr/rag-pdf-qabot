"""Tests for the vector store and retriever, with the fake keyword embeddings."""

import pytest
from conftest import HANDBOOK, MANUAL
from langchain_chroma import Chroma

from qabot.ingest import load_pdfs, split_documents
from qabot.vectorstore import build_vector_store, get_retriever


def chunks_of(*pdfs):
    return split_documents(load_pdfs(list(pdfs)))


def test_retrieves_from_the_right_file(embeddings):
    retriever = get_retriever(build_vector_store(chunks_of(HANDBOOK, MANUAL), embeddings), k=1)

    [top] = retriever.invoke("how long does the battery take to charge")
    assert top.metadata["file"] == MANUAL.name

    [top] = retriever.invoke("how many days of annual leave")
    assert top.metadata["file"] == HANDBOOK.name


def test_ranks_by_direction_not_length(embeddings):
    """Page 2 of the handbook says "leave" most often, so cosine ranks it first.

    With Chroma's default Euclidean distance, a short page that never says
    "leave" would rank first, because its vector is closer to the short query.
    """
    vectordb = build_vector_store(chunks_of(HANDBOOK, MANUAL), embeddings)
    [top] = get_retriever(vectordb, k=1).invoke("annual leave days")
    assert (top.metadata["file"], top.metadata["page"]) == (HANDBOOK.name, 1)


def test_k_sets_number_of_results(embeddings):
    vectordb = build_vector_store(chunks_of(HANDBOOK, MANUAL), embeddings)
    assert len(get_retriever(vectordb, k=2).invoke("warranty")) == 2


def test_uploads_do_not_mix(embeddings):
    """Two builds get separate collections, so the second sees only its own file."""
    build_vector_store(chunks_of(HANDBOOK), embeddings)
    second = build_vector_store(chunks_of(MANUAL), embeddings)
    files = {d.metadata["file"] for d in second.similarity_search("leave", k=10)}
    assert files == {MANUAL.name}


def test_lab_code_mixes_uploads(embeddings):
    """The lab's call (no collection name) shares one collection between uploads."""
    Chroma.from_documents(chunks_of(HANDBOOK), embeddings)
    second = Chroma.from_documents(chunks_of(MANUAL), embeddings)
    files = {d.metadata["file"] for d in second.similarity_search("leave", k=10)}
    assert HANDBOOK.name in files  # chunks from the earlier upload leak in
    second.delete_collection()  # clean up the shared default collection


def test_empty_chunks_raise(embeddings):
    with pytest.raises(ValueError, match="no chunks"):
        build_vector_store([], embeddings)
