"""Embed chunks into a Chroma vector store and search it (lab: vector_database, retriever).

Fix for a lab bug
    The lab calls Chroma.from_documents(chunks, embedding_model) with no
    collection name. Every call then adds to the same default in-memory
    collection, so chunks from a PDF uploaded earlier stay in the store and
    can be retrieved for questions about the next PDF. Here every build gets
    its own collection, so each upload is searched on its own.

Similarity measure
    Collections use cosine similarity instead of Chroma's default Euclidean
    distance (see build_vector_store).

Contents
    build_vector_store()  embed chunks into a new, separate collection
    get_retriever()       the store as a LangChain retriever (top-k similarity search)
"""

import os
import uuid

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # turn off Chroma's usage reporting

from langchain_chroma import Chroma  # noqa: E402  (must come after the line above)
from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_core.vectorstores import VectorStoreRetriever  # noqa: E402

# How many chunks the retriever returns. 4 is LangChain's default, as in the lab.
TOP_K = 4


def build_vector_store(chunks: list[Document], embeddings: Embeddings) -> Chroma:
    """Embed every chunk and store the vectors in a new in-memory collection."""
    if not chunks:
        raise ValueError("There are no chunks to embed.")
    collection = f"upload-{uuid.uuid4().hex[:12]}"  # unique name: uploads never mix
    return Chroma.from_documents(
        chunks,
        embeddings,
        collection_name=collection,
        # Chroma ranks by Euclidean (L2) distance unless told otherwise. Cosine
        # compares only the direction of the vectors, the usual measure for
        # text embeddings, so results don't depend on how a model scales them.
        collection_metadata={"hnsw:space": "cosine"},
    )


def get_retriever(vectordb: Chroma, k: int = TOP_K) -> VectorStoreRetriever:
    """Wrap the store as a retriever that returns the k most similar chunks."""
    return vectordb.as_retriever(search_kwargs={"k": k})
