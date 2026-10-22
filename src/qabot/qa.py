"""Answer questions from retrieved chunks (lab: retriever_qa with RetrievalQA).

The lab uses RetrievalQA with chain_type="stuff": retrieve chunks, "stuff"
them all into one prompt, ask the LLM. RetrievalQA is now a legacy chain
(moved to langchain-classic), so this module builds the same steps as an
LCEL pipeline, which makes each step visible and lets it return the
retrieved chunks alongside the answer.

Two changes to the lab's behaviour
    - The prompt tells the model to answer only from the documents and to
      say so when they don't contain the answer, instead of guessing.
    - Every answer comes with its sources (file and page), so a user can
      check it. With several uploaded files this is how you know which
      document an answer came from.

Contents
    QA_PROMPT         the instruction, the numbered context and the question
    format_docs()     join chunks into the context, labelled [1] file p.N
    sources_of()      the distinct file/page pairs, in ranking order
    build_qa_chain()  retriever -> prompt -> llm -> text, plus the chunks used
"""

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough

NO_ANSWER = "I couldn't find the answer in the uploaded documents."

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about the user's documents. Use only the numbered "
            "context passages below. If they don't contain the answer, reply exactly: "
            f'"{NO_ANSWER}" Keep answers short: at most three sentences.\n\n'
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def page_label(doc: Document) -> str:
    """"file.pdf p.3": the page counted from 1, the way people read it."""
    return f"{doc.metadata['file']} p.{doc.metadata['page'] + 1}"


def format_docs(docs: list[Document]) -> str:
    """Join chunks into one context string, each labelled with where it came from."""
    return "\n\n".join(
        f"[{number}] ({page_label(doc)})\n{doc.page_content}"
        for number, doc in enumerate(docs, start=1)
    )


def sources_of(docs: list[Document]) -> list[str]:
    """Distinct "file p.N" labels in ranking order (a page can give several chunks)."""
    return list(dict.fromkeys(page_label(doc) for doc in docs))


def build_qa_chain(retriever: BaseRetriever, llm: BaseChatModel) -> Runnable:
    """Build the QA pipeline.

    Input:  {"question": "..."}
    Output: {"question": ..., "docs": [retrieved chunks], "answer": "..."}

    Each .assign() step adds one key to the dict and keeps the others, so the
    retrieved chunks are still there at the end to show as sources.
    """
    answer_step = (
        (lambda x: {"context": format_docs(x["docs"]), "question": x["question"]})
        | QA_PROMPT
        | llm
        | StrOutputParser()  # message -> plain text (also handles Gemini's list content)
    )
    return RunnablePassthrough.assign(
        docs=lambda x: retriever.invoke(x["question"])
    ).assign(answer=answer_step)
