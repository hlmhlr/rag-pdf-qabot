"""Web app: upload PDFs and ask questions about them (lab section: Gradio interface).

The lab's interface: a file upload, a question box and an answer box, wired
to one function. Differences from the lab:
    - several PDFs can be uploaded at once (file_count="multiple");
    - allow_flagging is now called flagging_mode in Gradio 6;
    - no share=True: the lab's public tunnel link would let anyone on the
      internet use your models. The app runs on your own computer only.

Run:  uv run python app.py      then open http://127.0.0.1:7860
"""

import gradio as gr

from qabot.index import QABot
from qabot.llm import get_embeddings, get_llm

bot = QABot(get_llm(), get_embeddings())


def retriever_qa(files: list[str] | None, query: str) -> str:
    """The lab's function: answer a question about the uploaded files."""
    answer = bot.ask(files or [], query)
    return answer.text


rag_application = gr.Interface(
    fn=retriever_qa,
    flagging_mode="never",
    inputs=[
        gr.File(label="Upload PDF files", file_count="multiple", file_types=[".pdf"]),
        gr.Textbox(label="Input query", lines=2, placeholder="Type your question here..."),
    ],
    outputs=gr.Textbox(label="Output"),
    title="RAG PDF QA Bot",
    description="Upload one or more PDF documents and ask any question about them.",
)

if __name__ == "__main__":
    rag_application.launch(server_port=7860)
