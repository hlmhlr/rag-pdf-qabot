"""The Gradio web app: upload PDFs once, then ask as many questions as you like.

The lab's gr.Interface sends the file and the question together on every
click, so each question re-reads the PDF. This version:
    - indexes the files as soon as they are uploaded, so a problem (a scanned
      PDF, too many files) is shown right away, before any question;
    - answers each question from the cached index (see index.py);
    - shows the retrieved sources and how long the answer took;
    - turns every expected error into a message in the page, never a crash.

The handlers are plain functions that take the bot as an argument, so the
tests can call them with fake models and no browser.

Contents
    index_files()      runs on upload: build the index, return a status line
    answer_question()  runs on "Ask": answer, sources and status
    build_app()        the page layout, wired to the handlers
    server_settings()  where to listen: this computer only, or a host's PORT
"""

import logging
import os
from functools import partial

import gradio as gr

from qabot.index import QABot
from qabot.ingest import MAX_FILES, MAX_TOTAL_MB, IngestError

log = logging.getLogger(__name__)

START_MESSAGE = f"Upload up to {MAX_FILES} PDF files ({MAX_TOTAL_MB} MB in total) to start."
MODEL_ERROR = (
    "The model could not answer ({name}). If you run locally, check that Ollama is "
    "running; with Gemini, check your API key, or wait a minute if the free quota is used up."
)


def index_files(bot: QABot, files: list[str] | None) -> str:
    """Build (or reuse) the index for the uploaded files; return a status line."""
    if not files:
        return START_MESSAGE
    try:
        index, reused = bot.cache.get_or_build(files)
    except IngestError as error:
        return f"⚠️ {error}"
    except Exception as error:  # embedding model unreachable, out of memory, ...
        log.exception("Indexing failed")
        return "⚠️ " + MODEL_ERROR.format(name=type(error).__name__)

    if reused:
        return f"✅ Ready: {len(index.files)} file(s), already indexed."
    return (f"✅ Ready: {len(index.files)} file(s), {index.chunks} chunks indexed "
            f"in {index.build_seconds:.1f} s. Ask a question.")


def answer_question(bot: QABot, files: list[str] | None, question: str) -> tuple[str, str, str]:
    """Answer one question. Returns (answer, sources, status) for the three output boxes."""
    if not files:
        return "", "", START_MESSAGE
    try:
        answer = bot.ask(files, question)
    except (IngestError, ValueError) as error:  # bad files or an empty question
        return "", "", f"⚠️ {error}"
    except Exception as error:  # model unreachable, rate limit, ...
        log.exception("Answering failed")
        return "", "", "⚠️ " + MODEL_ERROR.format(name=type(error).__name__)

    sources = (
        "**Retrieved passages** (best match first): " + " · ".join(answer.sources)
        if answer.sources else ""
    )
    index_note = "index reused" if answer.reused_index else "index built"
    return answer.text, sources, f"Answered in {answer.seconds:.1f} s ({index_note})."


def build_app(bot: QABot, models: str = "", examples: list[list] | None = None) -> gr.Blocks:
    """Lay out the page and connect each control to its handler."""
    # Gradio deletes uploaded files older than an hour, checking every hour,
    # so a deployed app doesn't keep people's documents.
    with gr.Blocks(title="RAG PDF QA Bot", delete_cache=(3600, 3600)) as app:
        gr.Markdown(
            "# RAG PDF QA Bot\n"
            "Upload PDF documents and ask questions about them. Answers come only "
            "from your files, with the pages they were found on."
        )
        with gr.Row():
            with gr.Column(scale=2):
                files = gr.File(
                    label="PDF files", file_count="multiple", file_types=[".pdf"], type="filepath"
                )
                status = gr.Markdown(START_MESSAGE)
            with gr.Column(scale=3):
                question = gr.Textbox(
                    label="Question", lines=2, placeholder="e.g. How many days of annual leave do I get?"
                )
                ask = gr.Button("Ask", variant="primary")
                answer = gr.Textbox(label="Answer", lines=5, interactive=False)
                sources = gr.Markdown()

        if examples:
            gr.Examples(examples=examples, inputs=[files, question], label="Try the sample documents")
        if models:
            gr.Markdown(f"<small>Models: {models}. Each question is answered on its own "
                        "(no chat memory).</small>")

        # A new upload (or clearing the files) resets the answer and rebuilds the index.
        files.change(lambda: ("", ""), outputs=[answer, sources], api_visibility="private")
        files.change(partial(index_files, bot), inputs=files, outputs=status, api_name="index")

        # The button and the Enter key both ask (one public endpoint, named "ask").
        ask_event = {"fn": partial(answer_question, bot), "inputs": [files, question],
                     "outputs": [answer, sources, status]}
        ask.click(**ask_event, api_name="ask")
        question.submit(**ask_event, api_visibility="private")
    return app


def server_settings(env: dict[str, str] | None = None) -> dict:
    """Where the app listens, for app.launch().

    Locally: port 7860 on this computer only (Gradio's default address).
    On a host like Render, which sets PORT: that port on every network
    interface (0.0.0.0), so the host's router can reach the app.
    """
    env = os.environ if env is None else env
    if "PORT" in env:
        return {"server_name": "0.0.0.0", "server_port": int(env["PORT"])}
    return {"server_port": 7860}
