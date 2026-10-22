"""Start the web app: upload PDFs and ask questions about them.

The page and its handlers are in src/qabot/ui.py; this file only creates the
models and launches the app. Hugging Face Spaces runs this same file.

Differences from the lab's launch:
    - no share=True: the lab's public tunnel link would let anyone on the
      internet use your models. Locally the app is only on your computer;
      online it's deployed on purpose (Stage 5).
    - max_file_size makes Gradio refuse oversized files before they reach
      the bot.

Run:  uv run python app.py      then open http://127.0.0.1:7860
"""

import gradio as gr

from qabot.index import QABot
from qabot.ingest import MAX_TOTAL_MB
from qabot.llm import describe, get_embeddings, get_llm
from qabot.paths import SAMPLES_DIR
from qabot.ui import build_app

SAMPLES = sorted(str(p) for p in SAMPLES_DIR.glob("*.pdf"))
EXAMPLES = [
    [SAMPLES, "How many days of annual leave do full-time employees get?"],
    [SAMPLES, "What does error E07 on the display mean?"],
    [SAMPLES, "Can employees get a discount on an e-bike, and how long is the battery covered?"],
]

app = build_app(QABot(get_llm(), get_embeddings()), models=describe(), examples=EXAMPLES)

if __name__ == "__main__":
    app.launch(server_port=7860, max_file_size=f"{MAX_TOTAL_MB}mb", theme=gr.themes.Soft())
