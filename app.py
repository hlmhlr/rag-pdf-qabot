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

import sys
from pathlib import Path

import gradio as gr

# Locally, uv installs the qabot package. On Hugging Face Spaces only
# requirements.txt is installed, so the package is imported from src/.
try:
    import qabot  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from qabot.index import QABot  # noqa: E402
from qabot.ingest import MAX_TOTAL_MB  # noqa: E402
from qabot.llm import describe, get_embeddings, get_llm  # noqa: E402
from qabot.paths import SAMPLES_DIR  # noqa: E402
from qabot.ui import build_app  # noqa: E402

SAMPLES = sorted(str(p) for p in SAMPLES_DIR.glob("*.pdf"))
EXAMPLES = [
    [SAMPLES, "How many days of annual leave do full-time employees get?"],
    [SAMPLES, "What does error E07 on the display mean?"],
    [SAMPLES, "Can employees get a discount on an e-bike, and how long is the battery covered?"],
]

app = build_app(QABot(get_llm(), get_embeddings()), models=describe(), examples=EXAMPLES)

if __name__ == "__main__":
    app.launch(server_port=7860, max_file_size=f"{MAX_TOTAL_MB}mb", theme=gr.themes.Soft())
