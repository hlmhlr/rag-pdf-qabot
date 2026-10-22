"""Check that the answer model and the embedding model both work.

Aim
    Catch setup problems (Ollama not running, model not pulled, missing API
    key) before building anything on top of the models.

What you'll see
    The providers and models in use, then one line per check: a short answer
    from the chat model, and the size of one embedding vector (768 numbers
    for Granite Embedding 278M).

Run:  uv run python scripts/00_check_setup.py
"""

import json
import sys
import time
import urllib.request

from qabot.config import load_settings
from qabot.llm import describe, get_embeddings, get_llm


def pulled_ollama_models(base_url: str) -> list[str]:
    """Ask the Ollama server which models are downloaded (GET /api/tags)."""
    with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as response:
        return [m["name"] for m in json.load(response)["models"]]


def check_ollama(settings) -> bool:
    """Confirm Ollama is running and every Ollama model in use is pulled."""
    needed = []
    if settings.llm_provider == "ollama":
        needed.append(settings.ollama_model)
    if settings.embeddings_provider == "ollama":
        needed.append(settings.ollama_embed_model)
    if not needed:
        return True  # cloud setup: Ollama isn't used

    try:
        pulled = pulled_ollama_models(settings.ollama_base_url)
    except OSError:
        print(f"FAIL  Ollama is not reachable at {settings.ollama_base_url}. Is it running?")
        return False

    ok = True
    for model in needed:
        # Ollama adds ":latest" to names given without a tag.
        found = model in pulled or f"{model}:latest" in pulled
        print(f"{'OK  ' if found else 'FAIL'}  Ollama model {model}")
        if not found:
            print(f"      Download it with: ollama pull {model}")
            ok = False
    return ok


def main() -> None:
    settings = load_settings()
    print(f"Using {describe(settings)}\n")

    if not check_ollama(settings):
        sys.exit(1)

    try:
        start = time.perf_counter()
        answer = get_llm(settings).invoke("Answer in one word: what is the capital of Italy?")
        print(f"OK    Chat model answered {answer.content.strip()!r} "
              f"in {time.perf_counter() - start:.1f} s")

        start = time.perf_counter()
        vector = get_embeddings(settings).embed_query("A short test sentence.")
        print(f"OK    Embedding model returned {len(vector)} numbers "
              f"in {time.perf_counter() - start:.1f} s")
    except Exception as error:  # show any setup error in one readable line
        print(f"FAIL  {type(error).__name__}: {error}")
        sys.exit(1)

    print("\nSetup looks good.")


if __name__ == "__main__":
    main()
