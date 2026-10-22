"""Settings for the QA bot, read from environment variables (or a .env file).

The lab hard-codes its model ids, URL and project id inside each function.
Here they live in one place, so switching from local Ollama to the Gemini API
is a change in .env, not in the code.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Read the .env file (if there is one) into environment variables.
load_dotenv()

# Generation settings from the lab's get_llm(): at most 256 new tokens,
# temperature 0.5.
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.5


@dataclass(frozen=True)
class Settings:
    """Which providers and models to use. frozen=True makes it read-only."""

    llm_provider: str
    ollama_model: str
    ollama_base_url: str
    gemini_model: str
    google_api_key: str
    embeddings_provider: str
    ollama_embed_model: str
    hf_embed_model: str
    gemini_embed_model: str


def load_settings() -> Settings:
    """Build the settings from environment variables, with local defaults.

    Read at call time (not at import time) so tests can change the
    environment with monkeypatch.
    """
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "ollama").strip().lower(),
        ollama_model=os.getenv("OLLAMA_MODEL", "granite4:tiny-h"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        embeddings_provider=os.getenv("EMBEDDINGS_PROVIDER", "ollama").strip().lower(),
        ollama_embed_model=os.getenv("OLLAMA_EMBED_MODEL", "granite-embedding:278m"),
        hf_embed_model=os.getenv(
            "HF_EMBED_MODEL", "ibm-granite/granite-embedding-278m-multilingual"
        ),
        gemini_embed_model=os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-2"),
    )
