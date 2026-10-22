"""Create the answer model and the embedding model.

Replaces the lab's get_llm() (WatsonxLLM) and watsonx_embedding()
(WatsonxEmbeddings). Providers are chosen in .env:

    Answers      ollama (Granite 4, local, default) or gemini (free API tier)
    Embeddings   ollama       Granite Embedding 278M, local (default)
                 huggingface  the same Granite model on CPU, no Ollama needed
                 gemini       Gemini Embedding through the API: no model in
                              memory, so it fits small servers (the deployed demo)

The optional packages are imported inside the functions, so the local setup
works without installing them.
"""

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from qabot.config import MAX_NEW_TOKENS, TEMPERATURE, Settings, load_settings

LLM_PROVIDERS = ("ollama", "gemini")
EMBEDDINGS_PROVIDERS = ("ollama", "huggingface", "gemini")

# Gemini Embedding returns 3072 numbers by default. 768 matches Granite
# Embedding's size and uses a quarter of the memory in the vector store.
GEMINI_EMBED_DIMENSIONS = 768


class MissingCloudPackages(ImportError):
    """A provider was chosen but its optional packages aren't installed."""

    def __init__(self, setting: str, error: ImportError, extra: str):
        super().__init__(
            f"{setting} needs extra packages ({error.name} is missing). "
            f"Install them with: uv sync --extra {extra}"
        )


def _require_gemini_key(settings: Settings, setting: str) -> None:
    if not settings.google_api_key:
        raise ValueError(f"{setting} needs GOOGLE_API_KEY in .env.")


def get_llm(settings: Settings | None = None) -> BaseChatModel:
    """Return the chat model that writes the answers (lab: get_llm)."""
    settings = settings or load_settings()

    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            num_predict=MAX_NEW_TOKENS,  # Ollama's name for max_new_tokens
            temperature=TEMPERATURE,
        )

    if settings.llm_provider == "gemini":
        _require_gemini_key(settings, "LLM_PROVIDER=gemini")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as error:
            raise MissingCloudPackages("LLM_PROVIDER=gemini", error, "gemini") from error

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            max_output_tokens=MAX_NEW_TOKENS,  # Gemini's name for max_new_tokens
            temperature=TEMPERATURE,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER {settings.llm_provider!r}. Use one of {LLM_PROVIDERS}."
    )


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    """Return the model that turns text chunks into vectors (lab: watsonx_embedding).

    The lab sets TRUNCATE_INPUT_TOKENS: 3, which embeds only the first three
    tokens of each chunk. All options here embed the whole chunk.
    """
    settings = settings or load_settings()

    if settings.embeddings_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
        )

    if settings.embeddings_provider == "huggingface":
        try:
            import torch
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as error:
            raise MissingCloudPackages("EMBEDDINGS_PROVIDER=huggingface", error, "hf") from error

        return HuggingFaceEmbeddings(
            model_name=settings.hf_embed_model,
            # The model is stored in bfloat16, which most CPUs run slowly.
            # float32 is the fast precision on CPU.
            model_kwargs={"device": "cpu", "model_kwargs": {"dtype": torch.float32}},
            encode_kwargs={"normalize_embeddings": True},  # unit length, for cosine similarity
        )

    if settings.embeddings_provider == "gemini":
        _require_gemini_key(settings, "EMBEDDINGS_PROVIDER=gemini")
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError as error:
            raise MissingCloudPackages("EMBEDDINGS_PROVIDER=gemini", error, "gemini") from error

        # Chunks are embedded as RETRIEVAL_DOCUMENT and questions as
        # RETRIEVAL_QUERY (the class's default), which suits search.
        return GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embed_model,
            google_api_key=settings.google_api_key,
            output_dimensionality=GEMINI_EMBED_DIMENSIONS,
        )

    raise ValueError(
        f"Unknown EMBEDDINGS_PROVIDER {settings.embeddings_provider!r}. "
        f"Use one of {EMBEDDINGS_PROVIDERS}."
    )


def describe(settings: Settings | None = None) -> str:
    """One line naming the models in use, for scripts and the app."""
    settings = settings or load_settings()
    llm = settings.ollama_model if settings.llm_provider == "ollama" else settings.gemini_model
    emb = {
        "ollama": settings.ollama_embed_model,
        "huggingface": settings.hf_embed_model,
        "gemini": settings.gemini_embed_model,
    }.get(settings.embeddings_provider, settings.embeddings_provider)
    return f"answers: {llm} ({settings.llm_provider}), embeddings: {emb} ({settings.embeddings_provider})"
