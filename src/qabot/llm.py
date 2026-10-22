"""Create the answer model and the embedding model.

Replaces the lab's get_llm() (WatsonxLLM) and watsonx_embedding()
(WatsonxEmbeddings). Two setups are supported:

    Local (default)  Granite 4 for answers and Granite Embedding 278M,
                     both through Ollama on your own computer.
    Cloud            Gemini (free API tier) for answers, and the same Granite
                     embedding model run on CPU with sentence-transformers.

The cloud packages are imported inside the functions, so the local setup
works without installing them.
"""

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from qabot.config import MAX_NEW_TOKENS, TEMPERATURE, Settings, load_settings

LLM_PROVIDERS = ("ollama", "gemini")
EMBEDDINGS_PROVIDERS = ("ollama", "huggingface")


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
        if not settings.google_api_key:
            raise ValueError("LLM_PROVIDER=gemini needs GOOGLE_API_KEY in .env.")
        from langchain_google_genai import ChatGoogleGenerativeAI

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
    tokens of each chunk. Both options here embed the whole chunk (Granite
    Embedding reads up to 512 tokens).
    """
    settings = settings or load_settings()

    if settings.embeddings_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
        )

    if settings.embeddings_provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=settings.hf_embed_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # unit length, for cosine similarity
        )

    raise ValueError(
        f"Unknown EMBEDDINGS_PROVIDER {settings.embeddings_provider!r}. "
        f"Use one of {EMBEDDINGS_PROVIDERS}."
    )


def describe(settings: Settings | None = None) -> str:
    """One line naming the models in use, for scripts and the app."""
    settings = settings or load_settings()
    llm = settings.ollama_model if settings.llm_provider == "ollama" else settings.gemini_model
    emb = (
        settings.ollama_embed_model
        if settings.embeddings_provider == "ollama"
        else settings.hf_embed_model
    )
    return f"answers: {llm} ({settings.llm_provider}), embeddings: {emb} ({settings.embeddings_provider})"
