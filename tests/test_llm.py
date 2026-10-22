"""Tests for provider selection. No model is called and nothing is downloaded."""

import sys

import pytest

from qabot.config import MAX_NEW_TOKENS, TEMPERATURE, load_settings
from qabot.llm import MissingCloudPackages, describe, get_embeddings, get_llm


@pytest.fixture
def clean_env(monkeypatch):
    """Remove the settings variables so each test starts from the defaults."""
    for name in [
        "LLM_PROVIDER", "OLLAMA_MODEL", "OLLAMA_BASE_URL", "GEMINI_MODEL",
        "GOOGLE_API_KEY", "EMBEDDINGS_PROVIDER", "OLLAMA_EMBED_MODEL", "HF_EMBED_MODEL",
        "GEMINI_EMBED_MODEL",
    ]:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_defaults_are_local(clean_env):
    settings = load_settings()
    assert settings.llm_provider == "ollama"
    assert settings.embeddings_provider == "ollama"
    assert settings.ollama_model == "granite4:tiny-h"


def test_ollama_llm_uses_lab_parameters(clean_env):
    llm = get_llm()
    assert type(llm).__name__ == "ChatOllama"
    assert llm.num_predict == MAX_NEW_TOKENS
    assert llm.temperature == TEMPERATURE


def test_ollama_embeddings(clean_env):
    assert type(get_embeddings()).__name__ == "OllamaEmbeddings"


def test_provider_names_ignore_case_and_spaces(clean_env):
    clean_env.setenv("LLM_PROVIDER", "  Ollama ")
    assert load_settings().llm_provider == "ollama"


def test_unknown_providers_raise(clean_env):
    clean_env.setenv("LLM_PROVIDER", "watsonx")
    clean_env.setenv("EMBEDDINGS_PROVIDER", "watsonx")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        get_llm()
    with pytest.raises(ValueError, match="Unknown EMBEDDINGS_PROVIDER"):
        get_embeddings()


def test_gemini_needs_api_key(clean_env):
    clean_env.setenv("LLM_PROVIDER", "gemini")
    clean_env.setenv("EMBEDDINGS_PROVIDER", "gemini")
    with pytest.raises(ValueError, match="LLM_PROVIDER=gemini needs GOOGLE_API_KEY"):
        get_llm()
    with pytest.raises(ValueError, match="EMBEDDINGS_PROVIDER=gemini needs GOOGLE_API_KEY"):
        get_embeddings()


def test_gemini_embeddings(clean_env):
    pytest.importorskip("langchain_google_genai")  # only with the 'gemini' extra
    clean_env.setenv("EMBEDDINGS_PROVIDER", "gemini")
    clean_env.setenv("GOOGLE_API_KEY", "fake-key-for-tests")
    embeddings = get_embeddings()
    assert type(embeddings).__name__ == "GoogleGenerativeAIEmbeddings"
    assert (embeddings.model, embeddings.output_dimensionality) == ("gemini-embedding-2", 768)
    assert "gemini-embedding-2 (gemini)" in describe()


def test_gemini_llm(clean_env):
    pytest.importorskip("langchain_google_genai")  # only with the 'gemini' extra
    clean_env.setenv("LLM_PROVIDER", "gemini")
    clean_env.setenv("GOOGLE_API_KEY", "fake-key-for-tests")
    llm = get_llm()
    assert type(llm).__name__ == "ChatGoogleGenerativeAI"
    assert llm.max_output_tokens == MAX_NEW_TOKENS


def test_missing_cloud_packages_explain_the_fix(clean_env):
    # A None entry in sys.modules makes "import x" fail, as if x weren't installed.
    clean_env.setitem(sys.modules, "langchain_google_genai", None)
    clean_env.setitem(sys.modules, "torch", None)
    clean_env.setenv("LLM_PROVIDER", "gemini")
    clean_env.setenv("GOOGLE_API_KEY", "fake-key-for-tests")
    clean_env.setenv("EMBEDDINGS_PROVIDER", "huggingface")

    with pytest.raises(MissingCloudPackages, match="uv sync --extra gemini"):
        get_llm()
    with pytest.raises(MissingCloudPackages, match="torch is missing.*--extra hf"):
        get_embeddings()
    clean_env.setenv("EMBEDDINGS_PROVIDER", "gemini")
    with pytest.raises(MissingCloudPackages, match="uv sync --extra gemini"):
        get_embeddings()


def test_describe_names_both_models(clean_env):
    text = describe()
    assert "granite4:tiny-h" in text and "granite-embedding:278m" in text
