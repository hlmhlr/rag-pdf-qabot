"""Tests for the Render deployment settings. Nothing is deployed."""

import tomllib
from pathlib import Path

import yaml

from qabot.ui import server_settings

ROOT = Path(__file__).resolve().parent.parent


def render_service() -> dict:
    [service] = yaml.safe_load((ROOT / "render.yaml").read_text(encoding="utf-8"))["services"]
    return service


def test_render_service_is_free_and_installs_the_locked_versions():
    service = render_service()
    assert (service["type"], service["plan"]) == ("web", "free")
    assert "uv sync --locked" in service["buildCommand"]
    assert "--no-dev" in service["buildCommand"]


def test_render_uses_gemini_for_both_models_and_keeps_the_key_secret():
    env = {v["key"]: v for v in render_service()["envVars"]}
    assert env["LLM_PROVIDER"]["value"] == "gemini"
    assert env["EMBEDDINGS_PROVIDER"]["value"] == "gemini"
    assert env["GOOGLE_API_KEY"] == {"key": "GOOGLE_API_KEY", "sync": False}  # no value in the repo


def test_render_installs_an_extra_that_exists_and_has_no_pytorch():
    """The build's --extra must exist in pyproject.toml, and must not pull in
    PyTorch, which wouldn't fit in the free plan's 512 MB."""
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "optional-dependencies"]
    build = render_service()["buildCommand"].split()
    extra = build[build.index("--extra") + 1]
    assert extra in extras
    assert not any(p.startswith(("torch", "sentence-transformers", "qabot[")) for p in extras[extra])


def test_python_version_matches_the_project():
    env = {v["key"]: v.get("value") for v in render_service()["envVars"]}
    assert env["PYTHON_VERSION"].startswith("3.12.")  # full version, as Render requires


def test_server_listens_locally_by_default_and_on_the_hosts_port():
    assert server_settings({}) == {"server_port": 7860}
    assert server_settings({"PORT": "10000"}) == {"server_name": "0.0.0.0", "server_port": 10000}
