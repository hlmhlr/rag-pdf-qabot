"""Tests for the Hugging Face Space bundle. Nothing is uploaded."""

import re
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "deploy"))

import deploy_space  # noqa: E402


def package_name(requirement: str) -> str:
    """'langchain-core>=1.0' -> 'langchain-core'."""
    return re.split(r"[<>=!~\[ ;]", requirement.strip(), maxsplit=1)[0].lower()


def test_bundle_has_everything_the_space_needs(tmp_path):
    files = deploy_space.build_bundle(tmp_path)
    for needed in ["app.py", "README.md", "requirements.txt", "src/qabot/ui.py",
                   "data/samples/volta_x2_ebike_manual.pdf"]:
        assert needed in files
    assert not any("__pycache__" in f for f in files)


def test_space_readme_uses_the_installed_gradio_and_python_312(tmp_path):
    deploy_space.build_bundle(tmp_path)
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert f'sdk_version: "{version("gradio")}"' in readme
    assert 'python_version: "3.12"' in readme  # Spaces default to 3.10


def test_space_requirements_match_pyproject():
    """Every runtime and cloud dependency must also be installed on the Space."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    needed = {package_name(r) for r in project["dependencies"] + project["optional-dependencies"]["cloud"]}
    needed.discard("gradio")  # installed by the Space itself, from sdk_version

    lines = (ROOT / "deploy" / "requirements.txt").read_text(encoding="utf-8").splitlines()
    on_space = {package_name(line) for line in lines if line.strip() and not line.startswith(("#", "-"))}
    assert needed <= on_space, f"missing on the Space: {sorted(needed - on_space)}"
