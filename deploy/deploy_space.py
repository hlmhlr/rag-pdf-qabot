"""Build the Hugging Face Space bundle and publish it.

Aim
    Deploy the app with one command (run by GitHub Actions on every release
    tag), so the live demo always matches a tested version of the code.

What it does
    1. Copies what the Space needs into a staging folder: app.py, the qabot
       package, the sample PDFs, requirements.txt, and the Space's README
       with the Gradio version from this environment filled in.
    2. Creates the Space if it doesn't exist yet.
    3. Sets the Space's variables (cloud providers) and its secret
       (GOOGLE_API_KEY), which never appears in any file.
    4. Uploads the staging folder, replacing the previous version.

Needs (environment variables)
    HF_TOKEN        a Hugging Face token with write access
    HF_SPACE_ID     e.g. "your-name/rag-pdf-qabot"
    GOOGLE_API_KEY  the Gemini key, stored as a Space secret

Run:  uv run --with huggingface_hub python deploy/deploy_space.py
      uv run python deploy/deploy_space.py --dry-run     (build the bundle only)
"""

import argparse
import os
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / "deploy"

# Settings for the Space. The API key is set separately, as a secret.
SPACE_VARIABLES = {
    "LLM_PROVIDER": "gemini",
    "EMBEDDINGS_PROVIDER": "huggingface",
    "GEMINI_MODEL": os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
}


def build_bundle(out: Path) -> list[str]:
    """Copy everything the Space needs into out/ and return the file list."""
    shutil.copy(ROOT / "app.py", out / "app.py")
    shutil.copytree(ROOT / "src" / "qabot", out / "src" / "qabot",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (out / "data" / "samples").mkdir(parents=True)
    for pdf in (ROOT / "data" / "samples").glob("*.pdf"):
        shutil.copy(pdf, out / "data" / "samples" / pdf.name)
    shutil.copy(DEPLOY / "requirements.txt", out / "requirements.txt")

    # The Space installs exactly the Gradio version the tests ran with.
    readme = (DEPLOY / "README.md").read_text(encoding="utf-8")
    (out / "README.md").write_text(readme.replace("{gradio_version}", version("gradio")),
                                   encoding="utf-8")
    return sorted(str(p.relative_to(out)).replace("\\", "/") for p in out.rglob("*") if p.is_file())


def publish(out: Path, space_id: str, token: str, api_key: str) -> None:
    """Create or update the Space, set its settings, and upload the bundle."""
    from huggingface_hub import HfApi  # only needed for a real deployment

    api = HfApi(token=token)
    api.create_repo(space_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    for key, value in SPACE_VARIABLES.items():
        api.add_space_variable(space_id, key, value)
    api.add_space_secret(space_id, "GOOGLE_API_KEY", api_key)
    api.upload_folder(
        repo_id=space_id,
        repo_type="space",
        folder_path=out,
        commit_message=f"Deploy {os.getenv('GITHUB_REF_NAME', 'manual')} from GitHub",
        delete_patterns=["*.py", "src/**", "data/**"],  # remove files that no longer exist
    )
    print(f"Deployed: https://huggingface.co/spaces/{space_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="build the bundle and list it, don't upload")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        files = build_bundle(out)
        print("Bundle:\n  " + "\n  ".join(files))
        if args.dry_run:
            return

        missing = [n for n in ("HF_TOKEN", "HF_SPACE_ID", "GOOGLE_API_KEY") if not os.getenv(n)]
        if missing:
            raise SystemExit(f"Missing environment variables: {', '.join(missing)}")
        publish(out, os.environ["HF_SPACE_ID"], os.environ["HF_TOKEN"], os.environ["GOOGLE_API_KEY"])


if __name__ == "__main__":
    main()
