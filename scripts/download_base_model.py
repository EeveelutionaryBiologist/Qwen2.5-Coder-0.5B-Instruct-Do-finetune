#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["huggingface_hub"]
# ///
"""Download Qwen2.5-Coder-0.5B-Instruct from Hugging Face into model/ if not already present."""

from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


def main() -> None:
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir(exist_ok=True, parents=True)

    if (MODEL_DIR / "config.json").exists():
        print(f"Model already present in {MODEL_DIR}, skipping download.")
        return

    print(f"Downloading {REPO_ID} into {MODEL_DIR} ...")
    snapshot_download(repo_id=REPO_ID, local_dir=MODEL_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
