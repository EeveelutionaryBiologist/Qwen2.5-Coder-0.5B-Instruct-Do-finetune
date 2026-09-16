#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["huggingface_hub"]
# ///
"""Download the NL2SH-ALFA dataset from Hugging Face into data/example if not already present."""

from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "westenfelder/NL2SH-ALFA"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "example"


def main() -> None:
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(exist_ok=True, parents=True)

    if (DATA_DIR / "train.csv").exists():
        print(f"Dataset already present in {DATA_DIR}, skipping download.")
        return

    print(f"Downloading {REPO_ID} into {DATA_DIR} ...")
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=DATA_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
