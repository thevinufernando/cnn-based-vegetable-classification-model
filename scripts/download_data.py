"""
Download the Kaggle "Vegetable Image Dataset" into data/raw using kagglehub.

Dataset : https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset
          15 classes x 1,400 RGB images (224x224 JPG), shipped pre-split into
          train / validation / test folders.

kagglehub downloads the archive and extracts it automatically, so no manual
unzipping is needed. The dataset is public, so normally no Kaggle login is
required. If you get a 401/403 error, create an API token on kaggle.com
(Settings -> API -> Create New Token) and place kaggle.json in
%USERPROFILE%\\.kaggle\\

Usage (from the project root):
    python scripts/download_data.py
    python scripts/download_data.py --force     # re-download even if present
"""

import argparse
from pathlib import Path

import kagglehub

# Project root = parent of the scripts/ folder, so the script works from any cwd
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DATASET_HANDLE = "misrakahmed/vegetable-image-dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def download(force: bool = False) -> Path:
    """Download (and auto-extract) the dataset into data/raw and return its path."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = kagglehub.dataset_download(
        DATASET_HANDLE, output_dir=str(RAW_DIR), force_download=force
    )
    return Path(path)


def summarise(folder: Path) -> None:
    """Print the top-level folder structure and number of images found."""
    n_images = sum(1 for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    print(f"\nDataset location : {folder}")
    print(f"Images found     : {n_images:,}")
    print("Top-level contents:")
    for child in sorted(folder.iterdir()):
        print(f"  {child.name}{'/' if child.is_dir() else ''}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the vegetable image dataset.")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    dataset_path = download(force=args.force)
    summarise(dataset_path)
    print("\nDone. Next step: python scripts/prepare_data.py")
