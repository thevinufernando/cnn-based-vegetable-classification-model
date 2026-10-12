"""
Prepare the vegetable dataset for the resource-constrained CNN assignment.

Steps
-----
1. Collect every image under data/raw (ignoring Kaggle's original
   train/validation/test split). The class label is the image's parent folder.
2. Drop exact duplicate files (MD5 hash) so the same image can never end up
   in two splits (data leakage).
3. Stratified split into 70% train / 15% validation / 15% test, so every
   class keeps the same proportion in each split.
4. Downscale every image to 64x64 RGB (emulates a low-resolution edge sensor)
   and save it as a lossless PNG in:
       data/processed/{train,val,test}/<ClassName>/<image>.png
5. Write data/processed/split_manifest.csv (source path, class, split) so the
   exact same split can be audited / reproduced later.

Usage (from the project root):
    python scripts/prepare_data.py
    python scripts/prepare_data.py --overwrite          # rebuild data/processed
    python scripts/prepare_data.py --size 48 --seed 7   # other options
"""

import argparse
import csv
import hashlib
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = ROOT / "data" / "raw"
DEFAULT_OUT_DIR = ROOT / "data" / "processed"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")


def collect_images(raw_dir: Path) -> tuple[list[Path], list[str], int]:
    """Return image paths, their class labels and the number of duplicates removed."""
    paths = sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise SystemExit(f"No images found under {raw_dir}. Run scripts/download_data.py first.")

    seen_hashes = set()
    unique_paths, labels = [], []
    for path in tqdm(paths, desc="Scanning images"):
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest in seen_hashes:
            continue  # identical file already collected
        seen_hashes.add(digest)
        unique_paths.append(path)
        labels.append(path.parent.name)  # e.g. .../train/Broccoli/0001.jpg -> "Broccoli"

    return unique_paths, labels, len(paths) - len(unique_paths)


def split_dataset(paths, labels, seed):
    """Stratified 70/15/15 split -> dict {split_name: [(path, label), ...]}."""
    # 70% train, 30% temporary pool
    train_p, temp_p, train_y, temp_y = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=seed
    )
    # Split the 30% pool in half -> 15% validation, 15% test
    val_p, test_p, val_y, test_y = train_test_split(
        temp_p, temp_y, test_size=0.50, stratify=temp_y, random_state=seed
    )
    return {
        "train": list(zip(train_p, train_y)),
        "val": list(zip(val_p, val_y)),
        "test": list(zip(test_p, test_y)),
    }


def output_name(src: Path, raw_dir: Path) -> str:
    """Unique PNG file name built from the source path (original split + file stem),
    because Kaggle's train/validation/test folders may reuse the same file names."""
    rel_parts = src.relative_to(raw_dir).parts
    original_split = rel_parts[-3] if len(rel_parts) >= 3 else "img"
    return f"{original_split}_{src.stem}.png"


def resize_and_save(src: Path, dst: Path, size: int) -> None:
    """Load an image, force 3-channel RGB, downscale to size x size and save as PNG."""
    with Image.open(src) as img:
        img = img.convert("RGB")
        img = img.resize((size, size), Image.Resampling.LANCZOS)  # high-quality downsampling
        img.save(dst, format="PNG", optimize=True)


def main():
    parser = argparse.ArgumentParser(description="Re-split and downscale the vegetable dataset.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--size", type=int, default=64, help="output resolution (max 64)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for the split")
    parser.add_argument("--overwrite", action="store_true", help="delete existing out-dir first")
    args = parser.parse_args()

    if args.size > 64:
        raise SystemExit("The assignment caps the resolution at 64x64.")

    raw_dir, out_dir = args.raw_dir.resolve(), args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{out_dir} is not empty. Re-run with --overwrite to rebuild it.")
        shutil.rmtree(out_dir)

    # 1-2. Collect + de-duplicate
    paths, labels, n_duplicates = collect_images(raw_dir)
    classes = sorted(set(labels))
    print(f"Found {len(paths):,} unique images in {len(classes)} classes "
          f"({n_duplicates} duplicates removed).")
    if len(classes) != 15:
        print(f"WARNING: expected 15 classes, found {len(classes)}: {classes}")

    # 3. Stratified split
    splits = split_dataset(paths, labels, args.seed)

    # 4. Resize + save, 5. manifest
    manifest_rows = []
    for split in SPLITS:
        for cls in classes:
            (out_dir / split / cls).mkdir(parents=True, exist_ok=True)
        for src, label in tqdm(splits[split], desc=f"Writing {split}"):
            dst = out_dir / split / label / output_name(src, raw_dir)
            resize_and_save(src, dst, args.size)
            manifest_rows.append(
                {"source": src.relative_to(raw_dir).as_posix(), "class": label,
                 "split": split, "output": dst.relative_to(out_dir).as_posix()}
            )

    with open(out_dir / "split_manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "class", "split", "output"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Summary table: images per class per split
    counts = {split: Counter(label for _, label in splits[split]) for split in SPLITS}
    total = len(paths)
    print(f"\n{'Class':<15}" + "".join(f"{s:>8}" for s in SPLITS))
    for cls in classes:
        print(f"{cls:<15}" + "".join(f"{counts[s][cls]:>8}" for s in SPLITS))
    print(f"{'TOTAL':<15}" + "".join(f"{len(splits[s]):>8}" for s in SPLITS))
    print("Share          " + "".join(f"{len(splits[s]) / total:>8.1%}" for s in SPLITS))
    print(f"\nSaved {args.size}x{args.size} PNGs to {out_dir}")


if __name__ == "__main__":
    main()
