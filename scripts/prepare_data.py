"""
Prepare the vegetable dataset for the resource-constrained CNN assignment.

The Kaggle dataset already ships with a train / validation / test split that its
author describes as 70% / 15% / 15% (data/raw/.../{train,validation,test}/<Class>/).

Steps
-----
1. Collect every image under data/raw. The class label is the image's parent
   folder and the original split is the grand-parent folder.
2. Drop exact duplicate files (MD5 hash) so the same image can never appear
   in two splits (data leakage). The copy in the earliest split
   (train -> val -> test) is kept.
3. Assign splits, depending on --split-mode:
     original (default) : keep Kaggle's provided train/validation/test split
     resplit            : ignore it and make a fresh stratified 70/15/15 split
4. Downscale every image to 64x64 RGB (emulates a low-resolution edge sensor)
   and save it as a lossless PNG in:
       data/processed/{train,val,test}/<ClassName>/<image>.png
5. Write data/processed/split_manifest.csv (source path, class, split) so the
   exact split can be audited / reproduced later, and print the per-class
   counts and actual split percentages (report these numbers).

Usage (from the project root):
    python scripts/prepare_data.py                         # keep Kaggle's split
    python scripts/prepare_data.py --split-mode resplit    # exact 70/15/15 re-split
    python scripts/prepare_data.py --overwrite             # rebuild data/processed
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

# Map the folder names used by Kaggle to our split names
SPLIT_ALIASES = {
    "train": "train", "training": "train",
    "validation": "val", "valid": "val", "val": "val",
    "test": "test", "testing": "test",
}


def original_split_of(path: Path) -> str | None:
    """Original split from the folder layout .../<split>/<Class>/<image>, or None."""
    return SPLIT_ALIASES.get(path.parent.parent.name.lower())


def collect_images(raw_dir: Path) -> tuple[list[tuple[Path, str, str | None]], int]:
    """Return (path, class_label, original_split) records and the number of duplicates removed."""
    paths = [p for p in raw_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not paths:
        raise SystemExit(f"No images found under {raw_dir}. Run scripts/download_data.py first.")

    # Visit train first, then val, then test, so a duplicate is kept in the earliest split
    split_rank = {name: i for i, name in enumerate(SPLITS)}
    paths.sort(key=lambda p: (split_rank.get(original_split_of(p), len(SPLITS)), str(p)))

    seen_hashes = set()
    records = []
    for path in tqdm(paths, desc="Scanning images"):
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest in seen_hashes:
            continue  # identical file already collected
        seen_hashes.add(digest)
        # e.g. .../train/Broccoli/0001.jpg -> ("Broccoli", "train")
        records.append((path, path.parent.name, original_split_of(path)))

    return records, len(paths) - len(records)


def split_original(records):
    """Keep the split provided with the dataset -> {split_name: [(path, label), ...]}."""
    unknown = [p for p, _, split in records if split is None]
    if unknown:
        raise SystemExit(
            f"{len(unknown)} images are not inside a train/validation/test folder "
            f"(e.g. {unknown[0]}). Use --split-mode resplit instead."
        )
    return {s: [(p, label) for p, label, split in records if split == s] for s in SPLITS}


def split_stratified(records, seed):
    """Fresh stratified 70/15/15 split -> {split_name: [(path, label), ...]}."""
    paths = [p for p, _, _ in records]
    labels = [label for _, label, _ in records]
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


def output_name(src: Path) -> str:
    """Unique PNG file name (original split + file stem), because Kaggle's
    train/validation/test folders may reuse the same file names."""
    return f"{original_split_of(src) or 'img'}_{src.stem}.png"


def resize_and_save(src: Path, dst: Path, size: int) -> None:
    """Load an image, force 3-channel RGB, downscale to size x size and save as PNG."""
    with Image.open(src) as img:
        img = img.convert("RGB")
        img = img.resize((size, size), Image.Resampling.LANCZOS)  # high-quality downsampling
        img.save(dst, format="PNG", optimize=True)


def main():
    parser = argparse.ArgumentParser(description="Split and downscale the vegetable dataset.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--split-mode", choices=["original", "resplit"], default="original",
                        help="keep Kaggle's split (default) or make a new stratified 70/15/15 split")
    parser.add_argument("--size", type=int, default=64, help="output resolution (max 64)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (resplit mode only)")
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
    records, n_duplicates = collect_images(raw_dir)
    classes = sorted({label for _, label, _ in records})
    print(f"Found {len(records):,} unique images in {len(classes)} classes "
          f"({n_duplicates} duplicates removed).")
    if len(classes) != 15:
        print(f"WARNING: expected 15 classes, found {len(classes)}: {classes}")

    # 3. Assign splits
    if args.split_mode == "original":
        splits = split_original(records)
    else:
        splits = split_stratified(records, args.seed)
    print(f"Split mode: {args.split_mode}")

    # 4. Resize + save, 5. manifest
    manifest_rows = []
    for split in SPLITS:
        for cls in classes:
            (out_dir / split / cls).mkdir(parents=True, exist_ok=True)
        for src, label in tqdm(splits[split], desc=f"Writing {split}"):
            dst = out_dir / split / label / output_name(src)
            resize_and_save(src, dst, args.size)
            manifest_rows.append(
                {"source": src.relative_to(raw_dir).as_posix(), "class": label,
                 "split": split, "output": dst.relative_to(out_dir).as_posix()}
            )

    with open(out_dir / "split_manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "class", "split", "output"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Summary table: images per class per split + actual split percentages
    counts = {split: Counter(label for _, label in splits[split]) for split in SPLITS}
    total = len(records)
    print(f"\n{'Class':<15}" + "".join(f"{s:>8}" for s in SPLITS))
    for cls in classes:
        print(f"{cls:<15}" + "".join(f"{counts[s][cls]:>8}" for s in SPLITS))
    print(f"{'TOTAL':<15}" + "".join(f"{len(splits[s]):>8}" for s in SPLITS))
    print(f"{'Share':<15}" + "".join(f"{len(splits[s]) / total:>8.1%}" for s in SPLITS))
    print(f"\nSaved {args.size}x{args.size} PNGs to {out_dir}")


if __name__ == "__main__":
    main()
