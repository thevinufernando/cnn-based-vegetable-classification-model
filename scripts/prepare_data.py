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
DEFAULT_SPLIT_FILE = ROOT / "splits.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")
SPLIT_FILE_FIELDS = ["source", "class", "split", "md5"]

# Map the folder names used by Kaggle to our split names
SPLIT_ALIASES = {
    "train": "train", "training": "train",
    "validation": "val", "valid": "val", "val": "val",
    "test": "test", "testing": "test",
}


def md5_of(path: Path) -> str:
    """MD5 hash of a file's bytes (identifies identical images)."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def original_split_of(path: Path) -> str | None:
    """Original split from the folder layout .../<split>/<Class>/<image>, or None."""
    return SPLIT_ALIASES.get(path.parent.parent.name.lower())



# Mode "file": read the shared split

def load_split_file(split_file: Path, raw_dir: Path):
    """Read splits.csv -> ({split: [(path, label), ...]}, {path: md5}).
    Stops with an error if any listed image is missing or differs from the group's copy."""
    if not split_file.is_file():
        raise SystemExit(
            f"{split_file} not found. Pull it from the repository, or create it once with:\n"
            "  python scripts/prepare_data.py --split-mode original --save-split-file"
        )

    with open(split_file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    splits = {s: [] for s in SPLITS}
    hashes, missing, mismatched = {}, [], []
    for row in tqdm(rows, desc="Verifying images"):
        path = raw_dir / row["source"]
        if not path.is_file():
            missing.append(row["source"])
            continue
        if "md5" in row and row["md5"]:
            digest = md5_of(path)
            if digest != row["md5"]:
                mismatched.append(row["source"])
                continue
            hashes[path] = digest
        if row["split"] not in splits:
            raise SystemExit(f"Unknown split '{row['split']}' in {split_file} ({row['source']}).")
        splits[row["split"]].append((path, row["class"]))

    if missing or mismatched:
        for name, items in (("missing", missing), ("different (MD5 mismatch)", mismatched)):
            if items:
                print(f"{len(items)} images {name}, e.g. {items[:3]}")
        raise SystemExit(
            "Your data/raw does not match splits.csv. Re-download the dataset with:\n"
            "  python scripts/download_data.py --force"
        )
    return splits, hashes

# Modes "original" / "resplit": build a split from data/raw

def collect_images(raw_dir: Path):
    """Return (path, class_label, original_split) records, {path: md5} and the
    number of duplicates removed."""
    paths = [p for p in raw_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not paths:
        raise SystemExit(f"No images found under {raw_dir}. Run scripts/download_data.py first.")

    # Visit train first, then val, then test, so a duplicate is kept in the earliest split
    split_rank = {name: i for i, name in enumerate(SPLITS)}
    paths.sort(key=lambda p: (split_rank.get(original_split_of(p), len(SPLITS)), str(p)))

    hashes, records = {}, []
    seen_hashes = set()
    for path in tqdm(paths, desc="Scanning images"):
        digest = md5_of(path)
        if digest in seen_hashes:
            continue  # identical file already collected
        seen_hashes.add(digest)
        hashes[path] = digest
        # e.g. .../train/Broccoli/0001.jpg -> ("Broccoli", "train")
        records.append((path, path.parent.name, original_split_of(path)))

    return records, hashes, len(paths) - len(records)


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


def save_split_file(splits, hashes, raw_dir: Path, split_file: Path) -> None:
    """Write the split to splits.csv (sorted, with POSIX paths so it is identical on every OS)."""
    rows = [
        {"source": path.relative_to(raw_dir).as_posix(), "class": label,
         "split": split, "md5": hashes[path]}
        for split in SPLITS for path, label in splits[split]
    ]
    rows.sort(key=lambda r: (SPLITS.index(r["split"]), r["class"], r["source"]))
    with open(split_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SPLIT_FILE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved the split to {split_file}. Commit it so the whole group uses it.")



# Image processing

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
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--split-mode", choices=["file", "original", "resplit"], default="file",
                        help="use the shared splits.csv (default), Kaggle's split, "
                             "or a new stratified 70/15/15 split")
    parser.add_argument("--save-split-file", action="store_true",
                        help="(original/resplit modes) write the new split to --split-file")
    parser.add_argument("--size", type=int, default=64, help="output resolution (max 64)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (resplit mode only)")
    parser.add_argument("--overwrite", action="store_true", help="delete existing out-dir first")
    args = parser.parse_args()

    if args.size > 64:
        raise SystemExit("The assignment caps the resolution at 64x64.")
    if args.save_split_file and args.split_mode == "file":
        raise SystemExit("--save-split-file only applies to --split-mode original/resplit.")

    raw_dir, out_dir = args.raw_dir.resolve(), args.out_dir.resolve()
    split_file = args.split_file.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{out_dir} is not empty. Re-run with --overwrite to rebuild it.")
        shutil.rmtree(out_dir)

    # 1. Decide which image goes to which split
    if args.split_mode == "file":
        splits, hashes = load_split_file(split_file, raw_dir)
        print(f"Using the shared split from {split_file}")
    else:
        records, hashes, n_duplicates = collect_images(raw_dir)
        print(f"Found {len(records):,} unique images ({n_duplicates} duplicates removed).")
        if args.split_mode == "original":
            splits = split_original(records)
        else:
            splits = split_stratified(records, args.seed)
        print(f"Split mode: {args.split_mode}")
        if args.save_split_file:
            save_split_file(splits, hashes, raw_dir, split_file)

    classes = sorted({label for items in splits.values() for _, label in items})
    if len(classes) != 15:
        print(f"WARNING: expected 15 classes, found {len(classes)}: {classes}")

    # 2. Resize + save, and record where every image went
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

    # 3. Summary table: images per class per split + actual split percentages
    counts = {split: Counter(label for _, label in splits[split]) for split in SPLITS}
    total = sum(len(items) for items in splits.values())
    print(f"\n{'Class':<15}" + "".join(f"{s:>8}" for s in SPLITS))
    for cls in classes:
        print(f"{cls:<15}" + "".join(f"{counts[s][cls]:>8}" for s in SPLITS))
    print(f"{'TOTAL':<15}" + "".join(f"{len(splits[s]):>8}" for s in SPLITS))
    print(f"{'Share':<15}" + "".join(f"{len(splits[s]) / total:>8.1%}" for s in SPLITS))
    print(f"\nSaved {args.size}x{args.size} PNGs to {out_dir}")


if __name__ == "__main__":
    main()
