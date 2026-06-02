from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


VIDEO_EXTS_DEFAULT = {".mp4", ".mov", ".mkv", ".avi"}


@dataclass
class SplitCounts:
    train: int = 0
    test: int = 0
    val: int = 0


def list_class_dirs(root: Path) -> Dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(f"Root folder not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root is not a directory: {root}")
    class_dirs: Dict[str, Path] = {}
    for p in root.iterdir():
        if p.is_dir() and not p.name.startswith("."):
            class_dirs[p.name] = p
    if not class_dirs:
        raise RuntimeError(f"No class subfolders found under: {root}")
    return class_dirs


def ensure_dir(p: Path, dry_run: bool) -> None:
    if dry_run:
        return
    p.mkdir(parents=True, exist_ok=True)


def nonconflicting_path(dst_dir: Path, filename: str) -> Path:
    dst = dst_dir / filename
    if not dst.exists():
        return dst
    stem = Path(filename).stem
    suf = Path(filename).suffix
    i = 1
    while True:
        cand = dst_dir / f"{stem}__dup{i}{suf}"
        if not cand.exists():
            return cand
        i += 1


def transfer_file(src: Path, dst_dir: Path, move: bool, dry_run: bool) -> Path:
    ensure_dir(dst_dir, dry_run)
    dst = nonconflicting_path(dst_dir, src.name)
    if not dry_run:
        if move:
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))
    return dst


def split_counts(n: int) -> Tuple[int, int, int]:
    n_train = int(n * 0.6)
    n_test = int(n * 0.2)
    n_val = n - n_train - n_test
    return n_train, n_test, n_val


def main():
    ap = argparse.ArgumentParser(description="Split BIM Dataset V3 into train/test/val = 6:2:2.")
    ap.add_argument("--root", type=str, required=True, help="Input dataset root folder (class subfolders).")
    ap.add_argument("--out", type=str, required=True, help="Output folder for train/test/val structure.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for shuffling (default: 42).")
    ap.add_argument("--dry-run", action="store_true", help="Report only; do NOT copy/move files.")
    ap.add_argument("--move", action="store_true", help="Move files instead of copying.")
    ap.add_argument(
        "--ext",
        type=str,
        default="",
        help="Optional: only process one extension, e.g. '.mp4'. Leave empty for common video exts.",
    )
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()

    class_dirs = list_class_dirs(root)

    if args.ext.strip():
        ext = args.ext.lower()
        if not ext.startswith("."):
            ext = "." + ext
        exts = {ext}
    else:
        exts = VIDEO_EXTS_DEFAULT

    rng = random.Random(args.seed)

    train_root = out / "train"
    test_root = out / "test"
    val_root = out / "val"
    ensure_dir(train_root, args.dry_run)
    ensure_dir(test_root, args.dry_run)
    ensure_dir(val_root, args.dry_run)

    per_class: Dict[str, SplitCounts] = {}
    total = SplitCounts()

    for cls_name, cls_path in sorted(class_dirs.items(), key=lambda x: x[0].lower()):
        files = [p for p in cls_path.iterdir() if p.is_file() and p.suffix.lower() in exts]
        files.sort(key=lambda p: p.name.lower())
        n = len(files)

        if n == 0:
            per_class[cls_name] = SplitCounts(0, 0, 0)
            continue

        rng.shuffle(files)
        n_train, n_test, n_val = split_counts(n)

        train_files = files[:n_train]
        test_files = files[n_train:n_train + n_test]
        val_files = files[n_train + n_test:]

        # copy/move
        for f in train_files:
            transfer_file(f, train_root / cls_name, move=args.move, dry_run=args.dry_run)
        for f in test_files:
            transfer_file(f, test_root / cls_name, move=args.move, dry_run=args.dry_run)
        for f in val_files:
            transfer_file(f, val_root / cls_name, move=args.move, dry_run=args.dry_run)

        per_class[cls_name] = SplitCounts(len(train_files), len(test_files), len(val_files))
        total.train += len(train_files)
        total.test += len(test_files)
        total.val += len(val_files)

    mode = "DRY-RUN (no files changed)" if args.dry_run else ("MOVE" if args.move else "COPY")
    print("\n================ BIM Split Report (6:2:2) ================\n")
    print(f"Mode:        {mode}")
    print(f"Input root:  {root}")
    print(f"Output root: {out}")
    print(f"Extensions:  {', '.join(sorted(exts))}")
    print(f"Seed:        {args.seed}\n")

    print("--- Per-class counts ---")
    header = f"{'Class':30s} {'Train':>7s} {'Test':>7s} {'Val':>7s} {'Total':>7s}"
    print(header)
    print("-" * len(header))
    for cls_name, c in sorted(per_class.items(), key=lambda x: x[0].lower()):
        t = c.train + c.test + c.val
        print(f"{cls_name:30s} {c.train:7d} {c.test:7d} {c.val:7d} {t:7d}")

    all_total = total.train + total.test + total.val
    print("\n--- Overall totals ---")
    print(f"Train: {total.train}")
    print(f"Test:  {total.test}")
    print(f"Val:   {total.val}")
    print(f"All:   {all_total}")
    print("\nDone.\n==========================================================\n")


if __name__ == "__main__":
    main()
