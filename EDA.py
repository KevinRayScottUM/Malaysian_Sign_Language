from __future__ import annotations

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}


# ---------------------------
# Helpers: dataset structure
# ---------------------------
def list_class_dirs(root: Path) -> Dict[str, Path]:
    """Immediate subfolders under root are treated as class folders."""
    if not root.exists():
        raise FileNotFoundError(f"Root folder not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root is not a directory: {root}")

    out: Dict[str, Path] = {}
    for p in root.iterdir():
        if p.is_dir() and not p.name.startswith("."):
            out[p.name.lower()] = p
    if not out:
        raise RuntimeError(f"No class subfolders found under: {root}")
    return out


# ---------------------------
# Helpers: filename parsing
# ---------------------------
_DUP_SUFFIX_RE = re.compile(r"\s*\(\d+\)\s*$")  # e.g. "file (1).mp4"


def parse_label_and_indices(file_path: Path) -> Tuple[Optional[str], Optional[Tuple[str, str, str]]]:
    """
    Parse label and last-3 indices from filename stem (no extension).

    Expected:
      <label>_<i1>_<i2>_<i3>.mp4

    Returns:
      (label_raw, (i1,i2,i3))  OR (None, None) if not parseable.
    """
    stem = file_path.stem

    # remove common duplicate suffix like " (1)"
    stem = _DUP_SUFFIX_RE.sub("", stem).strip()

    parts = stem.split("_")
    if len(parts) < 4:
        return None, None

    idx = parts[-3:]
    if not all(re.fullmatch(r"\d+", x) for x in idx):
        return None, None

    label_raw = "_".join(parts[:-3])
    return label_raw, (idx[0], idx[1], idx[2])


# ---------------------------
# Helpers: label normalization
# ---------------------------
_ROMAN_II_RE = re.compile(r"(?:\(|\s|_)*ii(?:\)|\s|_)*$", re.IGNORECASE)


def normalize_label(label_raw: str, class_set: set[str]) -> str:
    """
    Normalize label strings to match folder naming conventions.

    Rules implemented from your notes:
    - "hot" -> "panas"
    - "baik (II)" / "baik ii" / "baik_ii" -> "baik_2"
    - "xxx2" (no underscore) -> "xxx_2" IF xxx_2 exists as a folder
      e.g. perlahan2 -> perlahan_2
    - remove spaces / parentheses and collapse to underscores (safe default)
    """
    s = label_raw.strip()

    # unify internal whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # explicit translation mistake
    if s.lower() == "hot":
        return "panas"

    # handle roman II at end -> suffix _2
    # Examples: "baik (II)" "baik II" "baik_ii" -> "baik_2"
    if _ROMAN_II_RE.search(s):
        base = _ROMAN_II_RE.sub("", s).strip()
        base = base.replace(" ", "_").replace("(", "").replace(")", "")
        cand = f"{base.lower()}_2"
        return cand

    # generic cleanup: spaces -> underscores, remove parentheses
    s = s.replace(" ", "_")
    s = s.replace("(", "").replace(")", "")
    s = re.sub(r"__+", "_", s)  # collapse multiple underscores
    s = s.lower().strip("_")

    # trailing '2' without underscore -> convert ONLY if folder exists
    if s.endswith("2") and not s.endswith("_2"):
        base = s[:-1]
        if f"{base}_2" in class_set:
            s = f"{base}_2"

    return s


# ---------------------------
# Helpers: safe rename/move
# ---------------------------
def make_nonconflicting_path(dst_dir: Path, filename: str) -> Path:
    """
    If dst_dir/filename exists, append __dupN before extension.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    candidate = dst_dir / filename
    if not candidate.exists():
        return candidate

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    i = 1
    while True:
        cand = dst_dir / f"{stem}__dup{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1


def move_or_rename(src: Path, dst_dir: Path, new_filename: str, dry_run: bool) -> Path:
    """
    Move src into dst_dir with new_filename (handles rename+move together).
    """
    dst = make_nonconflicting_path(dst_dir, new_filename)
    if not dry_run:
        shutil.move(str(src), str(dst))
    return dst


def main():
    ap = argparse.ArgumentParser(description="Fix BIM Dataset V3 misplacements + normalize filenames.")
    ap.add_argument(
        "--root",
        type=str,
        default="BIM Dataset V3",
        help='Dataset root folder path (default: "BIM Dataset V3" relative to current dir).',
    )
    ap.add_argument("--dry-run", action="store_true", help="Report only; do NOT move/rename files.")
    ap.add_argument(
        "--ext",
        type=str,
        default="",
        help="Optional: only process one extension, e.g. '.mp4'. Leave empty for common video types.",
    )
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    class_dirs = list_class_dirs(root)
    class_set = set(class_dirs.keys())

    # exts
    if args.ext.strip():
        ext = args.ext.lower()
        if not ext.startswith("."):
            ext = "." + ext
        exts = {ext}
    else:
        exts = VIDEO_EXTS

    # reporting
    misplacements: Dict[Tuple[str, str], List[str]] = defaultdict(list)  # (src_folder, dst_folder) -> [filename...]
    renames: Dict[str, List[Tuple[str, str]]] = defaultdict(list)        # folder -> [(old,new), ...]
    unknown_format: List[str] = []

    moved_count = 0
    renamed_count = 0
    scanned_count = 0

    # iterate folders/files
    for folder_name, folder_path in sorted(class_dirs.items()):
        for f in folder_path.iterdir():
            if not f.is_file():
                continue
            if f.suffix.lower() not in exts:
                continue

            scanned_count += 1
            label_raw, idx = parse_label_and_indices(f)

            if label_raw is None or idx is None:
                unknown_format.append(str(f.relative_to(root)))
                continue

            label_norm = normalize_label(label_raw, class_set)

            # determine intended folder
            intended_folder = label_norm if label_norm in class_set else None

            # if we can infer and it's not current folder -> misplacement
            final_folder = folder_name
            if intended_folder and intended_folder != folder_name:
                final_folder = intended_folder
                misplacements[(folder_name, intended_folder)].append(f.name)

            # final filename must match the final folder name (your rule #7 too)
            new_name = f"{final_folder}_{idx[0]}_{idx[1]}_{idx[2]}{f.suffix.lower()}"

            # decide if needs rename (including hot->panas, perlahan2->perlahan_2, baik(II)->baik_2, etc.)
            will_rename = (f.name != new_name)

            # apply (move + rename in one shot)
            if (final_folder != folder_name) or will_rename:
                dst_dir = class_dirs[final_folder] if final_folder in class_dirs else (root / final_folder)
                dst = move_or_rename(f, dst_dir, new_name, dry_run=args.dry_run)

                if final_folder != folder_name:
                    moved_count += 1
                if will_rename:
                    renamed_count += 1
                    # record rename under destination folder (more useful)
                    renames[final_folder].append((f.name, dst.name))

    # ---------------------------
    # Print report
    # ---------------------------
    print("\n================ BIM Dataset V3 Fix Report ================\n")
    print(f"Root: {root}")
    print(f"Class folders detected: {len(class_dirs)}")
    print(f"Video extensions processed: {', '.join(sorted(exts))}")
    print(f"Total videos scanned: {scanned_count}\n")

    if args.dry_run:
        print("MODE: DRY-RUN (no files were changed)\n")
    else:
        print("MODE: APPLY (files moved/renamed)\n")

    print(f"Moved (misplaced) files:  {moved_count}")
    print(f"Renamed files:           {renamed_count}\n")

    if misplacements:
        print("--- Misplacements by folder ---")
        for (src_cls, dst_cls), files in sorted(
            misplacements.items(), key=lambda x: (-len(x[1]), x[0][0], x[0][1])
        ):
            print(f'  "{src_cls}" folder has misplaced files of "{dst_cls}": {len(files)} file(s)')
    else:
        print("--- Misplacements by folder ---")
        print("  None ✅")

    if misplacements:
        print("\n--- Detailed moved list ---")
        for (src_cls, dst_cls), files in sorted(misplacements.items(), key=lambda x: (x[0][0], x[0][1])):
            print(f"\n[{src_cls} -> {dst_cls}]")
            for name in sorted(files):
                print(f"  - {name}")

    if renames:
        print("\n--- Detailed rename list (grouped by destination folder) ---")
        for folder, pairs in sorted(renames.items(), key=lambda x: (-len(x[1]), x[0])):
            print(f"\n[{folder}] ({len(pairs)} rename(s))")
            for old, new in sorted(pairs, key=lambda x: x[0].lower()):
                if old != new:
                    print(f"  - {old}  ->  {new}")
    else:
        print("\n--- Detailed rename list ---")
        print("  None ✅")

    if unknown_format:
        print("\n--- Files skipped (unknown filename format; not moved/renamed) ---")
        print(f"Count: {len(unknown_format)}")
        for p in unknown_format[:80]:
            print(f"  - {p}")
        if len(unknown_format) > 80:
            print("  ... (truncated)")

    print("\nDone.\n===========================================================\n")


if __name__ == "__main__":
    main()
