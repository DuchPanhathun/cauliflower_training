from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Iterable, Sequence, Tuple


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def collect_image_records(dataset_dir: Path, extensions: Iterable[str]) -> Sequence[Tuple[str, str]]:
    """Return sorted (relative_path, label) tuples for every image file under dataset_dir."""
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    records: list[Tuple[str, str]] = []
    for class_dir in sorted(dataset_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        for image_path in sorted(class_dir.rglob("*")):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in extensions:
                continue
            relative_path = image_path.relative_to(dataset_dir)
            records.append((relative_path.as_posix(), label))
    return records


def write_csv(output_path: Path, records: Sequence[Tuple[str, str]]) -> None:
    """Write the collected image records to a CSV file."""
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_path", "label"])
        writer.writerows(records)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a CSV catalog for an image dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset"),
        help="Root directory of the dataset (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_labels.csv"),
        help="Output CSV file (default: %(default)s).",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        type=str,
        default=sorted(IMAGE_EXTENSIONS),
        help="Image file extensions to include (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    records = collect_image_records(args.dataset_dir, {ext.lower() for ext in args.extensions})

    if not records:
        print(f"No images found under {args.dataset_dir}", file=sys.stderr)
        return 1

    write_csv(args.output, records)
    unique_labels = {label for _, label in records}
    print(f"Wrote {len(records)} records for {len(unique_labels)} labels to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

