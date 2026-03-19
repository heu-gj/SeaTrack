from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build small/medium/large evaluation subsets from a YOLO dataset.")
    parser.add_argument(
        "--dataset-root",
        default=r"E:\Dataset\716\ultralytics-visible-yolo26",
        help="Root directory of the exported Ultralytics dataset.",
    )
    parser.add_argument("--split", default="test", choices=("train", "val", "test"), help="Dataset split to analyze.")
    parser.add_argument(
        "--small-threshold",
        type=float,
        default=0.001,
        help="Normalized box-area threshold below which a target is considered small.",
    )
    parser.add_argument(
        "--medium-threshold",
        type=float,
        default=0.01,
        help="Normalized box-area threshold below which a target is considered medium.",
    )
    return parser.parse_args()


def classify_box(area: float, small_threshold: float, medium_threshold: float) -> str:
    if area < small_threshold:
        return "small"
    if area < medium_threshold:
        return "medium"
    return "large"


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    image_dir = dataset_root / "images" / args.split
    label_dir = dataset_root / "labels" / args.split
    subset_dir = dataset_root / "subsets"
    subset_dir.mkdir(parents=True, exist_ok=True)

    image_lists = {"small": [], "medium": [], "large": []}
    object_counts = {name: 0 for name in image_lists}
    image_counts = Counter()

    for label_path in sorted(label_dir.glob("*.txt")):
        image_path = image_dir / f"{label_path.stem}.jpg"
        if not image_path.exists():
            continue

        categories_in_image: set[str] = set()
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw_line.strip().split()
            if len(parts) != 5:
                continue
            _cls, _xc, _yc, w, h = parts
            area = float(w) * float(h)
            scale = classify_box(area, args.small_threshold, args.medium_threshold)
            categories_in_image.add(scale)
            object_counts[scale] += 1

        for scale in categories_in_image:
            image_lists[scale].append(str(image_path.resolve()))
            image_counts[scale] += 1

    dataset_yaml = (dataset_root / "dataset.yaml").read_text(encoding="utf-8")
    names_block = dataset_yaml.split("names:", 1)[1].strip() if "names:" in dataset_yaml else ""

    for scale, image_paths in image_lists.items():
        txt_path = subset_dir / f"{args.split}_{scale}.txt"
        yaml_path = subset_dir / f"dataset_{args.split}_{scale}.yaml"
        txt_path.write_text("\n".join(image_paths) + ("\n" if image_paths else ""), encoding="utf-8")

        yaml_lines = [
            f"path: {dataset_root.as_posix()}",
            "train: images/train",
            "val: " + txt_path.as_posix(),
            "test: " + txt_path.as_posix(),
            "",
            "names:",
        ]
        if names_block:
            yaml_lines.extend(names_block.splitlines())
        yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    report = {
        "dataset_root": str(dataset_root),
        "split": args.split,
        "small_threshold": args.small_threshold,
        "medium_threshold": args.medium_threshold,
        "image_counts": dict(image_counts),
        "object_counts": object_counts,
    }
    (subset_dir / f"{args.split}_size_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
