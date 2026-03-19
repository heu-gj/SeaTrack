from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2


CLASS_NAMES = {
    0: "cargo",
    1: "passenger_ship",
    2: "speedboat",
    3: "buoy",
    4: "floater",
    5: "others",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a horizontal-slice YOLO dataset from the visible 716 export.")
    parser.add_argument(
        "--src",
        default=r"E:\Dataset\716\ultralytics-visible-yolo26",
        help="Source dataset root in Ultralytics detection format.",
    )
    parser.add_argument(
        "--dst",
        default=r"E:\Dataset\716\ultralytics-visible-yolo26-slice2",
        help="Output sliced dataset root.",
    )
    parser.add_argument("--num-slices", type=int, default=2, help="Number of horizontal slices per image.")
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=0.15,
        help="Horizontal overlap ratio between neighboring slices, relative to full image width.",
    )
    parser.add_argument(
        "--min-retain-ratio",
        type=float,
        default=0.3,
        help="Keep a clipped box only if the retained area / original area is at least this value.",
    )
    parser.add_argument(
        "--min-box-size",
        type=float,
        default=2.0,
        help="Minimum clipped box width and height in pixels.",
    )
    parser.add_argument(
        "--include-empty-labels",
        action="store_true",
        help="Export slice images even if no labels remain after clipping.",
    )
    return parser.parse_args()


def ensure_structure(dst_root: Path) -> None:
    for split in ("train", "val", "test"):
        (dst_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def load_labels(label_path: Path, image_w: int, image_h: int) -> list[tuple[int, float, float, float, float]]:
    labels: list[tuple[int, float, float, float, float]] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        xc = float(parts[1]) * image_w
        yc = float(parts[2]) * image_h
        bw = float(parts[3]) * image_w
        bh = float(parts[4]) * image_h
        x1 = xc - bw / 2.0
        y1 = yc - bh / 2.0
        x2 = xc + bw / 2.0
        y2 = yc + bh / 2.0
        labels.append((cls, x1, y1, x2, y2))
    return labels


def build_slice_windows(image_w: int, num_slices: int, overlap_ratio: float) -> list[tuple[int, int]]:
    if num_slices < 1:
        raise ValueError("num_slices must be >= 1")
    if num_slices == 1:
        return [(0, image_w)]

    overlap_px = int(round(image_w * overlap_ratio))
    slice_w = int(math.ceil((image_w + overlap_px * (num_slices - 1)) / num_slices))
    stride = max(1, slice_w - overlap_px)

    windows: list[tuple[int, int]] = []
    start = 0
    for slice_idx in range(num_slices):
        if slice_idx == num_slices - 1:
            end = image_w
            start = max(0, end - slice_w)
        else:
            end = min(image_w, start + slice_w)
        windows.append((start, end))
        start += stride

    windows[0] = (0, min(image_w, windows[0][1]))
    windows[-1] = (max(0, image_w - (windows[-1][1] - windows[-1][0])), image_w)
    return windows


def clip_box_to_window(
    box: tuple[int, float, float, float, float],
    x_start: int,
    x_end: int,
    image_h: int,
    min_retain_ratio: float,
    min_box_size: float,
) -> tuple[int, float, float, float, float] | None:
    cls, x1, y1, x2, y2 = box
    inter_x1 = max(x1, x_start)
    inter_x2 = min(x2, x_end)
    inter_y1 = max(y1, 0.0)
    inter_y2 = min(y2, float(image_h))

    inter_w = inter_x2 - inter_x1
    inter_h = inter_y2 - inter_y1
    if inter_w < min_box_size or inter_h < min_box_size:
        return None

    orig_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    inter_area = max(0.0, inter_w) * max(0.0, inter_h)
    if orig_area <= 0.0:
        return None
    if inter_area / orig_area < min_retain_ratio:
        return None

    return cls, inter_x1 - x_start, inter_y1, inter_x2 - x_start, inter_y2


def to_yolo_line(cls: int, x1: float, y1: float, x2: float, y2: float, crop_w: int, crop_h: int) -> str | None:
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None

    xc = (x1 + x2) / 2.0 / crop_w
    yc = (y1 + y2) / 2.0 / crop_h
    wn = bw / crop_w
    hn = bh / crop_h

    if xc <= 0 or yc <= 0 or xc >= 1 or yc >= 1:
        return None
    if wn <= 0 or hn <= 0 or wn > 1 or hn > 1:
        return None
    return f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"


def write_dataset_yaml(dst_root: Path) -> None:
    lines = [
        f"path: {dst_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    for cls_id, cls_name in CLASS_NAMES.items():
        lines.append(f"  {cls_id}: {cls_name}")
    (dst_root / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_sequences_from_split(image_paths: list[Path]) -> list[str]:
    seqs = sorted({path.stem.split("_", 1)[0] for path in image_paths if "_" in path.stem})
    return seqs


def main() -> None:
    args = parse_args()
    src_root = Path(args.src)
    dst_root = Path(args.dst)

    ensure_structure(dst_root)

    export_stats: dict[str, dict[str, int]] = {}
    source_sequences: dict[str, list[str]] = {}

    for split in ("train", "val", "test"):
        src_image_dir = src_root / "images" / split
        src_label_dir = src_root / "labels" / split
        dst_image_dir = dst_root / "images" / split
        dst_label_dir = dst_root / "labels" / split

        images = sorted(src_image_dir.glob("*.jpg"))
        source_sequences[split] = collect_sequences_from_split(images)

        split_images = 0
        split_labels = 0
        split_objects = 0
        images_without_labels = 0

        for image_path in images:
            label_path = src_label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                continue
            image_h, image_w = image.shape[:2]
            boxes = load_labels(label_path, image_w, image_h)
            windows = build_slice_windows(image_w, args.num_slices, args.overlap_ratio)

            for slice_idx, (x_start, x_end) in enumerate(windows):
                crop = image[:, x_start:x_end]
                crop_h, crop_w = crop.shape[:2]
                yolo_lines: list[str] = []

                for box in boxes:
                    clipped = clip_box_to_window(
                        box,
                        x_start=x_start,
                        x_end=x_end,
                        image_h=image_h,
                        min_retain_ratio=args.min_retain_ratio,
                        min_box_size=args.min_box_size,
                    )
                    if clipped is None:
                        continue
                    cls, x1, y1, x2, y2 = clipped
                    line = to_yolo_line(cls, x1, y1, x2, y2, crop_w, crop_h)
                    if line is not None:
                        yolo_lines.append(line)

                if not yolo_lines and not args.include_empty_labels:
                    images_without_labels += 1
                    continue

                stem = f"{image_path.stem}_s{slice_idx}"
                out_image = dst_image_dir / f"{stem}.jpg"
                out_label = dst_label_dir / f"{stem}.txt"
                cv2.imwrite(str(out_image), crop)
                out_label.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

                split_images += 1
                split_labels += 1
                split_objects += len(yolo_lines)

        export_stats[split] = {
            "images": split_images,
            "labels": split_labels,
            "objects": split_objects,
            "skipped_empty_slices": images_without_labels,
        }

    write_dataset_yaml(dst_root)

    manifest = {
        "source_dataset": str(src_root),
        "num_slices": args.num_slices,
        "overlap_ratio": args.overlap_ratio,
        "min_retain_ratio": args.min_retain_ratio,
        "min_box_size": args.min_box_size,
        "include_empty_labels": args.include_empty_labels,
        "source_sequences": source_sequences,
        "export_stats": export_stats,
    }
    (dst_root / "slice_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    readme_lines = [
        "# Sliced Dataset Summary",
        "",
        f"Source dataset: `{src_root}`",
        f"Output dataset: `{dst_root}`",
        "",
        "## Slice Rules",
        "",
        f"- Horizontal slices per image: {args.num_slices}",
        f"- Horizontal overlap ratio: {args.overlap_ratio}",
        f"- Minimum retained-area ratio for clipped boxes: {args.min_retain_ratio}",
        f"- Minimum clipped box size in pixels: {args.min_box_size}",
        f"- Export empty slice labels: {args.include_empty_labels}",
        "",
        "## Split Summary",
        "",
    ]

    for split in ("train", "val", "test"):
        stats = export_stats[split]
        seqs = source_sequences[split]
        readme_lines.extend(
            [
                f"### {split}",
                "",
                f"- Source sequences: {len(seqs)}",
                f"- Exported slice images: {stats['images']}",
                f"- Exported label files: {stats['labels']}",
                f"- Exported objects: {stats['objects']}",
                f"- Skipped empty slices: {stats['skipped_empty_slices']}",
                "",
                "Source sequences:",
                "",
            ]
        )
        readme_lines.extend([f"- `{seq}`" for seq in seqs] if seqs else ["- none"])
        readme_lines.append("")

    (dst_root / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
