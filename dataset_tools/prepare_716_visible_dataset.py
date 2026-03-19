from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CLASS_NAMES = {
    0: "cargo",
    1: "passenger_ship",
    2: "speedboat",
    3: "buoy",
    4: "floater",
    5: "others",
}


@dataclass
class SequenceStats:
    name: str
    image_dir: Path
    gt_file: Path
    image_frames: set[int]
    annotated_frames: set[int]
    class_counts: Counter
    image_size: tuple[int, int]

    @property
    def valid_frames(self) -> list[int]:
        return sorted(self.image_frames & self.annotated_frames)

    @property
    def num_valid_frames(self) -> int:
        return len(self.image_frames & self.annotated_frames)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare visible-light detection dataset for Ultralytics/YOLO26.")
    parser.add_argument("--src", default=r"E:\Dataset\716\raw-match", help="Source raw-match root.")
    parser.add_argument(
        "--dst",
        default=r"E:\Dataset\716\ultralytics-visible-yolo26",
        help="Output dataset root.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Sequence split ratio for training.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Sequence split ratio for validation.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Sequence split ratio for test.")
    parser.add_argument("--seed", type=int, default=716, help="Random seed used for sequence split.")
    parser.add_argument(
        "--include-empty-labels",
        action="store_true",
        help="Export valid image frames even if a frame has zero boxes after parsing.",
    )
    return parser.parse_args()


def read_image_size(image_path: Path) -> tuple[int, int]:
    suffix = image_path.suffix.lower()
    if suffix not in {".jpg", ".jpeg"}:
        raise ValueError(f"Unsupported image suffix: {image_path}")

    with image_path.open("rb") as f:
        data = f.read(24)
        if len(data) < 24 or data[0:2] != b"\xff\xd8":
            raise ValueError(f"Invalid JPEG header: {image_path}")

        f.seek(2)
        while True:
            marker_prefix = f.read(1)
            if not marker_prefix:
                break
            if marker_prefix != b"\xff":
                continue
            marker = f.read(1)
            while marker == b"\xff":
                marker = f.read(1)
            if marker in {b"\xc0", b"\xc1", b"\xc2", b"\xc3", b"\xc5", b"\xc6", b"\xc7", b"\xc9", b"\xca", b"\xcb", b"\xcd", b"\xce", b"\xcf"}:
                segment_length = int.from_bytes(f.read(2), "big")
                precision = f.read(1)
                height = int.from_bytes(f.read(2), "big")
                width = int.from_bytes(f.read(2), "big")
                _ = precision
                return width, height
            if marker in {b"\xd8", b"\xd9"}:
                continue
            segment_length = int.from_bytes(f.read(2), "big")
            if segment_length < 2:
                raise ValueError(f"Invalid JPEG segment length in {image_path}")
            f.seek(segment_length - 2, 1)

    raise ValueError(f"Could not parse JPEG size: {image_path}")


def parse_gt_file(gt_file: Path) -> tuple[dict[int, list[tuple[int, float, float, float, float]]], Counter]:
    frame_to_labels: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    class_counts: Counter = Counter()

    with gt_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 8:
                raise ValueError(f"{gt_file}: line {line_no} has fewer than 8 columns.")

            frame = int(float(parts[0]))
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            cls = int(float(parts[7]))

            if cls not in CLASS_NAMES:
                raise ValueError(f"{gt_file}: line {line_no} has unknown class id {cls}.")
            if w <= 0 or h <= 0:
                continue

            frame_to_labels[frame].append((cls, x, y, w, h))
            class_counts[cls] += 1

    return frame_to_labels, class_counts


def collect_sequences(src_root: Path) -> tuple[list[SequenceStats], list[dict[str, str]]]:
    usable_sequences: list[SequenceStats] = []
    skipped_sequences: list[dict[str, str]] = []

    for seq_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        visible_dir = seq_dir / "visible"
        image_dir = visible_dir / "image"
        gt_file = visible_dir / "gt" / "gt_id.txt"

        if not image_dir.exists():
            skipped_sequences.append({"sequence": seq_dir.name, "reason": "missing_visible_image_dir"})
            continue
        if not gt_file.exists():
            skipped_sequences.append({"sequence": seq_dir.name, "reason": "missing_visible_gt"})
            continue

        image_paths = sorted(image_dir.glob("*.jpg"))
        if not image_paths:
            skipped_sequences.append({"sequence": seq_dir.name, "reason": "no_visible_images"})
            continue

        try:
            frame_to_labels, class_counts = parse_gt_file(gt_file)
        except Exception as exc:  # pragma: no cover - defensive path
            skipped_sequences.append({"sequence": seq_dir.name, "reason": f"gt_parse_error: {exc}"})
            continue

        if not frame_to_labels:
            skipped_sequences.append({"sequence": seq_dir.name, "reason": "empty_visible_gt"})
            continue

        image_frames = {int(p.stem) for p in image_paths}
        annotated_frames = set(frame_to_labels)
        valid_frames = image_frames & annotated_frames
        if not valid_frames:
            skipped_sequences.append({"sequence": seq_dir.name, "reason": "no_intersection_between_images_and_gt"})
            continue

        try:
            image_size = read_image_size(image_paths[0])
        except Exception as exc:  # pragma: no cover - defensive path
            skipped_sequences.append({"sequence": seq_dir.name, "reason": f"image_size_error: {exc}"})
            continue

        usable_sequences.append(
            SequenceStats(
                name=seq_dir.name,
                image_dir=image_dir,
                gt_file=gt_file,
                image_frames=image_frames,
                annotated_frames=annotated_frames,
                class_counts=class_counts,
                image_size=image_size,
            )
        )

    return usable_sequences, skipped_sequences


def allocate_split_counts(num_sequences: int, train_ratio: float, val_ratio: float, test_ratio: float) -> dict[str, int]:
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("Split ratios must sum to 1.0")

    raw_counts = {
        "train": num_sequences * train_ratio,
        "val": num_sequences * val_ratio,
        "test": num_sequences * test_ratio,
    }
    counts = {k: int(math.floor(v)) for k, v in raw_counts.items()}
    remainder = num_sequences - sum(counts.values())
    for split, _ in sorted(raw_counts.items(), key=lambda x: x[1] - math.floor(x[1]), reverse=True):
        if remainder <= 0:
            break
        counts[split] += 1
        remainder -= 1
    return counts


def split_sequences(
    sequences: list[SequenceStats],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[SequenceStats]]:
    split_counts = allocate_split_counts(len(sequences), train_ratio, val_ratio, test_ratio)
    rng = random.Random(seed)

    shuffled = sequences[:]
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda s: (s.num_valid_frames, sum(s.class_counts.values())), reverse=True)

    split_map: dict[str, list[SequenceStats]] = {"train": [], "val": [], "test": []}
    current_load: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for seq in shuffled:
        candidates = [split for split, count in split_counts.items() if len(split_map[split]) < count]
        if not candidates:
            raise RuntimeError("No split candidate left while assigning sequences.")
        chosen = min(candidates, key=lambda split: (current_load[split], len(split_map[split])))
        split_map[chosen].append(seq)
        current_load[chosen] += seq.num_valid_frames

    return split_map


def ensure_structure(dst_root: Path) -> None:
    for split in ("train", "val", "test"):
        (dst_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def format_yolo_label(cls: int, x: float, y: float, w: float, h: float, image_size: tuple[int, int]) -> str | None:
    image_w, image_h = image_size
    if image_w <= 0 or image_h <= 0:
        return None

    xc = (x + w / 2.0) / image_w
    yc = (y + h / 2.0) / image_h
    wn = w / image_w
    hn = h / image_h

    values = [xc, yc, wn, hn]
    if any(v <= 0 for v in (wn, hn)):
        return None
    if xc <= 0 or yc <= 0 or xc >= 1 or yc >= 1:
        return None
    if wn > 1 or hn > 1:
        return None

    return f"{cls} " + " ".join(f"{value:.6f}" for value in values)


def export_split(
    split_name: str,
    sequences: Iterable[SequenceStats],
    dst_root: Path,
    include_empty_labels: bool,
) -> dict[str, int]:
    image_out_dir = dst_root / "images" / split_name
    label_out_dir = dst_root / "labels" / split_name

    exported_images = 0
    exported_labels = 0
    exported_objects = 0

    for seq in sequences:
        frame_to_labels, _ = parse_gt_file(seq.gt_file)
        for frame in seq.valid_frames:
            image_name = f"{frame:06d}.jpg"
            src_image = seq.image_dir / image_name
            if not src_image.exists():
                continue

            stem = f"{seq.name}_{frame:06d}"
            dst_image = image_out_dir / f"{stem}.jpg"
            dst_label = label_out_dir / f"{stem}.txt"

            labels = []
            for cls, x, y, w, h in frame_to_labels.get(frame, []):
                line = format_yolo_label(cls, x, y, w, h, seq.image_size)
                if line is not None:
                    labels.append(line)

            if not labels and not include_empty_labels:
                continue

            dst_image.write_bytes(src_image.read_bytes())
            dst_label.write_text("\n".join(labels), encoding="utf-8")

            exported_images += 1
            exported_labels += 1
            exported_objects += len(labels)

    return {
        "images": exported_images,
        "labels": exported_labels,
        "objects": exported_objects,
    }


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


def write_report(
    dst_root: Path,
    split_map: dict[str, list[SequenceStats]],
    skipped_sequences: list[dict[str, str]],
    export_stats: dict[str, dict[str, int]],
) -> None:
    manifest = {
        "used_sequences": {
            split: [seq.name for seq in seqs] for split, seqs in split_map.items()
        },
        "skipped_sequences": skipped_sequences,
        "export_stats": export_stats,
    }
    (dst_root / "split_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Dataset Summary",
        "",
        "This dataset was exported from `E:\\Dataset\\716\\raw-match` using only visible-light images and visible GT.",
        "",
        "## Export Rules",
        "",
        "- Only sequences with `visible/image` and `visible/gt/gt_id.txt` were used.",
        "- Only frames present in both visible images and visible GT were exported.",
        "- Output format is Ultralytics detection format.",
        "",
        "## Split Summary",
        "",
    ]
    for split in ("train", "val", "test"):
        seqs = split_map[split]
        stats = export_stats[split]
        lines.extend(
            [
                f"### {split}",
                "",
                f"- Sequences: {len(seqs)}",
                f"- Images: {stats['images']}",
                f"- Label files: {stats['labels']}",
                f"- Objects: {stats['objects']}",
                "",
                "Used sequences:",
                "",
            ]
        )
        if seqs:
            lines.extend([f"- `{seq.name}`" for seq in seqs])
        else:
            lines.append("- none")
        lines.append("")

    lines.extend(
        [
            "## Skipped Sequences",
            "",
        ]
    )
    if skipped_sequences:
        for item in skipped_sequences:
            lines.append(f"- `{item['sequence']}`: {item['reason']}")
    else:
        lines.append("- none")
    lines.append("")

    (dst_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if not src_root.exists():
        raise FileNotFoundError(f"Source dataset root does not exist: {src_root}")

    sequences, skipped_sequences = collect_sequences(src_root)
    if not sequences:
        raise RuntimeError("No usable visible-light sequences with GT were found.")

    split_map = split_sequences(
        sequences,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    ensure_structure(dst_root)
    export_stats = {
        split: export_split(split, seqs, dst_root, args.include_empty_labels)
        for split, seqs in split_map.items()
    }
    write_dataset_yaml(dst_root)
    write_report(dst_root, split_map, skipped_sequences, export_stats)

    summary = {
        "usable_sequences": len(sequences),
        "skipped_sequences": len(skipped_sequences),
        "splits": {split: [seq.name for seq in seqs] for split, seqs in split_map.items()},
        "export_stats": export_stats,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
