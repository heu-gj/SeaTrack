from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)  # type: ignore[attr-defined]

import motmetrics as mm


DEFAULT_RAW_MATCH_ROOT = Path(r"E:\Dataset\716\raw-match")
DEFAULT_SEQUENCE_NAMES = [
    "0001",
    "0002",
    "0003",
    "0005",
    "0009",
    "0010",
    "0011",
    "0012",
    "0014",
    "0020",
    "0028",
    "0040",
    "0043",
    "0047",
    "0051",
    "0058",
    "0081",
    "0086",
    "0087",
    "0088",
    "0089",
    "0090",
    "0091",
    "0092",
    "0096",
    "0129",
    "0131",
    "0137",
    "0143",
    "0145",
    "0146",
    "0151",
    "0157",
    "0158",
    "0166",
    "0176",
    "0177",
    "0181",
]

MM_METRICS = [
    "num_frames",
    "mota",
    "idf1",
    "idp",
    "idr",
    "precision",
    "recall",
    "num_switches",
    "num_false_positives",
    "num_misses",
]

OUTPUT_COLUMNS = ["Sequence", "HOTA", "IDF1", "MOTA", "IDs", "FP", "FN", "Precision", "Recall", "IDP", "IDR", "Frames"]


def parse_sequence_names(sequence_names: str | None) -> list[str]:
    if not sequence_names:
        return list(DEFAULT_SEQUENCE_NAMES)
    parsed: list[str] = []
    for item in sequence_names.split(","):
        token = item.strip()
        if not token:
            continue
        parsed.append(token.zfill(4) if token.isdigit() else token)
    return parsed


def discover_result_sequences(result_root: Path) -> list[str]:
    seqs = []
    for child in sorted([p for p in result_root.iterdir() if p.is_dir()]):
        result_file = child / f"{child.name}.txt"
        if result_file.exists():
            seqs.append(child.name)
    return seqs


def read_sequence_image_size(gt_root: Path, seq_name: str) -> tuple[int, int]:
    image_dir = gt_root / seq_name / "visible" / "image"
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    if not images:
        raise FileNotFoundError(f"No visible images found for sequence {seq_name} under {image_dir}")
    image = cv2.imread(str(images[0]))
    if image is None:
        raise FileNotFoundError(f"Failed to read image for sequence {seq_name}: {images[0]}")
    h, w = image.shape[:2]
    return h, w


def clip_tlwh_to_image(tlwh: np.ndarray, image_h: int, image_w: int) -> np.ndarray | None:
    x, y, w, h = map(float, tlwh[:4])
    x1 = float(np.clip(x, 0, image_w))
    y1 = float(np.clip(y, 0, image_h))
    x2 = float(np.clip(x + w, 0, image_w))
    y2 = float(np.clip(y + h, 0, image_h))
    clipped_w = x2 - x1
    clipped_h = y2 - y1
    if clipped_w <= 0 or clipped_h <= 0:
        return None
    return np.array([x1, y1, clipped_w, clipped_h], dtype=np.float64)


def load_mot_rows(txt_path: Path) -> dict[int, list[dict[str, Any]]]:
    frames: dict[int, list[dict[str, Any]]] = {}
    if not txt_path.exists():
        return frames
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        frame_id = int(float(parts[0]))
        obj_id = int(float(parts[1]))
        x = float(parts[2])
        y = float(parts[3])
        w = float(parts[4])
        h = float(parts[5])
        score = float(parts[6]) if len(parts) > 6 else 1.0
        cls = int(float(parts[7])) if len(parts) > 7 else -1
        frames.setdefault(frame_id, []).append({"id": obj_id, "tlwh": np.array([x, y, w, h], dtype=np.float64), "score": score, "cls": cls})
    return frames


def evaluate_sequence(result_file: Path, gt_file: Path, image_h: int, image_w: int, iou_thresh: float) -> mm.MOTAccumulator:
    gt_frames = load_mot_rows(gt_file)
    pred_frames = load_mot_rows(result_file)
    all_frames = sorted(set(gt_frames) | set(pred_frames))

    acc = mm.MOTAccumulator(auto_id=False)
    for frame_id in all_frames:
        gt_entries = gt_frames.get(frame_id, [])
        pred_entries = pred_frames.get(frame_id, [])

        gt_ids: list[int] = []
        gt_boxes: list[np.ndarray] = []
        for item in gt_entries:
            clipped = clip_tlwh_to_image(item["tlwh"], image_h, image_w)
            if clipped is None:
                continue
            gt_ids.append(item["id"])
            gt_boxes.append(clipped)

        pred_ids: list[int] = []
        pred_boxes: list[np.ndarray] = []
        for item in pred_entries:
            clipped = clip_tlwh_to_image(item["tlwh"], image_h, image_w)
            if clipped is None:
                continue
            pred_ids.append(item["id"])
            pred_boxes.append(clipped)

        if gt_boxes and pred_boxes:
            distances = mm.distances.iou_matrix(np.asarray(gt_boxes), np.asarray(pred_boxes), max_iou=1.0 - iou_thresh)
        else:
            distances = np.empty((len(gt_boxes), len(pred_boxes)))
        acc.update(gt_ids, pred_ids, distances, frameid=frame_id)
    return acc


def try_compute_hota_placeholder() -> float | None:
    return None


def evaluate_result_root(
    result_root: Path,
    gt_root: Path,
    sequence_names: list[str] | None = None,
    gt_file_name: str = "gt.txt",
    iou_thresh: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result_root = Path(result_root)
    gt_root = Path(gt_root)
    seqs = sequence_names or discover_result_sequences(result_root)

    accumulators: dict[str, mm.MOTAccumulator] = {}
    skipped: dict[str, str] = {}
    for seq_name in seqs:
        result_file = result_root / seq_name / f"{seq_name}.txt"
        gt_file = gt_root / seq_name / "visible" / "gt" / gt_file_name
        if not result_file.exists():
            skipped[seq_name] = "missing_result"
            continue
        if not gt_file.exists():
            skipped[seq_name] = "missing_gt"
            continue
        try:
            image_h, image_w = read_sequence_image_size(gt_root, seq_name)
            accumulators[seq_name] = evaluate_sequence(result_file, gt_file, image_h, image_w, iou_thresh)
        except Exception as exc:  # noqa: BLE001
            skipped[seq_name] = f"error: {exc}"

    mh = mm.metrics.create()
    summary = mh.compute_many(
        list(accumulators.values()),
        names=list(accumulators.keys()),
        metrics=MM_METRICS,
        generate_overall=True,
    )

    rows: list[dict[str, Any]] = []
    for seq_name in list(accumulators.keys()) + ["OVERALL"]:
        if seq_name not in summary.index:
            continue
        item = summary.loc[seq_name]
        row = {
            "Sequence": seq_name,
            "HOTA": try_compute_hota_placeholder(),
            "IDF1": float(item["idf1"] * 100.0),
            "MOTA": float(item["mota"] * 100.0),
            "IDs": int(item["num_switches"]),
            "FP": int(item["num_false_positives"]),
            "FN": int(item["num_misses"]),
            "Precision": float(item["precision"] * 100.0),
            "Recall": float(item["recall"] * 100.0),
            "IDP": float(item["idp"] * 100.0),
            "IDR": float(item["idr"] * 100.0),
            "Frames": int(item["num_frames"]),
        }
        rows.append(row)

    manifest = {
        "result_root": str(result_root),
        "gt_root": str(gt_root),
        "gt_file_name": gt_file_name,
        "iou_thresh": iou_thresh,
        "evaluated_sequences": list(accumulators.keys()),
        "skipped_sequences": skipped,
        "metrics": OUTPUT_COLUMNS,
        "note": "HOTA is left null in this script unless a dedicated HOTA evaluator is integrated later.",
    }
    return rows, manifest


def write_outputs(output_dir: Path, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "mot_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    txt_path = output_dir / "mot_summary.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        headers = OUTPUT_COLUMNS
        f.write("\t".join(headers) + "\n")
        for row in rows:
            f.write("\t".join("" if row.get(h) is None else str(row.get(h)) for h in headers) + "\n")

    json_path = output_dir / "evaluation_manifest.json"
    json_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MOT-format tracking results.")
    parser.add_argument("--result-root", required=True, help="Directory containing per-sequence MOT txt outputs.")
    parser.add_argument("--gt-root", default=str(DEFAULT_RAW_MATCH_ROOT), help="raw-match root directory.")
    parser.add_argument(
        "--sequence-names",
        default=",".join(DEFAULT_SEQUENCE_NAMES),
        help="Comma-separated sequence names to evaluate. Defaults to the current train-38 list.",
    )
    parser.add_argument("--gt-file-name", default="gt.txt", help="GT file name under visible/gt.")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold used for MOT matching.")
    parser.add_argument("--output-dir", default="", help="Optional explicit output directory. Defaults to <result-root>/evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, manifest = evaluate_result_root(
        result_root=Path(args.result_root),
        gt_root=Path(args.gt_root),
        sequence_names=parse_sequence_names(args.sequence_names),
        gt_file_name=args.gt_file_name,
        iou_thresh=args.iou_thresh,
    )
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.result_root) / "evaluation"
    write_outputs(output_dir, rows, manifest)
    print(json.dumps({"rows": rows, "manifest": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
