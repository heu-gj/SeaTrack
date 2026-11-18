from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


INVALID_COST = 1e6


def parse_gt_file(
    path: Path,
    mark_column: int = 7,
    class_column: int = 8,
) -> dict[int, list[dict[str, Any]]]:
    """Read MOT-style GT rows: frame,id,x,y,w,h[,mark,class,...]."""
    frames: dict[int, list[dict[str, Any]]] = {}
    if not path.is_file():
        raise FileNotFoundError(f"GT file not found: {path}")

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            frame_id = int(float(parts[0]))
            object_id = int(float(parts[1]))
            x, y, width, height = map(float, parts[2:6])
        except (ValueError, OverflowError):
            continue
        if width <= 0 or height <= 0 or not np.isfinite([x, y, width, height]).all():
            continue

        if mark_column > 0 and len(parts) >= mark_column:
            try:
                if float(parts[mark_column - 1]) <= 0:
                    continue
            except ValueError:
                continue

        class_id = -1
        if class_column > 0 and len(parts) >= class_column:
            try:
                class_id = int(float(parts[class_column - 1]))
            except ValueError:
                class_id = -1

        frames.setdefault(frame_id, []).append(
            {
                "id": object_id,
                "xyxy": np.asarray([x, y, x + width, y + height], dtype=np.float64),
                "class_id": class_id,
            }
        )
    return frames


def frame_id_from_name(image_name: str, fallback: int, mode: str, offset: int) -> int:
    if mode == "ordinal":
        return fallback + offset
    match = re.search(r"(\d+)$", Path(image_name).stem)
    if match is None:
        return fallback + offset
    return int(match.group(1)) + offset


def pairwise_detection_gt_cost(
    detections: np.ndarray,
    ground_truth: list[dict[str, Any]],
    max_center_distance: float,
    max_log_size_ratio: float,
    match_classes: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a permissive one-to-one matching cost, including some zero-IoU pairs."""
    det = np.asarray(detections, dtype=np.float64).reshape(-1, 6)
    n_gt, n_det = len(ground_truth), len(det)
    costs = np.full((n_gt, n_det), INVALID_COST, dtype=np.float64)
    ious = np.zeros((n_gt, n_det), dtype=np.float64)
    if n_gt == 0 or n_det == 0:
        return costs, ious

    det_boxes = det[:, :4]
    det_widths = det_boxes[:, 2] - det_boxes[:, 0]
    det_heights = det_boxes[:, 3] - det_boxes[:, 1]
    valid_det = (det_widths > 0) & (det_heights > 0)
    for gt_index, gt in enumerate(ground_truth):
        gx1, gy1, gx2, gy2 = gt["xyxy"]
        gw, gh = gx2 - gx1, gy2 - gy1
        gcx, gcy = 0.5 * (gx1 + gx2), 0.5 * (gy1 + gy2)
        gt_class = int(gt["class_id"])

        for det_index in range(n_det):
            if not valid_det[det_index]:
                continue
            if match_classes and gt_class >= 0 and int(det[det_index, 5]) != gt_class:
                continue

            dx1, dy1, dx2, dy2 = det_boxes[det_index]
            dw, dh = det_widths[det_index], det_heights[det_index]
            dcx, dcy = 0.5 * (dx1 + dx2), 0.5 * (dy1 + dy2)
            center_distance = float(
                np.hypot((dcx - gcx) / max(gw, 1e-6), (dcy - gcy) / max(gh, 1e-6))
            )
            log_width = abs(float(np.log(max(dw, 1e-6) / max(gw, 1e-6))))
            log_height = abs(float(np.log(max(dh, 1e-6) / max(gh, 1e-6))))
            if center_distance > max_center_distance:
                continue
            if max(log_width, log_height) > max_log_size_ratio:
                continue

            inter_w = max(0.0, min(gx2, dx2) - max(gx1, dx1))
            inter_h = max(0.0, min(gy2, dy2) - max(gy1, dy1))
            intersection = inter_w * inter_h
            union = gw * gh + dw * dh - intersection
            iou = intersection / union if union > 0 else 0.0
            ious[gt_index, det_index] = iou

            costs[gt_index, det_index] = (
                center_distance
                + 0.25 * (log_width + log_height)
                + 0.10 * (1.0 - iou)
            )
    return costs, ious


def collect_sequence(
    sequence_name: str,
    cache_path: Path,
    gt_path: Path,
    frame_id_mode: str,
    frame_id_offset: int,
    gt_mark_column: int,
    gt_class_column: int,
    match_classes: bool,
    max_center_distance: float,
    max_log_size_ratio: float,
    max_match_cost: float,
) -> tuple[dict[str, list[Any]], dict[str, int]]:
    with cache_path.open("rb") as stream:
        cache = pickle.load(stream)
    if "frames" not in cache:
        raise ValueError(f"invalid detection cache (missing frames): {cache_path}")

    gt_frames = parse_gt_file(gt_path, gt_mark_column, gt_class_column)
    output: dict[str, list[Any]] = {
        "residuals": [],
        "scales": [],
        "scores": [],
        "sequence_ids": [],
        "frame_ids": [],
        "gt_ids": [],
        "detection_indices": [],
        "ious": [],
    }
    counts = {"frames": 0, "detections": 0, "gt": 0, "matches": 0, "zero_iou_matches": 0}

    for ordinal, frame in enumerate(cache["frames"], start=1):
        counts["frames"] += 1
        frame_id = frame_id_from_name(
            frame.get("image_name", ""),
            ordinal,
            frame_id_mode,
            frame_id_offset,
        )
        height, width = map(int, frame["shape"])
        detections = np.asarray(frame.get("boxes", []), dtype=np.float64).reshape(-1, 6)
        gt_entries = gt_frames.get(frame_id, [])
        counts["detections"] += len(detections)
        counts["gt"] += len(gt_entries)
        if len(detections) == 0 or len(gt_entries) == 0:
            continue

        costs, ious = pairwise_detection_gt_cost(
            detections,
            gt_entries,
            max_center_distance=max_center_distance,
            max_log_size_ratio=max_log_size_ratio,
            match_classes=match_classes,
        )
        rows, columns = linear_sum_assignment(costs)
        for gt_index, det_index in zip(rows, columns):
            match_cost = float(costs[gt_index, det_index])
            if not np.isfinite(match_cost) or match_cost >= INVALID_COST or match_cost > max_match_cost:
                continue

            gt_box = gt_entries[gt_index]["xyxy"]
            det_box = detections[det_index, :4]
            gt_width = float(gt_box[2] - gt_box[0])
            gt_height = float(gt_box[3] - gt_box[1])
            det_width = float(det_box[2] - det_box[0])
            det_height = float(det_box[3] - det_box[1])
            if min(gt_width, gt_height, det_width, det_height) <= 0:
                continue

            gt_cx = 0.5 * float(gt_box[0] + gt_box[2])
            gt_cy = 0.5 * float(gt_box[1] + gt_box[3])
            det_cx = 0.5 * float(det_box[0] + det_box[2])
            det_cy = 0.5 * float(det_box[1] + det_box[3])
            residual = [
                (det_cx - gt_cx) / gt_width,
                (det_cy - gt_cy) / gt_height,
                np.log(det_width / gt_width),
                np.log(det_height / gt_height),
            ]
            scale = np.sqrt((det_width * det_height) / max(float(width * height), 1.0))
            score = float(np.clip(detections[det_index, 4], 0.0, 1.0))
            pair_iou = float(ious[gt_index, det_index])

            output["residuals"].append(residual)
            output["scales"].append(scale)
            output["scores"].append(score)
            output["sequence_ids"].append(sequence_name)
            output["frame_ids"].append(frame_id)
            output["gt_ids"].append(int(gt_entries[gt_index]["id"]))
            output["detection_indices"].append(int(det_index))
            output["ious"].append(pair_iou)
            counts["matches"] += 1
            counts["zero_iou_matches"] += int(pair_iou <= 0.0)

    return output, counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect detector-to-GT localization residuals from SeaTrack detection caches."
    )
    parser.add_argument("--raw-match-root", required=True)
    parser.add_argument("--cache-dir", required=True, help="Directory containing one SEQUENCE.pkl cache per sequence.")
    parser.add_argument("--sequence-names", required=True, help="Comma-separated non-test sequence names only.")
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--modality-dir", default="visible", help="Dataset modality folder, e.g. visible or infrared.")
    parser.add_argument("--gt-file-name", default="gt.txt")
    parser.add_argument("--frame-id-mode", choices=("filename", "ordinal"), default="filename")
    parser.add_argument("--frame-id-offset", type=int, default=0)
    parser.add_argument("--gt-mark-column", type=int, default=7, help="1-based MOT mark/validity column; set 0 to disable.")
    parser.add_argument("--gt-class-column", type=int, default=8, help="1-based class column; set 0 if absent.")
    parser.add_argument("--match-classes", action="store_true", help="Require detector class IDs to equal GT class IDs.")
    parser.add_argument("--max-center-distance", type=float, default=2.0)
    parser.add_argument("--max-log-size-ratio", type=float, default=2.0)
    parser.add_argument("--max-match-cost", type=float, default=2.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_match_root)
    cache_dir = Path(args.cache_dir)
    sequence_names = [item.strip() for item in args.sequence_names.split(",") if item.strip()]
    if not sequence_names:
        raise ValueError("no sequence names were supplied")

    merged: dict[str, list[Any]] = {
        "residuals": [],
        "scales": [],
        "scores": [],
        "sequence_ids": [],
        "frame_ids": [],
        "gt_ids": [],
        "detection_indices": [],
        "ious": [],
    }
    sequence_reports: dict[str, Any] = {}
    for sequence_name in sequence_names:
        cache_path = cache_dir / f"{sequence_name}.pkl"
        gt_path = raw_root / sequence_name / args.modality_dir / "gt" / args.gt_file_name
        sequence_data, counts = collect_sequence(
            sequence_name=sequence_name,
            cache_path=cache_path,
            gt_path=gt_path,
            frame_id_mode=args.frame_id_mode,
            frame_id_offset=args.frame_id_offset,
            gt_mark_column=args.gt_mark_column,
            gt_class_column=args.gt_class_column,
            match_classes=args.match_classes,
            max_center_distance=args.max_center_distance,
            max_log_size_ratio=args.max_log_size_ratio,
            max_match_cost=args.max_match_cost,
        )
        for key in merged:
            merged[key].extend(sequence_data[key])
        sequence_reports[sequence_name] = counts

    if not merged["residuals"]:
        raise RuntimeError(
            "no detection/GT matches were collected; check cache/GT paths, frame indexing, "
            "GT mark column and matching gates"
        )

    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        residuals=np.asarray(merged["residuals"], dtype=np.float64).reshape(-1, 4),
        scales=np.asarray(merged["scales"], dtype=np.float64),
        scores=np.asarray(merged["scores"], dtype=np.float64),
        sequence_ids=np.asarray(merged["sequence_ids"], dtype="U"),
        frame_ids=np.asarray(merged["frame_ids"], dtype=np.int64),
        gt_ids=np.asarray(merged["gt_ids"], dtype=np.int64),
        detection_indices=np.asarray(merged["detection_indices"], dtype=np.int64),
        ious=np.asarray(merged["ious"], dtype=np.float64),
    )

    report = {
        "output_npz": str(output_path),
        "modality_dir": args.modality_dir,
        "sequence_names": sequence_names,
        "total_matches": len(merged["residuals"]),
        "zero_iou_matches": int(np.sum(np.asarray(merged["ious"]) <= 0.0)),
        "zero_iou_match_ratio": float(np.mean(np.asarray(merged["ious"]) <= 0.0)),
        "matching": {
            "frame_id_mode": args.frame_id_mode,
            "frame_id_offset": args.frame_id_offset,
            "max_center_distance": args.max_center_distance,
            "max_log_size_ratio": args.max_log_size_ratio,
            "max_match_cost": args.max_match_cost,
            "match_classes": args.match_classes,
            "gt_mark_column": args.gt_mark_column,
        },
        "sequences": sequence_reports,
        "warning": (
            "Use only train/validation sequences for fitting and coverage calibration. "
            "Do not include benchmark test sequences. Review zero-IoU matches and "
            "manually inspect a sample of matched detection/GT pairs before fitting."
        ),
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
