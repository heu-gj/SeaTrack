from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detector testing with two horizontal slices and merged results.")
    parser.add_argument("--model", required=True, help="Path to detector weights.")
    parser.add_argument("--source", required=True, help="Directory containing original full images.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size per slice.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=0.15,
        help="Overlap ratio between the two horizontal slices, relative to image width.",
    )
    parser.add_argument(
        "--out-dir",
        default=r"E:\Code\PRCV\ultralytics\experiments_tool\runs\test_two_slice",
        help="Output directory for merged visualizations and txt predictions.",
    )
    parser.add_argument("--save-vis", action="store_true", help="Save visualization images.")
    return parser.parse_args()


def compute_two_windows(width: int, overlap_ratio: float) -> list[tuple[int, int]]:
    overlap_px = int(round(width * overlap_ratio))
    half = int(np.ceil((width + overlap_px) / 2.0))
    left = (0, min(width, half))
    right = (max(0, width - half), width)
    return [left, right]


def box_iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_xyxy(detections: list[tuple[int, float, np.ndarray]], iou_thresh: float) -> list[tuple[int, float, np.ndarray]]:
    detections = sorted(detections, key=lambda x: x[1], reverse=True)
    kept: list[tuple[int, float, np.ndarray]] = []
    while detections:
        best = detections.pop(0)
        kept.append(best)
        remaining = []
        for det in detections:
            same_cls = det[0] == best[0]
            overlap = box_iou_xyxy(det[2], best[2]) > iou_thresh
            if same_cls and overlap:
                continue
            remaining.append(det)
        detections = remaining
    return kept


def draw_detections(image: np.ndarray, detections: list[tuple[int, float, np.ndarray]]) -> np.ndarray:
    canvas = image.copy()
    for cls_id, score, box in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            canvas,
            f"{cls_id}:{score:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source)
    out_dir = Path(args.out_dir)
    txt_dir = out_dir / "labels"
    vis_dir = out_dir / "vis"
    txt_dir.mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    images = sorted([p for p in source_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])

    summary = {"source": str(source_dir), "images": len(images), "detections_before_merge": 0, "detections_after_merge": 0}

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        windows = compute_two_windows(width, args.overlap_ratio)

        merged_detections: list[tuple[int, float, np.ndarray]] = []
        for x_start, x_end in windows:
            crop = image[:, x_start:x_end]
            result = model.predict(
                source=crop,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )[0]

            boxes = result.boxes
            if boxes is None:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clses = boxes.cls.cpu().numpy().astype(int)
            summary["detections_before_merge"] += len(xyxy)

            for box, score, cls_id in zip(xyxy, confs, clses):
                mapped = np.array([box[0] + x_start, box[1], box[2] + x_start, box[3]], dtype=np.float32)
                merged_detections.append((int(cls_id), float(score), mapped))

        kept = nms_xyxy(merged_detections, args.iou)
        summary["detections_after_merge"] += len(kept)

        lines = []
        for cls_id, score, box in kept:
            x1, y1, x2, y2 = box.tolist()
            lines.append(f"{cls_id} {score:.6f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}")
        (txt_dir / f"{image_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        if args.save_vis:
            plotted = draw_detections(image, kept)
            cv2.imwrite(str(vis_dir / image_path.name), plotted)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
