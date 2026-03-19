from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-image detector testing on a directory of images.")
    parser.add_argument("--model", required=True, help="Path to detector weights.")
    parser.add_argument("--source", required=True, help="Directory containing test images.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument(
        "--out-dir",
        default=r"E:\Code\PRCV\ultralytics\experiments_tool\runs\test_full_image",
        help="Output directory for visualizations and txt predictions.",
    )
    parser.add_argument("--save-vis", action="store_true", help="Save visualization images.")
    return parser.parse_args()


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

    summary = {"source": str(source_dir), "images": len(images), "detections": 0}
    for image_path in images:
        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )[0]

        lines = []
        boxes = result.boxes
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clses = boxes.cls.cpu().numpy().astype(int)
            summary["detections"] += len(xyxy)
            for box, score, cls_id in zip(xyxy, confs, clses):
                x1, y1, x2, y2 = box.tolist()
                lines.append(f"{cls_id} {score:.6f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}")

        (txt_dir / f"{image_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        if args.save_vis:
            plotted = result.plot()
            cv2.imwrite(str(vis_dir / image_path.name), plotted)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
