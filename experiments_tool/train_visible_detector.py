from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a visible-light detector on the exported 716 dataset.")
    parser.add_argument("--model", default="yolo26n.pt", help="Model checkpoint or YAML to train from.")
    parser.add_argument(
        "--data",
        default=r"E:\Dataset\716\ultralytics-visible-yolo26\dataset.yaml",
        help="Dataset YAML path.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default="0", help="Training device, e.g. 0 or cpu.")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument("--project", default="runs/prcv_detect", help="Ultralytics project directory.")
    parser.add_argument("--name", default="yolo26_visible_716", help="Run name.")
    parser.add_argument("--patience", type=int, default=30, help="Early-stopping patience.")
    parser.add_argument("--seed", type=int, default=716, help="Random seed.")
    parser.add_argument("--pretrained", action="store_true", help="Force pretrained=True if the selected model supports it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        seed=args.seed,
        pretrained=args.pretrained,
    )


if __name__ == "__main__":
    main()
