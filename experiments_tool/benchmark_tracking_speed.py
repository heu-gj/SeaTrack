from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark tracking FPS for a model and tracker config.")
    parser.add_argument("--model", required=True, help="Path to the detector weights.")
    parser.add_argument("--source", required=True, help="Video path, image directory, or stream source.")
    parser.add_argument("--tracker", required=True, help="Tracker YAML path.")
    parser.add_argument("--device", default="", help="Ultralytics device argument, e.g. 0 or cpu.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs not included in the final average.")
    parser.add_argument("--runs", type=int, default=3, help="Measured benchmark runs.")
    parser.add_argument("--save-json", default="", help="Optional path to save the benchmark JSON report.")
    return parser.parse_args()


def run_once(model_path: str, source: str, tracker: str, device: str, imgsz: int, conf: float) -> dict[str, float]:
    model = YOLO(model_path)
    frame_count = 0
    start = time.perf_counter()
    results = model.track(
        source=source,
        tracker=tracker,
        stream=True,
        save=False,
        verbose=False,
        device=device,
        imgsz=imgsz,
        conf=conf,
        persist=False,
    )
    for _result in results:
        frame_count += 1
    elapsed = time.perf_counter() - start
    fps = frame_count / elapsed if elapsed > 0 else 0.0
    return {"frames": frame_count, "elapsed": elapsed, "fps": fps}


def main() -> None:
    args = parse_args()
    source = str(Path(args.source)) if Path(args.source).exists() else args.source

    for _ in range(max(args.warmup, 0)):
        run_once(args.model, source, args.tracker, args.device, args.imgsz, args.conf)

    runs = [run_once(args.model, source, args.tracker, args.device, args.imgsz, args.conf) for _ in range(args.runs)]
    avg_frames = sum(item["frames"] for item in runs) / max(len(runs), 1)
    avg_elapsed = sum(item["elapsed"] for item in runs) / max(len(runs), 1)
    avg_fps = sum(item["fps"] for item in runs) / max(len(runs), 1)

    report = {
        "model": args.model,
        "source": source,
        "tracker": args.tracker,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "warmup": args.warmup,
        "runs": args.runs,
        "average_frames": avg_frames,
        "average_elapsed": avg_elapsed,
        "average_fps": avg_fps,
        "per_run": runs,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.save_json:
        Path(args.save_json).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
