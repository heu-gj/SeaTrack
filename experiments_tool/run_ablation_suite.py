from __future__ import annotations

import argparse
import json
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ultralytics.trackers.track import TRACKER_MAP
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml

from evaluate_mot_results import DEFAULT_SEQUENCE_NAMES, evaluate_result_root, write_outputs
from evaluate_mot_results_trackeval import (
    discover_trackeval_root,
    load_trackeval_summary,
    prepare_trackeval_workspace,
    run_trackeval,
    write_manifest as write_trackeval_manifest,
    find_summary_file as find_trackeval_summary_file,
)
from sealine_mot_runner import (
    DEFAULT_RAW_MATCH_ROOT,
    FullImageUltralyticsDetector,
    TwoSliceUltralyticsDetector,
    draw_tracks,
    infer_sequence_name_from_dir,
    parse_sequence_names,
    resolve_raw_match_sequence_dirs,
    sanitize_track_row,
    sorted_image_paths,
)


DEFAULT_TRACKERS = [
    r"E:\Code\PRCV\ultralytics\ultralytics\cfg\trackers\ablations\bytetrack_baseline.yaml",
    r"E:\Code\PRCV\ultralytics\ultralytics\cfg\trackers\ablations\sealinetracker_innov1_only.yaml",
    r"E:\Code\PRCV\ultralytics\ultralytics\cfg\trackers\ablations\sealinetracker_innov2_only.yaml",
    r"E:\Code\PRCV\ultralytics\ultralytics\cfg\trackers\ablations\sealinetracker_full.yaml",
]


class CachedBoxes:
    def __init__(self, data: list[list[float]] | np.ndarray):
        array = np.asarray(data, dtype=np.float32)
        self.data = array.reshape(-1, 6) if array.size else np.zeros((0, 6), dtype=np.float32)

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, idx) -> "CachedBoxes":
        sliced = self.data[idx]
        if sliced.ndim == 1:
            sliced = sliced[None, :]
        return CachedBoxes(sliced)

    @property
    def conf(self) -> np.ndarray:
        return self.data[:, 4]

    @property
    def cls(self) -> np.ndarray:
        return self.data[:, 5]

    @property
    def xyxy(self) -> np.ndarray:
        return self.data[:, :4]

    @property
    def xywh(self) -> np.ndarray:
        boxes = self.xyxy.copy()
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        boxes[:, 0] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes[:, 1] = boxes[:, 1] + boxes[:, 3] / 2.0
        return boxes


def build_detector(mode: str, detector_weights: str, imgsz: int, conf: float, iou: float, device: str, overlap_ratio: float):
    if mode == "full":
        return FullImageUltralyticsDetector(detector_weights, imgsz, conf, iou, device, classes=None)
    if mode == "two-slice":
        return TwoSliceUltralyticsDetector(detector_weights, imgsz, conf, iou, device, overlap_ratio=overlap_ratio, classes=None)
    raise ValueError(f"Unsupported detector mode: {mode}")


def cache_sequence_detections(sequence_dir: Path, detector, cache_file: Path) -> dict[str, Any]:
    images = sorted_image_paths(sequence_dir)
    seq_name = infer_sequence_name_from_dir(sequence_dir)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    frames: list[dict[str, Any]] = []
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        result = detector.infer(image, image_path)
        boxes = result.boxes.data.cpu().numpy().astype(np.float32) if result.boxes is not None and len(result.boxes) else np.zeros((0, 6), dtype=np.float32)
        frames.append(
            {
                "image_path": str(image_path),
                "image_name": image_path.name,
                "shape": image.shape[:2],
                "boxes": boxes.tolist(),
            }
        )

    payload = {"seq_name": seq_name, "sequence_dir": str(sequence_dir), "frames": frames}
    with cache_file.open("wb") as f:
        pickle.dump(payload, f)
    return {"seq_name": seq_name, "cache_file": str(cache_file), "frames": len(frames)}


def load_cache(cache_file: Path) -> dict[str, Any]:
    with cache_file.open("rb") as f:
        return pickle.load(f)


def run_tracker_from_cache(
    tracker_yaml: str,
    cache_files: list[str],
    output_root: str,
    save_vis: bool = False,
) -> dict[str, Any]:
    tracker_path = check_yaml(tracker_yaml)
    cfg = IterableSimpleNamespace(**YAML.load(tracker_path))
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}
    for cache_path_str in cache_files:
        cache = load_cache(Path(cache_path_str))
        seq_name = cache["seq_name"]
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)

        seq_out_dir = output_root_path / seq_name
        seq_out_dir.mkdir(parents=True, exist_ok=True)
        mot_path = seq_out_dir / f"{seq_name}.txt"
        vis_dir = seq_out_dir / "vis" if save_vis else None
        if vis_dir is not None:
            vis_dir.mkdir(parents=True, exist_ok=True)

        mot_lines: list[str] = []
        total_tracks = 0
        for frame_idx, frame in enumerate(cache["frames"], start=1):
            image = cv2.imread(frame["image_path"])
            if image is None:
                continue
            image_h, image_w = image.shape[:2]
            det_boxes = CachedBoxes(frame["boxes"])
            tracks = tracker.update(det_boxes, image)
            vis_tracks: list[tuple[float, float, float, float, int, float, int]] = []

            for row in tracks:
                sanitized = sanitize_track_row(row, image_w, image_h)
                if sanitized is None:
                    continue
                x1, y1, x2, y2, track_id, score, cls_id = sanitized
                vis_tracks.append(sanitized)
                mot_lines.append(
                    f"{frame_idx},{track_id},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{score:.6f},{cls_id},-1"
                )
                total_tracks += 1

            if vis_dir is not None:
                plotted = draw_tracks(image, vis_tracks)
                cv2.imwrite(str(vis_dir / frame["image_name"]), plotted)

        mot_path.write_text("\n".join(mot_lines) + ("\n" if mot_lines else ""), encoding="utf-8")
        summary[seq_name] = {
            "mot_txt": str(mot_path),
            "frames": len(cache["frames"]),
            "tracks_written": total_tracks,
            "tracker_yaml": str(tracker_path),
            "tracker_type": cfg.tracker_type,
        }

    (output_root_path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def worker_run_tracker(job: dict[str, Any]) -> dict[str, Any]:
    tracker_yaml = job["tracker_yaml"]
    tracker_name = Path(tracker_yaml).stem
    result_root = Path(job["output_root"]) / tracker_name
    run_tracker_from_cache(tracker_yaml, job["cache_files"], str(result_root), save_vis=job["save_vis"])
    rows, manifest = evaluate_result_root(
        result_root=result_root,
        gt_root=Path(job["gt_root"]),
        sequence_names=job["sequence_names"],
        gt_file_name=job["gt_file_name"],
        iou_thresh=job["iou_thresh"],
    )
    write_outputs(result_root / "evaluation", rows, manifest)
    trackeval_summary_csv = None
    trackeval_rows = None
    if job["run_trackeval"]:
        trackeval_root = discover_trackeval_root(job["trackeval_root"])
        prep = prepare_trackeval_workspace(
            result_root=result_root,
            gt_root=Path(job["gt_root"]),
            sequence_names=job["sequence_names"],
            workspace_root=result_root / "trackeval_evaluation" / "trackeval_workspace",
            benchmark_name=job["trackeval_benchmark"],
            split_name=job["trackeval_split"],
        )
        run_trackeval(trackeval_root, prep, result_root / "trackeval_evaluation" / "trackeval_output")
        trackeval_summary_csv = find_trackeval_summary_file(result_root / "trackeval_evaluation" / "trackeval_output", prep["tracker_name"])
        trackeval_rows = load_trackeval_summary(trackeval_summary_csv)
        write_trackeval_manifest(result_root / "trackeval_evaluation", prep, trackeval_summary_csv, trackeval_rows)
    overall = next((row for row in rows if row["Sequence"] == "OVERALL"), None)
    return {
        "tracker_name": tracker_name,
        "result_root": str(result_root),
        "overall": overall,
        "trackeval_summary_csv": str(trackeval_summary_csv) if trackeval_summary_csv else None,
        "trackeval_rows": trackeval_rows,
    }


def parse_tracker_yamls(trackers: str | None) -> list[str]:
    if not trackers:
        return list(DEFAULT_TRACKERS)
    return [item.strip() for item in trackers.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a one-click ablation suite with cached detections.")
    parser.add_argument("--detector", required=True, help="Path to detector weights.")
    parser.add_argument("--detector-mode", default="full", choices=("full", "two-slice"))
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    parser.add_argument("--overlap-ratio", type=float, default=0.15)
    parser.add_argument("--raw-match-root", default=str(DEFAULT_RAW_MATCH_ROOT))
    parser.add_argument("--sequence-names", default=",".join(DEFAULT_SEQUENCE_NAMES))
    parser.add_argument("--trackers", default="", help="Comma-separated tracker yaml paths. Defaults to the main four ablation configs.")
    parser.add_argument("--output-root", required=True, help="Root directory for cache, tracker outputs, and evaluations.")
    parser.add_argument("--cache-dir", default="", help="Optional detection cache directory. Defaults to <output-root>/det_cache.")
    parser.add_argument("--gt-file-name", default="gt.txt")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold used by MOT evaluation.")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel tracker jobs.")
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--run-trackeval", action="store_true", help="Also run TrackEval for each tracker result to compute HOTA.")
    parser.add_argument("--trackeval-root", default="", help="Optional local TrackEval root.")
    parser.add_argument("--trackeval-benchmark", default="PRCV716", help="Benchmark name passed to TrackEval.")
    parser.add_argument("--trackeval-split", default="train38", help="Split name passed to TrackEval.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence_names = parse_sequence_names(args.sequence_names)
    raw_match_root = Path(args.raw_match_root)
    sequence_dirs = resolve_raw_match_sequence_dirs(raw_match_root, sequence_names)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_root / "det_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    detector = build_detector(args.detector_mode, args.detector, args.imgsz, args.conf, args.iou, args.device, args.overlap_ratio)

    cache_manifest: dict[str, Any] = {}
    cache_files: list[str] = []
    for sequence_dir in sequence_dirs:
        seq_name = infer_sequence_name_from_dir(sequence_dir)
        cache_file = cache_dir / f"{seq_name}.pkl"
        if not cache_file.exists():
            cache_manifest[seq_name] = cache_sequence_detections(sequence_dir, detector, cache_file)
        else:
            cache_manifest[seq_name] = {"seq_name": seq_name, "cache_file": str(cache_file), "cached": True}
        cache_files.append(str(cache_file))

    (cache_dir / "cache_manifest.json").write_text(json.dumps(cache_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    tracker_yamls = parse_tracker_yamls(args.trackers)
    jobs = [
        {
            "tracker_yaml": tracker_yaml,
            "cache_files": cache_files,
            "output_root": str(output_root),
            "gt_root": str(raw_match_root),
            "sequence_names": sequence_names,
            "gt_file_name": args.gt_file_name,
            "iou_thresh": args.iou_thresh,
            "save_vis": args.save_vis,
            "run_trackeval": args.run_trackeval,
            "trackeval_root": args.trackeval_root,
            "trackeval_benchmark": args.trackeval_benchmark,
            "trackeval_split": args.trackeval_split,
        }
        for tracker_yaml in tracker_yamls
    ]

    results: list[dict[str, Any]] = []
    max_workers = max(1, min(args.workers, len(jobs)))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker_run_tracker, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())

    suite_summary = {
        "detector": args.detector,
        "detector_mode": args.detector_mode,
        "imgsz": args.imgsz,
        "sequence_names": sequence_names,
        "trackers": tracker_yamls,
        "results": sorted(results, key=lambda x: x["tracker_name"]),
    }
    (output_root / "suite_summary.json").write_text(json.dumps(suite_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(suite_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
