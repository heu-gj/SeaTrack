from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_ablation_suite import parse_sequence_names

import run_ablation_suite as suite


DEFAULT_FINE_GRAINED_TRACKERS = [
    r"E:\Code\PRCV\ultralytics\ultralytics\cfg\trackers\ablations\seatrack_sgl.yaml",
    r"E:\Code\PRCV\ultralytics\ultralytics\cfg\trackers\ablations\seatrack_sgl_wo_scl.yaml",
    r"E:\Code\PRCV\ultralytics\ultralytics\cfg\trackers\ablations\seatrack_sgl_wo_srl.yaml",
    r"E:\Code\PRCV\ultralytics\ultralytics\cfg\trackers\ablations\seatrack_sgl_wo_ssl.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fine-grained SeaTrack SGL ablation suite.")
    parser.add_argument("--detector", required=True, help="Path to detector weights.")
    parser.add_argument("--detector-mode", default="full", choices=("full", "two-slice"))
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    parser.add_argument("--overlap-ratio", type=float, default=0.15)
    parser.add_argument("--raw-match-root", default=str(suite.DEFAULT_RAW_MATCH_ROOT))
    parser.add_argument("--sequence-names", default=",".join(suite.DEFAULT_SEQUENCE_NAMES))
    parser.add_argument(
        "--trackers",
        default=",".join(DEFAULT_FINE_GRAINED_TRACKERS),
        help="Comma-separated tracker yaml paths. Defaults to SGL full, w/o SCL, w/o SRL, and w/o SSL.",
    )
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
    sequence_dirs = suite.resolve_raw_match_sequence_dirs(raw_match_root, sequence_names)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_root / "det_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    detector = suite.build_detector(args.detector_mode, args.detector, args.imgsz, args.conf, args.iou, args.device, args.overlap_ratio)

    cache_manifest: dict[str, object] = {}
    cache_files: list[str] = []
    for sequence_dir in sequence_dirs:
        seq_name = suite.infer_sequence_name_from_dir(sequence_dir)
        cache_file = cache_dir / f"{seq_name}.pkl"
        if not cache_file.exists():
            cache_manifest[seq_name] = suite.cache_sequence_detections(sequence_dir, detector, cache_file)
        else:
            cache_manifest[seq_name] = {"seq_name": seq_name, "cache_file": str(cache_file), "cached": True}
        cache_files.append(str(cache_file))

    (cache_dir / "cache_manifest.json").write_text(json.dumps(cache_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    tracker_yamls = [item.strip() for item in args.trackers.split(",") if item.strip()]
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

    results: list[dict[str, object]] = []
    max_workers = max(1, min(args.workers, len(jobs)))
    with suite.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(suite.worker_run_tracker, job) for job in jobs]
        for future in suite.as_completed(futures):
            results.append(future.result())

    suite_summary = {
        "detector": args.detector,
        "detector_mode": args.detector_mode,
        "imgsz": args.imgsz,
        "sequence_names": sequence_names,
        "trackers": tracker_yamls,
        "results": sorted(results, key=lambda x: x["tracker_name"]),
        "suite_type": "fine_grained_ablation",
    }
    (output_root / "suite_summary.json").write_text(json.dumps(suite_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(suite_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
