from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
from ultralytics.utils import IterableSimpleNamespace, YAML

from evaluate_mot_results import DEFAULT_SEQUENCE_NAMES, evaluate_result_root, write_outputs
from evaluate_mot_results_trackeval import (
    discover_trackeval_root,
    find_summary_file as find_trackeval_summary_file,
    load_trackeval_summary,
    prepare_trackeval_workspace,
    run_trackeval,
    write_manifest as write_trackeval_manifest,
)
from run_ablation_suite import CachedBoxes, build_detector, cache_sequence_detections, load_cache
from sealine_mot_runner import (
    DEFAULT_RAW_MATCH_ROOT,
    draw_tracks,
    infer_sequence_name_from_dir,
    parse_sequence_names,
    resolve_raw_match_sequence_dirs,
    sanitize_track_row,
)


BOXMOT_REPO_ROOT = Path(r"E:\Code\PRCV\boxmot-master")
BASELINE_DIR = Path(r"E:\Code\PRCV\ultralytics\ultralytics\cfg\trackers\baselines")
DEFAULT_BASELINE_TRACKERS = [
    str(BASELINE_DIR / "boxmot_ocsort.yaml"),
    str(BASELINE_DIR / "boxmot_botsort.yaml"),
    str(BASELINE_DIR / "boxmot_deepocsort.yaml"),
    str(BASELINE_DIR / "boxmot_strongsort.yaml"),
    str(BASELINE_DIR / "boxmot_boosttrack.yaml"),
    str(BASELINE_DIR / "boxmot_hybridsort.yaml"),
    str(BASELINE_DIR / "boxmot_sfsort.yaml"),
]
REID_TRACKERS = {"strongsort", "botsort", "deepocsort", "hybridsort", "boosttrack"}


def ensure_boxmot_imports():
    repo_root = str(BOXMOT_REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from boxmot import create_tracker

    return create_tracker


def load_wrapper_cfg(tracker_yaml: str) -> IterableSimpleNamespace:
    cfg = IterableSimpleNamespace(**YAML.load(tracker_yaml))
    if getattr(cfg, "tracker_type", "") != "boxmot":
        raise ValueError(f"{tracker_yaml} is not a BoxMOT baseline wrapper YAML.")
    return cfg


def resolve_reid_weights(cfg: IterableSimpleNamespace, default_reid_weights: str) -> str | None:
    tracker_name = cfg.boxmot_tracker.lower()
    yaml_reid = str(getattr(cfg, "reid_weights", "") or "").strip()
    cli_reid = str(default_reid_weights or "").strip()
    resolved = yaml_reid or cli_reid
    if tracker_name in REID_TRACKERS and not resolved:
        raise ValueError(
            f"Tracker '{tracker_name}' requires ReID weights. Set 'reid_weights' in the wrapper YAML or pass --default-reid-weights."
        )
    return resolved or None


def normalize_boxmot_device(device: str) -> str:
    token = str(device or "").strip()
    if token.isdigit():
        return f"cuda:{token}"
    return token or "cpu"


def create_boxmot_tracker(cfg: IterableSimpleNamespace, default_device: str, default_reid_weights: str):
    create_tracker = ensure_boxmot_imports()
    tracker_config = str(getattr(cfg, "boxmot_config", "") or "").strip() or None
    tracker_device = normalize_boxmot_device(str(getattr(cfg, "device", "") or "").strip() or default_device)
    return create_tracker(
        tracker_type=cfg.boxmot_tracker,
        tracker_config=tracker_config,
        reid_weights=resolve_reid_weights(cfg, default_reid_weights),
        device=tracker_device,
        half=bool(getattr(cfg, "half", False)),
        per_class=bool(getattr(cfg, "per_class", False)),
    )


def run_boxmot_from_cache(
    tracker_yaml: str,
    cache_files: list[str],
    output_root: str,
    default_device: str,
    default_reid_weights: str,
    save_vis: bool = False,
) -> dict[str, Any]:
    cfg = load_wrapper_cfg(tracker_yaml)
    tracker_name = getattr(cfg, "display_name", None) or Path(tracker_yaml).stem
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}
    for cache_path_str in cache_files:
        cache = load_cache(Path(cache_path_str))
        seq_name = cache["seq_name"]
        tracker = create_boxmot_tracker(cfg, default_device=default_device, default_reid_weights=default_reid_weights)

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
            tracks = tracker.update(det_boxes.data, image)
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
            "tracker_yaml": str(Path(tracker_yaml)),
            "tracker_name": tracker_name,
            "boxmot_tracker": cfg.boxmot_tracker,
        }

    (output_root_path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def worker_run_baseline(job: dict[str, Any]) -> dict[str, Any]:
    tracker_yaml = job["tracker_yaml"]
    cfg = load_wrapper_cfg(tracker_yaml)
    tracker_name = getattr(cfg, "display_name", None) or Path(tracker_yaml).stem
    result_root = Path(job["output_root"]) / tracker_name

    run_boxmot_from_cache(
        tracker_yaml=tracker_yaml,
        cache_files=job["cache_files"],
        output_root=str(result_root),
        default_device=job["device"],
        default_reid_weights=job["default_reid_weights"],
        save_vis=job["save_vis"],
    )

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
        return list(DEFAULT_BASELINE_TRACKERS)
    return [item.strip() for item in trackers.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-click BoxMOT baseline experiments with cached detections.")
    parser.add_argument("--detector", required=True, help="Path to detector weights.")
    parser.add_argument("--detector-mode", default="full", choices=("full", "two-slice"))
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0", help="Device passed to the detector and used as BoxMOT fallback device.")
    parser.add_argument("--overlap-ratio", type=float, default=0.15)
    parser.add_argument("--raw-match-root", default=str(DEFAULT_RAW_MATCH_ROOT))
    parser.add_argument("--sequence-names", default=",".join(DEFAULT_SEQUENCE_NAMES))
    parser.add_argument(
        "--trackers",
        default="",
        help="Comma-separated BoxMOT wrapper YAMLs. Defaults to ocsort, botsort, deepocsort, strongsort, boosttrack, hybridsort, and sfsort.",
    )
    parser.add_argument("--default-reid-weights", default="", help="Fallback ReID weights for BoxMOT ReID-based trackers.")
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
            "device": args.device,
            "default_reid_weights": args.default_reid_weights,
        }
        for tracker_yaml in tracker_yamls
    ]

    results: list[dict[str, Any]] = []
    max_workers = max(1, min(args.workers, len(jobs)))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker_run_baseline, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())

    suite_summary = {
        "suite_type": "boxmot_baseline",
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
