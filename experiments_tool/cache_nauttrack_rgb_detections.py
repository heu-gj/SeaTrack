from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_ablation_suite import build_detector, cache_sequence_detections
from sealine_mot_runner import (
    DEFAULT_RAW_MATCH_ROOT,
    infer_sequence_name_from_dir,
    parse_sequence_names,
    resolve_raw_match_sequence_dirs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache visible-RGB detections for NautTrack calibration without running a tracker."
    )
    parser.add_argument("--detector", required=True, help="Path to detector weights.")
    parser.add_argument("--raw-match-root", default=str(DEFAULT_RAW_MATCH_ROOT))
    parser.add_argument("--sequence-names", required=True, help="Comma-separated RGB train/calibration sequence names.")
    parser.add_argument("--cache-dir", required=True, help="Directory for per-sequence detection pickle files.")
    parser.add_argument("--detector-mode", default="full", choices=("full", "two-slice"))
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    parser.add_argument("--overlap-ratio", type=float, default=0.15)
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing sequence caches.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence_names = parse_sequence_names(args.sequence_names)
    if not sequence_names:
        raise ValueError("no sequence names were supplied")

    raw_match_root = Path(args.raw_match_root)
    sequence_dirs = resolve_raw_match_sequence_dirs(raw_match_root, sequence_names)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    detector = build_detector(
        args.detector_mode,
        args.detector,
        args.imgsz,
        args.conf,
        args.iou,
        args.device,
        args.overlap_ratio,
    )

    manifest: dict[str, Any] = {}
    for sequence_dir in sequence_dirs:
        sequence_name = infer_sequence_name_from_dir(sequence_dir)
        cache_file = cache_dir / f"{sequence_name}.pkl"
        if cache_file.exists() and not args.overwrite:
            manifest[sequence_name] = {
                "seq_name": sequence_name,
                "cache_file": str(cache_file),
                "cached": True,
                "reused": True,
            }
            continue

        result = cache_sequence_detections(sequence_dir, detector, cache_file)
        result["reused"] = False
        manifest[sequence_name] = result

    manifest_path = cache_dir / "cache_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "cache_dir": str(cache_dir),
                "sequence_count": len(manifest),
                "manifest": str(manifest_path),
                "sequences": manifest,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
