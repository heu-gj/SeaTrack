from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2


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


def infer_tracker_name(result_root: Path) -> str:
    return result_root.name


def discover_trackeval_root(explicit_root: str | None) -> Path:
    candidates = []
    if explicit_root:
        candidates.append(Path(explicit_root))
    candidates.extend(
        [
            Path(r"E:\Code\PRCV\TrackEval"),
            Path(r"E:\Code\PRCV\ultralytics\TrackEval"),
            Path(r"E:\Code\PRCV\third_party\TrackEval"),
        ]
    )
    for candidate in candidates:
        if (candidate / "scripts" / "run_mot_challenge.py").exists():
            return candidate
    raise FileNotFoundError(
        "TrackEval root not found. Provide --trackeval-root and make sure scripts/run_mot_challenge.py exists."
    )


def read_sequence_meta(gt_root: Path, seq_name: str) -> dict[str, Any]:
    image_dir = gt_root / seq_name / "visible" / "image"
    gt_file = gt_root / seq_name / "visible" / "gt" / "gt.txt"
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    if not images:
        raise FileNotFoundError(f"No images found for sequence {seq_name}: {image_dir}")
    image = cv2.imread(str(images[0]))
    if image is None:
        raise FileNotFoundError(f"Failed to read first image for sequence {seq_name}: {images[0]}")
    h, w = image.shape[:2]
    numeric_stems = [int(p.stem) for p in images if p.stem.isdigit()]
    seq_length = max(numeric_stems) if numeric_stems else len(images)
    return {
        "seq_name": seq_name,
        "image_dir": image_dir,
        "gt_file": gt_file,
        "seq_length": seq_length,
        "im_width": w,
        "im_height": h,
        "frame_rate": 30,
        "im_ext": images[0].suffix,
    }


def write_seqinfo_ini(seq_dir: Path, meta: dict[str, Any]) -> None:
    content = "\n".join(
        [
            "[Sequence]",
            f"name={meta['seq_name']}",
            "imDir=img1",
            f"frameRate={meta['frame_rate']}",
            f"seqLength={meta['seq_length']}",
            f"imWidth={meta['im_width']}",
            f"imHeight={meta['im_height']}",
            f"imExt={meta['im_ext']}",
            "",
        ]
    )
    (seq_dir / "seqinfo.ini").write_text(content, encoding="utf-8")


def normalize_mot_class_ids(src: Path, dst: Path, target_class_id: int = 1) -> None:
    """Copy a MOT-format txt file while collapsing all category ids to one class for TrackEval."""
    normalized_lines: list[str] = []
    with src.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 8:
                parts[7] = str(target_class_id)
            normalized_lines.append(",".join(parts))
    dst.write_text("\n".join(normalized_lines) + ("\n" if normalized_lines else ""), encoding="utf-8")


def prepare_trackeval_workspace(
    result_root: Path,
    gt_root: Path,
    sequence_names: list[str],
    workspace_root: Path,
    benchmark_name: str,
    split_name: str,
) -> dict[str, Any]:
    if workspace_root.exists():
        shutil.rmtree(workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)

    tracker_name = infer_tracker_name(result_root)
    gt_benchmark_dir = workspace_root / "gt" / "mot_challenge" / f"{benchmark_name}-{split_name}"
    tracker_benchmark_dir = workspace_root / "trackers" / "mot_challenge" / f"{benchmark_name}-{split_name}" / tracker_name / "data"
    seqmaps_dir = workspace_root / "gt" / "mot_challenge" / "seqmaps"
    gt_benchmark_dir.mkdir(parents=True, exist_ok=True)
    tracker_benchmark_dir.mkdir(parents=True, exist_ok=True)
    seqmaps_dir.mkdir(parents=True, exist_ok=True)

    used_sequences: list[str] = []
    skipped: dict[str, str] = {}
    for seq_name in sequence_names:
        result_file = result_root / seq_name / f"{seq_name}.txt"
        try:
            meta = read_sequence_meta(gt_root, seq_name)
        except Exception as exc:  # noqa: BLE001
            skipped[seq_name] = f"meta_error: {exc}"
            continue
        if not meta["gt_file"].exists():
            skipped[seq_name] = "missing_gt"
            continue
        if not result_file.exists():
            skipped[seq_name] = "missing_result"
            continue

        seq_gt_dir = gt_benchmark_dir / seq_name
        (seq_gt_dir / "gt").mkdir(parents=True, exist_ok=True)
        write_seqinfo_ini(seq_gt_dir, meta)
        normalize_mot_class_ids(meta["gt_file"], seq_gt_dir / "gt" / "gt.txt")
        normalize_mot_class_ids(result_file, tracker_benchmark_dir / f"{seq_name}.txt")
        used_sequences.append(seq_name)

    seqmap_path = seqmaps_dir / f"{benchmark_name}-{split_name}.txt"
    seqmap_lines = ["name", *used_sequences]
    seqmap_path.write_text("\n".join(seqmap_lines) + "\n", encoding="utf-8")

    return {
        "tracker_name": tracker_name,
        "workspace_root": str(workspace_root),
        "benchmark_name": benchmark_name,
        "split_name": split_name,
        "used_sequences": used_sequences,
        "skipped_sequences": skipped,
        "gt_folder": str(workspace_root / "gt" / "mot_challenge"),
        "trackers_folder": str(workspace_root / "trackers" / "mot_challenge"),
        "seqmap_path": str(seqmap_path),
    }


def run_trackeval(trackeval_root: Path, prep: dict[str, Any], output_root: Path) -> None:
    command = [
        sys.executable,
        str(trackeval_root / "scripts" / "run_mot_challenge.py"),
        "--GT_FOLDER",
        prep["gt_folder"],
        "--TRACKERS_FOLDER",
        prep["trackers_folder"],
        "--BENCHMARK",
        prep["benchmark_name"],
        "--SPLIT_TO_EVAL",
        prep["split_name"],
        "--TRACKERS_TO_EVAL",
        prep["tracker_name"],
        "--METRICS",
        "HOTA",
        "CLEAR",
        "Identity",
        "--USE_PARALLEL",
        "False",
        "--PRINT_RESULTS",
        "False",
        "--OUTPUT_FOLDER",
        str(output_root),
    ]
    subprocess.run(command, check=True)


def find_summary_file(trackeval_output_root: Path, tracker_name: str) -> Path:
    candidates = list(trackeval_output_root.rglob("*.csv"))
    for candidate in candidates:
        if candidate.name.lower().endswith("_summary.csv") or candidate.name.lower() == "pedestrian_summary.csv":
            return candidate
    for candidate in candidates:
        if candidate.name.lower().endswith("_detailed.csv") or candidate.name.lower() == "pedestrian_detailed.csv":
            return candidate
    for candidate in candidates:
        return candidate
    raise FileNotFoundError(f"Could not find TrackEval summary csv under {trackeval_output_root}")


def load_trackeval_summary(summary_csv: Path) -> list[dict[str, Any]]:
    with summary_csv.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if summary_csv.name.lower().endswith("_detailed.csv") or summary_csv.name.lower() == "pedestrian_detailed.csv":
        combined_rows = [row for row in rows if str(row.get("seq", "")).upper() == "COMBINED"]
        return combined_rows or rows[-1:] if rows else []
    return rows


def write_manifest(output_dir: Path, prep: dict[str, Any], summary_csv: Path, summary_rows: list[dict[str, Any]]) -> None:
    manifest = {
        "workspace": prep,
        "summary_csv": str(summary_csv),
        "rows": summary_rows,
    }
    (output_dir / "trackeval_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MOT results with TrackEval to compute HOTA.")
    parser.add_argument("--result-root", required=True, help="Directory containing per-sequence result folders.")
    parser.add_argument("--gt-root", default=str(DEFAULT_RAW_MATCH_ROOT), help="raw-match root directory.")
    parser.add_argument("--trackeval-root", default="", help="Path to local TrackEval root.")
    parser.add_argument("--sequence-names", default=",".join(DEFAULT_SEQUENCE_NAMES), help="Comma-separated sequence names.")
    parser.add_argument("--output-dir", default="", help="Optional output directory. Defaults to <result-root>/trackeval_evaluation.")
    parser.add_argument("--benchmark-name", default="PRCV716", help="TrackEval benchmark name.")
    parser.add_argument("--split-name", default="train38", help="TrackEval split name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_root = Path(args.result_root)
    gt_root = Path(args.gt_root)
    output_dir = Path(args.output_dir) if args.output_dir else result_root / "trackeval_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    trackeval_root = discover_trackeval_root(args.trackeval_root)
    prep = prepare_trackeval_workspace(
        result_root=result_root,
        gt_root=gt_root,
        sequence_names=parse_sequence_names(args.sequence_names),
        workspace_root=output_dir / "trackeval_workspace",
        benchmark_name=args.benchmark_name,
        split_name=args.split_name,
    )
    if not prep["used_sequences"]:
        payload = {
            "summary_csv": "",
            "rows": [],
            "skipped": True,
            "reason": "no_usable_sequences",
            "workspace": prep,
        }
        (output_dir / "trackeval_manifest.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    run_trackeval(trackeval_root, prep, output_dir / "trackeval_output")
    summary_csv = find_summary_file(output_dir / "trackeval_output", prep["tracker_name"])
    summary_rows = load_trackeval_summary(summary_csv)
    write_manifest(output_dir, prep, summary_csv, summary_rows)
    print(json.dumps({"summary_csv": str(summary_csv), "rows": summary_rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
