from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import optuna
except ImportError as exc:  # pragma: no cover - import guard for runtime use
    raise SystemExit(
        "Optuna is required for this script. Install it with `pip install optuna` in the experiment environment."
    ) from exc

import run_ablation_suite as suite
from evaluate_mot_results import DEFAULT_SEQUENCE_NAMES, evaluate_result_root, write_outputs


DEFAULT_BASE_TRACKER = Path(r"E:\Code\PRCV\ultralytics\ultralytics\cfg\trackers\sealinetracker.yaml")
DEFAULT_OUTPUT_ROOT = Path(r"E:\Code\PRCV\ultralytics\output\optuna_seatrack")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune SeaTrack with Optuna using cached detector outputs.")
    parser.add_argument("--detector", required=True, help="Path to detector weights.")
    parser.add_argument("--detector-mode", default="full", choices=("full", "two-slice"))
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    parser.add_argument("--overlap-ratio", type=float, default=0.15)
    parser.add_argument("--raw-match-root", default=str(suite.DEFAULT_RAW_MATCH_ROOT))
    parser.add_argument("--sequence-names", default=",".join(DEFAULT_SEQUENCE_NAMES))
    parser.add_argument("--base-tracker", default=str(DEFAULT_BASE_TRACKER))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--cache-dir", default="", help="Optional detection cache directory. Defaults to <output-root>/det_cache.")
    parser.add_argument("--gt-file-name", default="gt.txt")
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--study-name", default="seatrack_optuna")
    parser.add_argument("--sampler-seed", type=int, default=3407)
    parser.add_argument(
        "--objective",
        default="composite",
        choices=("composite", "mota", "idf1", "fn"),
        help="Optimization target. `composite` maximizes MOTA and IDF1 while penalizing IDs and FN.",
    )
    parser.add_argument("--mota-weight", type=float, default=0.6, help="Used only when --objective composite.")
    parser.add_argument("--idf1-weight", type=float, default=0.4, help="Used only when --objective composite.")
    parser.add_argument("--ids-penalty", type=float, default=0.002, help="Used only when --objective composite.")
    parser.add_argument("--fn-penalty", type=float, default=0.0005, help="Used only when --objective composite.")
    parser.add_argument("--save-vis", action="store_true", help="Save tracking visualizations for each trial.")
    return parser.parse_args()


def load_flat_yaml(path: Path) -> tuple[list[str], dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    values: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return lines, values


def apply_overrides(lines: list[str], overrides: dict[str, str]) -> list[str]:
    updated_lines: list[str] = []
    seen: set[str] = set()
    for raw_line in lines:
        prefix, *comment = raw_line.split("#", 1)
        if ":" not in prefix:
            updated_lines.append(raw_line)
            continue
        key, _value = prefix.split(":", 1)
        key = key.strip()
        if key not in overrides:
            updated_lines.append(raw_line)
            continue
        seen.add(key)
        new_line = f"{key}: {overrides[key]}"
        if comment:
            new_line += f" # {comment[0].strip()}"
        updated_lines.append(new_line)
    for key, value in overrides.items():
        if key not in seen:
            updated_lines.append(f"{key}: {value}")
    return updated_lines


def write_trial_tracker(base_lines: list[str], tracker_path: Path, overrides: dict[str, str]) -> None:
    tracker_path.parent.mkdir(parents=True, exist_ok=True)
    tracker_path.write_text("\n".join(apply_overrides(base_lines, overrides)) + "\n", encoding="utf-8")


def build_cache(args: argparse.Namespace, output_root: Path) -> tuple[list[str], list[str], Path]:
    sequence_names = suite.parse_sequence_names(args.sequence_names)
    raw_match_root = Path(args.raw_match_root)
    sequence_dirs = suite.resolve_raw_match_sequence_dirs(raw_match_root, sequence_names)

    cache_dir = Path(args.cache_dir) if args.cache_dir else output_root / "det_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    detector = suite.build_detector(args.detector_mode, args.detector, args.imgsz, args.conf, args.iou, args.device, args.overlap_ratio)

    cache_manifest: dict[str, Any] = {}
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
    return sequence_names, cache_files, raw_match_root


def suggest_overrides(trial: optuna.Trial) -> dict[str, str]:
    return {
        "match_thresh": f"{trial.suggest_float('match_thresh', 0.70, 0.90):.4f}",
        "adaptive_iou_alpha": f"{trial.suggest_float('adaptive_iou_alpha', 2.0, 6.0):.4f}",
        "adaptive_iou_beta": f"{trial.suggest_float('adaptive_iou_beta', 8.0, 16.0):.4f}",
        "adaptive_iou_lambda_tau": f"{trial.suggest_float('adaptive_iou_lambda_tau', 0.02, 0.08):.4f}",
        "geom_center_y_weight": f"{trial.suggest_float('geom_center_y_weight', 0.4, 0.9):.4f}",
        "geom_scale_height_weight": f"{trial.suggest_float('geom_scale_height_weight', 0.30, 0.50):.4f}",
        "use_recovery_matching": "False",
    }


def compute_objective(row: dict[str, Any], args: argparse.Namespace) -> float:
    if args.objective == "mota":
        return float(row["MOTA"])
    if args.objective == "idf1":
        return float(row["IDF1"])
    if args.objective == "fn":
        return -float(row["FN"])
    return (
        args.mota_weight * float(row["MOTA"])
        + args.idf1_weight * float(row["IDF1"])
        - args.ids_penalty * float(row["IDs"])
        - args.fn_penalty * float(row["FN"])
    )


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    base_lines, _ = load_flat_yaml(Path(args.base_tracker))
    sequence_names, cache_files, raw_match_root = build_cache(args, output_root)

    tracker_dir = output_root / "trial_trackers"
    result_dir = output_root / "trial_results"
    trials_dir = output_root / "trials"
    tracker_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    trials_dir.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    study = optuna.create_study(direction="maximize", study_name=args.study_name, sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        overrides = suggest_overrides(trial)
        tracker_path = tracker_dir / f"trial_{trial.number:04d}.yaml"
        write_trial_tracker(base_lines, tracker_path, overrides)

        trial_result_root = result_dir / f"trial_{trial.number:04d}"
        suite.run_tracker_from_cache(str(tracker_path), cache_files, str(trial_result_root), save_vis=args.save_vis)
        rows, manifest = evaluate_result_root(
            result_root=trial_result_root,
            gt_root=raw_match_root,
            sequence_names=sequence_names,
            gt_file_name=args.gt_file_name,
            iou_thresh=args.iou_thresh,
        )
        write_outputs(trial_result_root / "evaluation", rows, manifest)

        overall = next((row for row in rows if row["Sequence"] == "OVERALL"), None)
        if overall is None:
            raise RuntimeError(f"Missing OVERALL row for trial {trial.number}")

        score = compute_objective(overall, args)
        trial.set_user_attr("tracker_yaml", str(tracker_path))
        trial.set_user_attr("result_root", str(trial_result_root))
        for key, value in overall.items():
            if key != "Sequence":
                trial.set_user_attr(key, value)
        for key, value in overrides.items():
            trial.set_user_attr(key, value)

        trial_record = {
            "trial": trial.number,
            "score": score,
            "params": trial.params,
            "overrides": overrides,
            "overall": overall,
            "tracker_yaml": str(tracker_path),
            "result_root": str(trial_result_root),
        }
        (trials_dir / f"trial_{trial.number:04d}.json").write_text(
            json.dumps(trial_record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return score

    study.optimize(objective, n_trials=args.n_trials)

    best = study.best_trial
    best_summary = {
        "study_name": args.study_name,
        "objective": args.objective,
        "best_trial": best.number,
        "best_score": best.value,
        "best_params": best.params,
        "best_attrs": best.user_attrs,
    }
    (output_root / "best_trial.json").write_text(json.dumps(best_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if "tracker_yaml" in best.user_attrs:
        best_yaml_src = Path(best.user_attrs["tracker_yaml"])
        best_yaml_dst = output_root / "best_seatrack.yaml"
        best_yaml_dst.write_text(best_yaml_src.read_text(encoding="utf-8"), encoding="utf-8")

    all_trials = []
    for trial in study.trials:
        all_trials.append(
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": str(trial.state),
                "user_attrs": trial.user_attrs,
            }
        )
    (output_root / "study_trials.json").write_text(json.dumps(all_trials, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(best_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
