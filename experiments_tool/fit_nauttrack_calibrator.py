from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ultralytics.trackers.uncertainty_calibration import ScaleConfidenceCalibrator


def load_samples(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {"residuals", "scales", "scores"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        return (
            np.asarray(data["residuals"], dtype=np.float64),
            np.asarray(data["scales"], dtype=np.float64),
            np.asarray(data["scores"], dtype=np.float64),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a NautTrack scale/confidence-conditioned localization calibrator."
    )
    parser.add_argument("--fit-npz", required=True, help="NPZ with residuals, scales and scores from fit sequences.")
    parser.add_argument(
        "--coverage-npz",
        default="",
        help="Optional sequence-disjoint NPZ used only for empirical coverage calibration.",
    )
    parser.add_argument("--output", required=True, help="Output calibrator JSON path.")
    parser.add_argument("--scale-bins", type=int, default=3)
    parser.add_argument("--confidence-bins", type=int, default=3)
    parser.add_argument("--shrinkage", type=float, default=50.0)
    parser.add_argument("--eigenvalue-floor", type=float, default=1e-6)
    parser.add_argument("--coverage", type=float, default=0.90)
    parser.add_argument("--min-coverage-samples", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fit_path = Path(args.fit_npz)
    residuals, scales, scores = load_samples(fit_path)

    calibrator = ScaleConfidenceCalibrator().fit(
        residuals=residuals,
        scales=scales,
        scores=scores,
        scale_bins=args.scale_bins,
        confidence_bins=args.confidence_bins,
        shrinkage=args.shrinkage,
        eigenvalue_floor=args.eigenvalue_floor,
    )

    coverage_summary = None
    if args.coverage_npz:
        coverage_path = Path(args.coverage_npz)
        coverage_residuals, coverage_scales, coverage_scores = load_samples(coverage_path)
        multiplier = calibrator.fit_coverage(
            residuals=coverage_residuals,
            scales=coverage_scales,
            scores=coverage_scores,
            coverage=args.coverage,
            min_samples=args.min_coverage_samples,
        )
        coverage_summary = {
            "coverage_npz": str(coverage_path),
            "target_coverage": args.coverage,
            "covariance_multiplier": multiplier,
            "samples": int(len(coverage_residuals)),
        }

    calibrator.metadata.update(
        {
            "fit_npz": str(fit_path),
            "fit_samples": int(calibrator.n_samples),
            "coverage_split": coverage_summary,
            "warning": (
                "Fit and coverage files must contain sequence-disjoint non-test samples. "
                "Do not fit or calibrate on benchmark test sequences."
            ),
        }
    )
    output_path = Path(args.output)
    calibrator.save(output_path)

    report = {
        "output": str(output_path),
        "fit_samples": int(calibrator.n_samples),
        "scale_edges_log": calibrator.scale_edges.tolist(),
        "confidence_edges": calibrator.score_edges.tolist(),
        "bin_counts": calibrator.bin_counts.tolist(),
        "global_mean": calibrator.global_mean.tolist(),
        "global_covariance": calibrator.global_covariance.tolist(),
        "coverage_scale": float(calibrator.coverage_scale),
        "coverage": coverage_summary,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
