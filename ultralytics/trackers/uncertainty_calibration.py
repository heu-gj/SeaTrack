# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Post-hoc scale/confidence-conditioned calibration of detector localization errors.

The calibrator is detector-agnostic. It consumes matched detection/GT residuals
exported by an experiment script and returns a corrected XYAH measurement and
its 4x4 observation covariance for the existing XYAH Kalman filter.

Residual convention used during fitting:
    e = [(cx_det-cx_gt)/w_det, (cy_det-cy_gt)/h_det,
         log(w_det/w_gt), log(h_det/h_gt)]

At inference, the first two residual dimensions are mapped back to pixels using
the same detected box width/height normalization. This is a statistical calibration model, not a
claim that consecutive-frame GT motion is detector localization noise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class ScaleConfidenceCalibrator:
    """Binned, shrinkage-regularized localization-error calibrator.

    Scale and confidence bins are learned from the fitting split. Sparse cells
    shrink toward the global residual distribution. A separate held-out
    calibration split may be used by fit_coverage() to adjust one global
    ellipsoidal coverage scale.
    """

    FORMAT_VERSION = 1

    def __init__(self) -> None:
        self.scale_edges = np.empty((0,), dtype=np.float64)
        self.score_edges = np.empty((0,), dtype=np.float64)
        self.bin_means = np.zeros((1, 1, 4), dtype=np.float64)
        self.bin_covariances = np.eye(4, dtype=np.float64).reshape(1, 1, 4, 4)
        self.bin_counts = np.zeros((1, 1), dtype=np.int64)
        self.global_mean = np.zeros((4,), dtype=np.float64)
        self.global_covariance = np.eye(4, dtype=np.float64)
        self.coverage_scale = 1.0
        self.n_samples = 0
        self.metadata: dict[str, Any] = {}

    @staticmethod
    def _validate_training_arrays(
        residuals: np.ndarray, scales: np.ndarray, scores: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        residuals = np.asarray(residuals, dtype=np.float64)
        scales = np.asarray(scales, dtype=np.float64).reshape(-1)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if residuals.ndim != 2 or residuals.shape[1] != 4:
            raise ValueError("residuals must have shape (N, 4)")
        if len(residuals) != len(scales) or len(scales) != len(scores):
            raise ValueError("residuals, scales and scores must have the same length")
        valid = (
            np.isfinite(residuals).all(axis=1)
            & np.isfinite(scales)
            & (scales > 0.0)
            & np.isfinite(scores)
        )
        residuals = residuals[valid]
        scales = scales[valid]
        scores = np.clip(scores[valid], 0.0, 1.0)
        if len(residuals) < 2:
            raise ValueError("at least two finite calibration samples are required")
        return residuals, scales, scores

    @staticmethod
    def _quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
        if n_bins <= 1 or len(values) < 2:
            return np.empty((0,), dtype=np.float64)
        quantiles = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
        return np.unique(np.asarray(quantiles, dtype=np.float64))

    @staticmethod
    def _make_spd(matrix: np.ndarray, eigenvalue_floor: float) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float64)
        matrix = 0.5 * (matrix + matrix.T)
        values, vectors = np.linalg.eigh(matrix)
        values = np.maximum(values, float(eigenvalue_floor))
        result = (vectors * values[None, :]) @ vectors.T
        return 0.5 * (result + result.T)

    @staticmethod
    def _sample_covariance(samples: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        if len(samples) < 2:
            return fallback.copy()
        covariance = np.cov(samples, rowvar=False, ddof=1)
        return np.asarray(covariance, dtype=np.float64).reshape(4, 4)

    def _bin_indices(self, scales: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        log_scales = np.log(np.maximum(np.asarray(scales, dtype=np.float64), 1e-12))
        scores = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
        scale_ids = np.digitize(log_scales, self.scale_edges)
        score_ids = np.digitize(scores, self.score_edges)
        return scale_ids, score_ids

    def fit(
        self,
        residuals: np.ndarray,
        scales: np.ndarray,
        scores: np.ndarray,
        scale_bins: int = 3,
        confidence_bins: int = 3,
        shrinkage: float = 50.0,
        eigenvalue_floor: float = 1e-6,
    ) -> "ScaleConfidenceCalibrator":
        """Fit conditional residual moments using quantile bins and shrinkage."""
        residuals, scales, scores = self._validate_training_arrays(residuals, scales, scores)
        if scale_bins < 1 or confidence_bins < 1:
            raise ValueError("scale_bins and confidence_bins must be >= 1")
        if shrinkage < 0:
            raise ValueError("shrinkage must be >= 0")
        if eigenvalue_floor <= 0:
            raise ValueError("eigenvalue_floor must be > 0")

        self.scale_edges = self._quantile_edges(np.log(scales), int(scale_bins))
        self.score_edges = self._quantile_edges(scores, int(confidence_bins))
        scale_ids, score_ids = self._bin_indices(scales, scores)
        n_scale = len(self.scale_edges) + 1
        n_score = len(self.score_edges) + 1

        self.global_mean = residuals.mean(axis=0)
        global_cov = self._sample_covariance(residuals, np.eye(4, dtype=np.float64))
        self.global_covariance = self._make_spd(global_cov, eigenvalue_floor)

        self.bin_means = np.zeros((n_scale, n_score, 4), dtype=np.float64)
        self.bin_covariances = np.zeros((n_scale, n_score, 4, 4), dtype=np.float64)
        self.bin_counts = np.zeros((n_scale, n_score), dtype=np.int64)

        for scale_id in range(n_scale):
            for score_id in range(n_score):
                mask = (scale_ids == scale_id) & (score_ids == score_id)
                cell = residuals[mask]
                count = len(cell)
                self.bin_counts[scale_id, score_id] = count
                if count == 0:
                    self.bin_means[scale_id, score_id] = self.global_mean
                    self.bin_covariances[scale_id, score_id] = self.global_covariance
                    continue

                local_mean = cell.mean(axis=0)
                local_cov = self._sample_covariance(cell, self.global_covariance)
                weight = count / (count + float(shrinkage)) if shrinkage > 0 else 1.0
                mean = weight * local_mean + (1.0 - weight) * self.global_mean
                mean_delta = local_mean - self.global_mean
                covariance = (
                    weight * local_cov
                    + (1.0 - weight) * self.global_covariance
                    + weight * (1.0 - weight) * np.outer(mean_delta, mean_delta)
                )
                self.bin_means[scale_id, score_id] = mean
                self.bin_covariances[scale_id, score_id] = self._make_spd(
                    covariance, eigenvalue_floor
                )

        self.n_samples = int(len(residuals))
        self.coverage_scale = 1.0
        self.metadata = {
            "scale_bins_requested": int(scale_bins),
            "confidence_bins_requested": int(confidence_bins),
            "shrinkage": float(shrinkage),
            "eigenvalue_floor": float(eigenvalue_floor),
            "residual_definition": "detection_normalized_center_xy_and_log_wh",
        }
        return self

    def predict_distribution(
        self, scales: np.ndarray, scores: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return conditional normalized residual means/covariances for samples."""
        scales = np.asarray(scales, dtype=np.float64).reshape(-1)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(scales) != len(scores):
            raise ValueError("scales and scores must have the same length")
        if not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError("scales must be finite and positive")
        if not np.isfinite(scores).all():
            raise ValueError("scores must be finite")
        scale_ids, score_ids = self._bin_indices(scales, scores)
        means = self.bin_means[scale_ids, score_ids].copy()
        covariances = self.bin_covariances[scale_ids, score_ids].copy()
        flat_ids = scale_ids * self.bin_means.shape[1] + score_ids
        return means, covariances, flat_ids

    def fit_coverage(
        self,
        residuals: np.ndarray,
        scales: np.ndarray,
        scores: np.ndarray,
        coverage: float = 0.90,
        min_samples: int = 50,
        min_scale: float = 0.1,
        max_scale: float = 10.0,
    ) -> float:
        """Calibrate ellipsoid coverage using a sequence-disjoint held-out split."""
        residuals, scales, scores = self._validate_training_arrays(residuals, scales, scores)
        if not 0.0 < coverage < 1.0:
            raise ValueError("coverage must be between 0 and 1")
        if len(residuals) < min_samples:
            raise ValueError(
                "not enough held-out samples for coverage calibration: "
                f"{len(residuals)} < {min_samples}"
            )

        means, covariances, _ = self.predict_distribution(scales, scores)
        distances = np.empty((len(residuals),), dtype=np.float64)
        for index, (residual, mean, covariance) in enumerate(
            zip(residuals, means, covariances)
        ):
            delta = residual - mean
            distances[index] = float(delta @ np.linalg.solve(covariance, delta))

        try:
            from scipy.stats import chi2
        except ImportError as exc:  # pragma: no cover
            raise ImportError("scipy is required for chi-square coverage calibration") from exc

        empirical_quantile = float(np.quantile(distances, coverage))
        reference_quantile = float(chi2.ppf(coverage, df=4))
        if not np.isfinite(reference_quantile) or reference_quantile <= 0:
            raise RuntimeError("failed to compute the chi-square reference quantile")
        self.coverage_scale = float(
            np.clip(empirical_quantile / reference_quantile, min_scale, max_scale)
        )
        self.metadata["coverage_target"] = float(coverage)
        self.metadata["coverage_empirical_quantile"] = empirical_quantile
        self.metadata["coverage_reference_quantile"] = reference_quantile
        self.metadata["coverage_scale"] = self.coverage_scale
        return self.coverage_scale

    def predict_xyah(
        self,
        boxes_xyxy: np.ndarray,
        scores: np.ndarray,
        image_shape: tuple[int, int],
        covariance_floor: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Correct detections and return XYAH measurements plus 4x4 covariances."""
        boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(boxes) != len(scores):
            raise ValueError("boxes_xyxy and scores must have the same length")
        if len(image_shape) != 2:
            raise ValueError("image_shape must be (height, width)")
        image_height, image_width = map(float, image_shape)
        if image_height <= 0 or image_width <= 0:
            raise ValueError("image dimensions must be positive")
        if len(boxes) == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0, 4, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )
        if not np.isfinite(boxes).all():
            raise ValueError("boxes_xyxy must be finite")

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        if np.any(widths <= 0) or np.any(heights <= 0):
            raise ValueError("all detection boxes must have positive width and height")

        scales = np.sqrt((widths * heights) / (image_width * image_height))
        means, covariances, bin_ids = self.predict_distribution(scales, scores)

        centers_x = 0.5 * (boxes[:, 0] + boxes[:, 2])
        centers_y = 0.5 * (boxes[:, 1] + boxes[:, 3])
        raw_z = np.column_stack((centers_x, centers_y, np.log(widths), np.log(heights)))
        jacobian_scale = np.column_stack((widths, heights, np.ones_like(widths), np.ones_like(heights)))
        corrected_z = raw_z - jacobian_scale * means

        cov_z = covariances * (
            jacobian_scale[:, :, None] * jacobian_scale[:, None, :]
        ) * float(self.coverage_scale)

        corrected_width = np.exp(corrected_z[:, 2])
        corrected_height = np.exp(corrected_z[:, 3])
        aspect = corrected_width / np.maximum(corrected_height, 1e-12)
        corrected_xyah = np.column_stack(
            (corrected_z[:, 0], corrected_z[:, 1], aspect, corrected_height)
        )

        transform = np.zeros((len(boxes), 4, 4), dtype=np.float64)
        transform[:, 0, 0] = 1.0
        transform[:, 1, 1] = 1.0
        transform[:, 2, 2] = aspect
        transform[:, 2, 3] = -aspect
        transform[:, 3, 3] = corrected_height
        cov_xyah = transform @ cov_z @ np.transpose(transform, (0, 2, 1))

        for index in range(len(cov_xyah)):
            cov_xyah[index] = self._make_spd(cov_xyah[index], covariance_floor)

        return (
            corrected_xyah.astype(np.float32),
            cov_xyah.astype(np.float32),
            bin_ids.astype(np.int64),
        )

    def save(self, path: str | Path) -> None:
        """Save the fitted calibrator as a portable JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": self.FORMAT_VERSION,
            "scale_edges": self.scale_edges.tolist(),
            "score_edges": self.score_edges.tolist(),
            "bin_means": self.bin_means.tolist(),
            "bin_covariances": self.bin_covariances.tolist(),
            "bin_counts": self.bin_counts.tolist(),
            "global_mean": self.global_mean.tolist(),
            "global_covariance": self.global_covariance.tolist(),
            "coverage_scale": float(self.coverage_scale),
            "n_samples": int(self.n_samples),
            "metadata": self.metadata,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ScaleConfidenceCalibrator":
        """Load a calibrator saved by save()."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError(
                f"unsupported calibrator format version: {payload.get('format_version')}"
            )
        calibrator = cls()
        calibrator.scale_edges = np.asarray(payload["scale_edges"], dtype=np.float64)
        calibrator.score_edges = np.asarray(payload["score_edges"], dtype=np.float64)
        calibrator.bin_means = np.asarray(payload["bin_means"], dtype=np.float64)
        calibrator.bin_covariances = np.asarray(payload["bin_covariances"], dtype=np.float64)
        calibrator.bin_counts = np.asarray(payload["bin_counts"], dtype=np.int64)
        calibrator.global_mean = np.asarray(payload["global_mean"], dtype=np.float64)
        calibrator.global_covariance = np.asarray(payload["global_covariance"], dtype=np.float64)
        calibrator.coverage_scale = float(payload.get("coverage_scale", 1.0))
        calibrator.n_samples = int(payload.get("n_samples", 0))
        calibrator.metadata = dict(payload.get("metadata", {}))
        expected_shape = (len(calibrator.scale_edges) + 1, len(calibrator.score_edges) + 1)
        if calibrator.bin_means.shape != expected_shape + (4,):
            raise ValueError("invalid bin_means shape in calibrator file")
        if calibrator.bin_covariances.shape != expected_shape + (4, 4):
            raise ValueError("invalid bin_covariances shape in calibrator file")
        if calibrator.bin_counts.shape != expected_shape:
            raise ValueError("invalid bin_counts shape in calibrator file")
        return calibrator


__all__ = ("ScaleConfidenceCalibrator",)
