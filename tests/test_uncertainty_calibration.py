from __future__ import annotations

import numpy as np
import pytest

from ultralytics.trackers.uncertainty_calibration import ScaleConfidenceCalibrator


def make_samples(seed: int = 7, count: int = 900):
    rng = np.random.default_rng(seed)
    scales = np.exp(rng.uniform(np.log(0.001), np.log(0.05), count))
    scores = rng.uniform(0.2, 0.99, count)
    base_std = 0.012 + 0.0015 / np.sqrt(scales)
    residuals = rng.normal(size=(count, 4)) * base_std[:, None]
    residuals[:, 1] *= 1.2
    residuals[:, 2] *= 0.7
    return residuals, scales, scores


def test_fit_and_predict_xyah_returns_finite_spd_covariances():
    residuals, scales, scores = make_samples()
    calibrator = ScaleConfidenceCalibrator().fit(
        residuals, scales, scores, scale_bins=3, confidence_bins=3, shrinkage=20.0
    )

    boxes = np.asarray(
        [
            [100.0, 120.0, 114.0, 130.0],
            [300.0, 200.0, 380.0, 260.0],
            [500.0, 300.0, 510.0, 308.0],
        ],
        dtype=np.float32,
    )
    measurements, covariances, bin_ids = calibrator.predict_xyah(
        boxes, np.asarray([0.9, 0.7, 0.4]), image_shape=(720, 1280)
    )

    assert measurements.shape == (3, 4)
    assert covariances.shape == (3, 4, 4)
    assert bin_ids.shape == (3,)
    assert np.isfinite(measurements).all()
    assert np.isfinite(covariances).all()
    for covariance in covariances:
        assert np.allclose(covariance, covariance.T, atol=1e-6)
        assert np.linalg.eigvalsh(covariance).min() > 0.0


def test_save_load_round_trip(tmp_path):
    residuals, scales, scores = make_samples(count=300)
    calibrator = ScaleConfidenceCalibrator().fit(residuals, scales, scores)
    path = tmp_path / "calibrator.json"
    calibrator.save(path)
    restored = ScaleConfidenceCalibrator.load(path)

    boxes = np.asarray([[20.0, 30.0, 42.0, 48.0], [100.0, 120.0, 220.0, 200.0]])
    conf = np.asarray([0.8, 0.6])
    expected = calibrator.predict_xyah(boxes, conf, (480, 640))
    actual = restored.predict_xyah(boxes, conf, (480, 640))

    for expected_array, actual_array in zip(expected, actual):
        assert np.allclose(expected_array, actual_array)


def test_coverage_calibration_uses_held_out_samples():
    fit_residuals, fit_scales, fit_scores = make_samples(seed=1, count=500)
    held_residuals, held_scales, held_scores = make_samples(seed=2, count=500)
    calibrator = ScaleConfidenceCalibrator().fit(
        fit_residuals, fit_scales, fit_scores, scale_bins=3, confidence_bins=3
    )

    multiplier = calibrator.fit_coverage(
        held_residuals, held_scales, held_scores, coverage=0.9, min_samples=100
    )
    assert np.isfinite(multiplier)
    assert multiplier > 0.0
    assert calibrator.coverage_scale == multiplier


def test_empty_detection_input_has_well_defined_shapes():
    residuals, scales, scores = make_samples(count=100)
    calibrator = ScaleConfidenceCalibrator().fit(residuals, scales, scores)
    measurements, covariances, bin_ids = calibrator.predict_xyah(
        np.zeros((0, 4)), np.zeros((0,)), image_shape=(480, 640)
    )
    assert measurements.shape == (0, 4)
    assert covariances.shape == (0, 4, 4)
    assert bin_ids.shape == (0,)


def test_invalid_training_shapes_are_rejected():
    calibrator = ScaleConfidenceCalibrator()
    with pytest.raises(ValueError, match="shape"):
        calibrator.fit(np.zeros((10, 3)), np.ones(10), np.ones(10))
