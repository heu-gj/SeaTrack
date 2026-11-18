# NautTrack M1: localization-residual collection and calibration

This branch adds the first isolated NautTrack implementation milestone. It does **not**
replace or modify the existing SeaTrack tracker. The current milestone only builds and
tests a post-hoc detector localization calibrator.

## Why this is the first step

The available GT-to-GT statistics support scale-dependent inter-frame overlap degradation:

| Dataset / split (RGB or visible only) | Small zero-IoU pairs | Medium | Large |
|---|---:|---:|---:|
| JMT-RGBT visible RGB (59 sequences) | 53.72% | 12.92% | 2.12% |
| JMT2022 train (38 sequences) | 42.63% | 13.07% | 2.24% |
| JMT2022 test1 (45 sequences) | 32.10% | 9.64% | 1.43% |
| JMT2022 all 83 sequences | 37.00% | 11.18% | 1.83% |

These are **consecutive-frame GT box overlap statistics**. They motivate a small-object
association study, but they are not detector localization residuals and must not be used
as the covariance targets for SCLUC.

SCLUC must be fitted from same-frame detector-to-GT residuals collected using detector
outputs and GT annotations.

## Expected detection-cache format

The collector reuses the cache format written by experiments_tool/run_ablation_suite.py:

- one pickle file per sequence, named SEQUENCE.pkl;
- cache["frames"] is a list of frames;
- each frame contains image_name, shape=(height, width), and boxes;
- boxes are an N x 6 array: x1, y1, x2, y2, confidence, class_id.

This milestone is RGB-only. The GT path is fixed to:

    RAW_MATCH_ROOT/SEQUENCE/visible/gt/gt.txt

No infrared detector, infrared GT, or RGB-T fusion path is used in NautTrack M1.

## Step 1: collect detector-to-GT residuals

Pass only explicitly selected **training or calibration sequences**. Do not include
benchmark test sequences.

    python experiments_tool/collect_nauttrack_residuals.py ^
      --raw-match-root E:\Dataset\716\raw-match ^
      --cache-dir E:\Code\PRCV\SeaTrack\output\det_cache ^
      --sequence-names 0001,0002,0003 ^
      --output-npz output\nauttrack\visible_fit.npz

The collector uses a one-to-one Hungarian assignment with a permissive center/size
gate. It can retain some zero-IoU detection/GT pairs when their centers and sizes remain
plausibly compatible. Review the JSON report and manually inspect a sample of matches,
especially zero-IoU matches, before fitting.

The default GT parser assumes MOT-style columns:
frame, id, x, y, width, height, mark, class, ...
The mark column is 1-based column 7 by default. Verify this against the actual annotation
format. Use --gt-mark-column 0 if the file has no validity/mark column. Class matching is
disabled by default because detector and GT class IDs may use different mappings.

If image names encode frame numbers, the default filename mode uses the trailing digits.
If GT frame numbering follows the sorted cache order instead, pass --frame-id-mode ordinal.
Use --frame-id-offset to handle a verified zero-based/one-based offset.

## Step 2: fit the conditional calibrator

The NPZ file contains:
- residuals: N x 4 normalized center/log-width/log-height residuals;
- scales: relative detection scales sqrt(box_area / image_area);
- scores: detector confidence values;
- sequence_ids, frame_ids, GT IDs, detection indices and matched IoUs for auditing.

Fit the scale/confidence bins on one sequence-disjoint split:

    python experiments_tool/fit_nauttrack_calibrator.py ^
      --fit-npz output\nauttrack\visible_fit.npz ^
      --output output\nauttrack\visible_calibrator.json ^
      --scale-bins 3 ^
      --confidence-bins 3 ^
      --shrinkage 50

Optionally use a second, sequence-disjoint non-test split to calibrate ellipsoidal
coverage:

    python experiments_tool/fit_nauttrack_calibrator.py ^
      --fit-npz output\nauttrack\visible_fit.npz ^
      --coverage-npz output\nauttrack\visible_coverage.npz ^
      --output output\nauttrack\visible_calibrator.json ^
      --coverage 0.90

The coverage split must not overlap the fitting sequences. The final benchmark test
sequences must remain locked until all design and thresholds are frozen.

## Dataset-specific recommendations

### JMT2022

There are 38 training and 45 test1 sequences. Use only the 38 training sequences for
model fitting and validation. A practical first split is sequence-disjoint fitting,
coverage calibration and tracker-validation subsets; the exact sequence IDs must be
recorded in the experiment manifest. The 45 test1 sequences are final evaluation only.

The 31-sequence diagnostic subset is not a substitute for the full 45-sequence test1.

### JMT-RGBT

The latest visible-RGB GT audit covers 59 sequences. The split membership of those
sequences must be supplied explicitly. Do not infer train/validation/test membership
from the fact that a GT file exists. This NautTrack milestone uses visible RGB only.

### Important methodological boundary

The calibration residuals are detector-to-GT errors in the same RGB frame. Their center
components are normalized by the matched detection width and height, so inference can
map the fitted residual distribution back into pixel coordinates consistently. They
must not be replaced with consecutive-frame same-ID GT displacement, artificial pixel
shifts, or the difference between a track prediction and a detection. Those quantities
describe different phenomena.

## Current implementation scope

Implemented in this milestone:
- quantile scale/confidence bins;
- shrinkage-regularized conditional means and full 4 x 4 covariance matrices;
- positive-definite covariance flooring;
- optional held-out global Mahalanobis coverage calibration;
- conversion from calibrated center/log-wh covariance to XYAH measurement covariance;
- NPZ collector and fitting CLI;
- unit tests for fitting, prediction, covariance validity and JSON round-trip.

Not implemented yet:
- NautTracker integration with the two-stage association pipeline;
- uncertainty-aware Hungarian cost and gating;
- measurement-covariance-aware Kalman update;
- end-to-end MOT evaluation.

Those parts should only be added after the residual collector and calibration diagnostics
have been checked on the user's actual train/validation caches.
