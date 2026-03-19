# Experiment Tools

This folder contains lightweight utilities for the PRCV experiments around `SealineTracker`.

Files:

- `train_visible_detector.py`
  - Train a YOLO detector on the visible-light 716 dataset export.
- `test_detector_full_image.py`
  - Run detector inference on full images and save per-image prediction txt files.
- `test_detector_two_slice.py`
  - Run detector inference on two horizontal slices, remap boxes to the original image, and merge them with NMS.
- `sealine_mot_runner.py`
  - Wrap a detector and SealineTracker together and export MOT-format tracking results from image sequences.
- `evaluate_mot_results.py`
  - Evaluate per-sequence MOT txt outputs against `raw-match/<seq>/visible/gt/gt_id.txt` and write summary tables.
- `generate_ablation_configs.py`
  - Generate tracker YAML variants for the two-innovation paper version.
- `build_small_target_subsets.py`
  - Create small/medium/large subset manifests and dataset YAML files from a detection dataset.
- `benchmark_tracking_speed.py`
  - Benchmark end-to-end tracking FPS for a model and tracker config on a video or image sequence source.

Recommended experiment order:

1. Train the visible-light detector.
2. Generate tracker ablation configs.
3. Run main tracking experiments with a fixed detector.
4. Build small-target subsets from the visible dataset and evaluate on the small subset.
5. Run tracking speed benchmarks on the same detector/tracker setup.
