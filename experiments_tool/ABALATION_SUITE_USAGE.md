# Ablation Suite Usage

## Purpose

This pipeline is designed for:

- caching detector outputs once
- running multiple tracker YAMLs on the same cached detections
- evaluating each tracker result with a unified metric set

## Scripts

- `E:\Code\PRCV\ultralytics\experiments_tool\run_ablation_suite.py`
- `E:\Code\PRCV\ultralytics\experiments_tool\evaluate_mot_results.py`

## Final metric set

Main metrics written by the evaluation script:

- `HOTA`
- `IDF1`
- `MOTA`
- `IDs`
- `FP`
- `FN`
- `Precision`
- `Recall`
- `IDP`
- `IDR`
- `Frames`

Note:

- `HOTA` is currently left empty unless a dedicated HOTA evaluator is integrated later.
- The current script fully supports `IDF1`, `MOTA`, `IDs`, `FP`, `FN`, `Precision`, and `Recall`.
- TrackEval integration is now available through `evaluate_mot_results_trackeval.py` or `run_ablation_suite.py --run-trackeval`.

## Default sequence list

By default, the scripts use the current train-38 sequence list:

- `0001`
- `0002`
- `0003`
- `0005`
- `0009`
- `0010`
- `0011`
- `0012`
- `0014`
- `0020`
- `0028`
- `0040`
- `0043`
- `0047`
- `0051`
- `0058`
- `0081`
- `0086`
- `0087`
- `0088`
- `0089`
- `0090`
- `0091`
- `0092`
- `0096`
- `0129`
- `0131`
- `0137`
- `0143`
- `0145`
- `0146`
- `0151`
- `0157`
- `0158`
- `0166`
- `0176`
- `0177`
- `0181`

## One-click ablation example

```bash
python E:\Code\PRCV\ultralytics\experiments_tool\run_ablation_suite.py \
  --detector E:\Code\PRCV\ultralytics\prcv_detect\yolo26_visible_7168\weights\best.pt \
  --detector-mode full \
  --raw-match-root E:\Dataset\716\raw-match \
  --output-root E:\Code\PRCV\ultralytics\output\ablation_suite_full \
  --imgsz 1280 \
  --device 0 \
  --workers 2
```

This will:

1. cache detections to `det_cache`
2. run the default four ablation tracker YAMLs
3. evaluate each result directory
4. write a `suite_summary.json`

## One-click ablation with TrackEval

```bash
python E:\Code\PRCV\ultralytics\experiments_tool\run_ablation_suite.py \
  --detector E:\Code\PRCV\ultralytics\prcv_detect\yolo26_visible_7168\weights\best.pt \
  --detector-mode full \
  --raw-match-root E:\Dataset\716\raw-match \
  --output-root E:\Code\PRCV\ultralytics\output\ablation_suite_full \
  --imgsz 1280 \
  --device 0 \
  --workers 2 \
  --run-trackeval \
  --trackeval-root E:\Code\PRCV\TrackEval
```

## Standalone TrackEval example

```bash
python E:\Code\PRCV\ultralytics\experiments_tool\evaluate_mot_results_trackeval.py \
  --result-root E:\Code\PRCV\ultralytics\output\ablation_suite_full\sealinetracker_full \
  --gt-root E:\Dataset\716\raw-match \
  --trackeval-root E:\Code\PRCV\TrackEval
```

## Custom tracker YAML list

```bash
python E:\Code\PRCV\ultralytics\experiments_tool\run_ablation_suite.py \
  --detector E:\Code\PRCV\ultralytics\prcv_detect\yolo26_visible_7168\weights\best.pt \
  --trackers E:\Code\PRCV\ultralytics\ultralytics\cfg\trackers\ablations\bytetrack_baseline.yaml,E:\Code\PRCV\ultralytics\ultralytics\cfg\trackers\ablations\sealinetracker_full.yaml \
  --raw-match-root E:\Dataset\716\raw-match \
  --output-root E:\Code\PRCV\ultralytics\output\ablation_custom \
  --imgsz 1280 \
  --device 0
```

## Manual evaluation example

```bash
python E:\Code\PRCV\ultralytics\experiments_tool\evaluate_mot_results.py \
  --result-root E:\Code\PRCV\ultralytics\output\mot_results\train38_raw_match \
  --gt-root E:\Dataset\716\raw-match
```
