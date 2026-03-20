# Ablation Suite Usage

## Scripts

- `E:\Code\PRCV\ultralytics\experiments_tool\run_ablation_suite.py`
- `E:\Code\PRCV\ultralytics\experiments_tool\run_fine_grained_ablation_suite.py`
- `E:\Code\PRCV\ultralytics\experiments_tool\evaluate_mot_results.py`
- `E:\Code\PRCV\ultralytics\experiments_tool\evaluate_mot_results_trackeval.py`

## Current Main Ablation Set

- `bytetrack_baseline.yaml`
- `seatrack_rsa_iou.yaml`
- `seatrack_sgl.yaml`
- `seatrack_rsa_iou_sgl.yaml`

## Current Fine-Grained Ablation Set

- `seatrack_sgl.yaml`
- `seatrack_sgl_wo_scl.yaml`
- `seatrack_sgl_wo_srl.yaml`
- `seatrack_sgl_wo_ssl.yaml`

Interpretation:

- `seatrack_sgl.yaml`: `SGL full = SCL + SRL + SSL`
- `seatrack_sgl_wo_scl.yaml`: `w/o SCL`
- `seatrack_sgl_wo_srl.yaml`: `w/o SRL`
- `seatrack_sgl_wo_ssl.yaml`: `w/o SSL`

## Main Ablation Example

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

## Fine-Grained Ablation Example

```bash
python E:\Code\PRCV\ultralytics\experiments_tool\run_fine_grained_ablation_suite.py \
  --detector E:\Code\PRCV\ultralytics\prcv_detect\yolo26_visible_7168\weights\best.pt \
  --detector-mode full \
  --raw-match-root E:\Dataset\716\raw-match \
  --output-root E:\Code\PRCV\ultralytics\output\fine_grained_ablation_suite_innov2 \
  --imgsz 1280 \
  --device 0 \
  --workers 2
```

## Standard Evaluation Example

```bash
python E:\Code\PRCV\ultralytics\experiments_tool\evaluate_mot_results.py \
  --result-root E:\Code\PRCV\ultralytics\output\ablation_suite_full\sealinetracker_full \
  --gt-root E:\Dataset\716\raw-match
```

## TrackEval Example

```bash
python E:\Code\PRCV\ultralytics\experiments_tool\evaluate_mot_results_trackeval.py \
  --result-root E:\Code\PRCV\ultralytics\output\ablation_suite_full\sealinetracker_full \
  --gt-root E:\Dataset\716\raw-match \
  --trackeval-root E:\Code\PRCV\TrackEval
```
