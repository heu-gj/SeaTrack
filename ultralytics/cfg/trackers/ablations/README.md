SealineTracker ablation tracker YAMLs.

Recommended usage:

- `bytetrack_baseline.yaml`
  - Pure ByteTrack baseline.
- `sealinetracker_full.yaml`
  - Paper main version with innovation 1 + innovation 2 enabled.
- `sealinetracker_innov1_only.yaml`
  - Only Relative-Scale Adaptive IoU enabled.
- `sealinetracker_no_adaptive_iou.yaml`
  - Disable innovation 1 while keeping Sea-Geo Cost enabled.
- `sealinetracker_no_sea_geo_cost.yaml`
  - Disable innovation 2 and fall back to Adaptive IoU cost.
- `sealinetracker_no_direction_center.yaml`
  - Disable direction-aware center weighting.
- `sealinetracker_no_asymmetric_scale.yaml`
  - Disable asymmetric width-height scale weighting.
- `sealinetracker_no_scale_aware_weights.yaml`
  - Disable target-scale-dependent geometry weights.

Typical command:

```bash
python E:\Code\PRCV\ultralytics\experiments_tool\sealine_mot_runner.py \
  --detector E:\path\to\best.pt \
  --tracker E:\Code\PRCV\ultralytics\ultralytics\cfg\trackers\ablations\sealinetracker_full.yaml \
  --sequence-names 0001,0002,0003 \
  --raw-match-root E:\Dataset\716\raw-match \
  --out-dir E:\Code\PRCV\ultralytics\output\mot_results\ablation_main \
  --imgsz 1280 \
  --device 0
```
