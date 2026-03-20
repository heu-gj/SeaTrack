SeaTrack paper ablation tracker YAMLs.

Retained paper-oriented configs:

- `bytetrack_baseline.yaml`
  - Pure ByteTrack baseline.
- `seatrack_rsa_iou.yaml`
  - Only `RSA-IoU` enabled.
- `seatrack_sgl.yaml`
  - Only `SGL` enabled.
- `seatrack_rsa_iou_sgl.yaml`
  - Full paper version: `RSA-IoU + SGL`.

Fine-grained `SGL` configs:

- `seatrack_sgl_wo_scl.yaml`
  - `SGL` without `SCL`.
- `seatrack_sgl_wo_srl.yaml`
  - `SGL` without `SRL`.
- `seatrack_sgl_wo_ssl.yaml`
  - `SGL` without `SSL`.

Removed from the current paper set:

- duplicate legacy `sealinetracker_no_*` variants
- `scale-aware weighting` ablation variants

Typical commands:

```bash
python E:\Code\PRCV\ultralytics\experiments_tool\run_ablation_suite.py \
  --detector E:\path\to\best.pt \
  --raw-match-root E:\Dataset\716\raw-match \
  --output-root E:\Code\PRCV\ultralytics\output\ablation_main \
  --imgsz 1280 \
  --device 0
```

```bash
python E:\Code\PRCV\ultralytics\experiments_tool\run_fine_grained_ablation_suite.py \
  --detector E:\path\to\best.pt \
  --raw-match-root E:\Dataset\716\raw-match \
  --output-root E:\Code\PRCV\ultralytics\output\ablation_sgl \
  --imgsz 1280 \
  --device 0
```
