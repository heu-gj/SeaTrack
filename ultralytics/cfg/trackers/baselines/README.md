BoxMOT baseline tracker wrapper YAMLs.

These YAMLs are used by:

- `E:\Code\PRCV\ultralytics\experiments_tool\run_boxmot_baseline_suite.py`

Retained wrappers:

- `boxmot_ocsort.yaml`
- `boxmot_botsort.yaml`
- `boxmot_deepocsort.yaml`
- `boxmot_strongsort.yaml`
- `boxmot_boosttrack.yaml`
- `boxmot_hybridsort.yaml`
- `boxmot_sfsort.yaml`

Notes:

- Motion-only trackers such as `ocsort` and `sfsort` can run without ReID weights.
- ReID-based trackers such as `botsort`, `deepocsort`, and `strongsort` require either:
  - `reid_weights` set inside the wrapper YAML, or
  - `--default-reid-weights` passed to the runner.
