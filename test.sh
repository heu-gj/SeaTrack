python E:\Code\PRCV\ultralytics\experiments_tool\sealine_mot_runner.py --detector E:\Code\PRCV\ultralytics\prcv_detect\yolo26_visible_7168\weights\best.pt --detector-mode full --tracker E:\Code\PRCV\ultralytics\ultralytics\cfg\trackers\sealinetracker.yaml --sequence-names 0011,0040,0081,0086,0092,0129,0137,0157,0188 --raw-match-root E:\Dataset\716\raw-match --out-dir E:\Code\PRCV\ultralytics\output\mot_results\raw_match_debug --imgsz 1280 --device 0 --save-vis

python E:\Code\PRCV\ultralytics\experiments_tool\evaluate_mot_results.py --result-root E:\Code\PRCV\ultralytics\output\mot_results\raw_match_debug --gt-root E:\Dataset\716\raw-match

python E:\Code\PRCV\ultralytics\experiments_tool\run_ablation_suite.py  --detector E:\Code\PRCV\ultralytics\prcv_detect\yolo26_visible_7168\weights\best.pt  --detector-mode full  --raw-match-root E:\Dataset\716\raw-match  --output-root E:\Code\PRCV\ultralytics\output\ablation_main  --imgsz 1280  --device 0  --workers 2

python E:\Code\PRCV\ultralytics\experiments_tool\run_fine_grained_ablation_suite.py --detector E:\Code\PRCV\ultralytics\prcv_detect\yolo26_visible_7168\weights\best.pt --detector-mode full --raw-match-root E:\Dataset\716\raw-match --output-root E:\Code\PRCV\ultralytics\output\ablation_sgl --imgsz 1280 --device 0 --workers 2

python E:\Code\PRCV\ultralytics\experiments_tool\run_boxmot_baseline_suite.py --detector E:\Code\PRCV\ultralytics\prcv_detect\yolo26_visible_7168\weights\best.pt --detector-mode full --raw-match-root E:\Dataset\716\raw-match --cache-dir E:\Code\PRCV\ultralytics\output\ablation_main\det_cache --output-root E:\Code\PRCV\ultralytics\output\boxmot_baselines --imgsz 1280 --device 0 --workers 2 --default-reid-weights osnet_x0_25_msmt17.pt
