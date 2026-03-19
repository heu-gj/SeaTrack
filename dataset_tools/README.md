# Dataset Tools

This folder contains dataset preparation utilities for the PRCV project.

Current script:

- `prepare_716_visible_dataset.py`
- `prepare_716_visible_slice_dataset.py`
- `prepare_716_visible_square_slice_dataset.py`

Purpose:

- Read `E:\Dataset\716\raw-match`
- Use only visible-light images and visible GT
- Split sequences into train/val/test at the sequence level
- Export an Ultralytics-compatible detection dataset
- Write dataset documentation into the generated dataset directory
- Build a derived horizontal-slice dataset from the visible-light export while clipping labels correctly
- Build a derived square-slice dataset where each crop uses image height as the square side
