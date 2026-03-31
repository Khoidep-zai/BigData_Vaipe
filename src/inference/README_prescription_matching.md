# Prescription Matching

This module provides a lightweight inference-ready post-processing step for the new task:

- Input: 1 prescription image + N pill images.
- Output: for each pill image, list of detections that are `in_prescription` and `out_of_prescription` with positions.

## Data source

Run this first to generate CSV artifacts:

```bash
python data/Loc_du_lieu.py
```

Required CSV files:

- `data/csv/Prescription_Image_Index.csv`
- `data/csv/Prescription_Pill_Annotations.csv`
- `data/csv/Medicine_Details_Training.csv`

## Quick run via train_cli

```bash
python train_cli.py --mode prescription_match --prescription-image data/images/public_train/prescription/image/VAIPE_P_TRAIN_0.png --pill-images data/images/public_train/pill/image/VAIPE_P_0_0.jpg data/images/public_train/pill/image/VAIPE_P_0_1.jpg
```

Save JSON output:

```bash
python train_cli.py --mode prescription_match --prescription-image data/images/public_train/prescription/image/VAIPE_P_TRAIN_0.png --pill-images data/images/public_train/pill/image/VAIPE_P_0_0.jpg data/images/public_train/pill/image/VAIPE_P_0_1.jpg --output-json models/reports/latest/prescription_match_example.json
```

Save both JSON and CSV, and print terminal summary:

```bash
python train_cli.py --mode prescription_match --prescription-image data/images/public_train/prescription/image/VAIPE_P_TRAIN_0.png --pill-images data/images/public_train/pill/image/VAIPE_P_0_0.jpg data/images/public_train/pill/image/VAIPE_P_0_1.jpg --output-json models/reports/latest/prescription_match_example.json --output-csv models/reports/latest/prescription_match_example.csv --pretty
```

## Quick run

```bash
python -m src.inference.prescription_matching \
  --prescription-image data/images/public_train/prescription/image/VAIPE_P_TRAIN_0.png \
  --pill-images data/images/public_train/pill/image/VAIPE_P_0_0.jpg data/images/public_train/pill/image/VAIPE_P_0_1.jpg \
  --pretty
```

Optional output file:

```bash
python -m src.inference.prescription_matching \
  --prescription-image data/images/public_train/prescription/image/VAIPE_P_TRAIN_0.png \
  --pill-images data/images/public_train/pill/image/VAIPE_P_0_0.jpg \
  --output-json models/reports/latest/prescription_match_example.json \
  --output-csv models/reports/latest/prescription_match_example.csv
```
