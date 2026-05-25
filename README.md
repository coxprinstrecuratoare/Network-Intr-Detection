# Network Intrusion Detection

Machine learning project to classify network connections into 5 categories:
Normal, DoS, Probe, R2L and U2R using the NSL-KDD dataset.

## Dataset

NSL-KDD dataset loaded directly from GitHub - no manual download needed.
- Training set: 125,973 records
- Test set: 22,544 records

## Experiments

| Experiment | Approach | Macro F1 |
|---|---|---|
| 1 | XGBoost baseline | 0.5599 |
| 2 | XGBoost + SMOTE | 0.6273 |
| 3 | XGBoost + SMOTETomek | 0.6485 |
| 4 | XGBoost + SMOTE + Threshold Tuning | 0.6743 |
| 5 | LightGBM + SMOTE | 0.6034 |
| 6 | Feature Engineering + SMOTETomek + Threshold Tuning | 0.6752 |
| 7 | RandomizedSearchCV + Feature Engineering | 0.6689 |

## How to Run

Install dependencies:
```bash
pip install -r requirements.txt
```

Then run any experiment file:
```bash
python exp1_baseline.py
```

## Results

All confusion matrices and classification reports are in the `results/` folder,
organised by experiment.
