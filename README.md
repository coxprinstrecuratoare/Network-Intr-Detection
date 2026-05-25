# Network Intrusion Detection

This project trains and compares machine learning models for network intrusion detection using the NSL-KDD dataset.

The task is to classify network connections into five categories:

- Normal
- DoS
- Probe
- R2L
- U2R

The main evaluation metric is **macro F1-score**, because the dataset is imbalanced and rare classes such as R2L and U2R are important.

## Repository Structure

~~~text
Network-Intr-Detection/
│
├── README.md
├── requirements.txt
├── report.md
├── NIDS-Konya-Scholz.pdf
│
├── Results/
│   ├── final_confusion_matrix.png
│   │
│   ├── Random_Forest/
│   │   ├── random_forest.py
│   │   ├── requirements.txt
│   │   ├── README.md
│   │   └── results/
│   │
│   ├── XGboost_default/
│   │   ├── experiment1.py
│   │   └── results.txt
│   │
│   ├── XG_SMOTE/
│   │   ├── experiment2.py
│   │   └── results.txt
│   │
│   ├── XG_SMOTETomek/
│   │   ├── experiment3.py
│   │   └── results.txt
│   │
│   ├── XG_SMOTE_Threshold/
│   │   ├── experiment4.py
│   │   └── results.txt
│   │
│   ├── LightGBM_SMOTE/
│   │   ├── experiment5.py
│   │   └── results.txt
│   │
│   ├── Feature_SMOTETomek_Threshold/
│   │   ├── experiment6.py
│   │   └── results.txt
│   │
│   └── Hyperpar_tuning+exp6/
│       ├── experiment7.py
│       └── results.txt
~~~

## Setup

Install the required Python libraries:

~~~bash
pip install -r requirements.txt
~~~

Python 3.9 or later is recommended.

## How to Run

The project contains separate experiment tracks. Each track can be run separately.

### Random Forest Experiments

~~~bash
python Results/Random_Forest/random_forest.py
~~~

This script runs the Random Forest-based experiments and saves classification reports, confusion matrices and summary scores under:

~~~text
Results/Random_Forest/results/
~~~

### XGBoost / LightGBM Experiments

The boosting experiments are stored in separate folders under `Results`.

Run individual experiments with:

~~~bash
python Results/XGboost_default/experiment1.py
python Results/XG_SMOTE/experiment2.py
python Results/XG_SMOTETomek/experiment3.py
python Results/XG_SMOTE_Threshold/experiment4.py
python Results/LightGBM_SMOTE/experiment5.py
python Results/Feature_SMOTETomek_Threshold/experiment6.py
python Results/Hyperpar_tuning+exp6/experiment7.py
~~~

The final selected model is documented in `report.md`.

The final confusion matrix image is saved as:

~~~text
Results/final_confusion_matrix.png
~~~

## Experiments

The project compares two main experiment tracks.

### Random Forest Track

The Random Forest track includes:

- Default Random Forest
- Random Forest with `class_weight='balanced'`
- Tuned balanced Random Forest
- Random oversampling + Random Forest
- SMOTENC + Random Forest
- Balanced Random Forest
- Balanced Random Forest with 500 trees

### Boosting Track

The boosting track includes:

- XGBoost baseline
- XGBoost + SMOTE
- XGBoost + SMOTETomek
- XGBoost + SMOTE + threshold tuning
- LightGBM + SMOTE
- Feature engineering + SMOTETomek + threshold tuning
- RandomizedSearchCV tuning

## Results Summary

| Experiment | Approach | Macro F1 |
|---|---|---:|
| Random Forest baseline | Default Random Forest | 0.4911 |
| Random Forest best | Balanced Random Forest with 500 trees | 0.6053 |
| XGBoost baseline | Default XGBoost | 0.5599 |
| XGBoost + SMOTE | XGBoost with SMOTE | 0.6273 |
| XGBoost + SMOTETomek | XGBoost with SMOTETomek | 0.6485 |
| XGBoost + SMOTE + thresholds | XGBoost with SMOTE and threshold tuning | 0.6743 |
| LightGBM + SMOTE | LightGBM with SMOTE | 0.6034 |
| Feature engineering + SMOTETomek + thresholds | Final selected model | 0.6752 |
| RandomizedSearchCV + feature engineering | Additional tuning experiment | 0.6698 |

The final model was selected based on macro F1-score and minority-class detection. The full explanation, classification report and confusion matrix are included in `report.md`.

## Final Model

The final selected model was the boosting-based model using:

- feature engineering
- SMOTETomek
- threshold tuning

Final KDDTest+ result:

| Metric | Score |
|---|---:|
| Accuracy | 0.7901 |
| Macro F1-score | 0.6752 |

The final confusion matrix is available at:

~~~text
Results/final_confusion_matrix.png
~~~

## Notes on Reproducibility

- The NSL-KDD train and test files are loaded directly from the public dataset URLs.
- Random seeds are set where applicable.
- The repository contains separate scripts for each experiment track.
- The final report includes a comparison between training validation or cross-validation performance and final KDDTest+ performance.
- The test set is used only for final evaluation.
