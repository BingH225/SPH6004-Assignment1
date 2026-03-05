# SPH6004 Assignment 1

This is my SPH6004 Assignment 1 project.

I use 3 models:
- Logistic Regression
- Random Forest
- SVM (RBF)

## Folders
- `scripts/`: python scripts for preprocessing, feature selection, training, and report generation
- `outputs/metrics/`: model metric csv files
- `outputs/figures/`: plots (ROC curve and feature importance)

## How to run
Run full pipeline:
```powershell
$env:PYTHONNOUSERSITE='1'; python scripts/run_pipeline.py
```

## Current result (latest)
- Feature count: 234 -> 174 -> 46
- Best AUC-ROC: Random Forest
- Best Recall: Logistic Regression
