# SPH6004 Assignment 1 (Leakage-Free Version)

## Project Structure
- `data/raw/`: original dataset (`Assignment1_mimic_dataset.csv`)
- `data/intermediate/`: pipeline intermediate files (processed/reduced train-test sets)
- `scripts/`: all runnable scripts
- `outputs/metrics/`: metrics and feature-selection summary CSVs
- `outputs/figures/`: generated plots
- `docs/`: task text and walkthrough
- `reports/`: final report files (`.docx`)
- `archive/`: legacy outputs kept for reference

## How To Run
Use this to avoid local user-package conflicts (NumPy 2 user-site issue):

```powershell
$env:PYTHONNOUSERSITE='1'; python scripts/run_pipeline.py
```

Then generate report:

```powershell
python scripts/generate_report_docx.py
```

## Current Results (Latest Run)
- Train/Test split: 52,292 / 13,074 (stratified)
- Death rate: train 8.68%, test 8.67%
- Feature count: 225 -> 165 -> 50
- Best AUC-ROC: Random Forest (`0.9176`)
- Best Recall: Logistic Regression (`0.8095`)

## Key Method Update
All preprocessing and feature selection are now fitted **only on training data** and then applied to test data, removing data leakage from the previous workflow.
