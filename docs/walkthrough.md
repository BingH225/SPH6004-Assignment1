# SPH6004 Assignment 1 Walkthrough (Updated)

## 1. Objective
Predict `icu_death_flag` (ICU in-unit mortality) from static admission snapshot features in the provided MIMIC-derived dataset.

Clinical interpretation: this is an imbalanced clinical dataset with lower death prevalence, which is expected and normal.

## 2. Leakage-Free Pipeline
The workflow was updated based on review findings to eliminate data leakage.

1. Split data first (stratified 80/20)
2. Fit preprocessing on **train only**:
   - Drop IDs, timestamps, and leakage columns (`hospital_expire_flag`, `los`)
   - Numerical median imputation + scaling
   - Categorical mode imputation + one-hot encoding
3. Fit feature selection on **train only**:
   - Variance threshold
   - L1 logistic selection
   - Random forest importance
   - Ensemble (intersection/union strategy)
4. Train/evaluate models on reduced features using hold-out test set.

## 3. Output Locations
- Intermediate data: `data/intermediate/`
- Metrics: `outputs/metrics/`
- Figures: `outputs/figures/`
- Report: `reports/SPH6004_Assignment1_Report.docx`

## 4. Latest Results
### Data split and class ratio
- Train rows: 52,292
- Test rows: 13,074
- Death rate: train 8.68%, test 8.67%

### Feature reduction
- Initial features: 225
- After variance threshold: 165
- Final selected features: 50

### Model metrics (test set)
| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1 | AUC-ROC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.9279 | 0.6078 | 0.8117 | 0.2205 | 0.3467 | **0.9176** | **0.5977** |
| Logistic Regression | 0.8366 | **0.8244** | 0.3235 | **0.8095** | 0.4622 | 0.9055 | 0.5767 |
| Multi-Layer Perceptron | 0.9222 | 0.7144 | 0.5627 | 0.4630 | **0.5080** | 0.8794 | 0.5390 |

## 5. How to reproduce
```powershell
$env:PYTHONNOUSERSITE='1'; python scripts/run_pipeline.py
python scripts/generate_report_docx.py
```
