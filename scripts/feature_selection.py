from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.linear_model import LogisticRegression


RANDOM_STATE = 42
TARGET_COL = "icu_death_flag"

BASE_DIR = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = BASE_DIR / "data" / "intermediate"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_path = INTERMEDIATE_DIR / "processed_train.csv"
    test_path = INTERMEDIATE_DIR / "processed_test.csv"

    print(f"Loading processed train data: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"Loading processed test data: {test_path}")
    test_df = pd.read_csv(test_path)

    y_train = train_df[TARGET_COL].astype(int)
    X_train = train_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL].astype(int)
    X_test = test_df.drop(columns=[TARGET_COL])

    print(f"Initial train features: {X_train.shape[1]}")

    print("\nStep 1: Variance thresholding (fit on train only)")
    vt = VarianceThreshold(threshold=0.01)
    X_train_vt_arr = vt.fit_transform(X_train)
    X_test_vt_arr = vt.transform(X_test)
    vt_cols = X_train.columns[vt.get_support()].tolist()

    X_train_vt = pd.DataFrame(X_train_vt_arr, columns=vt_cols)
    X_test_vt = pd.DataFrame(X_test_vt_arr, columns=vt_cols)
    print(f"Features after variance thresholding: {X_train_vt.shape[1]}")

    print("\nStep 2: L1 selection (fit on train only)")
    lr_l1 = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        random_state=RANDOM_STATE,
        max_iter=2000,
        class_weight="balanced",
    )
    selector_l1 = SelectFromModel(lr_l1, prefit=False)
    selector_l1.fit(X_train_vt, y_train)
    l1_support = selector_l1.get_support()
    X_l1_cols = X_train_vt.columns[l1_support].tolist()
    print(f"Features selected by L1: {len(X_l1_cols)}")

    print("\nStep 3: RF importance (fit on train only)")
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=1,
        class_weight="balanced_subsample",
    )
    rf.fit(X_train_vt, y_train)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    cumulative_importance = np.cumsum(importances[indices])

    n_top_rf = int(np.searchsorted(cumulative_importance, 0.90) + 1)
    n_top_rf = max(min(n_top_rf, 50), 20)
    n_top_rf = min(n_top_rf, X_train_vt.shape[1])

    rf_support = np.zeros(X_train_vt.shape[1], dtype=bool)
    rf_support[indices[:n_top_rf]] = True
    X_rf_cols = X_train_vt.columns[rf_support].tolist()
    print(f"Features selected by RF (top {n_top_rf}): {len(X_rf_cols)}")

    print("\nStep 4: Ensemble selection")
    selected_features = sorted(list(set(X_l1_cols).intersection(set(X_rf_cols))))
    print(f"Intersection size: {len(selected_features)}")

    if len(selected_features) < 15:
        print("Intersection too small; using union.")
        selected_features = sorted(list(set(X_l1_cols).union(set(X_rf_cols))))
        print(f"Union size: {len(selected_features)}")

    selected_features_path = METRICS_DIR / "selected_features.csv"
    pd.DataFrame({"selected_features": selected_features}).to_csv(
        selected_features_path, index=False
    )

    X_train_final = X_train_vt[selected_features]
    X_test_final = X_test_vt[selected_features]

    reduced_train_df = pd.concat([X_train_final, y_train.rename(TARGET_COL)], axis=1)
    reduced_test_df = pd.concat([X_test_final, y_test.rename(TARGET_COL)], axis=1)

    reduced_train_path = INTERMEDIATE_DIR / "reduced_train.csv"
    reduced_test_path = INTERMEDIATE_DIR / "reduced_test.csv"
    reduced_train_df.to_csv(reduced_train_path, index=False)
    reduced_test_df.to_csv(reduced_test_path, index=False)

    summary_df = pd.DataFrame(
        {
            "stage": ["initial", "after_variance", "final_selected"],
            "feature_count": [X_train.shape[1], X_train_vt.shape[1], len(selected_features)],
        }
    )
    summary_path = METRICS_DIR / "feature_selection_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved selected feature list: {selected_features_path}")
    print(f"Saved reduced train set: {reduced_train_path}")
    print(f"Saved reduced test set: {reduced_test_path}")
    print(f"Saved feature selection summary: {summary_path}")

    top_n = min(20, len(indices))
    plt.figure(figsize=(10, 8))
    plt.title("Top Feature Importances (Random Forest, train only)")
    plt.barh(range(top_n), importances[indices[:top_n]][::-1], color="b", align="center")
    plt.yticks(range(top_n), [X_train_vt.columns[i] for i in indices[:top_n]][::-1])
    plt.xlabel("Gini Importance")
    plt.tight_layout()

    feature_importance_path = FIGURES_DIR / "feature_importances.png"
    plt.savefig(feature_importance_path, dpi=300)
    print(f"Saved feature importance plot: {feature_importance_path}")


if __name__ == "__main__":
    main()
