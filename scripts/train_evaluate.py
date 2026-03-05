from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neural_network import MLPClassifier


RANDOM_STATE = 42
TARGET_COL = "icu_death_flag"

BASE_DIR = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = BASE_DIR / "data" / "intermediate"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_path = INTERMEDIATE_DIR / "reduced_train.csv"
    test_path = INTERMEDIATE_DIR / "reduced_test.csv"

    print(f"Loading reduced train data: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"Loading reduced test data: {test_path}")
    test_df = pd.read_csv(test_path)

    y_train = train_df[TARGET_COL].astype(int)
    X_train = train_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL].astype(int)
    X_test = test_df.drop(columns=[TARGET_COL])

    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")

    models = {
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=2000,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=1,
            class_weight="balanced_subsample",
        ),
        "Multi-Layer Perceptron": MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=500,
            random_state=RANDOM_STATE,
        ),
    }

    results = []

    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], "k--", label="Random Chance")

    print("\nTraining and evaluating models...")
    for name, model in models.items():
        print(f"-> Training {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)

        results.append(
            {
                "Model": name,
                "Accuracy": acc,
                "Balanced_Accuracy": bal_acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1,
                "AUC-ROC": auc,
                "PR-AUC": pr_auc,
            }
        )
        print(
            f"{name} -> AUC-ROC: {auc:.4f}, Recall: {rec:.4f}, "
            f"PR-AUC: {pr_auc:.4f}"
        )

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    results_df = pd.DataFrame(results).sort_values(by="AUC-ROC", ascending=False)
    print("\n--- Model Comparison Summary ---")
    print(results_df.to_string(index=False))

    metrics_path = METRICS_DIR / "metrics_summary.csv"
    results_df.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics to: {metrics_path}")

    plt.title("Receiver Operating Characteristic (ROC) Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    roc_path = FIGURES_DIR / "roc_curves.png"
    plt.savefig(roc_path, dpi=300)
    print(f"Saved ROC curves to: {roc_path}")


if __name__ == "__main__":
    main()
