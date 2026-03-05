from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TARGET_COL = "icu_death_flag"
TEST_SIZE = 0.2

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "Assignment1_mimic_dataset.csv"
INTERMEDIATE_DIR = BASE_DIR / "data" / "intermediate"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"


def main() -> None:
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    print("Initial shape:", df.shape)

    cols_to_drop = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "intime",
        "outtime",
        "deathtime",
        "hospital_expire_flag",
        "los",
    ]
    df_clean = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    if TARGET_COL not in df_clean.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    y = df_clean[TARGET_COL].astype(int)
    X = df_clean.drop(columns=[TARGET_COL])

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    categorical_cols = X_train_raw.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X_train_raw.select_dtypes(include=["int64", "float64"]).columns.tolist()
    print(f"Numerical columns: {len(numerical_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        sparse_threshold=0.0,
    )

    print("Fitting preprocessing pipeline on training set...")
    X_train_arr = preprocessor.fit_transform(X_train_raw)
    X_test_arr = preprocessor.transform(X_test_raw)

    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_feature_names = (
        cat_encoder.get_feature_names_out(categorical_cols).tolist()
        if categorical_cols
        else []
    )
    feature_names = numerical_cols + cat_feature_names

    X_train = pd.DataFrame(X_train_arr, columns=feature_names)
    X_test = pd.DataFrame(X_test_arr, columns=feature_names)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    train_df = pd.concat([X_train, y_train.rename(TARGET_COL)], axis=1)
    test_df = pd.concat([X_test, y_test.rename(TARGET_COL)], axis=1)

    train_path = INTERMEDIATE_DIR / "processed_train.csv"
    test_path = INTERMEDIATE_DIR / "processed_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    split_info = pd.DataFrame(
        {
            "split": ["train", "test"],
            "rows": [len(train_df), len(test_df)],
            "death_rate": [y_train.mean(), y_test.mean()],
        }
    )
    split_path = METRICS_DIR / "split_summary.csv"
    split_info.to_csv(split_path, index=False)

    print(f"Saved processed training data to: {train_path}")
    print(f"Saved processed testing data to: {test_path}")
    print(f"Saved split summary to: {split_path}")


if __name__ == "__main__":
    main()
