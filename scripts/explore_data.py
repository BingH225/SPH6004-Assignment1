from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "Assignment1_mimic_dataset.csv"
OUTPUT_PATH = BASE_DIR / "docs" / "data_info.txt"


def main() -> None:
    df = pd.read_csv(RAW_DATA_PATH)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        f.write("--- Columns ---\n")
        f.write(", ".join(df.columns.tolist()) + "\n\n")

        f.write("--- Possible Target Variables ---\n")
        for col in df.columns:
            if any(
                keyword in col.lower()
                for keyword in [
                    "predict",
                    "target",
                    "discharg",
                    "death",
                    "mortality",
                    "status",
                    "label",
                    "stay",
                    "icu",
                    "los",
                    "surviv",
                    "outcome",
                ]
            ):
                f.write(f"Column: {col}\n")
                f.write(f"Type: {df[col].dtype}\n")
                f.write(f"Unique values (up to 10): {df[col].unique()[:10]}\n")
                f.write("-" * 20 + "\n")


if __name__ == "__main__":
    main()
