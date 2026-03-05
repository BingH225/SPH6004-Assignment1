from pathlib import Path
import subprocess
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = BASE_DIR / "scripts"

STEPS = [
    "preprocess.py",
    "feature_selection.py",
    "train_evaluate.py",
]


def run_step(step: str) -> None:
    script_path = SCRIPTS_DIR / step
    print(f"\n===== Running {script_path.name} =====")
    result = subprocess.run([sys.executable, str(script_path)], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {step} (exit code {result.returncode})")


if __name__ == "__main__":
    for step in STEPS:
        run_step(step)
    print("\nPipeline completed successfully.")
