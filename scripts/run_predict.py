"""CLI entry point for a validated Kaggle submission."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.predict import make_submission


if __name__ == "__main__":
    print(make_submission())
