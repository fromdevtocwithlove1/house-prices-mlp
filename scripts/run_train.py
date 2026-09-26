"""CLI entry point for baseline, experiments, and selected final model."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train import run_all_experiments, run_baseline, run_final


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all-experiments", action="store_true")
    group.add_argument("--final", action="store_true")
    args = parser.parse_args()
    if args.all_experiments:
        print(run_all_experiments().to_string(index=False))
    elif args.final:
        print(run_final())
    else:
        print(run_baseline())


if __name__ == "__main__":
    main()
