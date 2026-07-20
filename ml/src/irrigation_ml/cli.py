"""Command-line entry point for training and EDA."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from .training import generate_eda_artifacts, run_training

load_dotenv(Path(__file__).resolve().parents[3] / ".env")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart Irrigation AI")
    parser.add_argument("--fast", action="store_true", help="smoke test reproducible")
    parser.add_argument("--full", action="store_true", help="full configuration")
    parser.add_argument("--eda-only", action="store_true")
    parser.add_argument("--data", default=None)
    parser.add_argument("--folds", type=int, default=int(os.getenv("CV_FOLDS", "5")))
    args = parser.parse_args()
    fast = not args.full
    if args.eda_only:
        from .data import load_and_validate
        validation = load_and_validate(args.data)
        generate_eda_artifacts(validation.dataframe, __import__("pathlib").Path(os.getenv("ARTIFACTS_DIR", "artifacts")))
        print("EDA generada correctamente")
        return
    print(run_training(data_path=args.data, folds=args.folds, fast=fast, seed=int(os.getenv("SEED", "42"))))


if __name__ == "__main__":
    main()
