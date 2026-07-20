"""Rebuild statistics and reports from an existing trained run without retraining."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml" / "src"))

from irrigation_ml.data import data_quality_report, discover_dataset, load_and_validate  # noqa: E402
from irrigation_ml.evaluation import statistical_tests  # noqa: E402
from irrigation_ml.reports import generate_reports  # noqa: E402


def main() -> None:
    artifacts = ROOT / "artifacts"
    cv_path = artifacts / "metrics" / "cv_results.csv"
    comparison_path = artifacts / "metrics" / "model_comparison.csv"
    if not cv_path.exists() or not comparison_path.exists():
        raise SystemExit("Faltan cv_results.csv o model_comparison.csv; ejecute primero el entrenamiento.")
    cv = pd.read_csv(cv_path)
    stats = statistical_tests(cv, {})
    (artifacts / "metrics" / "statistical_tests.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    validation = load_and_validate(discover_dataset(ROOT / "data" / "raw"))
    quality = data_quality_report(validation)
    quality_path = artifacts / "metrics" / "data_quality.json"
    quality_path.write_text(json.dumps(quality, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    generate_reports(artifacts, validation.dataframe, pd.read_csv(comparison_path), cv)
    friedman = stats.get("friedman", {})
    print(json.dumps({"t_paired": len(stats.get("paired_t", [])), "wilcoxon": len(stats.get("wilcoxon", [])), "friedman": friedman}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
