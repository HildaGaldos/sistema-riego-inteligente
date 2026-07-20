from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml" / "src"))

from irrigation_ml.data import load_and_validate
from irrigation_ml.reports import generate_reports

if __name__ == "__main__":
    artifacts = ROOT / "artifacts"
    validation = load_and_validate()
    comparison = pd.read_csv(artifacts / "metrics" / "model_comparison.csv")
    cv = pd.read_csv(artifacts / "metrics" / "cv_results.csv")
    generate_reports(artifacts, validation.dataframe, comparison, cv)
    print("Reportes PDF, DOCX y XLSX generados")
