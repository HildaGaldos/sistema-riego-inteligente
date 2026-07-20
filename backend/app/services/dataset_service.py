"""Service used by the manual upload endpoint."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ML_SRC = ROOT / "ml" / "src"
if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))

from irrigation_ml.data import data_quality_report, load_and_validate, read_table, validate_dataframe


def save_uploaded_dataset(uploaded_file, data_dir: str | Path = "data/raw") -> dict:
    name = Path(uploaded_file.filename or "dataset.csv").name
    suffix = Path(name).suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise ValueError("Formato no soportado. Suba un archivo CSV, XLSX o XLS.")
    target_dir = Path(data_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"uploaded_{time.strftime('%Y%m%d_%H%M%S')}{suffix}"
    with target.open("wb") as destination:
        shutil.copyfileobj(uploaded_file.file, destination)
    try:
        validation = validate_dataframe(read_table(target))
    except Exception:
        target.unlink(missing_ok=True)
        raise
    quality = data_quality_report(validation)
    Path("artifacts/metrics").mkdir(parents=True, exist_ok=True)
    Path("artifacts/metrics/data_quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    processed_dir = target_dir.parent / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = processed_dir / "cleaned_dataset.csv"
    validation.dataframe.to_csv(cleaned_path, index=False)
    quality["cleaned_dataset"] = str(cleaned_path)
    Path("artifacts/metrics/data_quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"filename": target.name, "path": str(target), "cleaned_path": str(cleaned_path), "quality": quality, "warnings": validation.warnings}


def dataset_status(data_dir: str | Path = "data/raw") -> dict:
    root = Path(data_dir)
    files = sorted([*root.glob("*.csv"), *root.glob("*.xlsx"), *root.glob("*.xls")], key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return {"available": False, "message": "No hay dataset. Súbalo manualmente desde Datos → Cargar dataset."}
    try:
        validation = load_and_validate(files[0])
        return {"available": True, "filename": files[0].name, "quality": data_quality_report(validation), "warnings": validation.warnings}
    except Exception as exc:
        return {"available": False, "filename": files[0].name, "message": str(exc)}
