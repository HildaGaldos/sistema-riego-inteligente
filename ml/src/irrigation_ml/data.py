"""Dataset discovery, manual-upload validation and leakage-safe preprocessing."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REQUIRED_COLUMNS = ["temperature", "humidity", "moi", "soil_type", "crop_stage", "result"]
NUMERIC_COLUMNS = ["temperature", "humidity", "moi"]
CATEGORICAL_COLUMNS = ["soil_type", "crop_stage"]
OPTIONAL_COLUMNS = ["crop_id", "crop_type", "crop"]


def _slug(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


ALIASES = {
    "temperature": {"temp", "temperature", "ambient_temperature", "air_temperature"},
    "humidity": {"humidity", "relative_humidity", "air_humidity"},
    "moi": {"moi", "moisture_index", "soil_moisture_index", "soil_moisture"},
    "soil_type": {"soil_type", "soiltype", "soil"},
    "crop_stage": {"seedling_stage", "crop_stage", "cropstage", "phenological_stage", "stage"},
    "result": {"result", "target", "label", "irrigation_required", "irrigation"},
    "crop_id": {"crop_id", "cropid", "crop_identifier"},
    "crop_type": {"crop_type", "croptype", "crop"},
}


@dataclass
class DatasetValidation:
    dataframe: pd.DataFrame
    original_columns: list[str]
    normalized_columns: list[str]
    warnings: list[str]
    excluded_rows: int = 0


def discover_dataset(directory: str | Path = "data/raw") -> Path:
    """Return the newest supported table in a directory, with an actionable error."""

    root = Path(directory)
    files = sorted(
        [*root.glob("*.csv"), *root.glob("*.CSV"), *root.glob("*.xlsx"), *root.glob("*.xls")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError(
            f"No se encontró un CSV/XLSX en {root}. Suba el dataset desde Datos → Cargar dataset "
            "o ejecute scripts/download_data.py."
        )
    return files[0]


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("Formato no soportado. Use un archivo .csv, .xlsx o .xls.")


def normalize_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Normalize published Kaggle names while preserving a clear canonical schema."""

    used: set[str] = set()
    rename: dict[str, str] = {}
    for original in frame.columns:
        token = _slug(original)
        canonical = next((name for name, names in ALIASES.items() if token in names), token)
        if canonical in used:
            canonical = f"{canonical}_duplicate"
        used.add(canonical)
        rename[original] = canonical
    return frame.rename(columns=rename), list(rename.values())


def validate_dataframe(frame: pd.DataFrame) -> DatasetValidation:
    """Validate and coerce a manually uploaded table; never creates replacement rows."""

    if frame.empty:
        raise ValueError("El archivo está vacío; cargue el dataset público con filas de datos.")
    original = [str(c) for c in frame.columns]
    normalized, columns = normalize_columns(frame.copy())
    missing = [column for column in REQUIRED_COLUMNS if column not in normalized.columns]
    if missing:
        raise ValueError(
            "Faltan columnas requeridas: "
            + ", ".join(missing)
            + ". Aliases aceptados: temp/temperature, humidity, MOI, soil type, "
            "Seedling Stage/crop stage y result/target."
        )

    warnings: list[str] = []
    for column in NUMERIC_COLUMNS:
        before = normalized[column].isna().sum()
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        after = normalized[column].isna().sum()
        if after > before:
            warnings.append(f"{after - before} valores no numéricos se tratarán como faltantes en {column}.")

    for column in [*CATEGORICAL_COLUMNS, *[c for c in OPTIONAL_COLUMNS if c in normalized]]:
        # Keep missing categories as numpy.nan.  This is compatible with the
        # persisted SimpleImputer and also allows single-row inference when an
        # optional field such as crop_id is omitted from the form.
        values = normalized[column].astype("string")
        normalized[column] = values.astype(object).where(values.notna(), np.nan)

    result = normalized["result"].astype("string").str.strip().str.lower()
    mapping = {"yes": 1, "true": 1, "si": 1, "sí": 1, "required": 1, "1": 1, "no": 0, "false": 0, "0": 0, "not_required": 0}
    mapped = result.map(mapping)
    numeric_result = pd.to_numeric(result, errors="coerce")
    normalized["result"] = mapped.fillna(numeric_result)
    # The public dataset contains a third label: 2 means excess water. The
    # requested study is binary, so exclude it explicitly instead of silently
    # remapping it to irrigation=1.
    excess_water = normalized["result"] == 2
    excluded_rows = int(excess_water.sum())
    if excluded_rows:
        normalized = normalized.loc[~excess_water].copy()
        warnings.append(
            f"Se excluyeron {excluded_rows} filas con result=2 (exceso de agua) "
            "porque el estudio solicitado es binario: 0=no riego, 1=requiere riego."
        )

    invalid = normalized["result"].isna() | ~normalized["result"].isin([0, 1])
    if invalid.any():
        raise ValueError(
            f"La columna result debe ser binaria (0/1); hay {int(invalid.sum())} valores inválidos."
        )
    normalized["result"] = normalized["result"].astype(int)
    if normalized["result"].nunique() < 2:
        warnings.append("El dataset contiene una sola clase; no puede entrenarse una clasificación binaria.")
    return DatasetValidation(normalized, original, columns, warnings, excluded_rows)


def load_and_validate(path: str | Path | None = None, directory: str | Path = "data/raw") -> DatasetValidation:
    return validate_dataframe(read_table(path or discover_dataset(directory)))


def data_quality_report(validation: DatasetValidation) -> dict[str, Any]:
    frame = validation.dataframe
    quality = {
        "rows": int(len(frame)),
        "rows_excluded": int(validation.excluded_rows),
        "columns": int(len(frame.columns)),
        "duplicates": int(frame.duplicated().sum()),
        "missing_by_column": {str(k): int(v) for k, v in frame.isna().sum().items()},
        "dtypes": {str(k): str(v) for k, v in frame.dtypes.items()},
        "class_distribution": {str(k): int(v) for k, v in frame["result"].value_counts().sort_index().items()},
        "numeric_describe": json.loads(frame[NUMERIC_COLUMNS].describe().to_json()),
        "warnings": validation.warnings,
        "normalized_columns": validation.normalized_columns,
    }
    return quality


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = [*CATEGORICAL_COLUMNS]
    for optional in OPTIONAL_COLUMNS:
        if optional in frame.columns:
            categorical.append(optional)
    return NUMERIC_COLUMNS, categorical


def build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric, categorical = feature_columns(frame)
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)])
    return ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def split_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    numeric, categorical = feature_columns(frame)
    return frame[[*numeric, *categorical]], frame["result"]
