import pandas as pd
import pytest

from irrigation_ml.data import build_preprocessor, validate_dataframe


def test_aliases_and_binary_target():
    frame = pd.DataFrame({"Temperature": [20, 30], "Humidity": [60, 40], "MOI": [20, 80], "Soil Type": ["A", "B"], "Seedling Stage": ["x", "y"], "Result": [0, 1]})
    validation = validate_dataframe(frame)
    assert set(["temperature", "humidity", "moi", "soil_type", "crop_stage", "result"]).issubset(validation.dataframe.columns)
    transformed = build_preprocessor(validation.dataframe).fit_transform(validation.dataframe.drop(columns="result"))
    assert transformed.shape[0] == 2


def test_missing_required_column_is_actionable():
    with pytest.raises(ValueError, match="Faltan columnas requeridas"):
        validate_dataframe(pd.DataFrame({"temp": [20], "result": [1]}))


def test_excess_water_class_is_explicitly_excluded_for_binary_study():
    frame = pd.DataFrame({"temp": [20, 30, 35], "humidity": [60, 40, 35], "MOI": [20, 80, 70], "soil_type": ["A", "B", "C"], "Seedling Stage": ["x", "y", "z"], "result": [0, 1, 2]})
    validation = validate_dataframe(frame)
    assert validation.excluded_rows == 1
    assert validation.dataframe["result"].tolist() == [0, 1]
    assert any("exceso de agua" in warning for warning in validation.warnings)


def test_optional_missing_category_is_imputed_for_inference():
    frame = pd.DataFrame({"temp": [20, 30], "humidity": [60, 40], "MOI": [20, 80], "soil_type": ["A", "B"], "Seedling Stage": ["x", "y"], "crop_id": [None, "crop-2"], "result": [0, 1]})
    validation = validate_dataframe(frame)
    transformed = build_preprocessor(validation.dataframe).fit_transform(validation.dataframe.drop(columns="result"))
    assert transformed.shape[0] == 2
