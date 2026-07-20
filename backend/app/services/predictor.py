"""Loads persisted artifacts once and serves inference without retraining."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ML_SRC = ROOT / "ml" / "src"
if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))

from irrigation_ml.data import validate_dataframe
from irrigation_ml.models import rbf_layer_class


class Predictor:
    def __init__(self, artifacts_dir: str | Path = "artifacts"):
        self.artifacts = Path(artifacts_dir)
        self.model = None
        self.preprocessor = None
        self.metadata: dict = {}
        self.threshold = 0.5
        self.reload()

    @property
    def available(self) -> bool:
        return self.model is not None and self.preprocessor is not None

    def reload(self) -> None:
        metadata_path = self.artifacts / "models" / "model_metadata.json"
        preprocessor_path = self.artifacts / "preprocessors" / "preprocessor.joblib"
        model_path = self.artifacts / "models" / "best_irrigation_model.keras"
        threshold_path = self.artifacts / "models" / "threshold.json"
        if not (metadata_path.exists() and preprocessor_path.exists() and model_path.exists()):
            self.model = self.preprocessor = None
            self.metadata = {}
            return
        import joblib
        import tensorflow as tf

        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.preprocessor = joblib.load(preprocessor_path)
        self.model = tf.keras.models.load_model(model_path, custom_objects={"RBFLayer": rbf_layer_class()})
        if threshold_path.exists():
            self.threshold = float(json.loads(threshold_path.read_text(encoding="utf-8")).get("value", 0.5))

    def predict_one(self, payload: dict) -> dict:
        if not self.available:
            raise RuntimeError("No hay modelo persistido. Suba el dataset y lance el entrenamiento explícitamente.")
        frame = pd.DataFrame([payload])
        # Reuse the same schema normalization and feature list as training.
        validation = validate_dataframe(frame.assign(result=0))
        features = validation.dataframe.drop(columns="result")
        transformed = self.preprocessor.transform(features).astype("float32")
        probability = float(self.model.predict(transformed, verbose=0).reshape(-1)[0])
        predicted = int(probability >= self.threshold)
        return {"probability_irrigation": probability, "predicted_class": predicted, "recommendation": "Requiere riego" if predicted else "No requiere riego", "threshold": self.threshold, "model_name": self.metadata.get("model_name", "unknown"), "model_version": self.metadata.get("model_version", "unknown")}
