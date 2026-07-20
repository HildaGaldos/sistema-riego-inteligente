"""TensorFlow/Keras constructors for the five required tabular models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def _tf():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow no está instalado. Ejecute: pip install -r ml/requirements.txt"
        ) from exc
    return tf


@dataclass
class ModelConfig:
    name: str
    learning_rate: float = 1e-3
    dropout: float = 0.2
    hidden_units: tuple[int, ...] = (64, 32)
    filters: int = 32
    kernel_size: int = 3
    lstm_units: int = 32
    rbf_centers: int = 16
    rbf_gamma: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def rbf_layer_class():
    tf = _tf()

    @tf.keras.utils.register_keras_serializable(package="smart_irrigation")
    class RBFLayer(tf.keras.layers.Layer):
        def __init__(self, units: int, gamma: float = 1.0, centers: Any = None, **kwargs):
            super().__init__(**kwargs)
            self.units = int(units)
            self.gamma = float(gamma)
            self.initial_centers = centers

        def build(self, input_shape):
            input_dim = int(input_shape[-1])
            initializer = "zeros" if self.initial_centers is not None else "glorot_uniform"
            self.centers = self.add_weight(
                name="centers", shape=(self.units, input_dim), initializer=initializer, trainable=True
            )
            if self.initial_centers is not None:
                values = self.initial_centers
                if values.shape != (self.units, input_dim):
                    raise ValueError("Los centros RBF no coinciden con el número de features transformadas.")
                self.centers.assign(values)
            super().build(input_shape)

        def call(self, inputs):
            distances = tf.reduce_sum(tf.square(tf.expand_dims(inputs, 1) - self.centers), axis=-1)
            return tf.exp(-self.gamma * distances)

        def get_config(self):
            config = super().get_config()
            config.update({"units": self.units, "gamma": self.gamma})
            return config

    return RBFLayer


def _compile(model, config: ModelConfig):
    tf = _tf()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc")],
    )
    return model


def build_model(config: ModelConfig, input_dim: int, rbf_centers: Any = None):
    tf = _tf()
    inputs = tf.keras.Input(shape=(input_dim,), name="tabular_features")
    name = config.name.upper()
    if name == "MLP":
        x = inputs
        for units in config.hidden_units:
            x = tf.keras.layers.Dense(units, activation="relu")(x)
            x = tf.keras.layers.Dropout(config.dropout)(x)
    elif name == "DNN":
        x = inputs
        for units in (128, 64, 32, 16):
            x = tf.keras.layers.Dense(units)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.Dropout(config.dropout)(x)
    elif name == "RBF":
        x = rbf_layer_class()(config.rbf_centers, gamma=config.rbf_gamma, centers=rbf_centers)(inputs)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
    elif name == "CNN_MLP":
        x = tf.keras.layers.Reshape((input_dim, 1))(inputs)
        x = tf.keras.layers.Conv1D(config.filters, config.kernel_size, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.GlobalMaxPooling1D()(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        x = tf.keras.layers.Dropout(config.dropout)(x)
    elif name == "LSTM_MLP":
        x = tf.keras.layers.Reshape((input_dim, 1))(inputs)
        x = tf.keras.layers.LSTM(config.lstm_units)(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        x = tf.keras.layers.Dropout(config.dropout)(x)
    else:
        raise ValueError(f"Modelo no soportado: {config.name}")
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="irrigation_probability")(x)
    return _compile(tf.keras.Model(inputs, outputs, name=config.name), config)


def default_configs() -> list[ModelConfig]:
    return [
        ModelConfig("MLP", hidden_units=(64, 32)),
        ModelConfig("DNN", hidden_units=(128, 64, 32, 16)),
        ModelConfig("RBF", rbf_centers=16, rbf_gamma=1.0),
        ModelConfig("CNN_MLP", filters=32, kernel_size=3),
        ModelConfig("LSTM_MLP", lstm_units=32),
    ]
