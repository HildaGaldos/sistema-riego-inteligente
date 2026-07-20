"""End-to-end leakage-safe training, cross-validation and artifact persistence."""

from __future__ import annotations

import json
import os
import platform
import random
import time
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split

from .data import data_quality_report, load_and_validate, split_features
from .evaluation import bootstrap_ci, classification_metrics, statistical_tests
from .models import ModelConfig, build_model, default_configs


class PipelineCancelled(RuntimeError):
    """Raised when the administrator cancels a long-running pipeline."""


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _centers(x_train: np.ndarray, config: ModelConfig, seed: int) -> np.ndarray | None:
    if config.name != "RBF":
        return None
    count = min(config.rbf_centers, len(x_train))
    return KMeans(n_clusters=count, random_state=seed, n_init=10).fit(x_train).cluster_centers_.astype("float32")


def _fit(config: ModelConfig, x_train, y_train, x_valid, y_valid, epochs: int, batch_size: int, seed: int, checkpoint: Path | None = None):
    import tensorflow as tf

    set_seed(seed)
    centers = _centers(x_train, config, seed)
    model = build_model(config, x_train.shape[1], rbf_centers=centers)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=max(2, min(8, epochs // 3)), restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]
    if checkpoint:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_loss", save_best_only=True))
    counts = np.bincount(np.asarray(y_train).astype(int), minlength=2)
    class_weight = None
    if counts.min() and counts.max() / counts.min() >= 1.5:
        total = counts.sum()
        class_weight = {0: total / (2 * counts[0]), 1: total / (2 * counts[1])}
    started = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_valid, y_valid),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=0,
    )
    return model, history.history, time.perf_counter() - started, class_weight


def _tune_configs(base_configs: list[ModelConfig], x_train, y_train, fast: bool, seed: int, artifacts: Path, trials_override: int | None = None) -> dict[str, dict[str, Any]]:
    """Tune a compact configuration space with Optuna when available.

    The objective uses only a validation split from training data. Test is never inspected.
    """
    trials = int(trials_override) if trials_override is not None else (3 if fast else int(os.getenv("TUNING_TRIALS", "20")))
    try:
        import optuna
    except ImportError:
        return {c.name: {"backend": "unavailable", "trials": 0, "warning": "Instale optuna para tuning."} for c in base_configs}

    x_a, x_b, y_a, y_b = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train, random_state=seed)
    tuned: dict[str, dict[str, Any]] = {}
    for base in base_configs:
        def objective(trial):
            config = ModelConfig(
                name=base.name,
                learning_rate=trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
                dropout=trial.suggest_float("dropout", 0.0, 0.4),
                filters=trial.suggest_categorical("filters", [16, 32, 64]),
                lstm_units=trial.suggest_categorical("lstm_units", [16, 32, 64]),
                rbf_centers=trial.suggest_categorical("rbf_centers", [8, 16, 32]),
                rbf_gamma=trial.suggest_float("rbf_gamma", 0.1, 3.0, log=True),
            )
            model, _, _, _ = _fit(config, x_a, y_a, x_b, y_b, epochs=3 if fast else 12, batch_size=64, seed=seed + trial.number)
            probabilities = model.predict(x_b, verbose=0).reshape(-1)
            return classification_metrics(y_b, probabilities)["f1"]

        study = optuna.create_study(direction="maximize", study_name=f"{base.name}_irrigation", storage=f"sqlite:///{(artifacts / 'tuning.db').resolve()}", load_if_exists=True)
        study.optimize(objective, n_trials=trials, show_progress_bar=False)
        params = dict(study.best_params)
        tuned[base.name] = {"backend": "optuna", "trials": len(study.trials), "best_value": study.best_value, "best_params": params}
    _json_dump(artifacts / "metrics" / "tuning.json", tuned)
    return tuned


def generate_eda_artifacts(frame: pd.DataFrame, artifacts: Path) -> None:
    metrics = artifacts / "metrics"
    figures = artifacts / "figures"
    metrics.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    _json_dump(metrics / "data_quality.json", data_quality_report(load_validation(frame)))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        numeric = [c for c in ["temperature", "humidity", "moi"] if c in frame]
        frame[numeric].hist(figsize=(10, 6), bins=24)
        plt.tight_layout(); plt.savefig(figures / "numeric_histograms.png", dpi=180); plt.close()
        plt.figure(figsize=(8, 5)); sns.boxplot(data=frame[numeric]); plt.tight_layout(); plt.savefig(figures / "numeric_boxplots.png", dpi=180); plt.close()
        plt.figure(figsize=(7, 5)); sns.heatmap(frame[numeric + ["result"]].corr(method="pearson"), annot=True, cmap="vlag", center=0); plt.tight_layout(); plt.savefig(figures / "correlation_heatmap.png", dpi=180); plt.close()
        for column in ["soil_type", "crop_stage"]:
            plt.figure(figsize=(10, 5)); sns.countplot(data=frame, x=column, hue="result"); plt.xticks(rotation=35, ha="right"); plt.tight_layout(); plt.savefig(figures / f"distribution_{column}.png", dpi=180); plt.close()
    except ImportError:
        _json_dump(metrics / "eda_warning.json", {"warning": "matplotlib/seaborn no instalados"})


def generate_evaluation_artifacts(predictions: dict[str, dict[str, Any]], comparison: pd.DataFrame, best_name: str, artifacts: Path) -> None:
    """Persist the visual evidence shown in Training and Reports."""

    figures = artifacts / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        best = predictions[best_name]
        matrix = confusion_matrix(best["y_true"], best["predictions"], labels=[0, 1])
        plt.figure(figsize=(5.5, 4.5))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="YlGn", cbar=False, xticklabels=["No riego", "Riego"], yticklabels=["No riego", "Riego"])
        plt.xlabel("Predicción")
        plt.ylabel("Real")
        plt.title(f"Matriz de confusión · {best_name}")
        plt.tight_layout()
        plt.savefig(figures / "confusion_matrix_best.png", dpi=180)
        plt.close()

        plt.figure(figsize=(7, 5))
        for name, values in predictions.items():
            y_true = np.asarray(values["y_true"])
            probabilities = np.asarray(values["probabilities"])
            fpr, tpr, _ = roc_curve(y_true, probabilities)
            auc = roc_auc_score(y_true, probabilities) if len(np.unique(y_true)) > 1 else 0.0
            plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.3f})")
        plt.plot([0, 1], [0, 1], "--", color="#9aa8ad", linewidth=1)
        plt.xlabel("Tasa de falsos positivos")
        plt.ylabel("Tasa de verdaderos positivos")
        plt.title("Curvas ROC comparativas")
        plt.legend(fontsize=8, loc="lower right")
        plt.tight_layout()
        plt.savefig(figures / "roc_curves_models.png", dpi=180)
        plt.close()

        columns = [column for column in ["accuracy", "precision", "recall", "f1", "roc_auc", "balanced_accuracy"] if column in comparison]
        heatmap = comparison.set_index("model")[columns].sort_values("f1", ascending=False)
        plt.figure(figsize=(9, 4.5))
        sns.heatmap(heatmap, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, cbar=True)
        plt.title("Mapa de calor de métricas por modelo")
        plt.xlabel("Métrica")
        plt.ylabel("Modelo")
        plt.tight_layout()
        plt.savefig(figures / "model_metrics_heatmap.png", dpi=180)
        plt.close()
    except ImportError:
        _json_dump(artifacts / "metrics" / "evaluation_figures_warning.json", {"warning": "matplotlib/seaborn no instalados"})


def load_validation(frame):
    from .data import validate_dataframe, DatasetValidation
    return validate_dataframe(frame)


def run_training(
    data_path: str | Path | None = None,
    artifacts_dir: str | Path = "artifacts",
    seed: int = 42,
    folds: int = 5,
    fast: bool = True,
    epochs: int | None = None,
    tuning_trials: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run EDA, CV, tuning, final fitting and artifact/report generation on real data."""

    def notify(stage: str, progress: float, message: str, detail: str = "") -> None:
        if progress_callback:
            progress_callback({"stage": stage, "progress": float(progress), "message": message, "detail": detail})

    set_seed(seed)
    notify("data_validation", 2, "Buscando y validando el dataset real")
    artifacts = Path(artifacts_dir)
    for sub in ["models", "preprocessors", "metrics", "figures", "reports"]:
        (artifacts / sub).mkdir(parents=True, exist_ok=True)
    validation = load_and_validate(data_path)
    frame = validation.dataframe
    if frame["result"].nunique() < 2:
        raise ValueError("El dataset cargado debe contener las clases result=0 y result=1.")
    notify("data_cleaning", 12, "Limpieza y normalización completadas", f"{len(frame)} filas utilizables; {validation.excluded_rows} excluidas")
    notify("eda", 15, "Generando EDA, calidad y figuras")
    generate_eda_artifacts(frame, artifacts)
    notify("preprocessing", 22, "Preparando preprocesamiento sin fuga de datos")
    x_frame, y = split_features(frame)
    x_train_frame, x_test_frame, y_train, y_test = train_test_split(x_frame, y, test_size=0.2, stratify=y, random_state=seed)
    configs = default_configs()
    actual_epochs = epochs or (2 if fast else int(os.getenv("EPOCHS", "30")))
    batch_size = 256 if fast else 128
    effective_folds = max(2, min(folds, 2 if fast else folds))
    # Fit preprocessing per fold, preventing validation/test leakage.
    cv_rows: list[dict[str, Any]] = []
    splitter = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
    tuned: dict[str, dict[str, Any]] = {}
    total_cv = len(configs) * effective_folds
    cv_completed = 0
    notify("cross_validation", 25, "Ejecutando validación cruzada estratificada", f"0/{total_cv} entrenamientos de fold")
    for config in configs:
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(x_train_frame, y_train), start=1):
            from .data import build_preprocessor
            preprocessor = build_preprocessor(frame)
            x_a = preprocessor.fit_transform(x_train_frame.iloc[train_idx]).astype("float32")
            x_b = preprocessor.transform(x_train_frame.iloc[valid_idx]).astype("float32")
            model, _, train_seconds, _ = _fit(config, x_a, y_train.iloc[train_idx].values, x_b, y_train.iloc[valid_idx].values, actual_epochs, batch_size, seed + fold)
            probabilities = model.predict(x_b, verbose=0).reshape(-1)
            row = classification_metrics(y_train.iloc[valid_idx].values, probabilities)
            row.update({"model": config.name, "fold": fold, "training_seconds": train_seconds})
            cv_rows.append(row)
            cv_completed += 1
            notify("cross_validation", 25 + 25 * cv_completed / total_cv, f"Validación cruzada: {config.name}", f"Fold {fold}/{effective_folds}; {cv_completed}/{total_cv} completados")
    cv_frame = pd.DataFrame(cv_rows)
    cv_frame.to_csv(artifacts / "metrics" / "cv_results.csv", index=False)
    from .data import build_preprocessor
    tuning_preprocessor = build_preprocessor(frame)
    x_tuning = tuning_preprocessor.fit_transform(x_train_frame).astype("float32")
    notify("tuning", 52, "Ajustando hiperparámetros sin usar test")
    if fast:
        tuned = {config.name: {"backend": "skipped_fast", "trials": 0, "message": "Se omite tuning para acelerar el smoke test; use modo completo para Optuna."} for config in configs}
        _json_dump(artifacts / "metrics" / "tuning.json", tuned)
        notify("tuning", 62, "Tuning omitido en modo rápido", "El modo completo ejecuta Optuna con TUNING_TRIALS")
    else:
        tuned = _tune_configs(configs, x_tuning, y_train.values, False, seed, artifacts, trials_override=tuning_trials)
        notify("tuning", 64, "Tuning completado", f"{len(configs)} estudios Optuna")

    comparison: list[dict[str, Any]] = []
    test_predictions: dict[str, dict[str, Any]] = {}
    trained_models: dict[str, Any] = {}
    notify("model_training", 66, "Entrenando los cinco modelos finales")
    for model_index, config in enumerate(configs, start=1):
        from .data import build_preprocessor
        preprocessor = build_preprocessor(frame)
        x_a = preprocessor.fit_transform(x_train_frame).astype("float32")
        x_b = preprocessor.transform(x_test_frame).astype("float32")
        model, history, train_seconds, class_weight = _fit(config, x_a, y_train.values, x_b, y_test.values, actual_epochs, batch_size, seed, checkpoint=artifacts / "models" / f"{config.name}.keras")
        probabilities = model.predict(x_b, verbose=0).reshape(-1)
        started = time.perf_counter(); model.predict(x_b, verbose=0); elapsed = (time.perf_counter() - started) / max(1, len(x_b))
        inference_samples = []
        for sample in x_b[: min(20, len(x_b))]:
            sample_started = time.perf_counter(); model.predict(sample[None, :], verbose=0); inference_samples.append(time.perf_counter() - sample_started)
        metrics = classification_metrics(y_test, probabilities)
        cv_model = cv_frame.loc[cv_frame.model == config.name]
        cv_std = cv_model["f1"].std(ddof=1)
        metrics.update({"model": config.name, "training_seconds": train_seconds, "inference_seconds_per_row": elapsed, "inference_p50_seconds": float(np.percentile(inference_samples, 50)) if inference_samples else elapsed, "inference_p95_seconds": float(np.percentile(inference_samples, 95)) if inference_samples else elapsed, "parameters": int(model.count_params()), "artifact_bytes": 0, "cv_f1_mean": float(cv_model["f1"].mean()), "cv_f1_std": float(0.0 if pd.isna(cv_std) else cv_std), "cv_roc_auc_mean": float(cv_model["roc_auc"].mean()), "cv_recall_mean": float(cv_model["recall"].mean()), "bootstrap_f1": bootstrap_ci(y_test, probabilities, "f1", seed=seed, n_bootstrap=200 if fast else 1000), "bootstrap_roc_auc": bootstrap_ci(y_test, probabilities, "roc_auc", seed=seed, n_bootstrap=200 if fast else 1000)})
        comparison.append(metrics)
        test_predictions[config.name] = {"predictions": (probabilities >= 0.5).astype(int).tolist(), "y_true": np.asarray(y_test).astype(int).tolist(), "probabilities": probabilities.tolist()}
        trained_models[config.name] = (model, preprocessor, history, metrics)
        notify("model_training", 66 + 18 * model_index / len(configs), f"Modelo {config.name} listo", f"{model_index}/{len(configs)} modelos entrenados")
    comparison_frame = pd.DataFrame(comparison)
    ranking_columns = ["cv_f1_mean", "cv_roc_auc_mean", "cv_recall_mean", "cv_f1_std", "inference_seconds_per_row"]
    comparison_frame.sort_values(ranking_columns, ascending=[False, False, False, True, True]).to_csv(artifacts / "metrics" / "model_comparison.csv", index=False)
    best_name = comparison_frame.sort_values(ranking_columns, ascending=[False, False, False, True, True]).iloc[0]["model"]
    best_model, best_preprocessor, history, best_metrics = trained_models[best_name]
    notify("evaluation", 86, "Calculando métricas, selección y pruebas estadísticas")
    best_model_path = artifacts / "models" / "best_irrigation_model.keras"
    best_model.save(best_model_path)
    try:
        best_model.save(artifacts / "models" / "best_irrigation_model.h5")
    except Exception as exc:
        _json_dump(artifacts / "metrics" / "h5_warning.json", {"warning": str(exc)})
    joblib.dump(best_preprocessor, artifacts / "preprocessors" / "preprocessor.joblib")
    threshold = {"value": 0.5, "method": "fixed_initial_threshold", "optimized_on": "none"}
    _json_dump(artifacts / "models" / "threshold.json", threshold)
    metadata = {"model_name": best_name, "model_version": time.strftime("%Y%m%d-%H%M%S"), "seed": seed, "folds": effective_folds, "requested_folds": folds, "fast_mode": fast, "epochs": actual_epochs, "feature_columns": list(x_train_frame.columns), "metrics": best_metrics, "platform": platform.platform(), "tuning": tuned, "artifact_files": ["best_irrigation_model.h5", "best_irrigation_model.keras", "preprocessor.joblib", "threshold.json"], "statistical_tests": ["t_student_pareada", "friedman", "wilcoxon_holm", "mcnemar"]}
    _json_dump(artifacts / "models" / "model_metadata.json", metadata)
    ordered_predictions = {best_name: test_predictions[best_name], **{name: values for name, values in test_predictions.items() if name != best_name}}
    generate_evaluation_artifacts(ordered_predictions, comparison_frame, best_name, artifacts)
    _json_dump(artifacts / "metrics" / "statistical_tests.json", statistical_tests(cv_frame, ordered_predictions))
    # Keep the best model's history for the dashboard.
    _json_dump(artifacts / "metrics" / "training_history.json", {k: [float(v) for v in values] for k, values in history.items()})
    notify("reports", 94, "Generando PDF, Word y Excel")
    from .reports import generate_reports
    generate_reports(artifacts, frame, comparison_frame, cv_frame)
    notify("ready", 100, "Pipeline completado: módulos listos", f"Modelo seleccionado: {best_name}")
    return {"status": "completed", "model_name": best_name, "rows": len(frame), "artifacts_dir": str(artifacts.resolve()), "metrics": comparison, "excluded_rows": validation.excluded_rows, "folds": effective_folds}
