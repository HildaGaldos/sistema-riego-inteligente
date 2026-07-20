"""Metrics and statistical comparisons used by the training pipeline."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, probabilities, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities).reshape(-1)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) > 1 else 0.0,
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "threshold": float(threshold),
        "n": int(len(y_true)),
    }


def bootstrap_ci(y_true, probabilities, metric: str = "f1", seed: int = 42, n_bootstrap: int = 1000):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    values = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2 and metric == "roc_auc":
            continue
        values.append(classification_metrics(y_true[indices], probabilities[indices])[metric])
    if not values:
        return {"lower": None, "upper": None}
    lower, upper = np.percentile(values, [2.5, 97.5])
    return {"lower": float(lower), "upper": float(upper)}


def paired_rank_biserial(x, y) -> float:
    """Rank-biserial effect for paired differences, preserving ties."""

    from scipy.stats import rankdata

    differences = np.asarray(x) - np.asarray(y)
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return float((positive - negative) / (positive + negative))


def _holm_adjust(p_values: list[float | None]) -> list[float | None]:
    """Holm-Bonferroni adjustment preserving the original comparison order."""

    valid = [(index, float(value)) for index, value in enumerate(p_values) if value is not None and np.isfinite(value)]
    adjusted: list[float | None] = [None] * len(p_values)
    previous = 0.0
    for rank, (index, value) in enumerate(sorted(valid, key=lambda item: item[1]), start=1):
        corrected = min(1.0, max(previous, (len(valid) - rank + 1) * value))
        adjusted[index] = float(corrected)
        previous = corrected
    return adjusted


def statistical_tests(cv_results, test_predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Run paired t-Student, Friedman, Wilcoxon/Holm and McNemar tests.

    CV scores are paired by fold, so every model comparison uses the same
    partitions. This is the appropriate unit of comparison for the requested
    model-level statistical analysis; test-set predictions are reserved for
    McNemar and the classification metrics.
    """
    try:
        from scipy.stats import friedmanchisquare, rankdata, t as t_distribution, ttest_rel, wilcoxon
    except ImportError:
        return {"warning": "scipy no está instalado", "friedman": None, "paired_t": [], "wilcoxon": [], "pairwise": [], "interpretation_es": "Instale scipy para calcular las pruebas estadísticas."}
    import pandas as pd

    frame = cv_results if hasattr(cv_results, "columns") else pd.DataFrame(cv_results)
    pivot = frame.pivot_table(index="fold", columns="model", values="f1", aggfunc="mean")
    models = list(pivot.columns)
    complete = pivot.dropna()
    result: dict[str, Any] = {
        "alpha": 0.05,
        "models": models,
        "n_folds": int(len(complete)),
        "friedman": None,
        "paired_t": [],
        "wilcoxon": [],
        "pairwise": [],
        "mcnemar": [],
    }
    if len(models) >= 3 and len(complete) >= 3:
        statistic, pvalue = friedmanchisquare(*(complete[m].values for m in models))
        kendall_w = float(statistic / (len(complete) * (len(models) - 1))) if len(models) > 1 else 0.0
        result["friedman"] = {
            "statistic": float(statistic),
            "p_value": float(pvalue),
            "degrees_of_freedom": int(len(models) - 1),
            "kendall_w": kendall_w,
            "significant": bool(pvalue < 0.05),
            "null_hypothesis_es": "Los modelos tienen el mismo rendimiento mediano por fold.",
            "interpretation_es": (
                f"p={pvalue:.5f} < 0.05: se rechaza H0 y hay diferencias globales entre los {len(models)} modelos; "
                "Friedman no identifica por sí solo qué pares son diferentes, por eso se revisan t pareada y Wilcoxon."
                if pvalue < 0.05 else
                f"p={pvalue:.5f} >= 0.05: no se rechaza H0; no hay evidencia suficiente de diferencias globales entre los modelos."
            ),
        }
    else:
        result["friedman"] = {
            "available": False,
            "reason": "Friedman requiere al menos 3 modelos y 3 folds completos.",
            "n_folds": int(len(complete)),
            "interpretation_es": "No se puede concluir con Friedman: ejecute el modo completo con al menos 3 folds completos.",
        }

    t_raw: list[dict[str, Any]] = []
    w_raw: list[dict[str, Any]] = []
    for left, right in itertools.combinations(models, 2):
        pair = complete[[left, right]].dropna()
        left_values = pair[left].to_numpy(dtype=float)
        right_values = pair[right].to_numpy(dtype=float)
        difference = left_values - right_values
        mean_difference = float(np.mean(difference)) if len(difference) else None
        sd_difference = float(np.std(difference, ddof=1)) if len(difference) > 1 else None
        cohen_d = float(mean_difference / sd_difference) if sd_difference and sd_difference > 0 else 0.0
        degrees_of_freedom = int(len(difference) - 1) if len(difference) >= 2 else None
        ci_low = ci_high = None
        if len(difference) >= 2:
            if sd_difference and sd_difference > 0:
                t_statistic, t_pvalue = ttest_rel(left_values, right_values, nan_policy="omit")
                critical = float(t_distribution.ppf(0.975, degrees_of_freedom))
                ci_low = float(mean_difference - critical * sd_difference / np.sqrt(len(difference)))
                ci_high = float(mean_difference + critical * sd_difference / np.sqrt(len(difference)))
            else:
                t_statistic = float(np.copysign(np.inf, mean_difference)) if mean_difference else 0.0
                t_pvalue = 0.0 if mean_difference else 1.0
                ci_low = ci_high = mean_difference
            try:
                w_statistic, w_pvalue = wilcoxon(left_values, right_values, zero_method="wilcox", alternative="two-sided", method="auto")
            except ValueError:
                w_statistic, w_pvalue = 0.0, 1.0
            t_value = float(t_pvalue) if np.isfinite(t_pvalue) else 0.0
            w_value = float(w_pvalue) if np.isfinite(w_pvalue) else 1.0
            t_stat_value = float(t_statistic) if np.isfinite(t_statistic) else float(t_statistic)
            w_stat_value = float(w_statistic) if np.isfinite(w_statistic) else 0.0
            nonzero = difference[difference != 0]
            ranks = rankdata(np.abs(nonzero), method="average") if len(nonzero) else np.array([])
            positive_rank_sum = float(ranks[nonzero > 0].sum()) if len(nonzero) else 0.0
            negative_rank_sum = float(ranks[nonzero < 0].sum()) if len(nonzero) else 0.0
        else:
            t_value = w_value = None
            t_stat_value = w_stat_value = None
            positive_rank_sum = negative_rank_sum = 0.0
        direction = "may be higher" if (mean_difference or 0) > 0 else "may be lower" if (mean_difference or 0) < 0 else "is equal"
        base = {
            "model_a": left,
            "model_b": right,
            "n": int(len(difference)),
            "mean_difference": mean_difference,
            "sd_difference": sd_difference,
            "degrees_of_freedom": degrees_of_freedom,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "cohen_d_paired": cohen_d,
            "direction": direction,
        }
        t_raw.append({**base, "t_statistic": t_stat_value, "p_value": t_value})
        w_raw.append({**base, "statistic": w_stat_value, "positive_rank_sum": positive_rank_sum, "negative_rank_sum": negative_rank_sum, "n_nonzero": int(np.count_nonzero(difference)), "effect": paired_rank_biserial(left_values, right_values) if len(difference) else 0.0, "p_value": w_value})

    for item, adjusted in zip(t_raw, _holm_adjust([row["p_value"] for row in t_raw])):
        item["p_value_holm"] = adjusted
        item["significant"] = bool(adjusted is not None and adjusted < 0.05)
        item["null_hypothesis_es"] = "La diferencia media de F1 entre los dos modelos es igual a cero."
        item["interpretation_es"] = (
            f"p ajustado={adjusted:.5f} < 0.05: diferencia media significativa; {item['model_a']} {item['direction']} que {item['model_b']} en F1 por fold."
            if adjusted is not None and adjusted < 0.05 else
            f"p ajustado={adjusted:.5f} >= 0.05: no se demuestra una diferencia media significativa entre {item['model_a']} y {item['model_b']}."
            if adjusted is not None else "No fue posible calcular la interpretación por falta de folds."
        )
        result["paired_t"].append(item)
    for item, adjusted in zip(w_raw, _holm_adjust([row["p_value"] for row in w_raw])):
        item["p_value_holm"] = adjusted
        item["significant"] = bool(adjusted is not None and adjusted < 0.05)
        item["null_hypothesis_es"] = "La distribución de las diferencias pareadas está centrada en cero."
        item["interpretation_es"] = (
            f"p ajustado={adjusted:.5f} < 0.05: Wilcoxon detecta una diferencia significativa en los rangos pareados entre {item['model_a']} y {item['model_b']}."
            if adjusted is not None and adjusted < 0.05 else
            f"p ajustado={adjusted:.5f} >= 0.05: Wilcoxon no detecta una diferencia significativa entre {item['model_a']} y {item['model_b']}."
            if adjusted is not None else "No fue posible calcular la interpretación por falta de folds."
        )
        result["wilcoxon"].append(item)
        result["pairwise"].append(item)

    # Keep a compact ranking for the Reports and Statistics screens.
    means = {model: float(complete[model].mean()) if model in complete else None for model in models}
    result["model_means_f1"] = dict(sorted(means.items(), key=lambda item: item[1] if item[1] is not None else -1, reverse=True))

    try:
        from statsmodels.stats.contingency_tables import mcnemar
    except ImportError:
        mcnemar = None
    if test_predictions:
        best = next(iter(test_predictions))
        base = test_predictions[best]["predictions"]
        for model, values in test_predictions.items():
            if model == best:
                continue
            if mcnemar:
                table = np.zeros((2, 2), dtype=int)
                for a, b, y in zip(base, values["predictions"], values["y_true"]):
                    table[int(a != y), int(b != y)] += 1
                test = mcnemar(table, exact=False, correction=True)
                result["mcnemar"].append({"best": best, "competitor": model, "p_value": float(test.pvalue), "table": table.tolist()})
            else:
                result["mcnemar"].append({"best": best, "competitor": model, "warning": "statsmodels no instalado"})
    result["interpretation_es"] = (
        "La t de Student pareada evalúa diferencias medias de F1 por fold; Friedman contrasta la diferencia global entre tres o más modelos; "
        "Wilcoxon contrasta diferencias pareadas sin asumir normalidad. Se aplicó Holm a las comparaciones múltiples y no se declara superioridad "
        "si el p ajustado es mayor o igual que 0.05."
    )
    result["formulas_es"] = {
        "paired_t": "d_i = F1_Ai - F1_Bi; t = media(d) / (sd(d) / sqrt(n)); gl = n - 1; IC95% = media(d) ± t_0.975,gl · sd(d)/sqrt(n).",
        "friedman": "Se asignan rangos dentro de cada fold; Q = 12/[n·k·(k+1)] · suma(R_j²) - 3n(k+1), con gl=k-1; W de Kendall = Q/[n·(k-1)].",
        "wilcoxon": "Se eliminan diferencias cero, se ordenan |d_i|, se suman rangos positivos y negativos y W es el menor de ambas sumas; Holm ajusta las comparaciones.",
    }
    result["methodology_es"] = "La unidad pareada es el mismo fold estratificado para cada modelo. El conjunto de prueba no se utiliza para t, Friedman ni Wilcoxon."
    return result
