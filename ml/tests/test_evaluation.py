import pandas as pd

from irrigation_ml.evaluation import statistical_tests


def test_paired_t_friedman_and_wilcoxon_are_calculated_by_fold():
    rows = []
    scores = {
        "MLP": [0.80, 0.82, 0.81, 0.83, 0.84],
        "DNN": [0.779, 0.792, 0.797, 0.813, 0.819],
        "RBF": [0.743, 0.761, 0.752, 0.771, 0.779],
    }
    for model, values in scores.items():
        rows.extend({"model": model, "fold": fold, "f1": score} for fold, score in enumerate(values, start=1))

    result = statistical_tests(pd.DataFrame(rows), {})

    assert result["n_folds"] == 5
    assert result["friedman"]["p_value"] < 0.05
    assert len(result["paired_t"]) == 3
    assert len(result["wilcoxon"]) == 3
    assert all(row["p_value_holm"] is not None and 0 <= row["p_value_holm"] <= 1 for row in result["wilcoxon"])
    assert result["friedman"]["degrees_of_freedom"] == 2
    assert result["friedman"]["kendall_w"] > 0
    assert all(row["ci95_low"] <= row["mean_difference"] <= row["ci95_high"] for row in result["paired_t"])
    assert all(row["interpretation_es"] for row in result["paired_t"] + result["wilcoxon"])
    assert result["formulas_es"]["paired_t"]
