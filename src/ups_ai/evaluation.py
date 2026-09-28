"""Metrics and validation selection used by the GRAPE-UPS alert pathway."""
import numpy as np
import pandas as pd
from typing import Any
from scipy.stats import t
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

def threshold_from_validation(y_true: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(np.quantile(scores, np.linspace(0.50, 0.995, 250)))
    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for threshold in candidates:
        value = f1_score(
            y_true,
            scores >= threshold,
            zero_division=0,
        )
        if value > best_f1:
            best_f1 = float(value)
            best_threshold = float(threshold)
    return best_threshold

def binary_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    prediction = scores >= threshold
    return {
        "average_precision": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
    }

def event_ndcg(frame: pd.DataFrame, score_column: str) -> float:
    values: list[float] = []
    for _, event in frame.groupby(["string_id", "event_id"], sort=False):
        truth = event["label_anomaly"].to_numpy(float)
        if not np.any(truth > 0):
            continue
        order = np.argsort(-event[score_column].to_numpy(float), kind="stable")
        discounts = 1.0 / np.log2(np.arange(2, len(event) + 2))
        dcg = float(np.sum(truth[order] * discounts))
        ideal = float(np.sum(np.sort(truth)[::-1] * discounts))
        values.append(dcg / ideal if ideal > 0 else 0.0)
    return float(np.mean(values))

def interval(values: pd.Series) -> dict[str, float]:
    array = values.to_numpy(float)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    half = (
        float(t.ppf(0.975, len(array) - 1) * std / np.sqrt(len(array)))
        if len(array) > 1
        else 0.0
    )
    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }

def repeated_event_ewma(
    frame: pd.DataFrame,
    scores: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Apply causal per-battery EWMA in increasing event order."""
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1].")
    output = np.empty(len(frame), dtype=float)
    for _, indices in frame.groupby(
        ["string_id", "battery_id"],
        sort=False,
    ).groups.items():
        ordered = np.asarray(list(indices), dtype=int)
        ordered = ordered[
            np.argsort(frame.loc[ordered, "event_index"].to_numpy(int))
        ]
        output[ordered] = (
            pd.Series(scores[ordered])
            .ewm(alpha=alpha, adjust=False)
            .mean()
            .to_numpy(float)
        )
    return output
