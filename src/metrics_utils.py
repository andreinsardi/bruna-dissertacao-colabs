"""
metrics_utils.py
Utilitários para métricas, seed, e bootstrap.
"""

import os
import random
import time
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def fix_seed(seed: int = 42):
    """Fixa seeds para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinismo
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_all_metrics(y_true, y_pred, class_names=None) -> dict:
    """
    Calcula todas as métricas necessárias para o artigo.

    Returns:
        dict com accuracy, macro_f1, balanced_accuracy, classification_report (dict),
        confusion_matrix (np.ndarray)
    """
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": cm,
    }


def bootstrap_metric(y_true, y_pred, metric_fn, n_bootstrap=200, seed=42):
    """
    Calcula intervalo de confiança via bootstrap para uma métrica.

    Args:
        y_true, y_pred: arrays de labels
        metric_fn: função(y_true, y_pred) -> float
        n_bootstrap: número de reamostragens
        seed: seed

    Returns:
        dict com mean, std, ci_lower (2.5%), ci_upper (97.5%)
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        score = metric_fn(y_true[idx], y_pred[idx])
        scores.append(score)

    scores = np.array(scores)
    return {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "ci_lower": float(np.percentile(scores, 2.5)),
        "ci_upper": float(np.percentile(scores, 97.5)),
    }


class TrainTimer:
    """Context manager simples para medir tempo de treino."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed = 0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time

    @property
    def elapsed_str(self):
        m, s = divmod(self.elapsed, 60)
        return f"{int(m)}m {s:.1f}s"
