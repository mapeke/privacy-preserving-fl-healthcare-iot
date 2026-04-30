"""Classification metrics tailored to imbalanced ECG arrhythmia data.

The MIT-BIH AAMI 5-class problem is highly imbalanced (~90% normal beats), so accuracy alone is
misleading. We compute macro-F1 and per-class precision / recall / F1 in addition.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

AAMI_CLASSES = ("N", "S", "V", "F", "Q")  # AAMI 5-class scheme


@dataclass(frozen=True)
class ClassificationReport:
    """Snapshot of one evaluation pass."""

    accuracy: float
    macro_f1: float
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_f1: dict[str, float]
    confusion: list[list[int]] = field(default_factory=list)
    num_samples: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "per_class_precision": self.per_class_precision,
            "per_class_recall": self.per_class_recall,
            "per_class_f1": self.per_class_f1,
            "confusion": self.confusion,
            "num_samples": self.num_samples,
        }


def compute_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: tuple[str, ...] = AAMI_CLASSES,
) -> ClassificationReport:
    """Compute accuracy, macro-F1, and per-class P/R/F1 on integer labels."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

    labels = list(range(len(class_names)))
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))

    per_p = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_r = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_f = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return ClassificationReport(
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class_precision={c: float(v) for c, v in zip(class_names, per_p)},
        per_class_recall={c: float(v) for c, v in zip(class_names, per_r)},
        per_class_f1={c: float(v) for c, v in zip(class_names, per_f)},
        confusion=cm,
        num_samples=int(y_true.size),
    )


def aggregate_reports(
    reports: list[ClassificationReport],
    weights: list[int] | None = None,
) -> ClassificationReport:
    """Sample-weighted aggregation of per-client classification reports."""
    if not reports:
        raise ValueError("aggregate_reports() called with empty list")
    if weights is None:
        weights = [r.num_samples for r in reports]
    if len(weights) != len(reports):
        raise ValueError("weights length must match reports length")
    total = sum(weights)
    if total == 0:
        raise ValueError("total weight is zero")

    accuracy = sum(r.accuracy * w for r, w in zip(reports, weights)) / total
    macro_f1 = sum(r.macro_f1 * w for r, w in zip(reports, weights)) / total

    classes = list(reports[0].per_class_f1.keys())
    per_p = {c: sum(r.per_class_precision[c] * w for r, w in zip(reports, weights)) / total for c in classes}
    per_r = {c: sum(r.per_class_recall[c] * w for r, w in zip(reports, weights)) / total for c in classes}
    per_f = {c: sum(r.per_class_f1[c] * w for r, w in zip(reports, weights)) / total for c in classes}

    return ClassificationReport(
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class_precision=per_p,
        per_class_recall=per_r,
        per_class_f1=per_f,
        confusion=[],
        num_samples=total,
    )
