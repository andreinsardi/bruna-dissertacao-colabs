"""Scientific analysis primitives for the definitive CREMA-D Colab.

The functions in this module are intentionally independent from Colab, CUDA,
Google Drive and the training implementation.  This keeps the actor-level
split contract and the quantitative post-processing directly unit-testable.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


DEFAULT_GROUP_ATTRIBUTES = ("Race", "Ethnicity", "Sex", "age_bin")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _candidate_strata(actors: pd.DataFrame, n_required: int) -> tuple[pd.Series | None, str]:
    candidates: list[tuple[str, pd.Series]] = []
    if {"Race", "Sex"}.issubset(actors.columns):
        candidates.append(
            (
                "Race__Sex",
                actors["Race"].fillna("Unknown").astype(str)
                + "__"
                + actors["Sex"].fillna("Unknown").astype(str),
            )
        )
    for column in ("Race", "Sex"):
        if column in actors.columns:
            candidates.append((column, actors[column].fillna("Unknown").astype(str)))

    for name, strata in candidates:
        counts = strata.value_counts(dropna=False)
        if len(counts) >= 2 and int(counts.min()) >= n_required:
            return strata, name
    return None, "unstratified"


def _split_train_validation_actors(
    actors: pd.DataFrame,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    if len(actors) < 3:
        raise ValueError("At least three training actors are required")

    strata, _ = _candidate_strata(actors, n_required=2)
    stratify = strata
    if stratify is not None:
        n_classes = int(stratify.nunique())
        n_validation = int(np.ceil(len(actors) * validation_fraction))
        n_training = len(actors) - n_validation
        if n_validation < n_classes or n_training < n_classes:
            stratify = None

    try:
        train_positions, validation_positions = train_test_split(
            np.arange(len(actors)),
            test_size=validation_fraction,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError:
        train_positions, validation_positions = train_test_split(
            np.arange(len(actors)),
            test_size=validation_fraction,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )
    return np.sort(train_positions), np.sort(validation_positions)


def build_actor_folds(
    manifest: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    validation_fraction: float = 0.15,
    actor_column: str = "ActorID",
) -> tuple[list[dict[str, object]], pd.DataFrame, str]:
    """Build outer test folds and inner validation splits without actor leakage.

    The outer assignment is stratified by ``Race x Sex`` when the actor counts
    allow it, then progressively falls back to Race, Sex and finally KFold.
    Every sample is held out exactly once across the outer folds.
    """

    _require_columns(manifest, [actor_column])
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")

    demographic_columns = [c for c in ("Race", "Sex", "Ethnicity", "age_bin", "Age") if c in manifest.columns]
    actor_frame = manifest[[actor_column, *demographic_columns]].copy()
    for column in demographic_columns:
        inconsistent = actor_frame.groupby(actor_column, dropna=False)[column].nunique(dropna=False)
        if (inconsistent > 1).any():
            offenders = inconsistent[inconsistent > 1].index.tolist()
            raise ValueError(f"Demographic value changes within actor for {column}: {offenders[:5]}")
    actors = actor_frame.drop_duplicates(actor_column).reset_index(drop=True)
    if len(actors) < n_splits:
        raise ValueError(f"Need at least {n_splits} actors; received {len(actors)}")

    strata, strategy = _candidate_strata(actors, n_required=n_splits)
    if strata is None:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        outer_splits = splitter.split(actors)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        outer_splits = splitter.split(actors, strata)

    sample_actors = manifest[actor_column].to_numpy()
    actor_test_fold: dict[object, int] = {}
    folds: list[dict[str, object]] = []
    for fold_number, (train_validation_positions, test_positions) in enumerate(outer_splits, start=1):
        train_validation_actors = actors.iloc[train_validation_positions].reset_index(drop=True)
        inner_train_positions, validation_positions = _split_train_validation_actors(
            train_validation_actors,
            validation_fraction=validation_fraction,
            seed=seed + fold_number,
        )

        train_actor_values = train_validation_actors.iloc[inner_train_positions][actor_column].to_numpy()
        validation_actor_values = train_validation_actors.iloc[validation_positions][actor_column].to_numpy()
        test_actor_values = actors.iloc[test_positions][actor_column].to_numpy()
        for actor in test_actor_values:
            actor_test_fold[actor] = fold_number

        folds.append(
            {
                "fold": fold_number,
                "train_idx": np.flatnonzero(np.isin(sample_actors, train_actor_values)),
                "val_idx": np.flatnonzero(np.isin(sample_actors, validation_actor_values)),
                "test_idx": np.flatnonzero(np.isin(sample_actors, test_actor_values)),
                "train_actors": train_actor_values.tolist(),
                "val_actors": validation_actor_values.tolist(),
                "test_actors": test_actor_values.tolist(),
                "outer_stratification": strategy,
            }
        )

    assignments = actors.copy()
    assignments["outer_test_fold"] = assignments[actor_column].map(actor_test_fold).astype(int)
    assignments["outer_stratification"] = strategy
    validate_actor_folds(manifest, folds, actor_column=actor_column)
    return folds, assignments, strategy


def validate_actor_folds(
    manifest: pd.DataFrame,
    folds: Sequence[dict[str, object]],
    actor_column: str = "ActorID",
) -> dict[str, object]:
    """Validate index bounds, actor disjointness and exact outer-fold coverage."""

    _require_columns(manifest, [actor_column])
    appearances = np.zeros(len(manifest), dtype=np.int64)
    leakage_rows: list[dict[str, object]] = []
    for fold in folds:
        fold_number = int(fold["fold"])
        split_indices: dict[str, np.ndarray] = {}
        split_actors: dict[str, set[object]] = {}
        for role in ("train", "val", "test"):
            index = np.asarray(fold[f"{role}_idx"], dtype=np.int64)
            if len(index) == 0:
                raise ValueError(f"Fold {fold_number} has an empty {role} split")
            if index.min() < 0 or index.max() >= len(manifest):
                raise ValueError(f"Fold {fold_number} contains an out-of-range index")
            split_indices[role] = index
            split_actors[role] = set(manifest.iloc[index][actor_column].tolist())

        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = split_actors[left].intersection(split_actors[right])
            if overlap:
                leakage_rows.append(
                    {
                        "fold": fold_number,
                        "left": left,
                        "right": right,
                        "actors": sorted(map(str, overlap)),
                    }
                )
        appearances[split_indices["test"]] += 1

    valid = not leakage_rows and bool(np.all(appearances == 1))
    if not valid:
        raise ValueError(
            "Invalid actor folds: "
            f"leakage={leakage_rows[:3]}, test appearances={np.unique(appearances, return_counts=True)}"
        )
    return {
        "valid": True,
        "n_folds": len(folds),
        "n_samples": len(manifest),
        "n_actors": int(manifest[actor_column].nunique()),
        "test_appearances_min": int(appearances.min()),
        "test_appearances_max": int(appearances.max()),
        "actor_leakage_count": 0,
    }


def _normalized_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    raw = confusion_matrix(y_true, y_pred, labels=list(labels)).astype(np.float64)
    return np.divide(raw, raw.sum(axis=1, keepdims=True), out=np.zeros_like(raw), where=raw.sum(axis=1, keepdims=True) > 0)


def aggregate_cv_results(
    records: Sequence[dict[str, object]],
    labels: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """Aggregate fold metrics and row-normalized confusion matrices by model."""

    if not records:
        raise ValueError("At least one cross-validation record is required")
    metric_rows: list[dict[str, object]] = []
    confusions: dict[str, dict[str, np.ndarray]] = {}
    model_names = sorted({str(record["model"]) for record in records})
    for record in records:
        y_true = np.asarray(record["y_true"], dtype=np.int64)
        y_pred = np.asarray(record["y_pred"], dtype=np.int64)
        if y_true.shape != y_pred.shape or y_true.ndim != 1:
            raise ValueError("y_true and y_pred must be aligned one-dimensional arrays")
        metric_rows.append(
            {
                "model": str(record["model"]),
                "fold": int(record["fold"]),
                "n_test": int(len(y_true)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            }
        )

    fold_metrics = pd.DataFrame(metric_rows).sort_values(["model", "fold"]).reset_index(drop=True)
    summary_rows: list[dict[str, object]] = []
    for model in model_names:
        selected_records = [record for record in records if str(record["model"]) == model]
        model_metrics = fold_metrics[fold_metrics["model"] == model]
        row: dict[str, object] = {"model": model, "n_folds": int(len(model_metrics))}
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"):
            row[f"{metric}_mean"] = float(model_metrics[metric].mean())
            row[f"{metric}_std"] = float(model_metrics[metric].std(ddof=1)) if len(model_metrics) > 1 else 0.0
            row[f"{metric}_min"] = float(model_metrics[metric].min())
            row[f"{metric}_max"] = float(model_metrics[metric].max())
        summary_rows.append(row)

        matrices = np.stack(
            [
                _normalized_confusion(
                    np.asarray(record["y_true"], dtype=np.int64),
                    np.asarray(record["y_pred"], dtype=np.int64),
                    labels,
                )
                for record in selected_records
            ]
        )
        pooled_true = np.concatenate([np.asarray(record["y_true"], dtype=np.int64) for record in selected_records])
        pooled_pred = np.concatenate([np.asarray(record["y_pred"], dtype=np.int64) for record in selected_records])
        pooled_raw = confusion_matrix(pooled_true, pooled_pred, labels=list(labels))
        confusions[model] = {
            "mean": matrices.mean(axis=0),
            "std": matrices.std(axis=0, ddof=1) if len(matrices) > 1 else np.zeros_like(matrices[0]),
            "pooled_raw": pooled_raw,
            "pooled_normalized": _normalized_confusion(pooled_true, pooled_pred, labels),
        }

    summary = pd.DataFrame(summary_rows).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)
    return fold_metrics, summary, confusions


def group_emotion_metrics(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_attributes: Sequence[str] = DEFAULT_GROUP_ATTRIBUTES,
    class_names: Sequence[str] | None = None,
    min_support: int = 10,
) -> pd.DataFrame:
    """Return recall, precision, F1 and false-positive rate per group/emotion."""

    _require_columns(metadata, group_attributes)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if len(metadata) != len(y_true) or y_true.shape != y_pred.shape:
        raise ValueError("metadata, y_true and y_pred must be aligned")
    if class_names is None:
        n_classes = int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)
        class_names = [str(index) for index in range(n_classes)]

    rows: list[dict[str, object]] = []
    for attribute in group_attributes:
        group_values = metadata[attribute].fillna("Unknown").astype(str).to_numpy()
        for group in pd.unique(group_values):
            group_mask = group_values == group
            for class_id, emotion in enumerate(class_names):
                true_mask = group_mask & (y_true == class_id)
                predicted_mask = group_mask & (y_pred == class_id)
                support_true = int(true_mask.sum())
                support_predicted = int(predicted_mask.sum())
                if support_true < min_support:
                    continue
                true_positive = int((true_mask & (y_pred == class_id)).sum())
                false_positive = int((predicted_mask & (y_true != class_id)).sum())
                negative_support = int((group_mask & (y_true != class_id)).sum())
                recall = true_positive / support_true
                precision = true_positive / support_predicted if support_predicted else np.nan
                f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
                rows.append(
                    {
                        "protected_attribute": attribute,
                        "group": group,
                        "emotion_id": class_id,
                        "emotion": str(emotion),
                        "support_true": support_true,
                        "support_predicted": support_predicted,
                        "true_positive": true_positive,
                        "false_positive": false_positive,
                        "recall": float(recall),
                        "precision": float(precision) if not np.isnan(precision) else np.nan,
                        "f1": float(f1),
                        "false_positive_rate": float(false_positive / negative_support) if negative_support else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def confusion_pairs_by_group(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_attributes: Sequence[str] = DEFAULT_GROUP_ATTRIBUTES,
    class_names: Sequence[str] | None = None,
    min_true_support: int = 10,
) -> pd.DataFrame:
    """Quantify every non-diagonal true-to-predicted confusion within groups."""

    _require_columns(metadata, group_attributes)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if len(metadata) != len(y_true) or y_true.shape != y_pred.shape:
        raise ValueError("metadata, y_true and y_pred must be aligned")
    if class_names is None:
        n_classes = int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)
        class_names = [str(index) for index in range(n_classes)]

    rows: list[dict[str, object]] = []
    for attribute in group_attributes:
        groups = metadata[attribute].fillna("Unknown").astype(str).to_numpy()
        for group in pd.unique(groups):
            group_mask = groups == group
            group_error_count = int((group_mask & (y_true != y_pred)).sum())
            if not group_error_count:
                continue
            for true_id, true_emotion in enumerate(class_names):
                true_mask = group_mask & (y_true == true_id)
                true_support = int(true_mask.sum())
                if true_support < min_true_support:
                    continue
                for predicted_id, predicted_emotion in enumerate(class_names):
                    if true_id == predicted_id:
                        continue
                    count = int((true_mask & (y_pred == predicted_id)).sum())
                    if count == 0:
                        continue
                    rows.append(
                        {
                            "protected_attribute": attribute,
                            "group": group,
                            "true_emotion_id": true_id,
                            "true_emotion": str(true_emotion),
                            "predicted_emotion_id": predicted_id,
                            "predicted_emotion": str(predicted_emotion),
                            "count": count,
                            "true_emotion_support": true_support,
                            "group_error_count": group_error_count,
                            "error_rate_within_true": float(count / true_support),
                            "share_of_group_errors": float(count / group_error_count),
                        }
                    )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["protected_attribute", "group", "count"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def sample_distribution_tables(
    metadata: pd.DataFrame,
    group_attributes: Sequence[str] = DEFAULT_GROUP_ATTRIBUTES,
    emotion_column: str = "label_name",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create long-form demographic counts and emotion-by-demographic counts."""

    _require_columns(metadata, [emotion_column, *group_attributes])
    distribution_rows: list[dict[str, object]] = []
    cross_rows: list[pd.DataFrame] = []
    for attribute in group_attributes:
        values = metadata[attribute].fillna("Unknown").astype(str)
        counts = values.value_counts(dropna=False)
        for group, count in counts.items():
            distribution_rows.append(
                {
                    "protected_attribute": attribute,
                    "group": group,
                    "n": int(count),
                    "percent": float(100 * count / len(metadata)),
                }
            )
        cross = pd.crosstab(metadata[emotion_column], values, dropna=False).stack().rename("n").reset_index()
        cross.columns = ["emotion", "group", "n"]
        cross.insert(0, "protected_attribute", attribute)
        cross["percent_within_emotion"] = cross["n"] / cross.groupby("emotion")["n"].transform("sum") * 100
        cross_rows.append(cross)
    return pd.DataFrame(distribution_rows), pd.concat(cross_rows, ignore_index=True)


def balance_strategy_table(
    metadata: pd.DataFrame,
    label_column: str = "label_code",
    race_column: str = "Race",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Document class weights and label-by-race sampling weights."""

    _require_columns(metadata, [label_column, race_column])
    label_counts = metadata[label_column].fillna("Unknown").astype(str).value_counts().sort_index()
    class_rows = pd.DataFrame({"label": label_counts.index, "n": label_counts.values})
    raw_weights = len(metadata) / np.maximum(class_rows["n"].to_numpy(dtype=float), 1.0)
    class_rows["class_weight_normalized"] = raw_weights / raw_weights.mean()
    class_rows["strategy"] = "weighted_cross_entropy"

    keys = (
        metadata[label_column].fillna("Unknown").astype(str)
        + "__"
        + metadata[race_column].fillna("Unknown").astype(str)
    )
    key_counts = keys.value_counts().sort_index()
    sampler_rows = []
    raw_sample_weights = keys.map(lambda key: 1.0 / key_counts[key]).to_numpy(dtype=float)
    normalized_sample_weights = raw_sample_weights / raw_sample_weights.mean()
    for key, count in key_counts.items():
        label, race = key.split("__", 1)
        mask = keys == key
        sampler_rows.append(
            {
                "label": label,
                "Race": race,
                "n": int(count),
                "mean_sampling_weight_normalized": float(normalized_sample_weights[mask].mean()),
                "expected_share_after_resampling": float(1 / len(key_counts)),
                "strategy": "weighted_random_sampler_label_x_race",
            }
        )
    return class_rows, pd.DataFrame(sampler_rows)


def safe_slug(value: object) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    return slug or "unknown"
