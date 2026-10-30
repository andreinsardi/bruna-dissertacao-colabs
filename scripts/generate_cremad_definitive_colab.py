"""Generate the definitive, resumable CREMA-D Colab Pro notebook.

Notebook 08 extends the preserved scientific baseline (06) and recovery
notebook (07).  It adds the analyses requested after qualification while
keeping every write inside Google Drive; it never writes to Overleaf.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from generate_cremad_resumable_colab import (
    build_notebook as build_resumable_notebook,
    cell_source,
    replace_or_fail,
    set_source,
    wrap_stage,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "08_cremad_fairness_xai_emot_definitivo_colab_pro.ipynb"
ANALYSIS_MODULE = ROOT / "scripts" / "definitive_analysis.py"


def markdown_cell(source: str, role: str | None = None) -> dict:
    metadata = {"cremad_role": role} if role else {}
    return {
        "cell_type": "markdown",
        "metadata": metadata,
        "source": source.splitlines(keepends=True),
    }


def code_cell(source: str, role: str | None = None, stage: str | None = None) -> dict:
    compile(source, f"<notebook-{role or stage or 'code'}>", "exec")
    metadata = {}
    if role:
        metadata["cremad_role"] = role
    if stage:
        metadata["cremad_stage"] = stage
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def find_cell(cells: list[dict], token: str) -> int:
    matches = [index for index, cell in enumerate(cells) if token in cell_source(cell)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one cell containing {token!r}; found {matches}")
    return matches[0]


SCOPE_AND_DISTRIBUTION_SOURCE = r'''
scope_rows = [
    {
        "dataset": "CREMA-D",
        "dataset_modalities_available": "audio+video",
        "modality_used_in_this_experiment": "visual_only",
        "visual_representation": f"MediaPipe landmarks ({LANDMARK_SET}); temporal_derivatives={ADD_TEMPORAL_DERIVATIVES}",
        "audio_features_used": False,
        "unit_of_evaluation": "video",
        "identity_control": "actor-disjoint holdout and actor-disjoint five-fold CV",
        "claim_boundary": "Results cannot be compared directly with audio-only or audiovisual fusion systems.",
    }
]
modality_scope = pd.DataFrame(scope_rows)
atomic_write_text(TABLE_DIR / "modality_scope.csv", modality_scope.to_csv(index=False))
scope_text = """# Escopo experimental

- O CREMA-D disponibiliza áudio e vídeo.
- Esta execução usa exclusivamente a modalidade visual: frames de vídeo são convertidos em landmarks faciais; nenhuma feature de áudio entra nos modelos.
- Comparações de desempenho com a literatura devem usar métodos visual-only e protocolo de separação compatível. Resultados áudio-only ou multimodais devem aparecer apenas como contexto, nunca como comparação direta.
- A unidade de avaliação é o vídeo e todo split é feito por ator para impedir vazamento de identidade.
"""
atomic_write_text(REPORT_DIR / "experimental_scope.md", scope_text)

distribution_groups = ["Race", "Ethnicity", "Sex", "age_bin"]
sample_distribution, emotion_distribution = sample_distribution_tables(
    meta_run,
    group_attributes=distribution_groups,
    emotion_column="label_name",
)
atomic_write_text(TABLE_DIR / "sample_distribution_by_demographic.csv", sample_distribution.to_csv(index=False))
atomic_write_text(TABLE_DIR / "sample_distribution_emotion_by_demographic.csv", emotion_distribution.to_csv(index=False))

intersectional_specs = [
    ["Race", "Sex"],
    ["Race", "age_bin"],
    ["Sex", "age_bin"],
    ["Race", "Sex", "age_bin"],
]
intersectional_counts = []
for columns in intersectional_specs:
    table = meta_run.groupby(columns, dropna=False).size().rename("n").reset_index()
    table.insert(0, "intersection", " x ".join(columns))
    table["percent"] = 100 * table["n"] / len(meta_run)
    intersectional_counts.append(table)
sample_distribution_intersectional = pd.concat(intersectional_counts, ignore_index=True, sort=False)
atomic_write_text(TABLE_DIR / "sample_distribution_intersectional.csv", sample_distribution_intersectional.to_csv(index=False))

runtime_estimate_definitive_a100 = pd.DataFrame([
    {"scenario": "A100 sem cache de landmarks", "estimated_hours_low": 5.0, "estimated_hours_high": 8.0, "dominant_cost": "MediaPipe em CPU + 5-fold CV"},
    {"scenario": "A100 com landmarks íntegros em cache", "estimated_hours_low": 1.5, "estimated_hours_high": 3.5, "dominant_cost": "treino holdout + 20 treinos do 5-fold + XAI"},
    {"scenario": "retomada após falha", "estimated_hours_low": 0.1, "estimated_hours_high": 3.0, "dominant_cost": "somente shards/épocas/folds ainda incompletos"},
])
atomic_write_text(TABLE_DIR / "runtime_estimate_definitive_a100.csv", runtime_estimate_definitive_a100.to_csv(index=False))

display(modality_scope)
display(sample_distribution)
display(emotion_distribution.head(40))
display(runtime_estimate_definitive_a100)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for ax, attribute in zip(axes.flat, distribution_groups):
    plot_data = sample_distribution[sample_distribution["protected_attribute"] == attribute]
    sns.barplot(data=plot_data, x="group", y="n", ax=ax, color="#4C78A8")
    ax.set_title(f"Distribuição por {attribute}")
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
fig.savefig(FIG_DIR / "02b_sample_distribution_demographics.png", bbox_inches="tight")
plt.show()
log_event("definitive_scope_and_sample_distribution", n_samples=int(len(meta_run)), n_distribution_rows=int(len(sample_distribution)))
'''


BALANCE_SOURCE = r'''
balance_class_weights, balance_sampler = balance_strategy_table(meta_train)
balance_documentation = pd.DataFrame([
    {
        "model_scope": "FlatMLP, TemporalCNN1D, EmoT",
        "label_imbalance_control": "CrossEntropy ponderada pelo inverso da frequência da emoção",
        "demographic_imbalance_control": "nenhum resampling demográfico",
        "downsampling": False,
        "reason": "Preserva todas as amostras e controla o desbalanceamento de classe no loss.",
    },
    {
        "model_scope": "EmoT_Reweighted",
        "label_imbalance_control": "CrossEntropy ponderada pelo inverso da frequência da emoção",
        "demographic_imbalance_control": "WeightedRandomSampler por emoção x Race",
        "downsampling": False,
        "reason": "Mitigação comparável que aumenta a exposição de combinações minoritárias sem descartar vídeos.",
    },
])
atomic_write_text(TABLE_DIR / "balance_class_weights.csv", balance_class_weights.to_csv(index=False))
atomic_write_text(TABLE_DIR / "balance_sampler_label_x_race.csv", balance_sampler.to_csv(index=False))
atomic_write_text(TABLE_DIR / "balance_strategy_documentation.csv", balance_documentation.to_csv(index=False))
display(balance_documentation)
display(balance_class_weights)
display(balance_sampler)
log_event("balance_strategy_documentation", n_train=int(len(meta_train)), n_label_race_cells=int(len(balance_sampler)))
'''


CROSS_VALIDATION_SOURCE = r'''
cross_validation_start = time.time()
CV_DIR = OUTPUT_DIR / "cross_validation"
CV_MODEL_DIR = CV_DIR / "models"
CV_LOG_DIR = CV_DIR / "logs"
CV_FIG_DIR = CV_DIR / "figures"
CV_TABLE_DIR = CV_DIR / "tables"
for directory in [CV_DIR, CV_MODEL_DIR, CV_LOG_DIR, CV_FIG_DIR, CV_TABLE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

cv_folds, cv_actor_assignments, cv_stratification = build_actor_folds(
    meta_valid,
    n_splits=CV_N_SPLITS,
    seed=SEED,
    validation_fraction=CV_VALIDATION_FRACTION,
)
cv_fold_audit = validate_actor_folds(meta_valid, cv_folds)
atomic_write_text(TABLE_DIR / "cv_actor_assignments.csv", cv_actor_assignments.to_csv(index=False))
atomic_write_json(REPORT_DIR / "cv_actor_split_audit.json", cv_fold_audit)
valid_sample_distribution, valid_emotion_distribution = sample_distribution_tables(
    meta_valid,
    group_attributes=PROTECTED_ATTRIBUTES_DEFINITIVE,
    emotion_column="label_name",
)
atomic_write_text(TABLE_DIR / "sample_distribution_valid_by_demographic.csv", valid_sample_distribution.to_csv(index=False))
atomic_write_text(TABLE_DIR / "sample_distribution_valid_emotion_by_demographic.csv", valid_emotion_distribution.to_csv(index=False))
display(cv_actor_assignments.groupby(["outer_test_fold", "Race", "Sex"], dropna=False).size().rename("n_actors").reset_index())
print(f"Estratificação externa: {cv_stratification} | auditoria: {cv_fold_audit}")

def instantiate_cv_model(model_name):
    if model_name == "FlatMLP":
        return FlatMLP(T=T, D=D, n_classes=N_CLASSES)
    if model_name == "TemporalCNN1D":
        return TemporalCNN1D(T=T, D=D, n_classes=N_CLASSES)
    if model_name in {"EmoT", "EmoT_Reweighted"}:
        return EmoTransformer(
            D=D,
            n_classes=N_CLASSES,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            dim_ff=DIM_FF,
            dropout=DROPOUT,
            mlp_head_size=MLP_HEAD_SIZE,
            max_len=T,
        )
    raise ValueError(f"Modelo CV desconhecido: {model_name}")

cv_records = []
cv_prediction_frames = []
cv_history_frames = []
cv_balance_rows = []
original_ckpt_dir, original_log_dir = CKPT_DIR, LOG_DIR
main_mu, main_sigma = mu.copy(), sigma.copy()
force_cv = "cross_validation_5fold" in FORCE_STAGES

try:
    for fold in cv_folds:
        fold_id = int(fold["fold"])
        fold_train_idx = np.asarray(fold["train_idx"], dtype=int)
        fold_val_idx = np.asarray(fold["val_idx"], dtype=int)
        fold_test_idx = np.asarray(fold["test_idx"], dtype=int)
        fold_dir = CV_DIR / f"fold_{fold_id:02d}"
        fold_ckpt_dir = CV_MODEL_DIR / f"fold_{fold_id:02d}"
        fold_log_dir = CV_LOG_DIR / f"fold_{fold_id:02d}"
        for directory in [fold_dir, fold_ckpt_dir, fold_log_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        X_fold_train_raw = X[fold_train_idx]
        X_fold_val_raw = X[fold_val_idx]
        X_fold_test_raw = X[fold_test_idx]
        y_fold_train = y[fold_train_idx]
        y_fold_val = y[fold_val_idx]
        y_fold_test = y[fold_test_idx]
        meta_fold_train = meta_valid.iloc[fold_train_idx].reset_index(drop=True)

        fold_mu = X_fold_train_raw.reshape(-1, D).mean(axis=0).astype(np.float32)
        fold_sigma = X_fold_train_raw.reshape(-1, D).std(axis=0).astype(np.float32)
        fold_sigma[fold_sigma == 0] = 1.0
        X_fold_train = ((X_fold_train_raw - fold_mu) / fold_sigma).astype(np.float32)
        X_fold_val = ((X_fold_val_raw - fold_mu) / fold_sigma).astype(np.float32)
        X_fold_test = ((X_fold_test_raw - fold_mu) / fold_sigma).astype(np.float32)
        atomic_npz_save(
            fold_dir / "normalization_and_indices.npz",
            mu=fold_mu,
            sigma=fold_sigma,
            train_idx=fold_train_idx,
            val_idx=fold_val_idx,
            test_idx=fold_test_idx,
        )

        fold_class_weights, fold_sampler_plan = balance_strategy_table(meta_fold_train)
        fold_class_weights.insert(0, "fold", fold_id)
        fold_sampler_plan.insert(0, "fold", fold_id)
        cv_balance_rows.extend(fold_sampler_plan.to_dict("records"))

        # Os dois loaders compartilham o mesmo TensorDataset para evitar duas
        # cópias integrais do fold na RAM do Colab.
        fold_dataset = TensorDataset(
            torch.from_numpy(X_fold_train),
            torch.from_numpy(y_fold_train.astype(np.int64, copy=False)),
        )
        standard_loader = DataLoader(
            fold_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        mitigated_loader = DataLoader(
            fold_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            sampler=make_group_label_sampler(meta_fold_train),
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        fold_criterion = nn.CrossEntropyLoss(
            weight=class_weights_from_y(y_fold_train),
            label_smoothing=LABEL_SMOOTHING,
        )

        CKPT_DIR, LOG_DIR, mu, sigma = fold_ckpt_dir, fold_log_dir, fold_mu, fold_sigma
        for model_name in CV_MODELS:
            ledger_model_name = f"CV_F{fold_id:02d}_{model_name}"
            prediction_cache = fold_dir / f"predictions_{model_name}.npz"
            if file_is_valid(prediction_cache) and not force_cv:
                with np.load(prediction_cache, allow_pickle=False) as cached:
                    fold_pred = cached["y_pred"].astype(int)
                    fold_prob = cached["y_prob"].astype(np.float32)
                    fold_logits = cached["logits"].astype(np.float32)
                    cached_test_idx = cached["test_idx"].astype(int)
                    cached_y_true = cached["y_true"].astype(int)
                if not np.array_equal(cached_test_idx, fold_test_idx) or not np.array_equal(cached_y_true, y_fold_test):
                    raise RuntimeError(f"Cache CV incompatível: {prediction_cache}")
                history_path = fold_log_dir / f"training_history_{ledger_model_name}.csv"
                if file_is_valid(history_path):
                    hist = pd.read_csv(history_path)
                    hist["display_model"] = model_name
                    hist["fold"] = fold_id
                    cv_history_frames.append(hist)
                print(f"Fold {fold_id}/{CV_N_SPLITS} | {model_name}: predição retomada do cache")
            else:
                if force_cv:
                    FORCE_STAGES.add(f"train_{ledger_model_name}")
                model = instantiate_cv_model(model_name)
                loader = mitigated_loader if model_name == "EmoT_Reweighted" else standard_loader
                trained, hist, _ = fit_model(
                    ledger_model_name,
                    model,
                    loader,
                    X_fold_val,
                    y_fold_val,
                    fold_criterion,
                )
                fold_pred, fold_prob, fold_logits = predict_model(trained, X_fold_test)
                hist["display_model"] = model_name
                hist["fold"] = fold_id
                cv_history_frames.append(hist)
                atomic_npz_save(
                    prediction_cache,
                    test_idx=fold_test_idx,
                    y_true=y_fold_test,
                    y_pred=fold_pred,
                    y_prob=fold_prob,
                    logits=fold_logits,
                )
                del trained, model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            cv_records.append(
                {
                    "model": model_name,
                    "fold": fold_id,
                    "y_true": y_fold_test.copy(),
                    "y_pred": fold_pred.copy(),
                }
            )
            prediction_frame = pd.DataFrame(
                {
                    "sample_idx": fold_test_idx,
                    "fold": fold_id,
                    "model": model_name,
                    "y_true": y_fold_test,
                    "y_pred": fold_pred,
                }
            )
            for class_id, class_name in enumerate(CLASS_NAMES):
                prediction_frame[f"prob_{class_name}"] = fold_prob[:, class_id]
            cv_prediction_frames.append(prediction_frame)
            LEDGER.register_artifact(RUN_ID, "cross_validation_5fold", prediction_cache, kind="cv_predictions")

        del X_fold_train_raw, X_fold_val_raw, X_fold_test_raw
        del X_fold_train, X_fold_val, X_fold_test
        del fold_dataset, standard_loader, mitigated_loader, fold_criterion
        gc.collect()
finally:
    CKPT_DIR, LOG_DIR, mu, sigma = original_ckpt_dir, original_log_dir, main_mu, main_sigma

cv_fold_metrics, cv_summary, cv_confusions = aggregate_cv_results(
    cv_records,
    labels=list(range(N_CLASSES)),
)
cv_pooled_predictions = pd.concat(cv_prediction_frames, ignore_index=True).sort_values(["model", "sample_idx"])
cv_training_history = pd.concat(cv_history_frames, ignore_index=True) if cv_history_frames else pd.DataFrame()
cv_balance_by_fold = pd.DataFrame(cv_balance_rows)

expected_rows = len(meta_valid) * len(CV_MODELS)
if len(cv_pooled_predictions) != expected_rows:
    raise RuntimeError(f"Predições CV incompletas: esperado={expected_rows}, obtido={len(cv_pooled_predictions)}")
coverage = cv_pooled_predictions.groupby("model")["sample_idx"].agg(["count", "nunique"])
if not ((coverage["count"] == len(meta_valid)) & (coverage["nunique"] == len(meta_valid))).all():
    raise RuntimeError(f"Cobertura CV inválida:\n{coverage}")

atomic_write_text(TABLE_DIR / "cv_fold_metrics.csv", cv_fold_metrics.to_csv(index=False))
atomic_write_text(TABLE_DIR / "cv_summary_mean_std.csv", cv_summary.to_csv(index=False))
atomic_write_text(TABLE_DIR / "cv_pooled_predictions.csv", cv_pooled_predictions.to_csv(index=False))
atomic_write_text(TABLE_DIR / "cv_training_history.csv", cv_training_history.to_csv(index=False))
atomic_write_text(TABLE_DIR / "cv_balance_by_fold.csv", cv_balance_by_fold.to_csv(index=False))

for model_name, matrices in cv_confusions.items():
    safe_model = safe_slug(model_name)
    for matrix_name, matrix in matrices.items():
        matrix_frame = pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES)
        atomic_write_text(TABLE_DIR / f"cv_confusion_{matrix_name}_{safe_model}.csv", matrix_frame.to_csv())
    annotations = np.empty_like(matrices["mean"], dtype=object)
    for row_idx in range(N_CLASSES):
        for col_idx in range(N_CLASSES):
            annotations[row_idx, col_idx] = f"{matrices['mean'][row_idx, col_idx]:.2f}\n±{matrices['std'][row_idx, col_idx]:.2f}"
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        matrices["mean"],
        annot=annotations,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_title(f"5-fold: matriz de confusão média ± DP — {model_name}")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"cv_confusion_mean_{safe_model}.png", bbox_inches="tight")
    fig.savefig(CV_FIG_DIR / f"cv_confusion_mean_{safe_model}.png", bbox_inches="tight")
    plt.show()

cv_group_metric_frames = []
cv_group_metric_by_fold_frames = []
cv_group_emotion_frames = []
cv_confusion_pair_frames = []
for cv_model_name in CV_MODELS:
    pooled_model = cv_pooled_predictions[cv_pooled_predictions["model"] == cv_model_name].sort_values("sample_idx").reset_index(drop=True)
    pooled_model_meta = meta_valid.iloc[pooled_model["sample_idx"].to_numpy(dtype=int)].reset_index(drop=True)
    pooled_model_y_true = pooled_model["y_true"].to_numpy(dtype=int)
    pooled_model_y_pred = pooled_model["y_pred"].to_numpy(dtype=int)
    pooled_model_prob = pooled_model[[f"prob_{name}" for name in CLASS_NAMES]].to_numpy(dtype=float)

    for attribute in PROTECTED_ATTRIBUTES_DEFINITIVE:
        group_table = metrics_by_group(
            pooled_model_y_true,
            pooled_model_y_pred,
            pooled_model_prob,
            pooled_model_meta[attribute].values,
            min_n=CV_MIN_GROUP_N,
        )
        if len(group_table):
            group_table.insert(0, "model", cv_model_name)
            group_table.insert(1, "protected_attribute", attribute)
            cv_group_metric_frames.append(group_table)

    model_emotion_table = group_emotion_metrics(
        pooled_model_meta,
        pooled_model_y_true,
        pooled_model_y_pred,
        group_attributes=PROTECTED_ATTRIBUTES_DEFINITIVE,
        class_names=CLASS_NAMES,
        min_support=CV_MIN_EMOTION_SUPPORT,
    )
    model_emotion_table.insert(0, "model", cv_model_name)
    cv_group_emotion_frames.append(model_emotion_table)

    model_confusion_pairs = confusion_pairs_by_group(
        pooled_model_meta,
        pooled_model_y_true,
        pooled_model_y_pred,
        group_attributes=PROTECTED_ATTRIBUTES_DEFINITIVE,
        class_names=CLASS_NAMES,
        min_true_support=CV_MIN_EMOTION_SUPPORT,
    )
    if len(model_confusion_pairs):
        model_confusion_pairs.insert(0, "model", cv_model_name)
        cv_confusion_pair_frames.append(model_confusion_pairs)

    for fold_id in range(1, CV_N_SPLITS + 1):
        fold_predictions = pooled_model[pooled_model["fold"] == fold_id].sort_values("sample_idx").reset_index(drop=True)
        fold_metadata = meta_valid.iloc[fold_predictions["sample_idx"].to_numpy(dtype=int)].reset_index(drop=True)
        fold_y_true = fold_predictions["y_true"].to_numpy(dtype=int)
        fold_y_pred = fold_predictions["y_pred"].to_numpy(dtype=int)
        fold_prob = fold_predictions[[f"prob_{name}" for name in CLASS_NAMES]].to_numpy(dtype=float)
        for attribute in PROTECTED_ATTRIBUTES_DEFINITIVE:
            fold_group_table = metrics_by_group(
                fold_y_true,
                fold_y_pred,
                fold_prob,
                fold_metadata[attribute].values,
                min_n=max(10, CV_MIN_GROUP_N // CV_N_SPLITS),
            )
            if len(fold_group_table):
                fold_group_table.insert(0, "model", cv_model_name)
                fold_group_table.insert(1, "fold", fold_id)
                fold_group_table.insert(2, "protected_attribute", attribute)
                cv_group_metric_by_fold_frames.append(fold_group_table)

cv_group_metrics = pd.concat(cv_group_metric_frames, ignore_index=True)
cv_group_metrics_by_fold = pd.concat(cv_group_metric_by_fold_frames, ignore_index=True)
cv_group_emotion_metrics = pd.concat(cv_group_emotion_frames, ignore_index=True)
cv_confusion_pairs = pd.concat(cv_confusion_pair_frames, ignore_index=True)
cv_group_gap_summary = cv_group_metrics.groupby(["model", "protected_attribute"]).agg(
    n_groups=("group", "nunique"),
    worst_group_accuracy=("accuracy", "min"),
    best_group_accuracy=("accuracy", "max"),
    worst_group_macro_f1=("macro_f1", "min"),
    best_group_macro_f1=("macro_f1", "max"),
).reset_index()
cv_group_gap_summary["accuracy_gap"] = cv_group_gap_summary["best_group_accuracy"] - cv_group_gap_summary["worst_group_accuracy"]
cv_group_gap_summary["macro_f1_gap"] = cv_group_gap_summary["best_group_macro_f1"] - cv_group_gap_summary["worst_group_macro_f1"]
cv_group_metrics_fold_summary = cv_group_metrics_by_fold.groupby(["model", "protected_attribute", "group"]).agg(
    n_folds=("fold", "nunique"),
    n_mean=("n", "mean"),
    accuracy_mean=("accuracy", "mean"),
    accuracy_std=("accuracy", "std"),
    macro_f1_mean=("macro_f1", "mean"),
    macro_f1_std=("macro_f1", "std"),
    ece_mean=("ece", "mean"),
    ece_std=("ece", "std"),
).reset_index()

pooled_emot = cv_pooled_predictions[cv_pooled_predictions["model"] == "EmoT"].sort_values("sample_idx").reset_index(drop=True)
pooled_emot_meta = meta_valid.iloc[pooled_emot["sample_idx"].to_numpy(dtype=int)].reset_index(drop=True)
pooled_emot_y_true = pooled_emot["y_true"].to_numpy(dtype=int)
pooled_emot_y_pred = pooled_emot["y_pred"].to_numpy(dtype=int)
pooled_emot_prob = pooled_emot[[f"prob_{name}" for name in CLASS_NAMES]].to_numpy(dtype=float)
cv_top_confusions = cv_confusion_pairs[cv_confusion_pairs["model"] == "EmoT"].groupby(["protected_attribute", "group"], group_keys=False).head(TOP_CONFUSIONS_PER_GROUP)

atomic_write_text(TABLE_DIR / "cv_group_metrics.csv", cv_group_metrics.to_csv(index=False))
atomic_write_text(TABLE_DIR / "cv_group_metrics_by_fold.csv", cv_group_metrics_by_fold.to_csv(index=False))
atomic_write_text(TABLE_DIR / "cv_group_metrics_fold_summary.csv", cv_group_metrics_fold_summary.to_csv(index=False))
atomic_write_text(TABLE_DIR / "cv_group_gap_summary.csv", cv_group_gap_summary.to_csv(index=False))
atomic_write_text(TABLE_DIR / "cv_group_emotion_metrics.csv", cv_group_emotion_metrics.to_csv(index=False))
atomic_write_text(TABLE_DIR / "cv_confusion_pairs_by_group.csv", cv_confusion_pairs.to_csv(index=False))
atomic_write_text(TABLE_DIR / "cv_top_confusions_to_discuss.csv", cv_top_confusions.to_csv(index=False))

for attribute in PROTECTED_ATTRIBUTES_DEFINITIVE:
    attribute_values = pooled_emot_meta[attribute].fillna("Unknown").astype(str).to_numpy()
    for group in pd.unique(attribute_values):
        group_mask = attribute_values == group
        if int(group_mask.sum()) < CV_MIN_GROUP_N:
            continue
        safe_group = safe_slug(group)
        cm = confusion_matrix(
            pooled_emot_y_true[group_mask],
            pooled_emot_y_pred[group_mask],
            labels=list(range(N_CLASSES)),
        ).astype(float)
        cm = np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm), where=cm.sum(axis=1, keepdims=True) > 0)
        cm_frame = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
        atomic_write_text(TABLE_DIR / f"cv_confusion_EmoT_{safe_slug(attribute)}_{safe_group}.csv", cm_frame.to_csv())
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_title(f"EmoT 5-fold — {attribute}={group} (n={int(group_mask.sum())})")
        ax.set_xlabel("Predito")
        ax.set_ylabel("Verdadeiro")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        fig.savefig(FIG_DIR / f"cv_confusion_EmoT_{safe_slug(attribute)}_{safe_group}.png", bbox_inches="tight")
        plt.close(fig)

emot_comparison = cv_summary[cv_summary["model"].isin(["EmoT", "EmoT_Reweighted"])].copy()
if set(emot_comparison["model"]) == {"EmoT", "EmoT_Reweighted"}:
    baseline_row = emot_comparison.set_index("model").loc["EmoT"]
    mitigated_row = emot_comparison.set_index("model").loc["EmoT_Reweighted"]
    race_gaps = cv_group_gap_summary[cv_group_gap_summary["protected_attribute"] == "Race"].set_index("model")
    race_accuracy_gap_delta = (
        race_gaps.loc["EmoT_Reweighted", "accuracy_gap"] - race_gaps.loc["EmoT", "accuracy_gap"]
        if {"EmoT", "EmoT_Reweighted"}.issubset(race_gaps.index)
        else np.nan
    )
    race_macro_f1_gap_delta = (
        race_gaps.loc["EmoT_Reweighted", "macro_f1_gap"] - race_gaps.loc["EmoT", "macro_f1_gap"]
        if {"EmoT", "EmoT_Reweighted"}.issubset(race_gaps.index)
        else np.nan
    )
    balance_mitigation_comparison = pd.DataFrame([
        {
            "comparison": "EmoT_Reweighted - EmoT",
            "macro_f1_delta": mitigated_row["macro_f1_mean"] - baseline_row["macro_f1_mean"],
            "balanced_accuracy_delta": mitigated_row["balanced_accuracy_mean"] - baseline_row["balanced_accuracy_mean"],
            "accuracy_delta": mitigated_row["accuracy_mean"] - baseline_row["accuracy_mean"],
            "race_accuracy_gap_delta": race_accuracy_gap_delta,
            "race_macro_f1_gap_delta": race_macro_f1_gap_delta,
            "interpretation_rule": "A mitigação é favorável somente se os gaps diminuírem sem perda global substantivamente inaceitável.",
        }
    ])
else:
    balance_mitigation_comparison = pd.DataFrame()
atomic_write_text(TABLE_DIR / "balance_mitigation_cv_comparison.csv", balance_mitigation_comparison.to_csv(index=False))

literature_template = pd.DataFrame([
    {
        "study": "Esta dissertação — EmoT",
        "year": time.strftime("%Y"),
        "dataset": "CREMA-D",
        "modality": "visual-only landmarks",
        "identity_disjoint": True,
        "evaluation_protocol": "5-fold actor-disjoint CV",
        "metric_name": "macro_f1",
        "metric_mean": float(cv_summary.set_index("model").loc["EmoT", "macro_f1_mean"]),
        "metric_std": float(cv_summary.set_index("model").loc["EmoT", "macro_f1_std"]),
        "source_doi_or_url": "resultado desta execução",
        "directly_comparable": True,
        "validation_status": "generated",
    },
    {
        "study": "PREENCHER_APENAS_COM_FONTE_PRIMARIA_VALIDADA",
        "year": np.nan,
        "dataset": "CREMA-D",
        "modality": "visual-only required",
        "identity_disjoint": np.nan,
        "evaluation_protocol": "documentar split/folds",
        "metric_name": "documentar exatamente",
        "metric_mean": np.nan,
        "metric_std": np.nan,
        "source_doi_or_url": "obrigatório",
        "directly_comparable": False,
        "validation_status": "pending documentary validation",
    },
])
atomic_write_text(TABLE_DIR / "literature_visual_only_comparison_template.csv", literature_template.to_csv(index=False))

observation_lines = [
    "# Observações quantitativas para discussão",
    "",
    "As linhas abaixo são achados descritivos gerados automaticamente. Não implicam causalidade e devem ser interpretadas com suporte teórico e validação humana.",
    "",
]
for row in cv_top_confusions.itertuples(index=False):
    observation_lines.append(
        f"- {row.protected_attribute}={row.group}: {row.true_emotion} confundida com {row.predicted_emotion} "
        f"em {100 * row.error_rate_within_true:.1f}% dos casos verdadeiros (n={row.count}/{row.true_emotion_support})."
    )
atomic_write_text(REPORT_DIR / "cv_observations_to_discuss.md", "\n".join(observation_lines) + "\n")

display(cv_fold_metrics)
display(cv_summary)
display(cv_group_metrics)
display(cv_group_emotion_metrics.head(60))
display(cv_top_confusions.head(60))
display(balance_mitigation_comparison)
log_event(
    "cross_validation_5fold",
    start_time=cross_validation_start,
    n_folds=int(CV_N_SPLITS),
    n_models=int(len(CV_MODELS)),
    n_training_jobs=int(CV_N_SPLITS * len(CV_MODELS)),
    outer_stratification=cv_stratification,
)
'''


CAPTUM_SOURCE = r'''
captum_validation_start = time.time()
from captum.attr import IntegratedGradients as CaptumIntegratedGradients

CAPTUM_CACHE = XAI_DIR / "xai_captum_integrated_gradients_validation.npz"
captum_n = min(int(CAPTUM_VERIFY_SAMPLES), len(X_ig))
if file_is_valid(CAPTUM_CACHE) and "captum_integrated_gradients_validation" not in FORCE_STAGES:
    with np.load(CAPTUM_CACHE, allow_pickle=False) as cached:
        captum_attr = cached["captum_attr"].astype(np.float32)
        captum_targets = cached["targets"].astype(int)
    print(f"Validação Captum retomada: {CAPTUM_CACHE}")
else:
    captum_engine = CaptumIntegratedGradients(emot_model)
    captum_batches = []
    captum_targets = xai_preds_ig[:captum_n].astype(int)
    for start in tqdm(range(0, captum_n, CAPTUM_BATCH_SIZE), desc="Captum Integrated Gradients"):
        stop = min(captum_n, start + CAPTUM_BATCH_SIZE)
        inputs = torch.tensor(X_ig[start:stop], dtype=torch.float32, device=DEVICE, requires_grad=True)
        baselines = torch.zeros_like(inputs)
        targets = torch.tensor(captum_targets[start:stop], dtype=torch.long, device=DEVICE)
        attribution = captum_engine.attribute(
            inputs,
            baselines=baselines,
            target=targets,
            n_steps=IG_STEPS,
            internal_batch_size=max(1, min(CAPTUM_INTERNAL_BATCH_SIZE, len(inputs) * IG_STEPS)),
        )
        captum_batches.append(attribution.abs().detach().cpu().numpy())
        del inputs, baselines, targets, attribution
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    captum_attr = np.concatenate(captum_batches, axis=0).astype(np.float32)
    atomic_npz_save(CAPTUM_CACHE, captum_attr=captum_attr, targets=captum_targets)

manual_attr_comparison = ig_attr[:captum_n].astype(np.float32)
captum_flat = captum_attr.reshape(captum_n, -1)
manual_flat = manual_attr_comparison.reshape(captum_n, -1)
dot = (captum_flat * manual_flat).sum(axis=1)
denominator = np.linalg.norm(captum_flat, axis=1) * np.linalg.norm(manual_flat, axis=1)
cosine_similarity = np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator > 0)

captum_landmarks = aggregate_attr_to_landmarks(captum_attr)
manual_landmarks = aggregate_attr_to_landmarks(manual_attr_comparison)
rank_correlations = []
for sample_index in range(captum_n):
    left = pd.Series(captum_landmarks[sample_index]).rank(method="average").to_numpy()
    right = pd.Series(manual_landmarks[sample_index]).rank(method="average").to_numpy()
    correlation = np.corrcoef(left, right)[0, 1]
    rank_correlations.append(float(correlation) if np.isfinite(correlation) else np.nan)

captum_validation = pd.DataFrame({
    "sample_idx_within_ig": np.arange(captum_n),
    "target_class": captum_targets,
    "target_emotion": [CLASS_NAMES[value] for value in captum_targets],
    "cosine_similarity_manual_vs_captum": cosine_similarity,
    "spearman_landmark_rank_manual_vs_captum": rank_correlations,
})
atomic_write_text(TABLE_DIR / "xai_captum_integrated_gradients_validation.csv", captum_validation.to_csv(index=False))
atomic_write_text(XAI_DIR / "xai_captum_integrated_gradients_validation.csv", captum_validation.to_csv(index=False))

captum_report = f"""# Validação do Integrated Gradients com Captum

- Amostras comparadas: {captum_n}
- Passos de integração: {IG_STEPS}
- Similaridade cosseno média entre implementação manual e Captum: {captum_validation['cosine_similarity_manual_vs_captum'].mean():.4f}
- Correlação de postos média dos landmarks: {captum_validation['spearman_landmark_rank_manual_vs_captum'].mean():.4f}

Esta verificação documenta correspondência numérica entre os dois motores. Diferenças residuais podem decorrer da regra de integração utilizada internamente pelo Captum.
"""
atomic_write_text(REPORT_DIR / "xai_captum_validation.md", captum_report)
display(captum_validation.describe(include="all"))
log_event("captum_integrated_gradients_validation", start_time=captum_validation_start, n_samples=int(captum_n), ig_steps=int(IG_STEPS))
'''


DEFINITIVE_SUMMARY_SOURCE = r'''
definitive_summary_lines = [
    "# Índice da execução definitiva",
    "",
    f"- Run ID: `{RUN_ID}`",
    f"- Base válida: {len(meta_valid):,} vídeos; execução integral: {FULL_RUN}.",
    f"- Protocolo principal de robustez: {CV_N_SPLITS}-fold por ator, estratificação `{cv_stratification}`.",
    f"- Melhor modelo no CV por macro-F1 médio: {cv_summary.iloc[0]['model']} ({cv_summary.iloc[0]['macro_f1_mean']:.4f} ± {cv_summary.iloc[0]['macro_f1_std']:.4f}).",
    "- Modalidade: exclusivamente visual; nenhuma feature de áudio foi utilizada.",
    "- Fairness: Race, Ethnicity, Sex, age_bin e análises interseccionais.",
    "- XAI: Attention Rollout, Gradient x Input, Integrated Gradients, Captum, oclusão regional e Deletion Test.",
    "- O arquivo `cv_observations_to_discuss.md` contém achados descritivos; eles não devem ser apresentados como causalidade.",
    "- O arquivo `literature_visual_only_comparison_template.csv` exige validação documental de cada comparador externo.",
    "- Nenhum conteúdo foi escrito no Overleaf.",
]
atomic_write_text(REPORT_DIR / "definitive_execution_index.md", "\n".join(definitive_summary_lines) + "\n")
print("\n".join(definitive_summary_lines))
'''


def build_notebook() -> dict:
    notebook = build_resumable_notebook()
    cells = notebook["cells"]

    set_source(
        cells[0],
        "# CREMA-D + EmoT — Colab definitivo da dissertação\n\n"
        "Execução integral visual-only, resumível no Google Drive, com SQLite, "
        "holdout por ator, validação cruzada de cinco folds por ator, fairness, XAI, "
        "Captum, evidências, logs, prints e pacote final. Os notebooks 06 e 07 "
        "permanecem preservados; este notebook não escreve no Overleaf.\n",
    )

    pip_matches = [
        index
        for index, cell in enumerate(cells)
        if cell.get("cell_type") == "code"
        and cell_source(cell).lstrip().startswith("%pip install")
    ]
    if len(pip_matches) != 1:
        raise RuntimeError(f"Expected one pip cell; found {pip_matches}")
    pip_index = pip_matches[0]
    pip_source = cell_source(cells[pip_index])
    if "captum" not in pip_source:
        set_source(cells[pip_index], pip_source.rstrip() + " captum\n")

    analysis_source = ANALYSIS_MODULE.read_text(encoding="utf-8")
    analysis_runtime_cell = code_cell(
        "# Módulo científico testável embutido em scripts/definitive_analysis.py\n"
        f"exec(compile({analysis_source!r}, 'definitive_analysis.py', 'exec'), globals())\n"
        "print('Módulo definitivo de análise carregado.')\n",
        role="definitive_analysis_runtime",
    )
    setup_index = find_cell(cells, "EXPERIMENT_NAME =")
    cells.insert(setup_index, analysis_runtime_cell)

    setup_index = find_cell(cells, "EXPERIMENT_NAME =")
    setup = cell_source(cells[setup_index])
    setup = replace_or_fail(
        setup,
        "FULL_RUN = True\nFAST_DEV_SAMPLES = 360\n",
        """FULL_RUN = True
DEFINITIVE_RUN = True
ALLOW_DEV_MODE = False
FAST_DEV_SAMPLES = 360

# Contrato científico pós-qualificação.
CV_N_SPLITS = 5
CV_VALIDATION_FRACTION = 0.15
CV_MODELS = ("FlatMLP", "TemporalCNN1D", "EmoT", "EmoT_Reweighted")
PROTECTED_ATTRIBUTES_DEFINITIVE = ("Race", "Ethnicity", "Sex", "age_bin")
CV_MIN_GROUP_N = 30
CV_MIN_EMOTION_SUPPORT = 10
TOP_CONFUSIONS_PER_GROUP = 5
CAPTUM_VERIFY_SAMPLES = 64
CAPTUM_BATCH_SIZE = 8
CAPTUM_INTERNAL_BATCH_SIZE = 64

if DEFINITIVE_RUN and not FULL_RUN and not ALLOW_DEV_MODE:
    raise RuntimeError("A execução definitiva exige FULL_RUN=True. Use ALLOW_DEV_MODE=True apenas para depuração explícita.")
""",
        "definitive configuration",
    )
    setup = replace_or_fail(
        setup,
        'EXPERIMENT_NAME = "cremad_fairness_xai_emot"',
        'EXPERIMENT_NAME = "cremad_fairness_xai_emot_definitive"',
        "definitive experiment name",
    )
    setup = replace_or_fail(
        setup,
        'recovery_config = {\n    "seed": SEED,\n',
        'recovery_config = {\n    "seed": SEED,\n    "definitive_run": DEFINITIVE_RUN,\n    "cv_n_splits": CV_N_SPLITS,\n    "cv_models": list(CV_MODELS),\n    "captum_verify_samples": CAPTUM_VERIFY_SAMPLES,\n',
        "definitive recovery hash",
    )
    setup = replace_or_fail(
        setup,
        'run_config = {\n    "run_id": RUN_ID,\n',
        'run_config = {\n    "run_id": RUN_ID,\n    "definitive_run": DEFINITIVE_RUN,\n    "cv_n_splits": CV_N_SPLITS,\n    "cv_models": list(CV_MODELS),\n    "protected_attributes_definitive": list(PROTECTED_ATTRIBUTES_DEFINITIVE),\n    "captum_verify_samples": CAPTUM_VERIFY_SAMPLES,\n',
        "definitive run config",
    )
    set_source(cells[setup_index], setup)

    download_markdown_index = find_cell(cells, "## 7. Download dos vídeos")
    cells[download_markdown_index:download_markdown_index] = [
        markdown_cell(
            "## Escopo, composição da base e tempo estimado\n\n"
            "O CREMA-D é áudio-visual, mas o experimento abaixo é estritamente visual-only. "
            "Esta seção materializa o escopo, as distribuições por raça/etnia, sexo, faixa "
            "etária e emoção, além das interseções e da estimativa para A100.\n",
            role="definitive_scope",
        ),
        code_cell(SCOPE_AND_DISTRIBUTION_SOURCE, role="scope_and_distribution"),
    ]

    loader_index = find_cell(cells, "fair_train_loader =")
    cells[loader_index + 1:loader_index + 1] = [
        markdown_cell(
            "## Documentação do balanceamento\n\n"
            "A execução não descarta vídeos. Todos os modelos usam loss ponderado por emoção; "
            "a variante `EmoT_Reweighted` acrescenta amostragem por emoção × Race. As tabelas "
            "abaixo tornam os pesos e a decisão metodológica auditáveis.\n",
            role="balance_documentation",
        ),
        code_cell(BALANCE_SOURCE, role="balance_strategy"),
    ]

    robustness_index = find_cell(cells, "## 21. Robustez por alto consenso facial")
    cv_stage_source = wrap_stage(
        "cross_validation_5fold",
        CROSS_VALIDATION_SOURCE,
        '[TABLE_DIR / "cv_fold_metrics.csv", TABLE_DIR / "cv_summary_mean_std.csv", TABLE_DIR / "cv_pooled_predictions.csv", TABLE_DIR / "cv_group_emotion_metrics.csv", TABLE_DIR / "cv_confusion_pairs_by_group.csv"]',
    )
    cells[robustness_index:robustness_index] = [
        markdown_cell(
            "## Validação cruzada definitiva: cinco folds por ator\n\n"
            "Cada ator aparece no teste de exatamente um fold e nunca aparece simultaneamente "
            "em treino, validação ou teste. Os quatro modelos são treinados em todos os folds. "
            "Checkpoints por época e predições por fold permitem retomar do último ponto íntegro.\n",
            role="cross_validation_5fold",
        ),
        code_cell(cv_stage_source, stage="cross_validation_5fold"),
    ]

    ig_index = find_cell(cells, "## 24. Fairness explicativo")
    captum_stage_source = wrap_stage(
        "captum_integrated_gradients_validation",
        CAPTUM_SOURCE,
        '[XAI_DIR / "xai_captum_integrated_gradients_validation.npz", TABLE_DIR / "xai_captum_integrated_gradients_validation.csv"]',
    )
    cells[ig_index:ig_index] = [
        markdown_cell(
            "## Verificação independente do Integrated Gradients com Captum\n\n"
            "Uma amostra estratificada das explicações é recalculada pelo Captum. A execução "
            "salva similaridade cosseno e correlação de postos dos landmarks em relação à "
            "implementação manual, produzindo evidência numérica de consistência.\n",
            role="captum_validation",
        ),
        code_cell(captum_stage_source, stage="captum_integrated_gradients_validation"),
    ]

    logs_index = find_cell(cells, "## 29. Logs de tempo")
    cells[logs_index:logs_index] = [
        markdown_cell(
            "## Síntese e índice dos artefatos definitivos\n\n"
            "Este índice aponta os resultados que serão usados na redação posterior. A redação "
            "da dissertação e qualquer mudança no Overleaf continuam condicionadas à validação do usuário.\n",
            role="definitive_summary",
        ),
        code_cell(DEFINITIVE_SUMMARY_SOURCE, role="definitive_summary_index"),
    ]

    # nbformat 4.5+ requires stable cell IDs.  The source and position make the
    # generated notebook deterministic while avoiding random IDs on each run.
    for index, cell in enumerate(cells):
        if not cell.get("id"):
            digest = hashlib.sha1(
                f"{index}\n{cell.get('cell_type')}\n{cell_source(cell)}".encode("utf-8")
            ).hexdigest()[:10]
            cell["id"] = f"cremad-{index:02d}-{digest}"

    notebook["cells"] = cells
    notebook.setdefault("metadata", {})["accelerator"] = "GPU"
    notebook["metadata"].setdefault("colab", {})["gpuType"] = "A100"
    notebook["metadata"]["colab"]["provenance"] = [
        "generated-from-notebook-06",
        "extends-notebook-07-resilience",
    ]
    notebook["metadata"]["cremad_definitive"] = {
        "schema": 1,
        "full_run_default": True,
        "cross_validation_folds": 5,
        "cross_validation_unit": "ActorID",
        "models_per_fold": 4,
        "visual_only": True,
        "captum_validation": True,
        "state_database": "cremad_fairness_xai/state/experiment_state.sqlite",
        "overleaf_write": False,
    }
    return notebook


def main() -> None:
    if not ANALYSIS_MODULE.is_file():
        raise FileNotFoundError(ANALYSIS_MODULE)
    notebook = build_notebook()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Notebook generated: {OUTPUT}")


if __name__ == "__main__":
    main()
