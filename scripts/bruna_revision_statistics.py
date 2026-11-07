"""Actor-cluster analysis of preserved CREMA-D out-of-fold predictions.

Dependencies are limited to NumPy and pandas. This module never trains a model.
The intervals are conditional on the saved out-of-fold fitted models, not a
replacement for repeated training or external validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ("angry", "disgust", "fear", "happy", "neutral", "sad")
ATTRIBUTES = ("Race", "Ethnicity", "Sex", "age_bin")
PROB_COLUMNS = tuple("prob_" + name for name in CLASSES)


def metrics_from_confusion(cm: np.ndarray) -> dict[str, np.ndarray]:
    """Metrics with six prespecified classes; undefined class F1 is zero."""
    cm = np.asarray(cm, dtype=float)
    if cm.shape[-2:] != (6, 6) or np.any(cm < 0) or not np.isfinite(cm).all():
        raise ValueError("Expected finite nonnegative [...,6,6] confusion matrices")
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    true = cm.sum(axis=-1)
    pred = cm.sum(axis=-2)
    f1 = np.divide(2 * tp, true + pred, out=np.zeros_like(tp), where=(true + pred) > 0)
    recall = np.divide(tp, true, out=np.zeros_like(tp), where=true > 0)
    total = cm.sum(axis=(-1, -2))
    accuracy = np.divide(tp.sum(axis=-1), total, out=np.zeros_like(total), where=total > 0)
    return {"accuracy": accuracy, "macro_f1": f1.mean(axis=-1), "balanced_accuracy": recall.mean(axis=-1)}


def confusion(y_true, y_pred, weights=None):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) != len(y_pred) or np.any((y_true < 0) | (y_true > 5)) or np.any((y_pred < 0) | (y_pred > 5)):
        raise ValueError("Invalid labels or inconsistent lengths")
    return np.bincount(6 * y_true + y_pred, weights=weights, minlength=36).reshape(6, 6)


def actor_confusions(frame: pd.DataFrame, actor_ids: np.ndarray) -> np.ndarray:
    lookup = {actor: i for i, actor in enumerate(actor_ids)}
    codes = frame["ActorID"].map(lookup)
    if codes.isna().any():
        raise ValueError("Prediction refers to an unknown actor")
    index = codes.to_numpy(int) * 36 + frame["y_true"].to_numpy(int) * 6 + frame["y_pred"].to_numpy(int)
    return np.bincount(index, minlength=len(actor_ids) * 36).reshape(len(actor_ids), 6, 6)


def validate_inputs(predictions, manifest, assignments, expected_samples=None, expected_actors=None):
    """Align sample_idx to the persisted manifest and audit identity/fold labels."""
    required = {"sample_idx", "fold", "model", "y_true", "y_pred", *PROB_COLUMNS}
    if not required.issubset(predictions.columns):
        raise ValueError(f"Missing prediction columns: {sorted(required - set(predictions.columns))}")
    if not {"ActorID", "y", "file_stem", *ATTRIBUTES}.issubset(manifest.columns):
        raise ValueError("Manifest is missing labels, identities or demographic attributes")
    if not {"ActorID", "outer_test_fold", *ATTRIBUTES}.issubset(assignments.columns):
        raise ValueError("Assignments are missing identities, folds or demographics")
    if manifest["file_stem"].duplicated().any() or assignments["ActorID"].duplicated().any():
        raise ValueError("Manifest sample or actor assignment is duplicated")
    if expected_samples is not None and len(manifest) != expected_samples:
        raise ValueError("Unexpected manifest size")
    if expected_actors is not None and manifest["ActorID"].nunique() != expected_actors:
        raise ValueError("Unexpected actor count")
    for attr in ATTRIBUTES:
        if manifest.groupby("ActorID")[attr].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Demographic category varies inside actor: {attr}")
    if set(manifest.ActorID) != set(assignments.ActorID):
        raise ValueError("Actor sets disagree between manifest and assignments")
    actors = manifest.groupby("ActorID", sort=True).first()[list(ATTRIBUTES)]
    actors = actors.join(assignments.set_index("ActorID")[["outer_test_fold"]])
    reference = assignments.set_index("ActorID")
    for attr in ATTRIBUTES:
        if not actors[attr].fillna("Unknown").astype(str).equals(reference.loc[actors.index, attr].fillna("Unknown").astype(str)):
            raise ValueError(f"Actor demographic metadata disagrees: {attr}")
    by_model = {}
    expected = np.arange(len(manifest))
    for model, frame in predictions.groupby("model", sort=True):
        f = frame.sort_values("sample_idx").reset_index(drop=True)
        if not np.array_equal(f.sample_idx.to_numpy(), expected):
            raise ValueError(f"OOF samples not covered once for {model}")
        if not np.array_equal(f.y_true.to_numpy(), manifest.y.to_numpy()):
            raise ValueError(f"True labels disagree with manifest for {model}")
        probs = f[list(PROB_COLUMNS)].to_numpy(float)
        if not np.isfinite(probs).all() or probs.min() < -1e-7 or probs.max() > 1 + 1e-7 or not np.allclose(probs.sum(1), 1, atol=2e-6):
            raise ValueError(f"Invalid probabilities for {model}")
        if not np.array_equal(probs.argmax(1), f.y_pred.to_numpy()):
            raise ValueError(f"Predicted labels disagree with probability argmax for {model}")
        f["ActorID"] = manifest.ActorID.to_numpy()
        for attr in ATTRIBUTES:
            f[attr] = manifest[attr].fillna("Unknown").astype(str).to_numpy()
        actor_folds = f.ActorID.map(actors.outer_test_fold)
        if not np.array_equal(f.fold.to_numpy(), actor_folds.to_numpy()):
            raise ValueError(f"Fold assignments disagree for {model}")
        if f.groupby("ActorID").fold.nunique().gt(1).any():
            raise ValueError(f"Actor identity leaks across OOF folds for {model}")
        by_model[str(model)] = f
    if not {"EmoT", "TemporalCNN1D"}.issubset(by_model):
        raise ValueError("EmoT and TemporalCNN1D are required")
    return by_model, actors.reset_index()


def cluster_bootstrap_pair(cm_a, cm_b, n_boot=5000, seed=20261107, actor_equal_weight=False):
    """Resample whole actors and apply identical actor multiplicities to models."""
    a, b = np.asarray(cm_a, float), np.asarray(cm_b, float)
    if a.shape != b.shape or a.ndim != 3 or a.shape[1:] != (6, 6):
        raise ValueError("Paired actor confusion arrays must have identical [actors,6,6] shapes")
    if not np.array_equal(a.sum((1, 2)), b.sum((1, 2))):
        raise ValueError("Models have different per-actor support")
    if np.any(a.sum((1, 2)) <= 0):
        raise ValueError("Every actor must contribute videos")
    if actor_equal_weight:
        totals = a.sum((1, 2))[:, None, None]
        a, b = a / totals, b / totals
    rng = np.random.default_rng(seed)
    n = len(a)
    multiplicities = rng.multinomial(n, np.full(n, 1 / n), size=n_boot)
    samples_a = metrics_from_confusion((multiplicities @ a.reshape(n, 36)).reshape(n_boot, 6, 6))
    samples_b = metrics_from_confusion((multiplicities @ b.reshape(n, 36)).reshape(n_boot, 6, 6))
    observed_a, observed_b = metrics_from_confusion(a.sum(0)), metrics_from_confusion(b.sum(0))
    rows, distributions = [], {}
    for metric in ("accuracy", "macro_f1"):
        delta = samples_a[metric] - samples_b[metric]
        for target, point, samples in (("EmoT", observed_a[metric], samples_a[metric]), ("TemporalCNN1D", observed_b[metric], samples_b[metric]), ("EmoT_minus_TemporalCNN1D", observed_a[metric] - observed_b[metric], delta)):
            lo, hi = np.percentile(samples, [2.5, 97.5])
            rows.append({"estimand": "equal_actor_weight" if actor_equal_weight else "video_weighted", "metric": metric, "target": target, "observed": float(point), "ci_low": float(lo), "ci_high": float(hi), "n_boot": n_boot, "seed": seed, "n_actors": n})
        distributions[metric + "_delta"] = delta
    return pd.DataFrame(rows), pd.DataFrame(distributions)


def permute_actor_categories(categories, blocks, rng, fixed_categories=()):
    """Permute complete actor categories within OOF fold blocks."""
    categories, blocks = np.asarray(categories), np.asarray(blocks)
    if len(categories) != len(blocks):
        raise ValueError("Categories and fold blocks have different lengths")
    result = categories.copy()
    movable = ~np.isin(categories, tuple(fixed_categories))
    for block in np.unique(blocks):
        index = np.flatnonzero((blocks == block) & movable)
        result[index] = rng.permutation(categories[index])
    return result


def holm_adjust(p_values):
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    ranked = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    result = np.empty(len(p));result[order] = ranked
    return result


def demographic_permutations(cm, actors, n_perm=5000, seed=20261107, min_videos=30, min_actors=2, hold_unknown_fixed=False):
    """Four planned accuracy-gap tests, with Holm correction across attributes."""
    totals = cm.sum((1, 2)).astype(float)
    correct = np.diagonal(cm, axis1=1, axis2=2).sum(1).astype(float)
    blocks = actors.outer_test_fold.to_numpy()
    rows, support_rows = [], []
    for j, attr in enumerate(ATTRIBUTES):
        labels = actors[attr].fillna("Unknown").astype(str).to_numpy()
        eligible = []
        for group in sorted(np.unique(labels)):
            mask = labels == group
            na, nv = int(mask.sum()), int(totals[mask].sum())
            ok = na >= min_actors and nv >= min_videos and group.casefold() not in {'unknown','nan','none',''}
            support_rows.append({"attribute": attr, "group": group, "n_actors": na, "n_videos": nv, "eligible_for_permutation": ok, "low_actor_support_flag": na < 10})
            if ok:eligible.append(group)
        if len(eligible) < 2:
            raise ValueError(f"Fewer than two eligible groups for {attr}")
        def gap(cats):
            accuracies = [correct[cats == group].sum() / totals[cats == group].sum() for group in eligible]
            return float(max(accuracies) - min(accuracies))
        observed = gap(labels)
        rng = np.random.default_rng(seed + 100 + j)
        null = np.array([gap(permute_actor_categories(labels, blocks, rng, fixed_categories=('Unknown',) if hold_unknown_fixed else ())) for _ in range(n_perm)])
        p_value = (1 + np.count_nonzero(null >= observed - 1e-14)) / (n_perm + 1)
        rows.append({"attribute": attr, "observed_accuracy_gap": observed, "p_actor_permutation": p_value, "n_perm": n_perm, "seed": seed + 100 + j, "permutation_blocks": "outer_test_fold", "unknown_category_fixed":hold_unknown_fixed, "eligible_groups": "|".join(eligible), "min_actors": min_actors, "min_videos": min_videos})
    result = pd.DataFrame(rows)
    result["p_holm_four_attributes"] = holm_adjust(result.p_actor_permutation)
    return result, pd.DataFrame(support_rows)


def group_emotion_metrics(by_model):
    group_rows, emotion_rows = [], []
    for model, df in by_model.items():
        for attr in ATTRIBUTES:
            for group, g in df.groupby(attr, sort=True):
                cm = confusion(g.y_true, g.y_pred)
                met = metrics_from_confusion(cm)
                prob = g[list(PROB_COLUMNS)].to_numpy(float)
                confidence = prob.max(1)
                correct = g.y_true.to_numpy() == g.y_pred.to_numpy()
                bins = np.minimum((confidence * 10).astype(int), 9)
                ece = 0.
                for bi in range(10):
                    mask = bins == bi
                    if mask.any():ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
                group_rows.append({"model": model, "attribute": attr, "group": group, "n_videos": len(g), "n_actors": g.ActorID.nunique(), **{key:float(value) for key,value in met.items()}, "ece_10_bins": ece})
                for k, emotion in enumerate(CLASSES):
                    true = g.y_true.to_numpy() == k
                    pred = g.y_pred.to_numpy() == k
                    tp = int((true & pred).sum())
                    n_true, n_pred = int(true.sum()), int(pred.sum())
                    recall = tp / n_true if n_true else np.nan
                    precision = tp / n_pred if n_pred else 0.
                    emotion_rows.append({"model": model, "attribute": attr, "group": group, "emotion_id": k, "emotion": emotion, "n_videos_true": n_true, "n_actors_true": g.loc[true, "ActorID"].nunique(), "n_videos_predicted": n_pred, "true_positive": tp, "recall": recall, "precision": precision, "f1": 2*tp/(n_true+n_pred) if n_true+n_pred else 0.})
    return pd.DataFrame(group_rows), pd.DataFrame(emotion_rows)


def gap_eligibility_sensitivity(group_metrics, min_videos=30):
    """Preserve the historical rule and separately declare an actor-aware rule."""
    rows=[]
    for (model,attr),frame in group_metrics.groupby(['model','attribute'],sort=True):
        for policy in ('historical_videos_ge30','known_groups_actors_ge2_videos_ge30'):
            chosen=frame.loc[frame.n_videos>=min_videos]
            if policy!='historical_videos_ge30':
                chosen=chosen.loc[(chosen.n_actors>=2)&~chosen.group.astype(str).str.casefold().isin(['unknown','nan','none',''])]
            if len(chosen)<2:raise ValueError('Fewer than two eligible groups')
            row={'model':model,'attribute':attr,'eligibility_policy':policy,'groups':'|'.join(sorted(chosen.group.astype(str))),
                 'n_groups':len(chosen),'minimum_group_actors':int(chosen.n_actors.min()),'minimum_group_videos':int(chosen.n_videos.min())}
            for metric in ('accuracy','macro_f1'):
                row[metric+'_gap']=float(chosen[metric].max()-chosen[metric].min())
                row['worst_group_'+metric]=float(chosen[metric].min())
            rows.append(row)
    return pd.DataFrame(rows)


def run_analysis(input_dir, output_dir, n_boot=5000, n_perm=5000, seed=20261107):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = ["cv_pooled_predictions.csv", "manifest_valid_with_splits.csv", "cv_actor_assignments.csv", "cv_fold_metrics.csv"]
    inputs = {name:pd.read_csv(input_dir/name) for name in names}
    by_model, actors = validate_inputs(inputs[names[0]], inputs[names[1]], inputs[names[2]], expected_samples=7441, expected_actors=91)
    actor_ids = actors.ActorID.to_numpy()
    matrices = {model:actor_confusions(df, actor_ids) for model, df in by_model.items()}
    metrics_rows = []
    for model, df in by_model.items():
        for fold, part in df.groupby("fold"):
            computed = metrics_from_confusion(confusion(part.y_true, part.y_pred))
            saved = inputs["cv_fold_metrics.csv"].query("model == @model and fold == @fold")
            if len(saved) != 1 or any(not np.isclose(float(computed[key]), float(saved.iloc[0][key]), atol=1e-12) for key in computed):
                raise ValueError(f"Recomputed fold metrics disagree with record for {model} fold {fold}")
            metrics_rows.append({"model":model, "fold":int(fold), "n_videos":len(part), "n_actors":part.ActorID.nunique(), **{key:float(value) for key,value in computed.items()}})
    fold_metrics = pd.DataFrame(metrics_rows)
    paired = fold_metrics.pivot(index="fold",columns="model",values="macro_f1")[["EmoT","TemporalCNN1D"]].reset_index()
    paired["delta_macro_f1"] = paired.EmoT-paired.TemporalCNN1D
    paired.to_csv(output_dir/'paired_fold_macro_f1.csv',index=False)
    fold_metrics.to_csv(output_dir/'fold_metrics_recomputed.csv',index=False)
    bootstrap_results = []
    for equal in (False, True):
        stats, samples = cluster_bootstrap_pair(matrices['EmoT'],matrices['TemporalCNN1D'],n_boot,seed,actor_equal_weight=equal)
        bootstrap_results.append(stats)
        samples.to_csv(output_dir/f"bootstrap_deltas_{'equal_actor' if equal else 'video_weighted'}.csv",index=False)
    boot = pd.concat(bootstrap_results,ignore_index=True)
    boot.to_csv(output_dir/'actor_cluster_bootstrap.csv',index=False)
    permutation, support = demographic_permutations(matrices['EmoT'],actors,n_perm,seed,hold_unknown_fixed=True)
    permutation.to_csv(output_dir/'actor_demographic_permutation.csv',index=False)
    permutation_unknown_movable,_=demographic_permutations(matrices['EmoT'],actors,n_perm,seed,hold_unknown_fixed=False)
    permutation_unknown_movable.to_csv(output_dir/'actor_demographic_permutation_unknown_movable_sensitivity.csv',index=False)
    support.to_csv(output_dir/'actor_group_support.csv',index=False)
    group, emotion = group_emotion_metrics(by_model)
    group.to_csv(output_dir/'group_metrics_with_actor_support.csv',index=False)
    emotion.to_csv(output_dir/'emotion_group_recall_with_actor_support.csv',index=False)
    gaps=gap_eligibility_sensitivity(group)
    gaps.to_csv(output_dir/'group_gap_eligibility_sensitivity.csv',index=False)
    gaps.query("attribute == 'Race'").to_csv(output_dir/'racial_gap_all_models_two_eligibility_rules.csv',index=False)
    mitigation=[]
    for policy,frame in gaps.query("attribute == 'Race'").groupby('eligibility_policy'):
        indexed=frame.set_index('model')
        base,weighted=indexed.loc['EmoT'],indexed.loc['EmoT_Reweighted']
        mitigation.append({'eligibility_policy':policy,'groups':base['groups'],
            'EmoT_accuracy_gap':float(base.accuracy_gap),'Reweighted_accuracy_gap':float(weighted.accuracy_gap),
            'Reweighted_minus_EmoT_accuracy_gap':float(weighted.accuracy_gap-base.accuracy_gap),
            'EmoT_macro_f1_gap':float(base.macro_f1_gap),'Reweighted_macro_f1_gap':float(weighted.macro_f1_gap),
            'Reweighted_minus_EmoT_macro_f1_gap':float(weighted.macro_f1_gap-base.macro_f1_gap)})
    mitigation=pd.DataFrame(mitigation)
    mitigation.to_csv(output_dir/'mitigation_gap_sensitivity.csv',index=False)
    fold_support=[]
    for attr in ATTRIBUTES:
        for (fold,group_name),part in by_model['EmoT'].groupby(['fold',attr],sort=True):
            fold_support.append({'fold':int(fold),'attribute':attr,'group':group_name,'n_actors':int(part.ActorID.nunique()),'n_videos':len(part),'low_actor_support_flag':part.ActorID.nunique()<10})
    pd.DataFrame(fold_support).to_csv(output_dir/'actor_group_support_by_fold.csv',index=False)
    emotion_reference_audit=None
    reference_path=input_dir/'cv_group_emotion_metrics.csv'
    if reference_path.exists():
        saved=pd.read_csv(reference_path).rename(columns={'protected_attribute':'attribute'})
        keys=['model','attribute','group','emotion_id']
        merged=emotion.merge(saved,on=keys,suffixes=('_recomputed','_saved'),validate='one_to_one',how='outer',indicator=True)
        if not merged['_merge'].eq('both').all():raise ValueError('Emotion/group coverage differs from saved table')
        max_error=float(np.max(np.abs(merged.recall_recomputed-merged.recall_saved)))
        if not np.allclose(merged.recall_recomputed,merged.recall_saved,atol=1e-12,equal_nan=True):raise ValueError('Saved group/emotion recalls were not reproduced')
        if not np.array_equal(merged.n_videos_true,merged.support_true):raise ValueError('Saved group/emotion supports were not reproduced')
        emotion_reference_audit={'source':reference_path.name,'rows':len(merged),'max_abs_recall_error':max_error,'support_matches':True,'sha256':hashlib.sha256(reference_path.read_bytes()).hexdigest()}
    actor_metrics = []
    for model, cm in matrices.items():
        met = metrics_from_confusion(cm)
        for i, actor in enumerate(actor_ids):
            actor_metrics.append({"ActorID":int(actor),"model":model,"n_videos":int(cm[i].sum()),**{key:float(value[i]) for key,value in met.items()}})
    pd.DataFrame(actor_metrics).to_csv(output_dir/'metrics_per_actor.csv',index=False)
    summary = {
      'status':'completed','analysis':'actor_cluster_reanalysis_of_saved_OOF_predictions','run_id':'20260809_223452_c0613d3c',
      'n_samples_per_model':7441,'n_actors':91,'models':sorted(by_model),'n_boot':n_boot,'n_perm':n_perm,'seed':seed,
      'fold_audit':'Each sample once per model; same labels and folds as manifest and actor assignment; saved fold metrics independently reproduced.',
      'estimand_primary':'Macro-F1/accuracy on pooled OOF videos; actor resampling keeps all videos of each sampled actor; identical multiplicities for the model pair.',
      'estimand_sensitivity':'Pooled confusion matrix after normalizing each actor total to 1; each actor receives equal total weight. Not the average of actor-specific F1 values.',
      'interval':'Percentile actor-cluster bootstrap, 2.5th and 97.5th percentiles; conditional on saved fitted models and actor sample; no new training.',
      'demographic_test':'Accuracy gap among fixed eligible known groups (>=30 videos and >=2 actors); permute complete actor demographic categories within external fold; 4 p-values corrected by Holm. Main racial randomization fixes Unknown and its actor because Unknown denotes missing information, not a substantive racial category. A separately labelled sensitivity moves Unknown as a category in the null randomization. These are null randomizations, not demographic imputation. The actor remains included in other attributes and global metrics.',
      'analysis_plan_status':'Escolhas metodológicas explicitadas nesta reanálise, posterior à dissertação original. Não há alegação de pré-registro.',
      'limitations':[
       'Os intervalos condicionais não incorporam a variabilidade de retreinamento, a dependência induzida pela sobreposição dos conjuntos de treino dos folds, deslocamentos para populações não observadas ou toda a incerteza da seleção de modelos.',
       'A permutação demográfica pressupõe permutabilidade condicional das categorias dos atores dentro dos folds. A análise é observacional e associativa, sem identificação causal nem ajuste por outros atributos correlacionados.',
       'A análise racial principal mantém Unknown e seu único ator fixos, fora das trocas, pois esse rótulo representa ausência de informação racial. O ator permanece nas outras comparações demográficas e nas métricas globais. Na sensibilidade, Unknown pode mudar de ator dentro do fold4 sob a hipótese nula; isso não altera nem imputa os metadados originais.',
       'Grupos com poucos atores continuam muito incertos. O baixo suporte é sinalizado; o bootstrap não cria novas identidades.',
       'Macro-F1 médio dos folds e macro-F1 das predições externas concatenadas são agregações diferentes. A diferença entre esses valores é esperada.',
       'Não se afirma um valor p confirmatório a partir dos cinco folds, nem um valor p confirmatório de superioridade entre modelos.'
      ],
      'paired_fold_wins':int((paired.delta_macro_f1>0).sum()),'paired_fold_mean_delta':float(paired.delta_macro_f1.mean()),
      'bootstrap':boot.to_dict(orient='records'),'permutation':permutation.to_dict(orient='records'),
      'permutation_unknown_movable_sensitivity':permutation_unknown_movable.to_dict(orient='records'),
      'racial_gap_sensitivity':gaps.query("attribute == 'Race'").to_dict(orient='records'),
      'mitigation_sensitivity':mitigation.to_dict(orient='records'),
      'saved_emotion_metrics_audit':emotion_reference_audit,
      'inputs':[{'name':name,'sha256':hashlib.sha256((input_dir/name).read_bytes()).hexdigest(),'bytes':(input_dir/name).stat().st_size} for name in names]
    }
    (output_dir/'analysis_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    def num(v):return f'{v:.6f}'.replace('.',',')
    text = ['# Reanálise por ator das predições externas preservadas','',
      '## Evidência e auditoria','',
      'Foram alinhadas 7.441 amostras por modelo com os 91 atores do manifesto. Cada vídeo aparece uma vez no teste externo de cada modelo. Rótulos, probabilidades, folds e metadados de ator foram conferidos. As métricas de cada fold foram recalculadas e coincidem com o CSV preservado. Não houve treinamento novo.','',
      '## Comparação pareada','',
      f'O EmoT vence a CNN1D nos {summary["paired_fold_wins"]} folds. A diferença média dos cinco macro-F1 é {num(summary["paired_fold_mean_delta"])}. Esse resultado descritivo não é apresentado como cinco experimentos independentes.','',
      'O estimando primário é o desempenho nas predições externas concatenadas, com peso dos vídeos. O bootstrap sorteia 91 atores com reposição e transporta todos os vídeos de cada ator. As multiplicidades de atores são idênticas para os dois modelos, em 5.000 réplicas, semente 20261107. Intervalos percentis de 95% são condicionais aos modelos externos já ajustados.','']
    for _,row in boot.query("estimand == 'video_weighted'").iterrows():
        text.append(f'- {row.target}, {row.metric}: {num(row.observed)}; IC condicional por ator [{num(row.ci_low)}; {num(row.ci_high)}].')
    text+=['','## Sensibilidade ao peso por pessoa','',
      'A análise secundária normaliza a matriz de confusão de cada ator para massa total 1 antes de agregá-las. Cada pessoa recebe o mesmo peso; não se trata da média dos F1 individuais. Os dois estimandos respondem a perguntas diferentes.','']
    for _,row in boot.query("estimand == 'equal_actor_weight' and target == 'EmoT_minus_TemporalCNN1D'").iterrows():
        text.append(f'- Diferença {row.metric}: {num(row.observed)}; IC condicional [{num(row.ci_low)}; {num(row.ci_high)}].')
    text+=['','## Permutação demográfica por ator','',
      'As categorias completas dos atores foram permutadas dentro de cada fold externo, preservando as contagens de cada categoria por fold. A estatística é a amplitude de acurácia entre categorias conhecidas com pelo menos 30 vídeos e dois atores. As escolhas desta reanálise são posteriores à dissertação original, sem alegação de pré-registro. Foram usadas 5.000 permutações por atributo e p=(1+extremos)/(5001). O ajuste de Holm cobre a família de quatro atributos. Atributos correlacionados não são controlados causalmente.',
      'Na análise racial principal, Unknown e seu único ator ficam fixos e fora das trocas: esse rótulo representa ausência de informação, não uma categoria racial substantiva. A decisão tem justificativa metodológica independente do valor p. O ator continua incluído nas análises dos outros atributos e nas métricas globais. A estatística racial usa somente os três grupos conhecidos elegíveis. A sensibilidade permite trocar também o rótulo Unknown dentro do fold4, gerando categorias hipotéticas sob uma hipótese nula mais ampla. Essa randomização não é imputação nem altera os metadados originais. Cada análise tem sua própria correção de Holm para os quatro atributos.','']
    for _,row in permutation.iterrows():
        text.append(f'- {row.attribute}: gap={num(row.observed_accuracy_gap)}; p por ator={num(row.p_actor_permutation)}; p Holm={num(row.p_holm_four_attributes)}; grupos={row.eligible_groups}.')
    text+=['','## Sensibilidade à regra de elegibilidade racial','',
      'A análise histórica admite grupos com pelo menos 30 vídeos. Em OOF isso inclui Unknown, que representa somente um ator e 82 vídeos. A regra alternativa é uma escolha metodológica explicitada nesta reanálise, posterior à dissertação: categorias conhecidas, pelo menos 30 vídeos e dois atores. Não há alegação de pré-registro. Essa regra produz outro estimando de disparidade, apresentado ao lado do histórico, sem substituição silenciosa. Dois atores é um mínimo operacional, não evidência de suporte estatístico suficiente. Asian tem somente sete atores no total; três folds contêm um ator Asian cada.','']
    for _,row in gaps.query("attribute == 'Race'").iterrows():
        text.append(f'- {row.eligibility_policy}, {row.model}: gap de acurácia={num(row.accuracy_gap)}; gap de macro-F1={num(row.macro_f1_gap)}; grupos={row.groups}.')
    text+=['','Na comparação de mitigação, valores negativos abaixo significam redução da amplitude racial, não melhoria automática de todo o sistema.','']
    for _,row in mitigation.iterrows():
        text.append(f'- {row.eligibility_policy}: Reweighted menos EmoT no gap de acurácia={num(row.Reweighted_minus_EmoT_accuracy_gap)}; no gap de macro-F1={num(row.Reweighted_minus_EmoT_macro_f1_gap)}.')
    text+=['','Na sensibilidade que permite mover Unknown sob a hipótese nula, os resultados de permutação são:','']
    for _,row in permutation_unknown_movable.iterrows():
        text.append(f'- {row.attribute}: p por ator={num(row.p_actor_permutation)}; p Holm desta família={num(row.p_holm_four_attributes)}.')
    text+=['','## Suporte e resultados por emoção','',
      'Os CSVs incluem número de atores e vídeos por categoria e emoção. Grupos com menos de dez atores são sinalizados. A exclusão de uma categoria de um teste não elimina seus resultados descritivos. A categoria Unknown com suporte insuficiente em atores permanece nas tabelas descritivas.','',
      '## Limites da interpretação','']
    text+=['- '+limit for limit in summary['limitations']]
    text+=['','A conclusão deve ser restrita ao estudo de caso e à análise condicional. Os intervalos de diferenças não substituem o pacote de sementes adicionais, ablação ou validação em outra base.']
    (output_dir/'RELATORIO_ESTATISTICA_POR_ATOR.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--n-boot',type=int,default=5000)
    parser.add_argument('--n-perm',type=int,default=5000)
    parser.add_argument('--seed',type=int,default=20261107)
    args=parser.parse_args()
    result=run_analysis(args.input_dir,args.output_dir,args.n_boot,args.n_perm,args.seed)
    print(json.dumps({'status':result['status'],'n_actors':result['n_actors'],'n_samples':result['n_samples_per_model'],'mean_fold_delta':result['paired_fold_mean_delta'],'output_dir':str(args.output_dir)},ensure_ascii=False))
