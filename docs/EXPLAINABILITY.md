# Explainability evaluation

Use the CREMA-D fixed-test checkpoint and the saved outputs of parameter randomization from notebook 11. The script verifies clip order, original target classes and agreement with the stored attribution maps before writing summaries.

```bash
python scripts/evaluate_explainability.py --base-dir /path/to/cremad_fairness_xai --output-dir /path/to/results --cpu
```

The base directory contains `cache/cremad_landmarks_T100_semantic.npz` and `runs/<run-id>/outputs`. The checkpoint is read from `models/best_EmoT.pt` under that run. The `xai` directory supplies `xai_sample_manifest.csv` and `xai_deletion_test_temporal.csv`. The parameter-randomization directory supplies the sample manifest, baseline attributions and fully randomized maps for seeds 11, 29 and 47. Use `--sanity-dir` to select that directory if it differs from the path in the notebook.

| Output | Measurement |
|---|---|
| `blocos_ig.json` | Absolute IG share and mean magnitude per attribute and time point in each block |
| `convergencia.json` | Completeness error for 16 right-Riemann steps |
| `adi.json` | Actor-level ADI permutation with joint maximum-statistic adjustment |
| `delecao.json` | Aggregate curves and AUC for attention-ordered and random deletion |
| `delecao_contraste.csv` | Matched deletion curves by clip |

Render the two figures using the regional IG CSV produced by the CREMA-D run and the matched deletion CSV.

```bash
python scripts/plot_explainability_evaluation.py --region-scores /path/to/xai_integrated_gradients_region_importance_by_race.csv --deletion-scores /path/to/results/delecao_contraste.csv --output-dir /path/to/figures
```
