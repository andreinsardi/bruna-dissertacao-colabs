# Video facial expression recognition with EmoT

Computational materials for Bruna Carolina Rodrigues's dissertation at the School of Technology, University of Campinas (UNICAMP), supervised by Marco Antônio Garcia de Carvalho.

The study evaluates six acted facial expressions using 155 landmarks at 100 time points, with position and three orders of temporal differences. The workflow covers actor-disjoint evaluation on CREMA-D, stability across initializations, ablation with retraining, demographic auditing, explainability, and transfer to RAVDESS.

| Procedure | Executable notebook |
|---|---|
| CREMA-D acquisition, feature extraction, fixed test and cross-validation, fairness and attribution | [CREMA-D evaluation](notebooks/publication_en/cremad_emot_definitive_colab.ipynb) |
| Actor-level inference, stability, temporal ablation and parameter randomization | [Stability and ablation](notebooks/11_cremad_metodologia_final_colab.ipynb) |
| Attribution by block, IG completeness, actor-level ADI and random deletion | [Explainability evaluation](notebooks/13_cremad_explainability_audit.ipynb) · [Open in Colab](https://colab.research.google.com/github/andreinsardi/bruna-dissertacao-colabs/blob/v1.2.0/notebooks/13_cremad_explainability_audit.ipynb) |
| Frozen-model transfer from CREMA-D to RAVDESS | [Transfer evaluation](notebooks/12_ravdess_validacao_externa_cremad_colab.ipynb) |
| Figure generation | [Plotting notebook](notebooks/publication_en/generate_paper_figures.ipynb), [experimental and model panels](notebooks/10_cremad_figuras_corrigidas_v4_colab.ipynb) |

Use the CREMA-D run artifacts to reproduce attribution and transfer. The explainability notebook also requires the parameter-randomization artifacts from the stability notebook. See [reproduction instructions](docs/REPRODUCAO.md) and [explainability inputs and outputs](docs/EXPLAINABILITY.md).

The [notebook inventory](docs/INVENTARIO.md) includes the original notebooks and the preserved executions of the CREMA-D experiment and figure generation. The [manifest](manifest/notebooks.json) and [checksums](SHA256SUMS) identify all 17 notebooks.

## Validation

```bash
python -m pip install -r requirements-validation.txt
python scripts/validate_notebooks.py
python -m unittest discover -s tests -v
```

Validation checks notebook structure, code syntax, hashes, archived outputs, and the actor-disjoint evaluation helpers. Full experiment reproduction requires the official datasets and corresponding artifacts. Raw videos, landmarks and checkpoints are obtained through the workflow. See [reuse terms](COPYING.md) and [citation metadata](CITATION.cff).
