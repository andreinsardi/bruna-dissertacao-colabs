# Inventário dos notebooks

O acervo contém **13 notebooks**: nove etapas/versões numeradas, duas execuções preservadas e duas versões em inglês. Todos os arquivos `.ipynb` foram copiados sem alteração de conteúdo. O manifesto registra origem, tamanho, SHA-256, contagem de células e saídas, além do hash do código sem as saídas. Cópias idênticas encontradas em outros pacotes e diretórios não foram repetidas.

| Arquivo | Finalidade | Saídas preservadas | Google Colab |
|---|---|---:|---|
| [notebooks/01_ingest_qc_manifest.ipynb](../notebooks/01_ingest_qc_manifest.ipynb) | Ingestão e controle de qualidade dos CSVs RAVDESS | 19 | [Abrir no Colab](https://colab.research.google.com/github/andreinsardi/bruna-dissertacao-colabs/blob/main/notebooks/01_ingest_qc_manifest.ipynb) |
| [notebooks/02_preprocess_T100_dataset.ipynb](../notebooks/02_preprocess_T100_dataset.ipynb) | Normalização temporal em 100 passos | 13 | [Abrir no Colab](https://colab.research.google.com/github/andreinsardi/bruna-dissertacao-colabs/blob/main/notebooks/02_preprocess_T100_dataset.ipynb) |
| [notebooks/03_split_actor_holdout.ipynb](../notebooks/03_split_actor_holdout.ipynb) | Separação de treinamento e teste por ator | 7 | [Abrir no Colab](https://colab.research.google.com/github/andreinsardi/bruna-dissertacao-colabs/blob/main/notebooks/03_split_actor_holdout.ipynb) |
| [notebooks/04_train_eval_models.ipynb](../notebooks/04_train_eval_models.ipynb) | Treinamento e avaliação dos modelos da prova de conceito | 21 | [Abrir no Colab](https://colab.research.google.com/github/andreinsardi/bruna-dissertacao-colabs/blob/main/notebooks/04_train_eval_models.ipynb) |
| [notebooks/05_xai_attention_deletion.ipynb](../notebooks/05_xai_attention_deletion.ipynb) | Atenção, Gradiente × Entrada e deleção na prova de conceito | 19 | [Abrir no Colab](https://colab.research.google.com/github/andreinsardi/bruna-dissertacao-colabs/blob/main/notebooks/05_xai_attention_deletion.ipynb) |
| [notebooks/06_cremad_fairness_xai_emot_final_colab.ipynb](../notebooks/06_cremad_fairness_xai_emot_final_colab.ipynb) | Primeira versão completa em CREMA-D | 0 | [Abrir no Colab](https://colab.research.google.com/github/andreinsardi/bruna-dissertacao-colabs/blob/main/notebooks/06_cremad_fairness_xai_emot_final_colab.ipynb) |
| [notebooks/07_cremad_fairness_xai_emot_resumable_colab_pro.ipynb](../notebooks/07_cremad_fairness_xai_emot_resumable_colab_pro.ipynb) | Versão com recuperação por etapas e SQLite | 0 | [Abrir no Colab](https://colab.research.google.com/github/andreinsardi/bruna-dissertacao-colabs/blob/main/notebooks/07_cremad_fairness_xai_emot_resumable_colab_pro.ipynb) |
| [notebooks/08_cremad_fairness_xai_emot_definitivo_colab_pro.ipynb](../notebooks/08_cremad_fairness_xai_emot_definitivo_colab_pro.ipynb) | Protocolo definitivo: cinco partições por ator, equidade e XAI | 0 | [Abrir no Colab](https://colab.research.google.com/github/andreinsardi/bruna-dissertacao-colabs/blob/main/notebooks/08_cremad_fairness_xai_emot_definitivo_colab_pro.ipynb) |
| [notebooks/09_cremad_figuras_cientificas_v3_colab.ipynb](../notebooks/09_cremad_figuras_cientificas_v3_colab.ipynb) | Figuras científicas a partir dos artefatos da execução definitiva | 0 | [Abrir no Colab](https://colab.research.google.com/github/andreinsardi/bruna-dissertacao-colabs/blob/main/notebooks/09_cremad_figuras_cientificas_v3_colab.ipynb) |
| [notebooks/executed/08_cremad_fairness_xai_emot_definitivo_colab_pro.ipynb](../notebooks/executed/08_cremad_fairness_xai_emot_definitivo_colab_pro.ipynb) | Execução definitiva preservada, com 202 saídas e hash citado na dissertação | 202 | [Abrir no Colab](https://colab.research.google.com/github/andreinsardi/bruna-dissertacao-colabs/blob/main/notebooks/executed/08_cremad_fairness_xai_emot_definitivo_colab_pro.ipynb) |
| [notebooks/executed/09_cremad_figuras_cientificas_v3_colab.ipynb](../notebooks/executed/09_cremad_figuras_cientificas_v3_colab.ipynb) | Execução preservada da geração de figuras, com 17 saídas | 17 | [Abrir no Colab](https://colab.research.google.com/github/andreinsardi/bruna-dissertacao-colabs/blob/main/notebooks/executed/09_cremad_figuras_cientificas_v3_colab.ipynb) |
| [notebooks/publication_en/cremad_emot_definitive_colab.ipynb](../notebooks/publication_en/cremad_emot_definitive_colab.ipynb) | Versão em inglês preparada para o artigo associado | 0 | [Abrir no Colab](https://colab.research.google.com/github/andreinsardi/bruna-dissertacao-colabs/blob/main/notebooks/publication_en/cremad_emot_definitive_colab.ipynb) |
| [notebooks/publication_en/generate_paper_figures.ipynb](../notebooks/publication_en/generate_paper_figures.ipynb) | Versão em inglês preparada para o artigo associado | 0 | [Abrir no Colab](https://colab.research.google.com/github/andreinsardi/bruna-dissertacao-colabs/blob/main/notebooks/publication_en/generate_paper_figures.ipynb) |

## Qual notebook representa o resultado da dissertação?

`08_cremad_fairness_xai_emot_definitivo_colab_pro.ipynb` é o protocolo definitivo em CREMA-D. Sua cópia em `notebooks/executed/` preserva exatamente o arquivo cujo SHA-256 aparece no apêndice da dissertação. O notebook 09 gera figuras a partir desses artefatos. As versões em inglês estão em `notebooks/publication_en/`; elas não substituem o arquivo executado citado no texto.

## Acervo histórico

Os notebooks 01–05 foram recuperados do commit `989ac5ab1cfbb938f690dfbad571b57362e4b53f` de `vit-emot-poc`, juntamente com os módulos `src/`. A recuperação ocorreu neste novo repositório, sem restaurar ou sobrescrever os arquivos da pasta de trabalho original.

Eles registram uma prova de conceito em RAVDESS. O notebook 04 conserva uma configuração de oito classes, enquanto o 05 remapeia os rótulos para sete classes e espera um checkpoint correspondente. São estados históricos: não se deve executar 04 e 05 em sequência presumindo compatibilidade entre os checkpoints. Sua reprodução depende dos CSVs e do checkpoint compatível com cada configuração. Esses resultados não sustentam os números definitivos em CREMA-D.

Os notebooks 06 e 07 conservam a evolução do experimento CREMA-D. Para uma nova execução da dissertação, use o 08.
