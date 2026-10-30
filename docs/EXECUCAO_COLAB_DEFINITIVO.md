# Execução do Colab definitivo — CREMA-D, fairness e XAI

## Arquivo canônico

Use o notebook:

`notebooks/08_cremad_fairness_xai_emot_definitivo_colab_pro.ipynb`

Os notebooks `06` e `07` permanecem preservados. O notebook `08` não acessa nem altera o Overleaf. Qualquer mudança futura no texto da dissertação depende de validação prévia do usuário.

## Ambiente recomendado

- Google Colab Pro;
- GPU A100;
- Google Drive com espaço para vídeos, landmarks, checkpoints, tabelas, figuras e ZIP final;
- execução integral com `FULL_RUN=True` e `DEFINITIVE_RUN=True`.

O notebook bloqueia por padrão uma execução marcada como definitiva com amostra reduzida. `ALLOW_DEV_MODE=True` existe somente para depuração deliberada e não produz evidência final válida.

## Como executar

1. Envie o arquivo `.ipynb` ao Google Drive ou abra-o diretamente no Colab.
2. Selecione **Ambiente de execução > Alterar tipo de ambiente de execução > A100 GPU**.
3. Use **Executar tudo**.
4. Autorize a montagem do Google Drive quando solicitado.
5. Acompanhe os prints, os logs de cada etapa e os PNGs em `outputs/progress_snapshots`.
6. Ao final, preserve o diretório completo da execução e o ZIP gerado.

## Retomada depois de falha ou desconexão

Abra o mesmo notebook e execute tudo novamente. O notebook usa:

- `state/experiment_state.sqlite` para o estado operacional;
- `state/active_run.json` para localizar a execução compatível;
- shards de landmarks para não repetir vídeos já processados;
- `latest_*.pt` e `best_*.pt` para retomar treinamento por época;
- predições `.npz` por fold e modelo para não repetir folds concluídos;
- caches separados para cada técnica XAI;
- escrita atômica para evitar arquivos parciais considerados válidos.

Para iniciar uma execução independente, altere `FORCE_NEW_RUN=True`. Para invalidar uma etapa específica, adicione o identificador em `FORCE_STAGES`, por exemplo `{"cross_validation_5fold"}`.

## Escopo científico executado

- 100% dos vídeos válidos da base selecionada;
- modalidade exclusivamente visual, embora o CREMA-D original seja áudio-visual;
- landmarks faciais do MediaPipe com posição, velocidade, aceleração e jerk;
- holdout por ator para a análise principal;
- validação cruzada de cinco folds por ator;
- quatro modelos em cada fold: FlatMLP, TemporalCNN1D, EmoT e EmoT_Reweighted;
- normalização ajustada apenas no treino de cada fold;
- loss ponderado por emoção em todos os modelos;
- amostragem por emoção × Race somente no EmoT_Reweighted;
- métricas globais, por grupo, por emoção e interseccionais;
- matrizes de confusão globais, por grupo e média ± desvio-padrão entre folds;
- Attention Rollout, Gradient × Input, Integrated Gradients, validação com Captum, oclusão regional e Deletion Test;
- logs de recursos, tempos, checkpoints, hashes, manifesto e snapshot do SQLite.

## Artefatos indispensáveis para a análise final

| Pergunta | Artefato principal |
|---|---|
| A base está desbalanceada? | `sample_distribution_by_demographic.csv`, `sample_distribution_emotion_by_demographic.csv`, `sample_distribution_intersectional.csv`, `sample_distribution_valid_by_demographic.csv`, `sample_distribution_valid_emotion_by_demographic.csv` |
| Como o balanceamento foi tratado? | `balance_strategy_documentation.csv`, `balance_class_weights.csv`, `balance_sampler_label_x_race.csv` |
| O resultado é robusto entre folds? | `cv_fold_metrics.csv`, `cv_summary_mean_std.csv` |
| Houve vazamento de ator? | `cv_actor_assignments.csv`, `cv_actor_split_audit.json` |
| Quais emoções são mais confundidas? | `cv_confusion_pairs_by_group.csv`, `cv_top_confusions_to_discuss.csv` |
| O desempenho muda por raça, etnia, sexo ou idade? | `cv_group_metrics.csv`, `cv_group_gap_summary.csv`, `cv_group_metrics_by_fold.csv`, `cv_group_metrics_fold_summary.csv`, `cv_group_emotion_metrics.csv` |
| Como é a matriz média? | `cv_confusion_mean_<modelo>.csv` e `cv_confusion_mean_<modelo>.png` |
| A mitigação ajudou? | `balance_mitigation_cv_comparison.csv` e métricas de fairness do EmoT_Reweighted |
| O Integrated Gradients foi verificado? | `xai_captum_integrated_gradients_validation.csv`, `xai_captum_validation.md` |
| O que discutir na dissertação? | `cv_observations_to_discuss.md`, sempre com interpretação humana e suporte teórico |
| Como comparar com a literatura? | `literature_visual_only_comparison_template.csv`, preenchido somente com fontes primárias validadas |
| Qual foi o ambiente e o custo? | `resource_usage_log_<run_id>.csv`, `processing_time_by_stage.csv`, `runtime_estimate_definitive_a100.csv` |

## Tempo esperado na A100

- sem cache de landmarks: aproximadamente 5 a 8 horas;
- com landmarks íntegros em cache: aproximadamente 1,5 a 3,5 horas;
- depois de uma falha: depende do último shard, época ou fold íntegro e pode ser muito menor.

O MediaPipe é o principal gargalo porque a extração de landmarks ocorre majoritariamente em CPU. A A100 acelera o treinamento e o XAI, mas não elimina esse custo.

## Critério para encerrar a execução

A execução só deve ser considerada concluída quando:

1. o painel final mostrar todas as etapas como concluídas;
2. existir `cv_summary_mean_std.csv` com cinco folds para cada um dos quatro modelos;
3. `cv_actor_split_audit.json` indicar cobertura integral e zero vazamento de ator;
4. o ZIP final e o manifesto de hashes tiverem sido produzidos;
5. o arquivo `completed_<run_id>.json` existir no diretório de estado;
6. os logs não tiverem erro terminal posterior ao empacotamento.

Depois da execução, envie o ZIP final, o diretório `reports` e as tabelas principais para validação. Somente após essa validação os resultados devem ser convertidos em texto de dissertação.
