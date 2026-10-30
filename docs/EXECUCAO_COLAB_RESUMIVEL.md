# Execução resumível do experimento CREMA-D no Colab Pro

## Objetivo

O notebook `07_cremad_fairness_xai_emot_resumable_colab_pro.ipynb` executa o
pipeline completo de classificação de emoções, auditoria de fairness e XAI. A
execução usa o Google Drive como armazenamento durável e um banco SQLite como
ledger operacional, permitindo continuar após desconexões, erros ou reinícios
da máquina virtual do Colab.

O notebook anterior, `06_cremad_fairness_xai_emot_final_colab.ipynb`, permanece
inalterado como referência científica e técnica.

## Primeira execução

1. Envie o notebook 07 ao Google Drive e abra-o com o Google Colab.
2. No Colab, selecione **Ambiente de execução > Alterar tipo de ambiente de
   execução > GPU**. Quando disponível, escolha **A100**.
3. Mantenha `RESUME_MODE = True` e `FORCE_NEW_RUN = False`.
4. Use **Ambiente de execução > Executar tudo**.
5. Autorize a montagem do Google Drive quando solicitado.
6. Acompanhe os blocos `[STAGE]`, `[HEARTBEAT]` e `[LOG]` impressos nas células.

Enquanto uma etapa cara estiver ativa, `latest_progress.png` é atualizado a cada
120 segundos. Ao concluir ou falhar, o notebook também preserva um PNG histórico
com o nome da etapa e seu estado.

Não feche a janela durante a autorização inicial do Drive. Depois que os dados
forem persistidos, uma queda do Colab não invalida os estágios já concluídos.

## Retomada após erro ou desconexão

1. Reconecte o ambiente do Colab.
2. Abra o mesmo notebook.
3. Use **Executar tudo** novamente, desde a primeira célula.

O notebook remonta o Drive, reinstala as dependências, encontra a execução
incompleta no SQLite e reconstrói as variáveis em memória. Em seguida:

- arquivos de vídeo existentes não são baixados novamente;
- landmarks já concluídos são carregados em shards de 50 vídeos;
- cada modelo continua a partir da última época integralmente persistida;
- Attention Rollout, Gradient × Input, Integrated Gradients, oclusão facial e
  Deletion Test carregam seus caches quando válidos;
- o empacotamento final é reconstruído de forma atômica.

Nunca comece a retomada no meio do notebook, pois as células iniciais recriam o
ambiente Python necessário. Use sempre **Executar tudo**.

## Estrutura criada no Google Drive

```text
MyDrive/cremad_fairness_xai/
├── raw/                         # metadados e vídeos originais
├── cache/                       # landmarks consolidados e shards
├── state/
│   ├── experiment_state.sqlite # estado operacional canônico
│   └── active_run.json          # ponteiro para a execução incompleta
└── runs/<run_id>/
    └── outputs/
        ├── fairness/
        ├── figures/
        ├── logs/
        │   └── stage_logs/      # saída textual por etapa
        ├── models/              # best_*.pt e latest_*.pt
        ├── progress_snapshots/  # PNGs de acompanhamento e latest_progress.png
        ├── reports/             # síntese e snapshot consistente do SQLite
        ├── tables/
        ├── xai/
        └── zips/                # pacote final e manifesto com SHA-256
```

Cada `run_id` possui saídas próprias. Os dados brutos, os caches e o banco de
estado são compartilhados para evitar retrabalho, sem sobrescrever os resultados
de execuções anteriores.

## O que o SQLite registra

O arquivo `experiment_state.sqlite` contém as tabelas:

- `runs`: configuração, situação e timestamps de cada execução;
- `stages`: tentativas, duração, erro e traceback de cada etapa;
- `events`: eventos e heartbeats durante o processamento;
- `items`: situação individual de downloads e vídeos processados;
- `training_epochs`: métricas e checkpoint de cada época por modelo;
- `artifacts`: caminho, tamanho e hash SHA-256 dos artefatos registrados.

Consultas úteis, executáveis com qualquer visualizador de SQLite:

```sql
SELECT run_id, status, started_at, updated_at, error
FROM runs
ORDER BY updated_at DESC;

SELECT stage_id, status, attempts, duration_sec, error_message
FROM stages
WHERE run_id = '<run_id>'
ORDER BY started_at;

SELECT model_name, MAX(epoch) AS ultima_epoca
FROM training_epochs
WHERE run_id = '<run_id>'
GROUP BY model_name;
```

## Execução nova e reprocessamento seletivo

Para criar outro `run_id` sem reutilizar uma execução incompleta, altere
temporariamente:

```python
FORCE_NEW_RUN = True
```

Depois que o novo `run_id` for criado, retorne o valor para `False` antes de uma
eventual retomada.

Para invalidar somente um cache caro, informe o identificador da etapa:

```python
FORCE_STAGES = {"xai_integrated_gradients"}
```

Os principais identificadores são `landmark_processing`,
`training_all_models`, `xai_attention_rollout`, `xai_gradient_x_input`,
`xai_integrated_gradients`, `xai_region_occlusion` e `xai_deletion_test`.

## Evidência de conclusão

Uma execução é considerada concluída quando a última célula imprime
`EXECUÇÃO CONCLUÍDA E PERSISTIDA`, a tabela `runs` apresenta `completed` e o
diretório `zips/` contém:

- o arquivo `cremad_fairness_xai_artifacts_<run_id>.zip`;
- o manifesto `download_manifest_<run_id>.csv`, com tamanho e SHA-256;
- o snapshot `reports/experiment_state_snapshot.sqlite` dentro do pacote.

## Regra de segurança do texto acadêmico

Este notebook não acessa nem altera o Overleaf. Resultados, tabelas e figuras
podem subsidiar uma proposta de revisão da dissertação, mas qualquer mudança no
texto acadêmico deve ser apresentada ao autor e aprovada explicitamente antes de
qualquer sincronização com o Overleaf.
