# Colabs da dissertação de Bruna Carolina Rodrigues

Acervo computacional da dissertação **Inteligência artificial explicável e equidade no reconhecimento de expressões faciais em vídeo com Transformers espaço–temporais**, edição de 11/10/2026, Faculdade de Tecnologia da UNICAMP. Autora: Bruna Carolina Rodrigues. Orientador: Prof. Dr. Marco Antônio Garcia de Carvalho.

Este repositório reúne **16 notebooks**: a sequência 01–12, as execuções preservadas de 08 e 09 e duas versões em inglês do artigo associado. Inclui os módulos de apoio, geradores, testes e o manifesto de integridade. O [inventário completo](docs/INVENTARIO.md) contém um link para abrir cada notebook no Google Colab. Os Colabs 10–12 documentam a [revisão metodológica de novembro de 2026](docs/REVISAO_20261107.md); suas cópias são distribuídas sem saídas para reprodução.

## Reprodução do núcleo histórico

| Notebook | Uso | Google Colab |
|---|---|---|
| [08 — Experimento definitivo](notebooks/08_cremad_fairness_xai_emot_definitivo_colab_pro.ipynb) | Preparação do CREMA-D, treinamento, cinco partições por ator, equidade e explicabilidade | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andreinsardi/bruna-dissertacao-colabs/blob/main/notebooks/08_cremad_fairness_xai_emot_definitivo_colab_pro.ipynb) |
| [09 — Figuras científicas](notebooks/09_cremad_figuras_cientificas_v3_colab.ipynb) | Reconstrução das figuras a partir de vídeos, tensores, modelos e resultados da execução | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andreinsardi/bruna-dissertacao-colabs/blob/main/notebooks/09_cremad_figuras_cientificas_v3_colab.ipynb) |

Para consultar os resultados já produzidos, abra a [execução definitiva com as saídas](notebooks/executed/08_cremad_fairness_xai_emot_definitivo_colab_pro.ipynb) ou a [execução das figuras](notebooks/executed/09_cremad_figuras_cientificas_v3_colab.ipynb). Essas cópias são evidências preservadas; use as versões sem saídas para iniciar uma nova execução.

## Revisão metodológica de novembro de 2026

| Notebook | Uso |
|---|---|
| [10 — Figuras com layout corrigido](notebooks/10_cremad_figuras_corrigidas_v4_colab.ipynb) | Regeneração das figuras a partir dos artefatos históricos, com correção das sobreposições |
| [11 — Robustez e explicabilidade](notebooks/11_cremad_metodologia_final_colab.ipynb) | Três sementes, cinco folds, ablação conjunta das diferenças temporais e aleatorização de parâmetros para IG |
| [12 — Avaliação entre bases](notebooks/12_ravdess_validacao_externa_cremad_colab.ipynb) | Modelos CREMA-D congelados avaliados no RAVDESS, com conferência do extrator e bootstrap por ator |

Essas fontes exigem as entradas completas descritas em [REPRODUCAO.md](docs/REPRODUCAO.md). O registro de execução e a interpretação dos novos resultados ficam em [REVISAO_20261107.md](docs/REVISAO_20261107.md).

## Execução no Google Colab

1. Abra o notebook 08 e selecione uma GPU; a execução de referência usou uma A100 de 80 GB.
2. Execute as células em ordem e autorize a montagem do seu Google Drive.
3. Mantenha `FULL_RUN=True` e `DEFINITIVE_RUN=True` para o protocolo completo. Modos de desenvolvimento não reproduzem o experimento definitivo.
4. Use um diretório próprio para dados e resultados, configurado por `DRIVE_PROJECT_DIR`. Para uma execução independente, use `FORCE_NEW_RUN=True`; para recuperar uma execução interrompida, mantenha o diretório e use a retomada documentada.
5. Após concluir o experimento, abra o notebook 09. Ajuste `RUN_ID` e `BASE_DIR` à sua execução antes de executá-lo. O valor distribuído aponta para a execução de referência dos autores.

O notebook 08 contém o código de apoio embutido e baixa os dados a partir das fontes configuradas no próprio notebook. O notebook 09 precisa dos artefatos completos da execução e dos vídeos; apenas os arquivos `.ipynb` não substituem essas entradas. Consulte [como reproduzir](docs/REPRODUCAO.md) e os [detalhes de execução e retomada](docs/EXECUCAO_COLAB_DEFINITIVO.md).

## Rastreabilidade científica

- Execução definitiva: `20260809_223452_c0613d3c`.
- Notebook executado: **91 células, 45 de código, 202 saídas e zero saídas de erro**.
- SHA-256 do arquivo executado: `3e90e89858655dfbba39a1824ddf48c0b44004f82d0c33121d453fe15e224318`.
- O código das versões limpa e executada do 08 é idêntico; o mesmo foi conferido para o 09.
- Protocolo: canal visual do CREMA-D, 7.441 vídeos válidos, 155 marcos faciais, 100 passos temporais e cinco partições por ator.

O [mapa de resultados](docs/RESULTADOS_E_EXPLICABILIDADE.md) relaciona os métodos de XAI às figuras da edição histórica da dissertação. Consulte [manifest/notebooks.json](manifest/notebooks.json) e [SHA256SUMS](SHA256SUMS) para verificar os arquivos. As saídas preservadas de 08 e 09 continuam sendo as da execução registrada. Os 45 novos ajustes do Colab 11 e as demais análises complementares são identificados separadamente em [REVISAO_20261107.md](docs/REVISAO_20261107.md).

## Organização

```text
notebooks/                 # sequência 01–12
  executed/                # cópias 08 e 09 com as saídas originais
  publication_en/          # duas versões do artigo em inglês
src/                       # módulos históricos da prova de conceito RAVDESS
scripts/                   # recuperação, análise, geradores e validação
tests/                     # testes dos módulos operacionais e científicos
docs/                      # inventário, reprodução e relação com a dissertação
manifest/notebooks.json    # origens, hashes e contagens
CITATION.cff               # metadados para citar o acervo
```

Os notebooks 01–05 são **históricos em RAVDESS**; 06 e 07 são versões anteriores do experimento CREMA-D. Eles estão preservados para documentar a trajetória metodológica. Consulte as particularidades de reprodução no [inventário](docs/INVENTARIO.md).

## Verificação local

```bash
python -m pip install -r requirements-validation.txt
python scripts/validate_notebooks.py
python -m unittest discover -s tests -v
```

A validação confere formato, hashes, sintaxe das células e dos módulos embutidos, ausência de saídas de erro e equivalência do código entre as cópias limpas e executadas. Os testes exercitam a recuperação de etapas, os contratos científicos e a separação por ator. Eles não executam um novo treinamento completo nem exigem acesso ao Drive dos autores. O [registro de validação](docs/VALIDACAO.md) descreve os 49 testes aprovados e as verificações realizadas no acervo.

## Dados, ambiente e autoria

Os vídeos brutos do CREMA-D e do RAVDESS e os CSVs históricos do RAVDESS não estão incluídos. Obtenha os dados em suas fontes e observe os respectivos termos. Os notebooks executados preservam figuras, tabelas e mensagens produzidas pelo experimento. Tensores, checkpoints e demais arquivos necessários à reconstrução integral das figuras devem ser gerados pelo protocolo ou disponibilizados pelos autores.

`requirements.txt` é uma lista de dependências para instalação, não um congelamento integral do ambiente de 2026. Os notebooks históricos preservam seus comandos originais de instalação. Consulte o registro do ambiente em [REPRODUCAO.md](docs/REPRODUCAO.md).

As predições se referem a expressões faciais aparentes encenadas. Categorias demográficas são utilizadas para auditoria do experimento, e diferenças observadas não estabelecem causalidade.

Use [CITATION.cff](CITATION.cff) para citar esta versão do acervo. A edição do texto associada foi organizada no [projeto Overleaf da dissertação](https://www.overleaf.com/project/6acc3fb1f33af398a9f02bd9). Os termos de reutilização do código estão em [COPYING.md](COPYING.md).
