# Reprodução e preservação

## Referência da dissertação

A edição de 11/10/2026 identifica a execução `20260809_223452_c0613d3c`. A exportação executada do notebook 08 tem SHA-256 `3e90e89858655dfbba39a1824ddf48c0b44004f82d0c33121d453fe15e224318`, igual ao informado no apêndice. A cópia limpa e a executada têm exatamente o mesmo texto nas 45 células de código. A exportação do notebook 09 também mantém o mesmo código da cópia limpa.

Não foi repetido o treinamento durante a preparação deste repositório. As saídas dos notebooks executados foram preservadas sem reescrita. Os notebooks 01–05, seus módulos `src/` e as versões 06–09 também foram copiados sem alteração. Os geradores estão presentes para rastreabilidade, mas executá-los pode regravar os notebooks de destino; use uma cópia separada se desejar regenerá-los.

## Entradas e persistência

O notebook 08 organiza os dados em `MyDrive/<DRIVE_PROJECT_DIR>/`, com `raw/`, `cache/`, `state/` e `runs/<run_id>/outputs/`. As células iniciais reconstroem o ambiente. O SQLite, os shards, os checkpoints e os caches permitem recuperar etapas concluídas quando os artefatos esperados permanecem válidos. Não inicie a retomada no meio do notebook.

Ao executar pela primeira vez, use uma pasta própria para evitar misturar sua execução com as evidências preservadas. Para uma repetição independente, selecione `FORCE_NEW_RUN=True`; após criar a execução, retorne a `False` se precisar retomá-la. O `RUN_ID` será novo: a correspondência de protocolo não implica identidade bit a bit dos resultados.

O notebook 09 não treina. Ele espera o manifesto de amostras, o cache de marcos, os modelos, as predições, as atribuições, as tabelas Captum e os vídeos. Ajuste `RUN_ID` e `BASE_DIR` para a mesma execução. O diretório de resultados de figuras é separado dos artefatos científicos de entrada.

## Ambiente observado

A saída preservada do notebook 08 registra Google Colab Pro, GPU NVIDIA A100-SXM4-80GB, PyTorch `2.11.0+cu128` e MediaPipe `1.0.0`, com backend `tasks`. São versões observadas no arquivo executado, não uma alegação de disponibilidade atual em todo ambiente Colab.

Não foi localizado um `pip freeze` integral do ambiente. A célula original usa instalações com `--upgrade`; versões futuras podem mudar APIs e dependências. `requirements.txt` descreve as bibliotecas exigidas e não garante reprodução exata do ambiente histórico. A saída original de instalação também conserva um aviso de incompatibilidade de versão de `requests` com o pacote `google-colab`; a execução terminou com o marcador de conclusão e sem saídas de erro do Jupyter. O aviso foi preservado como parte do registro.

## Notebooks históricos 01–05

Esses notebooks usam os módulos `src/` e calculam a raiz como o diretório pai da pasta de execução. Localmente, abra o Jupyter na pasta `notebooks/`. No Colab, prepare uma cópia do repositório e mude a pasta antes das células originais:

```python
from pathlib import Path
import os
import subprocess

repo = Path('/content/bruna-dissertacao-colabs')
if not repo.exists():
    subprocess.run(['git', 'clone', 'https://github.com/andreinsardi/bruna-dissertacao-colabs.git', str(repo)], check=True)
os.chdir(repo / 'notebooks')
```

Disponibilize os CSVs em `data/ravdess_landmarks_kaggle/00_raw_kaggle_csv/`. Instale as dependências antes dos imports. Os dados gerados na máquina virtual do Colab são efêmeros se não forem persistidos no Drive. As saídas originais de 01–05 estão preservadas para consulta, mas não equivalem a uma confirmação de que todos os estados históricos formam uma cadeia única compatível. O checkpoint de sete classes esperado pelo 05 exige a configuração histórica correspondente; o 04 mantém uma configuração de oito classes.

## Escopo da conferência do acervo

Foram conferidos a pasta de trabalho `vit-emot-poc`, o commit histórico registrado, as exportações locais do Colab e os pacotes do artigo. A consulta ao Drive confirmou as pastas do projeto, a execução de referência e a presença do notebook 09. A busca global do conector não expôs todos os notebooks nativos do Colab; a listagem da pasta Colab é limitada a 100 itens. Portanto, este inventário cobre o conjunto 01–09 identificado para a dissertação, suas duas execuções e as duas versões em inglês; não afirma inventariar todos os notebooks de toda a conta Google.

## Conferência de integridade

O manifesto registra o SHA-256 dos arquivos integrais e o hash do código, calculado pela concatenação das células de código com uma quebra de linha entre elas. O validador lê os notebooks sem salvar ou normalizar seus arquivos. Os hashes devem permanecer iguais para preservar a correspondência com a dissertação.

## Complementos de novembro de 2026

O inventário passou a incluir os Colabs 10–12, descritos em [REVISAO_20261107.md](REVISAO_20261107.md). O 10 requer os artefatos históricos completos; o 11 reutiliza cache e partições, com uma grade nova de 45 ajustes; o 12 exige o checkpoint congelado, o modelo de detecção preservado e a conferência numérica do extrator antes de obter os vídeos externos. Todos são distribuídos sem saídas e mantêm o código de apoio embutido. Os geradores ficam em `scripts/`; o manifesto oficial dos arquivos externos fica em `manifest/`.
