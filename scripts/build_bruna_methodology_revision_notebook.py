"""Empacota módulos auditados num Colab autossuficiente, sem executar a rodada."""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'notebooks/11_cremad_metodologia_final_colab.ipynb'

def cell(kind,text):
    out={'cell_type':kind,'metadata':{},'source':text.strip().splitlines(keepends=True)}
    if kind=='code':out.update({'execution_count':None,'outputs':[]})
    return out

def build():
    modules={name:(ROOT/'scripts'/name).read_text(encoding='utf-8') for name in (
        'bruna_revision_models.py','bruna_revision_experiments.py','bruna_revision_statistics.py')}
    hashes={name:hashlib.sha256(text.encode()).hexdigest() for name,text in modules.items()}
    cells=[cell('markdown', '''# Bruna: revisão metodológica e experimentos complementares

Execução de revisão de 07/11/2026, posterior aos resultados históricos. Este notebook usa os artefatos da execução `20260809_223452_c0613d3c`; não refaz a extração de marcos e não modifica esses artefatos.

## Entregas

1. Reanálise das predições externas: bootstrap pareado por ator, permutação demográfica por ator, suportes e revocação por emoção/grupo.
2. 45 ajustes NOVOS: três sementes (42, 2026 e 2027), cinco folds congelados e três condições (EmoT completo, CNN1D completa e EmoT com posição apenas).
3. Aleatorização cumulativa dos parâmetros do EmoT para IG: 64 vídeos do teste fixo, alvos originais fixos, três inicializações e oito estágios. É uma propriedade de dependência dos parâmetros, sem certificado universal de explicação.

Selecione GPU, preferencialmente A100 com RAM alta, e execute todas as células. Autorize a montagem do Drive. O tempo depende da interrupção antecipada e do runtime; não há promessa de término em poucos minutos. Checkpoints permitem retomada, desde que código, configuração e ambiente sejam compatíveis.

As figuras são regeneradas separadamente pelo notebook `10_cremad_figuras_corrigidas_v4_colab.ipynb`. O manuscrito só recebe novos números e figuras depois da execução e da validação. Este notebook não acessa o Overleaf.
'''),cell('code','''from pathlib import Path
import os, sys, json, hashlib, platform, shutil, time
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

BASE_DIR = Path('/content/drive/MyDrive/cremad_fairness_xai')
OUTPUT_DIR = BASE_DIR / 'revision_final_20261107'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_ID = '20260809_223452_c0613d3c'
CANONICAL_TABLES = BASE_DIR / 'runs' / RUN_ID / 'outputs' / 'tables'

required = [BASE_DIR/'cache/cremad_landmarks_T100_semantic.npz',
            BASE_DIR/'runs'/RUN_ID/'outputs/models/best_EmoT.pt']
required += [CANONICAL_TABLES/name for name in (
    'manifest_valid_with_splits.csv','cv_pooled_predictions.csv','cv_actor_assignments.csv','cv_fold_metrics.csv')]
missing=[str(p) for p in required if not p.is_file()]
if missing:raise FileNotFoundError('Artefatos canônicos ausentes: '+str(missing))
print('Execução original:',RUN_ID)
print('Saídas complementares:',OUTPUT_DIR)
print('Python:',platform.python_version(),'Torch:',torch.__version__,'NumPy:',np.__version__,'pandas:',pd.__version__)
'''),cell('code',f'''# Módulos locais auditados, incorporados sem depender de um GitHub mutável.
MODULES = {modules!r}
EXPECTED_HASHES = {hashes!r}
CODE_DIR=Path('/content/bruna_revision_code')
CODE_DIR.mkdir(parents=True, exist_ok=True)
for name,source in MODULES.items():
    digest=hashlib.sha256(source.encode()).hexdigest()
    if digest != EXPECTED_HASHES[name]:raise RuntimeError('Hash do código divergente: '+name)
    compile(source,name,'exec')
    (CODE_DIR/name).write_text(source,encoding='utf-8')
sys.path.insert(0,str(CODE_DIR))
from bruna_revision_statistics import run_analysis
from bruna_revision_experiments import run_revision_training, run_parameter_sanity
code_record={{'notebook_revision':'2026-11-07','module_sha256':EXPECTED_HASHES,'source_run':RUN_ID}}
(OUTPUT_DIR/'notebook_code_manifest.json').write_text(json.dumps(code_record,indent=2),encoding='utf-8')
print('Código íntegro e importado.')
'''),cell('markdown','''## 1. Reanálise por ator das predições históricas

O estimando primário pondera vídeos, mantendo o ator como cluster. A sensibilidade normaliza a matriz de confusão de cada ator para peso igual por pessoa. Os intervalos são condicionais aos modelos externos já ajustados e não incluem a variabilidade de treinamento. A reanálise foi definida após os resultados históricos; não é apresentada como pré-registrada.

A categoria racial Unknown é preservada nas métricas globais, mas não é tratada como uma raça substantiva. O relatório mostra o impacto de sua inclusão histórica e uma sensibilidade da permutação. As quatro comparações demográficas têm ajuste de Holm e não identificam causalidade.
'''),cell('code', '''actor_result = run_analysis(
    CANONICAL_TABLES, OUTPUT_DIR/'actor_statistics',
    n_boot=5000, n_perm=5000, seed=20261107)
display(pd.read_csv(OUTPUT_DIR/'actor_statistics/actor_cluster_bootstrap.csv'))
display(pd.read_csv(OUTPUT_DIR/'actor_statistics/actor_demographic_permutation.csv'))
display(pd.read_csv(OUTPUT_DIR/'actor_statistics/mitigation_gap_sensitivity.csv'))
'''),cell('markdown','''## 2. Preflight de GPU e recursos

O protocolo definitivo requer CUDA para os novos treinamentos. A reanálise anterior funciona em CPU. Se esta etapa falhar, selecione GPU e RAM alta no Colab e execute novamente; não reduza a rodada para apresentá-la como definitiva.
'''),cell('code','''if not torch.cuda.is_available():
    raise RuntimeError('Selecione uma GPU no Colab antes de iniciar os 45 ajustes.')
import psutil
resources={'gpu':torch.cuda.get_device_name(0),
           'gpu_memory_gib':torch.cuda.get_device_properties(0).total_memory/2**30,
           'host_total_memory_gib':psutil.virtual_memory().total/2**30,
           'host_available_memory_gib':psutil.virtual_memory().available/2**30,
           'local_disk_free_gib':shutil.disk_usage('/content').free/2**30,
           'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
display(pd.DataFrame([resources]))
(OUTPUT_DIR/'preflight_resources.json').write_text(json.dumps(resources,indent=2),encoding='utf-8')
if resources['host_available_memory_gib']<14:
    raise RuntimeError('Selecione runtime com RAM alta: menos de 14 GiB de RAM livre para os tensores por fold.')
if resources['local_disk_free_gib']<8:
    raise RuntimeError('Espaço local livre insuficiente para uma execução confortável.')
'''),cell('markdown','''## 3. Sementes adicionais e ablação

Os cinco conjuntos de treino/validação/teste são os mesmos da execução original. Todas as condições usam o mesmo critério de seleção por validação e o mesmo orçamento de otimização. A padronização é estimada no treino. A versão com posição possui 310 atributos, enquanto a completa tem 1.240; essa diferença altera o número de parâmetros e será reportada.

São três condições × três sementes × cinco folds = 45 ajustes novos. Os ajustes históricos não são reutilizados como réplicas com inicialização controlada. Logs, melhor checkpoint, último checkpoint, estado aleatório e predições são persistidos por condição. Ao retomar, cada resultado tem sua identidade e seus hashes conferidos.
'''),cell('code','''training_result = run_revision_training(
    BASE_DIR, OUTPUT_DIR, seeds=(42,2026,2027), folds=(1,2,3,4,5))
print(json.dumps(training_result,indent=2))
if not training_result.get('definitive_grid_complete') or training_result.get('fits')!=45:
    raise RuntimeError('Rodada de treinamentos incompleta.')
display(pd.read_csv(OUTPUT_DIR/'training/pooled_metrics_by_seed.csv'))
'''),cell('markdown','''## 4. Aleatorização de parâmetros para os IG

Usa o checkpoint fixo original, referência zero padronizada, IG de 16 passos e classe originalmente prevista mantida fixa. A escolha de 64 vídeos equilibra a inspeção entre grupos e atores dentro do teste fixo; é uma coorte de auditoria, sem estimativa representativa de equidade populacional. Cada inicialização aleatoriza cumulativamente o classificador, os blocos da última à primeira camada e os parâmetros de entrada. As comparações registram magnitudes, rankings e sobreposição, sem escolher um limiar de aprovação depois de observar os resultados.
'''),cell('code','''sanity_result = run_parameter_sanity(
    BASE_DIR, OUTPUT_DIR, n_samples=64, random_seeds=(11,29,47))
print(json.dumps(sanity_result,indent=2))
if not sanity_result.get('definitive_grid_complete'):
    raise RuntimeError('Rodada de sanity check incompleta.')
display(pd.read_csv(OUTPUT_DIR/'parameter_sanity/summary.csv'))
'''),cell('markdown','''## 5. Pacote de revisão e estado verificável

O marcador abaixo descreve a conclusão dos experimentos complementares, não a aprovação final da dissertação. A integração e a compilação do manuscrito, a inspeção das figuras regeneradas e a revisão de autora/orientador continuam sendo etapas próprias. O ZIP não inclui os vídeos ou o cache original.
'''),cell('code','''if not training_result.get('definitive_grid_complete') or not sanity_result.get('definitive_grid_complete'):
    raise RuntimeError('Não é possível concluir: existe grade experimental pendente.')
records=[]
for p in OUTPUT_DIR.rglob('*'):
    if p.is_file() and p.suffix!='.partial':
        digest=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(2**20),b''):digest.update(block)
        records.append({'file':str(p.relative_to(OUTPUT_DIR)),'bytes':p.stat().st_size,'sha256':digest.hexdigest()})
pd.DataFrame(records).to_csv(OUTPUT_DIR/'artifact_manifest.csv',index=False)
completion={'status':'completed','scope':'supplementary_experiments_only',
            'canonical_run':RUN_ID,'training':training_result,'sanity':sanity_result,
            'actor_statistics_completed':(OUTPUT_DIR/'actor_statistics/analysis_summary.json').is_file(),
            'figure_regeneration_separate':True,'manuscript_integration_not_asserted':True}
(OUTPUT_DIR/'completed_supplementary_experiments.json').write_text(json.dumps(completion,indent=2),encoding='utf-8')
zip_path=shutil.make_archive(str(OUTPUT_DIR.parent/'bruna_revision_final_20261107_experiments'),'zip',root_dir=OUTPUT_DIR)
print('EXPERIMENTOS COMPLEMENTARES CONCLUÍDOS:',zip_path)
''')]
    # A pasta de treino é definida pelo módulo e conferida na geração.
    experiments=modules['bruna_revision_experiments.py']
    if '/"training"' not in experiments and '/ "training"' not in experiments:
        raise RuntimeError('Conferir o nome da pasta de resultados de treinamento no módulo.')
    for i,c in enumerate(cells):
        if c['cell_type']=='code':compile(''.join(c['source']),f'methodology-cell-{i}','exec')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'colab':{'name':DEST.name},'accelerator':'GPU','bruna_revision':{'executed':False,'date':'2026-11-07','module_sha256':hashes,'expected_new_fits':45}},'cells':cells}
    DEST.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(json.dumps({'notebook':str(DEST),'cells':len(cells),'executed':False,'sha256':hashlib.sha256(DEST.read_bytes()).hexdigest()},ensure_ascii=False))
    return DEST
if __name__=='__main__':build()
