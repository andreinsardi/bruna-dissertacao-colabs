"""Build the external-validation Colab without executing downloads or inference."""
from pathlib import Path
import ast
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parent/'bruna-dissertacao-colabs/notebooks/executed/08_cremad_fairness_xai_emot_definitivo_colab_pro.ipynb'
EXTRACTOR=ROOT/'scripts/bruna_ravdess_canonical_extractor.py'
DEST=ROOT/'notebooks/12_ravdess_validacao_externa_cremad_colab.ipynb'
CANONICAL_HASH='3e90e89858655dfbba39a1824ddf48c0b44004f82d0c33121d453fe15e224318'
CANONICAL_FUNCTIONS=('create_face_landmarker','sample_video_frames','normalize_landmarks_xy',
                     'extract_frame_landmarks','interpolate_missing_frames','process_one_video')


def canonical_sources():
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest()!=CANONICAL_HASH:
        raise ValueError('Executed canonical notebook changed.')
    nb=json.loads(SOURCE.read_text(encoding='utf-8'))
    functions={};regions=None
    for index in (26,28):
        text=''.join(nb['cells'][index]['source'])
        tree=ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='compile' and node.args and isinstance(node.args[0],ast.Constant):
                text=node.args[0].value;tree=ast.parse(text);break
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name in CANONICAL_FUNCTIONS:
                functions[node.name]=ast.get_source_segment(text,node)
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REGION_IDS' for t in node.targets):
                regions=ast.literal_eval(node.value)
    if set(functions)!=set(CANONICAL_FUNCTIONS) or regions is None:
        raise ValueError('Missing canonical extraction definitions.')
    return functions,sorted(set().union(*regions.values()))


def build():
    functions,ids=canonical_sources()
    if len(ids)!=155:raise ValueError('Expected 155 canonical landmarks.')
    header='''"""Canonical extraction functions from executed notebook08; no extraction runs on import.
Only the task-file helper is stricter: a preserved model is required, never downloaded as latest.
"""
from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks_python
from mediapipe.tasks.python import vision as mp_tasks_vision

MEDIAPIPE_BACKEND="tasks"
mp_face_mesh=None
BaseOptions=mp_tasks_python.BaseOptions
FaceLandmarker=mp_tasks_vision.FaceLandmarker
FaceLandmarkerOptions=mp_tasks_vision.FaceLandmarkerOptions
VisionRunningMode=mp_tasks_vision.RunningMode
T_FRAMES=100
BASE_FEATURE_DIM=310
MIN_FACE_DETECTION_RATE=0.60
FACE_LANDMARKER_MODEL_PATH=None

def ensure_face_landmarker_model():
    if FACE_LANDMARKER_MODEL_PATH is None or not Path(FACE_LANDMARKER_MODEL_PATH).is_file():
        raise FileNotFoundError("Use the preserved CREMA-D face_landmarker.task; no latest model substitution.")
    return Path(FACE_LANDMARKER_MODEL_PATH)
'''
    extractor=header+'\nSELECTED_LANDMARK_IDS='+repr(ids)+'\n\n'+'\n\n'.join(functions[name] for name in CANONICAL_FUNCTIONS)+'\n'
    compile(extractor,str(EXTRACTOR),'exec')
    EXTRACTOR.write_text(extractor,encoding='utf-8')
    # AST comparison is independent of comments and whitespace.
    tree=ast.parse(extractor)
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in functions:
            expected=ast.parse(functions[node.name]).body[0]
            if ast.dump(node,include_attributes=False)!=ast.dump(expected,include_attributes=False):
                raise ValueError('Canonical extractor function changed: '+node.name)
    names=('bruna_revision_models.py','bruna_revision_experiments.py','bruna_revision_statistics.py',
           'bruna_ravdess_canonical_extractor.py','bruna_ravdess_external.py')
    modules={name:(ROOT/'scripts'/name).read_text(encoding='utf-8') for name in names}
    hashes={name:hashlib.sha256(source.encode()).hexdigest() for name,source in modules.items()}
    archive_path=ROOT/'dissertacao_bruna_revisao_20261107/evidencias/ravdess_zenodo_archive_manifest.json'
    if not archive_path.exists():archive_path=ROOT/'manifest/ravdess_zenodo_archive_manifest.json'
    archives=json.loads(archive_path.read_text(encoding='utf-8-sig'))
    if len(archives['files'])!=24:raise ValueError('Incomplete official archive manifest.')
    contents=[('markdown','''# Bruna: validação externa CREMA-D → RAVDESS

**Estado distribuído: não executado.** Os resultados só existirão depois das células de download, extração e inferência. Este notebook complementa os Colabs 10 (figuras) e 11 (45 ajustes/IG), sem alterá-los.

O teste usa os checkpoints originais congelados EmoT e CNN1D treinados no CREMA-D. Não há fine-tuning, seleção de checkpoint ou normalização ajustada ao RAVDESS. O projeto já usou RAVDESS em etapas históricas; esta é uma avaliação externa entre bases definida após esse desenvolvimento, sem alegação de base inteiramente inédita ou pré-registro.

São 24 atores, vídeos somente visuais de fala e seis emoções comuns, totalizando 1.056 estímulos nominais. Calm e surprised ficam fora; calm não é fundido a neutral. As versões AV e VO do mesmo estímulo não são duplicadas.

Os ZIPs oficiais por ator somam aproximadamente **13,36 GB transferidos**. O processamento mantém um ZIP temporário por vez e persiste somente os vídeos selecionados no Drive. A extração é em CPU pelo MediaPipe Tasks/IMAGE 0.10.35 e OpenCV headless 4.13.0.92; a inferência usa GPU se disponível. Use uma sessão própria, sem interromper um treinamento do Colab 11. O tempo depende do download e da extração de 105.600 quadros amostrados, com retomada por ator e vídeo.

Recuperação do ambiente em 07/11/2026: MediaPipe 1.0.0/OpenCV 5.0 produziram erro médio acima da tolerância tanto na sessão CPU quanto na A100. O notebook original de agosto carregou o cache consolidado, criado em 25/06/2026, antes dessas versões. As versões aqui fixadas eram as mais recentes disponíveis na data do cache; são uma hipótese de recuperação do ambiente, não um registro histórico comprovado. A aceitação depende de repetir a conferência numérica com as mesmas tolerâncias, sem observar desempenho RAVDESS e sem afrouxar os critérios. Os diagnósticos anteriores foram preservados.

Dados: https://zenodo.org/records/1188976 — Livingstone e Russo (2018), DOI 10.1371/journal.pone.0196391. Observe os termos CC BY-NC-SA 4.0. Os vídeos não devem ser incluídos no repositório de notebooks.
'''),('code', '''%pip install -q "mediapipe==0.10.35" "opencv-python-headless==4.13.0.92" requests
'''),('code','''from pathlib import Path
import sys, json, hashlib, platform, shutil
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

BASE_DIR=Path('/content/drive/MyDrive/cremad_fairness_xai')
OUTPUT_DIR=BASE_DIR/'revision_ravdess_external_20261107'
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
RUN_ID='20260809_223452_c0613d3c'
required=[BASE_DIR/'cache/face_landmarker.task', BASE_DIR/'cache/cremad_landmarks_T100_semantic.npz',
          BASE_DIR/'runs'/RUN_ID/'outputs/tables/manifest_valid_with_splits.csv']
required += [BASE_DIR/'runs'/RUN_ID/'outputs/models'/name for name in ('best_EmoT.pt','best_TemporalCNN1D.pt')]
missing=[str(p) for p in required if not p.is_file()]
if missing:raise FileNotFoundError('Artefatos canônicos ausentes: '+str(missing))
import psutil
if psutil.virtual_memory().available<4*2**30:raise RuntimeError('Necessários ao menos 4 GiB de RAM livre.')
if shutil.disk_usage('/content').free<4*2**30:raise RuntimeError('Necessários ao menos 4 GiB de disco temporário livre.')
print('Python:',platform.python_version(),'Torch:',torch.__version__,'GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
print('Saídas:',OUTPUT_DIR)
'''),('code',f'''MODULES={modules!r}
EXPECTED_HASHES={hashes!r}
ARCHIVE_MANIFEST={archives!r}
CODE_DIR=Path('/content/bruna_ravdess_revision_code')
CODE_DIR.mkdir(exist_ok=True)
for name,source in MODULES.items():
    if hashlib.sha256(source.encode()).hexdigest()!=EXPECTED_HASHES[name]:raise RuntimeError('Código alterado: '+name)
    compile(source,name,'exec')
    (CODE_DIR/name).write_text(source,encoding='utf-8')
sys.path.insert(0,str(CODE_DIR))
from bruna_ravdess_external import verify_canonical_extraction, prepare_ravdess_videos, extract_ravdess, evaluate_external
from bruna_revision_experiments import atomic_json
atomic_json(OUTPUT_DIR/'notebook_code_manifest.json',{{'canonical_notebook_sha256':{CANONICAL_HASH!r},'module_sha256':EXPECTED_HASHES}})
atomic_json(OUTPUT_DIR/'official_archive_manifest.json',ARCHIVE_MANIFEST)
print('Módulos íntegros; manifesto oficial com',len(ARCHIVE_MANIFEST['files']),'arquivos.')
'''),('markdown','''## 1. Conferir o extrator em vídeos originais CREMA-D

Antes do download externo, reextrair seis vídeos CREMA-D: uma classe por vídeo, com atores distintos e escolha determinística. Comparar com o tensor original. O notebook reutiliza a cópia preservada de `face_landmarker.task`, sem baixar um modelo chamado latest. Se a representação não coincidir dentro das tolerâncias registradas, a execução para e preserva o diagnóstico. Não afrouxar a tolerância para obter um resultado externo.
'''),('code','''preflight=verify_canonical_extraction(BASE_DIR,OUTPUT_DIR)
display(pd.read_csv(OUTPUT_DIR/'canonical_extractor_preflight.csv'))
assert preflight['all_six_passed']
'''),('markdown','''## 2. Obter e validar os vídeos oficiais

Cada ZIP é verificado com tamanho e MD5 publicados pelo Zenodo. A seleção é auditada contra os 1.056 nomes esperados. O manifesto preserva hashes dos vídeos, identidade, classe, intensidade, frase e repetição. Não usa CSVs antigos de 68 marcos nem duplica versões com e sem áudio.
'''),('code','''nominal=prepare_ravdess_videos(OUTPUT_DIR,ARCHIVE_MANIFEST)
display(nominal.groupby(['ActorID','label_name']).size().unstack(fill_value=0))
assert len(nominal)==1056 and nominal.ActorID.nunique()==24
'''),('markdown','''## 3. Extrair os mesmos 155 marcos

São as mesmas funções canônicas de amostragem, centralização/escala por quadro, interpolação e limiar de detecção de 60%, com 100 instantes. As falhas são registradas por vídeo e não desaparecem do denominador nominal. Uma falha inesperada interrompe a rotina em vez de virar exclusão silenciosa. A retomada confere configuração e hashes dos resultados por vídeo.
'''),('code','''valid,x_base=extract_ravdess(BASE_DIR,OUTPUT_DIR,nominal)
display(pd.read_csv(OUTPUT_DIR/'coverage_actor_emotion.csv'))
print('Vídeos nominais:',len(nominal),'válidos:',len(valid),'atores válidos:',valid.ActorID.nunique())
'''),('markdown','''## 4. Inferência congelada e incerteza por ator

Aplicar posição e diferenças exatamente como no CREMA-D; padronizar somente com média e desvio-padrão do treino salvos em cada checkpoint. Exportar predições, métricas, confusão e revocação. Bootstrap de 5.000 amostras de atores, com as mesmas multiplicidades para os dois modelos. Os intervalos são condicionais aos ajustes congelados e aos vídeos cuja extração foi válida; não representam a variação de treinamento nem eliminam a limitação de cobertura.
'''),('code','''external=evaluate_external(BASE_DIR,OUTPUT_DIR,valid,x_base,n_boot=5000)
display(pd.read_csv(OUTPUT_DIR/'evaluation/metrics.csv'))
display(pd.read_csv(OUTPUT_DIR/'evaluation/actor_cluster_intervals.csv'))
display(pd.read_csv(OUTPUT_DIR/'evaluation/recall_by_emotion.csv'))
print(json.dumps(external,indent=2))
'''),('markdown','''## 5. Figuras derivadas das predições externas reais

As figuras abaixo usam somente as tabelas recém-produzidas. Não substituem as figuras históricas do CREMA-D e não equiparam o teste entre bases ao protocolo de estudos treinados dentro do RAVDESS.
'''),('code','''import matplotlib.pyplot as plt
from bruna_revision_experiments import CLASSES, hash_file
FIG_DIR=OUTPUT_DIR/'figures'
FIG_DIR.mkdir(exist_ok=True)
labels=['Raiva','Nojo','Medo','Alegria','Neutro','Tristeza']
for model in ('EmoT','TemporalCNN1D'):
    cm=pd.read_csv(OUTPUT_DIR/f'evaluation/{model}_confusion.csv').iloc[:,1:].to_numpy(float)
    rates=np.divide(cm,cm.sum(1,keepdims=True),out=np.zeros_like(cm),where=cm.sum(1,keepdims=True)>0)
    fig,ax=plt.subplots(figsize=(7.4,6.4),layout='constrained')
    im=ax.imshow(rates,vmin=0,vmax=1,cmap='Blues')
    ax.set_xticks(range(6),labels,rotation=35,ha='right');ax.set_yticks(range(6),labels)
    ax.set_xlabel('Classe prevista');ax.set_ylabel('Classe verdadeira')
    ax.set_title(f'{model}: CREMA-D → RAVDESS\\nTeste externo visual, sem ajuste no RAVDESS')
    for i in range(6):
        for j in range(6):ax.text(j,i,f'{int(cm[i,j])}\\n{rates[i,j]:.2f}',ha='center',va='center',color='white' if rates[i,j]>.55 else 'black',fontsize=9)
    fig.colorbar(im,ax=ax,label='Fração dentro da classe verdadeira')
    fig.savefig(FIG_DIR/f'{model}_confusao_externa.pdf');fig.savefig(FIG_DIR/f'{model}_confusao_externa.png',dpi=220)
    plt.show();plt.close(fig)
recall=pd.read_csv(OUTPUT_DIR/'evaluation/recall_by_emotion.csv')
fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
for offset,model in zip((-.18,.18),('EmoT','TemporalCNN1D')):
    values=recall[recall.model.eq(model)].set_index('emotion').loc[CLASSES,'recall'].to_numpy()
    ax.bar(np.arange(6)+offset,values,width=.36,label=model)
ax.set_xticks(range(6),labels);ax.set_ylim(0,1);ax.set_ylabel('Revocação')
ax.set_title('CREMA-D → RAVDESS: revocação por emoção');ax.legend(loc='upper left',bbox_to_anchor=(1.01,1))
fig.savefig(FIG_DIR/'revocacao_externa_por_emocao.pdf');fig.savefig(FIG_DIR/'revocacao_externa_por_emocao.png',dpi=220)
plt.show();plt.close(fig)
fig_manifest=[{'file':p.name,'sha256':hash_file(p),'bytes':p.stat().st_size} for p in sorted(FIG_DIR.iterdir()) if p.is_file()]
atomic_json(OUTPUT_DIR/'figure_manifest.json',fig_manifest)
print('Avaliação externa concluída; vídeos e cache permanecem separados do repositório de notebooks.')
''')]
    cells=[]
    for index,(kind,source) in enumerate(contents):
        source=source.strip()+'\n'
        if kind=='code':compile('\n'.join('pass # Colab magic' if l.startswith('%') else l for l in source.splitlines()),f'ravdess-cell-{index}','exec')
        cell={'cell_type':kind,'id':hashlib.sha256((str(index)+source).encode()).hexdigest()[:12],'metadata':{},'source':source.splitlines(keepends=True)}
        if kind=='code':cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    nb={'nbformat':4,'nbformat_minor':5,'cells':cells,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'},'colab':{'name':DEST.name},'bruna_revision':{'executed':False,'canonical_run':'20260809_223452_c0613d3c','canonical_notebook_sha256':CANONICAL_HASH,'module_sha256':hashes,'nominal_videos':1056,'nominal_actors':24,'new_training':False}}}
    DEST.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(json.dumps({'notebook':str(DEST),'cells':len(cells),'bytes':DEST.stat().st_size,'sha256':hashlib.sha256(DEST.read_bytes()).hexdigest(),'executed':False}))
    return DEST


if __name__=='__main__':build()
