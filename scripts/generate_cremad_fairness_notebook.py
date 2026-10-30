"""Generate the final CREMA-D fairness/XAI Colab notebook.

The generated notebook is intentionally self-contained: it installs Colab
dependencies, downloads CREMA-D metadata/vídeos, extracts facial landmarks,
trains the landmark-based EmoT experiment, and reports predictive fairness,
XAI, explanatory fairness, and high-consensus robustness.
"""

from pathlib import Path

import nbformat as nbf


OUT = Path("notebooks/06_cremad_fairness_xai_emot_final_colab.ipynb")


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.rstrip() + "\n")


nb = nbf.v4.new_notebook()
nb.metadata.update(
    {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "colab": {
            "name": OUT.name,
            "provenance": [],
            "gpuType": "A100",
        },
        "accelerator": "GPU",
    }
)

nb.cells = [
    md(
        r"""
# Experimento final - CREMA-D, EmoT, XAI e fairness étnico-racial

Este notebook implementa o experimento final da pesquisa ajustada: **auditar um modelo
Emotion Transformer (EmoT) baseado em landmarks faciais para reconhecimento de
emoções no CREMA-D**, com foco em fairness por grupos etnico-raciais.

Decisões metodológicas:

- **Dataset principal:** CREMA-D.
- **Modelo principal:** EmoT/AcT no estilo da implementação do Pedro, adaptado para
  sequências temporais de landmarks faciais do CREMA-D.
- **Baselines:** FlatMLP e TemporalCNN1D usando os mesmos landmarks.
- **Rótulo principal:** emoção codificada no nome do arquivo do CREMA-D.
- **Robustez:** subconjunto de alto consenso dos avaliadores faciais já existentes no CREMA-D.
- **Fairness principal:** `Race`.
- **Fairness secundário:** `Ethnicity`, `Sex` e faixa etária.
- **XAI:** Attention Rollout temporal, Gradient x Input por landmark, Integrated Gradients
  e Deletion Test temporal.
- **Ajuste pós-revisão bibliográfica:** análise interseccional, Integrated Gradients,
  oclusão por regiões faciais e correlação exploratória entre XAI e fairness.

Observação ética: o experimento não infere estados emocionais reais nem realiza diagnóstico
psicológico. Ele avalia reconhecimento de **expressões faciais aparentes encenadas** em
um corpus controlado, com foco em auditabilidade e viés algorítmico.

Execução recomendada: Google Colab Pro com GPU, preferencialmente A100. Todos os artefatos
são salvos no Google Drive em `MyDrive/cremad_fairness_xai/`, para que o experimento possa
ser interrompido e retomado sem perder downloads, landmarks, modelos, tabelas e figuras.
"""
    ),
    md(
        r"""
## Fluxo detalhado do experimento

```mermaid
flowchart TD
    A[Colab Pro com GPU] --> A1[Montar Google Drive]
    A1 --> A2[Criar raw/cache/outputs no Drive]
    A2 --> B[Instalar dependências]
    B --> C[Baixar e cachear metadados oficiais CREMA-D]
    C --> D[Manifesto experimental]
    D --> D1[Label original do arquivo]
    D --> D2[Race, Ethnicity, Sex, Age]
    D --> D3[Votos faciais e agreement]
    D --> E[Estatísticas descritivas]
    E --> F[Baixar vídeos FLV]
    F --> G[Amostrar 100 frames por video]
    G --> H[MediaPipe Face Mesh]
    H --> I[Landmarks normalizados]
    I --> J[Controle de qualidade por grupo]
    J --> K[Tensor X: video x tempo x features]
    K --> L[Derivadas temporais: pos, vel, acc, jerk]
    L --> M[Split por ator sem vazamento]
    M --> N[FlatMLP baseline]
    M --> O[TemporalCNN1D baseline]
    M --> P[EmoT principal]
    M --> Q[EmoT com reweighting]
    N --> R[Avaliação preditiva]
    O --> R
    P --> R
    Q --> R
    R --> S[Fairness preditivo por Race]
    S --> S1[Worst-group accuracy]
    S --> S2[Equal opportunity]
    S --> S3[Equalized odds]
    S --> S4[Calibration/ECE]
    P --> T[XAI temporal e espacial]
    T --> T1[Attention Rollout]
    T --> T2[Gradient x Input]
    T --> T3[Integrated Gradients]
    T --> T4[Oclusao por regioes faciais]
    T --> T5[Deletion Test]
    T --> U[Fairness explicativo]
    U --> U1[ADI]
    U --> U2[Temporal Bias Drift]
    U --> U3[Deletion AUC por grupo]
    U --> U4[Correlacao XAI-Fairness]
    U --> V[Robustez: agreement >= 0.70]
    V --> W[Tabelas, figuras e modelos]
    W --> X[Zip final no Drive e download local]
```
"""
    ),
    md(
        r"""
## 1. Instalação e verificação do ambiente

Esta célula prepara o Colab. O treinamento usa GPU via PyTorch; a extracao de landmarks
com MediaPipe roda principalmente em CPU e costuma ser a etapa mais longa.
"""
    ),
    code(
        r"""
%pip install -q --upgrade mediapipe opencv-python-headless seaborn scikit-learn tqdm requests
"""
    ),
    md(
        r"""
## 2. Imports, semente e configuração

`FULL_RUN=True` faz o notebook usar a base inteira. Para testar rapidamente o fluxo, mude para
`False`; o experimento final deve permanecer com `True`.

Os atributos sensíveis são usados apenas para auditoria. Eles não entram como entrada do modelo.

No Colab, esta célula também monta o Google Drive e cria a estrutura:
`raw/`, `cache/` e `outputs/`. O objetivo é persistir os artefatos fora da VM temporária
do Colab.
"""
    ),
    code(
        r"""
import os, re, gc, json, time, math, random, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from IPython.display import display

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

FULL_RUN = True
FAST_DEV_SAMPLES = 360
T_FRAMES = 100
MIN_FACE_DETECTION_RATE = 0.60
HIGH_CONSENSUS_THRESHOLD = 0.70

# "semantic" usa regiões expressivas da face; "full" usa os 468 landmarks.
LANDMARK_SET = "semantic"
ADD_TEMPORAL_DERIVATIVES = True

# Configuração Pedro/AcT: mantemos a base CREMA-D + MediaPipe, mas aproximamos a
# camada Transformer da implementação original usada pelo Pedro.
EMOT_REFERENCE = "Pedro/AcT base adaptado para CREMA-D+MediaPipe"
EMOT_MODEL_SIZE = "base"
PEDRO_ACT_SPECS = {
    "micro": {"N_HEADS": 1, "N_LAYERS": 4, "DROPOUT": 0.30, "MLP": 256},
    "small": {"N_HEADS": 2, "N_LAYERS": 5, "DROPOUT": 0.30, "MLP": 256},
    "base": {"N_HEADS": 3, "N_LAYERS": 6, "DROPOUT": 0.30, "MLP": 256},
    "large": {"N_HEADS": 4, "N_LAYERS": 6, "DROPOUT": 0.40, "MLP": 512},
    "enormous": {"N_HEADS": 8, "N_LAYERS": 10, "DROPOUT": 0.40, "MLP": 512},
}
EMOT_SPEC = PEDRO_ACT_SPECS[EMOT_MODEL_SIZE]
N_HEADS = EMOT_SPEC["N_HEADS"]
N_LAYERS = EMOT_SPEC["N_LAYERS"]
D_MODEL = 64 * N_HEADS
DIM_FF = 4 * D_MODEL
DROPOUT = EMOT_SPEC["DROPOUT"]
MLP_HEAD_SIZE = EMOT_SPEC["MLP"]

BATCH_SIZE = 256
EPOCHS = 3000
# Pedro treina até N_EPOCHS salvando o melhor checkpoint. No Colab, manter
# early stopping evita desperdício quando a validação estabiliza; use None para
# forçar as 3000 épocas completas.
PATIENCE = 50
USE_PEDRO_LR_SCHEDULE = True
WARMUP_PERC = 0.10
LR = 1.0
WEIGHT_DECAY = 3e-5
LABEL_SMOOTHING = 0.10
GRAD_CLIP_VALUE = 0.5
GRAD_CLIP_NORM = 1.0

# XAI: None usa todo o teste; se faltar tempo/memória, use por exemplo 600.
MAX_XAI_SAMPLES = None
XAI_BATCH_SIZE = 32
DELETION_FRACTIONS = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00]
TOP_K_LANDMARKS = 20
MAX_IG_SAMPLES = 600
IG_STEPS = 16
REGION_OCCLUSION_BATCH_SIZE = 64

DOWNLOAD_WORKERS = 16
DOWNLOAD_TIMEOUT = 60

try:
    import google.colab  # type: ignore
    IN_COLAB = True
except Exception:
    IN_COLAB = False

USE_GOOGLE_DRIVE = True
DRIVE_PROJECT_DIR = "cremad_fairness_xai"

if IN_COLAB and USE_GOOGLE_DRIVE:
    from google.colab import drive

    print("Montando Google Drive em /content/drive ...")
    drive.mount("/content/drive", force_remount=False)
    BASE_DIR = Path("/content/drive/MyDrive") / DRIVE_PROJECT_DIR
else:
    BASE_DIR = Path("/content/cremad_fairness_xai") if IN_COLAB else Path.cwd().parent / "data" / "cremad_fairness_xai"

RAW_DIR = BASE_DIR / "raw"
VIDEO_DIR = RAW_DIR / "videos_flv"
METADATA_DIR = RAW_DIR / "metadata"
CACHE_DIR = BASE_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
CKPT_DIR = OUTPUT_DIR / "models"
LOG_DIR = OUTPUT_DIR / "logs"
REPORT_DIR = OUTPUT_DIR / "reports"
XAI_DIR = OUTPUT_DIR / "xai"
FAIRNESS_DIR = OUTPUT_DIR / "fairness"
ZIP_DIR = OUTPUT_DIR / "zips"

RUN_ID = time.strftime("%Y%m%d_%H%M%S")
for d in [RAW_DIR, VIDEO_DIR, METADATA_DIR, CACHE_DIR, OUTPUT_DIR, FIG_DIR, TABLE_DIR, CKPT_DIR, LOG_DIR, REPORT_DIR, XAI_DIR, FAIRNESS_DIR, ZIP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

print(f"IN_COLAB: {IN_COLAB}")
print(f"USE_GOOGLE_DRIVE: {USE_GOOGLE_DRIVE}")
print(f"Device: {DEVICE}")
print(f"Base dir: {BASE_DIR}")
print(f"Raw videos: {VIDEO_DIR}")
print(f"Cache: {CACHE_DIR}")
print(f"Outputs: {OUTPUT_DIR}")
print(f"PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    if "A100" not in gpu_name.upper():
        print("Aviso: o experimento roda em outras GPUs, mas a estimativa principal assume Colab Pro com A100.")
else:
    gpu_name = "CPU"

run_config = {
    "run_id": RUN_ID,
    "full_run": FULL_RUN,
    "fast_dev_samples": FAST_DEV_SAMPLES,
    "t_frames": T_FRAMES,
    "landmark_set": LANDMARK_SET,
    "add_temporal_derivatives": ADD_TEMPORAL_DERIVATIVES,
    "emot_reference": EMOT_REFERENCE,
    "emot_model_size": EMOT_MODEL_SIZE,
    "n_heads": N_HEADS,
    "n_layers": N_LAYERS,
    "d_model": D_MODEL,
    "dim_ff": DIM_FF,
    "dropout": DROPOUT,
    "mlp_head_size": MLP_HEAD_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "patience": PATIENCE,
    "use_pedro_lr_schedule": USE_PEDRO_LR_SCHEDULE,
    "warmup_perc": WARMUP_PERC,
    "weight_decay": WEIGHT_DECAY,
    "label_smoothing": LABEL_SMOOTHING,
    "high_consensus_threshold": HIGH_CONSENSUS_THRESHOLD,
    "max_xai_samples": MAX_XAI_SAMPLES,
    "base_dir": str(BASE_DIR),
    "device": str(DEVICE),
    "gpu_name": gpu_name,
}
with open(LOG_DIR / f"run_config_{RUN_ID}.json", "w", encoding="utf-8") as f:
    json.dump(run_config, f, ensure_ascii=False, indent=2)
print(f"Configuração da execução salva em: {LOG_DIR / f'run_config_{RUN_ID}.json'}")

RUN_EVENTS = []

def current_gpu_stats():
    stats = {
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": gpu_name,
    }
    if torch.cuda.is_available():
        stats.update({
            "gpu_memory_allocated_mb": round(torch.cuda.memory_allocated() / (1024 ** 2), 2),
            "gpu_memory_reserved_mb": round(torch.cuda.memory_reserved() / (1024 ** 2), 2),
            "gpu_max_memory_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
            "gpu_max_memory_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024 ** 2), 2),
        })
        try:
            import subprocess as _subprocess
            query = "utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu"
            out = _subprocess.check_output(
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                text=True,
                stderr=_subprocess.STDOUT,
            ).strip().splitlines()[0]
            util, mem_used, mem_total, power, temp = [x.strip() for x in out.split(",")]
            stats.update({
                "nvidia_gpu_utilization_percent": float(util),
                "nvidia_memory_used_mb": float(mem_used),
                "nvidia_memory_total_mb": float(mem_total),
                "nvidia_power_draw_w": float(power),
                "nvidia_temperature_c": float(temp),
            })
        except Exception as exc:
            stats["nvidia_smi_error"] = str(exc)[:200]
    return stats

def log_event(stage, status="done", start_time=None, **extra):
    now = time.time()
    row = {
        "run_id": RUN_ID,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage,
        "status": status,
    }
    if start_time is not None:
        row["duration_sec"] = round(now - start_time, 3)
        row["duration_min"] = round((now - start_time) / 60, 3)
    row.update(current_gpu_stats())
    row.update(extra)
    RUN_EVENTS.append(row)
    log_df = pd.DataFrame(RUN_EVENTS)
    log_df.to_csv(LOG_DIR / f"resource_usage_log_{RUN_ID}.csv", index=False)
    log_df.to_csv(LOG_DIR / "resource_usage_log_latest.csv", index=False)
    msg = f"[LOG] {stage}: {status}"
    if "duration_sec" in row:
        msg += f" | {row['duration_sec']:.1f}s"
    if row.get("cuda_available"):
        msg += f" | VRAM usada={row.get('nvidia_memory_used_mb', row.get('gpu_memory_allocated_mb', 'n/a'))} MB"
    print(msg)
    return row

def show_gpu_snapshot(label):
    row = log_event(f"gpu_snapshot_{label}", status="snapshot")
    display(pd.DataFrame([row]).T.rename(columns={0: label}))
    return row

log_event("setup_colab_drive_and_dirs", status="ready")
"""
    ),
    md(
        r"""
## 3. Diagnóstico da GPU

O print da GPU ajuda a documentar o ambiente computacional usado no experimento.
"""
    ),
    code(
        r"""
import subprocess

if torch.cuda.is_available():
    subprocess.run(["nvidia-smi"], check=False)
    show_gpu_snapshot("inicio")
else:
    print("Nenhuma GPU foi detectada. No Colab, ative Runtime > Change runtime type > GPU.")
    show_gpu_snapshot("sem_gpu")
"""
    ),
    md(
        r"""
## 4. Fontes oficiais e classes do CREMA-D

O label principal vem do nome do arquivo. Exemplo: `1001_IEO_HAP_HI.flv` tem emoção
original `HAP`. Os votos dos avaliadores do CREMA-D entram apenas na analise de robustez.
"""
    ),
    code(
        r"""
CREMAD_RAW_BASE = "https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master"
CREMAD_MEDIA_BASE = "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master"

URL_DEMOGRAPHICS = f"{CREMAD_RAW_BASE}/VideoDemographics.csv"
URL_SUMMARY = f"{CREMAD_RAW_BASE}/processedResults/summaryTable.csv"
URL_TABULATED = f"{CREMAD_RAW_BASE}/processedResults/tabulatedVotes.csv"

LABEL_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}
CLASS_CODES = list(LABEL_MAP.keys())
CLASS_NAMES = [LABEL_MAP[c] for c in CLASS_CODES]
CODE_TO_ID = {code: idx for idx, code in enumerate(CLASS_CODES)}
ID_TO_CODE = {idx: code for code, idx in CODE_TO_ID.items()}
N_CLASSES = len(CLASS_NAMES)

print("Classes usadas:")
for code, name in LABEL_MAP.items():
    print(f"  {code}: {name}")
"""
    ),
    md(
        r"""
## 5. Manifesto experimental

Esta célula junta metadados demográficos, labels originais e consenso facial. O campo
`face_agreement` vem dos avaliadores do próprio CREMA-D, não de avaliadores novos.
"""
    ),
    code(
        r"""
def read_csv_cached(url, path):
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        print(f"Carregando metadado em cache: {path}")
        return pd.read_csv(path)
    print(f"Baixando metadado oficial: {url}")
    df = pd.read_csv(url)
    df.to_csv(path, index=False)
    print(f"Metadado salvo em: {path}")
    return df


def load_cremad_metadata():
    demographics = read_csv_cached(URL_DEMOGRAPHICS, METADATA_DIR / "VideoDemographics.csv")
    summary = read_csv_cached(URL_SUMMARY, METADATA_DIR / "summaryTable.csv").rename(columns={"FileName": "file_stem"})
    tabulated = read_csv_cached(URL_TABULATED, METADATA_DIR / "tabulatedVotes.csv")

    summary["ActorID"] = summary["file_stem"].str.extract(r"^(\d{4})").astype(int)

    # IDs 200001..299999 correspondem aos votos faciais no tabulatedVotes.
    face_votes = tabulated[(tabulated["Unnamed: 0"] >= 200000) & (tabulated["Unnamed: 0"] < 300000)].copy()
    face_votes = face_votes.rename(
        columns={
            "fileName": "file_stem",
            "agreement": "face_agreement",
            "emoVote": "face_emo_vote",
            "numResponses": "face_num_responses",
        }
    )
    face_votes = face_votes[["file_stem", "face_num_responses", "face_agreement", "face_emo_vote"]]

    meta = summary.merge(demographics, on="ActorID", how="left")
    meta = meta.merge(face_votes, on="file_stem", how="left")

    parts = meta["file_stem"].str.split("_", expand=True)
    meta["sentence_code"] = parts[1]
    meta["label_code"] = parts[2]
    meta["intensity_code"] = parts[3]
    meta["label_name"] = meta["label_code"].map(LABEL_MAP)
    meta["y"] = meta["label_code"].map(CODE_TO_ID)
    meta["video_url"] = meta["file_stem"].map(lambda s: f"{CREMAD_MEDIA_BASE}/VideoFlash/{s}.flv")
    meta["video_path"] = meta["file_stem"].map(lambda s: str(VIDEO_DIR / f"{s}.flv"))
    meta["age_bin"] = pd.cut(
        meta["Age"],
        bins=[0, 29, 39, 49, 59, 200],
        labels=["20-29", "30-39", "40-49", "50-59", "60+"],
        include_lowest=True,
    ).astype(str)

    meta = meta[meta["label_code"].isin(CLASS_CODES)].copy()
    return meta.sort_values("file_stem").reset_index(drop=True)

metadata_start = time.time()
meta_full = load_cremad_metadata()
meta_full.to_csv(TABLE_DIR / "manifest_full_cremad.csv", index=False)
if FULL_RUN:
    meta_run = meta_full.copy()
else:
    meta_run = (
        meta_full.groupby(["label_code", "Race"], group_keys=False)
        .apply(lambda x: x.sample(min(len(x), max(1, FAST_DEV_SAMPLES // 24)), random_state=SEED))
        .reset_index(drop=True)
    )

print(f"Manifesto completo: {len(meta_full):,} vídeos")
print(f"Execução atual: {len(meta_run):,} vídeos")
print(f"Atores unicos: {meta_run['ActorID'].nunique()}")
meta_run.to_csv(TABLE_DIR / "manifest_run_cremad.csv", index=False)
print(f"Manifesto completo salvo em: {TABLE_DIR / 'manifest_full_cremad.csv'}")
print(f"Manifesto desta execução salvo em: {TABLE_DIR / 'manifest_run_cremad.csv'}")
log_event(
    "metadata_manifest",
    start_time=metadata_start,
    n_full=int(len(meta_full)),
    n_run=int(len(meta_run)),
    n_actors=int(meta_run["ActorID"].nunique()),
)
display(meta_run.head())
"""
    ),
    md(
        r"""
## 5.1 Estimativa de tempo da execução completa

Esta estimativa ajuda a planejar o uso do Colab Pro. Ela não substitui os logs reais:
ao final do notebook, a seção de processamento salva os tempos medidos por etapa em
`outputs/tables/processing_time_by_stage.csv`.

O ponto crítico é que a A100 acelera treino e XAI, mas a extração de landmarks com
MediaPipe depende bastante de CPU, leitura dos vídeos e velocidade do Google Drive.
"""
    ),
    code(
        r"""
estimated_sampled_frames = len(meta_run) * T_FRAMES

runtime_estimate_rows = [
    {
        "stage": "Instalação, Drive e metadados",
        "first_run_a100": "5-15 min",
        "cached_run_a100": "2-5 min",
        "main_bottleneck": "pip/Colab/Drive",
    },
    {
        "stage": "Download dos vídeos CREMA-D",
        "first_run_a100": "20-60 min",
        "cached_run_a100": "0-5 min",
        "main_bottleneck": "rede e GitHub media",
    },
    {
        "stage": "Estatísticas de vídeo e amostras visuais",
        "first_run_a100": "5-20 min",
        "cached_run_a100": "5-20 min",
        "main_bottleneck": "I/O de vídeo",
    },
    {
        "stage": "MediaPipe landmarks em 100 frames/vídeo",
        "first_run_a100": "3-7 h",
        "cached_run_a100": "0-10 min",
        "main_bottleneck": "CPU + leitura dos frames",
    },
    {
        "stage": "Treino dos modelos",
        "first_run_a100": "1.5-4 h",
        "cached_run_a100": "1.5-4 h",
        "main_bottleneck": "GPU/early stopping",
    },
    {
        "stage": "Fairness, robustez e matrizes",
        "first_run_a100": "10-30 min",
        "cached_run_a100": "10-30 min",
        "main_bottleneck": "CPU",
    },
    {
        "stage": "XAI: attention, gradientes, IG, oclusão e deletion",
        "first_run_a100": "1-3 h",
        "cached_run_a100": "1-3 h",
        "main_bottleneck": "GPU + número de amostras XAI",
    },
    {
        "stage": "Zip final e download dos artefatos",
        "first_run_a100": "10-30 min",
        "cached_run_a100": "10-30 min",
        "main_bottleneck": "tamanho do cache/outputs",
    },
]
runtime_estimate = pd.DataFrame(runtime_estimate_rows)
runtime_estimate_path = TABLE_DIR / "runtime_estimate_table.csv"
runtime_estimate.to_csv(runtime_estimate_path, index=False)

print(f"Vídeos na execução: {len(meta_run):,}")
print(f"Frames amostrados para MediaPipe: {estimated_sampled_frames:,} ({T_FRAMES} por vídeo)")
print("Estimativa A100 Colab Pro, primeira execução sem cache: 6-15 horas.")
print("Estimativa A100 com vídeos/landmarks já em cache: 3-8 horas.")
print("Se PATIENCE=None e as 3000 épocas forem forçadas, o treino pode passar de 24 horas.")
print(f"Tabela de estimativa salva em: {runtime_estimate_path}")
display(runtime_estimate)
log_event(
    "runtime_estimate",
    status="estimated",
    n_videos=int(len(meta_run)),
    sampled_frames=int(estimated_sampled_frames),
    first_run_a100_total="6-15 h",
    cached_run_a100_total="3-8 h",
)
"""
    ),
    md(
        r"""
## 6. Estatísticas descritivas

Antes de treinar, documentamos a composição da base. Isso evita discutir fairness sem
saber o tamanho dos grupos e o balanceamento das emoções.
"""
    ),
    code(
        r"""
def show_counts(df, column, title):
    counts = df[column].value_counts(dropna=False).rename_axis(column).reset_index(name="n")
    counts["percent"] = 100 * counts["n"] / counts["n"].sum()
    print(f"\n{title}")
    display(counts)
    return counts

descriptive_start = time.time()
descriptive_outputs = {
    "label_name": show_counts(meta_run, "label_name", "Distribuição por emoção"),
    "Race": show_counts(meta_run, "Race", "Distribuição por Race"),
    "Ethnicity": show_counts(meta_run, "Ethnicity", "Distribuição por Ethnicity"),
    "Sex": show_counts(meta_run, "Sex", "Distribuição por Sex"),
    "age_bin": show_counts(meta_run, "age_bin", "Distribuição por faixa etária"),
}
for name, table in descriptive_outputs.items():
    out_path = TABLE_DIR / f"descriptive_{name}.csv"
    table.to_csv(out_path, index=False)
    print(f"Tabela descritiva salva: {out_path}")

print("\nEmocao x Race")
crosstab_emotion_race = pd.crosstab(meta_run["label_name"], meta_run["Race"], margins=True)
display(crosstab_emotion_race)
crosstab_emotion_race.to_csv(TABLE_DIR / "crosstab_emotion_by_race.csv")

print("\nEmocao x Ethnicity")
crosstab_emotion_ethnicity = pd.crosstab(meta_run["label_name"], meta_run["Ethnicity"], margins=True)
display(crosstab_emotion_ethnicity)
crosstab_emotion_ethnicity.to_csv(TABLE_DIR / "crosstab_emotion_by_ethnicity.csv")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
sns.countplot(data=meta_run, y="label_name", order=CLASS_NAMES, ax=axes[0, 0], color="#4C78A8")
axes[0, 0].set_title("Amostras por emoção original")
sns.countplot(data=meta_run, y="Race", order=meta_run["Race"].value_counts().index, ax=axes[0, 1], color="#F58518")
axes[0, 1].set_title("Amostras por Race")
sns.countplot(data=meta_run, y="Ethnicity", order=meta_run["Ethnicity"].value_counts().index, ax=axes[1, 0], color="#54A24B")
axes[1, 0].set_title("Amostras por Ethnicity")
sns.histplot(data=meta_run, x="face_agreement", bins=20, ax=axes[1, 1], color="#B279A2")
axes[1, 1].axvline(HIGH_CONSENSUS_THRESHOLD, color="black", linestyle="--")
axes[1, 1].set_title("Consenso facial dos avaliadores CREMA-D")
plt.tight_layout()
fig.savefig(FIG_DIR / "01_descriptive_distributions.png", bbox_inches="tight")
plt.show()
log_event("descriptive_statistics", start_time=descriptive_start, n_tables=len(descriptive_outputs), figure="01_descriptive_distributions.png")
"""
    ),
    md(
        r"""
## 7. Download dos vídeos

Os arquivos já existentes são pulados. Assim, se o Colab cair, basta executar novamente.
"""
    ),
    code(
        r"""
def download_file(url, path, timeout=DOWNLOAD_TIMEOUT, retries=3):
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return {"path": str(path), "status": "exists", "bytes": path.stat().st_size, "error": None}
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                tmp = path.with_suffix(path.suffix + ".tmp")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
                tmp.replace(path)
            return {"path": str(path), "status": "downloaded", "bytes": path.stat().st_size, "error": None}
        except Exception as exc:
            last_error = str(exc)
            time.sleep(1.5 * attempt)
    return {"path": str(path), "status": "failed", "bytes": 0, "error": last_error}

records = meta_run[["video_url", "video_path"]].drop_duplicates().to_dict("records")
start = time.time()
results = []
with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:
    futures = [ex.submit(download_file, r["video_url"], r["video_path"]) for r in records]
    for fut in tqdm(as_completed(futures), total=len(futures), desc="Baixando vídeos"):
        results.append(fut.result())

download_report = pd.DataFrame(results)
download_report.to_csv(TABLE_DIR / "download_report.csv", index=False)
print(f"Tempo de download: {(time.time() - start) / 60:.1f} min")
display(download_report["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="n"))
failed = download_report[download_report["status"] == "failed"]
if len(failed):
    display(failed.head(20))
log_event(
    "download_videos",
    start_time=start,
    n_files=int(len(download_report)),
    n_downloaded=int((download_report["status"] == "downloaded").sum()),
    n_existing=int((download_report["status"] == "exists").sum()),
    n_failed=int((download_report["status"] == "failed").sum()),
)
"""
    ),
    md(
        r"""
## 8. Estatísticas dos vídeos e amostragem temporal

Antes dos landmarks, medimos duração, FPS e número de frames dos vídeos baixados. Essa tabela
responde à pergunta metodológica: quando usamos `T_FRAMES=100`, aproximando a configuração
temporal do EmoT do Pedro, estamos amostrando quantos quadros em vídeos de tamanhos diferentes?
"""
    ),
    code(
        r"""
def inspect_video_file(path):
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {
            "video_path": str(path),
            "video_status": "open_failed",
            "frame_count": np.nan,
            "fps": np.nan,
            "duration_sec": np.nan,
            "width": np.nan,
            "height": np.nan,
            "sample_stride_frames": np.nan,
            "sample_stride_sec": np.nan,
        }
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration_sec = frame_count / fps if fps > 0 else np.nan
    sample_stride_frames = frame_count / max(T_FRAMES - 1, 1) if frame_count > 0 else np.nan
    sample_stride_sec = duration_sec / max(T_FRAMES - 1, 1) if np.isfinite(duration_sec) else np.nan
    return {
        "video_path": str(path),
        "video_status": "ok",
        "frame_count": frame_count,
        "fps": fps,
        "duration_sec": duration_sec,
        "width": width,
        "height": height,
        "sample_stride_frames": sample_stride_frames,
        "sample_stride_sec": sample_stride_sec,
    }

video_stats_start = time.time()
video_stats_rows = []
for _, row in tqdm(meta_run[["file_stem", "video_path"]].drop_duplicates().iterrows(), total=meta_run["file_stem"].nunique(), desc="Inspecionando vídeos"):
    stats = inspect_video_file(row["video_path"])
    stats["file_stem"] = row["file_stem"]
    video_stats_rows.append(stats)

video_stats = pd.DataFrame(video_stats_rows)
video_stats = video_stats.merge(meta_run[["file_stem", "label_name", "Race", "Ethnicity", "Sex", "age_bin"]], on="file_stem", how="left")
video_stats.to_csv(TABLE_DIR / "video_duration_stats.csv", index=False)

print(f"Estatísticas dos vídeos salvas em: {TABLE_DIR / 'video_duration_stats.csv'}")
display(video_stats[["frame_count", "fps", "duration_sec", "sample_stride_frames", "sample_stride_sec"]].describe().T)
display(video_stats.groupby("Race", dropna=False)[["duration_sec", "frame_count", "sample_stride_sec"]].agg(["count", "mean", "median", "min", "max"]))
log_event(
    "video_duration_statistics",
    start_time=video_stats_start,
    n_videos=int(len(video_stats)),
    mean_duration_sec=float(video_stats["duration_sec"].mean(skipna=True)),
    mean_frame_count=float(video_stats["frame_count"].mean(skipna=True)),
)
"""
    ),
    md(
        r"""
## 8. Landmarks faciais com MediaPipe

Usamos regiões expressivas para reduzir memória e melhorar interpretabilidade: labios,
olhos, sobrancelhas, nariz e contorno facial. O modelo recebe apenas geometria facial, nao
cor/textura diretamente.
"""
    ),
    code(
        r"""
import mediapipe as mp

print(f"MediaPipe version: {getattr(mp, '__version__', 'unknown')}")

# O MediaPipe tem duas APIs Python em circulação:
# 1. API clássica: `mp.solutions.face_mesh.FaceMesh`
# 2. API moderna: `mp.tasks.vision.FaceLandmarker`
#
# Alguns ambientes recentes expõem apenas a API moderna. Por isso, esta célula
# detecta a API disponível e usa a alternativa correta automaticamente.
MEDIAPIPE_BACKEND = None
mp_face_mesh = None
BaseOptions = None
FaceLandmarker = None
FaceLandmarkerOptions = None
VisionRunningMode = None

if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
    mp_face_mesh = mp.solutions.face_mesh
    MEDIAPIPE_BACKEND = "solutions"
elif hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    MEDIAPIPE_BACKEND = "tasks"
else:
    try:
        from mediapipe.tasks import python as mp_tasks_python
        from mediapipe.tasks.python import vision as mp_tasks_vision

        BaseOptions = mp_tasks_python.BaseOptions
        FaceLandmarker = mp_tasks_vision.FaceLandmarker
        FaceLandmarkerOptions = mp_tasks_vision.FaceLandmarkerOptions
        VisionRunningMode = mp_tasks_vision.RunningMode
        MEDIAPIPE_BACKEND = "tasks"
    except Exception as exc:
        raise ImportError(
            "A instalação de mediapipe não expõe `mp.solutions` nem a API "
            "`mediapipe.tasks`. Rode novamente a célula de instalação com "
            "`%pip install --upgrade mediapipe` e reinicie o runtime/kernel "
            "antes de continuar."
        ) from exc

print(f"MediaPipe backend selecionado: {MEDIAPIPE_BACKEND}")

# IDs semânticos baseados na topologia canônica do MediaPipe Face Mesh.
# Eles evitam depender de constantes `FACEMESH_*`, que só existem na API clássica.
REGION_IDS = {
    "lips": {
        0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95,
        146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312,
        314, 317, 318, 321, 324, 375, 402, 405, 409, 415,
    },
    "left_eye": {
        249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388,
        390, 398, 466,
    },
    "right_eye": {
        7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163,
        173, 246,
    },
    "left_eyebrow": {276, 282, 283, 285, 293, 295, 296, 300, 334, 336},
    "right_eyebrow": {46, 52, 53, 55, 63, 65, 66, 70, 105, 107},
    "nose": {
        1, 2, 4, 5, 45, 48, 64, 94, 97, 98, 115, 168, 195, 197, 219,
        220, 235, 240, 275, 278, 290, 305, 326, 327, 344, 439, 440,
    },
    "face_oval": {
        10, 21, 54, 58, 67, 93, 103, 109, 127, 132, 136, 148, 149, 150,
        152, 162, 172, 176, 234, 251, 284, 288, 297, 323, 332, 338, 356,
        361, 365, 377, 378, 379, 389, 397, 400, 454,
    },
}

FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
FACE_LANDMARKER_MODEL_PATH = CACHE_DIR / "face_landmarker.task"

def ensure_face_landmarker_model():
    # Baixa o modelo da API MediaPipe Tasks quando necessário.
    if MEDIAPIPE_BACKEND != "tasks":
        return None
    if FACE_LANDMARKER_MODEL_PATH.exists() and FACE_LANDMARKER_MODEL_PATH.stat().st_size > 0:
        return FACE_LANDMARKER_MODEL_PATH
    print("Baixando modelo MediaPipe FaceLandmarker...")
    response = requests.get(FACE_LANDMARKER_MODEL_URL, timeout=120)
    response.raise_for_status()
    FACE_LANDMARKER_MODEL_PATH.write_bytes(response.content)
    print(f"Modelo salvo em: {FACE_LANDMARKER_MODEL_PATH}")
    return FACE_LANDMARKER_MODEL_PATH

def create_face_landmarker():
    # Cria o detector de landmarks compatível com a API disponível.
    if MEDIAPIPE_BACKEND == "solutions":
        return mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    model_path = ensure_face_landmarker_model()
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return FaceLandmarker.create_from_options(options)

if LANDMARK_SET == "semantic":
    SELECTED_LANDMARK_IDS = sorted(set().union(*REGION_IDS.values()))
elif LANDMARK_SET == "full":
    SELECTED_LANDMARK_IDS = list(range(468))
else:
    raise ValueError("LANDMARK_SET deve ser 'semantic' ou 'full'.")

LANDMARK_TO_REGIONS = {}
for region, ids in REGION_IDS.items():
    for idx in ids:
        LANDMARK_TO_REGIONS.setdefault(idx, []).append(region)

N_LANDMARKS = len(SELECTED_LANDMARK_IDS)
BASE_FEATURE_DIM = N_LANDMARKS * 2
print(f"Landmarks selecionados: {N_LANDMARKS}")
print(f"Features base por frame: {BASE_FEATURE_DIM}")
for region, ids in REGION_IDS.items():
    print(region, len(set(ids).intersection(SELECTED_LANDMARK_IDS)))
log_event("mediapipe_landmark_setup", status=MEDIAPIPE_BACKEND, n_landmarks=int(N_LANDMARKS), feature_dim=int(BASE_FEATURE_DIM))
"""
    ),
    md(
        r"""
## 9. Extração e cache dos landmarks

Para cada video: amostramos frames uniformemente, detectamos a face, extraimos landmarks,
normalizamos centro/escala e interpolamos frames sem detecção quando a taxa de detecção
e suficiente.

O relatório de processamento permite avaliar se a própria detecção facial falha mais em
algum grupo demográfico.
"""
    ),
    code(
        r"""
LANDMARK_CACHE = CACHE_DIR / f"cremad_landmarks_T{T_FRAMES}_{LANDMARK_SET}.npz"
PROCESSING_REPORT_CSV = TABLE_DIR / f"processing_report_T{T_FRAMES}_{LANDMARK_SET}.csv"

def sample_video_frames(video_path, t_frames=T_FRAMES):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        return []
    idx = np.linspace(0, len(frames) - 1, t_frames).round().astype(int)
    return [frames[i] for i in idx]

def normalize_landmarks_xy(coords_xy):
    center = coords_xy.mean(axis=0, keepdims=True)
    centered = coords_xy - center
    span = coords_xy.max(axis=0) - coords_xy.min(axis=0)
    scale = max(float(span.max()), 1e-6)
    return centered / scale

def extract_frame_landmarks(frame_bgr, face_landmarker):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if MEDIAPIPE_BACKEND == "solutions":
        result = face_landmarker.process(rgb)
        if not result.multi_face_landmarks:
            return None
        landmarks = result.multi_face_landmarks[0].landmark
    else:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = face_landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        landmarks = result.face_landmarks[0]
    coords = np.array([[landmarks[idx].x, landmarks[idx].y] for idx in SELECTED_LANDMARK_IDS], dtype=np.float32)
    coords = normalize_landmarks_xy(coords)
    return coords.reshape(-1).astype(np.float32)

def interpolate_missing_frames(seq):
    valid = ~np.isnan(seq).any(axis=1)
    if valid.sum() == 0:
        return None
    if valid.all():
        return seq.astype(np.float32)
    x = np.arange(seq.shape[0])
    out = seq.copy()
    for d in range(seq.shape[1]):
        col = seq[:, d]
        out[~valid, d] = np.interp(x[~valid], x[valid], col[valid])
    return out.astype(np.float32)

def process_one_video(row, face_landmarker):
    frames = sample_video_frames(row["video_path"], T_FRAMES)
    if not frames:
        return None, {"file_stem": row["file_stem"], "status": "no_frames", "n_frames_read": 0, "detected_frames": 0, "detection_rate": 0.0}
    seq = np.full((T_FRAMES, BASE_FEATURE_DIM), np.nan, dtype=np.float32)
    detected = 0
    for i, frame in enumerate(frames):
        landmarks = extract_frame_landmarks(frame, face_landmarker)
        if landmarks is not None:
            seq[i] = landmarks
            detected += 1
    rate = detected / T_FRAMES
    if rate < MIN_FACE_DETECTION_RATE:
        return None, {"file_stem": row["file_stem"], "status": "low_detection_rate", "n_frames_read": len(frames), "detected_frames": detected, "detection_rate": rate}
    seq = interpolate_missing_frames(seq)
    if seq is None:
        return None, {"file_stem": row["file_stem"], "status": "interpolation_failed", "n_frames_read": len(frames), "detected_frames": detected, "detection_rate": rate}
    return seq, {"file_stem": row["file_stem"], "status": "ok", "n_frames_read": len(frames), "detected_frames": detected, "detection_rate": rate}

landmark_start = time.time()
if LANDMARK_CACHE.exists():
    cached = np.load(LANDMARK_CACHE, allow_pickle=True)
    X_base = cached["X_base"]
    y = cached["y"]
    valid_file_stems = cached["file_stems"].astype(str)
    processing_report = pd.read_csv(PROCESSING_REPORT_CSV)
    print(f"Cache carregado: {LANDMARK_CACHE}")
    landmark_status = "cache_loaded"
else:
    X_list, y_list, valid_file_stems, report_rows = [], [], [], []
    rows = meta_run.to_dict("records")
    with create_face_landmarker() as face_landmarker:
        for row in tqdm(rows, desc="Extraindo landmarks"):
            seq, report = process_one_video(row, face_landmarker)
            report_rows.append(report)
            if seq is not None:
                X_list.append(seq)
                y_list.append(int(row["y"]))
                valid_file_stems.append(row["file_stem"])
    X_base = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    valid_file_stems = np.array(valid_file_stems, dtype=str)
    processing_report = pd.DataFrame(report_rows)
    processing_report.to_csv(PROCESSING_REPORT_CSV, index=False)
    np.savez_compressed(LANDMARK_CACHE, X_base=X_base, y=y, file_stems=valid_file_stems, selected_landmark_ids=np.array(SELECTED_LANDMARK_IDS, dtype=np.int64))
    print(f"Cache salvo: {LANDMARK_CACHE}")
    print(f"Tempo de extracao: {(time.time() - landmark_start) / 60:.1f} min")
    landmark_status = "extracted"

print(f"X_base: {X_base.shape} | y: {y.shape}")
display(processing_report["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="n"))
log_event(
    "landmark_processing",
    status=landmark_status,
    start_time=landmark_start,
    n_valid_videos=int(len(valid_file_stems)),
    n_total_videos=int(len(processing_report)),
    ok_rate=float((processing_report["status"] == "ok").mean()),
    mean_detection_rate=float(processing_report["detection_rate"].mean()),
)
"""
    ),
    md(
        r"""
## 9.1 Amostras visuais do pré-processamento

Esta célula gera prints visuais para a dissertação e para a reunião com os orientadores:

1. amostras de frames reais do CREMA-D por grupo;
2. sequência temporal amostrada;
3. frame com landmarks detectados;
4. landmarks normalizados, que são a entrada efetiva do modelo.
"""
    ),
    code(
        r"""
VISUAL_SAMPLE_MAX_GROUPS = 6
VISUAL_STRIP_FRAMES = 8
VISUAL_FRAME_HEIGHT = 96

REGION_COLORS = {
    "lips": (214, 39, 40),
    "left_eye": (31, 119, 180),
    "right_eye": (31, 119, 180),
    "left_eyebrow": (148, 103, 189),
    "right_eyebrow": (148, 103, 189),
    "nose": (44, 160, 44),
    "face_oval": (127, 127, 127),
}

def extract_selected_landmarks_pixels(frame_bgr, face_landmarker):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if MEDIAPIPE_BACKEND == "solutions":
        result = face_landmarker.process(rgb)
        if not result.multi_face_landmarks:
            return None
        landmarks = result.multi_face_landmarks[0].landmark
    else:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = face_landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        landmarks = result.face_landmarks[0]
    return np.array([[landmarks[idx].x * w, landmarks[idx].y * h] for idx in SELECTED_LANDMARK_IDS], dtype=np.float32)

def draw_landmark_overlay(frame_bgr, pixel_coords):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).copy()
    if pixel_coords is None:
        return rgb
    id_to_pos = {lm_id: pos for pos, lm_id in enumerate(SELECTED_LANDMARK_IDS)}
    for region, ids in REGION_IDS.items():
        color = REGION_COLORS.get(region, (255, 255, 255))
        for lm_id in ids:
            pos = id_to_pos.get(lm_id)
            if pos is None:
                continue
            x, y = pixel_coords[pos]
            cv2.circle(rgb, (int(round(x)), int(round(y))), 2, color, -1, lineType=cv2.LINE_AA)
    return rgb

def make_temporal_strip(frames, max_frames=VISUAL_STRIP_FRAMES, height=VISUAL_FRAME_HEIGHT):
    if not frames:
        return None
    idx = np.linspace(0, len(frames) - 1, min(max_frames, len(frames))).round().astype(int)
    imgs = []
    for i in idx:
        rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        scale = height / max(1, rgb.shape[0])
        width = max(1, int(rgb.shape[1] * scale))
        imgs.append(cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA))
    return np.concatenate(imgs, axis=1)

visual_start = time.time()
visual_candidates = processing_report.merge(
    meta_run[["file_stem", "video_path", "label_name", "Race", "Ethnicity", "Sex", "age_bin"]],
    on="file_stem",
    how="left",
)
visual_candidates = visual_candidates[visual_candidates["status"] == "ok"].drop_duplicates("file_stem")

if visual_candidates.empty:
    print("Nenhuma amostra processada com status ok para visualização.")
else:
    sample_parts = []
    for _, group in visual_candidates.groupby("Race", dropna=False):
        sample_parts.append(group.sample(1, random_state=SEED))
    visual_samples = pd.concat(sample_parts, ignore_index=True).head(VISUAL_SAMPLE_MAX_GROUPS)
    visual_samples.to_csv(TABLE_DIR / "visual_sample_manifest.csv", index=False)
    display(visual_samples[["file_stem", "label_name", "Race", "Ethnicity", "Sex", "age_bin", "detection_rate"]])

    n = len(visual_samples)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6))
    axes = np.atleast_1d(axes)

    visual_payload = []
    with create_face_landmarker() as face_landmarker:
        for ax, (_, row) in zip(axes, visual_samples.iterrows()):
            frames = sample_video_frames(row["video_path"], T_FRAMES)
            frame = frames[len(frames) // 2] if frames else None
            if frame is None:
                ax.text(0.5, 0.5, "sem frame", ha="center", va="center")
                ax.axis("off")
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ax.imshow(rgb)
            ax.set_title(f"{row['Race']}\n{row['label_name']}", fontsize=9)
            ax.axis("off")
            pixel_coords = extract_selected_landmarks_pixels(frame, face_landmarker)
            overlay = draw_landmark_overlay(frame, pixel_coords)
            norm_flat = extract_frame_landmarks(frame, face_landmarker)
            norm_xy = norm_flat.reshape(-1, 2) if norm_flat is not None else None
            strip = make_temporal_strip(frames)
            visual_payload.append((row, rgb, strip, overlay, norm_xy))

    plt.tight_layout()
    sample_fig_path = FIG_DIR / "02_visual_sample_frames_by_race.png"
    fig.savefig(sample_fig_path, bbox_inches="tight")
    plt.show()

    n_pipe = len(visual_payload)
    fig, axes = plt.subplots(n_pipe, 4, figsize=(15, 3.2 * n_pipe))
    axes = np.asarray(axes).reshape(n_pipe, 4)
    for i, (row, rgb, strip, overlay, norm_xy) in enumerate(visual_payload):
        title_prefix = f"{row['file_stem']} | {row['Race']} | {row['label_name']}"
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title("Frame bruto\n" + title_prefix, fontsize=8)
        if strip is not None:
            axes[i, 1].imshow(strip)
        else:
            axes[i, 1].text(0.5, 0.5, "sem strip", ha="center", va="center")
        axes[i, 1].set_title(f"Amostragem temporal\n{VISUAL_STRIP_FRAMES} de {T_FRAMES} frames", fontsize=8)
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("Processamento\nlandmarks detectados", fontsize=8)
        if norm_xy is not None:
            axes[i, 3].scatter(norm_xy[:, 0], -norm_xy[:, 1], s=8, alpha=0.8)
            axes[i, 3].set_aspect("equal", adjustable="box")
        else:
            axes[i, 3].text(0.5, 0.5, "sem landmarks", ha="center", va="center")
        axes[i, 3].set_title("Entrada do modelo\nlandmarks normalizados", fontsize=8)
        for j in range(4):
            axes[i, j].axis("off")
    plt.tight_layout()
    pipeline_fig_path = FIG_DIR / "02_visual_preprocessing_pipeline.png"
    fig.savefig(pipeline_fig_path, bbox_inches="tight")
    plt.show()

    print(f"Figura de amostras salva em: {sample_fig_path}")
    print(f"Figura do pipeline visual salva em: {pipeline_fig_path}")
    log_event(
        "visual_preprocessing_samples",
        start_time=visual_start,
        n_samples=int(n),
        sample_figure=str(sample_fig_path),
        pipeline_figure=str(pipeline_fig_path),
    )
"""
    ),
    md(
        r"""
## 10. Fairness do pré-processamento

Antes do modelo, avaliamos a taxa de detecção facial por grupo. Se a detecção falha mais
em um grupo, isso pode contaminar as métricas posteriores.
"""
    ),
    code(
        r"""
meta_valid = meta_run[meta_run["file_stem"].isin(valid_file_stems)].copy()
meta_valid = meta_valid.set_index("file_stem").loc[valid_file_stems].reset_index()
meta_valid["y"] = y
meta_valid["label_name"] = meta_valid["label_code"].map(LABEL_MAP)

proc_demo = processing_report.merge(meta_run[["file_stem", "Race", "Ethnicity", "Sex", "age_bin", "label_name"]], on="file_stem", how="left")
preprocessing_fairness_start = time.time()
processing_summaries = []
for group_col in ["Race", "Ethnicity", "Sex", "age_bin"]:
    print(f"\nProcessamento por {group_col}")
    summary = proc_demo.groupby(group_col, dropna=False).agg(
        n=("file_stem", "count"),
        ok_rate=("status", lambda s: float((s == "ok").mean())),
        mean_detection_rate=("detection_rate", "mean"),
        median_detection_rate=("detection_rate", "median"),
    ).reset_index()
    display(summary)
    summary.insert(0, "group_col", group_col)
    processing_summaries.append(summary)

processing_quality_by_group = pd.concat(processing_summaries, ignore_index=True)
processing_quality_by_group.to_csv(TABLE_DIR / "processing_quality_by_demographic_group.csv", index=False)
processing_report.to_csv(LOG_DIR / f"processing_report_{RUN_ID}.csv", index=False)
print(f"Resumo de qualidade por grupo salvo em: {TABLE_DIR / 'processing_quality_by_demographic_group.csv'}")
print(f"Log de processamento salvo em: {LOG_DIR / f'processing_report_{RUN_ID}.csv'}")

fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=proc_demo, x="Race", y="detection_rate", ax=ax)
ax.set_title("Taxa de detecção facial por Race")
ax.set_ylim(0, 1.05)
plt.xticks(rotation=20)
plt.tight_layout()
fig.savefig(FIG_DIR / "02_preprocessing_detection_rate_by_race.png", bbox_inches="tight")
plt.show()
log_event(
    "preprocessing_fairness_quality",
    start_time=preprocessing_fairness_start,
    n_groups=int(processing_quality_by_group["group_col"].nunique()),
    mean_detection_rate=float(proc_demo["detection_rate"].mean()),
)
"""
    ),
    md(
        r"""
## 11. Derivadas temporais

Adicionamos velocidade, aceleração e jerk aos landmarks. Assim, o EmoT recebe informação
de movimento facial, não apenas a geometria estática de cada frame.
"""
    ),
    code(
        r"""
def compute_temporal_derivatives(X):
    vel = np.zeros_like(X, dtype=np.float32)
    vel[:, 1:, :] = X[:, 1:, :] - X[:, :-1, :]
    acc = np.zeros_like(X, dtype=np.float32)
    acc[:, 1:-1, :] = X[:, 2:, :] - 2 * X[:, 1:-1, :] + X[:, :-2, :]
    jerk = np.zeros_like(X, dtype=np.float32)
    jerk[:, 2:-2, :] = X[:, 4:, :] - 2 * X[:, 3:-1, :] + 2 * X[:, 1:-3, :] - X[:, :-4, :]
    return np.concatenate([X, vel, acc, jerk], axis=-1).astype(np.float32)

derivative_start = time.time()
if ADD_TEMPORAL_DERIVATIVES:
    X = compute_temporal_derivatives(X_base)
    FEATURE_BLOCKS = ["pos", "vel", "acc", "jerk"]
else:
    X = X_base.astype(np.float32)
    FEATURE_BLOCKS = ["pos"]

T, D = X.shape[1], X.shape[2]
print(f"Tensor final X: {X.shape}")
print(f"Blocos de feature: {FEATURE_BLOCKS}")
log_event("temporal_feature_engineering", start_time=derivative_start, x_shape=str(X.shape), feature_blocks=",".join(FEATURE_BLOCKS))
"""
    ),
    md(
        r"""
## 12. Split por ator

O split é feito por ator para evitar vazamento de identidade. Tentamos estratificar por
`Race`; se isso falhar por grupos pequenos, usamos split aleatório por ator como fallback.
"""
    ),
    code(
        r"""
def make_actor_split(meta_df, seed=SEED):
    actors = meta_df[["ActorID", "Race", "Sex", "Ethnicity", "Age", "age_bin"]].drop_duplicates("ActorID").copy()
    known = actors[actors["Race"] != "Unknown"].copy()
    unknown = actors[actors["Race"] == "Unknown"].copy()
    try:
        train_actors, temp_actors = train_test_split(known, test_size=0.30, random_state=seed, stratify=known["Race"])
        val_actors, test_actors = train_test_split(temp_actors, test_size=0.50, random_state=seed, stratify=temp_actors["Race"])
    except ValueError as exc:
        print("Stratified split falhou; usando split aleatório por ator.")
        print(exc)
        train_actors, temp_actors = train_test_split(known, test_size=0.30, random_state=seed)
        val_actors, test_actors = train_test_split(temp_actors, test_size=0.50, random_state=seed)
    train_actors = pd.concat([train_actors, unknown], ignore_index=True)
    split_map = {int(a): "train" for a in train_actors["ActorID"]}
    split_map.update({int(a): "val" for a in val_actors["ActorID"]})
    split_map.update({int(a): "test" for a in test_actors["ActorID"]})
    return split_map

split_start = time.time()
split_map = make_actor_split(meta_valid)
meta_valid["split"] = meta_valid["ActorID"].map(split_map)
train_idx = np.where(meta_valid["split"].values == "train")[0]
val_idx = np.where(meta_valid["split"].values == "val")[0]
test_idx = np.where(meta_valid["split"].values == "test")[0]

print(f"Train: {len(train_idx):,} vídeos | atores: {meta_valid.loc[train_idx, 'ActorID'].nunique()}")
print(f"Val:   {len(val_idx):,} vídeos | atores: {meta_valid.loc[val_idx, 'ActorID'].nunique()}")
print(f"Test:  {len(test_idx):,} vídeos | atores: {meta_valid.loc[test_idx, 'ActorID'].nunique()}")
display(pd.crosstab(meta_valid["Race"], meta_valid["split"], margins=True))
display(pd.crosstab(meta_valid["label_name"], meta_valid["split"], margins=True))
log_event(
    "actor_holdout_split",
    start_time=split_start,
    n_train=int(len(train_idx)),
    n_val=int(len(val_idx)),
    n_test=int(len(test_idx)),
    train_actors=int(meta_valid.loc[train_idx, "ActorID"].nunique()),
    val_actors=int(meta_valid.loc[val_idx, "ActorID"].nunique()),
    test_actors=int(meta_valid.loc[test_idx, "ActorID"].nunique()),
)
"""
    ),
    md(
        r"""
## 13. Normalizacao e DataLoaders

O z-score e ajustado apenas no treino. Isso evita vazamento estatístico para validação e
teste.
"""
    ),
    code(
        r"""
normalization_start = time.time()
X_train_raw, y_train = X[train_idx], y[train_idx]
X_val_raw, y_val = X[val_idx], y[val_idx]
X_test_raw, y_test = X[test_idx], y[test_idx]

flat_train = X_train_raw.reshape(-1, D)
mu = flat_train.mean(axis=0).astype(np.float32)
sigma = flat_train.std(axis=0).astype(np.float32)
sigma[sigma == 0] = 1.0

X_train = ((X_train_raw - mu) / sigma).astype(np.float32)
X_val = ((X_val_raw - mu) / sigma).astype(np.float32)
X_test = ((X_test_raw - mu) / sigma).astype(np.float32)

meta_train = meta_valid.iloc[train_idx].reset_index(drop=True)
meta_val = meta_valid.iloc[val_idx].reset_index(drop=True)
meta_test = meta_valid.iloc[test_idx].reset_index(drop=True)
meta_valid.to_csv(TABLE_DIR / "manifest_valid_with_splits.csv", index=False)
np.savez_compressed(CACHE_DIR / "normalization_split_cache.npz", mu=mu, sigma=sigma, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

print(f"X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")
print(f"Media treino normalizado: {X_train.mean():.4f}")
print(f"Desvio treino normalizado: {X_train.std():.4f}")

def make_loader(X_np, y_np, batch_size=BATCH_SIZE, shuffle=False, sampler=None):
    ds = TensorDataset(torch.tensor(X_np, dtype=torch.float32), torch.tensor(y_np, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle if sampler is None else False, sampler=sampler, num_workers=0, pin_memory=torch.cuda.is_available())

def class_weights_from_y(y_np):
    counts = np.bincount(y_np, minlength=N_CLASSES).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)

def make_group_label_sampler(meta_df):
    keys = meta_df["label_code"].astype(str) + "__" + meta_df["Race"].astype(str)
    counts = keys.value_counts().to_dict()
    weights = keys.map(lambda k: 1.0 / counts[k]).values.astype(np.float64)
    weights = weights / weights.mean()
    return WeightedRandomSampler(torch.DoubleTensor(weights), num_samples=len(weights), replacement=True)

train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader = make_loader(X_val, y_val)
fair_train_loader = make_loader(X_train, y_train, sampler=make_group_label_sampler(meta_train))
criterion_weighted = nn.CrossEntropyLoss(weight=class_weights_from_y(y_train), label_smoothing=LABEL_SMOOTHING)
log_event(
    "normalization_and_dataloaders",
    start_time=normalization_start,
    x_train_shape=str(X_train.shape),
    x_val_shape=str(X_val.shape),
    x_test_shape=str(X_test.shape),
)
"""
    ),
    md(
        r"""
## 14. Modelos

Os três modelos usam a mesma entrada `(B, T, D)`. O modelo principal é uma adaptação
PyTorch do EmoT/AcT usado pelo Pedro: projeção linear dos landmarks, token de classe,
embedding posicional aprendível, encoder Transformer pós-normalização e cabeça MLP.

Mantemos CREMA-D + MediaPipe porque o objetivo agora é fairness étnico-racial; portanto,
a fidelidade aqui está na camada Transformer e nos metaparâmetros, não no dataset
RAVDESS/OpenFace original.
"""
    ),
    code(
        r"""
class FlatMLP(nn.Module):
    def __init__(self, T, D, n_classes, hidden=256, dropout=0.30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(T * D, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )
    def forward(self, x):
        return self.net(x)

class TemporalCNN1D(nn.Module):
    def __init__(self, T, D, n_classes, n_filters=96, dropout=0.30):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(D, n_filters, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(n_filters),
            nn.MaxPool1d(2),
            nn.Conv1d(n_filters, n_filters * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(n_filters * 2),
            nn.MaxPool1d(2),
            nn.Conv1d(n_filters * 2, n_filters * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_filters * 2, n_classes))
    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.classifier(self.features(x).squeeze(-1))

class PedroStyleTransformerBlockWithAttention(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_ff), nn.GELU(), nn.Linear(dim_ff, d_model))
        self.drop2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
    def forward(self, x, return_attention=False):
        attn_out, attn_w = self.attn(x, x, x, need_weights=return_attention, average_attn_weights=False)
        x = self.norm1(x + self.drop1(attn_out))
        x = self.norm2(x + self.drop2(self.ff(x)))
        return (x, attn_w) if return_attention else (x, None)

class EmoTransformer(nn.Module):
    def __init__(
        self,
        D,
        n_classes,
        d_model=192,
        n_heads=3,
        n_layers=6,
        dim_ff=768,
        dropout=0.30,
        mlp_head_size=256,
        max_len=100,
    ):
        super().__init__()
        self.input_proj = nn.Linear(D, d_model)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.empty(1, max_len + 1, d_model))
        self.blocks = nn.ModuleList([PedroStyleTransformerBlockWithAttention(d_model, n_heads, dim_ff, dropout) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(d_model, mlp_head_size), nn.Linear(mlp_head_size, n_classes))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.cls_token)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, x, return_attention=False):
        B = x.size(0)
        x = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : x.size(1), :]
        attentions = []
        for block in self.blocks:
            x, attn = block(x, return_attention=return_attention)
            if return_attention:
                attentions.append(attn)
        logits = self.head(x[:, 0])
        return (logits, attentions) if return_attention else logits

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
"""
    ),
    md(
        r"""
## 15. Métricas e fairness preditivo

Definimos desempenho global, desempenho por grupo, calibration/ECE, equal opportunity,
equalized odds e teste de permutação para o gap de acurácia.
"""
    ),
    code(
        r"""
def softmax_np(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)

def compute_basic_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

def multiclass_ece(y_true, y_prob, n_bins=10):
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    ok = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(y_true)) * abs(ok[mask].mean() - conf[mask].mean())
    return float(ece)

def metrics_by_group(y_true, y_pred, y_prob, groups, min_n=30):
    rows = []
    groups = np.array(groups)
    for group in pd.Series(groups).dropna().unique():
        mask = groups == group
        if mask.sum() < min_n:
            continue
        m = compute_basic_metrics(y_true[mask], y_pred[mask])
        m.update({"group": group, "n": int(mask.sum()), "ece": multiclass_ece(y_true[mask], y_prob[mask])})
        rows.append(m)
    return pd.DataFrame(rows).sort_values("group") if rows else pd.DataFrame()

def recall_by_class_group(y_true, y_pred, groups, min_n=10):
    rows = []
    groups = np.array(groups)
    for group in pd.Series(groups).dropna().unique():
        for c, name in enumerate(CLASS_NAMES):
            mask = (groups == group) & (y_true == c)
            if mask.sum() >= min_n:
                rows.append({"group": group, "class_name": name, "recall": float((y_pred[mask] == c).mean()), "n": int(mask.sum())})
    return pd.DataFrame(rows)

def fpr_by_class_group(y_true, y_pred, groups, min_n=10):
    rows = []
    groups = np.array(groups)
    for group in pd.Series(groups).dropna().unique():
        for c, name in enumerate(CLASS_NAMES):
            mask = (groups == group) & (y_true != c)
            if mask.sum() >= min_n:
                rows.append({"group": group, "class_name": name, "fpr": float((y_pred[mask] == c).mean()), "n": int(mask.sum())})
    return pd.DataFrame(rows)

def summarize_fairness(y_true, y_pred, y_prob, groups, protected_name, min_n=30):
    group_df = metrics_by_group(y_true, y_pred, y_prob, groups, min_n=min_n)
    if group_df.empty:
        return group_df, pd.DataFrame()
    rec = recall_by_class_group(y_true, y_pred, groups)
    fpr = fpr_by_class_group(y_true, y_pred, groups)
    if len(rec):
        eo = rec.groupby("class_name")["recall"].agg(lambda s: s.max() - s.min())
        equal_opp_mean, equal_opp_max = eo.mean(), eo.max()
    else:
        equal_opp_mean = equal_opp_max = np.nan
    if len(rec) and len(fpr):
        tpr_gap = rec.groupby("class_name")["recall"].agg(lambda s: s.max() - s.min())
        fpr_gap = fpr.groupby("class_name")["fpr"].agg(lambda s: s.max() - s.min())
        eod = pd.concat([tpr_gap.rename("tpr"), fpr_gap.rename("fpr")], axis=1).max(axis=1)
        eod_mean, eod_max = eod.mean(), eod.max()
    else:
        eod_mean = eod_max = np.nan
    summary = pd.DataFrame([{
        "protected_attribute": protected_name,
        "n_groups_used": len(group_df),
        "worst_group_accuracy": group_df["accuracy"].min(),
        "performance_gap_accuracy": group_df["accuracy"].max() - group_df["accuracy"].min(),
        "equal_opportunity_gap_mean": equal_opp_mean,
        "equal_opportunity_gap_max": equal_opp_max,
        "equalized_odds_gap_mean": eod_mean,
        "equalized_odds_gap_max": eod_max,
        "calibration_ece_gap": group_df["ece"].max() - group_df["ece"].min(),
    }])
    return group_df, summary

def bootstrap_metric_ci(y_true, y_pred, metric_fn, n_boot=500, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        vals.append(metric_fn(y_true[idx], y_pred[idx]))
    vals = np.array(vals)
    return {"mean": float(vals.mean()), "ci_low": float(np.percentile(vals, 2.5)), "ci_high": float(np.percentile(vals, 97.5))}

def permutation_gap_test(y_true, y_pred, groups, n_perm=500, min_n=30, seed=SEED):
    rng = np.random.default_rng(seed)
    groups = np.array(groups)
    def gap(g):
        vals = []
        for group in pd.Series(g).dropna().unique():
            mask = g == group
            if mask.sum() >= min_n:
                vals.append(accuracy_score(y_true[mask], y_pred[mask]))
        return np.nan if len(vals) < 2 else max(vals) - min(vals)
    obs = gap(groups)
    if np.isnan(obs):
        return {"observed_gap": np.nan, "p_value": np.nan}
    null = []
    for _ in range(n_perm):
        shuffled = groups.copy()
        rng.shuffle(shuffled)
        null.append(gap(shuffled))
    null = np.array(null)
    return {"observed_gap": float(obs), "p_value": float((np.sum(null >= obs) + 1) / (len(null) + 1))}
"""
    ),
    md(
        r"""
## 16. Treino e inferência

O loop usa mixed precision quando há GPU. Para aproximar o treino do Pedro, usamos AdamW,
label smoothing, clipping de gradiente e schedule Transformer com warmup. O early stopping
fica ativo por viabilidade no Colab; para uma réplica estrita do número de épocas, defina
`PATIENCE = None`.
"""
    ),
    code(
        r"""
def clip_gradients(model):
    if GRAD_CLIP_VALUE is not None:
        torch.nn.utils.clip_grad_value_(model.parameters(), GRAD_CLIP_VALUE)
    if GRAD_CLIP_NORM is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

def make_pedro_lr_scheduler(optimizer, total_steps):
    warmup_steps = max(1, int(WARMUP_PERC * total_steps))
    def lr_lambda(step):
        step = max(1, step)
        return (D_MODEL ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda), warmup_steps

def train_one_epoch(model, loader, optimizer, criterion, scaler=None, scheduler=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if USE_AMP:
            with torch.amp.autocast("cuda"):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_gradients(model)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            clip_gradients(model)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def predict_model(model, X_np, batch_size=BATCH_SIZE):
    model.eval()
    logits_all = []
    dummy = np.zeros(len(X_np), dtype=np.int64)
    for xb, _ in make_loader(X_np, dummy, batch_size=batch_size):
        xb = xb.to(DEVICE, non_blocking=True)
        logits_all.append(model(xb).detach().cpu().numpy())
    logits_all = np.concatenate(logits_all)
    prob = softmax_np(logits_all)
    return prob.argmax(axis=1), prob, logits_all

def evaluate_on_arrays(model, X_np, y_np):
    pred, prob, logits = predict_model(model, X_np)
    return compute_basic_metrics(y_np, pred), pred, prob, logits

def fit_model(model_name, model, loader, val_X, val_y, criterion, epochs=EPOCHS):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler, warmup_steps = (None, None)
    if USE_PEDRO_LR_SCHEDULE:
        total_steps = max(1, len(loader) * epochs)
        scheduler, warmup_steps = make_pedro_lr_scheduler(optimizer, total_steps)
    scaler = torch.amp.GradScaler("cuda") if USE_AMP else None
    best_f1, best_state, best_epoch = -1, None, -1
    history, patience_left = [], PATIENCE
    start = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, loader, optimizer, criterion, scaler, scheduler)
        val_metrics, _, _, _ = evaluate_on_arrays(model, val_X, val_y)
        current_lr = optimizer.param_groups[0]["lr"]
        row = {"model": model_name, "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "lr": current_lr, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(f"[{model_name}] epoch {epoch:04d}/{epochs} loss={train_loss:.4f} train_acc={train_acc:.3f} val_macro_f1={val_metrics['macro_f1']:.3f} lr={current_lr:.2e}")
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if PATIENCE is not None:
                patience_left = PATIENCE
        else:
            if PATIENCE is not None:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"Early stopping em epoch {epoch}.")
                    break
    model.load_state_dict(best_state)
    ckpt_path = CKPT_DIR / f"best_{model_name}.pt"
    torch.save({"model_state_dict": model.state_dict(), "best_epoch": best_epoch, "best_val_macro_f1": best_f1, "T": T, "D": D, "class_names": CLASS_NAMES, "mu": mu, "sigma": sigma}, ckpt_path)
    print(f"Melhor {model_name}: epoch={best_epoch}, val_macro_f1={best_f1:.4f}, tempo={(time.time()-start)/60:.1f} min")
    log_event(
        f"train_{model_name}",
        start_time=start,
        best_epoch=int(best_epoch),
        best_val_macro_f1=float(best_f1),
        checkpoint=str(ckpt_path),
        params=int(count_parameters(model)),
        warmup_steps=int(warmup_steps) if warmup_steps is not None else None,
        label_smoothing=float(LABEL_SMOOTHING),
        weight_decay=float(WEIGHT_DECAY),
    )
    return model, pd.DataFrame(history), ckpt_path
"""
    ),
    md(
        r"""
## 17. Treinamento dos modelos

Treinamos os baselines, o EmoT principal e uma variante mitigada com reweighting por
classe e `Race`.
"""
    ),
    code(
        r"""
models_to_train = {
    "FlatMLP": FlatMLP(T=T, D=D, n_classes=N_CLASSES),
    "TemporalCNN1D": TemporalCNN1D(T=T, D=D, n_classes=N_CLASSES),
    "EmoT": EmoTransformer(D=D, n_classes=N_CLASSES, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, dim_ff=DIM_FF, dropout=DROPOUT, mlp_head_size=MLP_HEAD_SIZE, max_len=T),
    "EmoT_Reweighted": EmoTransformer(D=D, n_classes=N_CLASSES, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, dim_ff=DIM_FF, dropout=DROPOUT, mlp_head_size=MLP_HEAD_SIZE, max_len=T),
}
trained_models, histories, ckpt_paths = {}, [], {}
training_all_start = time.time()
for name, model in models_to_train.items():
    print("\n" + "=" * 80)
    print(f"Treinando {name} | parametros: {count_parameters(model):,}")
    loader = fair_train_loader if name == "EmoT_Reweighted" else train_loader
    trained, hist, ckpt = fit_model(name, model, loader, X_val, y_val, criterion_weighted)
    trained_models[name] = trained
    histories.append(hist)
    ckpt_paths[name] = ckpt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

history_df = pd.concat(histories, ignore_index=True)
history_df.to_csv(TABLE_DIR / "training_history.csv", index=False)
history_df.to_csv(LOG_DIR / f"training_history_{RUN_ID}.csv", index=False)
display(history_df.tail())
log_event("training_all_models", start_time=training_all_start, n_models=int(len(models_to_train)))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.lineplot(data=history_df, x="epoch", y="train_loss", hue="model", ax=axes[0])
axes[0].set_title("Loss de treino")
sns.lineplot(data=history_df, x="epoch", y="val_macro_f1", hue="model", ax=axes[1])
axes[1].set_title("Macro-F1 de validação")
plt.tight_layout()
fig.savefig(FIG_DIR / "03_training_curves.png", bbox_inches="tight")
plt.show()
"""
    ),
    md(
        r"""
## 18. Avaliação preditiva global

Esta e a avaliação clássica: acurácia, balanced accuracy, macro-F1 e relatório por classe.
"""
    ),
    code(
        r"""
evaluation_start = time.time()
predictions, global_rows = {}, []
for name, model in trained_models.items():
    metrics, pred, prob, logits = evaluate_on_arrays(model, X_test, y_test)
    predictions[name] = {"pred": pred, "prob": prob, "logits": logits}
    ci = bootstrap_metric_ci(y_test, pred, lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0))
    global_rows.append({"model": name, **metrics, "macro_f1_ci_low": ci["ci_low"], "macro_f1_ci_high": ci["ci_high"], "n_test": len(y_test), "params": count_parameters(model)})
global_results = pd.DataFrame(global_rows).sort_values("macro_f1", ascending=False)
global_results.to_csv(TABLE_DIR / "global_test_metrics.csv", index=False)
global_results.to_csv(REPORT_DIR / "global_test_metrics.csv", index=False)
display(global_results)
print("Relatorio por classe do EmoT principal:")
print(classification_report(y_test, predictions["EmoT"]["pred"], target_names=CLASS_NAMES, zero_division=0))
log_event("global_predictive_evaluation", start_time=evaluation_start, n_models=int(len(trained_models)), n_test=int(len(y_test)))
"""
    ),
    md(
        r"""
## 19. Matrizes de confusão global e por Race

As matrizes por grupo atendem diretamente ao ajuste pós-qualificação: observar erros do
EmoT por grupos demográficos.
"""
    ),
    code(
        r"""
def plot_confusion(y_true, y_pred, title, path=None, normalize=True):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    cm_plot = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalize else cm
    fmt = ".2f" if normalize else "d"
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_plot, annot=True, fmt=fmt, cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight")
    plt.show()

emot_pred = predictions["EmoT"]["pred"]
confusion_start = time.time()
plot_confusion(y_test, emot_pred, "Matriz de confusao global - EmoT", FIG_DIR / "04_confusion_global_emot.png")

for race in meta_test["Race"].dropna().unique():
    mask = meta_test["Race"].values == race
    if mask.sum() < 30:
        print(f"Pulando Race={race}: n={mask.sum()} menor que 30")
        continue
    safe = re.sub(r"[^A-Za-z0-9]+", "_", str(race)).strip("_")
    plot_confusion(y_test[mask], emot_pred[mask], f"Matriz de confusao - EmoT - Race={race} (n={mask.sum()})", FIG_DIR / f"05_confusion_emot_race_{safe}.png")
log_event("confusion_matrices", start_time=confusion_start, n_race_groups=int(meta_test["Race"].nunique()))
"""
    ),
    md(
        r"""
## 20. Fairness preditivo

Calculamos métricas por grupo e gaps para `Race`, `Ethnicity`, `Sex` e faixa etária.
O foco interpretativo principal deve ser `Race`.
"""
    ),
    code(
        r"""
fairness_start = time.time()
fairness_group_tables, fairness_summary_tables = [], []
protected_attributes = ["Race", "Ethnicity", "Sex", "age_bin"]
for model_name, pred_obj in predictions.items():
    for attr in protected_attributes:
        group_df, summary_df = summarize_fairness(y_test, pred_obj["pred"], pred_obj["prob"], meta_test[attr].values, attr, min_n=30)
        if len(group_df):
            group_df.insert(0, "model", model_name)
            group_df.insert(1, "protected_attribute", attr)
            fairness_group_tables.append(group_df)
        if len(summary_df):
            summary_df.insert(0, "model", model_name)
            perm = permutation_gap_test(y_test, pred_obj["pred"], meta_test[attr].values, min_n=30)
            summary_df["permutation_accuracy_gap"] = perm["observed_gap"]
            summary_df["permutation_p_value"] = perm["p_value"]
            fairness_summary_tables.append(summary_df)

fairness_by_group = pd.concat(fairness_group_tables, ignore_index=True)
fairness_summary = pd.concat(fairness_summary_tables, ignore_index=True)
fairness_by_group.to_csv(TABLE_DIR / "fairness_metrics_by_group.csv", index=False)
fairness_summary.to_csv(TABLE_DIR / "fairness_summary.csv", index=False)
fairness_by_group.to_csv(FAIRNESS_DIR / "fairness_metrics_by_group.csv", index=False)
fairness_summary.to_csv(FAIRNESS_DIR / "fairness_summary.csv", index=False)
print(f"Tabelas de fairness salvas em: {FAIRNESS_DIR}")

display(fairness_by_group[fairness_by_group["protected_attribute"] == "Race"].sort_values(["model", "group"]))
display(fairness_summary.sort_values(["protected_attribute", "performance_gap_accuracy"]))

race_summary = fairness_summary[fairness_summary["protected_attribute"] == "Race"].copy()
plot_df = race_summary.melt(
    id_vars="model",
    value_vars=["performance_gap_accuracy", "equal_opportunity_gap_mean", "equalized_odds_gap_mean", "calibration_ece_gap"],
    var_name="metric",
    value_name="gap",
)
fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(data=plot_df, x="metric", y="gap", hue="model", ax=ax)
ax.set_title("Gaps de fairness por Race")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
fig.savefig(FIG_DIR / "06_fairness_gaps_by_race.png", bbox_inches="tight")
plt.show()
log_event(
    "predictive_fairness_metrics",
    start_time=fairness_start,
    n_group_rows=int(len(fairness_by_group)),
    n_summary_rows=int(len(fairness_summary)),
)
"""
    ),
    md(
        r"""
## 20.1 Fairness interseccional

A revisão bibliográfica destacou que analisar `Race`, `Sex`, `Ethnicity` e idade separadamente
pode esconder grupos mais vulneráveis. Aqui criamos grupos interseccionais, como
`Race x Sex` e `Race x faixa etária`, e calculamos as mesmas métricas quando há tamanho
amostral mínimo.

Esta análise é exploratória: grupos muito pequenos são removidos para evitar conclusões
estatisticamente frágeis.
"""
    ),
    code(
        r"""
intersectional_start = time.time()
INTERSECTIONAL_MIN_N = 20
intersection_specs = {
    "Race_x_Sex": ["Race", "Sex"],
    "Race_x_Ethnicity": ["Race", "Ethnicity"],
    "Race_x_age_bin": ["Race", "age_bin"],
    "Race_x_Sex_x_age_bin": ["Race", "Sex", "age_bin"],
}

meta_test_inter = meta_test.copy()
intersection_group_tables, intersection_summary_tables = [], []
for inter_name, cols in intersection_specs.items():
    meta_test_inter[inter_name] = meta_test_inter[cols].astype(str).agg(" | ".join, axis=1)
    group_counts = meta_test_inter[inter_name].value_counts()
    eligible_groups = set(group_counts[group_counts >= INTERSECTIONAL_MIN_N].index)
    eligible_mask = meta_test_inter[inter_name].isin(eligible_groups).values
    print(f"{inter_name}: {len(eligible_groups)} grupos com n >= {INTERSECTIONAL_MIN_N}")
    if not eligible_groups:
        continue
    for model_name, pred_obj in predictions.items():
        group_df, summary_df = summarize_fairness(
            y_test[eligible_mask],
            pred_obj["pred"][eligible_mask],
            pred_obj["prob"][eligible_mask],
            meta_test_inter.loc[eligible_mask, inter_name].values,
            inter_name,
            min_n=INTERSECTIONAL_MIN_N,
        )
        if len(group_df):
            group_df.insert(0, "model", model_name)
            group_df.insert(1, "intersectional_attribute", inter_name)
            intersection_group_tables.append(group_df)
        if len(summary_df):
            summary_df.insert(0, "model", model_name)
            summary_df.insert(1, "intersectional_attribute", inter_name)
            intersection_summary_tables.append(summary_df)

if intersection_group_tables:
    fairness_intersectional = pd.concat(intersection_group_tables, ignore_index=True)
    fairness_intersectional_summary = pd.concat(intersection_summary_tables, ignore_index=True)
    fairness_intersectional.to_csv(TABLE_DIR / "fairness_intersectional_metrics.csv", index=False)
    fairness_intersectional_summary.to_csv(TABLE_DIR / "fairness_intersectional_summary.csv", index=False)
    fairness_intersectional.to_csv(FAIRNESS_DIR / "fairness_intersectional_metrics.csv", index=False)
    fairness_intersectional_summary.to_csv(FAIRNESS_DIR / "fairness_intersectional_summary.csv", index=False)
    display(fairness_intersectional.sort_values(["model", "intersectional_attribute", "accuracy"]).head(30))
    display(fairness_intersectional_summary.sort_values(["model", "performance_gap_accuracy"], ascending=[True, False]))

    plot_inter = fairness_intersectional_summary[fairness_intersectional_summary["model"] == "EmoT"].copy()
    if len(plot_inter):
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.barplot(data=plot_inter, x="intersectional_attribute", y="performance_gap_accuracy", ax=ax, color="#E45756")
        ax.set_title("Gap de acurácia interseccional - EmoT")
        ax.set_xlabel("")
        ax.set_ylabel("accuracy gap")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        fig.savefig(FIG_DIR / "06b_fairness_intersectional_gaps_emot.png", bbox_inches="tight")
        plt.show()
else:
    fairness_intersectional = pd.DataFrame()
    fairness_intersectional_summary = pd.DataFrame()
    print("Nenhum grupo interseccional atingiu o tamanho mínimo.")

log_event(
    "intersectional_fairness",
    start_time=intersectional_start,
    n_rows=int(len(fairness_intersectional)),
    min_n=int(INTERSECTIONAL_MIN_N),
)
"""
    ),
    md(
        r"""
## 21. Robustez por alto consenso facial

O experimento principal usa o label original. Aqui avaliamos apenas exemplos com
`face_agreement >= 0.70` para verificar se desempenho e gaps permanecem quando a
ambiguidade perceptual é menor.
"""
    ),
    code(
        r"""
robustness_start = time.time()
high_mask = meta_test["face_agreement"].astype(float).values >= HIGH_CONSENSUS_THRESHOLD
print(f"Teste completo: {len(y_test)}")
print(f"Alto consenso facial: {high_mask.sum()} ({100*high_mask.mean():.1f}%)")
robust_rows = []
for model_name, obj in predictions.items():
    yt = y_test[high_mask]
    pred = obj["pred"][high_mask]
    prob = obj["prob"][high_mask]
    row = {"model": model_name, "subset": "high_face_consensus", "n": len(yt), **compute_basic_metrics(yt, pred)}
    _, fair_sum = summarize_fairness(yt, pred, prob, meta_test.loc[high_mask, "Race"].values, "Race", min_n=20)
    if len(fair_sum):
        row.update({f"race_{k}": v for k, v in fair_sum.iloc[0].to_dict().items() if k != "protected_attribute"})
    robust_rows.append(row)
for _, row in global_results.iterrows():
    robust_rows.append({"model": row["model"], "subset": "full_test", "n": int(row["n_test"]), "accuracy": row["accuracy"], "balanced_accuracy": row["balanced_accuracy"], "macro_f1": row["macro_f1"], "weighted_f1": row["weighted_f1"]})
robustness_df = pd.DataFrame(robust_rows)
robustness_df.to_csv(TABLE_DIR / "robustness_high_consensus.csv", index=False)
robustness_df.to_csv(FAIRNESS_DIR / "robustness_high_consensus.csv", index=False)
display(robustness_df.sort_values(["model", "subset"]))
log_event("high_consensus_robustness", start_time=robustness_start, n_high_consensus=int(high_mask.sum()), n_test=int(len(y_test)))
"""
    ),
    md(
        r"""
## 22. XAI - Attention Rollout temporal

O Attention Rollout acumula a atenção das camadas do EmoT. Extraímos a atenção do token
`CLS` para cada frame, produzindo uma curva temporal de importância.
"""
    ),
    code(
        r"""
def select_xai_indices(meta_df, max_samples=None, seed=SEED):
    if max_samples is None or max_samples >= len(meta_df):
        return np.arange(len(meta_df))
    rng = np.random.default_rng(seed)
    selected = []
    grouped = meta_df.groupby(["Race", "label_code"], dropna=False)
    per_group = max(1, max_samples // max(1, grouped.ngroups))
    for _, idx in grouped.indices.items():
        idx = np.array(list(idx))
        selected.extend(rng.choice(idx, size=min(len(idx), per_group), replace=False).tolist())
    if len(selected) < max_samples:
        remaining = np.setdiff1d(np.arange(len(meta_df)), np.array(selected))
        selected.extend(rng.choice(remaining, size=min(len(remaining), max_samples-len(selected)), replace=False).tolist())
    return np.array(sorted(selected[:max_samples]))

def attention_rollout(attentions):
    result = None
    for attn in attentions:
        attn_mean = attn.mean(dim=1)
        eye = torch.eye(attn_mean.size(-1), device=attn_mean.device).unsqueeze(0).expand_as(attn_mean)
        attn_res = 0.5 * attn_mean + 0.5 * eye
        attn_res = attn_res / attn_res.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        result = attn_res if result is None else torch.bmm(attn_res, result)
    return result

@torch.no_grad()
def compute_attention_importance(model, X_np, batch_size=XAI_BATCH_SIZE):
    model.eval()
    curves, preds, probs = [], [], []
    for start in tqdm(range(0, len(X_np), batch_size), desc="Attention Rollout"):
        xb = torch.tensor(X_np[start:start+batch_size], dtype=torch.float32, device=DEVICE)
        logits, attentions = model(xb, return_attention=True)
        prob = F.softmax(logits, dim=1)
        rollout = attention_rollout(attentions)
        cls_to_frames = rollout[:, 0, 1:]
        cls_to_frames = cls_to_frames / cls_to_frames.sum(dim=1, keepdim=True).clamp_min(1e-8)
        curves.append(cls_to_frames.detach().cpu().numpy())
        preds.append(prob.argmax(dim=1).detach().cpu().numpy())
        probs.append(prob.detach().cpu().numpy())
    return np.concatenate(curves), np.concatenate(preds), np.concatenate(probs)

xai_idx = select_xai_indices(meta_test, MAX_XAI_SAMPLES)
X_xai, y_xai = X_test[xai_idx], y_test[xai_idx]
meta_xai = meta_test.iloc[xai_idx].reset_index(drop=True)
emot_model = trained_models["EmoT"].to(DEVICE)
xai_attention_start = time.time()
attention_curves, xai_preds, xai_probs = compute_attention_importance(emot_model, X_xai)
print(f"Amostras XAI: {len(xai_idx)}")
print(f"attention_curves: {attention_curves.shape}")
meta_xai.to_csv(XAI_DIR / "xai_sample_manifest.csv", index=False)
np.savez_compressed(
    XAI_DIR / "xai_attention_rollout_emot.npz",
    attention_curves=attention_curves,
    xai_preds=xai_preds,
    xai_probs=xai_probs,
    xai_idx=xai_idx,
    y_xai=y_xai,
)
print(f"Artefatos de Attention Rollout salvos em: {XAI_DIR}")

def normalized_entropy(p, axis=1):
    p = p / np.maximum(p.sum(axis=axis, keepdims=True), 1e-12)
    ent = -(p * np.log(np.maximum(p, 1e-12))).sum(axis=axis)
    return ent / np.log(p.shape[axis])

meta_xai["attention_entropy"] = normalized_entropy(attention_curves, axis=1)
fig, ax = plt.subplots(figsize=(11, 5))
for race, group in meta_xai.groupby("Race"):
    if len(group) < 10:
        continue
    curve = attention_curves[group.index.values].mean(axis=0)
    ax.plot(np.arange(T_FRAMES), curve, marker="o", label=f"{race} (n={len(group)})")
ax.set_title("Attention Rollout temporal médio por Race - EmoT")
ax.set_xlabel("frame amostrado")
ax.set_ylabel("importância normalizada")
ax.legend()
plt.tight_layout()
fig.savefig(FIG_DIR / "07_xai_temporal_attention_by_race.png", bbox_inches="tight")
plt.show()
display(meta_xai.groupby("Race").agg(n=("file_stem", "count"), mean_entropy=("attention_entropy", "mean"), std_entropy=("attention_entropy", "std")).reset_index())
log_event("xai_attention_rollout", start_time=xai_attention_start, n_xai_samples=int(len(xai_idx)), attention_shape=str(attention_curves.shape))
"""
    ),
    md(
        r"""
## 23. XAI - Gradient x Input por landmark

Gradient x Input estima a contribuicao das features de entrada para o logit explicado.
Agregamos por landmark e por regiao facial para interpretar **onde** o EmoT decidiu.
"""
    ),
    code(
        r"""
def compute_grad_x_input(model, X_np, target_classes, batch_size=16):
    model.eval()
    all_attr = []
    for start in tqdm(range(0, len(X_np), batch_size), desc="Gradient x Input"):
        xb = torch.tensor(X_np[start:start+batch_size], dtype=torch.float32, device=DEVICE, requires_grad=True)
        target = torch.tensor(target_classes[start:start+batch_size], dtype=torch.long, device=DEVICE)
        logits = model(xb)
        selected = logits[torch.arange(xb.size(0), device=DEVICE), target].sum()
        model.zero_grad(set_to_none=True)
        selected.backward()
        all_attr.append((xb.grad * xb).abs().detach().cpu().numpy())
        del xb, logits, selected
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.concatenate(all_attr, axis=0)

def aggregate_attr_to_landmarks(attr):
    attr_mean = attr.mean(axis=1)
    n_blocks = len(FEATURE_BLOCKS)
    scores = np.zeros((attr.shape[0], N_LANDMARKS), dtype=np.float64)
    for b in range(n_blocks):
        block = attr_mean[:, b * BASE_FEATURE_DIM : (b + 1) * BASE_FEATURE_DIM]
        block = block.reshape(attr.shape[0], N_LANDMARKS, 2).mean(axis=2)
        scores += block
    return scores / np.maximum(scores.sum(axis=1, keepdims=True), 1e-12)

def aggregate_landmarks_to_regions(landmark_scores):
    selected_to_pos = {lm_id: pos for pos, lm_id in enumerate(SELECTED_LANDMARK_IDS)}
    rows = []
    for region, ids in REGION_IDS.items():
        positions = [selected_to_pos[i] for i in ids if i in selected_to_pos]
        if positions:
            values = landmark_scores[:, positions].sum(axis=1)
            rows.append(pd.DataFrame({"sample_idx": np.arange(len(values)), "region": region, "importance": values}))
    return pd.concat(rows, ignore_index=True)

grad_x_input_start = time.time()
grad_attr = compute_grad_x_input(emot_model, X_xai, xai_preds, batch_size=16)
landmark_scores = aggregate_attr_to_landmarks(grad_attr)
np.savez_compressed(
    XAI_DIR / "xai_gradient_landmark_scores_emot.npz",
    landmark_scores=landmark_scores,
    selected_landmark_ids=np.array(SELECTED_LANDMARK_IDS, dtype=np.int64),
)
region_importance = aggregate_landmarks_to_regions(landmark_scores)
region_importance = region_importance.merge(meta_xai[["Race", "Ethnicity", "Sex", "label_name"]].reset_index(names="sample_idx"), on="sample_idx", how="left")
region_summary = region_importance.groupby(["Race", "region"]).agg(n=("importance", "count"), mean_importance=("importance", "mean"), std_importance=("importance", "std")).reset_index()
region_summary.to_csv(TABLE_DIR / "xai_region_importance_by_race.csv", index=False)
region_summary.to_csv(XAI_DIR / "xai_region_importance_by_race.csv", index=False)
display(region_summary)

fig, ax = plt.subplots(figsize=(13, 6))
sns.barplot(data=region_summary, x="region", y="mean_importance", hue="Race", ax=ax)
ax.set_title("Gradient x Input: importância media por regiao facial e Race")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
fig.savefig(FIG_DIR / "08_xai_region_importance_by_race.png", bbox_inches="tight")
plt.show()
log_event("xai_gradient_x_input", start_time=grad_x_input_start, n_xai_samples=int(len(X_xai)), landmark_scores_shape=str(landmark_scores.shape))
"""
    ),
    md(
        r"""
## 23.1 Integrated Gradients e importância estático-dinâmica

A revisão bibliográfica sugere comparar explicações por técnica e verificar se o modelo
depende mais de landmarks estáticos (`pos`) ou de dinâmica expressiva (`vel`, `acc`, `jerk`).

Como o modelo usa landmarks, não usamos Grad-CAM como técnica principal. Grad-CAM é mais
natural para CNNs sobre imagem. Aqui, a alternativa mais coerente é Integrated Gradients
diretamente sobre as features temporais de landmarks.
"""
    ),
    code(
        r"""
def compute_integrated_gradients(model, X_np, target_classes, steps=IG_STEPS, batch_size=16):
    model.eval()
    attrs = []
    baseline_np = np.zeros_like(X_np, dtype=np.float32)
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=DEVICE)[1:]
    for start in tqdm(range(0, len(X_np), batch_size), desc="Integrated Gradients"):
        x = torch.tensor(X_np[start:start+batch_size], dtype=torch.float32, device=DEVICE)
        baseline = torch.tensor(baseline_np[start:start+batch_size], dtype=torch.float32, device=DEVICE)
        target = torch.tensor(target_classes[start:start+batch_size], dtype=torch.long, device=DEVICE)
        total_grad = torch.zeros_like(x)
        for alpha in alphas:
            x_step = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
            logits = model(x_step)
            selected = logits[torch.arange(x_step.size(0), device=DEVICE), target].sum()
            model.zero_grad(set_to_none=True)
            selected.backward()
            total_grad += x_step.grad.detach()
        avg_grad = total_grad / len(alphas)
        attr = ((x - baseline) * avg_grad).abs().detach().cpu().numpy()
        attrs.append(attr)
        del x, baseline, target, total_grad, avg_grad
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.concatenate(attrs, axis=0)

def feature_block_importance(attr):
    rows = []
    for block_idx, block_name in enumerate(FEATURE_BLOCKS):
        start_d = block_idx * BASE_FEATURE_DIM
        end_d = (block_idx + 1) * BASE_FEATURE_DIM
        values = attr[:, :, start_d:end_d].mean(axis=(1, 2))
        rows.append(pd.DataFrame({"sample_idx": np.arange(len(values)), "feature_block": block_name, "importance": values}))
    out = pd.concat(rows, ignore_index=True)
    denom = out.groupby("sample_idx")["importance"].transform("sum").replace(0, np.nan)
    out["importance_norm"] = out["importance"] / denom
    return out.fillna(0)

ig_start = time.time()
ig_n = min(len(X_xai), MAX_IG_SAMPLES)
ig_indices = select_xai_indices(meta_xai, ig_n, seed=SEED) if ig_n < len(X_xai) else np.arange(len(X_xai))
X_ig = X_xai[ig_indices]
y_ig = y_xai[ig_indices]
meta_ig = meta_xai.iloc[ig_indices].reset_index(drop=True)
xai_preds_ig = xai_preds[ig_indices]

ig_attr = compute_integrated_gradients(emot_model, X_ig, xai_preds_ig, steps=IG_STEPS, batch_size=16)
ig_landmark_scores = aggregate_attr_to_landmarks(ig_attr)
ig_region_importance = aggregate_landmarks_to_regions(ig_landmark_scores)
ig_region_importance = ig_region_importance.merge(meta_ig[["Race", "Ethnicity", "Sex", "label_name"]].reset_index(names="sample_idx"), on="sample_idx", how="left")
ig_region_summary = ig_region_importance.groupby(["Race", "region"]).agg(n=("importance", "count"), mean_importance=("importance", "mean"), std_importance=("importance", "std")).reset_index()
ig_region_summary.to_csv(TABLE_DIR / "xai_integrated_gradients_region_importance_by_race.csv", index=False)
ig_region_summary.to_csv(XAI_DIR / "xai_integrated_gradients_region_importance_by_race.csv", index=False)

block_importance_gxi = feature_block_importance(grad_attr).merge(meta_xai[["Race", "Ethnicity", "Sex", "label_name"]].reset_index(names="sample_idx"), on="sample_idx", how="left")
block_importance_ig = feature_block_importance(ig_attr).merge(meta_ig[["Race", "Ethnicity", "Sex", "label_name"]].reset_index(names="sample_idx"), on="sample_idx", how="left")
block_summary_gxi = block_importance_gxi.groupby(["Race", "feature_block"]).agg(n=("importance_norm", "count"), mean_importance_norm=("importance_norm", "mean")).reset_index()
block_summary_ig = block_importance_ig.groupby(["Race", "feature_block"]).agg(n=("importance_norm", "count"), mean_importance_norm=("importance_norm", "mean")).reset_index()
block_summary_gxi["method"] = "Gradient_x_Input"
block_summary_ig["method"] = "Integrated_Gradients"
block_summary = pd.concat([block_summary_gxi, block_summary_ig], ignore_index=True)
block_summary.to_csv(TABLE_DIR / "xai_feature_block_importance_by_race.csv", index=False)
block_summary.to_csv(XAI_DIR / "xai_feature_block_importance_by_race.csv", index=False)

np.savez_compressed(
    XAI_DIR / "xai_integrated_gradients_emot.npz",
    ig_landmark_scores=ig_landmark_scores,
    ig_indices=ig_indices,
    selected_landmark_ids=np.array(SELECTED_LANDMARK_IDS, dtype=np.int64),
)

display(ig_region_summary.head(20))
display(block_summary.head(30))

fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(data=block_summary[block_summary["method"] == "Integrated_Gradients"], x="feature_block", y="mean_importance_norm", hue="Race", ax=ax)
ax.set_title("Integrated Gradients: importância estática vs dinâmica por Race")
ax.set_ylabel("importância normalizada média")
plt.tight_layout()
fig.savefig(FIG_DIR / "08b_xai_static_dynamic_importance_by_race.png", bbox_inches="tight")
plt.show()

log_event("xai_integrated_gradients_static_dynamic", start_time=ig_start, n_ig_samples=int(len(X_ig)), ig_steps=int(IG_STEPS))
"""
    ),
    md(
        r"""
## 24. Fairness explicativo: ADI e Temporal Bias Drift

Agora avaliamos disparidade nas explicações:

- **ADI:** distância entre distribuições médias de importância dos landmarks.
- **Temporal Bias Drift:** distância entre curvas temporais médias de atenção.
"""
    ),
    code(
        r"""
def js_distance(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / np.maximum(p.sum(), 1e-12)
    q = q / np.maximum(q.sum(), 1e-12)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(np.maximum(p, 1e-12) / np.maximum(m, 1e-12)))
    kl_qm = np.sum(q * np.log(np.maximum(q, 1e-12) / np.maximum(m, 1e-12)))
    return float(np.sqrt(0.5 * (kl_pm + kl_qm)))

def compute_explanatory_fairness(meta_df, attention, landmark_scores, group_col="Race", reference_group="Caucasian", min_n=10):
    rows = []
    groups = meta_df[group_col].values
    ref_mask = groups == reference_group if reference_group in set(groups) else np.ones(len(groups), dtype=bool)
    ref_attr = landmark_scores[ref_mask].mean(axis=0)
    ref_time = attention[ref_mask].mean(axis=0)
    for group in pd.Series(groups).dropna().unique():
        mask = groups == group
        if mask.sum() < min_n:
            continue
        rows.append({
            "group_col": group_col,
            "group": group,
            "n": int(mask.sum()),
            "ADI_js_distance_to_reference": js_distance(landmark_scores[mask].mean(axis=0), ref_attr),
            "Temporal_Bias_Drift_js_distance_to_reference": js_distance(attention[mask].mean(axis=0), ref_time),
            "attention_entropy_mean": float(normalized_entropy(attention[mask], axis=1).mean()),
        })
    return pd.DataFrame(rows)

xai_fairness_start = time.time()
xai_fairness_race = compute_explanatory_fairness(meta_xai, attention_curves, landmark_scores, "Race", "Caucasian")
xai_fairness_ethnicity = compute_explanatory_fairness(meta_xai, attention_curves, landmark_scores, "Ethnicity", "Not Hispanic")
xai_fairness = pd.concat([xai_fairness_race, xai_fairness_ethnicity], ignore_index=True)
xai_fairness.to_csv(TABLE_DIR / "xai_explanatory_fairness_adi_tbd.csv", index=False)
xai_fairness.to_csv(XAI_DIR / "xai_explanatory_fairness_adi_tbd.csv", index=False)
display(xai_fairness)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.barplot(data=xai_fairness_race, x="group", y="ADI_js_distance_to_reference", ax=axes[0], color="#4C78A8")
axes[0].set_title("ADI por Race")
axes[0].tick_params(axis="x", rotation=20)
sns.barplot(data=xai_fairness_race, x="group", y="Temporal_Bias_Drift_js_distance_to_reference", ax=axes[1], color="#F58518")
axes[1].set_title("Temporal Bias Drift por Race")
axes[1].tick_params(axis="x", rotation=20)
plt.tight_layout()
fig.savefig(FIG_DIR / "09_xai_adi_temporal_bias_drift_by_race.png", bbox_inches="tight")
plt.show()
log_event("xai_explanatory_fairness", start_time=xai_fairness_start, n_rows=int(len(xai_fairness)))
"""
    ),
    md(
        r"""
## 25. Top-k landmarks por grupo

O overlap de top-k landmarks verifica se os grupos compartilham os mesmos pontos faciais
mais importantes.
"""
    ),
    code(
        r"""
def top_landmarks_for_group(meta_df, scores, group_col, group, k=TOP_K_LANDMARKS):
    mask = meta_df[group_col].values == group
    mean_scores = scores[mask].mean(axis=0)
    top_pos = np.argsort(mean_scores)[::-1][:k]
    return set([SELECTED_LANDMARK_IDS[i] for i in top_pos]), mean_scores

topk_start = time.time()
races = [r for r in meta_xai["Race"].dropna().unique() if (meta_xai["Race"] == r).sum() >= 10]
rows = []
for i, g1 in enumerate(races):
    set1, _ = top_landmarks_for_group(meta_xai, landmark_scores, "Race", g1)
    for g2 in races[i+1:]:
        set2, _ = top_landmarks_for_group(meta_xai, landmark_scores, "Race", g2)
        rows.append({"group_1": g1, "group_2": g2, "top_k": TOP_K_LANDMARKS, "jaccard_overlap": len(set1 & set2) / max(1, len(set1 | set2))})
topk_overlap = pd.DataFrame(rows)
topk_overlap.to_csv(TABLE_DIR / "xai_topk_landmark_overlap_by_race.csv", index=False)
topk_overlap.to_csv(XAI_DIR / "xai_topk_landmark_overlap_by_race.csv", index=False)
display(topk_overlap)

top_rows = []
for race in races:
    _, mean_scores = top_landmarks_for_group(meta_xai, landmark_scores, "Race", race)
    for rank, pos in enumerate(np.argsort(mean_scores)[::-1][:TOP_K_LANDMARKS], start=1):
        lm_id = SELECTED_LANDMARK_IDS[pos]
        top_rows.append({"Race": race, "rank": rank, "landmark_id": lm_id, "importance": float(mean_scores[pos]), "regions": ",".join(LANDMARK_TO_REGIONS.get(lm_id, ["other"]))})
top_landmarks_df = pd.DataFrame(top_rows)
top_landmarks_df.to_csv(TABLE_DIR / "xai_top_landmarks_by_race.csv", index=False)
top_landmarks_df.to_csv(XAI_DIR / "xai_top_landmarks_by_race.csv", index=False)
display(top_landmarks_df.head(30))
log_event("xai_topk_landmarks", start_time=topk_start, n_pairs=int(len(topk_overlap)), n_top_rows=int(len(top_landmarks_df)))
"""
    ),
    md(
        r"""
## 25.1 Oclusão por regiões faciais

A revisão propõe testar se a remoção de regiões como boca, olhos, sobrancelhas e região
inferior/superior da face degrada o desempenho de modo diferente entre grupos. Esta etapa
faz uma ablação direta nas features de landmarks: zeramos os landmarks de uma região e
medimos queda de probabilidade e acurácia.
"""
    ),
    code(
        r"""
def feature_indices_for_landmarks(landmark_ids):
    selected_to_pos = {lm_id: pos for pos, lm_id in enumerate(SELECTED_LANDMARK_IDS)}
    positions = [selected_to_pos[i] for i in landmark_ids if i in selected_to_pos]
    feature_idx = []
    for block_idx in range(len(FEATURE_BLOCKS)):
        offset = block_idx * BASE_FEATURE_DIM
        for pos in positions:
            feature_idx.extend([offset + 2 * pos, offset + 2 * pos + 1])
    return sorted(set(feature_idx))

OCCLUSION_REGION_IDS = {
    **REGION_IDS,
    "upper_face": set(REGION_IDS["left_eye"]) | set(REGION_IDS["right_eye"]) | set(REGION_IDS["left_eyebrow"]) | set(REGION_IDS["right_eyebrow"]),
    "lower_face": set(REGION_IDS["lips"]),
    "expressive_core": set(REGION_IDS["lips"]) | set(REGION_IDS["left_eye"]) | set(REGION_IDS["right_eye"]) | set(REGION_IDS["left_eyebrow"]) | set(REGION_IDS["right_eyebrow"]),
}

def region_occlusion_test(model, X_np, y_true, meta_df, target_classes, regions=OCCLUSION_REGION_IDS):
    base_pred, base_prob, _ = predict_model(model, X_np, batch_size=REGION_OCCLUSION_BATCH_SIZE)
    base_target_prob = base_prob[np.arange(len(X_np)), target_classes]
    rows = []
    for region, ids in tqdm(regions.items(), desc="Oclusão por região"):
        feature_idx = feature_indices_for_landmarks(ids)
        if not feature_idx:
            continue
        X_masked = X_np.copy()
        X_masked[:, :, feature_idx] = 0.0
        masked_pred, masked_prob, _ = predict_model(model, X_masked, batch_size=REGION_OCCLUSION_BATCH_SIZE)
        masked_target_prob = masked_prob[np.arange(len(X_np)), target_classes]
        for i in range(len(X_np)):
            rows.append({
                "sample_idx": i,
                "region": region,
                "n_landmarks": len(set(ids).intersection(SELECTED_LANDMARK_IDS)),
                "base_pred": int(base_pred[i]),
                "masked_pred": int(masked_pred[i]),
                "true_label": int(y_true[i]),
                "base_correct": bool(base_pred[i] == y_true[i]),
                "masked_correct": bool(masked_pred[i] == y_true[i]),
                "base_target_prob": float(base_target_prob[i]),
                "masked_target_prob": float(masked_target_prob[i]),
                "target_prob_drop": float(base_target_prob[i] - masked_target_prob[i]),
            })
    return pd.DataFrame(rows)

occlusion_start = time.time()
occlusion_df = region_occlusion_test(emot_model, X_xai, y_xai, meta_xai, xai_preds)
occlusion_df = occlusion_df.merge(meta_xai[["Race", "Ethnicity", "Sex", "label_name"]].reset_index(names="sample_idx"), on="sample_idx", how="left")
occlusion_summary = occlusion_df.groupby(["Race", "region"]).agg(
    n=("sample_idx", "nunique"),
    n_landmarks=("n_landmarks", "first"),
    mean_target_prob_drop=("target_prob_drop", "mean"),
    base_accuracy=("base_correct", "mean"),
    masked_accuracy=("masked_correct", "mean"),
).reset_index()
occlusion_summary["accuracy_drop"] = occlusion_summary["base_accuracy"] - occlusion_summary["masked_accuracy"]

occlusion_df.to_csv(TABLE_DIR / "xai_region_occlusion_samples.csv", index=False)
occlusion_summary.to_csv(TABLE_DIR / "xai_region_occlusion_by_race.csv", index=False)
occlusion_df.to_csv(XAI_DIR / "xai_region_occlusion_samples.csv", index=False)
occlusion_summary.to_csv(XAI_DIR / "xai_region_occlusion_by_race.csv", index=False)

display(occlusion_summary.sort_values(["region", "accuracy_drop"], ascending=[True, False]).head(40))

fig, ax = plt.subplots(figsize=(13, 6))
sns.barplot(data=occlusion_summary, x="region", y="accuracy_drop", hue="Race", ax=ax)
ax.set_title("Oclusão por região facial: queda de acurácia por Race")
ax.set_ylabel("accuracy drop")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
fig.savefig(FIG_DIR / "10a_xai_region_occlusion_accuracy_drop_by_race.png", bbox_inches="tight")
plt.show()

log_event("xai_region_occlusion", start_time=occlusion_start, n_rows=int(len(occlusion_df)), n_regions=int(occlusion_summary["region"].nunique()))
"""
    ),
    md(
        r"""
## 26. Deletion Test temporal

Removemos progressivamente os frames mais importantes segundo Attention Rollout. Se a
explicacao for fiel, a probabilidade da classe explicada deve cair rapidamente.
"""
    ),
    code(
        r"""
@torch.no_grad()
def deletion_test_temporal(model, X_np, frame_importance, target_classes, fractions=DELETION_FRACTIONS, batch_size=XAI_BATCH_SIZE):
    _, prob_original, _ = predict_model(model, X_np, batch_size=batch_size)
    base_prob = prob_original[np.arange(len(X_np)), target_classes]
    rows = []
    for frac in tqdm(fractions, desc="Deletion fractions"):
        X_masked = X_np.copy()
        k = int(round(frac * X_np.shape[1]))
        if k > 0:
            order = np.argsort(frame_importance, axis=1)[:, ::-1]
            for i in range(len(X_masked)):
                X_masked[i, order[i, :k], :] = 0.0
        _, prob_masked, _ = predict_model(model, X_masked, batch_size=batch_size)
        target_prob = prob_masked[np.arange(len(X_np)), target_classes]
        for i in range(len(X_np)):
            rows.append({"sample_idx": i, "fraction_deleted": frac, "k_frames_deleted": k, "base_prob": float(base_prob[i]), "masked_prob": float(target_prob[i]), "prob_drop": float(base_prob[i] - target_prob[i])})
    return pd.DataFrame(rows)

deletion_start = time.time()
deletion_df = deletion_test_temporal(emot_model, X_xai, attention_curves, xai_preds)
deletion_df = deletion_df.merge(meta_xai[["Race", "Ethnicity", "Sex", "label_name"]].reset_index(names="sample_idx"), on="sample_idx", how="left")
deletion_df.to_csv(TABLE_DIR / "xai_deletion_test_temporal.csv", index=False)
deletion_df.to_csv(XAI_DIR / "xai_deletion_test_temporal.csv", index=False)
display(deletion_df.head())

def auc_deletion_for_group(df):
    curve = df.groupby("fraction_deleted")["masked_prob"].mean().reset_index().sort_values("fraction_deleted")
    return float(np.trapz(curve["masked_prob"].values, curve["fraction_deleted"].values))

auc_rows = []
for race, g in deletion_df.groupby("Race"):
    if g["sample_idx"].nunique() >= 10:
        auc_rows.append({"Race": race, "n_samples": g["sample_idx"].nunique(), "deletion_auc_masked_prob": auc_deletion_for_group(g), "mean_aopc_drop": g.groupby("sample_idx")["prob_drop"].mean().mean()})
deletion_auc_by_race = pd.DataFrame(auc_rows)
deletion_auc_by_race.to_csv(TABLE_DIR / "xai_deletion_auc_by_race.csv", index=False)
deletion_auc_by_race.to_csv(XAI_DIR / "xai_deletion_auc_by_race.csv", index=False)
display(deletion_auc_by_race)

delete_summary = deletion_df.groupby(["Race", "fraction_deleted"]).agg(mean_masked_prob=("masked_prob", "mean"), n=("masked_prob", "count")).reset_index()
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=delete_summary, x="fraction_deleted", y="mean_masked_prob", hue="Race", marker="o", ax=ax)
ax.set_title("Deletion Test temporal por Race")
ax.set_xlabel("fração de frames removidos")
ax.set_ylabel("probabilidade media da classe explicada")
plt.tight_layout()
fig.savefig(FIG_DIR / "10_xai_deletion_curves_by_race.png", bbox_inches="tight")
plt.show()
log_event("xai_deletion_test", start_time=deletion_start, n_rows=int(len(deletion_df)), n_auc_rows=int(len(deletion_auc_by_race)))
"""
    ),
    md(
        r"""
## 26.1 Correlação exploratória entre XAI e fairness

A revisão bibliográfica sugere testar se indicadores de explicabilidade ajudam a antecipar
ou explicar disparidades. Aqui agregamos, por `Race`, métricas preditivas e métricas XAI.

Importante: com poucos grupos demográficos, esta análise é **exploratória**. Correlação não
prova causalidade; ela apenas orienta interpretação e hipóteses para o texto.
"""
    ),
    code(
        r"""
correlation_start = time.time()

emot_race_metrics = fairness_by_group[
    (fairness_by_group["model"] == "EmoT") & (fairness_by_group["protected_attribute"] == "Race")
].copy()
emot_race_metrics = emot_race_metrics.rename(columns={"group": "Race"})

attention_group = meta_xai.groupby("Race").agg(
    xai_attention_entropy_mean=("attention_entropy", "mean"),
    xai_attention_entropy_std=("attention_entropy", "std"),
    xai_n=("file_stem", "count"),
).reset_index()

adi_group = xai_fairness_race.rename(columns={
    "group": "Race",
    "ADI_js_distance_to_reference": "xai_ADI",
    "Temporal_Bias_Drift_js_distance_to_reference": "xai_TBD",
    "attention_entropy_mean": "xai_attention_entropy_group_mean",
})

deletion_group = deletion_auc_by_race.rename(columns={
    "deletion_auc_masked_prob": "xai_deletion_auc",
    "mean_aopc_drop": "xai_mean_aopc_drop",
})

lower_occ = occlusion_summary[occlusion_summary["region"] == "lower_face"][
    ["Race", "accuracy_drop", "mean_target_prob_drop"]
].rename(columns={
    "accuracy_drop": "xai_lower_face_accuracy_drop",
    "mean_target_prob_drop": "xai_lower_face_prob_drop",
})

block_ig = block_summary[block_summary["method"] == "Integrated_Gradients"].pivot_table(
    index="Race",
    columns="feature_block",
    values="mean_importance_norm",
    aggfunc="mean",
).reset_index()
for col in FEATURE_BLOCKS:
    if col not in block_ig.columns:
        block_ig[col] = 0.0
dynamic_cols = [c for c in ["vel", "acc", "jerk"] if c in block_ig.columns]
block_ig["xai_static_pos_importance"] = block_ig.get("pos", 0.0)
block_ig["xai_dynamic_importance"] = block_ig[dynamic_cols].sum(axis=1) if dynamic_cols else 0.0
block_ig = block_ig[["Race", "xai_static_pos_importance", "xai_dynamic_importance"]]

xai_fairness_group_profile = emot_race_metrics.merge(attention_group, on="Race", how="left")
xai_fairness_group_profile = xai_fairness_group_profile.merge(adi_group[["Race", "xai_ADI", "xai_TBD"]], on="Race", how="left")
xai_fairness_group_profile = xai_fairness_group_profile.merge(deletion_group[["Race", "xai_deletion_auc", "xai_mean_aopc_drop"]], on="Race", how="left")
xai_fairness_group_profile = xai_fairness_group_profile.merge(lower_occ, on="Race", how="left")
xai_fairness_group_profile = xai_fairness_group_profile.merge(block_ig, on="Race", how="left")
xai_fairness_group_profile.to_csv(TABLE_DIR / "xai_fairness_group_profile_by_race.csv", index=False)
xai_fairness_group_profile.to_csv(XAI_DIR / "xai_fairness_group_profile_by_race.csv", index=False)
display(xai_fairness_group_profile)

fairness_metrics_for_corr = [c for c in ["accuracy", "macro_f1", "ece"] if c in xai_fairness_group_profile.columns]
xai_metrics_for_corr = [
    "xai_attention_entropy_mean",
    "xai_ADI",
    "xai_TBD",
    "xai_deletion_auc",
    "xai_mean_aopc_drop",
    "xai_lower_face_accuracy_drop",
    "xai_lower_face_prob_drop",
    "xai_static_pos_importance",
    "xai_dynamic_importance",
]
xai_metrics_for_corr = [c for c in xai_metrics_for_corr if c in xai_fairness_group_profile.columns]

try:
    from scipy.stats import pearsonr, spearmanr
except Exception:
    pearsonr = spearmanr = None

corr_rows = []
for fair_col in fairness_metrics_for_corr:
    for xai_col in xai_metrics_for_corr:
        tmp = xai_fairness_group_profile[[fair_col, xai_col]].dropna()
        row = {"fairness_metric": fair_col, "xai_metric": xai_col, "n_groups": int(len(tmp))}
        if len(tmp) >= 3 and tmp[fair_col].nunique() > 1 and tmp[xai_col].nunique() > 1:
            if pearsonr is not None:
                pr = pearsonr(tmp[fair_col], tmp[xai_col])
                sr = spearmanr(tmp[fair_col], tmp[xai_col])
                row.update({
                    "pearson_r": float(pr.statistic),
                    "pearson_p": float(pr.pvalue),
                    "spearman_rho": float(sr.statistic),
                    "spearman_p": float(sr.pvalue),
                })
            else:
                row.update({
                    "pearson_r": float(tmp[fair_col].corr(tmp[xai_col], method="pearson")),
                    "pearson_p": np.nan,
                    "spearman_rho": float(tmp[fair_col].corr(tmp[xai_col], method="spearman")),
                    "spearman_p": np.nan,
                })
        else:
            row.update({"pearson_r": np.nan, "pearson_p": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan})
        corr_rows.append(row)

xai_fairness_correlation = pd.DataFrame(corr_rows)
xai_fairness_correlation.to_csv(TABLE_DIR / "xai_fairness_correlation_by_race.csv", index=False)
xai_fairness_correlation.to_csv(XAI_DIR / "xai_fairness_correlation_by_race.csv", index=False)
display(xai_fairness_correlation.sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False).head(20))

if "ece" in xai_fairness_group_profile.columns and "xai_attention_entropy_mean" in xai_fairness_group_profile.columns:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=xai_fairness_group_profile, x="xai_attention_entropy_mean", y="ece", hue="Race", s=90, ax=ax)
    for _, row in xai_fairness_group_profile.iterrows():
        ax.text(row["xai_attention_entropy_mean"], row["ece"], str(row["Race"]), fontsize=8)
    ax.set_title("Correlação exploratória: entropia de atenção vs ECE")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "10b_xai_fairness_correlation_entropy_ece.png", bbox_inches="tight")
    plt.show()

log_event("xai_fairness_correlation", start_time=correlation_start, n_rows=int(len(xai_fairness_correlation)))
"""
    ),
    md(
        r"""
## 27. Robustez XAI em alto consenso facial

Repetimos ADI e Temporal Bias Drift no subconjunto `face_agreement >= 0.70`.
"""
    ),
    code(
        r"""
xai_robustness_start = time.time()
xai_high_mask = meta_xai["face_agreement"].astype(float).values >= HIGH_CONSENSUS_THRESHOLD
print(f"Amostras XAI totais: {len(meta_xai)}")
print(f"Amostras XAI de alto consenso: {xai_high_mask.sum()}")
if xai_high_mask.sum() >= 30:
    xai_fair_high = compute_explanatory_fairness(
        meta_xai.loc[xai_high_mask].reset_index(drop=True),
        attention_curves[xai_high_mask],
        landmark_scores[xai_high_mask],
        "Race",
        "Caucasian",
        min_n=5,
    )
    xai_fair_high["subset"] = "high_face_consensus"
    xai_fair_full = xai_fairness_race.copy()
    xai_fair_full["subset"] = "full_xai_test"
    xai_robustness = pd.concat([xai_fair_full, xai_fair_high], ignore_index=True)
    xai_robustness.to_csv(TABLE_DIR / "xai_robustness_high_consensus.csv", index=False)
    xai_robustness.to_csv(XAI_DIR / "xai_robustness_high_consensus.csv", index=False)
    display(xai_robustness)
else:
    print("Subconjunto pequeno demais para robustez XAI.")
log_event("xai_high_consensus_robustness", start_time=xai_robustness_start, n_high_consensus=int(xai_high_mask.sum()), n_xai=int(len(meta_xai)))
"""
    ),
    md(
        r"""
## 28. Síntese automática

Esta célula grava um resumo curto para apoiar a escrita da dissertação/artigo.
"""
    ),
    code(
        r"""
best_model_row = global_results.iloc[0].to_dict()
emot_global = global_results[global_results["model"] == "EmoT"].iloc[0].to_dict()
emot_race_fair = fairness_summary[(fairness_summary["model"] == "EmoT") & (fairness_summary["protected_attribute"] == "Race")]

lines = [
    f"Melhor modelo por macro-F1: {best_model_row['model']} (macro-F1={best_model_row['macro_f1']:.3f}).",
    f"EmoT: accuracy={emot_global['accuracy']:.3f}, balanced_accuracy={emot_global['balanced_accuracy']:.3f}, macro-F1={emot_global['macro_f1']:.3f}.",
]
if len(emot_race_fair):
    row = emot_race_fair.iloc[0]
    lines.append(
        f"Fairness preditivo do EmoT por Race: worst-group accuracy={row['worst_group_accuracy']:.3f}, "
        f"performance gap={row['performance_gap_accuracy']:.3f}, equal opportunity gap médio={row['equal_opportunity_gap_mean']:.3f}, "
        f"p_perm={row['permutation_p_value']:.3f}."
    )
if len(xai_fairness_race):
    lines.append(
        f"Fairness explicativo por Race: ADI maximo={xai_fairness_race['ADI_js_distance_to_reference'].max():.3f}, "
        f"Temporal Bias Drift maximo={xai_fairness_race['Temporal_Bias_Drift_js_distance_to_reference'].max():.3f}."
    )

summary_text = "\n".join(f"- {line}" for line in lines)
print(summary_text)
summary_path = REPORT_DIR / "summary_results.md"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("# Síntese dos resultados\n\n")
    f.write(summary_text + "\n")
print(f"Síntese salva em: {summary_path}")
"""
    ),
    md(
        r"""
## 29. Logs de tempo, GPU e processamento

Esta célula consolida a rastreabilidade computacional: quanto tempo cada etapa levou,
quanto de GPU/VRAM estava sendo usado e quais arquivos de log foram salvos.
"""
    ),
    code(
        r"""
show_gpu_snapshot("fim_experimento")

resource_log = pd.DataFrame(RUN_EVENTS)
resource_log_path = LOG_DIR / f"resource_usage_log_{RUN_ID}.csv"
resource_log_latest = LOG_DIR / "resource_usage_log_latest.csv"
resource_log.to_csv(resource_log_path, index=False)
resource_log.to_csv(resource_log_latest, index=False)

timing_summary = resource_log.dropna(subset=["duration_sec"]).copy() if "duration_sec" in resource_log.columns else pd.DataFrame()
if len(timing_summary):
    timing_summary = timing_summary.sort_values("duration_sec", ascending=False)
    timing_summary.to_csv(TABLE_DIR / "processing_time_by_stage.csv", index=False)
    timing_summary.to_csv(LOG_DIR / "processing_time_by_stage.csv", index=False)
    display_cols = [
        "stage",
        "status",
        "duration_sec",
        "duration_min",
        "gpu_name",
        "nvidia_gpu_utilization_percent",
        "nvidia_memory_used_mb",
        "gpu_memory_allocated_mb",
        "gpu_memory_reserved_mb",
    ]
    display_cols = [c for c in display_cols if c in timing_summary.columns]
    display(timing_summary[display_cols].head(30))

    fig, ax = plt.subplots(figsize=(12, max(5, 0.35 * min(25, len(timing_summary)))))
    plot_time = timing_summary.head(25).sort_values("duration_sec")
    sns.barplot(data=plot_time, x="duration_sec", y="stage", ax=ax, color="#4C78A8")
    ax.set_title("Tempo de processamento por etapa")
    ax.set_xlabel("segundos")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "11_processing_time_by_stage.png", bbox_inches="tight")
    plt.show()
else:
    print("Nenhuma etapa com duração registrada.")

gpu_cols = ["nvidia_gpu_utilization_percent", "nvidia_memory_used_mb", "nvidia_power_draw_w", "nvidia_temperature_c"]
available_gpu_cols = [c for c in gpu_cols if c in resource_log.columns and resource_log[c].notna().any()]
if available_gpu_cols:
    gpu_timeline = resource_log[["stage"] + available_gpu_cols].copy()
    gpu_timeline.to_csv(LOG_DIR / "gpu_usage_timeline.csv", index=False)
    display(gpu_timeline.tail(20))

    fig, axes = plt.subplots(len(available_gpu_cols), 1, figsize=(13, 2.8 * len(available_gpu_cols)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, available_gpu_cols):
        ax.plot(np.arange(len(resource_log)), resource_log[col], marker="o")
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("evento registrado")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "12_gpu_usage_timeline.png", bbox_inches="tight")
    plt.show()
else:
    print("Sem métricas detalhadas do nvidia-smi. O log ainda contém memória PyTorch quando CUDA está disponível.")

print(f"Log de recursos salvo em: {resource_log_path}")
print(f"Resumo de tempos salvo em: {TABLE_DIR / 'processing_time_by_stage.csv'}")
"""
    ),
    md(
        r"""
## 30. Artefatos gerados

Principais saídas:

- `outputs/tables/global_test_metrics.csv`
- `outputs/tables/fairness_metrics_by_group.csv`
- `outputs/tables/fairness_summary.csv`
- `outputs/tables/fairness_intersectional_metrics.csv`
- `outputs/tables/robustness_high_consensus.csv`
- `outputs/tables/xai_explanatory_fairness_adi_tbd.csv`
- `outputs/tables/xai_deletion_test_temporal.csv`
- `outputs/tables/xai_integrated_gradients_region_importance_by_race.csv`
- `outputs/tables/xai_feature_block_importance_by_race.csv`
- `outputs/tables/xai_region_occlusion_by_race.csv`
- `outputs/tables/xai_fairness_correlation_by_race.csv`
- `outputs/tables/visual_sample_manifest.csv`
- `outputs/tables/processing_time_by_stage.csv`
- `outputs/figures/*.png`
- `outputs/figures/02_visual_sample_frames_by_race.png`
- `outputs/figures/02_visual_preprocessing_pipeline.png`
- `outputs/figures/06b_fairness_intersectional_gaps_emot.png`
- `outputs/figures/08b_xai_static_dynamic_importance_by_race.png`
- `outputs/figures/10a_xai_region_occlusion_accuracy_drop_by_race.png`
- `outputs/figures/10b_xai_fairness_correlation_entropy_ece.png`
- `outputs/figures/11_processing_time_by_stage.png`
- `outputs/figures/12_gpu_usage_timeline.png`
- `outputs/models/best_*.pt`
- `outputs/fairness/*.csv`
- `outputs/xai/*.csv` e `outputs/xai/*.npz`
- `outputs/logs/*.json` e `outputs/logs/*.csv`
- `outputs/reports/summary_results.md`

Leitura recomendada para a dissertação:

1. estatistica descritiva do CREMA-D;
2. qualidade de processamento facial por grupo;
3. comparação dos baselines e EmoT;
4. fairness preditivo por `Race`;
5. XAI temporal e por landmarks;
6. fairness explicativo;
7. robustez com alto consenso facial.
"""
    ),
    md(
        r"""
## 31. Download de todos os artefatos

Esta célula compacta os artefatos gerados e inicia o download no Colab.

Por padrão, o `.zip` inclui:

- `outputs/`: tabelas, figuras, modelos, logs, relatórios, fairness e XAI;
- `cache/`: tensores processados, landmarks extraídos, split/normalização e modelo `.task` do MediaPipe, quando existir.

Os vídeos brutos ficam fora por padrão, porque podem deixar o arquivo muito grande e podem ser baixados novamente do CREMA-D. Se quiser incluir também os vídeos, altere `INCLUDE_RAW_VIDEOS_IN_ZIP = True`.
"""
    ),
    code(
        r"""
import zipfile
from datetime import datetime

INCLUDE_OUTPUTS_IN_ZIP = True
INCLUDE_CACHE_IN_ZIP = True
INCLUDE_RAW_VIDEOS_IN_ZIP = False

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
artifact_zip = ZIP_DIR / f"cremad_fairness_xai_artifacts_{timestamp}.zip"

def should_skip_artifact(path):
    path = Path(path)
    # Evita incluir zips/manifests antigos guardados em outputs/zips dentro do novo zip.
    try:
        path.resolve().relative_to(ZIP_DIR.resolve())
        return True
    except ValueError:
        return False

artifact_sources = []
if INCLUDE_OUTPUTS_IN_ZIP and OUTPUT_DIR.exists():
    artifact_sources.append(("outputs", OUTPUT_DIR))
if INCLUDE_CACHE_IN_ZIP and CACHE_DIR.exists():
    artifact_sources.append(("cache", CACHE_DIR))
if INCLUDE_RAW_VIDEOS_IN_ZIP and VIDEO_DIR.exists():
    artifact_sources.append(("raw/videos_flv", VIDEO_DIR))

manifest_rows = []
for source_name, source_dir in artifact_sources:
    for path in sorted(source_dir.rglob("*")):
        if path.is_file() and not should_skip_artifact(path):
            rel = Path(source_name) / path.relative_to(source_dir)
            manifest_rows.append({
                "source": source_name,
                "relative_path": str(rel),
                "bytes": int(path.stat().st_size),
            })

download_manifest = pd.DataFrame(manifest_rows)
download_manifest_path = ZIP_DIR / f"download_manifest_{timestamp}.csv"
download_manifest.to_csv(download_manifest_path, index=False)
download_manifest.to_csv(TABLE_DIR / "download_manifest_latest.csv", index=False)

with zipfile.ZipFile(artifact_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    zf.write(download_manifest_path, arcname="outputs/download_manifest.csv")
    for source_name, source_dir in artifact_sources:
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue
            if should_skip_artifact(path):
                continue
            arcname = Path(source_name) / path.relative_to(source_dir)
            zf.write(path, arcname=str(arcname))

zip_size_mb = artifact_zip.stat().st_size / (1024 ** 2)
print(f"Arquivo criado: {artifact_zip}")
print(f"Tamanho: {zip_size_mb:.1f} MB")
print(f"Arquivos incluídos: {len(download_manifest):,}")
display(download_manifest.sort_values("bytes", ascending=False).head(20))

if IN_COLAB:
    from google.colab import files
    files.download(str(artifact_zip))
else:
    print("Fora do Colab, o arquivo ficou salvo no caminho acima.")
"""
    ),
]

nbf.write(nb, OUT)
print(f"Wrote {OUT} with {len(nb.cells)} cells")
