"""
ravdess_utils.py
Utilitários para leitura e parsing dos CSVs do RAVDESS (Kaggle landmarks).
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path

# ── Mapeamento oficial RAVDESS (8 emoções) ──────────────────────────────────
EMOTION_MAP = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}

EMOTION_LABELS = list(EMOTION_MAP.values())  # ordenado por código

# ── Mapeamento 7 classes (merge neutral + calm) ─────────────────────────────
# Seguindo o protocolo do EmoT (Mazzia et al., 2021):
# calm (código RAVDESS 02) é fundido com neutral (código 01) → classe 0
EMOTION_REMAP_7 = {
    1: 0,  # neutral → 0
    2: 0,  # calm → 0 (merged with neutral)
    3: 1,  # happy → 1
    4: 2,  # sad → 2
    5: 3,  # angry → 3
    6: 4,  # fearful → 4
    7: 5,  # disgust → 5
    8: 6,  # surprised → 6
}

EMOTION_LABELS_7 = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"]


def remap_to_7classes(y_8class):
    """
    Remapeia labels 0-indexed de 8 classes para 7 classes (merge neutral+calm).

    Input:  y com valores 0-7  (8 classes, 0-indexed do RAVDESS)
    Output: y com valores 0-6  (7 classes)

    Mapping (0-indexed):
        0→0 (neutral), 1→0 (calm→neutral), 2→1 (happy), 3→2 (sad),
        4→3 (angry), 5→4 (fearful), 6→5 (disgust), 7→6 (surprised)
    """
    _REMAP_TABLE = np.array([0, 0, 1, 2, 3, 4, 5, 6])
    return _REMAP_TABLE[np.asarray(y_8class)]


def parse_ravdess_filename(filename: str) -> dict:
    """
    Extrai metadados do nome de arquivo padrão RAVDESS.
    Formato: 03-01-{emotion}-{intensity}-{statement}-{rep}-{actor}.csv
    Também aceita variantes com 'Actor_XX' no nome.

    Retorna dict com: emotion_code, emotion_label, actor_id, intensity, etc.
    """
    stem = Path(filename).stem  # remove extensão

    # Tentar extrair actor_id de padrão "Actor_XX" no path ou nome
    actor_match = re.search(r'Actor_(\d+)', str(filename), re.IGNORECASE)

    # Extrair os códigos do padrão RAVDESS (XX-XX-XX-XX-XX-XX-XX)
    code_match = re.search(r'(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})', stem)

    if code_match:
        modality = int(code_match.group(1))
        vocal_channel = int(code_match.group(2))
        emotion_code = int(code_match.group(3))
        intensity = int(code_match.group(4))
        statement = int(code_match.group(5))
        repetition = int(code_match.group(6))
        actor_id = int(code_match.group(7))
    else:
        return None

    # Se Actor_XX existe no path, usar como fallback
    if actor_match:
        actor_id_from_path = int(actor_match.group(1))
        # Priorizar o que está no nome do arquivo codificado

    emotion_label = EMOTION_MAP.get(emotion_code, f"unknown_{emotion_code}")

    return {
        "emotion_code": emotion_code,
        "emotion_label": emotion_label,
        "actor_id": actor_id,
        "intensity": intensity,
        "statement": statement,
        "repetition": repetition,
        "modality": modality,
        "vocal_channel": vocal_channel,
    }


def discover_csv_structure(csv_path: str, nrows: int = 5) -> dict:
    """
    Lê as primeiras linhas de um CSV e retorna info sobre a estrutura.
    """
    df = pd.read_csv(csv_path, nrows=nrows)
    return {
        "columns": list(df.columns),
        "n_cols": len(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "shape_sample": df.shape,
    }


def load_landmark_csv(csv_path: str) -> pd.DataFrame:
    """
    Carrega CSV de landmarks. Tenta diferentes separadores se necessário.
    """
    try:
        df = pd.read_csv(csv_path)
        if df.shape[1] <= 1:
            df = pd.read_csv(csv_path, sep=';')
        return df
    except Exception as e:
        print(f"Erro ao ler {csv_path}: {e}")
        return None


def find_all_csvs(base_dir: str) -> list:
    """
    Encontra recursivamente todos os CSVs na pasta base_dir.
    """
    csvs = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith('.csv'):
                csvs.append(os.path.join(root, f))
    csvs.sort()
    return csvs


def build_manifest(csv_paths: list) -> pd.DataFrame:
    """
    Constrói manifest.csv com metadados de cada amostra.
    Colunas: filepath, filename, actor_id, emotion_code, emotion_label,
             n_frames, n_features, has_nan, status_ok
    """
    records = []
    for path in csv_paths:
        filename = os.path.basename(path)
        meta = parse_ravdess_filename(filename)

        if meta is None:
            records.append({
                "filepath": path,
                "filename": filename,
                "actor_id": None,
                "emotion_code": None,
                "emotion_label": None,
                "n_frames": 0,
                "n_features": 0,
                "has_nan": True,
                "status_ok": False,
                "parse_error": "filename_parse_failed",
            })
            continue

        df = load_landmark_csv(path)
        if df is None:
            records.append({
                "filepath": path,
                "filename": filename,
                "actor_id": meta["actor_id"],
                "emotion_code": meta["emotion_code"],
                "emotion_label": meta["emotion_label"],
                "n_frames": 0,
                "n_features": 0,
                "has_nan": True,
                "status_ok": False,
                "parse_error": "csv_read_failed",
            })
            continue

        # Identificar colunas numéricas (landmarks)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        has_nan = bool(df[numeric_cols].isna().any().any())
        n_frames = len(df)
        n_features = len(numeric_cols)

        records.append({
            "filepath": path,
            "filename": filename,
            "actor_id": meta["actor_id"],
            "emotion_code": meta["emotion_code"],
            "emotion_label": meta["emotion_label"],
            "n_frames": n_frames,
            "n_features": n_features,
            "has_nan": has_nan,
            "status_ok": (n_frames > 0 and n_features > 0),
            "parse_error": None,
        })

    return pd.DataFrame(records)
