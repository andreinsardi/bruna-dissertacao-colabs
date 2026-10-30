"""
temporal.py
Utilitários para normalização temporal (T=100) e construção do dataset.
"""

import numpy as np
from scipy.interpolate import interp1d


def normalize_sequence_length(sequence: np.ndarray, target_len: int = 100) -> np.ndarray:
    """
    Normaliza uma sequência de shape (T_orig, D) para (target_len, D).

    Estratégia:
    - Se T_orig == target_len: retorna como está
    - Se T_orig > target_len: subamostrar via interpolação linear
    - Se T_orig < target_len: interpolar (upsampling) via interpolação linear
    - Se T_orig == 0: retorna zeros

    Usa interpolação linear frame-a-frame (independente por feature).
    """
    T_orig, D = sequence.shape

    if T_orig == 0:
        return np.zeros((target_len, D), dtype=np.float32)

    if T_orig == target_len:
        return sequence.astype(np.float32)

    # Índices originais normalizados para [0, 1]
    x_orig = np.linspace(0, 1, T_orig)
    x_target = np.linspace(0, 1, target_len)

    # Interpolar cada feature independentemente
    interpolator = interp1d(x_orig, sequence, axis=0, kind='linear')
    result = interpolator(x_target)

    return result.astype(np.float32)


def compute_temporal_derivatives(X: np.ndarray) -> np.ndarray:
    """
    Adiciona velocidade, aceleração e jerk como canais adicionais.
    Seguindo o protocolo do EmoT (Mazzia et al., 2021).

    Args:
        X: shape (N, T, D) — posições dos landmarks

    Returns:
        X_aug: shape (N, T, D*4) — [posição, velocidade, aceleração, jerk]

    Fórmulas:
        Velocidade:    v[t] = p[t] - p[t-1]
        Aceleração:    a[t] = p[t+1] - 2·p[t] + p[t-1]
        Jerk:          j[t] = p[t+2] - 2·p[t+1] + 2·p[t-1] - p[t-2]

    Bordas: preenchidas com zero (frames sem vizinhos suficientes).
    """
    N, T, D = X.shape

    # Velocidade: v[t] = p[t] - p[t-1]
    vel = np.zeros_like(X)
    vel[:, 1:, :] = X[:, 1:, :] - X[:, :-1, :]

    # Aceleração: a[t] = p[t+1] - 2·p[t] + p[t-1]
    acc = np.zeros_like(X)
    acc[:, 1:-1, :] = X[:, 2:, :] - 2 * X[:, 1:-1, :] + X[:, :-2, :]

    # Jerk (3ª derivada): j[t] = p[t+2] - 2·p[t+1] + 2·p[t-1] - p[t-2]
    jrk = np.zeros_like(X)
    jrk[:, 2:-2, :] = (X[:, 4:, :] - 2 * X[:, 3:-1, :]
                        + 2 * X[:, 1:-3, :] - X[:, :-4, :])

    return np.concatenate([X, vel, acc, jrk], axis=-1).astype(np.float32)


def build_dataset_T100(
    manifest_df,
    load_csv_fn,
    target_len: int = 100,
    verbose: bool = True,
):
    """
    Constrói o dataset normalizado (N, T, D) a partir do manifest filtrado.

    Args:
        manifest_df: DataFrame filtrado (status_ok == True)
        load_csv_fn: função para carregar CSV -> DataFrame
        target_len: comprimento temporal alvo
        verbose: se True, mostra progresso

    Returns:
        X: np.ndarray shape (N, target_len, D)
        y: np.ndarray shape (N,) com emotion_code (0-indexed)
        meta: list de dicts com metadados por amostra
    """
    from tqdm import tqdm

    X_list = []
    y_list = []
    meta_list = []
    skipped = 0

    iterator = manifest_df.iterrows()
    if verbose:
        iterator = tqdm(list(iterator), desc="Construindo dataset T=100")

    for idx, row in iterator:
        df = load_csv_fn(row["filepath"])
        if df is None:
            skipped += 1
            continue

        # Selecionar apenas colunas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            skipped += 1
            continue

        values = df[numeric_cols].values  # shape (T_orig, D)

        # Preencher NaN com interpolação linear por coluna, ou 0 se não possível
        for col_idx in range(values.shape[1]):
            col = values[:, col_idx]
            nans = np.isnan(col)
            if nans.all():
                values[:, col_idx] = 0.0
            elif nans.any():
                not_nan = ~nans
                x = np.arange(len(col))
                values[nans, col_idx] = np.interp(x[nans], x[not_nan], col[not_nan])

        # Normalizar comprimento
        seq_normalized = normalize_sequence_length(values, target_len)
        X_list.append(seq_normalized)

        # emotion_code 1-indexed -> 0-indexed
        y_list.append(int(row["emotion_code"]) - 1)
        meta_list.append({
            "filepath": row["filepath"],
            "filename": row["filename"],
            "actor_id": int(row["actor_id"]),
            "emotion_code": int(row["emotion_code"]),
            "emotion_label": row["emotion_label"],
            "n_frames_orig": int(row["n_frames"]),
        })

    if verbose:
        print(f"Dataset construído: {len(X_list)} amostras, {skipped} puladas")

    X = np.stack(X_list, axis=0)  # (N, T, D)
    y = np.array(y_list, dtype=np.int64)  # (N,)

    return X, y, meta_list
