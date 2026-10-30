"""
models.py
Definição dos 3 modelos para o PoC: MLP, CNN1D, Transformer (EmoT).
Todos projetados para CPU-first com hiperparâmetros pequenos.
7 classes: neutral, happy, sad, angry, fearful, disgust, surprised.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlatMLP(nn.Module):
    """
    MLP simples que achata a entrada (T, D) -> (T*D) e classifica.
    Baseline sem modelagem temporal explícita.
    """

    def __init__(self, T: int, D: int, n_classes: int = 7, hidden: int = 256):
        super().__init__()
        self.flatten_dim = T * D
        self.net = nn.Sequential(
            nn.Linear(self.flatten_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        # x: (B, T, D)
        B = x.size(0)
        x = x.reshape(B, -1)  # (B, T*D)
        return self.net(x)


class TemporalCNN1D(nn.Module):
    """
    CNN 1D temporal: convoluções ao longo do eixo temporal.
    Input: (B, T, D) -> permute -> (B, D, T) para Conv1d.
    """

    def __init__(self, T: int, D: int, n_classes: int = 7,
                 n_filters: int = 64, kernel_size: int = 5):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(D, n_filters, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(n_filters),
            nn.MaxPool1d(2),

            nn.Conv1d(n_filters, n_filters * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(n_filters * 2),
            nn.MaxPool1d(2),

            nn.Conv1d(n_filters * 2, n_filters * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # (B, n_filters*2, 1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(n_filters * 2, n_classes),
        )

    def forward(self, x):
        # x: (B, T, D)
        x = x.permute(0, 2, 1)  # (B, D, T)
        x = self.conv_layers(x)  # (B, C, 1)
        x = x.squeeze(-1)       # (B, C)
        return self.classifier(x)


class PositionalEncoding(nn.Module):
    """Positional encoding sinusoidal padrão."""

    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, d_model)
        return x + self.pe[:, :x.size(1), :]


class EmoTransformer(nn.Module):
    """
    Transformer pequeno para classificação temporal de emoções (EmoT).
    CPU-first: d_model=64, n_layers=2, n_heads=4.

    Salva attention weights para XAI.
    """

    def __init__(self, D: int, n_classes: int = 7,
                 d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, dim_ff: int = 128,
                 dropout: float = 0.1, max_len: int = 200):
        super().__init__()
        self.d_model = d_model

        # Projeção de entrada: D features -> d_model
        self.input_proj = nn.Linear(D, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.input_dropout = nn.Dropout(dropout)

        # Encoder layers com batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # CLS token (aprendido)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Classificador
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

        # Storage para attention weights (para XAI)
        self._attention_weights = []

    def forward(self, x, return_attention=False):
        """
        x: (B, T, D)
        Se return_attention=True, também retorna lista de attention maps.
        """
        B, T, D = x.shape

        # Projetar features para d_model
        x = self.input_proj(x)  # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.input_dropout(x)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, d_model)

        if return_attention:
            # Passar manualmente por cada layer para capturar atenção
            attn_weights_all = []
            for layer in self.encoder.layers:
                # MultiheadAttention com need_weights=True
                # Precisamos chamar self_attn manualmente
                src = layer.norm1(x)
                attn_output, attn_w = layer.self_attn(
                    src, src, src, need_weights=True, average_attn_weights=False
                )
                x = x + layer.dropout1(attn_output)
                # Feed-forward
                src2 = layer.norm2(x)
                ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(src2))))
                x = x + layer.dropout2(ff_output)
                attn_weights_all.append(attn_w.detach())  # (B, n_heads, T+1, T+1)

            cls_repr = x[:, 0, :]  # (B, d_model)
            logits = self.classifier(cls_repr)
            return logits, attn_weights_all
        else:
            x = self.encoder(x)       # (B, T+1, d_model)
            cls_repr = x[:, 0, :]     # (B, d_model)
            logits = self.classifier(cls_repr)
            return logits


def count_parameters(model: nn.Module) -> int:
    """Conta parâmetros treináveis."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
