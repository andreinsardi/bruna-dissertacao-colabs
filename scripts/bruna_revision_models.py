"""Canonical model classes copied verbatim from executed notebook 08.
Run: 20260809_223452_c0613d3c. No architectural changes.
"""
import torch
import torch.nn as nn

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


