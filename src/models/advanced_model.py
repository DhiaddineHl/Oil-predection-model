"""TFT-inspired advanced model with variable selection, attention, and
quantile output (q10 / q50 / q90).

The architecture follows the Temporal Fusion Transformer pattern at a
reduced scale that is appropriate for the size of the oil-price dataset:

    input [B, T, F]
        |--> Per-feature linear embedding (F -> d_model)
        |--> Variable Selection Network (softmax across features per time-step)
        |--> LSTM encoder (2 layers)
        |--> Multi-Head Self-Attention
        |--> Gated Residual Network + skip connection
        |--> Quantile head (q10, q50, q90)
"""
from __future__ import annotations

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Gated Residual Network
# ---------------------------------------------------------------------------
class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.gate = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.elu = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.elu(self.fc1(x))
        h = self.dropout(self.fc2(h))
        gate = torch.sigmoid(self.gate(x))
        return self.layer_norm(x + gate * h)


# ---------------------------------------------------------------------------
# Variable Selection Network
# ---------------------------------------------------------------------------
class VariableSelectionNetwork(nn.Module):
    """Embeds each input feature separately and computes softmax weights
    over the features at every time-step.

    Returns
    -------
    selected : ``[B, T, d_model]``
    weights  : ``[B, T, n_features]`` — average across batch/time is exposed
        as the model's *feature importance*.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.feature_embeddings = nn.ModuleList(
            [nn.Linear(1, d_model) for _ in range(n_features)]
        )
        self.weight_grn = GatedResidualNetwork(
            input_dim=n_features * d_model,
            hidden_dim=n_features * d_model,
            dropout=dropout,
        )
        self.weight_layer = nn.Linear(n_features * d_model, n_features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, F]
        embeds = []
        for i, emb in enumerate(self.feature_embeddings):
            embeds.append(emb(x[..., i : i + 1]))  # [B, T, d_model]
        embeds = torch.stack(embeds, dim=-2)        # [B, T, F, d_model]

        flat = embeds.flatten(start_dim=-2)         # [B, T, F * d_model]
        flat = self.weight_grn(flat)
        weights = torch.softmax(self.weight_layer(flat), dim=-1)  # [B, T, F]

        weighted = (embeds * weights.unsqueeze(-1)).sum(dim=-2)   # [B, T, d_model]
        return weighted, weights


# ---------------------------------------------------------------------------
# TFT-inspired model
# ---------------------------------------------------------------------------
class TemporalFusionTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 32,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.quantiles = quantiles

        # Variable Selection
        self.vsn = VariableSelectionNetwork(
            n_features=n_features, d_model=d_model, dropout=dropout
        )

        # Project VSN output to LSTM hidden size so residuals align.
        self.input_proj = nn.Linear(d_model, lstm_hidden)

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Self-attention block
        self.attn = nn.MultiheadAttention(
            embed_dim=lstm_hidden,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(lstm_hidden)

        # Final GRN with skip connection
        self.post_grn = GatedResidualNetwork(
            input_dim=lstm_hidden,
            hidden_dim=lstm_hidden,
            dropout=dropout,
        )

        # Quantile head
        self.head = nn.Linear(lstm_hidden, len(quantiles))

    def forward(
        self, x: torch.Tensor, return_weights: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        # x: [B, T, F]
        selected, weights = self.vsn(x)         # [B, T, d_model], [B, T, F]
        projected = self.input_proj(selected)   # [B, T, lstm_hidden]

        lstm_out, _ = self.lstm(projected)      # [B, T, lstm_hidden]

        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out, need_weights=False)
        attn_out = self.attn_norm(lstm_out + attn_out)

        gated = self.post_grn(attn_out)
        last = gated[:, -1, :]                  # [B, lstm_hidden]

        quantile_preds = self.head(last)        # [B, n_quantiles]

        if return_weights:
            return quantile_preds, weights
        return quantile_preds


# ---------------------------------------------------------------------------
# Quantile (pinball) loss
# ---------------------------------------------------------------------------
class QuantileLoss(nn.Module):
    def __init__(self, quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)) -> None:
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # preds: [B, n_q],  target: [B]
        target = target.unsqueeze(-1).expand_as(preds)
        errors = target - preds
        q = torch.tensor(self.quantiles, device=preds.device).view(1, -1)
        loss = torch.maximum(q * errors, (q - 1.0) * errors)
        return loss.mean()
