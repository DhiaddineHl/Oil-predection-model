"""PyTorch deep learning models for next-day WTI price prediction.

Two architectures are provided:
    * ``BiLSTMRegressor``  — BiLSTM with attention pooling over time steps.
    * ``CNNGRURegressor``  — 1D-CNN feature extractor + GRU + dense head.

Both models map a tensor of shape ``[batch, seq_len, n_features]`` to a single
scalar (the next-day price).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TimeSeriesDataset(Dataset):
    """Sliding-window dataset for time-series regression.

    Each item is ``(x, y)`` where ``x`` has shape ``(seq_len, n_features)``
    and ``y`` is the target at ``index + seq_len - 1`` of the source frame.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        seq_len: int = 30,
    ) -> None:
        if len(features) != len(targets):
            raise ValueError(
                f"features ({len(features)}) and targets ({len(targets)}) "
                "must have matching length"
            )
        if len(features) <= seq_len:
            raise ValueError(
                f"Not enough rows ({len(features)}) for seq_len={seq_len}"
            )
        self.features = np.asarray(features, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.seq_len = int(seq_len)

    def __len__(self) -> int:
        return len(self.features) - self.seq_len + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len - 1]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "target",
        seq_len: int = 30,
    ) -> "TimeSeriesDataset":
        return cls(
            features=df[feature_cols].to_numpy(dtype=np.float32),
            targets=df[target_col].to_numpy(dtype=np.float32),
            seq_len=seq_len,
        )


# ---------------------------------------------------------------------------
# Attention pooling
# ---------------------------------------------------------------------------
class AttentionPool(nn.Module):
    """Additive attention pooling over the time dimension."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H]
        scores = self.attn(x).squeeze(-1)        # [B, T]
        weights = torch.softmax(scores, dim=1)   # [B, T]
        context = torch.einsum("bt,bth->bh", weights, x)
        return context


# ---------------------------------------------------------------------------
# BiLSTM
# ---------------------------------------------------------------------------
class BiLSTMRegressor(nn.Module):
    """BiLSTM (2 layers, hidden=128) + attention pooling + dense head."""

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        head_hidden: int = 256,
        head_dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = AttentionPool(hidden_dim * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        ctx = self.attn(out)
        return self.head(ctx).squeeze(-1)


# ---------------------------------------------------------------------------
# CNN + GRU hybrid
# ---------------------------------------------------------------------------
class CNNGRURegressor(nn.Module):
    """1D-CNN feature extractor + GRU + dense head."""

    def __init__(
        self,
        n_features: int,
        conv1_channels: int = 64,
        conv2_channels: int = 128,
        kernel_size: int = 3,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        gru_dropout: float = 0.3,
        head_hidden: int = 128,
        head_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(n_features, conv1_channels, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(conv1_channels)
        self.conv2 = nn.Conv1d(
            conv1_channels, conv2_channels, kernel_size, padding=pad
        )
        self.bn2 = nn.BatchNorm1d(conv2_channels)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.act = nn.ReLU()

        self.gru = nn.GRU(
            input_size=conv2_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] -> [B, F, T] for Conv1d
        h = x.transpose(1, 2)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.pool(h)
        # Back to [B, T', C] for GRU
        h = h.transpose(1, 2)
        out, _ = self.gru(h)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_deep_model(
    name: str,
    n_features: int,
    **kwargs,
) -> nn.Module:
    name = name.lower()
    if name == "bilstm":
        return BiLSTMRegressor(n_features=n_features, **kwargs)
    if name == "cnn_gru":
        return CNNGRURegressor(n_features=n_features, **kwargs)
    raise ValueError(f"Unknown deep model: {name!r}. Use 'bilstm' or 'cnn_gru'.")


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------
class EarlyStopping:
    """Stop training when the monitored value has not improved for ``patience`` epochs."""

    def __init__(self, patience: int = 15, min_delta: float = 0.0) -> None:
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best: Optional[float] = None
        self.bad_epochs = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Return ``True`` when a new best score has been observed."""
        if self.best is None or value < self.best - self.min_delta:
            self.best = value
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            self.should_stop = True
        return False
