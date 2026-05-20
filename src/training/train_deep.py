"""Train the deep-learning oil-price models (BiLSTM and CNN+GRU)."""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.deep_model import (
    EarlyStopping,
    TimeSeriesDataset,
    build_deep_model,
)
from src.training.train_baseline import compute_metrics

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
PALETTE = sns.color_palette("deep")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def _set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_abs = 0.0
    n_samples = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            preds = model(x)
            loss = loss_fn(preds, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            bs = y.size(0)
            total_loss += loss.item() * bs
            total_abs += torch.sum(torch.abs(preds.detach() - y)).item()
            n_samples += bs

    return total_loss / max(n_samples, 1), total_abs / max(n_samples, 1)


def _predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for x, _y in loader:
            x = x.to(device, non_blocking=True)
            preds = model(x).detach().cpu().numpy()
            out.append(preds)
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------
def train(
    model_name: str,
    processed_dir: Path,
    model_dir: Path,
    metrics_dir: Path,
    figures_dir: Path,
    seq_len: int = 30,
    batch_size: int = 64,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 15,
    device: str | None = None,
) -> dict:
    _set_seeds(42)

    train_df = pd.read_csv(processed_dir / "train.csv", parse_dates=["date"])
    val_df = pd.read_csv(processed_dir / "val.csv", parse_dates=["date"])
    test_df = pd.read_csv(processed_dir / "test.csv", parse_dates=["date"])
    feat_cols = json.loads((processed_dir / "feature_names.json").read_text())
    feat_cols = [c for c in feat_cols if c in train_df.columns]

    train_ds = TimeSeriesDataset.from_dataframe(train_df, feat_cols, seq_len=seq_len)
    val_ds = TimeSeriesDataset.from_dataframe(val_df, feat_cols, seq_len=seq_len)
    test_ds = TimeSeriesDataset.from_dataframe(test_df, feat_cols, seq_len=seq_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Using device: %s", device_t)

    model = build_deep_model(model_name, n_features=len(feat_cols)).to(device_t)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model %s — trainable params: %d", model_name, n_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    loss_fn = nn.MSELoss()
    stopper = EarlyStopping(patience=patience)

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []
    }

    best_state: dict | None = None
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss, train_mae = _run_epoch(
            model, train_loader, loss_fn, device_t, optimizer=optimizer
        )
        val_loss, val_mae = _run_epoch(model, val_loader, loss_fn, device_t)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)

        scheduler.step(val_loss)
        improved = stopper.step(val_loss)
        if improved:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        logger.info(
            "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  train_mae=%.4f  val_mae=%.4f  lr=%.2e%s",
            epoch, epochs, train_loss, val_loss, train_mae, val_mae,
            optimizer.param_groups[0]["lr"],
            "  *" if improved else "",
        )

        if stopper.should_stop:
            logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
            break

    train_time = time.perf_counter() - t0

    if best_state is not None:
        model.load_state_dict(best_state)

    # ----- Evaluation -------------------------------------------------------
    y_true = np.array(
        [test_ds[i][1].item() for i in range(len(test_ds))], dtype=np.float32
    )
    y_pred = _predict(model, test_loader, device_t)

    metrics = compute_metrics(y_true, y_pred)

    # Inference latency: time per single-sample prediction.
    if len(test_ds) > 0:
        x_one = test_ds[0][0].unsqueeze(0).to(device_t)
        model.eval()
        with torch.no_grad():
            _ = model(x_one)
            t_inf = time.perf_counter()
            for _ in range(50):
                _ = model(x_one)
            inference_ms = (time.perf_counter() - t_inf) / 50.0 * 1000.0
    else:
        inference_ms = 0.0

    logger.info("Test metrics: %s", metrics)

    # ----- Persist artefacts ------------------------------------------------
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = model_dir / f"deep_{model_name}.pt"
    torch.save(
        {
            "model_name": model_name,
            "state_dict": model.state_dict(),
            "seq_len": seq_len,
            "feature_columns": feat_cols,
            "n_features": len(feat_cols),
            "history": history,
        },
        ckpt_path,
    )
    logger.info("Saved model -> %s", ckpt_path)

    payload = {
        "model": f"deep_{model_name}",
        "seq_len": seq_len,
        "epochs_run": len(history["train_loss"]),
        "train_time_s": train_time,
        "inference_ms_per_sample": inference_ms,
        "n_params": int(n_params),
        "test": metrics,
        "history": history,
    }
    metrics_path = metrics_dir / f"deep_{model_name}_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved metrics -> %s", metrics_path)

    # ----- Plots ------------------------------------------------------------
    _plot_loss_curve(history, figures_dir / f"deep_{model_name}_loss_curve.png", model_name)

    # Align predictions to dates: each prediction corresponds to row index
    # ``seq_len - 1 + i`` of the test set.
    aligned_dates = test_df["date"].iloc[seq_len - 1 : seq_len - 1 + len(y_pred)].values
    _plot_predictions(
        aligned_dates,
        y_true,
        y_pred,
        figures_dir / f"deep_{model_name}_predictions.png",
        model_name,
    )
    _plot_residuals(
        aligned_dates,
        y_true,
        y_pred,
        figures_dir / f"deep_{model_name}_residuals.png",
        model_name,
    )

    return payload


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _plot_loss_curve(history: dict, path: Path, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["train_loss"], label="Train loss", color=PALETTE[0])
    ax.plot(history["val_loss"], label="Val loss", color=PALETTE[3])
    ax.set_title(f"Loss curve — {model_name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_predictions(
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
    model_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, y_true, label="Actual", color=PALETTE[0], linewidth=1.5)
    ax.plot(dates, y_pred, label="Predicted", color=PALETTE[2], linewidth=1.2)
    ax.set_title(f"Test-set predictions — {model_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("WTI price (USD / barrel)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_residuals(
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
    model_name: str,
) -> None:
    residuals = y_pred - y_true
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(dates, residuals, color=PALETTE[3], linewidth=0.9)
    axes[0].axhline(0, color="black", linewidth=0.7, linestyle="--")
    axes[0].set_title(f"Residuals over time — {model_name}")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Predicted − Actual (USD)")
    axes[0].tick_params(axis="x", rotation=30)

    sns.histplot(residuals, kde=True, ax=axes[1], color=PALETTE[2])
    axes[1].set_title("Residual distribution")
    axes[1].set_xlabel("Residual (USD)")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    root = _project_root()
    parser = argparse.ArgumentParser(description="Train a deep learning oil-price model.")
    parser.add_argument(
        "--model", choices=["bilstm", "cnn_gru"], default="bilstm",
        help="Which deep architecture to train.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default=None, help="cpu / cuda / mps (auto-detected)")
    parser.add_argument(
        "--processed-dir", type=Path, default=root / "data" / "processed"
    )
    parser.add_argument("--model-dir", type=Path, default=root / "models")
    parser.add_argument(
        "--metrics-dir", type=Path, default=root / "outputs" / "metrics"
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=root / "outputs" / "figures"
    )
    parser.add_argument(
        "--log-file", type=Path, default=root / "outputs" / "training.log"
    )
    args = parser.parse_args()

    _setup_logging(args.log_file)

    try:
        train(
            model_name=args.model,
            processed_dir=args.processed_dir,
            model_dir=args.model_dir,
            metrics_dir=args.metrics_dir,
            figures_dir=args.figures_dir,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            device=args.device,
        )
    except FileNotFoundError as exc:
        logger.error("Missing processed file: %s", exc)
        raise
    except Exception:
        logger.exception("Deep model training failed")
        raise


if __name__ == "__main__":
    main()
