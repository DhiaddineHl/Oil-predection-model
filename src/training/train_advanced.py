"""Train the TFT-inspired advanced oil-price model with quantile loss."""
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
from torch.utils.data import DataLoader

from src.models.advanced_model import QuantileLoss, TemporalFusionTransformer
from src.models.deep_model import EarlyStopping, TimeSeriesDataset
from src.training.train_baseline import compute_metrics

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
PALETTE = sns.color_palette("deep")

QUANTILES = (0.1, 0.5, 0.9)


def _set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def _run_epoch(
    model: TemporalFusionTransformer,
    loader: DataLoader,
    loss_fn: QuantileLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_abs, n_samples = 0.0, 0.0, 0

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

            q50 = preds[:, QUANTILES.index(0.5)]
            bs = y.size(0)
            total_loss += loss.item() * bs
            total_abs += torch.sum(torch.abs(q50.detach() - y)).item()
            n_samples += bs

    return total_loss / max(n_samples, 1), total_abs / max(n_samples, 1)


def _predict_quantiles_with_weights(
    model: TemporalFusionTransformer,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds_chunks: list[np.ndarray] = []
    weight_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for x, _y in loader:
            x = x.to(device, non_blocking=True)
            preds, weights = model(x, return_weights=True)
            preds_chunks.append(preds.cpu().numpy())
            # Average weights over time-steps to get per-sample feature importance.
            weight_chunks.append(weights.mean(dim=1).cpu().numpy())
    preds = np.concatenate(preds_chunks) if preds_chunks else np.empty((0, 3))
    weights = np.concatenate(weight_chunks) if weight_chunks else np.empty((0, 0))
    return preds, weights


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def train(
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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Using device: %s", device_t)

    model = TemporalFusionTransformer(
        n_features=len(feat_cols), quantiles=QUANTILES
    ).to(device_t)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("TFT — trainable params: %d", n_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    loss_fn = QuantileLoss(quantiles=QUANTILES)
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
            "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  q50_mae(val)=%.4f  lr=%.2e%s",
            epoch, epochs, train_loss, val_loss, val_mae,
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
    preds_q, vsn_weights = _predict_quantiles_with_weights(model, test_loader, device_t)
    q10, q50, q90 = preds_q[:, 0], preds_q[:, 1], preds_q[:, 2]
    metrics = compute_metrics(y_true, q50)
    metrics["pinball_q10"] = float(
        np.mean(np.maximum(0.1 * (y_true - q10), (0.1 - 1.0) * (y_true - q10)))
    )
    metrics["pinball_q90"] = float(
        np.mean(np.maximum(0.9 * (y_true - q90), (0.9 - 1.0) * (y_true - q90)))
    )
    metrics["coverage_80"] = float(np.mean((y_true >= q10) & (y_true <= q90)))

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

    logger.info("Test metrics (q50): %s", metrics)

    # ----- Persist artefacts ------------------------------------------------
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = model_dir / "advanced_tft.pt"
    torch.save(
        {
            "model_name": "tft",
            "state_dict": model.state_dict(),
            "seq_len": seq_len,
            "feature_columns": feat_cols,
            "n_features": len(feat_cols),
            "quantiles": list(QUANTILES),
            "history": history,
        },
        ckpt_path,
    )
    logger.info("Saved model -> %s", ckpt_path)

    feature_importance = vsn_weights.mean(axis=0) if vsn_weights.size else np.zeros(len(feat_cols))
    payload = {
        "model": "advanced_tft",
        "seq_len": seq_len,
        "epochs_run": len(history["train_loss"]),
        "train_time_s": train_time,
        "inference_ms_per_sample": inference_ms,
        "n_params": int(n_params),
        "quantiles": list(QUANTILES),
        "test": metrics,
        "history": history,
        "feature_importance": dict(zip(feat_cols, map(float, feature_importance))),
    }
    metrics_path = metrics_dir / "advanced_tft_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved metrics -> %s", metrics_path)

    # ----- Plots ------------------------------------------------------------
    _plot_loss_curve(history, figures_dir / "advanced_tft_loss_curve.png")
    aligned_dates = test_df["date"].iloc[seq_len - 1 : seq_len - 1 + len(q50)].values
    _plot_uncertainty(
        aligned_dates, y_true, q10, q50, q90,
        figures_dir / "advanced_tft_predictions_with_uncertainty.png",
    )
    _plot_feature_importance(
        feat_cols, feature_importance,
        figures_dir / "advanced_tft_feature_importance.png",
    )

    return payload


def _plot_loss_curve(history: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["train_loss"], label="Train loss", color=PALETTE[0])
    ax.plot(history["val_loss"], label="Val loss", color=PALETTE[3])
    ax.set_title("Loss curve — TFT (quantile loss)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pinball loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_uncertainty(
    dates: np.ndarray,
    y_true: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(
        dates, q10, q90, color=PALETTE[2], alpha=0.25, label="80% interval (q10–q90)"
    )
    ax.plot(dates, q50, color=PALETTE[2], linewidth=1.2, label="Predicted (q50)")
    ax.plot(dates, y_true, color=PALETTE[0], linewidth=1.5, label="Actual")
    ax.set_title("Test-set predictions with uncertainty — TFT")
    ax.set_xlabel("Date")
    ax.set_ylabel("WTI price (USD / barrel)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_feature_importance(
    feat_cols: list[str], weights: np.ndarray, path: Path
) -> None:
    order = np.argsort(weights)[::-1]
    sorted_cols = [feat_cols[i] for i in order]
    sorted_weights = weights[order]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(feat_cols))))
    sns.barplot(
        x=sorted_weights, y=sorted_cols, ax=ax, color=PALETTE[0], orient="h"
    )
    ax.set_title("Variable Selection Network — average feature importance")
    ax.set_xlabel("Mean softmax weight")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    root = _project_root()
    parser = argparse.ArgumentParser(description="Train the advanced TFT-inspired model.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default=None)
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
        logger.exception("Advanced model training failed")
        raise


if __name__ == "__main__":
    main()
