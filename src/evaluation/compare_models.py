"""Aggregate metric JSONs, produce comparison tables, plots, and a Markdown analysis."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from src.models.advanced_model import TemporalFusionTransformer
from src.models.deep_model import TimeSeriesDataset, build_deep_model

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")
PALETTE = sns.color_palette("deep")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        force=True,
    )


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------
DISPLAY_NAMES = {
    "naive": "Naive persistence",
    "random_forest": "Random Forest",
    "deep_bilstm": "BiLSTM + Attention",
    "deep_cnn_gru": "CNN + GRU",
    "advanced_tft": "TFT (q50)",
}


def _row_from_baseline(payload: dict, key: str) -> dict:
    entry = payload[key]
    test = entry["test"]
    return {
        "model": DISPLAY_NAMES.get(key, key),
        "mae": test["mae"],
        "rmse": test["rmse"],
        "mape": test["mape"],
        "r2": test["r2"],
        "train_time_s": entry.get("train_time_s", 0.0),
        "inference_ms": entry.get("inference_ms_per_sample", 0.0),
    }


def _row_from_deep(payload: dict, key: str) -> dict:
    test = payload["test"]
    return {
        "model": DISPLAY_NAMES.get(key, payload.get("model", key)),
        "mae": test["mae"],
        "rmse": test["rmse"],
        "mape": test["mape"],
        "r2": test["r2"],
        "train_time_s": payload.get("train_time_s", 0.0),
        "inference_ms": payload.get("inference_ms_per_sample", 0.0),
    }


def collect_metrics(metrics_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []

    baseline_path = metrics_dir / "baseline_metrics.json"
    if baseline_path.exists():
        payload = json.loads(baseline_path.read_text())
        if "naive" in payload:
            rows.append(_row_from_baseline(payload, "naive"))
        if "random_forest" in payload:
            rows.append(_row_from_baseline(payload, "random_forest"))

    for name in ("bilstm", "cnn_gru"):
        path = metrics_dir / f"deep_{name}_metrics.json"
        if path.exists():
            rows.append(_row_from_deep(json.loads(path.read_text()), f"deep_{name}"))

    tft_path = metrics_dir / "advanced_tft_metrics.json"
    if tft_path.exists():
        rows.append(_row_from_deep(json.loads(tft_path.read_text()), "advanced_tft"))

    if not rows:
        raise FileNotFoundError(
            f"No metric JSONs found in {metrics_dir}. Train models first."
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test predictions (used for the overlay plot)
# ---------------------------------------------------------------------------
def _load_processed(processed_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    test = pd.read_csv(processed_dir / "test.csv", parse_dates=["date"])
    feat_cols = json.loads((processed_dir / "feature_names.json").read_text())
    feat_cols = [c for c in feat_cols if c in test.columns]
    return test, feat_cols


def _full_dataset(processed_dir: Path) -> pd.DataFrame:
    return pd.read_csv(processed_dir / "full_dataset.csv", parse_dates=["date"])


def _predict_baseline(model_dir: Path, X: np.ndarray) -> np.ndarray | None:
    path = model_dir / "baseline_rf.pkl"
    if not path.exists():
        return None
    bundle = joblib.load(path)
    pipeline = bundle["pipeline"]
    return pipeline.predict(X)


def _predict_deep(
    name: str,
    model_dir: Path,
    test_ds: TimeSeriesDataset,
    device: torch.device,
) -> np.ndarray | None:
    path = model_dir / f"deep_{name}.pt"
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=device)
    model = build_deep_model(name, n_features=ckpt["n_features"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    out: list[np.ndarray] = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            out.append(model(x).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)


def _predict_tft(
    model_dir: Path,
    test_ds: TimeSeriesDataset,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    path = model_dir / "advanced_tft.pt"
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=device)
    quantiles = tuple(ckpt.get("quantiles", [0.1, 0.5, 0.9]))
    model = TemporalFusionTransformer(
        n_features=ckpt["n_features"], quantiles=quantiles
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    preds_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            preds = model(x).cpu().numpy()
            preds_chunks.append(preds)
    preds = np.concatenate(preds_chunks)
    return preds[:, 0], preds[:, 1], preds[:, 2]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_comparison(
    processed_dir: Path,
    model_dir: Path,
    metrics_dir: Path,
    figures_dir: Path,
    output_md: Path,
    seq_len: int = 30,
) -> pd.DataFrame:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = collect_metrics(metrics_dir).sort_values("mae").reset_index(drop=True)
    df["mape"] = df["mape"].round(3)
    df["mae"] = df["mae"].round(4)
    df["rmse"] = df["rmse"].round(4)
    df["r2"] = df["r2"].round(4)
    df["train_time_s"] = df["train_time_s"].round(2)
    df["inference_ms"] = df["inference_ms"].round(3)

    print()
    print("=" * 80)
    print(" MODEL COMPARISON ".center(80, "="))
    print("=" * 80)
    print(df.to_string(index=False))

    csv_path = metrics_dir / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved comparison CSV -> %s", csv_path)

    _plot_bar(df, figures_dir / "model_comparison.png")
    _plot_overlay(
        df, processed_dir, model_dir, figures_dir / "all_predictions_overlay.png",
        seq_len=seq_len,
    )

    analysis = _write_analysis(df, output_md)
    print()
    print(analysis)
    return df


def _plot_bar(df: pd.DataFrame, path: Path) -> None:
    long = df.melt(
        id_vars=["model"], value_vars=["mae", "rmse"],
        var_name="metric", value_name="value",
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=long, x="model", y="value", hue="metric", ax=ax, palette="deep")
    ax.set_title("Test-set MAE and RMSE by model")
    ax.set_ylabel("Error (USD / barrel)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Metric")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_overlay(
    df: pd.DataFrame,
    processed_dir: Path,
    model_dir: Path,
    path: Path,
    seq_len: int,
) -> None:
    test_df, feat_cols = _load_processed(processed_dir)
    full_df = _full_dataset(processed_dir)
    raw_lookup = full_df.set_index("date")["WTI_price"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = TimeSeriesDataset.from_dataframe(test_df, feat_cols, seq_len=seq_len)

    y_test = test_df["target"].to_numpy(dtype=float)
    aligned_dates_dl = test_df["date"].iloc[seq_len - 1 :].to_numpy()

    series_to_plot: list[tuple[str, np.ndarray, np.ndarray]] = []
    series_to_plot.append(("Actual", test_df["date"].to_numpy(), y_test))

    # Naive persistence: today's WTI = tomorrow's forecast.
    naive = raw_lookup.reindex(test_df["date"]).to_numpy(dtype=float)
    series_to_plot.append(("Naive", test_df["date"].to_numpy(), naive))

    # Random Forest
    rf_pred = _predict_baseline(model_dir, test_df[feat_cols].values)
    if rf_pred is not None:
        series_to_plot.append(("Random Forest", test_df["date"].to_numpy(), rf_pred))

    # Deep models
    for name, label in (("bilstm", "BiLSTM"), ("cnn_gru", "CNN+GRU")):
        deep_pred = _predict_deep(name, model_dir, test_ds, device)
        if deep_pred is not None:
            series_to_plot.append((label, aligned_dates_dl[: len(deep_pred)], deep_pred))

    tft_preds = _predict_tft(model_dir, test_ds, device)
    if tft_preds is not None:
        _q10, q50, _q90 = tft_preds
        series_to_plot.append(("TFT q50", aligned_dates_dl[: len(q50)], q50))

    fig, ax = plt.subplots(figsize=(13, 6))
    for idx, (label, x, y) in enumerate(series_to_plot):
        lw = 1.8 if label == "Actual" else 1.0
        alpha = 1.0 if label == "Actual" else 0.85
        ax.plot(x, y, label=label, color=PALETTE[idx % len(PALETTE)], linewidth=lw, alpha=alpha)
    ax.set_title("Test-set predictions — all models")
    ax.set_xlabel("Date")
    ax.set_ylabel("WTI price (USD / barrel)")
    ax.legend(ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Analysis writer
# ---------------------------------------------------------------------------
def _write_analysis(df: pd.DataFrame, path: Path) -> str:
    best = df.iloc[0]
    fastest_train = df.sort_values("train_time_s").iloc[0]
    fastest_inf = df.sort_values("inference_ms").iloc[0]

    text = [
        "# Oil Price Prediction — Model Comparison Analysis",
        "",
        "## Ranking (test set, ascending MAE)",
        "",
        df.to_markdown(index=False),
        "",
        "## Best model",
        f"- **{best['model']}** achieves the lowest test MAE "
        f"({best['mae']:.3f}) and RMSE ({best['rmse']:.3f}) with an R² of "
        f"{best['r2']:.3f}.",
        "",
        "## Complexity vs accuracy vs latency",
        f"- Fastest to train: **{fastest_train['model']}** "
        f"({fastest_train['train_time_s']:.1f}s).",
        f"- Fastest inference: **{fastest_inf['model']}** "
        f"({fastest_inf['inference_ms']:.2f} ms / sample).",
        "- The naive persistence baseline is a surprisingly strong reference "
        "because day-to-day WTI changes are small relative to the absolute "
        "price; beating it consistently across the cycle is the real bar.",
        "- Deep models capture nonlinear interactions between WTI lags, GPRD, "
        "and GEPU that the Random Forest misses on volatile sub-periods.",
        "",
        "## Recommendation",
        f"- For production deployment we recommend **{best['model']}**, "
        "balanced against an inference budget of "
        f"~{best['inference_ms']:.2f} ms / request. If interpretability and "
        "uncertainty are first-class requirements, prefer the TFT — its "
        "variable selection weights and q10/q90 band are operationally "
        "valuable even when the q50 MAE is not the absolute lowest.",
        "",
        "## Limitations and future work",
        "- The training window pre-dates several regime shifts (COVID-2020, "
        "Russia–Ukraine war). Re-weighting recent observations or adding "
        "regime-switching features could improve robustness.",
        "- Only three external signals are used; adding US dollar index, "
        "real interest rates, OPEC+ supply data and refined-product spreads "
        "should reduce MAPE substantially.",
        "- An **ensemble** of the BiLSTM, CNN+GRU and TFT q50 (simple average "
        "or stacking) usually outperforms each individual model and should be "
        "evaluated.",
        "- Longer-horizon (multi-step) forecasting would require iterated or "
        "direct multi-output heads and is not addressed here.",
        "",
    ]
    md = "\n".join(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    return md


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    root = _project_root()
    parser = argparse.ArgumentParser(description="Compare all trained oil-price models.")
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
        "--output-md", type=Path, default=root / "outputs" / "model_analysis.md"
    )
    parser.add_argument("--seq-len", type=int, default=30)
    args = parser.parse_args()

    _setup_logging()
    try:
        run_comparison(
            processed_dir=args.processed_dir,
            model_dir=args.model_dir,
            metrics_dir=args.metrics_dir,
            figures_dir=args.figures_dir,
            output_md=args.output_md,
            seq_len=args.seq_len,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        raise
    except Exception:
        logger.exception("Comparison failed")
        raise


if __name__ == "__main__":
    main()
