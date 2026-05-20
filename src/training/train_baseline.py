"""Train the baseline models (naive persistence + RandomForest)."""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib

matplotlib.use("Agg")  # non-interactive backend — figures are saved to disk.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from src.models.baseline import NaivePersistenceModel, build_random_forest_pipeline

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
PALETTE = sns.color_palette("deep")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def _set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred) * 100.0),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _feature_names_from(df: pd.DataFrame, feat_cols: Iterable[str]) -> list[str]:
    return [c for c in feat_cols if c in df.columns]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def train_and_evaluate(
    processed_dir: Path,
    model_dir: Path,
    metrics_dir: Path,
    figures_dir: Path,
    n_estimators: int = 500,
) -> dict[str, dict[str, float]]:
    _set_seeds(42)

    train = pd.read_csv(processed_dir / "train.csv", parse_dates=["date"])
    val = pd.read_csv(processed_dir / "val.csv", parse_dates=["date"])
    test = pd.read_csv(processed_dir / "test.csv", parse_dates=["date"])
    full = pd.read_csv(processed_dir / "full_dataset.csv", parse_dates=["date"])

    feat_cols = json.loads((processed_dir / "feature_names.json").read_text())
    feat_cols = _feature_names_from(train, feat_cols)

    X_train, y_train = train[feat_cols].values, train["target"].values
    X_val, y_val = val[feat_cols].values, val["target"].values
    X_test, y_test = test[feat_cols].values, test["target"].values

    # ----- Naive persistence (uses unscaled WTI_price) ----------------------
    raw_lookup = full.set_index("date")["WTI_price"]
    naive_pred_val = raw_lookup.reindex(val["date"]).to_numpy(dtype=float)
    naive_pred_test = raw_lookup.reindex(test["date"]).to_numpy(dtype=float)

    naive_val_metrics = compute_metrics(y_val, naive_pred_val)
    naive_test_metrics = compute_metrics(y_test, naive_pred_test)
    logger.info("Naive  val:  %s", naive_val_metrics)
    logger.info("Naive  test: %s", naive_test_metrics)

    # ----- Random Forest -----------------------------------------------------
    rf_pipeline = build_random_forest_pipeline(n_estimators=n_estimators)
    logger.info("Fitting RandomForest with %d trees on %d samples ...",
                n_estimators, len(X_train))
    t0 = time.perf_counter()
    rf_pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    logger.info("RF training finished in %.2fs", train_time)

    rf_val_pred = rf_pipeline.predict(X_val)
    rf_test_pred = rf_pipeline.predict(X_test)

    rf_val_metrics = compute_metrics(y_val, rf_val_pred)
    rf_test_metrics = compute_metrics(y_test, rf_test_pred)

    t0 = time.perf_counter()
    _ = rf_pipeline.predict(X_test[:1])
    inference_ms = (time.perf_counter() - t0) * 1000.0

    logger.info("RF     val:  %s", rf_val_metrics)
    logger.info("RF     test: %s", rf_test_metrics)

    # ----- Save artefacts ---------------------------------------------------
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rf_path = model_dir / "baseline_rf.pkl"
    joblib.dump({"pipeline": rf_pipeline, "feature_columns": feat_cols}, rf_path)
    logger.info("Saved RF model -> %s", rf_path)

    metrics_payload = {
        "naive": {
            "model": "naive_persistence",
            "val": naive_val_metrics,
            "test": naive_test_metrics,
            "train_time_s": 0.0,
            "inference_ms_per_sample": 0.0,
        },
        "random_forest": {
            "model": "random_forest",
            "val": rf_val_metrics,
            "test": rf_test_metrics,
            "train_time_s": train_time,
            "inference_ms_per_sample": inference_ms,
            "n_estimators": n_estimators,
        },
    }
    metrics_path = metrics_dir / "baseline_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))
    logger.info("Saved metrics  -> %s", metrics_path)

    # ----- Plot actual vs predicted on test set -----------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test["date"], y_test, label="Actual", color=PALETTE[0], linewidth=1.5)
    ax.plot(
        test["date"],
        naive_pred_test,
        label="Naive",
        color=PALETTE[1],
        linewidth=1.0,
        alpha=0.8,
    )
    ax.plot(
        test["date"],
        rf_test_pred,
        label="Random Forest",
        color=PALETTE[2],
        linewidth=1.2,
    )
    ax.set_title("Baseline models — test set predictions")
    ax.set_xlabel("Date")
    ax.set_ylabel("WTI price (USD / barrel)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    fig_path = figures_dir / "baseline_predictions.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info("Saved figure   -> %s", fig_path)

    return metrics_payload


def main() -> None:
    root = _project_root()
    parser = argparse.ArgumentParser(description="Train baseline oil-price models.")
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
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument(
        "--log-file", type=Path, default=root / "outputs" / "training.log"
    )
    args = parser.parse_args()

    _setup_logging(args.log_file)

    try:
        train_and_evaluate(
            processed_dir=args.processed_dir,
            model_dir=args.model_dir,
            metrics_dir=args.metrics_dir,
            figures_dir=args.figures_dir,
            n_estimators=args.n_estimators,
        )
    except FileNotFoundError as exc:
        logger.error("Missing processed file: %s", exc)
        raise
    except Exception:
        logger.exception("Baseline training failed")
        raise


if __name__ == "__main__":
    main()
