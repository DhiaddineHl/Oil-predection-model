"""Build the processed dataset used by every model.

Usage
-----
    python -m src.data.build_dataset
    python -m src.data.build_dataset --data-dir data/raw --processed-dir data/processed
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.preprocessing import (
    DROP_FROM_FEATURES,
    add_features,
    add_target,
    chronological_split,
    feature_columns,
    load_epu,
    load_gpr,
    load_wti,
    merge_sources,
)

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def build_dataset(
    data_dir: Path,
    processed_dir: Path,
    model_dir: Path,
    start: str = "1997-01-01",
    end: str = "2025-11-30",
) -> dict[str, pd.DataFrame]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load -----------------------------------------------------------------
    wti_path = data_dir / "RWTCd.xls"
    gpr_path = data_dir / "data_gpr_daily_recent.xls"
    epu_path = data_dir / "Global_Policy_Uncertainty_Data.xlsx"

    for p in (wti_path, gpr_path, epu_path):
        if not p.exists():
            raise FileNotFoundError(f"Required raw file is missing: {p}")

    logger.info("Loading WTI from %s", wti_path)
    wti = load_wti(wti_path)
    logger.info("Loading GPR from %s", gpr_path)
    gpr = load_gpr(gpr_path)
    logger.info("Loading EPU from %s", epu_path)
    epu = load_epu(epu_path)

    logger.info(
        "Loaded shapes — WTI: %s, GPR: %s, EPU: %s", wti.shape, gpr.shape, epu.shape
    )

    # 2. Merge ----------------------------------------------------------------
    merged = merge_sources(wti, gpr, epu, start=start, end=end)
    logger.info("Merged shape (filtered to %s..%s): %s", start, end, merged.shape)

    # 3. Feature engineering --------------------------------------------------
    featured = add_features(merged)
    featured = add_target(featured)

    # Drop rows where lag/rolling features are NaN (warm-up period).
    feat_cols_pre = feature_columns(featured)
    before = len(featured)
    featured = featured.dropna(subset=feat_cols_pre).reset_index(drop=True)
    logger.info(
        "Dropped %d warm-up rows with NaN features (kept %d)",
        before - len(featured),
        len(featured),
    )

    # 4. Persist the full (unscaled) dataset for the dashboard ---------------
    full_path = processed_dir / "full_dataset.csv"
    featured.to_csv(full_path, index=False)
    logger.info("Wrote %s (%d rows)", full_path, len(featured))

    # 5. Chronological split --------------------------------------------------
    splits = chronological_split(featured)
    for name, part in splits.items():
        logger.info(
            "Split %s: %d rows, %s -> %s",
            name,
            len(part),
            part["date"].min() if len(part) else "n/a",
            part["date"].max() if len(part) else "n/a",
        )

    # 6. Scale features (fit on train only) ----------------------------------
    feat_cols = feature_columns(featured)
    scaler = StandardScaler()
    scaler.fit(splits["train"][feat_cols].values)

    scaled = {}
    for name, part in splits.items():
        part = part.copy()
        part[feat_cols] = scaler.transform(part[feat_cols].values)
        out_path = processed_dir / f"{name}.csv"
        part.to_csv(out_path, index=False)
        scaled[name] = part
        logger.info("Wrote %s (%d rows)", out_path, len(part))

    scaler_path = model_dir / "scaler.pkl"
    joblib.dump({"scaler": scaler, "feature_columns": feat_cols}, scaler_path)
    logger.info("Wrote scaler to %s", scaler_path)

    feat_path = processed_dir / "feature_names.json"
    feat_path.write_text(json.dumps(feat_cols, indent=2))
    logger.info("Wrote feature names to %s", feat_path)

    # 7. Summary --------------------------------------------------------------
    _print_summary(featured, splits, feat_cols)
    return splits


def _print_summary(
    full: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    feat_cols: list[str],
) -> None:
    print()
    print("=" * 70)
    print(" DATASET SUMMARY ".center(70, "="))
    print("=" * 70)

    print(f"Full rows:     {len(full):,}")
    print(f"Date range:    {full['date'].min().date()}  ->  {full['date'].max().date()}")
    print(f"Features:      {len(feat_cols)}")
    print(f"Missing total: {int(full[feat_cols].isna().sum().sum())}")

    print("\nSplits:")
    for name, part in splits.items():
        if len(part):
            print(
                f"  {name:5s}: {len(part):6,} rows  "
                f"({part['date'].min().date()} -> {part['date'].max().date()})"
            )
        else:
            print(f"  {name:5s}: empty")

    print("\nTarget statistics (full):")
    print(full["target"].describe().to_string())

    print("\nFeature columns:")
    for i, col in enumerate(feat_cols, 1):
        print(f"  {i:2d}. {col}")
    print("=" * 70)


def _setup_logging(log_path: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def main() -> None:
    root = _project_root()
    parser = argparse.ArgumentParser(description="Build the oil-price dataset.")
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "raw")
    parser.add_argument(
        "--processed-dir", type=Path, default=root / "data" / "processed"
    )
    parser.add_argument("--model-dir", type=Path, default=root / "models")
    parser.add_argument("--start", default="1997-01-01")
    parser.add_argument("--end", default="2025-11-30")
    parser.add_argument(
        "--log-file", type=Path, default=root / "outputs" / "training.log"
    )
    args = parser.parse_args()

    _setup_logging(args.log_file)

    try:
        build_dataset(
            data_dir=args.data_dir,
            processed_dir=args.processed_dir,
            model_dir=args.model_dir,
            start=args.start,
            end=args.end,
        )
    except FileNotFoundError as exc:
        logger.error("Missing input file: %s", exc)
        raise
    except Exception:
        logger.exception("Failed to build dataset")
        raise


if __name__ == "__main__":
    main()
