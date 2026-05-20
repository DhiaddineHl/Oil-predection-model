"""Reusable cleaning and feature-engineering helpers used by ``build_dataset.py``.

The functions here are deliberately small and side-effect free so they can be
reused by the API (for live inference on raw user input) and by the notebook.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_wti(path: Path) -> pd.DataFrame:
    """Load WTI spot price from ``RWTCd.xls`` (legacy .xls, sheet 'Data 1')."""
    df = pd.read_excel(path, sheet_name="Data 1", skiprows=2, engine="xlrd")
    df.columns = [str(c).strip() for c in df.columns]
    # Drop duplicate column names that may sneak in from the source file —
    # otherwise df["date"] returns a DataFrame and to_datetime tries to assemble
    # year/month/day components and raises "cannot assemble with duplicate keys".
    df = df.loc[:, ~df.columns.duplicated()].copy()

    price_idx = next(
        (i for i, c in enumerate(df.columns) if "WTI" in c or "Cushing" in c),
        1 if len(df.columns) > 1 else 0,
    )
    date_series = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    price_series = pd.to_numeric(df.iloc[:, price_idx], errors="coerce")

    out = pd.DataFrame({"date": date_series, "WTI_price": price_series})
    out = out.dropna(subset=["date", "WTI_price"]).reset_index(drop=True)
    return out


GPR_COLUMNS = ["date", "GPRD", "GPRD_ACT", "GPRD_THREAT", "GPRD_MA7", "GPRD_MA30"]


def load_gpr(path: Path) -> pd.DataFrame:
    """Load the Daily Geopolitical Risk Index file."""
    df = pd.read_excel(path, engine="xlrd")
    df.columns = [str(c).strip() for c in df.columns]
    # Drop duplicate column names — pandas keeps both copies on read, which
    # then causes df["date"] to return a DataFrame and triggers the
    # "cannot assemble with duplicate keys" error from to_datetime.
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Pick the date column by name (case-insensitive). Common variants in the
    # published GPRD file are "DATE", "Date", "date", "DAY" or "month".
    date_idx = next(
        (
            i
            for i, c in enumerate(df.columns)
            if c.lower() in {"date", "day", "month"}
        ),
        0,
    )
    date_series = pd.to_datetime(df.iloc[:, date_idx], errors="coerce")

    out: dict[str, pd.Series] = {"date": date_series}
    for col in ("GPRD", "GPRD_ACT", "GPRD_THREAT", "GPRD_MA7", "GPRD_MA30"):
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")

    result = pd.DataFrame(out)
    result = result.dropna(subset=["date"]).reset_index(drop=True)
    return result


def load_epu(path: Path) -> pd.DataFrame:
    """Load the monthly Global Economic Policy Uncertainty file."""
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()].copy()

    if "Year" not in df.columns or "Month" not in df.columns:
        raise ValueError(
            f"EPU file is missing Year/Month columns. Found: {list(df.columns)}"
        )

    year = pd.to_numeric(df["Year"], errors="coerce")
    month = pd.to_numeric(df["Month"], errors="coerce")

    out: dict[str, pd.Series] = {"Year": year, "Month": month}
    for col in ("GEPU_current", "GEPU_ppp"):
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")

    result = pd.DataFrame(out).dropna(subset=["Year", "Month"]).reset_index(drop=True)
    result["Year"] = result["Year"].astype(int)
    result["Month"] = result["Month"].astype(int)
    return result


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------
def merge_sources(
    wti: pd.DataFrame,
    gpr: pd.DataFrame,
    epu: pd.DataFrame,
    start: str = "1997-01-01",
    end: str = "2025-11-30",
) -> pd.DataFrame:
    """Join the three sources on a daily grain and forward-fill monthly EPU."""
    df = pd.merge(wti, gpr, on="date", how="inner")
    df["Year"] = df["date"].dt.year
    df["Month"] = df["date"].dt.month
    df = pd.merge(df, epu, on=["Year", "Month"], how="left")

    # Forward fill EPU within each (Year, Month) block to keep daily granularity.
    for col in ("GEPU_current", "GEPU_ppp"):
        if col in df.columns:
            df[col] = df[col].ffill()

    df = df.sort_values("date").reset_index(drop=True)
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
LAGS = (1, 2, 3, 5, 10, 21)
ROLL_WINDOWS = (7, 30)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling, calendar and interaction features.

    Operates on a copy and returns a new dataframe ordered by date.
    """
    out = df.sort_values("date").copy().reset_index(drop=True)

    for lag in LAGS:
        out[f"WTI_lag_{lag}"] = out["WTI_price"].shift(lag)

    for window in ROLL_WINDOWS:
        out[f"WTI_roll_mean_{window}"] = (
            out["WTI_price"].shift(1).rolling(window=window).mean()
        )
        out[f"WTI_roll_std_{window}"] = (
            out["WTI_price"].shift(1).rolling(window=window).std()
        )

    out["WTI_log_return"] = np.log(out["WTI_price"] / out["WTI_price"].shift(1))
    out["day_of_week"] = out["date"].dt.dayofweek.astype(int)
    out["month"] = out["date"].dt.month.astype(int)

    if "GPRD" in out.columns and "GEPU_current" in out.columns:
        out["GPRD_x_GEPU"] = out["GPRD"] * out["GEPU_current"]

    return out


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Append the next-day price target and drop the trailing NaN row."""
    out = df.copy()
    out["target"] = out["WTI_price"].shift(-1)
    out = out.dropna(subset=["target"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def chronological_split(
    df: pd.DataFrame,
    train_end: str = "2020-12-31",
    val_end: str = "2022-12-31",
) -> dict[str, pd.DataFrame]:
    """Return train / val / test splits ordered chronologically (no shuffle)."""
    df = df.sort_values("date").reset_index(drop=True)
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    train = df[df["date"] <= train_end_ts].reset_index(drop=True)
    val = df[
        (df["date"] > train_end_ts) & (df["date"] <= val_end_ts)
    ].reset_index(drop=True)
    test = df[df["date"] > val_end_ts].reset_index(drop=True)
    return {"train": train, "val": val, "test": test}


# ---------------------------------------------------------------------------
# Feature-column helper
# ---------------------------------------------------------------------------
DROP_FROM_FEATURES = {"date", "target", "Year", "Month"}


def feature_columns(df: pd.DataFrame, exclude: Iterable[str] = ()) -> list[str]:
    """Return modelling feature column names (numeric, no leakage columns)."""
    excluded = set(exclude) | DROP_FROM_FEATURES
    return [
        c
        for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]
