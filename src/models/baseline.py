"""Baseline models: naive persistence and Random Forest regressor."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


# ---------------------------------------------------------------------------
# Naive persistence
# ---------------------------------------------------------------------------
@dataclass
class NaivePersistenceModel:
    """Predicts that tomorrow's price equals today's WTI price.

    The model expects a ``WTI_price`` (or scaled equivalent) column in the
    input, but for evaluation we work on the raw ``WTI_price`` column from
    ``full_dataset.csv``.
    """

    price_col: str = "WTI_price"

    def fit(self, df: pd.DataFrame) -> "NaivePersistenceModel":  # noqa: D401
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.price_col].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Random Forest pipeline
# ---------------------------------------------------------------------------
def build_random_forest_pipeline(
    n_estimators: int = 500,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Pipeline:
    """Identity scaler + RandomForestRegressor.

    The data fed to this pipeline is already standard-scaled by the dataset
    build step; the identity step is included so the pipeline structure
    matches downstream usage and feature names are preserved.
    """
    identity = FunctionTransformer(func=None, feature_names_out="one-to-one")
    return Pipeline(
        steps=[
            ("identity", identity),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )
