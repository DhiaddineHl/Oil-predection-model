"""Streamlit dashboard for the oil-price prediction project.

Run with:

    streamlit run src/app/streamlit_app.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = Path(os.environ.get("PROCESSED_DIR", PROJECT_ROOT / "data" / "processed"))
METRICS_DIR = Path(os.environ.get("METRICS_DIR", PROJECT_ROOT / "outputs" / "metrics"))
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Oil Price Prediction Dashboard",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_full_dataset() -> pd.DataFrame | None:
    path = PROCESSED_DIR / "full_dataset.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["date"])


@st.cache_data(show_spinner=False)
def load_comparison() -> pd.DataFrame | None:
    path = METRICS_DIR / "model_comparison.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_feature_names() -> list[str]:
    path = PROCESSED_DIR / "feature_names.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def fetch_api_models() -> dict:
    try:
        r = requests.get(f"{API_URL}/models", timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
api_info = fetch_api_models()
feature_names = api_info.get("feature_columns") or load_feature_names()
default_seq_len = int(api_info.get("seq_len", 30))

st.sidebar.title("🛢️ Controls")
api_status_ok = bool(api_info)
st.sidebar.markdown(
    f"**API:** {'🟢 connected' if api_status_ok else '🔴 unreachable'}  \n"
    f"`{API_URL}`"
)

available_models = (
    [m["name"] for m in api_info.get("models", []) if m.get("loaded")]
    if api_info
    else ["naive", "rf", "bilstm", "cnn_gru", "tft"]
)
selected_model = st.sidebar.selectbox(
    "Model", available_models, index=0 if available_models else None
)
seq_len = st.sidebar.slider("Sequence length", 5, 90, value=default_seq_len)

uploaded = st.sidebar.file_uploader(
    "Upload recent feature data (CSV)",
    type=["csv"],
    help="CSV with the same feature columns as feature_names.json — last rows are used.",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🛢️ Oil Price Prediction Dashboard")
st.caption(
    "WTI next-day price forecasting using WTI lags, Geopolitical Risk (GPRD) "
    "and Global Economic Policy Uncertainty (GEPU)."
)

tab_live, tab_history, tab_compare, tab_about = st.tabs(
    ["Live Prediction", "Historical Analysis", "Model Comparison", "About"]
)


# ---------------------------------------------------------------------------
# Tab 1 — Live prediction
# ---------------------------------------------------------------------------
with tab_live:
    st.subheader("Predict next-day WTI price")

    if not feature_names:
        st.warning(
            "No feature schema found. Build the dataset first: "
            "`python -m src.data.build_dataset`."
        )

    full = load_full_dataset()
    key_features = [
        c
        for c in ("WTI_price", "GPRD", "GPRD_THREAT", "GEPU_current")
        if c in feature_names
    ]

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("**Most recent values** (editable)")
        if full is not None and len(full):
            recent_row = full.iloc[-1]
            defaults = {c: float(recent_row[c]) for c in key_features if c in full.columns}
        else:
            defaults = {c: 0.0 for c in key_features}

        edited: dict[str, float] = {}
        for c in key_features:
            edited[c] = st.number_input(c, value=float(defaults.get(c, 0.0)))

    with col_b:
        if uploaded is not None:
            try:
                df_in = pd.read_csv(uploaded)
                st.markdown(f"Uploaded CSV — using last {seq_len} rows.")
                st.dataframe(df_in.tail(seq_len))
            except Exception as exc:
                st.error(f"Failed to read CSV: {exc}")
                df_in = None
        else:
            df_in = None

    predict_clicked = st.button("Predict", type="primary", use_container_width=True)

    if predict_clicked:
        if not api_status_ok:
            st.error(f"Cannot reach the API at {API_URL}.")
        elif not feature_names:
            st.error("Feature schema is missing. Build the dataset first.")
        else:
            # Build a (seq_len, n_features) array.
            if df_in is not None and all(c in df_in.columns for c in feature_names):
                arr = df_in[feature_names].tail(seq_len).to_numpy(dtype=float)
                if arr.shape[0] < seq_len:
                    st.error(
                        f"Uploaded CSV has only {arr.shape[0]} rows but {seq_len} are required."
                    )
                    arr = None
            elif full is not None and len(full) >= seq_len:
                arr = full[feature_names].tail(seq_len).to_numpy(dtype=float)
                for c, v in edited.items():
                    if c in feature_names:
                        arr[-1, feature_names.index(c)] = v
            else:
                st.error("Need either an uploaded CSV or a processed dataset.")
                arr = None

            if arr is not None:
                req_seq_len = int(api_info.get("seq_len", seq_len))
                if arr.shape[0] != req_seq_len:
                    # Pad or truncate to API's expected sequence length.
                    if arr.shape[0] > req_seq_len:
                        arr = arr[-req_seq_len:]
                    else:
                        pad = np.repeat(arr[:1], req_seq_len - arr.shape[0], axis=0)
                        arr = np.concatenate([pad, arr], axis=0)

                try:
                    resp = requests.post(
                        f"{API_URL}/predict",
                        json={"model": selected_model, "sequence": arr.tolist()},
                        timeout=15,
                    )
                    resp.raise_for_status()
                    payload = resp.json()
                except Exception as exc:
                    st.error(f"API call failed: {exc}")
                    payload = None

                if payload:
                    price = payload["predicted_price"]
                    last_price = (
                        float(full["WTI_price"].iloc[-1])
                        if full is not None and "WTI_price" in full.columns
                        else None
                    )
                    delta = (
                        f"{price - last_price:+.2f} USD vs last close"
                        if last_price is not None
                        else None
                    )
                    st.metric(
                        "Predicted next-day WTI price",
                        f"${price:.2f}",
                        delta=delta,
                    )

                    ci = payload.get("confidence_interval")
                    if ci:
                        st.markdown(
                            f"**80% interval:** ${ci['q10']:.2f} … ${ci['q90']:.2f}"
                        )
                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=["q10", "q50", "q90"],
                                y=[ci["q10"], ci["q50"], ci["q90"]],
                                mode="lines+markers",
                                line=dict(color="#1f77b4"),
                                name="Quantile forecast",
                            )
                        )
                        fig.update_layout(
                            yaxis_title="USD / barrel",
                            title="TFT quantile forecast",
                            height=320,
                        )
                        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2 — Historical analysis
# ---------------------------------------------------------------------------
with tab_history:
    st.subheader("Historical WTI price, GPR, and GEPU")
    full = load_full_dataset()
    if full is None:
        st.warning("No processed dataset found. Run `python -m src.data.build_dataset`.")
    else:
        st.write(f"{len(full):,} rows · {full['date'].min().date()} → {full['date'].max().date()}")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=full["date"], y=full["WTI_price"],
                name="WTI price", line=dict(color="#1f77b4"),
            )
        )
        if "GPRD" in full.columns:
            fig.add_trace(
                go.Scatter(
                    x=full["date"], y=full["GPRD"], name="GPRD",
                    line=dict(color="#d62728"), yaxis="y2", opacity=0.7,
                )
            )
        if "GEPU_current" in full.columns:
            fig.add_trace(
                go.Scatter(
                    x=full["date"], y=full["GEPU_current"], name="GEPU (current)",
                    line=dict(color="#2ca02c"), yaxis="y2", opacity=0.7,
                )
            )
        fig.update_layout(
            title="WTI price with GPR and GEPU overlays",
            xaxis_title="Date",
            yaxis=dict(title="WTI price (USD / barrel)"),
            yaxis2=dict(title="Index", overlaying="y", side="right"),
            height=520,
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Latest observations"):
            st.dataframe(full.tail(30))

        with st.expander("Correlation heatmap"):
            numeric = full.select_dtypes("number")
            corr = numeric.corr()
            fig_corr = px.imshow(
                corr, color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto",
                title="Feature correlation",
            )
            st.plotly_chart(fig_corr, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3 — Model comparison
# ---------------------------------------------------------------------------
with tab_compare:
    st.subheader("Model comparison")
    comp = load_comparison()
    if comp is None:
        st.warning(
            "No comparison file found. Run "
            "`python -m src.evaluation.compare_models` after training."
        )
    else:
        st.dataframe(comp, use_container_width=True)

        bar_df = comp.melt(
            id_vars=["model"], value_vars=["mae", "rmse"],
            var_name="metric", value_name="value",
        )
        fig_bar = px.bar(
            bar_df, x="model", y="value", color="metric", barmode="group",
            title="MAE and RMSE by model",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        if "inference_ms" in comp.columns:
            fig_inf = px.bar(
                comp.sort_values("inference_ms"),
                x="model", y="inference_ms",
                title="Inference latency (ms / sample)",
                color="model",
            )
            st.plotly_chart(fig_inf, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4 — About
# ---------------------------------------------------------------------------
with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
        This dashboard demonstrates a complete oil-price forecasting pipeline:

        **Data**
        - WTI spot price (EIA, daily)
        - Daily Geopolitical Risk Index (Caldara & Iacoviello)
        - Global Economic Policy Uncertainty (Davis, monthly)

        **Models**
        - Naive persistence baseline
        - Random Forest (500 trees)
        - BiLSTM + attention pooling
        - 1D-CNN + GRU hybrid
        - Temporal Fusion Transformer (TFT) — with q10/q50/q90 quantile output

        **Pipeline**
        - Chronological splits (no shuffling), `seq_len=30`, MSE / quantile losses,
          Adam + ReduceLROnPlateau, early stopping, full reproducibility.

        **Architecture**
        - `FastAPI` exposes `/predict`, `/models`, `/health` and is consumed by
          this Streamlit dashboard via `requests`.
        """
    )
    if api_info:
        st.markdown("**API metadata**")
        st.json(api_info)
