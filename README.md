# Oil Price Prediction (Deep Learning)

End-to-end deep learning project that predicts next-day WTI crude oil spot price
from a combination of:

- **WTI Spot Price** (EIA, daily, 1986–present)
- **Daily Geopolitical Risk Index — GPRD** (Caldara & Iacoviello, 1985–present)
- **Global Economic Policy Uncertainty — GEPU** (Davis, monthly, 1997–present)

Three models are trained and compared:

1. **Baseline** — Naive persistence + Random Forest regressor
2. **Deep Learning** — BiLSTM with attention pooling and a 1D-CNN + GRU hybrid
3. **Advanced** — Temporal Fusion Transformer (TFT)-inspired model with
   variable selection, multi-head self-attention, and a quantile output head
   (q10 / q50 / q90) for uncertainty estimation.

A FastAPI service and a Streamlit dashboard expose the trained models.

---

## Quickstart

```bash
# 1. Create environment and install
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate             # Windows
pip install -r requirements.txt

# 2. Place raw data files in data/raw/
#    - RWTCd.xls
#    - data_gpr_daily_recent.xls
#    - Global_Policy_Uncertainty_Data.xlsx

# 3. Build the processed dataset
python -m src.data.build_dataset

# 4. Train all models
python -m src.training.train_baseline
python -m src.training.train_deep --model bilstm
python -m src.training.train_deep --model cnn_gru
python -m src.training.train_advanced

# 5. Compare models
python -m src.evaluation.compare_models

# 6. Launch the API and the dashboard (two terminals)
uvicorn src.app.api:app --host 0.0.0.0 --port 8000
streamlit run src/app/streamlit_app.py
```

---

## Project layout

```
oil_price_prediction/
├── data/
│   ├── raw/                source .xls / .xlsx files
│   └── processed/          generated CSVs + scaler
├── src/
│   ├── data/               build_dataset.py, preprocessing.py
│   ├── models/             baseline.py, deep_model.py, advanced_model.py
│   ├── training/           train_baseline.py, train_deep.py, train_advanced.py
│   ├── evaluation/         compare_models.py
│   └── app/                api.py, streamlit_app.py
├── models/                 saved checkpoints (.pkl / .pt)
├── notebooks/              01_full_pipeline.ipynb
└── outputs/
    ├── figures/            PNG plots (150 DPI)
    └── metrics/            JSON / CSV reports
```

---

## Data sources

| Dataset | Source | License |
|---------|--------|---------|
| WTI Spot Price (`RWTCd.xls`) | U.S. Energy Information Administration (EIA) | Public domain |
| Geopolitical Risk Index (`data_gpr_daily_recent.xls`) | Caldara & Iacoviello, *Measuring Geopolitical Risk*, AER 2022 | Free for non-commercial use |
| Global Economic Policy Uncertainty (`Global_Policy_Uncertainty_Data.xlsx`) | Davis, *An Index of Global Economic Policy Uncertainty*, NBER 2016 | Free for non-commercial use |

---

## Results

| Model              | MAE | RMSE | MAPE (%) | R² |
|--------------------|-----|------|----------|----|
| Naive persistence  | TBD | TBD  | TBD      | TBD |
| Random Forest      | TBD | TBD  | TBD      | TBD |
| BiLSTM + Attention | TBD | TBD  | TBD      | TBD |
| CNN + GRU          | TBD | TBD  | TBD      | TBD |
| TFT (q50)          | TBD | TBD  | TBD      | TBD |

Run `python -m src.evaluation.compare_models` after training to populate this table.

---

## Notes

- All data splits are **chronological** (no shuffling).
- Splits: train ≤ 2020-12-31 · val 2021-01-01 → 2022-12-31 · test 2023-01-01 → 2025-11-30.
- Sequence length for all deep models is **30 trading days** (configurable).
- All figures saved at **150 DPI** under `outputs/figures/`.
