"""
data_utils.py

Shared data loading + feature engineering for the Retail Digital Twin.

IMPORTANT — observed vs derived variables:
  - Observed (from dataset): date, store, item, sales
  - Derived (computed here): calendar features, lag features, rolling
    mean/std features. These are legitimate features engineered FROM the
    observed data, not invented data.
Inventory, suppliers, lead times etc. are NOT touched here — those are
simulated later in twin.py / simulation.py.
"""

import pandas as pd
import numpy as np

LAGS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 28]

FEATURE_COLUMNS = (
    ["day_of_week", "day_of_month", "month", "year", "week_of_year", "is_weekend"]
    + [f"lag_{l}" for l in LAGS]
    + [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]
    + [f"rolling_std_{w}" for w in [7, 28]]
    + ["store", "item"]
)


def load_train(path="data/train.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["store", "item", "date"]).reset_index(drop=True)


def load_test(path="data/test.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["store", "item", "date"]).reset_index(drop=True)


def add_calendar_features(df):
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def add_lag_and_rolling_features(df):
    """
    Adds lag + rolling features computed strictly from PAST values within
    each store-item group (shift(1) before rolling => no leakage of the
    current day's own sales into its own features).
    """
    df = df.copy()
    df = df.sort_values(["store", "item", "date"])
    grp = df.groupby(["store", "item"])["sales"]

    for l in LAGS:
        df[f"lag_{l}"] = grp.shift(l)

    shifted = grp.shift(1)  # never include current day in rolling stats
    for w in ROLLING_WINDOWS:
        df[f"rolling_mean_{w}"] = (
            shifted.groupby([df["store"], df["item"]]).rolling(w).mean().reset_index(drop=True)
        )
    for w in [7, 28]:
        df[f"rolling_std_{w}"] = (
            shifted.groupby([df["store"], df["item"]]).rolling(w).std().reset_index(drop=True)
        )
    return df


def build_features(df):
    df = add_calendar_features(df)
    df = add_lag_and_rolling_features(df)
    return df


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred)
    out = np.where(denom == 0, 0.0, diff / denom)
    return 200.0 * np.mean(out)
