"""
forecasting.py

Produces PREDICTED demand for future (simulation) dates by recursively
applying the trained XGBoost model: each day's prediction becomes part of
the lag/rolling history used to predict the next day. This is necessary
because test.csv/simulation dates are beyond the training data and have no
real sales to compute lag features from.

This module only produces `predicted_demand`. It does NOT simulate
inventory, orders, or "actual" demand -- that happens in twin.py, which
consumes this module's output.
"""

import collections
import joblib
import numpy as np
import pandas as pd

from data_utils import FEATURE_COLUMNS, LAGS, ROLLING_WINDOWS

MAX_HISTORY = 35  # >= max(LAGS, ROLLING_WINDOWS) with headroom


def load_model_bundle(path="models/demand_model.pkl"):
    return joblib.load(path)


def _calendar_features(date):
    dow = date.dayofweek
    return {
        "day_of_week": dow,
        "day_of_month": date.day,
        "month": date.month,
        "year": date.year,
        "week_of_year": int(date.isocalendar()[1]),
        "is_weekend": int(dow >= 5),
    }


def _init_histories(train_df, store_item_pairs):
    """
    Build a dict {(store, item): deque[sales]} seeded with each combo's most
    recent MAX_HISTORY days of REAL observed sales from train_df.
    """
    histories = {}
    grouped = train_df.sort_values("date").groupby(["store", "item"])
    for (store, item), g in grouped:
        if (store, item) in store_item_pairs:
            histories[(store, item)] = collections.deque(
                g["sales"].tail(MAX_HISTORY).tolist(), maxlen=MAX_HISTORY
            )
    return histories


def _features_from_history(hist_deque, date):
    vals = list(hist_deque)  # oldest -> newest
    n = len(vals)
    row = _calendar_features(date)
    for l in LAGS:
        row[f"lag_{l}"] = vals[n - l] if n >= l else np.nan
    for w in ROLLING_WINDOWS:
        window = vals[max(0, n - w):]
        row[f"rolling_mean_{w}"] = float(np.mean(window)) if window else np.nan
    for w in [7, 28]:
        window = vals[max(0, n - w):]
        row[f"rolling_std_{w}"] = float(np.std(window)) if len(window) > 1 else 0.0
    return row


def recursive_forecast(
    model_bundle,
    train_df,
    store_item_pairs,
    start_date,
    horizon_days,
):
    """
    Batched day-by-day recursive forecast across many (store, item) combos.

    Returns a long-format DataFrame: date, store, item, predicted_demand
    """
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]

    histories = _init_histories(train_df, set(store_item_pairs))
    dates = pd.date_range(start=start_date, periods=horizon_days, freq="D")

    records = []
    for date in dates:
        rows = []
        keys = []
        for key in store_item_pairs:
            hist = histories.get(key)
            if hist is None:
                continue  # unseen combo, skip rather than fabricate
            feat = _features_from_history(hist, date)
            feat["store"], feat["item"] = key
            rows.append(feat)
            keys.append(key)

        if not rows:
            continue

        X = pd.DataFrame(rows)[feature_columns]
        preds = model.predict(X)
        preds = np.clip(preds, 0, None)

        for key, pred in zip(keys, preds):
            histories[key].append(float(pred))  # feed prediction back as history
            records.append(
                {"date": date, "store": key[0], "item": key[1], "predicted_demand": float(pred)}
            )

    return pd.DataFrame.from_records(records)


def historical_demand_stats(train_df, store, item, lookback_days=90):
    """
    Returns (avg_daily_demand, demand_std) computed from the last
    `lookback_days` of REAL observed sales for a store-item. Used to size
    safety stock / reorder points / initial inventory in twin.py.
    """
    g = train_df[(train_df.store == store) & (train_df.item == item)].sort_values("date")
    tail = g["sales"].tail(lookback_days)
    return float(tail.mean()), float(tail.std(ddof=0))


def bulk_historical_demand_stats(train_df, store_item_pairs, lookback_days=90):
    """
    Vectorized version of historical_demand_stats() for many pairs at once
    (e.g. a full 10-store x 50-item network). Returns
    {(store, item): (avg_daily_demand, demand_std)}.
    """
    pairs_set = set(store_item_pairs)
    mask = list(zip(train_df["store"], train_df["item"]))
    df = train_df[[p in pairs_set for p in mask]].sort_values("date")
    tail_df = df.groupby(["store", "item"], group_keys=False).tail(lookback_days)
    agg = tail_df.groupby(["store", "item"])["sales"].agg(["mean", "std"]).fillna(0.0)
    return {
        (s, i): (float(row["mean"]), float(row["std"]))
        for (s, i), row in agg.iterrows()
    }
