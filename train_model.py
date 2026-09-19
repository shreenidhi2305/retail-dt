"""
train_model.py

Trains the XGBoost demand forecasting model used by the Retail Digital Twin
and saves it to models/demand_model.pkl so the Streamlit app never has to
retrain on startup.

Run:
    python train_model.py
"""

import json
import time
import joblib
import numpy as np
import xgboost as xgb

from data_utils import load_train, build_features, FEATURE_COLUMNS, smape
from sklearn.metrics import mean_absolute_error, mean_squared_error

VALIDATION_DAYS = 90  # last 90 days of train.csv held out, chronologically


def main():
    t0 = time.time()
    print("Loading train.csv ...")
    df = load_train("data/train.csv")

    print("Building calendar + lag + rolling features ...")
    df = build_features(df)

    # Rows at the start of each store-item series don't have a full lag_28 /
    # rolling_28 history yet -> drop them rather than imputing fake demand.
    before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS + ["sales"]).reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows with insufficient lag history "
          f"({before} -> {len(df)})")

    # Chronological split: last VALIDATION_DAYS days as validation set.
    cutoff = df["date"].max() - pd.Timedelta(days=VALIDATION_DAYS)
    train_df = df[df["date"] <= cutoff]
    val_df = df[df["date"] > cutoff]
    print(f"Train: {train_df.date.min().date()} -> {train_df.date.max().date()} "
          f"({len(train_df)} rows)")
    print(f"Val:   {val_df.date.min().date()} -> {val_df.date.max().date()} "
          f"({len(val_df)} rows)")

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["sales"]
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df["sales"]

    model = xgb.XGBRegressor(
        n_estimators=600,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=30,
        eval_metric="mae",
    )

    print("Training XGBoost ...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    preds = model.predict(X_val)
    preds = np.clip(preds, 0, None)  # sales can't be negative

    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    smape_val = smape(y_val, preds)

    print("\n=== Validation metrics (last %d days, held out chronologically) ===" % VALIDATION_DAYS)
    print(f"MAE:   {mae:.3f}")
    print(f"RMSE:  {rmse:.3f}")
    print(f"sMAPE: {smape_val:.2f}%")

    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump(
        {"model": model, "feature_columns": FEATURE_COLUMNS},
        "models/demand_model.pkl",
    )
    print("\nSaved model -> models/demand_model.pkl")

    metrics = {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "sMAPE": round(float(smape_val), 4),
        "validation_days": VALIDATION_DAYS,
        "n_train_rows": int(len(train_df)),
        "n_val_rows": int(len(val_df)),
        "train_date_range": [str(train_df.date.min().date()), str(train_df.date.max().date())],
        "val_date_range": [str(val_df.date.min().date()), str(val_df.date.max().date())],
    }
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics -> models/metrics.json")
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    import pandas as pd
    main()
