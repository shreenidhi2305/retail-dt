"""
simulation.py

Orchestrates the Digital Twin at the "business decision" layer:
  - runs a scenario end-to-end (forecast -> twin -> summary)
  - compares baseline vs scenario
  - computes a transparent Business Operational Health score
  - ranks critical store-item pairs by a transparent risk score
  - exports machine-readable DSS evidence (dss_evidence.json)

Nothing here invents data. Every number traces back to either the trained
model's predictions or the twin's simulated state transitions.
"""

import json
from dataclasses import asdict
from datetime import datetime

import numpy as np
import pandas as pd

from forecasting import recursive_forecast, bulk_historical_demand_stats
from twin import DigitalTwin, ScenarioConfig, SupplierConfig


def run_scenario(model_bundle, train_df, pairs, scenario: ScenarioConfig,
                  supplier: SupplierConfig, start_date, horizon_days):
    """
    Runs one full scenario: forecast demand for the horizon, then run the
    twin's day-by-day state transitions. Returns (twin, steps_df, summary_df).
    """
    pred_df = recursive_forecast(model_bundle, train_df, pairs, start_date, horizon_days)
    demand_stats = bulk_historical_demand_stats(train_df, pairs)
    twin = DigitalTwin(pairs, scenario, supplier, demand_stats, pred_df)
    steps_df = twin.run()
    summary_df = twin.network_summary()
    return twin, steps_df, summary_df


def network_aggregate(summary_df: pd.DataFrame) -> dict:
    """Rolls up per-store-item summaries into network-level totals."""
    if summary_df.empty:
        return {}
    total_demand = summary_df["total_demand"].sum()
    fulfilled = summary_df["fulfilled_demand"].sum()
    unmet = summary_df["unmet_demand"].sum()
    return {
        "total_demand": round(float(total_demand), 1),
        "fulfilled_demand": round(float(fulfilled), 1),
        "unmet_demand": round(float(unmet), 1),
        "service_level": round(float(fulfilled / total_demand), 4) if total_demand > 0 else 1.0,
        "stockout_events": int(summary_df["stockout_days"].sum()),
        "average_inventory": round(float(summary_df["average_inventory"].mean()), 1),
        "min_inventory": round(float(summary_df["min_inventory"].min()), 1),
        "max_inventory": round(float(summary_df["max_inventory"].max()), 1),
        "replenishment_orders": int(summary_df["replenishment_orders"].sum()),
        "n_store_items": int(len(summary_df)),
        "critical_store_items": int((summary_df["stockout_rate"] > 0.1).sum()),
    }


def compare_baseline_vs_scenario(baseline_agg: dict, scenario_agg: dict) -> dict:
    """Transparent baseline-vs-scenario deltas at the network level."""
    if not baseline_agg or not scenario_agg:
        return {}
    return {
        "service_level_change": round(scenario_agg["service_level"] - baseline_agg["service_level"], 4),
        "stockout_change": scenario_agg["stockout_events"] - baseline_agg["stockout_events"],
        "unmet_demand_change": round(scenario_agg["unmet_demand"] - baseline_agg["unmet_demand"], 1),
        "inventory_change": round(scenario_agg["average_inventory"] - baseline_agg["average_inventory"], 1),
        "replenishment_orders_change": scenario_agg["replenishment_orders"] - baseline_agg["replenishment_orders"],
        "critical_store_items_change": scenario_agg["critical_store_items"] - baseline_agg["critical_store_items"],
    }


def business_health_score(agg: dict) -> dict:
    """
    Transparent 0-100 operational health score. Every component is derived
    directly from simulated outcomes -- documented here, not a black box:

      Service Level      -> up to 60 pts, linear in service_level (0-100%)
      Stockout Frequency  -> penalty, up to -25 pts, scaled by stockout rate
      Unmet Demand        -> penalty, up to -10 pts, scaled by unmet/total ratio
      Inventory Stability -> up to +15 pts, rewards a tight min/max inventory
                              spread relative to average inventory (less
                              volatile stock levels = more stable operation)
    Base offset of 60 keeps a "do-nothing, no stockouts" scenario near 100;
    components can push the score down as conditions worsen.
    """
    if not agg:
        return {"score": None, "components": {}}

    service_level = agg["service_level"]
    total_demand = max(agg["total_demand"], 1e-6)
    unmet_ratio = agg["unmet_demand"] / total_demand
    stockout_rate = agg["critical_store_items"] / max(agg["n_store_items"], 1)

    spread = agg["max_inventory"] - agg["min_inventory"]
    avg_inv = max(agg["average_inventory"], 1e-6)
    volatility_ratio = min(spread / avg_inv, 3.0)  # cap extreme cases
    inventory_stability_pts = 15.0 * (1 - volatility_ratio / 3.0)

    service_pts = 60.0 * service_level
    stockout_penalty = -25.0 * min(stockout_rate * 2, 1.0)
    unmet_penalty = -10.0 * min(unmet_ratio * 5, 1.0)

    raw_score = 25.0 + service_pts + stockout_penalty + unmet_penalty + inventory_stability_pts
    score = float(np.clip(raw_score, 0, 100))

    return {
        "score": round(score, 1),
        "components": {
            "service_level": round(service_pts, 1),
            "stockout_risk": round(stockout_penalty, 1),
            "unmet_demand": round(unmet_penalty, 1),
            "inventory_stability": round(inventory_stability_pts, 1),
            "base_offset": 25.0,
        },
    }


def rank_critical_entities(summary_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Transparent risk score per store-item = weighted combination of
    stockout rate and unmet demand volume, both directly from the twin.
    risk_score = 0.6 * stockout_rate (0-1) + 0.4 * normalized unmet demand
    """
    if summary_df.empty:
        return summary_df

    df = summary_df.copy()
    max_unmet = max(df["unmet_demand"].max(), 1e-6)
    df["unmet_norm"] = df["unmet_demand"] / max_unmet
    df["risk_score"] = 0.6 * df["stockout_rate"] + 0.4 * df["unmet_norm"]

    def risk_label(r):
        if r >= 0.5:
            return "HIGH"
        if r >= 0.2:
            return "MEDIUM"
        return "LOW"

    df["risk_level"] = df["risk_score"].apply(risk_label)
    df = df.sort_values("risk_score", ascending=False).reset_index(drop=True)
    cols = ["store", "item", "risk_level", "risk_score", "stockout_rate",
            "unmet_demand", "service_level", "replenishment_orders"]
    return df[cols].head(top_n)


def export_dss_evidence(
    path,
    model_metrics: dict,
    baseline_agg: dict,
    scenario_agg: dict,
    scenario_config: ScenarioConfig,
    comparison: dict,
    critical_df: pd.DataFrame,
    recommendations: list,
    n_stores: int,
    n_items: int,
    horizon_days: int,
    observed_columns=("date", "store", "item", "sales"),
):
    """Writes the machine-readable DSS evidence JSON described in the spec."""
    evidence_rows = []
    for label, base_key, scen_key in [
        ("Service level", "service_level", "service_level"),
        ("Stockout events", "stockout_events", "stockout_events"),
        ("Unmet demand", "unmet_demand", "unmet_demand"),
        ("Average inventory", "average_inventory", "average_inventory"),
        ("Replenishment orders", "replenishment_orders", "replenishment_orders"),
    ]:
        base_val = baseline_agg.get(base_key)
        scen_val = scenario_agg.get(scen_key)
        if base_val is None or scen_val is None:
            continue
        evidence_rows.append({
            "finding": f"{label} changed from baseline to scenario '{scenario_config.name}'",
            "metric": base_key,
            "baseline_value": base_val,
            "scenario_value": scen_val,
            "change": round(scen_val - base_val, 4) if isinstance(scen_val, (int, float)) else None,
        })

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "domain": "business",
        "digital_twin": {
            "type": "retail_demand_inventory",
            "stores": n_stores,
            "items": n_items,
            "simulation_horizon_days": horizon_days,
        },
        "data": {
            "source": "Store Item Demand Forecasting dataset (train.csv/test.csv)",
            "observed_variables": list(observed_columns),
        },
        "model": {
            "algorithm": "XGBoost",
            "forecast_horizon_days": horizon_days,
            "metrics": model_metrics,
        },
        "baseline": baseline_agg,
        "scenario": {
            "name": scenario_config.name,
            "demand_multiplier": scenario_config.demand_multiplier,
            "lead_time_change_days": scenario_config.lead_time_change_days,
            "service_level_target": scenario_config.service_level_target,
            **scenario_agg,
        },
        "impact": comparison,
        "critical_entities": critical_df.to_dict(orient="records") if critical_df is not None and not critical_df.empty else [],
        "recommendations": recommendations,
        "evidence": evidence_rows,
        "dt_value_indicators": {
            "predictive_capability": True,
            "scenario_simulation": True,
            "risk_detection": True,
            "decision_support": True,
            "operational_optimization": True,
        },
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return payload
